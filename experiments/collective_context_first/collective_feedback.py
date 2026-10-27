"""Training-only qualitative replay, diagnosis and cluster-prompt repair."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from collective_feedback_prompts import DIAGNOSIS_META, DIAGNOSIS_SPEC, UPDATE_META, UPDATE_SPEC

VALIDITY = {'supported', 'partially_supported', 'unsupported', 'uncertain'}
RELATIONS = {'case_better', 'equivalent', 'collective_better', 'uncertain'}
CHANGES = {'not_applicable', 'improved', 'unchanged', 'regressed', 'uncertain'}
ISSUES = {'rule_omission', 'overgeneralization', 'rule_conflict', 'data_or_judge_issue', 'other'}
FINDING_FIELDS = {'issue_type', 'actionable', 'case_prompt_excerpt', 'collective_prompt_excerpt',
                  'case_answer_excerpt', 'collective_answer_excerpt', 'context_excerpt', 'explanation', 'repair_principle'}
DIAG_FIELDS = {'case_answer_validity', 'collective_answer_validity', 'relation_to_case',
               'change_from_previous', 'summary', 'change_reason', 'findings'}
SCORE_KEYS = {'score', 'scores', 'rating', 'total_score', 'overall_score', 'factual_correctness', 'faithfulness', 'semantic_similarity'}


def enabled(cfg):
    return cfg.get('collective_feedback', {}).get('enabled', False)


def active_cluster_dir(run, cfg):
    run = Path(run)
    if not enabled(cfg):
        return run / 'clusters'
    manifest = json.loads((run / 'collective_feedback/final_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('Collective feedback has not completed; do not use unfinished prompts')
    if cfg['collective_feedback'].get('require_review', True):
        approval = json.loads((run / 'feedback_approval.json').read_text(encoding='utf-8'))
        if approval['artifact_sha256'] != manifest['artifact_sha256']:
            raise ValueError('Feedback approval does not match final prompts')
    return run / 'collective_feedback/final_clusters'


def diagnostic_payload(row, case_prompt, case_answer, prompt, answer, previous=None, runtime=None):
    payload={'question': row['question'], 'reference_answer': row['reference_answer'],
            'contexts': runtime.context_view(row), 'case_optimized': {'prompt': case_prompt, 'answer': case_answer},
            'current_collective': {'prompt': prompt, 'answer': answer}, 'previous_collective': previous}
    if getattr(runtime,'ALLOW_INTERNAL_KNOWLEDGE',False):payload['answer_policy']='context_first_allow_internal_knowledge'
    return payload


def validate_diagnosis(value, payload):
    if not isinstance(value, dict) or set(value) != DIAG_FIELDS:
        raise ValueError('Diagnosis must contain only qualitative comparison fields')
    def no_scores(obj):
        if isinstance(obj, dict):
            if set(obj) & SCORE_KEYS: raise ValueError('Numeric/three-dimension grading is not allowed in fusion feedback')
            for v in obj.values(): no_scores(v)
        elif isinstance(obj, list):
            for v in obj: no_scores(v)
    no_scores(value)
    for key in ('case_answer_validity', 'collective_answer_validity'):
        if not isinstance(value[key], str) or value[key] not in VALIDITY: raise ValueError('Unknown answer validity')
    if not isinstance(value['relation_to_case'],str) or not isinstance(value['change_from_previous'],str) or value['relation_to_case'] not in RELATIONS or value['change_from_previous'] not in CHANGES:
        raise ValueError('Unknown comparison decision')
    if bool(payload['previous_collective']) == (value['change_from_previous'] == 'not_applicable'):
        raise ValueError('Previous-answer comparison is inconsistent with its input')
    for key in ('summary', 'change_reason'):
        if not isinstance(value[key], str) or not value[key].strip(): raise ValueError('Comparison needs evidence-based explanations')
    if not isinstance(value['findings'], list): raise ValueError('findings must be a list, possibly empty')
    for item in value['findings']:
        if not isinstance(item, dict) or set(item) != FINDING_FIELDS: raise ValueError('Malformed finding')
        if item['issue_type'] not in ISSUES or type(item['actionable']) is not bool: raise ValueError('Malformed issue type/actionability')
        for key in FINDING_FIELDS - {'issue_type', 'actionable'}:
            if not isinstance(item[key], str): raise ValueError('Evidence and explanations must be text')
        if not item['explanation'].strip(): raise ValueError('Finding needs an explanation, not just a label')
        sources = {'case_prompt_excerpt': [payload['case_optimized']['prompt']],
                   'collective_prompt_excerpt': [payload['current_collective']['prompt']],
                   'case_answer_excerpt': [payload['case_optimized']['answer']],
                   'collective_answer_excerpt': [payload['current_collective']['answer']],
                   'context_excerpt': [d['text'] for d in payload['contexts']]}
        for key, texts in sources.items():
            if item[key] and not any(item[key] in text for text in texts): raise ValueError('Invented evidence excerpt: ' + key)
        if item['issue_type'] == 'data_or_judge_issue' and item['actionable']:
            raise ValueError('Data/judge conflicts cannot become mandatory QA rules')
        required_excerpts=('case_answer_excerpt','collective_answer_excerpt') if payload.get('answer_policy')=='context_first_allow_internal_knowledge' else ('case_answer_excerpt','collective_answer_excerpt','context_excerpt')
        if item['actionable'] and (not item['repair_principle'].strip() or any(not item[k].strip() for k in required_excerpts)):
            raise ValueError('Actionable finding requires both answers, context evidence and a transferable repair')
    return value


def verified_loss(record):
    d = record['diagnosis']
    return (d['relation_to_case'] == 'case_better'
            and d['case_answer_validity'] in {'supported', 'partially_supported'}
            and any(f['actionable'] and f['issue_type'] != 'data_or_judge_issue' for f in d['findings']))


def acceptance(before, after):
    if set(before) != set(after): raise ValueError('Candidate must be replayed on the same complete cluster')
    blocked = []; fixed = []
    for ident, old in before.items():
        new = after[ident]; d = new['diagnosis']; prior = old['diagnosis']
        if d['change_from_previous'] in {'regressed', 'uncertain'}:
            blocked.append(ident)
        if prior['relation_to_case'] in {'equivalent', 'collective_better'} and d['relation_to_case'] == 'case_better':
            blocked.append(ident)
        if (verified_loss(old) and d['relation_to_case'] in {'equivalent', 'collective_better'}
                and d['change_from_previous'] == 'improved' and old['answer'].strip() != new['answer'].strip()):
            fixed.append(ident)
    return {'accepted': bool(fixed) and not blocked, 'resolved_case_ids': sorted(fixed),
            'blocked_case_ids': sorted(set(blocked)),
            'reason': 'new_regression_or_uncertain_change' if blocked else ('verified_fusion_loss_resolved' if fixed else 'no_verified_loss_resolved')}


def diagnosis_messages(text, payload, errors):
    user=json.dumps(payload,ensure_ascii=False)
    if errors:
        user+='\n[格式与引文校验反馈]\n'+json.dumps(errors,ensure_ascii=False)+'\n仅纠正不合规的JSON结构或不属于输入原文的引文，依照原诊断要求重新核实；不根据方法优劣重采样有效判断。'
    return [{'role':'system','content':text},{'role':'user','content':user}]


def diagnose(api,key,text,payload,runtime):
    """First schema/evidence-valid judgment only; valid low/uncertain results stand."""
    r=runtime;errors=[];attempts=[]
    for index in range(3):
        result=api.ask(key+(':diagnosis' if index==0 else ':diagnosis-schema-recovery:'+str(index)),diagnosis_messages(text,payload,errors))
        attempts.append(result['raw_record'])
        try:
            value=json.loads(r.normalize(result['text']));validate_diagnosis(value,payload)
            return value,result,attempts
        except (ValueError,TypeError,KeyError) as exc:
            errors.append({'error':str(exc),'raw_record':result['raw_record']})
            r.save(r.RUN/'validation_failures'/(r.digest(key+':diagnosis')+'.json'),{'key':key+':diagnosis','errors':errors,'raw_records':attempts})
    raise RuntimeError('Qualitative diagnosis schema/evidence recovery exhausted: '+key)


def validate_feedback_stage(kind,text,runtime):
    """Check the operative contract, not literal prose or Markdown formatting."""
    r=runtime;text=r.normalize(text)
    if any(not r.section(text,h) for h in ('角色','任务','输入')):raise ValueError('Missing stage structure')
    task=r.section(text,'任务')
    task=re.sub(r'(?:不再|不得|禁止|不能|不要)[^。\n；]{0,20}(?:诊断指令|更新指令)', '', task)
    if re.search(r'(?:生成|设计)[^。\n]{0,16}(?:阶段|诊断|更新)指令',task):
        raise ValueError('Execute diagnosis/update, not another instruction generator')
    if kind=='diagnosis':
        required=DIAG_FIELDS|FINDING_FIELDS|VALIDITY|RELATIONS|CHANGES|ISSUES
        if any(token not in text for token in required):raise ValueError('Preserve all diagnosis fields and allowed decisions')
        if not all(token in text for token in ('案例级','群智','逐字','旧评分','评分','数据','findings')):
            raise ValueError('Preserve evidence, no-old-score and data boundaries')
        if not re.search(r'允许全部不存在|允许.*不存在|未发现.*问题',text):raise ValueError('Do not assume a fusion loss')
    else:
        if not all(token in text for token in ('案例','群智','参考答案','上下文','data_or_judge_issue','簇')):
            raise ValueError('Preserve case evidence and cluster/data boundaries')
        if any(h not in text for h in ('角色','任务','输入','证据使用','作答要求','输出')):
            raise ValueError('Preserve QA structure')
        if not re.search(r'不[^。\n]{0,20}(?:分数|评分|得分)',text):raise ValueError('Qualitative updater must not use numeric scores')
    return text


def optimize(api, cfg, train_rows, cases, clustering, initial_clusters, runtime):
    """No test rows, tests scores or numeric training grades enter this module."""
    r = runtime; run = r.RUN
    if not enabled(cfg): return initial_clusters
    options = cfg['collective_feedback']
    max_rounds = options.get('max_rounds', 3)
    if type(max_rounds) is not int or not 1 <= max_rounds <= 10: raise ValueError('Set a bounded feedback round limit (1..10)')
    base = run / 'collective_feedback'; final = base / 'final_manifest.json'
    if (run/'inference_frozen.json').exists() and not final.exists(): raise ValueError('Start feedback in a new isolated run, before test inference')
    by_id = {str(c['id']): c for c in cases}; rows = {str(row['id']): row for row in train_rows}
    if len(rows) != len(train_rows) or set(rows) != set(by_id) or set(rows) != set(clustering['train_labels']):
        raise ValueError('Feedback accepts exactly the original training-case bank')
    selected = {}
    for ident, case in by_id.items():
        version = case['history'][case['selected_iteration']]
        if case['status'] != 'complete' or version['prompt'] != case['prompt'] or not version['answer'].strip(): raise ValueError('Missing selected case answer')
        selected[ident] = {'prompt': version['prompt'], 'answer': version['answer']}
    lineage = {'training_ids': list(rows), 'training_inputs_sha256': r.digest(train_rows),
               'selected_cases_sha256': r.digest(selected), 'initial_clusters': {k:v['sha256'] for k,v in initial_clusters.items()},
               'max_rounds': max_rounds, 'diagnosis_meta_sha256': r.digest(DIAGNOSIS_META), 'update_meta_sha256': r.digest(UPDATE_META),
               'model':cfg.get('model'),'seed':cfg.get('seed'),'context_budget':cfg.get('context_budget',5),
               'selection_policy': 'resolve a verified case-to-collective loss; no observed regression or uncertain change; no scores'}
    revision_path=run/'feedback_protocol_revision.json'
    revision=r.read(revision_path)['revision'] if revision_path.exists() else None
    if revision:lineage['diagnostic_protocol_revision']=revision
    frozen = base/'input_frozen.json'
    if frozen.exists():
        if r.read(frozen) != lineage: raise ValueError('Feedback inputs/protocol changed during resume')
    else: r.save(frozen, lineage)
    if final.exists():
        folder = active_cluster_dir(run, cfg) if (not options.get('require_review', True) or (run/'feedback_approval.json').exists()) else None
        if folder is None: r.mark('awaiting_feedback_review'); return None
        values = {str(i): r.read(folder/('cluster_'+str(i)+'.json')) for i in range(clustering['k'])}
        if r.read(final)['clusters'] != {k:r.digest(r.validate_qa(v['text'])) for k,v in values.items()}: raise ValueError('Final feedback prompts changed')
        return values

    def stage(kind, meta, spec):
        p = base/'stages'/(kind+'.json')
        def validate(text):
            return validate_feedback_stage(kind,text,r)
        if p.exists():
            value=r.read(p); validate(value['text']); return value
        key='feedback:stage:'+kind;payload={'task':'生成可直接执行的阶段指令，不输入具体案例。'}
        if revision:key+=':protocol_'+str(revision)
        repair_path=run/'feedback_stage_repair_inputs.json'
        repairs=r.read(repair_path) if repair_path.exists() else {}
        if kind in repairs:
            payload['semantic_compliance_correction']=repairs[kind]
            key+=':contract_'+str(repairs[kind].get('revision','v1'))
        value=r.generated(api,key,meta,payload,validate);r.save(p,value);return value
    diagnosis_stage=stage('diagnosis',DIAGNOSIS_META,DIAGNOSIS_SPEC)
    update_stage=stage('update',UPDATE_META,UPDATE_SPEC)
    stages={'diagnosis':diagnosis_stage['sha256'],'update':update_stage['sha256']}
    r.save(run/'feedback_stage_review.json',{'artifact_sha256':r.digest(stages),'stages':stages,'status':'needs_semantic_review'})
    if options.get('require_review',True):
        p=run/'feedback_stage_approval.json'
        if not p.exists(): r.mark('awaiting_feedback_stage_review'); return None
        if r.read(p)['artifact_sha256']!=r.digest(stages): raise ValueError('Feedback stage approval mismatch')

    def replay(cid, round_number, prompt, members, previous=None):
        (base/'clusters'/cid/'rounds'/str(round_number)/'cases').mkdir(parents=True,exist_ok=True)
        def one(ident):
            row=rows[ident]; path=base/'clusters'/cid/'rounds'/str(round_number)/'cases'/(ident+'.json')
            prev={'prompt':previous[ident]['prompt'],'answer':previous[ident]['answer']} if previous else None
            expected={'question':row['question'],'reference_answer':row['reference_answer'],'contexts':r.context_view(row),
                      'case_optimized':selected[ident],'current_collective':{'prompt':prompt},'previous_collective':prev}
            if path.exists():
                result=r.read(path)
                if result['input_sha256']!=r.digest(expected): raise ValueError('Replay inputs changed')
                actual=diagnostic_payload(row,selected[ident]['prompt'],selected[ident]['answer'],prompt,result['answer'],prev,r)
                if result['diagnosis_input']!=actual or result['diagnosis_prompt_sha256']!=diagnosis_stage['sha256']:raise ValueError('Cached diagnosis changed')
                validate_diagnosis(result['diagnosis'],actual)
                return ident,result
            key='feedback:cluster:'+cid+':round:'+str(round_number)+':case:'+ident
            answer=api.ask(key+':answer',r.answer_messages(row,prompt))
            payload=diagnostic_payload(row,selected[ident]['prompt'],selected[ident]['answer'],prompt,answer['text'],prev,r)
            diagnosis_key=key+(':protocol_'+str(revision) if revision else '')
            value,diagnostic,attempts=diagnose(api,diagnosis_key,diagnosis_stage['text'],payload,r)
            result={'id':ident,'prompt':prompt,'answer':answer['text'],'answer_raw_record':answer['raw_record'],
                    'diagnosis':value,'diagnosis_raw_record':diagnostic['raw_record'],'diagnosis_attempts':attempts,'diagnosis_input':payload,
                    'diagnosis_prompt_sha256':diagnosis_stage['sha256'],'input_sha256':r.digest(expected)}
            r.save(path,result);return ident,result
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: return dict(pool.map(one,members))

    active={};states={}
    for cid, original in initial_clusters.items():
        cid=str(cid); members=[ident for ident in rows if str(clustering['train_labels'][ident])==cid]
        if members!=original['member_ids']: raise ValueError('Do not change cluster membership during feedback')
        state_path=base/'clusters'/cid/'state.json'
        if state_path.exists() and r.read(state_path).get('complete'):
            states[cid]=r.read(state_path);active[cid]=r.read(base/'final_clusters'/('cluster_'+cid+'.json'))
            if active[cid]['member_ids']!=members or r.digest(r.validate_qa(active[cid]['text']))!=states[cid]['selected_sha256']:raise ValueError('Completed cluster state changed')
            continue
        r.mark('collective_feedback_replay',cluster_id=cid,round=0,training_cases=len(members))
        prompt=original['text']; selected_round=0; origin=original
        history=[]; stop='round_limit'; current=None
        try:current=replay(cid,0,prompt,members)
        except RuntimeError as exc:
            if not str(exc).startswith('Qualitative diagnosis schema/evidence recovery exhausted: '):raise
            stop='initial_replay_unverifiable'
            history.append({'round':0,'accepted':False,'reason':stop,'error':str(exc)})
        for number in (range(1,max_rounds+1) if current is not None else []):
            if not any(verified_loss(v) for v in current.values()): stop='no_verified_fusion_loss';break
            r.mark('collective_feedback_update',cluster_id=cid,round=number)
            payload={'cluster_id':cid,'current_collective_prompt':prompt,
                     'training_cases':[{'id':ident,**current[ident]['diagnosis_input'],'diagnosis':current[ident]['diagnosis']} for ident in members]}
            candidate_path=base/'clusters'/cid/'rounds'/str(number)/'candidate.json'
            if candidate_path.exists(): candidate=r.read(candidate_path)
            else:
                try:candidate=r.generated(api,'feedback:cluster:'+cid+':update:'+str(number),update_stage['text'],payload,r.validate_qa)
                except RuntimeError as exc:
                    if not str(exc).startswith('Prompt compliance failed: '): raise
                    stop='candidate_noncompliant';history.append({'round':number,'accepted':False,'reason':stop});break
                r.save(candidate_path,candidate)
            if candidate['input']!=payload or candidate['sha256']!=r.digest(r.validate_qa(candidate['text'])): raise ValueError('Candidate lineage mismatch')
            if candidate['text']==prompt: stop='prompt_unchanged';break
            try:after=replay(cid,number,candidate['text'],members,current)
            except RuntimeError as exc:
                if not str(exc).startswith('Qualitative diagnosis schema/evidence recovery exhausted: '):raise
                stop='candidate_replay_unverifiable'
                history.append({'round':number,'accepted':False,'reason':stop,'error':str(exc),'candidate_sha256':candidate['sha256']})
                break
            decision=acceptance(current,after)
            history.append({'round':number,**decision,'candidate_sha256':candidate['sha256']})
            if not decision['accepted']: stop=decision['reason'];break
            current=after;prompt=candidate['text'];origin=candidate;selected_round=number
        value={**origin,'cluster_id':int(cid),'member_ids':members,'text':prompt,'sha256':r.digest(prompt),
               'question_type':r.section(prompt,'任务').split('。')[0][:70],
               'feedback':{'initial_sha256':original['sha256'],'accepted_round':selected_round,'state_path':str(state_path)}}
        state={'complete':True,'cluster_id':int(cid),'member_ids':members,'initial_sha256':original['sha256'],
               'selected_round':selected_round,'selected_sha256':value['sha256'],'stop_reason':stop,'history':history}
        r.save(base/'final_clusters'/('cluster_'+cid+'.json'),value);r.save(state_path,state);active[cid]=value;states[cid]=state
    hashes={k:v['sha256'] for k,v in active.items()}
    manifest={'complete':True,'artifact_sha256':r.digest(hashes),'clusters':hashes,'initial_clusters':lineage['initial_clusters'],
              'training_ids':list(rows),'stage_sha256':stages,'states':states,'numeric_scoring_used':False}
    r.save(final,manifest);r.save(run/'feedback_review.json',{'artifact_sha256':manifest['artifact_sha256'],'clusters':hashes,'status':'needs_semantic_review'})
    if options.get('require_review',True) and not (run/'feedback_approval.json').exists():r.mark('awaiting_feedback_review');return None
    active_cluster_dir(run,cfg)
    return active
