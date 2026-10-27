"""Fresh meta -> shared QA -> case updates -> dynamic clusters -> one-call QA.

One frozen three-dimension judge, one valid evaluation per answer. No old banks,
scores, answer ensembles, reference leakage, performance gates or schedulers.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get('RESTORED_RUN', str(ROOT / 'run')))
DIMS = ('factual_correctness','faithfulness','semantic_similarity')
LABELS = ('事实正确性','忠实度','语义相似性')
HEADINGS = ('角色','任务','输入','证据使用','作答要求','输出')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(x): return hashlib.sha256((x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()
def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def mark(stage, **kwargs):
    value={'stage':stage,'timestamp':time.time(),**kwargs};save(RUN/'job_status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
def section(text,name):
    m=re.search(r'(?m)^\['+re.escape(name)+r'\][ \t]*\n?(.*?)(?=^\[[^\]\n]+\]|\Z)',text,re.S|re.M)
    return m.group(1).strip() if m else ''
def normalize(text):
    text=text.strip();m=re.fullmatch(r'```(?:text|markdown|json)?\s*\n(.*?)\n```',text,re.S)
    return m.group(1).strip() if m else text
def contract(meta): return meta.split('[必须保留的评价规范]\n',1)[1].split('\n[生成指令的输出要求]',1)[0].strip()


def validate_qa(text):
    text=normalize(text)
    missing=[h for h in HEADINGS if not section(text,h)]
    if missing:raise ValueError('QA正文必须包含六部分，缺少：'+','.join(missing))
    task=section(text,'任务');output=section(text,'输出');evidence=section(text,'证据使用')
    if not re.search(r'回答|作答|答案|答复|解答',task):raise ValueError('QA任务必须直接回答当前问题')
    if re.search(r'(?:生成|改写|设计|融合)(?:一份|新版|新的|该|本)?(?:QA|问答|群智|初始)?提示',task):
        raise ValueError('成品QA不能要求再次生成、更新或融合提示；应直接回答问题')
    if not re.search(r'(?:只|仅)[^。\n]{0,15}(?:答案|回答)',output):raise ValueError('QA输出必须仅为最终答案')
    if re.search(r'(?:输出|展示)[^。；\n]{0,6}重排(?:序)?过程(?:简述|说明)',output):raise ValueError('不得输出重排过程')
    semantic_methods=(
        re.search(r'主体|关系',text) and re.search(r'核对|核验|一致|准确',text),
        re.search(r'断言|事实',text) and re.search(r'上下文|证据',text) and re.search(r'支持|依据',text),
        re.search(r'问项|子问|核心含义|核心问题|关键关系|必要条件',text) and re.search(r'覆盖|回应|完整|不遗漏',text))
    if any(label not in text for label in LABELS) and not all(semantic_methods):raise ValueError('QA须完整写明事实核对、断言有据、核心问项覆盖的操作方法；允许同义表达，不要求逐字写指标名称')
    for pattern in (r'噪音|无关|矛盾',r'全部|所有|逐条',r'rerank|重排|排序',r'保留|使用',r'不足|缺失'):
        if not re.search(pattern,evidence,re.I):raise ValueError('证据要求不完整，缺少：'+pattern)
    if re.search(r'(?:依照|依据|参照|读取|获取|结合)[^。\n]{0,6}(?:给定|提供的|当前|该)?(?:评分|评价)标准',task+'\n'+section(text,'作答要求')):
        raise ValueError('QA不得要求读取作答时未提供的评分标准；写明实际方法')
    if re.search(r'(?:必须|需要|应当|请)[^。\n]{0,8}(?:读取|获取|查看|参考)[^。\n]{0,5}(?:参考答案|标准答案|历史评分)',text):
        raise ValueError('QA不能索取未提供的参考答案或历史评分')
    if re.search(r'(?:放弃|舍弃|丢弃|排除|不使用)[^。\n]{0,5}无冲突',text):raise ValueError('不能舍弃无冲突的有效证据')
    return text


def validate_stage(kind,text,rubric):
    text=normalize(text)
    for h in ('角色','任务','输入'):
        if not section(text,h):raise ValueError('阶段指令缺少['+h+']')
    if kind=='qa':return validate_qa(text)
    task=section(text,'任务');inputs=section(text,'输入')
    if kind=='evaluation':
        if re.search(r'(?:生成|设计|编写)[^。\n]{0,15}(?:评价|评估)[^。\n]{0,5}(?:指令|提示)',task):raise ValueError('成品评价指令必须直接评价输入答案并输出三维评分JSON，不能再次生成评价指令。将[角色]改为答案评价专家，[任务]改为评价输入的实际答案，其余评分规范完整不变。')
        if rubric not in text:raise ValueError('必须在[评分要求]逐字保留以下完整评分规范，不改判据或加条件：\n'+rubric)
        if any(k not in inputs for k in ('question','reference_answer','candidate_answer','contexts')):raise ValueError('评价输入只能声明指定四项')
        if any(k not in text for k in (*DIMS,'score','reasoning')):raise ValueError('评价输出须为三个score/reasoning')
    elif kind=='update':
        if not re.search(r'更新|改写',task) or '提示' not in task:raise ValueError('本阶段任务必须改写当前QA提示，不是回答问题')
        if any(k not in inputs for k in ('问题','参考答案','答案','上下文','提示','评分','理由')):raise ValueError('更新输入须包含完整案例、当前提示、三维评分和理由')
        req=section(text,'更新要求')
        if any(k not in req for k in LABELS) or not re.search(r'分数|评分',req) or '理由' not in req:raise ValueError('更新要求中须明确逐维参考分数和理由进行改写')
    else:
        if not re.search(r'归纳|融合',task) or '提示' not in task:raise ValueError('融合任务须归纳问题类型并生成群智QA提示')
        if '簇' not in inputs or '参考答案' not in inputs or '提示' not in inputs:raise ValueError('融合输入应为簇编号和案例Q/参考A/最终提示，而非当前Q/top5')
        if not re.search(r'共同[^。\n]{0,10}问题类型|共同类型|问题类型归纳',text):raise ValueError('须归纳簇内共同问题类型并形成针对性要求')
        if re.search(r'产生参考答案|生成参考答案的QA提示',text):raise ValueError('参考答案独立提供，不能声称由最终QA提示产生')
        if re.search(r'不得[^。\n]{0,30}直接生成[^。\n]{0,40}作答提示|必须输出[^。\n]{0,40}生成器',section(text,'边界')):raise ValueError('成品融合指令执行时应直接输出簇级QA作答提示，不是继续输出提示生成器。删除边界中相反的生成层级要求；本指令输入簇内案例Q/参考A/最终提示，输出该类型的QA提示。')
    return text


def parse_grade_strict(text):
    text=normalize(text);start=text.find('{')
    if start<0:raise ValueError('No JSON in evaluation')
    value,end=json.JSONDecoder(strict=False).raw_decode(text[start:])
    if text[start+end:].strip():raise ValueError('Extra text after evaluation JSON')
    if not isinstance(value,dict) or set(value)!=set(DIMS):raise ValueError('Only three named dimensions allowed')
    for key in DIMS:
        item=value[key]
        if not isinstance(item,dict) or set(item)!={'score','reasoning'}:raise ValueError('Each dimension needs only score/reasoning')
        number=item['score']
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not 0<=number<=100:raise ValueError('Score outside0-100')
        if not isinstance(item['reasoning'],str) or not item['reasoning'].strip():raise ValueError('Missing score basis')
    return value


def grade_text_and_integrity(text):
    """Repair fixed-key delimiters/object closers, never score/reason values."""
    original=normalize(text)
    try:
        parse_grade_strict(original)
        return original,None
    except (ValueError,TypeError,KeyError):pass
    names='|'.join(DIMS)
    pattern=re.compile(r'("(?:'+names+r')"\s*:\s*\{\s*"score"\s*:\s*\d+(?:\.\d+)?\s*,\s*)("reasoning[ \t]*"[ \t]*[:：>]?[ \t]*"?)')
    matches=list(pattern.finditer(original))
    if len(matches)!=3:raise ValueError('Not an unambiguous fixed-field delimiter error')
    edits=[];repaired=original
    for m in reversed(matches):
        start,end=m.span(2);before=m.group(2);after='"reasoning": "'
        if before!=after:
            edits.append({'start':start,'end':end,'original':before,'repaired':after})
            repaired=repaired[:start]+after+repaired[end:]
    value=None
    for closers in ('','}','}}'):
        try:value=parse_grade_strict(repaired+closers)
        except (ValueError,TypeError,KeyError):continue
        if closers:
            edits.append({'start':len(original),'end':len(original),'original':'','repaired':closers})
            repaired+=closers
        break
    if value is None:
        # Escape only interior ASCII quotes inside three unambiguous,
        # completed reasoning lines. JSON decoding retains the original
        # literal quote characters, scores and all reason text.
        reason_lines=list(re.finditer(r'("reasoning"\s*:\s*")([^\r\n]*)(")(?=[ \t]*\r?\n[ \t]*\})',original))
        quote_edits=[]
        if len(reason_lines)==3:
            for m in reason_lines:
                body=m.group(2)
                for i,ch in enumerate(body):
                    if ch!='"':continue
                    backslashes=0;j=i-1
                    while j>=0 and body[j]=='\\':backslashes+=1;j-=1
                    if backslashes%2==0:quote_edits.append({'start':m.start(2)+i,'end':m.start(2)+i,'original':'','repaired':'\\'})
            candidate=original
            for edit in reversed(quote_edits):candidate=candidate[:edit['start']]+'\\'+candidate[edit['start']:]
            try:
                if quote_edits:value=parse_grade_strict(candidate)
            except (ValueError,TypeError,KeyError):value=None
            if value is not None:repaired=candidate;edits=quote_edits
    if value is None:
        # A completed one-line reasoning may end in a Chinese closing quote
        # instead of its required JSON delimiter. Insert the missing ASCII
        # delimiter; retain every original reason character, including ”.
        pattern_missing=re.compile(r'("reasoning"\s*:\s*")([^\r\n]*?)([ \t]*)(?=\r?\n[ \t]*\}[ \t]*(?:,|\r?\n))')
        missing=[]
        for m in pattern_missing.finditer(original):
            body=m.group(2)
            if body and not body.endswith('"') and body[-1] in '。！？；”’':
                try:json.loads('"'+body)
                except json.JSONDecodeError:missing.append({'start':m.end(2),'end':m.end(2),'original':'','repaired':'"'})
        if not missing:raise ValueError('Not repairable by fixed-key delimiters or final object closers')
        candidate=original
        for edit in reversed(missing):candidate=candidate[:edit['start']]+'"'+candidate[edit['start']:]
        value=parse_grade_strict(candidate)
        repaired=candidate;edits=missing
    original_scores={k:re.findall(r'"'+k+r'"\s*:\s*\{\s*"score"\s*:\s*([\d.]+)',original) for k in DIMS}
    assert all(len(original_scores[k])==1 and value[k]['score']==float(original_scores[k][0]) for k in DIMS)
    # Reconstruct byte-for-byte. All edits are anchored immediately after that
    # dimension's score and stop BEFORE its reason content; reasons are untouched.
    rebuilt=original
    for edit in sorted(edits,key=lambda e:e['start'],reverse=True):
        assert rebuilt[edit['start']:edit['end']]==edit['original']
        assert re.fullmatch(r'"reasoning[ \t]*"[ \t]*[:：>]?[ \t]*"?',edit['original']) or (edit['start']==len(original) and edit['original']=='' and edit['repaired'] in ('}','}}')) or (edit['original']=='' and edit['repaired']=='"' and original[edit['start']-1] in '。！？；”’' and original[edit['start']:].lstrip().startswith('}')) or (edit['original']=='' and edit['repaired']=='\\' and original[edit['start']]=='"' and any(m.start(2)<=edit['start']<m.end(2) for m in reason_lines))
        rebuilt=rebuilt[:edit['start']]+edit['repaired']+rebuilt[edit['end']:]
    assert rebuilt==repaired
    rule='missing_closing_reason_quote_insertion_only_v1' if any(e['original']=='' and e['repaired']=='"' for e in edits) else 'fixed_reasoning_delimiters_or_final_object_closers_only_v1'
    if any(e['repaired']=='\\' for e in edits):rule='interior_reason_quote_escaping_only_v1'
    proof={'rule':rule,'original_sha256':digest(text),
           'normalized_original_sha256':digest(original),'repaired_sha256':digest(repaired),
           'edits':sorted(edits,key=lambda e:e['start']),'scores_and_reason_content_unchanged':True}
    return repaired,proof


def parse_grade(text):
    canonical,_=grade_text_and_integrity(text)
    return parse_grade_strict(canonical)


def stop_decision(previous,current,min_gain=1.):
    if any(current[k]<previous[k] for k in DIMS):return 'dimension_declined'
    if statistics.mean(current.values())-statistics.mean(previous.values())<min_gain:return 'mean_gain_below_threshold'
    return None


def context_view(row):
    docs=row['context_documents'];assert len(docs)==5
    return [{'position':i+1,'chunk_id':d.get('chunk_id',''),'doc_title':d.get('doc_title',''),'text':d['text']} for i,d in enumerate(docs)]
def pure_input(row):
    return '[问题]\n'+row['question']+'\n\n[上下文]\n'+'\n\n'.join('[来源 '+str(d['position'])+'] '+d['doc_title']+'\n'+d['text'] for d in context_view(row))
def answer_messages(row,prompt=None):
    user=pure_input(row)
    if prompt:user+='\n\n[回答方法]\n'+prompt
    return [{'role':'user','content':user}]
def grade_payload(row,answer):
    return {'question':row['question'],'reference_answer':row['reference_answer'],'candidate_answer':answer,'contexts':context_view(row)}


def active_cluster_dir(cfg):
    if not cfg.get('collective_feedback',{}).get('enabled',False):return RUN/'clusters'
    from collective_feedback import active_cluster_dir as resolve
    return resolve(RUN,cfg)


class API:
    def __init__(self,config):self.config=config;self.gate=threading.BoundedSemaphore(config['workers'])
    def ask(self,key,messages,max_tokens=6500):
        payload={'model':self.config['model'],'messages':messages,'temperature':.1,'max_tokens':max_tokens,'stream':False,'seed':self.config['seed']}
        checkpoint=RUN/'request_checkpoints'/(digest(key)+'.json')
        if checkpoint.exists():
            r=read(checkpoint);assert r['request_sha256']==digest(payload),'Exact request changed';return r
        for attempt in range(3):
            p=RUN/'api_calls'/(str(time.time_ns())+'_'+digest(key)[:12]+'.json')
            record={'key':key,'request':payload,'request_sha256':digest(payload),'transport_attempt':attempt+1,'started_at':time.time()}
            try:
                with self.gate:
                    req=urllib.request.Request(self.config['base_url']+'/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
                    with urllib.request.urlopen(req,timeout=300) as res:record.update(http_status=res.status,response=json.loads(res.read()))
                choice=record['response']['choices'][0];content=choice['message'].get('content')
                if not isinstance(content,str) or not content.strip():raise ValueError('Empty response')
                if choice.get('finish_reason')=='length':raise ValueError('Truncated response')
                record.update(seconds=time.time()-record['started_at']);save(p,record)
                result={'text':content,'raw_record':str(p),'request_sha256':digest(payload)};save(checkpoint,result);return result
            except Exception as exc:
                record.update(error=type(exc).__name__+': '+str(exc),seconds=time.time()-record['started_at']);save(p,record)
                if isinstance(exc,urllib.error.HTTPError) and exc.code not in (429,500,502,503,504):raise
                if attempt==2:raise
                time.sleep(2*(attempt+1))


def generated(api,key,meta,payload,validator):
    attempts=[];errors=[];prior=''
    for i in range(3):
        user=json.dumps(payload,ensure_ascii=False)
        if errors:user+='\n[原要求校验反馈]\n'+errors[-1]+'\n[待修正输出]\n'+prior+'\n只纠正与原meta要求不一致的地方，不填入具体案例事实。'
        result=api.ask(key+':generation:'+str(i),[{'role':'system','content':meta},{'role':'user','content':user}])
        attempts.append(result['raw_record']);prior=result['text']
        try:return {'text':validator(prior),'raw_record':result['raw_record'],'attempts':attempts,'sha256':digest(validator(prior)),'input':payload}
        except ValueError as exc:
            errors.append(str(exc));save(RUN/'validation_failures'/(digest(key)+'.json'),{'key':key,'errors':errors,'raw_records':attempts})
    raise RuntimeError('Prompt compliance failed: '+key+' '+errors[-1])


def generate_stages(api,cfg):
    rubric=contract(cfg['metas']['evaluation']);records={}
    def one(kind):
        p=RUN/'stages'/(kind+'.json')
        if p.exists():return kind,read(p)
        payload={'task':'执行meta要求，生成本阶段提示正文；不输入具体案例，不再生成meta-prompt。','evaluation_criteria':rubric}
        repair_path=RUN/'stage_semantic_repair_inputs.json'
        repairs=read(repair_path) if repair_path.exists() else {}
        key='stage:'+kind
        if kind in repairs:
            payload.update(semantic_compliance_correction=repairs[kind])
            key+=':semantic_contract_'+str(repairs[kind].get('revision','v1'))
        value=generated(api,key,cfg['metas'][kind],payload,lambda t:validate_stage(kind,t,rubric));save(p,value);return kind,value
    with ThreadPoolExecutor(max_workers=4) as pool:
        for kind,value in pool.map(one,('qa','evaluation','update','collective')):records[kind]=value
    artifact={k:r['sha256'] for k,r in records.items()};save(RUN/'stage_review.json',{'artifact_sha256':digest(artifact),'stages':artifact,'status':'needs_semantic_review'})
    approval=RUN/'stage_approval.json'
    if not approval.exists():mark('awaiting_stage_review');return None
    assert read(approval)['artifact_sha256']==digest(artifact)
    return records


def grade(api,stages,key,row,answer):
    payload=grade_payload(row,answer)
    result=api.ask(key+':judge',[{'role':'system','content':stages['evaluation']['text']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=3600)
    try:
        canonical,format_integrity=grade_text_and_integrity(result['text'])
        value=parse_grade_strict(canonical)
    except (ValueError,TypeError,KeyError):
        # Syntax-only recovery may not change any named score or reason.
        original={k:re.findall(r'"'+k+r'"\s*:\s*\{\s*"score"\s*:\s*([\d.]+)',result['text']) for k in DIMS}
        if any(len(v)!=1 for v in original.values()):raise
        fix=api.ask(key+':json-syntax-only',[{'role':'system','content':'只修复JSON语法。不得重新评价，不改变任何维度分数、理由文字或字段。只输出合法JSON。'},{'role':'user','content':result['text']}],max_tokens=3600)
        value=parse_grade(fix['text'])
        assert all(value[k]['score']==float(original[k][0]) for k in DIMS),'Syntax repair changed score'
        raise RuntimeError('Score JSON repair requires independent word-preservation audit before use: '+fix['raw_record'])
    return {'dimensions':{k:value[k]['score'] for k in DIMS},'reasons':{k:value[k]['reasoning'] for k in DIMS},'raw_output':value,'raw_record':result['raw_record'],'evaluation_input':payload,'judge_sha256':stages['evaluation']['sha256'],'repetitions':1,'format_integrity':format_integrity}


def train_one(api,stages,cfg,row):
    path=RUN/'training/cases'/(row['id']+'.json')
    state=read(path) if path.exists() else {'id':row['id'],'question':row['question'],'history':[],'status':'running'}
    if state['status']=='complete':return state
    accepted=0
    for iteration in range(cfg['max_rounds']+1):
        if iteration<len(state['history']):
            if state['history'][iteration].get('accepted'):accepted=iteration
            continue
        p=RUN/'training/versions'/row['id']/(str(iteration)+'.json')
        if p.exists():version=read(p)
        else:
            if iteration==0:prompt=stages['qa']['text'];origin={'shared_initial_sha256':stages['qa']['sha256']}
            else:
                previous=state['history'][accepted]
                payload={**grade_payload(row,previous['answer']),'current_qa_prompt':previous['prompt'],'evaluation':previous['evaluation']['raw_output'],'evaluation_criteria':contract(cfg['metas']['evaluation'])}
                update_key='train:'+row['id']+':update:'+str(iteration)
                try:origin=generated(api,update_key,stages['update']['text'],payload,validate_qa)
                except RuntimeError as exc:
                    if not str(exc).startswith('Prompt compliance failed: '):raise
                    failure=RUN/'validation_failures'/(digest(update_key)+'.json')
                    assert failure.exists()
                    state.update(status='complete',selected_iteration=accepted,prompt=previous['prompt'],
                                 stop_reason='update_generation_noncompliant',rejected_update=read(failure))
                    save(path,state);print('CASE_COMPLETE '+row['id']+' update_generation_noncompliant',flush=True);return state
                prompt=origin['text']
                if prompt==previous['prompt']:
                    state.update(status='complete',selected_iteration=accepted,prompt=prompt,stop_reason='prompt_unchanged');save(path,state);return state
            version={'iteration':iteration,'prompt':prompt,'origin':origin,'answer':None,'evaluation':None};save(p,version)
        if version['answer'] is None:
            result=api.ask('train:'+row['id']+':answer:'+str(iteration),answer_messages(row,version['prompt']));version.update(answer=result['text'],answer_raw_record=result['raw_record']);save(p,version)
        if version['evaluation'] is None:version['evaluation']=grade(api,stages,'train:'+row['id']+':'+str(iteration),row,version['answer']);save(p,version)
        decision=stop_decision(state['history'][accepted]['evaluation']['dimensions'],version['evaluation']['dimensions'],cfg['min_gain']) if iteration else None
        version['accepted']=decision is None;save(p,version);state['history'].append(version)
        if decision is None:accepted=iteration
        if decision or all(s==100 for s in version['evaluation']['dimensions'].values()) or iteration==cfg['max_rounds']:
            state.update(status='complete',selected_iteration=accepted,prompt=state['history'][accepted]['prompt'],stop_reason=decision or ('all_dimensions_100' if all(s==100 for s in version['evaluation']['dimensions'].values()) else 'safety_round_limit'))
            save(path,state);print('CASE_COMPLETE '+row['id']+' '+state['stop_reason'],flush=True);return state
        save(path,state)
    raise AssertionError('Unreachable training state')


def train(api,stages,cfg,rows):
    mark('case_optimization',total=len(rows))
    values={}
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures={pool.submit(train_one,api,stages,cfg,r):r for r in rows}
        for future in as_completed(futures):
            value=future.result();values[value['id']]=value;mark('case_optimization',completed=len(values),total=len(rows))
    return [values[r['id']] for r in rows]


def cluster(cfg,train_rows,test_rows,cases):
    p=RUN/'clustering.json'
    if p.exists():return read(p)
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from embedding_runtime import EmbeddingModel
    mark('embedding_and_clustering',training_n=len(cases))
    model=EmbeddingModel(cfg['embedding_model'],device='cpu')
    def unit(a):return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    q=unit(np.asarray(model.encode([r['question'] for r in train_rows]),dtype=float))
    prompts=unit(np.asarray(model.encode([c['prompt'] for c in cases]),dtype=float))
    x=unit(np.concatenate((q,prompts),axis=1));candidates=[];assignments={}
    for k in cfg['clustering_candidates']:
        if k>=len(cases):continue
        fitted=KMeans(n_clusters=k,n_init=20,random_state=cfg['seed']).fit(x);labels=fitted.labels_
        if len(set(labels))<2 or len(set(labels))!=k:continue
        score=float(silhouette_score(x,labels,metric='cosine'));sizes=[int(sum(labels==i)) for i in range(k)]
        candidates.append({'k':k,'silhouette':score,'sizes':sizes,'minimum_size_ok':min(sizes)>=2});assignments[k]=labels
    assert candidates,'No valid clustering'
    eligible=[c for c in candidates if c['minimum_size_ok']] or candidates
    selected=max(eligible,key=lambda c:(round(c['silhouette'],12),-c['k']));labels=assignments[selected['k']]
    order=sorted(set(labels),key=lambda c:min(train_rows[i]['id'] for i in range(len(labels)) if labels[i]==c));remap={old:i for i,old in enumerate(order)}
    labels=np.asarray([remap[c] for c in labels]);centers=unit(np.asarray([q[labels==i].mean(axis=0) for i in range(len(order))]))
    test=unit(np.asarray(model.encode([r['question'] for r in test_rows]),dtype=float));similarities=test@centers.T
    result={'k':len(order),'selected':selected,'candidates':candidates,'train_labels':{r['id']:int(labels[i]) for i,r in enumerate(train_rows)},'test_routes':{r['id']:{'cluster_id':int(np.argmax(similarities[i])),'similarity':float(np.max(similarities[i]))} for i,r in enumerate(test_rows)},'route_rule':'nearest normalized training-question centroid of prompt-derived clusters; question only; no reference or score','created_at':time.time()}
    save(p,result);np.savez_compressed(RUN/'clustering_vectors.npz',questions=q,prompts=prompts,combined=x,centers=centers,test_questions=test)
    return result


def fuse(api,stages,cfg,train_rows,cases,clustering):
    mark('cluster_fusion',k=clustering['k']);records={}
    by_id={c['id']:c for c in cases}
    def one(cluster_id):
        p=RUN/'clusters'/('cluster_'+str(cluster_id)+'.json')
        if p.exists():return read(p)
        members=[r for r in train_rows if clustering['train_labels'][r['id']]==cluster_id]
        payload={'cluster_id':cluster_id,'cases':[{'question':r['question'],'reference_answer':r['reference_answer'],'final_qa_prompt':by_id[r['id']]['prompt']} for r in members],'evaluation_criteria':contract(cfg['metas']['evaluation'])}
        key='cluster:'+str(cluster_id)
        revision_path=RUN/'training_bank_revision.json'
        if revision_path.exists():key+=':bank_revision_'+str(read(revision_path)['revision'])
        repair_path=RUN/'cluster_semantic_repair_inputs.json'
        repairs=read(repair_path) if repair_path.exists() else {}
        correction=repairs.get(str(cluster_id),repairs.get('default'))
        if correction:
            payload['semantic_compliance_correction']=correction
            key+=':semantic_contract_'+str(correction.get('revision','v1'))
        value=generated(api,key,stages['collective']['text'],payload,validate_qa)
        value.update(cluster_id=cluster_id,member_ids=[r['id'] for r in members],question_type=section(value['text'],'任务').split('。')[0][:70]);save(p,value);return value
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        for value in pool.map(one,range(clustering['k'])):records[str(value['cluster_id'])]=value
    artifact={k:v['sha256'] for k,v in records.items()};save(RUN/'cluster_review.json',{'artifact_sha256':digest(artifact),'clusters':artifact,'k':clustering['k'],'status':'needs_semantic_review'})
    if not (RUN/'cluster_approval.json').exists():mark('awaiting_cluster_review',k=clustering['k']);return None
    assert read(RUN/'cluster_approval.json')['artifact_sha256']==digest(artifact)
    return records


def infer(api,stages,cfg,tests,clusters,clustering):
    artifact={'judge':stages['evaluation']['sha256'],'clusters':{k:v['sha256'] for k,v in clusters.items()},'k':clustering['k'],'selection_rule':f'No performance selection. All{len(tests)} both groups.','frozen_at':time.time()}
    if cfg.get('collective_feedback',{}).get('enabled',False):
        active_cluster_dir(cfg)
        artifact['feedback_manifest_sha256']=file_sha(RUN/'collective_feedback/final_manifest.json')
    p=RUN/'inference_frozen.json'
    if not p.exists():save(p,artifact)
    else:
        prior=read(p)
        for key in ('judge','clusters','k','selection_rule',*(['feedback_manifest_sha256'] if 'feedback_manifest_sha256' in artifact else [])):assert prior[key]==artifact[key]
    mark('full_inference_and_evaluation',total=2*len(tests))
    def one(group,row):
        path=RUN/'settings'/group/'records'/(row['id']+'.json')
        if path.exists() and read(path).get('evaluation'):return read(path)
        cluster_id=clustering['test_routes'][row['id']]['cluster_id'];prompt=clusters[str(cluster_id)]['text'] if group=='collective' else None
        value=read(path) if path.exists() else {'id':row['id'],'question':row['question'],'reference_answer':row['reference_answer'],'actual_top5':context_view(row),'group':group,'cluster_id':cluster_id,'question_type':clusters[str(cluster_id)]['question_type'],'prompt':prompt,'answer_calls':1,'answer':None,'evaluation':None}
        if value['answer'] is None:
            result=api.ask('final:'+group+':'+row['id']+':answer',answer_messages(row,prompt));value.update(answer=result['text'],answer_raw_record=result['raw_record']);save(path,value)
        if value['evaluation'] is None:value['evaluation']=grade(api,stages,'final:'+group+':'+row['id'],row,value['answer']);save(path,value)
        return value
    completed=0
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        # Interleave the two methods across questions, no method-specific scoring.
        futures=[pool.submit(one,g,r) for r in tests for g in cfg['groups']]
        for future in as_completed(futures):future.result();completed+=1;mark('full_inference_and_evaluation',completed=completed,total=2*len(tests))


def main():
    cfg=read(ROOT/'config.json');frozen=read(RUN/'frozen_manifest.json')
    for name,h in frozen['source_sha256'].items():assert file_sha(ROOT/name)==h,'Changed code '+name
    for name,h in frozen['input_sha256'].items():assert file_sha(RUN/'shared'/name)==h,'Changed data '+name
    assert cfg==frozen['config'] and cfg['judge_repetitions']==1
    api=API(cfg);stages=generate_stages(api,cfg)
    if stages is None:return
    train_rows=read(RUN/'shared/train_top5.json');tests=read(RUN/'shared/test_top5.json')
    cases=train(api,stages,cfg,train_rows);clustering=cluster(cfg,train_rows,tests,cases);clusters=fuse(api,stages,cfg,train_rows,cases,clustering)
    if clusters is None:return
    if cfg.get('collective_feedback',{}).get('enabled',False):
        from collective_feedback import optimize
        clusters=optimize(api,cfg,train_rows,cases,clustering,clusters,sys.modules[__name__])
        if clusters is None:return
        from audit_collective_feedback import audit
        audit(RUN,cfg,sys.modules[__name__])
    infer(api,stages,cfg,tests,clusters,clustering)
    mark('mechanical_metrics_and_report')
    subprocess.run([cfg['machine_python'],str(ROOT/'machine.py')],check=True)
    mark('complete',questions=len(tests),groups=cfg['groups'],k=clustering['k'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        mark('failed',error=type(exc).__name__+': '+str(exc));raise
