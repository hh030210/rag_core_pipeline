"""Read-only lineage and decision audit for qualitative cluster feedback."""
import json
from pathlib import Path

import collective_feedback as f
from collective_feedback_prompts import DIAGNOSIS_META, DIAGNOSIS_SPEC, UPDATE_META, UPDATE_SPEC


def audit(run, cfg, runtime, write_result=True):
    r=runtime;run=Path(run);base=run/'collective_feedback'
    if not f.enabled(cfg):return {'enabled':False}
    rows=r.read(run/'shared/train_top5.json');by_id={str(row['id']):row for row in rows}
    clustering=r.read(run/'clustering.json');manifest=r.read(base/'final_manifest.json')
    frozen=r.read(base/'input_frozen.json');stage_ids={};replays=0;comparisons=0;repairs=0;unverifiable=[]
    if frozen['training_ids']!=list(by_id) or frozen['training_inputs_sha256']!=r.digest(rows):raise ValueError('Feedback training inputs changed')
    selected={}
    for ident in by_id:
        case=r.read(run/'training/cases'/(ident+'.json'));v=case['history'][case['selected_iteration']]
        selected[ident]={'prompt':v['prompt'],'answer':v['answer']}
    if frozen['selected_cases_sha256']!=r.digest(selected):raise ValueError('Selected case-answer bank changed')

    def raw(path):
        p=run/'api_calls'/Path(path).name;v=r.read(p)
        if v.get('error') or v['request']['model']!=cfg['model']:raise ValueError('Invalid feedback source request')
        return v

    def failed_members(cid,number,records,members,rounds,prompt):
        missing=set(members)-set(records)
        if not missing:raise ValueError('Replay was marked unverifiable but no members are missing')
        for ident in sorted(missing):
            key='feedback:cluster:'+cid+':round:'+str(number)+':case:'+ident
            checkpoint=r.read(run/'request_checkpoints'/(r.digest(key+':answer')+'.json'))
            a=raw(checkpoint['raw_record']);answer=checkpoint['text']
            if a['request']['messages']!=r.answer_messages(by_id[ident],prompt) or a['response']['choices'][0]['message']['content']!=answer:
                raise ValueError('Unverifiable replay changed its answer/input')
            previous=None if number==0 else {'prompt':rounds[number-1][ident]['prompt'],'answer':rounds[number-1][ident]['answer']}
            payload=f.diagnostic_payload(by_id[ident],selected[ident]['prompt'],selected[ident]['answer'],prompt,answer,previous,r)
            revision=frozen.get('diagnostic_protocol_revision')
            diagnostic_key=key+(':protocol_'+str(revision) if revision else '')
            failure=r.read(run/'validation_failures'/(r.digest(diagnostic_key+':diagnosis')+'.json'))
            if len(failure['raw_records'])!=3:raise ValueError('Unverifiable diagnosis lacks bounded attempts')
            errors=[];stage=r.read(base/'stages/diagnosis.json')
            for path in failure['raw_records']:
                attempt=raw(path)
                if attempt['request']['messages']!=f.diagnosis_messages(stage['text'],payload,errors):raise ValueError('Failure recovery input changed')
                try:
                    value=json.loads(r.normalize(attempt['response']['choices'][0]['message']['content']));f.validate_diagnosis(value,payload)
                except (ValueError,TypeError,KeyError) as exc:errors.append({'error':str(exc),'raw_record':path})
                else:raise ValueError('A valid diagnosis was discarded as unavailable')
            unverifiable.append({'cluster_id':cid,'round':number,'id':ident,'retained_previous_prompt':True,'raw_attempts':failure['raw_records']})

    for kind,meta,contract in (('diagnosis',DIAGNOSIS_META,DIAGNOSIS_SPEC),('update',UPDATE_META,UPDATE_SPEC)):
        stage=r.read(base/'stages'/(kind+'.json'));v=raw(stage['raw_record'])
        if stage['sha256']!=r.digest(stage['text']):raise ValueError('Feedback contract changed')
        f.validate_feedback_stage(kind,stage['text'],r)
        if v['request']['messages'][0]!={'role':'system','content':meta}:raise ValueError('Stage was not generated from the declared meta')
        if r.normalize(v['response']['choices'][0]['message']['content'])!=stage['text']:raise ValueError('Generated stage text changed')
        stage_ids[kind]=stage['sha256']
    if stage_ids!=manifest['stage_sha256']:raise ValueError('Feedback stage lineage mismatch')
    if cfg['collective_feedback'].get('require_review',True):
        if r.read(run/'feedback_stage_approval.json')['artifact_sha256']!=r.digest(stage_ids):raise ValueError('Stage review mismatch')
    for cid,state in manifest['states'].items():
        original=r.read(run/'clusters'/('cluster_'+cid+'.json'))
        members=[ident for ident in by_id if str(clustering['train_labels'][ident])==cid]
        if members!=state['member_ids'] or members!=original['member_ids']:raise ValueError('Cluster membership changed')
        if r.digest(original['text'])!=frozen['initial_clusters'][cid]:raise ValueError('Original fusion prompt was overwritten')
        rounds={}
        for path in (base/'clusters'/cid/'rounds').glob('*/cases/*.json'):
            number=int(path.parents[1].name);value=r.read(path);ident=value['id']
            if ident not in members:raise ValueError('Non-training case used in feedback')
            rounds.setdefault(number,{})[ident]=value
        blocked={event['round'] for event in state['history'] if event['reason'] in {'initial_replay_unverifiable','candidate_replay_unverifiable'}}
        for number in blocked:rounds.setdefault(number,{})
        for number,records in sorted(rounds.items()):
            if set(records)!=set(members) and number not in blocked:raise ValueError('Not all cluster members were checked')
            round_prompt=original['text'] if number==0 else r.read(base/'clusters'/cid/'rounds'/str(number)/'candidate.json')['text']
            if number in blocked:failed_members(cid,number,records,members,rounds,round_prompt)
            for ident,value in records.items():
                if value['prompt']!=round_prompt:raise ValueError('Replay did not use the declared candidate prompt')
                previous=None if number==0 else {'prompt':rounds[number-1][ident]['prompt'],'answer':rounds[number-1][ident]['answer']}
                payload=f.diagnostic_payload(by_id[ident],selected[ident]['prompt'],selected[ident]['answer'],value['prompt'],value['answer'],previous,r)
                if payload!=value['diagnosis_input']:raise ValueError('Diagnostic inputs differ from actual replay')
                a=raw(value['answer_raw_record'])
                if a['request']['messages']!=r.answer_messages(by_id[ident],value['prompt']):raise ValueError('Replay answer saw extra data or multiple-message instructions')
                if a['response']['choices'][0]['message']['content']!=value['answer']:raise ValueError('Replay answer changed')
                d=raw(value['diagnosis_raw_record']);stage=r.read(base/'stages/diagnosis.json')
                attempts=value.get('diagnosis_attempts',[value['diagnosis_raw_record']]);errors=[]
                if not 1<=len(attempts)<=3 or attempts[-1]!=value['diagnosis_raw_record']:raise ValueError('Unbounded or wrong diagnosis selection')
                for index,path in enumerate(attempts):
                    attempt=raw(path)
                    expected=f.diagnosis_messages(stage['text'],payload,errors)
                    if attempt['request']['messages']!=expected:raise ValueError('Diagnostic request changed or included old scores')
                    try:
                        parsed=json.loads(r.normalize(attempt['response']['choices'][0]['message']['content']));f.validate_diagnosis(parsed,payload)
                    except (ValueError,TypeError,KeyError) as exc:
                        if index==len(attempts)-1:raise ValueError('Last diagnosis invalid')
                        errors.append({'error':str(exc),'raw_record':path})
                    else:
                        if index!=len(attempts)-1:raise ValueError('Earlier valid diagnosis discarded')
                actual=json.loads(r.normalize(d['response']['choices'][0]['message']['content']))
                f.validate_diagnosis(actual,payload)
                if actual!=value['diagnosis']:raise ValueError('Diagnostic outcome changed')
                replays+=1;comparisons+=1
        accepted_round=0
        for event in state['history']:
            number=event['round']
            if event['reason']=='initial_replay_unverifiable':
                if number!=0 or event['accepted']:raise ValueError('Invalid initial replay fallback')
                continue
            if event['reason']=='candidate_noncompliant':
                failure=r.read(run/'validation_failures'/(r.digest('feedback:cluster:'+cid+':update:'+str(number))+'.json'))
                if len(failure['raw_records'])!=3:raise ValueError('Unbounded or undocumented generation failure')
                for source in failure['raw_records']:
                    try:r.validate_qa(raw(source)['response']['choices'][0]['message']['content'])
                    except ValueError:pass
                    else:raise ValueError('A valid prompt was discarded as noncompliant')
                continue
            if event['reason']=='candidate_replay_unverifiable':
                if event['accepted']:raise ValueError('An unverifiable candidate was accepted')
            else:
                decision=f.acceptance(rounds[number-1],rounds[number])
                if any(event[k]!=v for k,v in decision.items()):raise ValueError('Acceptance did not follow qualitative comparison policy')
            candidate=r.read(base/'clusters'/cid/'rounds'/str(number)/'candidate.json')
            r.validate_qa(candidate['text']);request=raw(candidate['raw_record'])['request']['messages']
            update_stage=r.read(base/'stages/update.json')
            if request[0]!={'role':'system','content':update_stage['text']}:raise ValueError('Candidate used another updater')
            sent,_=json.JSONDecoder().raw_decode(request[1]['content'])
            expected_input={'cluster_id':cid,'current_collective_prompt':rounds[number-1][members[0]]['prompt'],
                'training_cases':[{'id':ident,**rounds[number-1][ident]['diagnosis_input'],'diagnosis':rounds[number-1][ident]['diagnosis']} for ident in members]}
            if sent!=expected_input or candidate['input']!=expected_input:raise ValueError('Updater did not see the whole original training cluster')
            if candidate['text']!=r.normalize(raw(candidate['raw_record'])['response']['choices'][0]['message']['content']):raise ValueError('Candidate text changed')
            if event['accepted']:accepted_round=number;repairs+=1
        value=r.read(base/'final_clusters'/('cluster_'+cid+'.json'))
        expected_prompt=original['text'] if accepted_round==0 else r.read(base/'clusters'/cid/'rounds'/str(accepted_round)/'candidate.json')['text']
        if state['selected_round']!=accepted_round or value['text']!=expected_prompt or value['sha256']!=r.digest(expected_prompt):raise ValueError('Final prompt is not the last accepted candidate')
        if manifest['clusters'][cid]!=value['sha256']:raise ValueError('Final prompt hash mismatch')
    f.active_cluster_dir(run,cfg)
    result={'passed':True,'training_ids':list(by_id),'clusters':len(manifest['clusters']),
            'replayed_answers_checked':replays,'qualitative_comparisons_checked':comparisons,'accepted_repairs':repairs,
            'numeric_scoring_used':False,'test_data_used_for_feedback':False,'initial_fusion_artifacts_preserved':True}
    result['unverifiable_replays']=unverifiable
    result['all_feedback_diagnoses_valid']=not unverifiable
    if write_result:r.save(base/'audit.json',result)
    return result
