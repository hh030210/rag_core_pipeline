"""Independent raw-input, score, training-selection and feedback audit."""
import json
from pathlib import Path
import sys

from rejudge_five_runner import parse_grade, DIMS
import five_metrics_workflow as f


def audit(run,training_only=False):
    run=Path(run); code=run/'code_snapshot'
    config=json.loads((code/'config.json').read_text(encoding='utf-8'))
    knowledge=None
    ten_point=config.get('protocol','').startswith('ten_point_')
    if ten_point:
        import ten_point_workflow as knowledge
        r=knowledge.bind()
    elif config.get('allow_internal_knowledge'):
        import knowledge_fallback_workflow as knowledge
        r=knowledge.bind()
    else:r=f.bind()
    r.RUN=run; r.ROOT=code
    dims=knowledge.DIMS if knowledge else DIMS
    cfg=r.read(code/'config.json'); frozen=r.read(run/'frozen_manifest.json')
    assert cfg==frozen['config'] and cfg['groups']==['collective'] and cfg['model']=='DeepSeek-V4-Pro' and cfg['workers']==8
    for name,h in frozen['source_sha256'].items(): assert r.file_sha(code/name)==h, name
    for name,h in frozen['input_sha256'].items(): assert r.file_sha(run/'shared'/name)==h, name
    train=r.read(run/'shared/train_top5.json'); tests=r.read(run/'shared/test_top5.json')
    assert len(train)==23 and len(tests)==378
    assert not {v['question'].strip() for v in train}&{v['question'].strip() for v in tests}
    judge=r.read(run/'stages/evaluation.json')
    assert judge['sha256']==r.digest(judge['text']) and r.contract(cfg['metas']['evaluation']) in judge['text']
    raw_count=0; grade_count=0; train_versions=0; final_count=0
    def raw(path):
        nonlocal raw_count
        v=r.read(run/'api_calls'/Path(path).name)
        assert not v.get('error') and v['request']['model']==cfg['model']
        assert v['response'].get('model')==cfg['model'], 'Unexpected response model identity'
        assert v['request_sha256']==r.digest(v['request'])
        raw_count+=1;return v
    def answer(row,version,prompt):
        v=raw(version['answer_raw_record'])
        assert v['request']['messages']==r.answer_messages(row,prompt)
        assert v['response']['choices'][0]['message']['content']==version['answer']
        return v
    def grade(row,version):
        nonlocal grade_count
        if knowledge:
            e=version['evaluation']
            payloads=knowledge.view_payloads(row,version['answer'],final=e['metric_phase']=='final') if ten_point else knowledge.view_payloads(row,version['answer'])
            expected_dims=knowledge.DIMS if not ten_point or e['metric_phase']=='final' else knowledge.PRIMARY
            merged={}
            assert e['physical_view_isolation'] and e['evaluation_input']==payloads and e['repetitions']==1
            assert set(e['views'])==set(payloads) and e['judge_sha256']==judge['sha256']
            for view,payload in payloads.items():
                item=e['views'][view];v=raw(item['raw_record'])
                assert item['input']==payload and item['judge_sha256']==judge['sha256']
                assert v['request']['messages']==[{'role':'system','content':judge['text']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
                parsed,proof=knowledge.parse_view_with_proof(v['response']['choices'][0]['message']['content'],view)
                assert parsed==item['value'] and item.get('format_integrity')==proof;merged.update(parsed)
            assert e['raw_output']==merged and e['raw_records']==[e['views'][view]['raw_record'] for view in payloads]
            assert e['dimensions']=={key:merged[key]['score'] for key in expected_dims} and e['reasons']=={key:merged[key]['reasoning'] for key in expected_dims}
            grade_count+=1;return
        e=version['evaluation']; v=raw(e['raw_record']);payload=r.grade_payload(row,version['answer'])
        assert v['request']['messages']==[{'role':'system','content':judge['text']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        value=parse_grade(v['response']['choices'][0]['message']['content'])
        assert e['raw_output']==value and e['judge_sha256']==judge['sha256'] and e['repetitions']==1
        assert e['dimensions']=={d:value[d]['score'] for d in DIMS}
        assert e['reasons']=={d:value[d]['reasoning'] for d in DIMS}
        grade_count+=1
    stage_hashes={}
    for kind in ('qa','evaluation','update','collective'):
        stage=r.read(run/'stages'/(kind+'.json'));source=raw(stage['raw_record'])
        assert source['request']['messages'][0]=={'role':'system','content':cfg['metas'][kind]}
        assert r.normalize(source['response']['choices'][0]['message']['content'])==stage['text']
        assert stage['sha256']==r.digest(stage['text'])
        stage_hashes[kind]=stage['sha256']
    assert r.read(run/'stage_approval.json')['artifact_sha256']==r.digest(stage_hashes)
    for row in train:
        case=r.read(run/'training/cases'/(row['id']+'.json'))
        assert case['status']=='complete'
        accepted=0
        for i,version in enumerate(case['history']):
            assert i==version['iteration'];answer(row,version,version['prompt']);grade(row,version);train_versions+=1
            if knowledge and i==0:assert version['prompt']==r.read(run/'stages/qa.json')['text']
            if i:
                previous=case['history'][accepted]
                decision=f.stop_decision(previous['evaluation']['dimensions'],version['evaluation']['dimensions'],cfg['min_gain'])
                assert version['accepted']==(decision is None)
                if version.get('origin',{}).get('raw_record'):
                    source=raw(version['origin']['raw_record'])
                    expected={**r.grade_payload(row,previous['answer']),'current_qa_prompt':previous['prompt'],
                        'evaluation':previous['evaluation']['raw_output'],'evaluation_criteria':r.contract(cfg['metas']['evaluation'])}
                    sent,_=json.JSONDecoder().raw_decode(source['request']['messages'][1]['content'])
                    assert sent==expected
            if version['accepted']:accepted=i
        assert case['selected_iteration']==accepted and case['prompt']==case['history'][accepted]['prompt']
    if training_only:
        result={'passed':True,'training_n':len(train),'training_versions_checked':train_versions,
            'numeric_evaluations_checked':grade_count,'raw_calls_checked':raw_count,
            'judge_sha256':judge['sha256'],'physically_isolated_evaluation_views':bool(knowledge),
            'metric_keys':list(knowledge.PRIMARY if ten_point else dims),'final_metric_keys':list(dims)}
        r.save(run/'reports/训练来源与隔离视图核验.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result
    from audit_collective_feedback import audit as audit_feedback
    by_id={row['id']:row for row in train}
    clustering=r.read(run/'clustering.json')
    for cid in range(clustering['k']):
        cluster=r.read(run/'clusters'/('cluster_'+str(cid)+'.json'))
        members=[row for row in train if clustering['train_labels'][row['id']]==cid]
        assert cluster['member_ids']==[row['id'] for row in members]
        source=raw(cluster['raw_record']);sent,_=json.JSONDecoder().raw_decode(source['request']['messages'][1]['content'])
        assert source['request']['messages'][0]=={'role':'system','content':r.read(run/'stages/collective.json')['text']}
        assert sent['cases']==[{'question':row['question'],'reference_answer':row['reference_answer'],
            'final_qa_prompt':r.read(run/'training/cases'/(row['id']+'.json'))['prompt']} for row in members]
        assert sent['evaluation_criteria']==r.contract(cfg['metas']['evaluation'])
        assert r.normalize(source['response']['choices'][0]['message']['content'])==cluster['text']
    feedback=audit_feedback(run,cfg,r,write_result=False)
    frozen_inference=r.read(run/'inference_frozen.json');clustering=r.read(run/'clustering.json')
    assert frozen_inference['groups']==['collective'] and frozen_inference['judge']==judge['sha256']
    assert frozen_inference['test_inputs_sha256']==r.digest(tests)
    assert frozen_inference['feedback_manifest_sha256']==r.file_sha(run/'collective_feedback/final_manifest.json')
    failures=[]
    for row in tests:
        p=run/'settings/collective/records'/(row['id']+'.json')
        if not p.exists():failures.append({'id':row['id'],'status':'missing'});continue
        v=r.read(p);cid=str(clustering['test_routes'][row['id']]['cluster_id'])
        cluster=r.read(run/'collective_feedback/final_clusters'/('cluster_'+cid+'.json'))
        assert v['prompt']==cluster['text'] and cluster['sha256']==frozen_inference['clusters'][cid]
        if v.get('answer'):
            a=answer(row,v,cluster['text']);assert a['started_at']>=frozen_inference['frozen_at'];final_count+=1
        if v.get('evaluation'):grade(row,v)
        else:failures.append({'id':row['id'],'status':v.get('status'),'error':v.get('error')})
    assert not (run/'settings/bare').exists(), 'No unrequested baseline'
    structural_recovery=None
    ten_folder=run/'technical_repairs/ten_point_structural_recovery'
    if (ten_folder/'proof.json').exists():
        proof=r.read(ten_folder/'proof.json')
        assert ten_point and proof['judge_sha256']==judge['sha256']
        assert r.file_sha(code/'config.json')==proof['config_sha256']
        assert r.file_sha(run/'inference_frozen.json')==proof['inference_frozen_sha256']
        assert r.read(ten_folder/'frozen_manifest.json')['config']==cfg
        for relative,h in proof['protected_sha256'].items():assert r.file_sha(run/relative)==h,relative
        eligible={v['id']:v for v in proof['eligible']}
        for p in (ten_folder/'failed_records').glob('*.json'):
            before=r.read(p);after=r.read(run/'settings/collective/records'/p.name)
            for key in ('answer','answer_raw_record','prompt','question','reference_answer','actual_top5'):assert before.get(key)==after.get(key),(p.name,key)
            if before['id'] not in eligible:assert before==after
        for ident,item in eligible.items():
            key='final:collective:'+ident
            cache=r.read(run/'evaluation_views'/(r.digest(key+':'+item['view'])+'.json'))
            assert cache['raw_record']==item['raw_record'] and cache['value']==item['original_metric_values']
            assert cache['format_integrity']==item['format_proof']
        new_raw=[r.read(p) for p in (run/'api_calls').glob('*.json') if p.name not in proof['original_raw_files']]
        assert all(v['key'] in proof['allowed_new_request_keys'] and ':judge:' in v['key'] for v in new_raw)
        current_result=r.read(run/'results.json')
        assert current_result['valid_answers']==proof['original_valid_answers']
        assert current_result['machine_metrics']==proof['original_machine_metrics']
        structural_recovery={'kind':'ten_point_named_structural','eligible_ids':list(eligible),
            'recovered_evaluations':sum(bool(r.read(run/'settings/collective/records'/(ident+'.json')).get('evaluation')) for ident in eligible),
            'original_scores_and_reasons_preserved':True,'new_raw_calls':len(new_raw),
            'no_answer_regeneration':True,'all_machine_metrics_unchanged':True,'unsafe_failures_not_resampled':proof['unsafe']}
    identity_folder=run/'technical_repairs/identity_echo_recovery'
    if (identity_folder/'proof.json').exists():
        proof=r.read(identity_folder/'proof.json')
        assert r.file_sha(run/'inference_frozen.json')==proof['inference_frozen_sha256']
        original_result=r.read(identity_folder/'results.json');current_result=r.read(run/'results.json')
        assert original_result['valid_answers']==current_result['valid_answers']
        assert set(original_result['machine_metrics'])==set(current_result['machine_metrics'])
        assert all(abs(v-current_result['machine_metrics'][key])<1e-12 for key,v in original_result['machine_metrics'].items())
        for relative,expected in proof['unchanged_artifact_sha256'].items():assert r.file_sha(run/relative)==expected,relative
        for name,expected in proof['failed_record_backups_sha256'].items():
            original=identity_folder/'failed_records'/name;assert r.file_sha(original)==expected
            before=r.read(original);after=r.read(run/'settings/collective/records'/name)
            for key in ('answer','answer_raw_record','prompt','question','reference_answer','actual_top5'):assert before.get(key)==after.get(key),(name,key)
        for item in proof['eligible']:
            current=r.read(run/'settings/collective/records'/(item['id']+'.json'))
            if current.get('evaluation'):
                view=current['evaluation']['views'][item['view']]
                assert view['raw_record']==item['original_raw_record'] and view['value']==item['original_metric_values']
        structural_recovery={'eligible_ids':[item['id'] for item in proof['eligible']],
            'recovered_evaluations':sum(bool(r.read(run/'settings/collective/records'/(item['id']+'.json')).get('evaluation')) for item in proof['eligible']),
            'original_scores_and_reasons_preserved':True,'empty_response_failures_not_retried':True}
        new_calls=[r.read(p) for p in (run/'api_calls').glob('*.json') if p.relative_to(run).as_posix() not in proof['unchanged_artifact_sha256']]
        assert all(v['key'].startswith('final:collective:') and v['key'].endswith(':judge:reference_answer') for v in new_calls)
        structural_recovery.update(new_raw_calls=len(new_calls),no_answer_regeneration=True,all_machine_metrics_unchanged=True)
    synonym_proof=run/'technical_repairs/direct_output_synonym/proof.json'
    if synonym_proof.exists():
        for relative,expected in r.read(synonym_proof)['unchanged_artifact_sha256'].items():
            assert r.file_sha(run/relative)==expected,relative
    guard_proof=run/'technical_repairs/qa_negation_guard/proof.json'
    if guard_proof.exists():
        saved=r.read(guard_proof)
        for name,h in saved['training_stage_cluster_hashes'].items():assert r.file_sha(run/name)==h,name
        for name,old in saved['pending_answers'].items():
            current=r.read(run/name)
            assert current['answer']==old['answer'] and current['answer_raw_record']==old['answer_raw_record']
    recovery=None
    if (run/'transport_recovery_latest.json').exists():
        pointer=r.read(run/'transport_recovery_latest.json')
        folder=run/'technical_repairs'/Path(pointer['manifest']).parent.name
        manifest=r.read(folder/'manifest.json')
        assert r.file_sha(run/'frozen_manifest.json')==manifest['frozen_manifest_sha256']
        assert r.file_sha(run/'inference_frozen.json')==manifest['inference_frozen_sha256']
        old_answers=old_grades=0
        for qid,h in manifest['record_backup_sha256'].items():
            backup=folder/'records'/(qid+'.json');assert r.file_sha(backup)==h
            previous=r.read(backup);current=r.read(run/'settings/collective/records'/(qid+'.json'))
            if previous.get('answer'):
                assert current['answer']==previous['answer'] and current['answer_raw_record']==previous['answer_raw_record'];old_answers+=1
            if previous.get('evaluation'):
                assert current==previous;old_grades+=1
            if qid in manifest['reset_timeout_ids']:
                assert previous.get('evaluation') is None and 'timed out' in previous.get('error','')
        assert old_answers==manifest['preserved_answers']==117 and old_grades==manifest['preserved_evaluations']==110
        recovery={'preserved_answers_checked':old_answers,'preserved_evaluations_checked':old_grades,
            'transport_failure_ids':manifest['reset_timeout_ids'],'frozen_sources_and_inference_unchanged':True}
    result={'passed':True,'training_n':23,'planned_test_n':378,'k':clustering['k'],
        'training_versions_checked':train_versions,'final_answers_checked':final_count,
        'raw_calls_checked':raw_count,'numeric_evaluations_checked':grade_count,'judge_sha256':judge['sha256'],
        'single_final_answer_call':True,'no_test_reference_in_answer_request':True,
        'same_judge_training_and_final':True,'feedback_audit':feedback,'transport_recovery':recovery,'structural_recovery':structural_recovery,'failures':failures,
        'physically_isolated_evaluation_views':bool(knowledge),'metric_keys':list(dims),
        'scope':'Structural/source audit, not certification of LLM judgment correctness'}
    r.save(run/'reports/独立核验.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':audit(sys.argv[1],training_only='--training-only' in sys.argv)
