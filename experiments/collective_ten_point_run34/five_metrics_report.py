"""One-method report; all planned questions retained, missing values never zeroed."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from collections import Counter

from rejudge_five_runner import DIMS

LABELS = ('事实准确性 ↑', '上下文忠实度 ↑', '问题响应完整性 ↑', '回答冗余度 ↓', '参考语义相似性 ↑（辅助）')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def report(run, code):
    run, code = Path(run), Path(code)
    spec = importlib.util.spec_from_file_location('chapter3_machine', code / 'machine_protocol.py')
    protocol = importlib.util.module_from_spec(spec); spec.loader.exec_module(protocol); protocol.ensure_metric_dependencies()
    tests = read(run / 'shared/test_top5.json'); cfg = read(code / 'config.json')
    dims=tuple(cfg.get('metric_keys',DIMS));labels=tuple(cfg.get('metric_labels',LABELS))
    clustering = read(run / 'clustering.json'); feedback = read(run / 'collective_feedback/final_manifest.json')
    rows = []; columns = []
    for row in tests:
        p = run / 'settings/collective/records' / (row['id'] + '.json')
        value = read(p) if p.exists() else {}
        assert not value or (value['question'] == row['question'] and value['reference_answer'] == row['reference_answer'])
        assert not value or [d['text'] for d in value['actual_top5']] == [d['text'] for d in row['context_documents']]
        grade = value.get('evaluation'); answer = value.get('answer')
        machine = {k:float(v) for k,v in protocol.score_answer(answer, row['reference_answer']).items()} if answer else None
        if machine: assert len(machine) == 13; columns = list(machine)
        if grade:
            assert set(grade['dimensions']) == set(dims) and grade['repetitions'] == 1
            assert grade['judge_sha256'] == read(run / 'stages/evaluation.json')['sha256']
        rows.append({'id':row['id'], 'question':row['question'], 'reference_answer':row['reference_answer'],
            'actual_top5':value.get('actual_top5', row['context_documents']),
            'question_type':'cluster_' + str(clustering['test_routes'][row['id']]['cluster_id']),
            'question_type_name':value.get('question_type', ''),
            'groups':{'collective':{'answer':answer, 'machine_metrics':machine,
                'llm_scores':grade['dimensions'] if grade else {d:None for d in dims},
                'score_basis':grade['reasons'] if grade else {d:value.get('error','未取得有效评价') for d in dims},
                'status':'complete' if grade else 'evaluation_unavailable' if answer else 'answer_unavailable',
                'answer_raw_record':value.get('answer_raw_record'), 'judge_raw_record':grade['raw_record'] if grade else None,
                'judge_raw_records':grade.get('raw_records',[grade['raw_record']]) if grade else []}}})
    valid = [row['groups']['collective'] for row in rows if row['groups']['collective']['status'] == 'complete']
    answered = [row['groups']['collective'] for row in rows if row['groups']['collective']['answer']]
    summary = {'planned_n':len(tests), 'training_n':cfg['training_n'], 'k':clustering['k'],
        'valid_answers':len(answered), 'valid_evaluations':len(valid), 'groups':['collective'],
        'model':cfg['model'], 'judge_sha256':read(run / 'stages/evaluation.json')['sha256'],
        'judge_repetitions':1, 'exploratory':True, 'llm_scores':{}, 'machine_metrics':{},
        'feedback_states':feedback['states'], 'complete':len(valid) == len(tests)}
    calls=[read(p) for p in (run/'api_calls').glob('*.json')]
    category=lambda v:'final_answer' if v['key'].startswith('final:') and v['key'].endswith(':answer') else 'final_judge' if v['key'].startswith('final:') and ':judge' in v['key'] else 'training_answer' if v['key'].startswith('train:') and ':answer:' in v['key'] else 'training_judge' if v['key'].startswith('train:') and ':judge' in v['key'] else 'feedback' if v['key'].startswith('feedback:') else 'prompt_generation'
    summary['raw_request_counts_including_failed_attempts']=dict(Counter(category(v) for v in calls))
    summary['all_raw_requests']=len(calls)
    for d in dims:
        values = [v['llm_scores'][d] for v in valid if v['llm_scores'][d] is not None]
        summary['llm_scores'][d] = {'mean':statistics.mean(values) if values else None, 'n':len(values)}
    if 'answer_preference' in dims:
        counts=Counter(v['llm_scores']['answer_preference'] for v in valid)
        summary['answer_preference_counts']={'candidate_better':counts[1],'undetermined':counts[0],'reference_better':counts[-1]}
        summary['score_scale']=cfg['score_scale']
    for key in columns:
        summary['machine_metrics'][key] = statistics.mean(v['machine_metrics'][key] for v in answered)
    save(run / 'results.json', summary)
    save(run / 'reports/全量逐题对照.json', {'n':len(rows), 'records':rows, 'summary':summary})
    md = ['# 旅游：新版评价与群智反馈完整流程', '',
        f'23题案例优化，{len(tests)}题测试；只运行q＋群智＋top5，{cfg["model"]}，并发8。', '',
        f'算法选择{clustering["k"]}簇。有效答案{len(answered)}，有效最终评价{len(valid)}；缺失保留为N/A，不填0。', '',
        '|指标|均分|有效题数|', '|---|---:|---:|']
    for d,label in zip(dims,labels):
        item=summary['llm_scores'][d]; score='N/A' if item['mean'] is None else f"{item['mean']:.3f}"
        md.append(f"|{label}|{score}|{item['n']}|")
    if 'answer_preference_counts' in summary:
        c=summary['answer_preference_counts']
        md += ['',f"答案优劣：实际答案更优{c['candidate_better']}题，无法判断{c['undetermined']}题，参考答案更优{c['reference_better']}题；不是与裸q+top5比较。"]
    md += ['', cfg.get('scoring_notes','参考相似性为辅助，不计算五项混合总分；冗余度越低越好。没有运行新基线，不与旧裁判结果直接作优劣比较。'), '',
        '## 机器指标', '', '|指标|均值|', '|---|---:|']
    md += [f'|{k}|{v:.6f}|' for k,v in summary['machine_metrics'].items()]
    md += ['', '## 群智反馈', '', '|簇|接受更新轮次|停止原因|', '|---|---:|---|']
    for cid,s in feedback['states'].items(): md.append(f'|{cid}|{s["selected_round"]}|{s["stop_reason"]}|')
    md += ['', '仅原训练题进入反馈；重点检查案例级能力在融合后是否丢失，不以固定指标分数驱动群智反馈。各簇独立修正，未修复或出现退步则保留先前提示。', '',
        '本批测试题此前已经检视，结果为探索性；同模型评价仍可能存在分数与理由不一致。'+('本轮三种视图物理隔离，每个指标一次有效评价；模型自身知识判断不等于外部事实核验。' if cfg.get('evaluation_views') else '裁判输入范围由提示约束，不是物理隔离的盲评。')]
    md+=['','## 原始请求数量（含失败尝试）','','|阶段|次数|','|---|---:|']
    md += [f'|{key}|{value}|' for key,value in summary['raw_request_counts_including_failed_attempts'].items()]
    (run / 'reports/实验结果.md').write_text('\n'.join(md),encoding='utf-8')
    prompts = ['# 本轮meta及实际阶段提示', '', (code / 'user_meta.md').read_text(encoding='utf-8')]
    for kind in ('qa','evaluation','update','collective'):
        prompts += ['', '## 实际阶段：'+kind, '', read(run / 'stages' / (kind+'.json'))['text']]
    for kind in ('diagnosis','update'):
        prompts += ['', '## 群智反馈阶段：'+kind, '', read(run / 'collective_feedback/stages' / (kind+'.json'))['text']]
    for cid in range(clustering['k']):
        prompts += ['', '## 最终簇'+str(cid)+' QA提示', '', read(run / 'collective_feedback/final_clusters' / ('cluster_'+str(cid)+'.json'))['text']]
    (run / 'reports/各阶段实际提示.md').write_text('\n'.join(prompts),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__': report(*sys.argv[1:3])
