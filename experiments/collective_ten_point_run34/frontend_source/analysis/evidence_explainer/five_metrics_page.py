"""Self-contained result viewer with an accessible draggable vertical split."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIMS = [('factual_accuracy','事实准确性 ↑'),('context_faithfulness','上下文忠实度 ↑'),
        ('question_completeness','问题响应完整性 ↑'),('answer_redundancy','回答冗余度 ↓'),
        ('reference_similarity','参考语义相似性 ↑（辅助）')]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def metric_dims(run):
    path=Path(run)/'code_snapshot/config.json';cfg=read(path) if path.exists() else {}
    dims=list(zip(cfg['metric_keys'],cfg['metric_labels'])) if cfg.get('metric_keys') else DIMS
    return [(key,'答案优劣比较' if key=='answer_preference' else label) for key,label in dims]


def build(run, sample=False):
    run = Path(run)
    dims=DIMS if sample else metric_dims(run)
    if sample:
        old = read(run / '展示数据.json')
        rows = []
        for q in old['questions']:
            rows.append({'id':q['id'],'question':q['question'],'reference_answer':q['reference'],
                'actual_top5':[old['documents'][key] for key in q['contexts']],
                'question_type_name':old['types'][q['type']], 'groups':{
                    g:{'answer':v['answer'],'llm_scores':v['scores'],'score_basis':v['basis']} for g,v in q['groups'].items()}})
        prompts = [{'id':'final','label':'最终评价','text':old['prompts']['final']}]
        for key,label in [('qa_instruction','初始QA'),('evaluation','原案例评价'),('update','提示更新'),('fusion','群智融合')]:
            prompts.append({'id':key,'label':label,'text':old['prompts'][key]})
        prompts.extend({'id':key,'label':key,'text':text} for key,text in old['prompts']['collective'].items())
        title = '旅游20题 · 原答案新版评分'; filename = '旅游20题_新版评价_可调布局.htm'
    else:
        report = run / 'reports/全量逐题对照.json'
        # A previous partial report must not hide newly resumed records.
        summary_path=run/'results.json'
        status_path=run/'job_status.json'
        batch_finished=status_path.exists() and read(status_path).get('stage')=='complete'
        report_complete=batch_finished or (summary_path.exists() and read(summary_path).get('complete',False))
        rows = read(report)['records'] if report.exists() and report_complete else []
        if not rows and (run/'shared/test_top5.json').exists():
            for row in read(run/'shared/test_top5.json'):
                path=run/'settings/collective/records'/(row['id']+'.json')
                value=read(path) if path.exists() else {};grade=value.get('evaluation')
                rows.append({'id':row['id'],'question':row['question'],'reference_answer':row['reference_answer'],
                    'actual_top5':row['context_documents'],'question_type_name':value.get('question_type',''),
                    'groups':{'collective':{'answer':value.get('answer'),
                        'llm_scores':grade['dimensions'] if grade else {d:None for d,_ in dims},
                        'score_basis':grade['reasons'] if grade else {d:'尚未取得有效最终评价' for d,_ in dims}}}})
        prompts = []
        for key,label in [('qa','初始QA'),('evaluation','案例与最终评价'),('update','案例更新'),('collective','群智融合')]:
            p = run / 'stages' / (key+'.json')
            if p.exists():prompts.append({'id':key,'label':label,'text':read(p)['text']})
        if not (run/'stages/evaluation.json').exists() and (run/'code_snapshot/evaluation_rubric.txt').exists():
            prompts.append({'id':'pending_evaluation_standard','label':'评价标准（执行指令待生成）','text':(run/'code_snapshot/evaluation_rubric.txt').read_text(encoding='utf-8')})
        config=run/'code_snapshot/config.json'
        if config.exists():
            for key,text in read(config).get('metas',{}).items():
                prompts.append({'id':'meta_'+key,'label':'meta：'+key,'text':text})
        for key,label in [('diagnosis','群智反馈诊断'),('update','群智反馈更新')]:
            p=run/'collective_feedback/stages'/(key+'.json')
            if p.exists():prompts.append({'id':'feedback_'+key,'label':label,'text':read(p)['text']})
        for p in sorted((run/'collective_feedback/final_clusters').glob('*.json')):
            prompts.append({'id':p.stem,'label':'最终簇'+str(read(p)['cluster_id']),'text':read(p)['text']})
        for p in sorted((run/'training/cases').glob('*.json')):
            case=read(p)
            if case.get('status')=='complete':prompts.append({'id':'case_'+case['id'],'label':'案例：'+case['question'],'text':case['prompt']})
        title=read(config).get('page_title','旅游 · 群智＋top5 · 新版完整流程') if config.exists() else '旅游 · 群智＋top5';filename='旅游群智_完整流程.htm'
    config_path=run/'code_snapshot/config.json'
    cfg=read(config_path) if config_path.exists() else {}
    integer_scale=cfg.get('protocol','').startswith('ten_point_')
    data={'title':title,'rows':rows,'prompts':prompts,'dims':dims,
          'scoreHeading':'最终LLM评分 · 1–10整数 · DeepSeek-V4-Pro' if integer_scale else '最终LLM评分 · 0–100 · DeepSeek-V4-Pro',
          'integerScale':integer_scale}
    template=(HERE/'five_metrics_page_template.htm').read_text(encoding='utf-8')
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    css=(ROOT/'sites/scenic-run25-review/src/styles.css').read_text(encoding='utf-8')
    css+='\n'+(ROOT/'sites/scenic-run25-review/src/basis.css').read_text(encoding='utf-8')
    html=template.replace('/*ORIGINAL_STYLES*/',css).replace('/*RESULT_DATA*/',encoded)
    (run/filename).write_text(html,encoding='utf-8')
    print(str(run/filename))


if __name__=='__main__':build(sys.argv[1],sample='--sample20' in sys.argv)
