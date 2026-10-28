"""Read-only post-run collation. Does not modify server answers/scores/prompts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

from five_metrics_page import build, DIMS, metric_dims


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def main(run):
    run=Path(run);result=read(run/'results.json');rows=read(run/'reports/全量逐题对照.json')['records']
    cfg=read(run/'code_snapshot/config.json');dims=metric_dims(run)
    audit=read(run/'reports/独立核验.json');feedback=audit['feedback_audit']
    batch_finished=read(run/'job_status.json').get('stage')=='complete'
    assert audit['passed'] and result['groups']==['collective']
    counts=Counter()
    for path in (run/'api_calls').glob('*.json'):
        value=read(path);key=value['key']
        category=('测试作答' if key.startswith('final:') and key.endswith(':answer') else
                  '测试评价' if key.startswith('final:') and ':judge' in key else
                  '案例作答' if key.startswith('train:') and ':answer:' in key else
                  '案例评价' if key.startswith('train:') and ':judge' in key else
                  '群智反馈（含阶段生成和无效尝试）' if key.startswith('feedback:') else
                  '提示生成（含校验失败尝试）')
        counts[category]+=1
    info={'raw_requests_including_failures':dict(counts),'total_raw_requests':sum(counts.values()),
          'server_results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
          'judge_sha256':result['judge_sha256'],'no_new_model_calls':True,
          'batch_finished':batch_finished,'all_planned_questions_valid':result['complete'],
          'feedback_unverifiable_replays':feedback['unverifiable_replays']}
    (run/'reports/交付核验.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(run/'reports/实验结果.md').read_text(encoding='utf-8')
    text+='\n\n## 补充核验\n\n独立核验已检查原始Q、实际top5、答案、各项评分和理由、训练选择、固定裁判和最终单调用；这是结构与来源核验，不保证LLM每条理由都正确。\n'
    missing=feedback['unverifiable_replays']
    text+=f'\n群智反馈有{len(missing)}条训练回放诊断在三次尝试后仍无法核验，相关簇按既定规则保留之前有效提示，不填造判断、不算修复成功。\n'
    text+='\n|原始请求阶段|次数（含失败尝试）|\n|---|---:|\n'
    text+='\n'.join(f'|{key}|{value}|' for key,value in counts.items())+'\n'
    text+='\n每个指标只采用第一次有效评价；提示生成、案例优化和反馈请求另计，不能将全部请求数理解为测试题数。\n'
    text+='\n机械指标按取得答案的题目计算；length是答案字符数，valid是该答案非空的布尔标识，其均值1不代表378题全部成功。实际答案与评价覆盖数见本报告开头。\n'
    if cfg.get('evaluation_views'):
        if cfg.get('protocol','').startswith('ten_point_'):
            text+='\n本轮案例仅请求事实/忠实度和完整性/精简性两个隔离视图；最终增加Q、上下文、参考A、实际A齐全的参考比较视图，给出相似分和独立的1/0/-1优劣判断。前四项判据训练与最终相同，后两项从不参与案例选择。最终答案仍为单次逻辑生成，重试另计。\n'
        else:
            text+='\n本轮每个答案分三个物理隔离请求评分：事实/忠实度、完整性/回答准确性、参考相似性。每个指标只取第一次有效评价，不是三次重复评分；最终答案仍为单次逻辑生成，空返回的有界重试另计。\n'
    if audit.get('structural_recovery'):
        info=audit['structural_recovery']
        detail=('仅移除唯一匹配的视图外壳，在明确的理由续段字段前补逗号并按原顺序连接全部理由文字，或删除完整理由后的尾逗号、补齐唯一缺失的根结束括号。' if info.get('kind')=='ten_point_named_structural' else '仅移除回显的视图身份字段，或补JSON转义/结束引号。')
        text+='\n## 评分结构恢复\n\n整批结束后，对'+str(info['recovered_evaluations'])+'题做了一次有界结构恢复：'+detail+'原始评分、理由字词、答案和已有有效视图均经核验保留；只补从未取得的评价视图，没有重评有效低分。不安全或无法机械恢复的记录保留为失败/N/A。\n'
    if audit.get('transport_recovery'):
        recovery=audit['transport_recovery']
        text+='\n## 服务中断与恢复\n\n8910曾持续超时，保留断点后停止本任务；用户恢复连接后，实际模型探测成功才续跑。仅恢复传输失败或未完成的精确请求，没有重训或更改裁判。独立核验确认恢复前的'+str(recovery['preserved_answers_checked'])+'份答案和'+str(recovery['preserved_evaluations_checked'])+'份有效评价完整保留。失败尝试也保留在上述请求数量中。\n'
    review='\n人工复核发现：“明十三陵顺便去居庸关长城方便吗，有什么看点”中，答案说“现有材料未提及……看点信息”，裁判却将其理解为断言看点不存在，两者并不等价。“败家石”案例的忠实度理由称“请人日夜看管”等细节无依据，但TOP1明确包含该叙述，该扣分点也需复核。这不是对历史真伪的认定，而是原文与评分理由的对照。原分数和理由均保留，未人工改分或重评。\n\n“孩子买学生票”一题的问题未指明景区，top5又混有多个景区，参考答案却默认南孔庙，因此还存在作答对象不明确的输入问题。答案整体拒答及重复说明值得改进，但不能把与参考答案不一致一律解释为模型事实错误。上述低分案例应区分生成不足、输入歧义与裁判误判，不作为已经证实的方法失败；本轮没有据此改提示、重选题或重评。\n'
    if cfg.get('allow_internal_knowledge'):
        review='\n本轮允许材料缺失时用模型内部知识补全。前四项指标不接收参考答案，完整性与回答准确性也不接收上下文；参考相似性仅作补充。事实正确性的内部知识判断不是外部权威核验，不能保证每条裁判理由正确。未知、无适用范围或服务失败均保留N/A，不当作0分或满分。未根据本批测试评分再调整提示或挑选题目。\n'
        if cfg.get('protocol','').startswith('ten_point_'):
            review=review.replace('完整性与回答准确性','完整性与回答精简性').replace('参考相似性仅作补充','参考相似性与答案优劣只在最终比较视图使用Q、上下文及两份答案，不能视为外部真值')
            review=review.replace('未知、无适用范围或服务失败均保留N/A，不当作0分或满分。','不适用的null或服务失败保留N/A；优劣比较的0是有效判断“无法判断”，不与缺失值混同。')
    text+='\n## 评分理由复核提醒\n'+review
    if audit['failures']:
        by_id={row['id']:row for row in rows}
        text+='\n## 未取得有效结果的题目\n\n以下为服务或格式故障，不解释为答案质量得0分。\n'
        for item in audit['failures']:
            text+='\n- '+by_id[item['id']]['question']+'：'+str(item.get('error') or item.get('status'))+'。\n'
    labels=[label for _,label in dims]
    text+='\n## 分簇描述性结果\n\n各列独立取有效数值均值；无法判断的null不按0分处理，优劣比较另见整体分布。\n\n|簇|题数|有效评价|'+'|'.join(labels)+'|F1|\n|---|'+'---:|'*(3+len(labels))+'\n'
    for cid in range(result['k']):
        members=[row for row in rows if row['question_type']=='cluster_'+str(cid)]
        values=[row['groups']['collective'] for row in members]
        means=[]
        for key,_ in dims:
            points=[v['llm_scores'][key] for v in values if v['llm_scores'][key] is not None]
            means.append(f'{statistics.mean(points):.2f}' if points else 'N/A')
        f1=[v['machine_metrics']['f1'] for v in values if v.get('machine_metrics')]
        text+='|'+str(cid)+'|'+str(len(members))+'|'+str(sum(v['status']=='complete' for v in values))+'|'+'|'.join(means)+'|'+(f'{statistics.mean(f1):.6f}' if f1 else 'N/A')+'|\n'
    if batch_finished and not result['complete']:
        text='# 本轮结果（整批已结束，含失败题）\n\n'+f"378题均已按有界协议处理；有效答案{result['valid_answers']}份、有效评价{result['valid_evaluations']}份，失败项保留N/A，均值只覆盖有效结果，不填0、不删除失败题。\n\n"+text
    elif not batch_finished:
        outage=run/'outage_stop.json'
        note='8910服务连续三次300秒请求超时，模型列表接口也超时；仅停止本任务的排队请求，所有有效答案、评分及原始记录保留。' if outage.exists() else '本轮尚未完成，以下均值只覆盖已取得有效结果的题目。'
        text='# 阶段结果（尚未完成全量测试）\n\n'+note+'\n\n机器指标覆盖'+str(result['valid_answers'])+'份已有答案，LLM均值覆盖'+str(result['valid_evaluations'])+'份有效评价，不能当成378题全量结果。\n\n'+text
    filename='完整结果.md' if result['complete'] else '本轮结果.md' if batch_finished else '阶段结果.md'
    (run/'reports'/filename).write_text(text,encoding='utf-8')
    def quality(row):
        values=[row['groups']['collective']['llm_scores'][key] for key,_ in dims[:3]]
        return sum(values)/3 if all(isinstance(v,(int,float)) for v in values) else float('inf')
    examples=['# 代表性问题案例','','按前三项质量评分由低到高列出3题，仅用于审阅，不据此筛选实验题或更新提示。原始评分理由可能存在误判，以下不作人工改分。']
    examples+=[review]
    scored=[row for row in rows if row['groups']['collective']['status']=='complete']
    for row in sorted(scored,key=lambda value:(quality(value),value['id']))[:3]:
        value=row['groups']['collective']
        examples+=['','## '+row['question'],'','### 参考答案','',row['reference_answer'],
                   '','### q＋群智＋top5','',value['answer'] or '未取得有效答案','','### 最终评分依据','']
        for key,label in dims:
            score=value['llm_scores'][key]
            examples += [f'- {label}：{score if score is not None else "N/A"}；{value["score_basis"][key]}']
        for index,doc in enumerate(row['actual_top5']):
            examples+=['',f'### TOP{index+1} '+doc.get('doc_title',''),'',doc['text']]
    (run/'reports/代表性问题案例.md').write_text('\n'.join(examples),encoding='utf-8')
    build(run)
    print(json.dumps({'planned':result['planned_n'],'answers':result['valid_answers'],
        'evaluations':result['valid_evaluations'],'llm_scores':result['llm_scores'],
        'feedback_unverifiable':len(missing),'requests':dict(counts)},ensure_ascii=False),flush=True)


if __name__=='__main__':main(sys.argv[1])
