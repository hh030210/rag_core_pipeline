"""Run34: four integer training metrics; six final metrics, same shared judge."""
import json
import re
import statistics

import knowledge_fallback_workflow as k
import five_metrics_workflow as b
import restored_meta_runner as r

DIMS=('factual_accuracy','context_faithfulness','question_completeness','answer_conciseness','reference_similarity','answer_preference')
LABELS=('事实正确性','上下文忠实度','问题响应完整性','回答精简性','参考语义相似性','答案优劣比较')
PRIMARY=DIMS[:4]
VIEWS={'facts_context':DIMS[:2],'question_answer':DIMS[2:4],'reference_comparison':DIMS[4:]}
EVIDENCE=k.EVIDENCE
GOALS='''事实正确性：已说出的事实应正确；先正确使用相关上下文，缺失部分允许用可靠内部知识补全，不确定时明确说明，不以参考答案定真伪。
上下文忠实度：材料对应部分保持主体、关系、条件、时间和确定性，不跨对象混用；知识补充不冒称材料原文。
问题响应完整性：识别并回应问题各问项与条件，不因参考或材料增加必答项；具体说明无法确定的问项，不笼统拒答。
回答精简性：保留作答和理解所需内容，去掉无关扩展、重复、无效铺垫和套话；必要条件、歧义澄清及有用提醒可以保留，不设固定字数。'''


def metas(rubric):
    common='\n[作答与评价边界]\n'+GOALS+'\n[证据使用]\n'+EVIDENCE
    structure=k.STRUCTURE
    return {
      'qa':'[角色]\n你是知识库问答的提示设计专家。\n[任务]\n生成一份通用初始QA提示。成品接收当前问题和未标注top5，直接生成答案，不再生成提示；此处不输入具体案例。'+common+'\n[输出要求]\n'+structure+'\n只输出QA提示正文。规则可迁移，不硬编码案例事实或答案。',
      'evaluation':'[角色]\n你是答案评价指令设计专家。\n[任务]\n生成一份直接评价答案的通用指令，不输入具体案例，不再生成评价提示。完整原样保留以下规范；案例与最终评价的前四项口径相同，后两项只在最终评价运行。\n[必须保留的评价规范]\n'+rubric.strip()+'\n[生成指令的输出要求]\n只输出成品评价指令，原样保留规范及视图范围，不增加锚点、小数或混合总分。',
      'update':'[角色]\n你是QA提示更新指令设计专家。\n[任务]\n生成一份直接改写当前QA提示的通用更新指令，不输入具体案例，不再生成更新指令。\n[成品输入]\n问题、参考答案、实际答案、实际top5、当前QA提示、前四项评分与理由、评价标准。\n[更新要求]\n逐项参考事实正确性、上下文忠实度、问题响应完整性、回答精简性的分数与理由，对照实际材料核实，再落实为可迁移的作答方法。前四项均1–10整数，越高越好；参考语义相似性和答案优劣比较不参与案例级评价或更新，不索取其分数。事实不以参考答案定真伪，不从参考新增必答项；不迎合与材料矛盾的错误归因，保留有效规则，不硬编码案例答案。'+common+'\n[输出结构]\n执行更新时只输出改写后的QA正文，包含[角色][任务][输入][证据使用][作答要求][输出]。最终作答只收到问题和上下文，不读取参考、评分或历史；把必要操作方法直接写进QA。',
      'collective':'[角色]\n你是群智提示融合指令设计专家。\n[任务]\n生成一份通用群智融合指令；成品从同簇Q、参考A和最终案例QA归纳共同问题类型、提问意图、答案形态与关键条件，生成该类型的群智QA，不再生成融合指令。\n[成品输入]\ncluster_id、cases（question、reference_answer、final_qa_prompt）、evaluation_criteria；不提供测试题或迭代记录。\n[融合要求]\n融合有效互补方法，去重、协调冲突，针对共同问题类型形成作答要求，不把个例约束泛化全簇，不复制答案。遵循前四项同一标准；后两项仅最终比较，不为贴近参考添加问题未问内容。'+common+'\n[输出结构]\n执行融合时只输出该簇QA正文，包含[角色][任务][输入][证据使用][作答要求][输出]。各问题类型分簇保留，不合并全局提示；最终仅问题、实际top5及本簇提示，一次调用输出答案，不读取参考、分数、其他答案或诊断。'}


def validate_stage(kind,text,rubric):
    text=r.normalize(text)
    if kind=='qa':return k.validate_qa(text)
    if kind=='evaluation':
        if rubric not in text:raise ValueError('须原样保留同一整数六指标规范及案例/最终范围')
        return text
    if any(not r.section(text,h) for h in ('角色','任务','输入')):raise ValueError('缺少角色/任务/输入')
    task=r.section(text,'任务')
    task=re.sub(r'(?:不再|不得|禁止|不能|不要)[^。\n；]{0,20}(?:更新指令|融合指令)','',task)
    if re.search(r'(?:生成|设计)[^。\n]{0,12}(?:更新指令|融合指令)',task):raise ValueError('成品直接更新或融合QA，不再生成指令')
    if kind=='update':
        if not re.search(r'更新|改写',task) or not all(label in text for label in LABELS[:4]) or not re.search(r'评分|分数',text) or '理由' not in text:raise ValueError('须按前四项分数与理由更新QA')
        if not re.search(r'不参与|不使用|不纳入|仅.*最终|只.*最终',text):raise ValueError('后两项不能参与案例更新')
    elif not re.search(r'问题类型|共同[^。\n]{0,12}(?:提问|意图)',text) or '簇' not in text:raise ValueError('须按簇归纳问题类型')
    if any(h not in text for h in r.HEADINGS):raise ValueError('输出须明确六段QA结构')
    if not re.search(r'内部知识|模型知识|自身知识',text) or not re.search(r'补全|补充',text):raise ValueError('保留内部知识补全边界')
    return text


def view_payloads(row,answer,final=True):
    result={'facts_context':{'evaluation_view':'facts_context','question':row['question'],'contexts':r.context_view(row),'candidate_answer':answer},
            'question_answer':{'evaluation_view':'question_answer','question':row['question'],'candidate_answer':answer}}
    if final:result['reference_comparison']={'evaluation_view':'reference_comparison','question':row['question'],'contexts':r.context_view(row),'reference_answer':row['reference_answer'],'candidate_answer':answer}
    return result


def parse_view_with_proof(text,view):
    from ten_point_format import canonicalize
    fields=VIEWS[view];canonical,proof=canonicalize(r.normalize(text),view,fields);value=json.loads(canonical)
    if isinstance(value,dict) and set(value)==set(fields)|{'evaluation_view'} and value['evaluation_view']==view:
        value={key:value[key] for key in fields}
        proof={'kind':'remove_exact_echoed_view_identity_only','removed_metadata':{'evaluation_view':view},'original_output_sha256':r.digest(text),'metric_values_sha256':r.digest(value),'prior_format_proof':proof,'no_score_or_reason_edits':True}
    if not isinstance(value,dict) or set(value)!=set(fields):raise ValueError('Wrong metrics for view '+view)
    for key,item in value.items():
        if not isinstance(item,dict) or set(item)!={'score','reasoning'}:raise ValueError('Each metric keeps score/reasoning only')
        score=item['score']
        if key=='answer_preference':
            if type(score) is not int or score not in (-1,0,1):raise ValueError('Preference must be integer -1/0/1')
        elif score is None:
            if key not in (DIMS[0],DIMS[1],DIMS[4]):raise ValueError('Question-only metrics cannot be null')
        elif type(score) is not int or not 1<=score<=10:raise ValueError('Score must be integer1-10')
        if not isinstance(item['reasoning'],str) or not item['reasoning'].strip():raise ValueError('Missing concrete reason')
    return value,proof


def grade(api,stages,key,row,answer):
    final=key.startswith('final:');payloads=view_payloads(row,answer,final=final);views={};merged={};judge=stages['evaluation']
    for view,payload in payloads.items():
        path=r.RUN/'evaluation_views'/(r.digest(key+':'+view)+'.json')
        if path.exists():
            item=r.read(path)
            assert item['input']==payload and item['judge_sha256']==judge['sha256']
        else:
            reply=api.ask(key+':judge:'+view,[{'role':'system','content':judge['text']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=5000 if view=='reference_comparison' else 3500)
            value,proof=parse_view_with_proof(reply['text'],view)
            item={'input':payload,'value':value,'raw_record':reply['raw_record'],'judge_sha256':judge['sha256'],'format_integrity':proof}
            r.save(path,item)
        views[view]=item;merged.update(item['value'])
    dims=DIMS if final else PRIMARY
    assert set(merged)==set(dims)
    return {'dimensions':{d:merged[d]['score'] for d in dims},'reasons':{d:merged[d]['reasoning'] for d in dims},'raw_output':merged,
            'raw_record':views['facts_context']['raw_record'],'raw_records':[v['raw_record'] for v in views.values()],
            'evaluation_input':payloads,'views':views,'judge_sha256':judge['sha256'],'repetitions':1,'format_integrity':None,
            'physical_view_isolation':True,'metric_phase':'final' if final else 'case'}


def effective_values(dimensions):
    return [dimensions[d] for d in PRIMARY if dimensions[d] is not None] or None


def stop_decision(previous,current,min_gain=.1):
    if {d for d in PRIMARY if previous[d] is None}!={d for d in PRIMARY if current[d] is None}:return 'primary_assessability_changed'
    old,new=effective_values(previous),effective_values(current)
    if old is None or new is None:return 'primary_metric_unassessable'
    if any(y<x for x,y in zip(old,new)):return 'primary_dimension_declined'
    if statistics.mean(new)-statistics.mean(old)<min_gain:return 'primary_gain_below_threshold'
    return None


def bind():
    # Reuse run33 answering/feedback mechanics; all scores and prompts are new.
    k.DIMS=DIMS;k.LABELS=LABELS;k.GOALS=GOALS
    runtime=k.bind()
    b.DIMS=DIMS;b.LABELS=LABELS;b.PRIMARY=PRIMARY;b.GOALS=GOALS
    b.grade=grade;b.validate_stage=validate_stage;b.effective_values=effective_values;b.stop_decision=stop_decision
    b.primary_perfect=lambda dims:bool(effective_values(dims)) and all(v==10 for v in effective_values(dims))
    return b.bind()


if __name__=='__main__':
    try:bind();b.main()
    except Exception as exc:r.mark('failed',error=type(exc).__name__+': '+str(exc));raise
