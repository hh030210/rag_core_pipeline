"""Context-first QA with knowledge fallback and physically scoped evaluation."""
import json
import math
from pathlib import Path
import re
import statistics

import five_metrics_workflow as b
import restored_meta_runner as r

DIMS=('factual_accuracy','context_faithfulness','question_completeness','answer_accuracy','reference_similarity')
LABELS=('事实正确性','上下文忠实度','问题响应完整性','回答准确性','参考语义相似性')
VIEWS={'facts_context':DIMS[:2], 'question_answer':DIMS[2:4], 'reference_answer':DIMS[4:]}
GOALS='''事实正确性：检查已说出的事实对不对；上下文有对应内容时正确使用，缺失部分允许用可靠的模型内部知识补全，不把不确定猜测写成确定事实。
上下文忠实度：与材料已有信息对应的内容应保持主体、关系、条件和确定性；知识补充不冒称材料原文，不跨对象混用。
问题响应完整性：先识别问题的各个问项并逐一回应，不从参考答案或材料增加必答要求；不能确定时说明具体不确定点，不笼统拒答。
回答准确性：紧扣问题的实际需求，完整回答后尽可能简洁，去掉无关扩展、重复及无效铺垫；必要条件和澄清可以保留，不设固定字数。
参考语义相似性仅为补充指标，不要求作答时读取参考答案，也不为贴近参考而添加问题未问的内容。'''
EVIDENCE='''给定上下文可能包含噪音、无关信息或相互矛盾的信息。无论给定多少条上下文，都必须先根据当前问题对全部上下文进行相关性和证据价值重排序（rerank），仅保留并使用有用、可靠的上下文；不设固定保留条数，允许全部保留或全部舍弃。
先按相关上下文回答；上下文未包含相关信息的部分，允许使用模型内部知识补全。不要把知识补充冒称材料原文；缺乏可靠知识时明确说明具体不确定点，不编造，不因局部缺失而整体拒答。材料冲突时说明冲突，不借补全混淆来源。'''
STRUCTURE=b.STRUCTURE
_original_qa=r.validate_qa


def metas(rubric):
    common='\n[作答与评价边界]\n'+GOALS+'\n[证据使用]\n'+EVIDENCE
    return {
      'qa':'[角色]\n你是知识库问答的提示设计专家。\n[任务]\n生成一份通用初始QA提示。成品接收当前问题和未标注top5，直接生成答案，不再生成提示。此处不输入具体案例。'+common+'\n[输出要求]\n'+STRUCTURE+'\n只输出QA提示正文；规则可迁移，不硬编码具体实体、数字、日期或答案。',
      'evaluation':'[角色]\n你是答案评价指令设计专家。\n[任务]\n生成可直接评价答案的通用指令，不输入具体案例，不再生成评价提示。案例迭代与最终评价使用以下同一规范，原样保留。\n[必须保留的评价规范]\n'+rubric.strip()+'\n[生成指令的输出要求]\n只输出成品评价指令，完整保留规范及三个视图的输入边界，不添加指标或总分。',
      'update':'[角色]\n你是QA提示更新指令设计专家。\n[任务]\n生成一份直接改写当前QA提示的通用更新指令，不输入具体案例，不再生成更新指令。\n[成品输入]\n问题、参考答案、实际答案、实际top5、当前QA提示，以及五项评分、理由和完整评价标准。\n[更新要求]\n逐项参考事实正确性、上下文忠实度、问题响应完整性、回答准确性、参考语义相似性的分数与理由，对照材料核实，落实为可迁移的方法。事实正确性不以参考答案定真伪；忠实度只检查材料相关部分，不惩罚知识补全；完整性和回答准确性仅按Q和实际A判断。参考相似性仅辅助，不新增必答项。不得迎合与材料矛盾的错误归因；保留有效规则，不硬编码案例答案。'+common+'\n[输出结构]\n执行更新时仅输出改写后的QA正文，包含[角色][任务][输入][证据使用][作答要求][输出]。最终作答不提供参考、分数、理由或历史，不要求读取这些内容。',
      'collective':'[角色]\n你是群智提示融合指令设计专家。\n[任务]\n生成一份通用群智融合指令。成品直接根据同一簇训练案例的问题、参考答案及最终QA提示，归纳共同问题类型、提问意图、答案形态与关键条件，再生成该类型的群智QA提示，不再生成融合指令。\n[成品输入]\n簇编号、簇内Q、参考A、最终QA提示及评价标准；不提供测试题或迭代记录。\n[融合要求]\n融合有效、互补的方法，协调冲突、去重，针对共同问题类型形成作答要求；不把个例约束泛化全簇，不复制具体答案。参考答案仅辅助理解案例语义，不能代替事实核实或给新问题增加必答项。'+common+'\n[输出结构]\n只输出该簇QA正文，包含[角色][任务][输入][证据使用][作答要求][输出]。不同问题类型分簇保留，不合并全局提示；最终只用问题、top5及本簇提示，一次调用输出答案，不读取参考、评分、其他答案或诊断。'
    }


def validate_qa(text):
    try:
        text=_original_qa(text)
    except ValueError as exc:
        # Accept the same direct-answer contract expressed without 只/仅.
        # A validation-only spelling substitute runs every remaining old guard;
        # the generated prompt itself is returned byte-for-byte normalized.
        normalized=r.normalize(text)
        output=r.section(normalized,'输出')
        phrase='直接输出最终答案文本，不输出思考过程'
        if str(exc)!='QA输出必须仅为最终答案' or not output.startswith(phrase):raise
        checked=normalized.replace('[输出]\n'+phrase,'[输出]\n只输出最终答案文本，不输出思考过程',1)
        if checked==normalized:raise
        _original_qa(checked)
        text=normalized
    evidence=r.section(text,'证据使用')
    if not re.search(r'(?:内部|自身|模型|已有)[^。\n]{0,12}知识',evidence):
        raise ValueError('证据使用须明确允许模型内部知识补全材料缺失部分')
    if not re.search(r'补全|补充',evidence) or not re.search(r'优先|先[^。\n]{0,12}上下文|首先',evidence):
        raise ValueError('须先按上下文回答，再对缺失部分知识补全')
    blanket_ban=r'(?:(?:不得|禁止|不允许|不能)(?:使用|利用|依赖)(?:任何|自己的)?|不使用)(?:模型内部知识|内部知识|自身知识)(?:进行)?(?:补全|补充|填补|回答|作答)?(?=[。；\n]|$)'
    if re.search(blanket_ban,text):
        raise ValueError('不得禁止本轮明确允许的内部知识补全')
    if not re.search(r'不确定|不能确定|无法确定|可靠知识',text):raise ValueError('补全不等于编造，须处理知识不确定性')
    if not re.search(r'简洁|重复|冗余|无关扩展',text):raise ValueError('须在完整回应后保持简洁')
    return text


def view_payloads(row,answer):
    return {
      'facts_context':{'evaluation_view':'facts_context','question':row['question'],'contexts':r.context_view(row),'candidate_answer':answer},
      'question_answer':{'evaluation_view':'question_answer','question':row['question'],'candidate_answer':answer},
      'reference_answer':{'evaluation_view':'reference_answer','reference_answer':row['reference_answer'],'candidate_answer':answer}}


def validate_stage(kind,text,rubric):
    text=r.normalize(text)
    if kind=='qa':return validate_qa(text)
    if kind=='evaluation':
        if rubric not in text:raise ValueError('案例与最终评价须完整保留同一隔离视图规范')
        return text
    if any(not r.section(text,h) for h in ('角色','任务','输入')):raise ValueError('阶段指令须有角色/任务/输入')
    task=r.section(text,'任务')
    task=re.sub(r'(?:不再|不得|禁止|不能|不要)[^。\n；]{0,20}(?:更新指令|融合指令)','',task)
    if re.search(r'(?:生成|设计)[^。\n]{0,12}(?:更新指令|融合指令)',task):raise ValueError('成品须直接更新或融合QA，不能继续生成指令')
    if kind=='update':
        if not re.search(r'更新|改写',task) or not all(label in text for label in LABELS) or not re.search(r'评分|分数',text) or '理由' not in text:raise ValueError('须按五项分数与理由直接更新QA')
    elif not re.search(r'问题类型|共同[^。\n]{0,12}(?:提问|意图)',text) or '簇' not in text:raise ValueError('须按簇共同问题类型融合QA')
    if any(h not in text for h in r.HEADINGS):raise ValueError('须声明成品QA的六段结构')
    if not re.search(r'内部知识|模型知识|自身知识',text) or not re.search(r'补全|补充',text):raise ValueError('须保留内部知识补全边界')
    return text


def parse_view(text,view):
    return parse_view_with_proof(text,view)[0]


def parse_view_with_proof(text,view):
    from knowledge_view_format import canonicalize
    canonical,proof=canonicalize(r.normalize(text),VIEWS[view]);value=json.loads(canonical)
    if isinstance(value,dict) and set(value)==set(VIEWS[view])|{'evaluation_view'} and value['evaluation_view']==view:
        # Identity echo only: retain every original score/reason, never rejudge.
        value={key:value[key] for key in VIEWS[view]}
        proof={'kind':'remove_exact_echoed_view_identity_only','removed_metadata':{'evaluation_view':view},
            'original_output_sha256':r.digest(text),'metric_values_sha256':r.digest(value),
            'prior_format_proof':proof,'no_score_or_reason_edits':True}
    if not isinstance(value,dict) or set(value)!=set(VIEWS[view]):raise ValueError('Wrong metrics for isolated view: '+view)
    for key,item in value.items():
        if not isinstance(item,dict) or set(item)!={'score','reasoning'}:raise ValueError('Each metric needs only score/reasoning')
        score=item['score']
        if score is None:
            if key not in (DIMS[0],DIMS[1],DIMS[4]):raise ValueError('Question-only metrics cannot be null')
        elif isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100:raise ValueError('Score outside0-100')
        if not isinstance(item['reasoning'],str) or not item['reasoning'].strip():raise ValueError('Missing reason')
    return value,proof


def grade(api,stages,key,row,answer):
    payloads=view_payloads(row,answer);views={};merged={};judge=stages['evaluation']
    for view,payload in payloads.items():
        path=r.RUN/'evaluation_views'/(r.digest(key+':'+view)+'.json')
        if path.exists():
            item=r.read(path)
            assert item['input']==payload and item['judge_sha256']==judge['sha256']
        else:
            reply=api.ask(key+':judge:'+view,[{'role':'system','content':judge['text']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=3500)
            value,proof=parse_view_with_proof(reply['text'],view)
            item={'input':payload,'value':value,'raw_record':reply['raw_record'],'judge_sha256':judge['sha256'],'format_integrity':proof}
            r.save(path,item)
        views[view]=item;merged.update(item['value'])
    assert set(merged)==set(DIMS)
    return {'dimensions':{k:merged[k]['score'] for k in DIMS},'reasons':{k:merged[k]['reasoning'] for k in DIMS},
      'raw_output':merged,'raw_record':views['facts_context']['raw_record'],'raw_records':[v['raw_record'] for v in views.values()],
      'evaluation_input':payloads,'views':views,'judge_sha256':judge['sha256'],'repetitions':1,'format_integrity':None,'physical_view_isolation':True}


def effective_values(dimensions):
    values=[dimensions[k] for k in DIMS[:4] if dimensions[k] is not None]
    return values or None


def stop_decision(previous,current,min_gain=1.):
    if {k for k in DIMS[:4] if previous[k] is None}!={k for k in DIMS[:4] if current[k] is None}:return 'primary_assessability_changed'
    old,new=effective_values(previous),effective_values(current)
    if old is None or new is None:return 'primary_metric_unassessable'
    if any(y<x for x,y in zip(old,new)):return 'primary_dimension_declined'
    if statistics.mean(new)-statistics.mean(old)<min_gain:return 'primary_gain_below_threshold'
    return None


def bind():
    b.DIMS=DIMS;b.LABELS=LABELS;b.PRIMARY=DIMS[:4];b.GOALS=GOALS;b.EVIDENCE=EVIDENCE
    b.validate_qa=validate_qa;b.validate_stage=validate_stage;b.grade=grade;b.effective_values=effective_values;b.stop_decision=stop_decision
    b.primary_perfect=lambda dims: bool(effective_values(dims)) and all(v==100 for v in effective_values(dims))
    runtime=b.bind();runtime.ALLOW_INTERNAL_KNOWLEDGE=True
    import collective_feedback_prompts as fp
    import knowledge_feedback_prompts as kp
    import collective_feedback as cf
    import audit_collective_feedback as af
    for module in (fp,cf,af):
        for name in ('DIAGNOSIS_META','DIAGNOSIS_SPEC','UPDATE_META','UPDATE_SPEC'):setattr(module,name,getattr(kp,name))
    return runtime


if __name__=='__main__':
    try:bind();b.main()
    except Exception as exc:r.mark('failed',error=type(exc).__name__+': '+str(exc));raise
