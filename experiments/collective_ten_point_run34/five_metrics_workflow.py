"""Isolated five-metric, collective-only workflow; old runtimes remain intact."""
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import restored_meta_runner as r
from rejudge_five_runner import DIMS, parse_grade

LABELS = ('事实准确性', '上下文忠实度', '问题响应完整性', '回答冗余度', '参考语义相似性')
PRIMARY = DIMS[:4]
GOALS = '''事实准确性：核对已陈述事实的主体、关系、数字、时间及条件，不编造或张冠李戴。
上下文忠实度：实质断言须由当前上下文支持，不跨对象拼接，不把建议写成已确认事实。
问题响应完整性：先识别问题必答要点，再覆盖有证据的部分；材料确实不足时准确说明缺失，不猜测补齐，也不整体拒答。
回答冗余度：只保留作答及理解结论必要的内容，避免无关扩展、重复和套话；完整回答后尽可能简洁，不限固定字数。
参考语义相似性为辅助：训练时可参考标准答案理解要点，不能复制具体答案；新问题没有参考答案，不要求读取或比较它。'''
EVIDENCE = '给定上下文可能包含噪音、无关信息或相互矛盾的信息。无论给定多少条上下文，都必须先根据当前问题对全部上下文进行相关性和证据价值重排序（rerank），仅保留并使用你判断对回答有用的上下文，丢弃无用或不可靠的信息；不设固定保留条数，允许全部保留或全部舍弃，证据不足时明确说明。'
STRUCTURE = '成品QA提示使用[角色][任务][输入][证据使用][作答要求][输出]六部分。直接接收当前问题和未标注top5并生成答案，不再生成提示；只输出最终答案，不输出思考过程、排序或保留编号。'


def metas(rubric):
    evaluation = '''[角色]
你是知识库问答评价指令设计专家。
[任务]
生成一份可直接执行的答案评价指令，不输入具体案例。成品直接评价答案，不再生成评价提示。以下完整规范是案例迭代和最终评价的共同标准，原样保留。
[必须保留的评价规范]
''' + rubric.strip() + '''
[生成指令的输出要求]
只输出评价指令正文，完整保留上述全部规范，不加总分、额外指标或案例。'''
    return {
        'qa': '''[角色]
你是面向知识库问答的提示设计专家。
[任务]
生成一份可直接回答问题的通用初始QA提示，不输入具体案例。
[成品输入]
当前问题和未标注的top5上下文。
[作答目标]
''' + GOALS + '\n[证据使用要求]\n' + EVIDENCE + '\n[输出要求]\n' + STRUCTURE + '\n规则可迁移，不硬编码实体、数值或案例答案。只输出QA提示正文，不输出JSON。',
        'evaluation': evaluation,
        'update': '''[角色]
你是知识库问答提示更新指令设计专家。
[任务]
生成一份可直接改写当前QA提示的通用更新指令，不输入具体案例，不再生成更新指令。
[成品输入]
问题、参考答案、实际答案、实际top5、当前QA提示，以及五项评价各自的分数和理由、完整评价标准。
[更新要求]
逐项读取事实准确性、上下文忠实度、问题响应完整性、回答冗余度和参考语义相似性的分数与理由，对照案例材料核实，再将有据的问题落实为可迁移的提示修改，不只看均分。
准确性只纠正已陈述事实错误；忠实度修正无依据补充和误用材料；完整性补上已有证据的遗漏，不要求编造缺失信息；冗余度去掉不必要内容，不删必答要点。参考相似性仅辅助理解语义要点，不为贴近参考而加入问题未问的内容。
评分与理由不符或与材料冲突时，不迎合错误评分。保留有效规则，避免修正一个问题引入新问题。作答要求与给出的评价标准一致，并明确告诉模型如何执行。
''' + GOALS + '\n' + EVIDENCE + '\n[输出要求]\n只输出改写后的QA提示正文，保持六段结构，不输出当前案例答案、诊断或JSON；不写入具体参考答案、实体、日期或数值。作答时不提供参考、评分或历史记录，不能要求读取它们。',
        'collective': '''[角色]
你是群智提示融合指令设计专家。
[任务]
生成一份通用群智融合指令。该指令直接从同一簇训练案例的问题、参考答案和最终QA提示中总结共同问题类型，生成该类型可直接作答的群智QA提示，不再生成融合指令。
[成品输入]
簇编号、簇内各案例的问题、参考答案、最终QA提示，以及完整评价标准；不提供测试题或案例迭代记录。
[融合要求]
先归纳共同提问意图、答案形态和关键条件，再形成针对该问题类型的作答要求。融合有效且互补的方法，去重并协调冲突，不机械拼接，不把个例限制推广到全簇。
''' + GOALS + '\n' + EVIDENCE + '\n[输出要求]\n只输出一份该簇的QA提示正文。' + STRUCTURE + '\n各类型分簇保留，不合并全局提示；不复制训练答案。最终作答没有参考、评分、其他答案或诊断，不要求读取这些信息，不引入多次答案生成。'
    }


_old_validate_qa = r.validate_qa


def validate_qa(text):
    try:
        text = _old_validate_qa(text)
    except ValueError:
        # Explicit semantic review of an unscored first output, not a score-based choice.
        normalized=r.normalize(text); p=r.ROOT/'qa_semantic_approvals.json'
        review=r.read(p).get(r.digest(normalized)) if p.exists() else None
        if not review or set(review.get('primary_goal_excerpts',{}))!=set(PRIMARY):raise
        if any(not quote or quote not in normalized for quote in review['primary_goal_excerpts'].values()):raise
        if any(not r.section(normalized,h) for h in r.HEADINGS):raise
        text=normalized
    if not re.search(r'简洁|冗余|重复|无关扩展', r.section(text, '作答要求')):
        raise ValueError('作答要求须在覆盖必要内容后保持简洁，去掉冗余')
    return text


def validate_stage(kind, text, rubric):
    text = r.normalize(text)
    if kind == 'qa': return validate_qa(text)
    if kind == 'evaluation':
        if rubric not in text: raise ValueError('原样保留给定完整评价提示，不改变指标或评分尺度')
        return text
    for heading in ('角色', '任务', '输入'):
        if not r.section(text, heading): raise ValueError('缺少[' + heading + ']')
    task = r.section(text, '任务')
    positive_task = re.sub(r'(?:不再|不得|禁止|不能|不要)(?:再次)?(?:生成|设计)(?:一份|新的|通用)?(?:更新指令|融合指令)', '', task)
    if re.search(r'(?:生成|设计)(?:一份|新的|通用)?(?:更新指令|融合指令)', positive_task):
        raise ValueError('成品必须直接更新QA或融合出QA，不再生成阶段指令')
    if kind == 'update':
        if not re.search(r'更新|改写', task): raise ValueError('任务须直接改写QA提示')
        if not all(label in text for label in LABELS): raise ValueError('逐项处理五项分数和理由')
        if not re.search(r'分数|评分', text) or '理由' not in text: raise ValueError('必须参考评分和理由')
        if not re.search(r'参考.*(?:辅助|仅)|相似性.*辅助', text): raise ValueError('参考相似性仅辅助，不新增必答项')
        output = text.split('[输出结构]',1)[-1].split('[边界]',1)[0] if '[输出结构]' in text else ''
        if not all('['+h+']' in output for h in r.HEADINGS):
            raise ValueError('输出结构须明确列出QA的[角色][任务][输入][证据使用][作答要求][输出]，不能换成其他六段')
    else:
        if not re.search(r'问题类型|共同[^。\n]{0,12}(?:提问意图|核心意图)', text) or '簇' not in text:
            raise ValueError('须归纳簇的共同提问意图、答案形态及条件，形成针对性要求')
    if not all(h in text for h in r.HEADINGS): raise ValueError('输出QA须保留六段结构')
    return text


def effective_values(dimensions):
    if any(dimensions[k] is None for k in PRIMARY): return None
    return [dimensions[k] if k != 'answer_redundancy' else 100 - dimensions[k] for k in PRIMARY]


def stop_decision(previous, current, min_gain=1.):
    old, new = effective_values(previous), effective_values(current)
    if old is None or new is None: return 'primary_metric_unassessable'
    if any(b < a for a, b in zip(old, new)): return 'primary_dimension_declined'
    if statistics.mean(new) - statistics.mean(old) < min_gain: return 'primary_gain_below_threshold'
    return None


def primary_perfect(dimensions):
    return effective_values(dimensions)==[100,100,100,100]


def grade(api, stages, key, row, answer):
    payload = r.grade_payload(row, answer)
    result = api.ask(key + ':judge', [{'role': 'system', 'content': stages['evaluation']['text']},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], max_tokens=5000)
    value = parse_grade(result['text'])
    return {'dimensions': {k:value[k]['score'] for k in DIMS}, 'reasons': {k:value[k]['reasoning'] for k in DIMS},
        'raw_output': value, 'raw_record':result['raw_record'], 'evaluation_input':payload,
        'judge_sha256':stages['evaluation']['sha256'], 'repetitions':1, 'format_integrity':None}


def train_one(api, stages, cfg, row):
    path = r.RUN / 'training/cases' / (row['id'] + '.json')
    state = r.read(path) if path.exists() else {'id':row['id'], 'question':row['question'], 'history':[], 'status':'running'}
    if state['status'] == 'complete': return state
    accepted = 0
    for iteration in range(cfg['max_rounds'] + 1):
        if iteration < len(state['history']):
            if state['history'][iteration].get('accepted'): accepted = iteration
            continue
        p = r.RUN / 'training/versions' / row['id'] / (str(iteration) + '.json')
        if p.exists(): version = r.read(p)
        else:
            if iteration == 0:
                prompt = stages['qa']['text']; origin = {'shared_initial_sha256':stages['qa']['sha256']}
            else:
                previous = state['history'][accepted]
                payload = {**r.grade_payload(row, previous['answer']), 'current_qa_prompt':previous['prompt'],
                    'evaluation':previous['evaluation']['raw_output'], 'evaluation_criteria':r.contract(cfg['metas']['evaluation'])}
                try:
                    origin = r.generated(api, 'train:' + row['id'] + ':update:' + str(iteration), stages['update']['text'], payload, validate_qa)
                except RuntimeError as exc:
                    if not str(exc).startswith('Prompt compliance failed: '): raise
                    state.update(status='complete', selected_iteration=accepted, prompt=previous['prompt'], stop_reason='update_generation_noncompliant')
                    r.save(path, state); return state
                prompt = origin['text']
                if prompt == previous['prompt']:
                    state.update(status='complete', selected_iteration=accepted, prompt=prompt, stop_reason='prompt_unchanged')
                    r.save(path, state); return state
            version = {'iteration':iteration, 'prompt':prompt, 'origin':origin, 'answer':None, 'evaluation':None}
            r.save(p, version)
        if version['answer'] is None:
            reply = api.ask('train:' + row['id'] + ':answer:' + str(iteration), r.answer_messages(row, version['prompt']))
            version.update(answer=reply['text'], answer_raw_record=reply['raw_record']); r.save(p, version)
        if version['evaluation'] is None:
            version['evaluation'] = grade(api, stages, 'train:' + row['id'] + ':' + str(iteration), row, version['answer']); r.save(p, version)
        dims = version['evaluation']['dimensions']
        decision = stop_decision(state['history'][accepted]['evaluation']['dimensions'], dims, cfg['min_gain']) if iteration else None
        version['accepted'] = decision is None; r.save(p, version); state['history'].append(version)
        if decision is None: accepted = iteration
        best = effective_values(dims)
        stop = decision or ('primary_metrics_perfect' if primary_perfect(dims) else
                           'primary_metric_unassessable' if best is None else
                           'safety_round_limit' if iteration == cfg['max_rounds'] else None)
        if stop:
            state.update(status='complete', selected_iteration=accepted, prompt=state['history'][accepted]['prompt'], stop_reason=stop)
            r.save(path, state); return state
        r.save(path, state)


def bind():
    r.DIMS = DIMS; r.LABELS = LABELS[:4]  # Auxiliary reference comparison is not a QA requirement.
    r.validate_qa = validate_qa; r.validate_stage = validate_stage
    r.grade = grade; r.train_one = train_one; r.stop_decision = stop_decision
    return r


def infer(api, stages, cfg, tests, clusters, clustering):
    artifact = {'judge':stages['evaluation']['sha256'], 'clusters':{k:v['sha256'] for k,v in clusters.items()},
        'groups':['collective'], 'test_inputs_sha256':r.digest(tests), 'frozen_at':time.time(),
        'feedback_manifest_sha256':r.file_sha(r.RUN / 'collective_feedback/final_manifest.json')}
    p = r.RUN / 'inference_frozen.json'
    if p.exists():
        prior = r.read(p)
        assert all(prior[k] == artifact[k] for k in artifact if k != 'frozen_at')
    else: r.save(p, artifact)
    r.mark('full_inference_and_evaluation', total=len(tests))

    def one(row):
        path = r.RUN / 'settings/collective/records' / (row['id'] + '.json')
        cid = clustering['test_routes'][row['id']]['cluster_id']; prompt = clusters[str(cid)]['text']
        value = r.read(path) if path.exists() else {'id':row['id'], 'question':row['question'],
            'reference_answer':row['reference_answer'], 'actual_top5':r.context_view(row), 'group':'collective',
            'cluster_id':cid, 'question_type':clusters[str(cid)]['question_type'], 'prompt':prompt,
            'answer_calls':1, 'answer':None, 'evaluation':None}
        if value.get('evaluation') or value.get('status') == 'failed': return value
        try:
            if value['answer'] is None:
                reply = api.ask('final:collective:' + row['id'] + ':answer', r.answer_messages(row, prompt))
                value.update(answer=reply['text'], answer_raw_record=reply['raw_record']); r.save(path, value)
            value['evaluation'] = grade(api, stages, 'final:collective:' + row['id'], row, value['answer'])
            value['status'] = 'complete'
        except Exception as exc:
            value.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        r.save(path, value); return value

    completed = valid = 0
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        for value in pool.map(one, tests):
            completed += 1; valid += bool(value.get('evaluation'))
            r.mark('full_inference_and_evaluation', completed=completed, valid=valid, total=len(tests))


def main():
    runtime = bind(); cfg = r.read(r.ROOT / 'config.json')
    assert cfg['groups'] == ['collective'] and cfg['judge_repetitions'] == 1
    frozen = r.read(r.RUN / 'frozen_manifest.json')
    assert cfg == frozen['config']
    for name, digest in frozen['source_sha256'].items(): assert r.file_sha(r.ROOT / name) == digest, name
    for name, digest in frozen['input_sha256'].items(): assert r.file_sha(r.RUN / 'shared' / name) == digest, name
    api = r.API(cfg); stages = r.generate_stages(api, cfg)
    if stages is None: return
    train = r.read(r.RUN / 'shared/train_top5.json'); tests = r.read(r.RUN / 'shared/test_top5.json')
    cases = r.train(api, stages, cfg, train)
    clustering = r.cluster(cfg, train, tests, cases)
    clusters = r.fuse(api, stages, cfg, train, cases, clustering)
    if clusters is None: return
    from collective_feedback import optimize
    clusters = optimize(api, cfg, train, cases, clustering, clusters, runtime)
    if clusters is None: return
    from audit_collective_feedback import audit
    audit(r.RUN, cfg, runtime)
    # Use the original single-call answer format, but only the requested method.
    infer(api, stages, cfg, tests, clusters, clustering)
    r.mark('mechanical_metrics_and_report')
    subprocess.run([cfg['machine_python'], str(r.ROOT / 'five_metrics_report.py'), str(r.RUN), str(r.ROOT)], check=True)
    r.mark('complete', questions=len(tests), groups=['collective'], k=clustering['k'])


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        r.mark('failed', error=type(exc).__name__ + ': ' + str(exc)); raise
