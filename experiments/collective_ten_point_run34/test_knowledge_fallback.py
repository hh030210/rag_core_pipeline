"""Offline checks: source isolation, score direction, first-valid caching, policy."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import knowledge_fallback_workflow as k


class Tests(unittest.TestCase):
    row={'question':'Q','reference_answer':'REFERENCE_SECRET','context_documents':[{'text':'CONTEXT_SECRET','doc_title':'D'} for _ in range(5)]}

    def dims(self,**updates):
        value=dict(zip(k.DIMS,(90,90,90,90,80)));value.update(updates);return value

    def test_view_inputs_physically_separate(self):
        views=k.view_payloads(self.row,'A')
        self.assertNotIn('REFERENCE_SECRET',json.dumps(views['facts_context']))
        self.assertNotIn('reference_answer',views['question_answer'])
        self.assertNotIn('contexts',views['question_answer'])
        self.assertNotIn('CONTEXT_SECRET',json.dumps(views['question_answer']))
        self.assertEqual(set(views['reference_answer']),{'evaluation_view','reference_answer','candidate_answer'})

    def test_answer_accuracy_high_is_good(self):
        self.assertIsNone(k.stop_decision(self.dims(),self.dims(answer_accuracy=95)))
        self.assertEqual(k.stop_decision(self.dims(),self.dims(answer_accuracy=85)),'primary_dimension_declined')

    def test_reference_only_gain_does_not_select(self):
        self.assertEqual(k.stop_decision(self.dims(),self.dims(reference_similarity=100)),'primary_gain_below_threshold')

    def test_nulls_not_zero_or_hidden_improvement(self):
        old=self.dims(context_faithfulness=None)
        self.assertEqual(k.effective_values(old),[90,90,90])
        self.assertIsNone(k.stop_decision(old,self.dims(context_faithfulness=None,answer_accuracy=96)))
        self.assertEqual(k.stop_decision(self.dims(),old),'primary_assessability_changed')

    def test_wrong_view_fields_rejected(self):
        with self.assertRaises(ValueError):k.parse_view(json.dumps({'reference_similarity':{'score':100,'reasoning':'x'}}),'facts_context')

    def test_exact_metric_schema(self):
        self.assertNotIn('answer_redundancy',k.DIMS)
        value={d:{'score':20,'reasoning':'低分仍有效'} for d in k.VIEWS['question_answer']}
        self.assertEqual(k.parse_view(json.dumps(value),'question_answer'),value)

    def test_knowledge_rule_and_shared_evaluation(self):
        rubric=(Path(__file__).parent/'knowledge_fallback_rubric.txt').read_text(encoding='utf-8')
        metas=k.metas(rubric)
        self.assertEqual(k.r.contract(metas['evaluation']),rubric.strip())
        for name in ('qa','update','collective'):
            self.assertIn(k.EVIDENCE,metas[name]);self.assertIn(k.GOALS,metas[name])

    def test_view_cache_never_resamples_valid_low(self):
        calls=[]
        def ask(key,messages,max_tokens):
            view=json.loads(messages[1]['content'])['evaluation_view'];calls.append(key)
            value={d:{'score':0,'reasoning':'有效低分不可丢弃'} for d in k.VIEWS[view]}
            return {'text':json.dumps(value),'raw_record':view+'.json'}
        old=k.r.RUN
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                assert Path(folder).resolve().parent==Path(__file__).resolve().parent
                k.r.RUN=Path(folder);api=SimpleNamespace(ask=ask);stages={'evaluation':{'text':'RULE','sha256':'HASH'}}
                first=k.grade(api,stages,'test',self.row,'A');second=k.grade(api,stages,'test',self.row,'A')
                self.assertEqual(len(calls),3);self.assertEqual(first,second);self.assertTrue(all(v==0 for v in first['dimensions'].values()))
        finally:k.r.RUN=old

    def test_feedback_has_no_context_only_prohibition(self):
        k.bind()
        import collective_feedback as cf
        payload=cf.diagnostic_payload(self.row,'P','A','P2','A2',runtime=k.r)
        self.assertEqual(payload['answer_policy'],'context_first_allow_internal_knowledge')
        self.assertIn('纯知识补全部分允许context_excerpt为空',cf.DIAGNOSIS_SPEC)
        self.assertIn(k.EVIDENCE,cf.UPDATE_SPEC)

    def test_forbidding_false_attribution_does_not_ban_knowledge(self):
        k.bind()
        text='[角色]\n问答专家。\n[任务]\n回答问题。\n[输入]\n问题和上下文。\n[证据使用]\n'+k.EVIDENCE+'\n不得将模型内部知识补充冒称材料原文。\n[作答要求]\n'+k.GOALS+'\n[输出]\n只输出最终答案。'
        self.assertEqual(k.validate_qa(text),text)
        with self.assertRaises(ValueError):k.validate_qa(text.replace('不得将模型内部知识补充冒称材料原文。','不得使用模型内部知识补全。'))

    def test_quote_only_repair_retains_original_scores_and_reason(self):
        raw='{\n"reference_similarity": {\n"score": 20,\n"reasoning": "原文说"两个"要点，不改分。"\n}\n}'
        value,proof=k.parse_view_with_proof(raw,'reference_answer')
        self.assertEqual(value['reference_similarity'],{'score':20,'reasoning':'原文说"两个"要点，不改分。'})
        self.assertTrue(proof['no_score_or_word_edits']);self.assertEqual(len(proof['edits']),2)

    def test_direct_final_answer_synonym_preserves_text_and_guards(self):
        k.bind()
        text='[角色]\n问答专家。\n[任务]\n回答问题。\n[输入]\n问题和上下文。\n[证据使用]\n'+k.EVIDENCE+'\n[作答要求]\n'+k.GOALS+'\n[输出]\n直接输出最终答案文本，不输出思考过程、排序结果或上下文编号。'
        self.assertEqual(k.validate_qa(text),text)
        with self.assertRaises(ValueError):k.validate_qa(text.replace('回答问题。','生成一份QA提示。'))
        with self.assertRaises(ValueError):k.validate_qa(text.replace('直接输出最终答案文本，不输出思考过程','输出思考过程及答案'))

    def test_exact_view_identity_echo_is_metadata_not_a_new_score(self):
        values={d:{'score':20,'reasoning':'原始低分和理由不能重评'} for d in k.VIEWS['question_answer']}
        text=json.dumps({'evaluation_view':'question_answer',**values},ensure_ascii=False)
        parsed,proof=k.parse_view_with_proof(text,'question_answer')
        self.assertEqual(parsed,values);self.assertTrue(proof['no_score_or_reason_edits'])
        with self.assertRaises(ValueError):k.parse_view(text.replace('"evaluation_view": "question_answer"','"evaluation_view": "facts_context"'),'question_answer')
        with self.assertRaises(ValueError):k.parse_view(json.dumps({**values,'unexpected':'x'}),'question_answer')

    def test_literal_reason_newline_preserves_score_and_all_characters(self):
        reason='相似，但分数仍为93。\n    原始说明保留。'
        raw='{ "reference_similarity": {"score": 93.0, "reasoning": "'+reason+'"}}'
        parsed,proof=k.parse_view_with_proof(raw,'reference_answer')
        self.assertEqual(parsed,{'reference_similarity':{'score':93.0,'reasoning':reason}})
        self.assertEqual(proof['kind'],'escape_literal_reason_control_characters')
        self.assertTrue(proof['decoded_values_unchanged'])

    def test_missing_ascii_closer_does_not_replace_chinese_reason_character(self):
        raw='{\n"reference_similarity": {\n"score": 20.0,\n"reasoning": "原始低分理由。”\n}\n}'
        parsed,proof=k.parse_view_with_proof(raw,'reference_answer')
        self.assertEqual(parsed,{'reference_similarity':{'score':20.0,'reasoning':'原始低分理由。”'}})
        self.assertEqual(proof['kind'],'insert_missing_reason_ascii_closer_only')
        self.assertEqual(proof['edits'][0]['insert'],'"')

    def test_quote_guard_does_not_rewrite_other_invalid_content(self):
        raw='{\n"reference_similarity": {\n"score": 20\n"reasoning": "没有逗号"\n}\n}'
        with self.assertRaises(ValueError):k.parse_view(raw,'reference_answer')


if __name__=='__main__':unittest.main()
