"""Offline contract tests; no real endpoint calls."""
import json
from pathlib import Path
import tempfile
import unittest

import ten_point_workflow as t


class Tests(unittest.TestCase):
    row={'question':'Q','reference_answer':'REF_ONLY','context_documents':[{'text':'CTX_ONLY','doc_title':'D'} for _ in range(5)]}

    def output(self,view,score=8):
        return json.dumps({key:{'score':0 if key=='answer_preference' else score,'reasoning':'具体依据'} for key in t.VIEWS[view]},ensure_ascii=False)

    def test_training_has_only_four_metrics_two_views(self):
        views=t.view_payloads(self.row,'A',final=False)
        self.assertEqual(set(views),{'facts_context','question_answer'})
        self.assertNotIn('REF_ONLY',json.dumps(views))
        self.assertNotIn('CTX_ONLY',json.dumps(views['question_answer']))

    def test_final_comparison_has_all_four_inputs(self):
        view=t.view_payloads(self.row,'A')['reference_comparison']
        self.assertEqual(set(view),{'evaluation_view','question','contexts','reference_answer','candidate_answer'})

    def test_integer_scores_only(self):
        for value in (0,11,8.0,True,'8'):
            with self.assertRaises(ValueError):t.parse_view_with_proof(self.output('facts_context',value),'facts_context')
        for value in (1,10):t.parse_view_with_proof(self.output('facts_context',value),'facts_context')

    def test_preference_distinct_scale(self):
        for score in (-1,0,1):
            value=json.loads(self.output('reference_comparison'));value['answer_preference']['score']=score
            t.parse_view_with_proof(json.dumps(value),'reference_comparison')
        for score in (2,None,True,1.0):
            value=json.loads(self.output('reference_comparison'));value['answer_preference']['score']=score
            with self.assertRaises(ValueError):t.parse_view_with_proof(json.dumps(value),'reference_comparison')

    def test_null_not_zero(self):
        t.parse_view_with_proof(self.output('facts_context',None),'facts_context')
        with self.assertRaises(ValueError):t.parse_view_with_proof(self.output('question_answer',None),'question_answer')

    def test_same_score_reason_structure(self):
        value,_=t.parse_view_with_proof(self.output('reference_comparison'),'reference_comparison')
        self.assertTrue(all(set(v)=={'score','reasoning'} for v in value.values()))

    def test_exact_view_envelope_preserves_low_scores_and_preference(self):
        source={'reference_similarity':{'score':1,'reasoning':'原理由'},'answer_preference':{'score':-1,'reasoning':'原比较'}}
        value,proof=t.parse_view_with_proof(json.dumps({'reference_comparison':source}),'reference_comparison')
        self.assertEqual(value,source);self.assertTrue(proof['scores_unchanged'])

    def test_reason_continuation_and_missing_comma_keep_all_words(self):
        text='{"factual_accuracy":{"score":1,\n"reasoning":"原文"\n"reasoning_additional":"后续原文"},"context_faithfulness":{"score":10,"reasoning":"另一理由"}}'
        value,proof=t.parse_view_with_proof(text,'facts_context')
        self.assertEqual(value['factual_accuracy'],{'score':1,'reasoning':'原文\n后续原文'})
        self.assertEqual(value['context_faithfulness']['score'],10)
        self.assertTrue(proof['all_reason_text_preserved'])

    def test_unknown_reason_fields_not_dropped(self):
        value=json.loads(self.output('facts_context'));value['factual_accuracy']['other']='不可静默丢弃'
        with self.assertRaises(ValueError):t.parse_view_with_proof(json.dumps(value),'facts_context')

    def test_wrong_view_envelope_not_unwrapped(self):
        with self.assertRaises(ValueError):t.parse_view_with_proof(json.dumps({'wrong':json.loads(self.output('facts_context'))}),'facts_context')

    def test_exact_metrics_envelope_keeps_low_score(self):
        original=json.loads(self.output('question_answer',1))
        parsed,proof=t.parse_view_with_proof(json.dumps({'evaluation_view':'question_answer','metrics':original}),'question_answer')
        self.assertEqual(parsed,original);self.assertTrue(proof['scores_unchanged'])

    def test_missing_root_closer_cannot_invent_reason_text(self):
        original=self.output('reference_comparison',3)
        parsed,proof=t.parse_view_with_proof(original[:-1],'reference_comparison')
        self.assertEqual(parsed,json.loads(original));self.assertTrue(proof['all_reason_text_preserved'])
        with self.assertRaises(ValueError):t.parse_view_with_proof(original[:original.index('具体依据')+2],'reference_comparison')

    def test_trailing_reason_comma_only_is_removed(self):
        text='{"reference_similarity":{"score":1,\n"reasoning":"原理由",\n},"answer_preference":{"score":-1,"reasoning":"比较原文"}}'
        parsed,proof=t.parse_view_with_proof(text,'reference_comparison')
        self.assertEqual(parsed['reference_similarity'],{'score':1,'reasoning':'原理由'})
        self.assertEqual(parsed['answer_preference']['score'],-1)
        self.assertTrue(proof['scores_unchanged'])

    def test_auxiliary_does_not_affect_selection(self):
        old=dict.fromkeys(t.PRIMARY,8);new={**old,'reference_similarity':10,'answer_preference':1}
        self.assertEqual(t.stop_decision(old,new),'primary_gain_below_threshold')
        self.assertIsNone(t.stop_decision(old,{**new,'answer_conciseness':9}))

    def test_perfect_training_score_is_ten(self):
        runtime=t.bind()
        self.assertTrue(t.b.primary_perfect(dict.fromkeys(t.PRIMARY,10)))
        self.assertFalse(t.b.primary_perfect(dict.fromkeys(t.PRIMARY,9)))
        self.assertIs(runtime.grade,t.grade)

    def test_cache_keeps_first_valid_integer_score(self):
        class API:
            def __init__(self):self.calls=[]
            def ask(inner,key,messages,**kwargs):
                inner.calls.append(key);view=json.loads(messages[1]['content'])['evaluation_view']
                return {'text':self.output(view,6),'raw_record':'raw_'+view}
        old=t.r.RUN
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                t.r.RUN=Path(folder);api=API();stages={'evaluation':{'text':'J','sha256':'JHASH'}}
                first=t.grade(api,stages,'train:id:0',self.row,'A')
                self.assertEqual(set(first['dimensions']),set(t.PRIMARY))
                second=t.grade(api,stages,'train:id:0',self.row,'A')
                self.assertEqual(first,second);self.assertEqual(len(api.calls),2)
                final=t.grade(api,stages,'final:collective:id',self.row,'A')
                self.assertEqual(set(final['dimensions']),set(t.DIMS));self.assertEqual(len(api.calls),5)
        finally:t.r.RUN=old

    def test_approved_prompt_removals(self):
        rubric=Path(__file__).with_name('ten_point_rubric.txt').read_text(encoding='utf-8')
        for text in ('80核心','允许一位小数','有效取得但完全未作答'):
            self.assertNotIn(text,rubric)
        self.assertIn('整数',rubric);self.assertIn('answer_preference',rubric)


if __name__=='__main__':unittest.main()
