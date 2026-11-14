import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import five_metrics_workflow as f


class Api:
    def ask(self, key, messages):
        return {'text': 'answer', 'raw_record': key}


class FixedIterationsTest(unittest.TestCase):
    def run_case(self, fixed, stop='primary_gain_below_threshold', unchanged=False):
        with tempfile.TemporaryDirectory() as directory:
            cfg={'max_rounds':5, 'min_gain':1, 'metas':{'evaluation':'rubric'}}
            if fixed is not None:cfg['case_iterations']=fixed
            stages={'qa':{'text':'initial','sha256':'initial-hash'},'update':{'text':'updater'}}
            row={'id':'example','question':'question'}
            generated=lambda api,key,*a: {'text':'initial' if unchanged else 'prompt-'+key.rsplit(':',1)[-1]}
            grade=lambda *args: {'dimensions':{'x':1},'raw_output':{'x':1}}
            with patch.object(f.r,'RUN',Path(directory)), patch.object(f.r,'generated',side_effect=generated), \
                 patch.object(f.r,'grade_payload',return_value={}),patch.object(f.r,'contract',return_value={}), \
                 patch.object(f.r,'answer_messages',return_value=[]),patch.object(f,'grade',side_effect=grade), \
                 patch.object(f,'effective_values',return_value=[1]),patch.object(f,'primary_perfect',return_value=False), \
                 patch.object(f,'stop_decision',return_value=stop):
                state=f.train_one(Api(),stages,cfg,row)
                if fixed is not None:
                    with self.assertRaises(ValueError):f.train_one(Api(),stages,{**cfg,'case_iterations':fixed+1},row)
                return state

    def test_fixed_counts_updates_even_if_prompt_unchanged(self):
        state=self.run_case(3, unchanged=True)
        self.assertEqual(len(state['history']),4)
        self.assertEqual(state['selected_iteration'],3)
        self.assertEqual(state['stop_reason'],'fixed_iterations_complete')

    def test_zero_updates_evaluates_initial(self):
        state=self.run_case(0)
        self.assertEqual(len(state['history']),1)
        self.assertEqual(state['selected_iteration'],0)

    def test_default_keeps_adaptive_stop_and_prior_version(self):
        state=self.run_case(None)
        self.assertEqual(len(state['history']),2)
        self.assertEqual(state['selected_iteration'],0)
        self.assertEqual(state['stop_reason'],'primary_gain_below_threshold')


if __name__=='__main__':unittest.main()
