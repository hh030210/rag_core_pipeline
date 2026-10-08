"""Synthetic regression checks for raw scores, head gaps and route absence."""
import copy
import unittest

from strategies import fuse_raw


def snapshot(scores, dimension=None, old_order=None):
    sem = [{'chunk_id': str(i), 'rank': i+1, 'score': score,
            'normalized_semantic_score': 0.123} for i, score in enumerate(scores)]
    dim = [{'chunk_id': str(cid), 'rank': i+1, 'score': score,
            'normalized_dimension_score': 0.456} for i, (cid, score) in enumerate(dimension or [])]
    ids = old_order or list(dict.fromkeys(x['chunk_id'] for x in sem+dim))
    return {'query': '', 'top_k': 10, 'semantic_candidates': sem, 'dimension_candidates': dim,
            'fusion_candidates': [{'chunk_id': cid, 'score': 0.5, 'chunk_text': ''} for cid in ids]}


class RawFusionTests(unittest.TestCase):
    def test_top_two_plateau_and_third_score_drop(self):
        data = snapshot([.89, .88, .7])
        head = fuse_raw(data, dict(mode='raw_lex_head', head=9, threshold=.1,
                                  lexical=.1, lexical_boost=.025))['diagnostics']
        top1 = fuse_raw(data, dict(mode='raw_lex_top1', threshold=.1,
                                  lexical=.1, lexical_boost=.025))['diagnostics']
        self.assertAlmostEqual(head['semantic_head_gap'], .18)
        self.assertEqual(head['semantic_head_gap_position'], 2)
        self.assertEqual(head['effective_lexical_weight'], .1)
        self.assertGreater(top1['effective_lexical_weight'], head['effective_lexical_weight'])

    def test_both_route_terms_use_original_scores(self):
        result = fuse_raw(snapshot([.7], [('0', 3)]),
                          dict(mode='raw_fixed_imputed', beta=.1, lexical=0))
        c = result['fusion_candidates'][0]
        self.assertAlmostEqual(c['score'], 1.0)
        self.assertAlmostEqual(c['score_components']['dimension'], .3)
        self.assertEqual(c['semantic_score'], .7)
        self.assertEqual(c['dimension_score'], 3)

    def test_semantic_absence_is_recorded_as_an_estimate(self):
        result = fuse_raw(snapshot([.89, .7], [('dimension-only', 2)]),
                          dict(mode='raw_fixed_imputed', lexical=0, missing_penalty=.025))
        c = next(x for x in result['fusion_candidates'] if x['chunk_id']=='dimension-only')
        self.assertIsNone(c['semantic_score'])
        self.assertTrue(c['semantic_score_imputed'])
        self.assertAlmostEqual(c['effective_semantic_score'], .675)

    def test_zero_is_an_observed_score(self):
        c = fuse_raw(snapshot([0]), dict(mode='raw_fixed_imputed', lexical=0))['fusion_candidates'][0]
        self.assertEqual(c['semantic_score'], 0)
        self.assertFalse(c['semantic_score_imputed'])

    def test_empty_semantic_route(self):
        result = fuse_raw(snapshot([], [('x', 2)]), dict(mode='raw_lex_head', beta=.1, lexical=0))
        self.assertEqual(result['fusion_candidates'][0]['score'], .2)
        self.assertIsNone(result['diagnostics']['semantic_head_gap_position'])
        self.assertEqual(result['diagnostics']['semantic_missing_estimate'], 0)

    def test_ties_use_saved_order_and_input_is_unchanged(self):
        data = snapshot([.7, .7], old_order=['1', '0'])
        before = copy.deepcopy(data)
        result = fuse_raw(data, dict(mode='raw_fixed_imputed', lexical=0))
        self.assertEqual([c['chunk_id'] for c in result['fusion_candidates']], ['1', '0'])
        self.assertEqual(data, before)

    def test_labels_in_snapshot_cannot_change_ranking(self):
        data = snapshot([.7, .6])
        config = dict(mode='raw_lex_head')
        before = fuse_raw(data, config)
        data['gold'] = {'chunk_ids': ['1']}
        data['reference_answer'] = '1'
        self.assertEqual(before, fuse_raw(data, config))


if __name__ == '__main__':
    unittest.main()
