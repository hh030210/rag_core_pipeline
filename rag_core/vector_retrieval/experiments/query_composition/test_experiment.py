"""Focused validation of query composition, fusion and metric boundaries."""
import unittest
from rag_core.vector_retrieval.experiments.query_composition.run import prepare, centroid, rrf
from rag_core.evaluation import route_metrics
from rag_core.vector_retrieval import VectorRetriever

class CompositionTests(unittest.TestCase):
    def test_explicit_queries_preserve_duplicates_and_missing_original(self):
        q = prepare({"question": "Q", "subqueries": ["A", "A", "", "Q"], "gold_chunk_ids": ["g"]})
        self.assertEqual(q["texts"]["subqueries_concat"], "A | A | Q")
        self.assertEqual(q["texts"]["original_plus_subqueries_concat"], "Q | A | A | Q")
        self.assertEqual(q["texts"]["original_plus_subqueries_dedup"], "Q | A")
        self.assertEqual(q["component_texts"], ["Q", "A"])
        with self.assertRaises(ValueError):
            prepare({"subqueries": ["A"]})

    def test_saved_input_and_empty_fallback(self):
        q = prepare({"question": "Q", "retrieval_query": " A | B "})
        self.assertEqual(q["parts"], ["A", "B"])
        q = prepare({"question": "Q", "subqueries": []})
        self.assertTrue(q["empty_subqueries_fallback"])
        self.assertEqual(q["texts"]["subqueries_concat"], "Q")
        with self.assertRaises(ValueError):
            prepare({"question": "Q"})

    def test_weighted_centroid_normalizes_each_component(self):
        v = centroid([[100, 0], [0, 1]], [2, 1])
        self.assertAlmostEqual(float(v[0]), 2 / (5 ** 0.5), places=6)
        self.assertAlmostEqual(float(v[1]), 1 / (5 ** 0.5), places=6)

    def test_rrf_overlap_and_unique_candidates(self):
        lists = [[{"chunk_id": "a"}, {"chunk_id": "b"}],
                 [{"chunk_id": "c"}, {"chunk_id": "b"}]]
        self.assertEqual([x["chunk_id"] for x in rrf(lists, 3)], ["b", "a", "c"])

    def test_metrics_cutoff_and_multiple_gold(self):
        m = route_metrics([{"chunk_id": "x"}, {"chunk_id": "g"}], ["g", "h"], [1, 5])
        self.assertFalse(m["hit_at_1"])
        self.assertEqual(m["rr_at_1"], 0)
        self.assertEqual(m["rr_at_5"], 0.5)
        self.assertEqual(m["gold_recall_at_5"], 0.5)

    def test_current_filter_is_preserved_with_supplied_vector(self):
        class Encoder:
            def encode(self, texts):
                raise AssertionError("Supplied vector must not be reencoded")
        class Store:
            def semantic_search(self, vector, limit):
                return [
                    {"id": "a", "score": 1, "payload": {"spot_name": "wrong"}},
                    {"id": "b", "score": 0.8, "payload": {"spot_name": "wanted"}},
                ]
        hits = VectorRetriever(embeddings=Encoder(), vector_store=Store()).search(
            ["A"], 20, original_query="Q", query_vector=[1, 0], spot_names=["wanted"])
        self.assertEqual([h["chunk_id"] for h in hits], ["b"])
        self.assertEqual(hits[0]["rank"], 1)

if __name__ == "__main__":
    unittest.main()

