import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag_core.qa import QAService
from rag_core.retrieval import Retriever
from rag_core.retrieval_fusion.fusion import apply_fusion


def candidates():
    """Independent semantic, dimension, and lexical signals, plus a route miss."""
    semantic = [
        {"chunk_id": "a", "score": 0.8, "rank": 1, "chunk_text_full": "alpha"},
        {"chunk_id": "b", "score": 0.6, "rank": 2, "chunk_text_full": "unrelated"},
    ]
    dimension = [
        {"chunk_id": "b", "score": 3.0, "rank": 1, "chunk_text_full": "unrelated"},
        {"chunk_id": "c", "score": 2.0, "rank": 2, "chunk_text_full": "alpha"},
        {"chunk_id": "a", "score": 1.0, "rank": 3, "chunk_text_full": "alpha"},
    ]
    return semantic, dimension


class FusionWeightTests(unittest.TestCase):
    def fuse(self, **weights):
        semantic, dimension = candidates()
        return apply_fusion(
            semantic, dimension, top_k=3, query="alpha",
            strategy_file="dimension_score.py", **weights,
        )

    def assert_scores(self, result, semantic_weight=1.0, dimension_weight=0.05,
                      lexical_weight=0.10):
        expected = {
            "a": semantic_weight * 0.8 + lexical_weight,
            "b": semantic_weight * 0.6 + dimension_weight,
            "c": semantic_weight * 0.575 + dimension_weight * 0.5 + lexical_weight,
        }
        actual = {row["chunk_id"]: row for row in result["fusion_candidates"]}
        self.assertEqual(set(actual), set(expected))
        for chunk_id, score in expected.items():
            self.assertAlmostEqual(actual[chunk_id]["score"], score, places=9)
        configuration = result["fusion_strategy"]["configuration"]
        self.assertEqual(configuration["semantic_weight"], semantic_weight)
        self.assertEqual(configuration["dimension_weight"], dimension_weight)
        self.assertEqual(configuration["lexical_weight"], lexical_weight)

    def test_default_formula_and_missing_semantic_penalty_are_preserved(self):
        result = self.fuse()
        self.assert_scores(result)
        self.assertEqual(
            [row["chunk_id"] for row in result["fusion_results"]], ["a", "c", "b"],
        )
        by_id = {row["chunk_id"]: row for row in result["fusion_candidates"]}
        self.assertTrue(by_id["c"]["semantic_score_imputed"])
        self.assertFalse(by_id["a"]["semantic_score_imputed"])

    def test_each_weight_can_be_set_independently_including_zero(self):
        for name, custom in (
            ("semantic_weight", 2.5), ("semantic_weight", 0.0),
            ("dimension_weight", 0.4), ("dimension_weight", 0.0),
            ("lexical_weight", 0.7), ("lexical_weight", 0.0),
        ):
            with self.subTest(name=name, value=custom):
                self.assert_scores(self.fuse(**{name: custom}), **{name: custom})

    def test_all_weights_can_be_zero(self):
        weights = dict(semantic_weight=0.0, dimension_weight=0.0, lexical_weight=0.0)
        self.assert_scores(self.fuse(**weights), **weights)

    def test_request_weights_do_not_change_later_defaults(self):
        before = copy.deepcopy(self.fuse())
        self.fuse(semantic_weight=0.0, dimension_weight=0.7, lexical_weight=2.0)
        after = self.fuse()
        self.assertEqual(after, before)

    def test_other_strategies_still_support_default_calls(self):
        for filename in ("online_adaptive.py", "recommended_fusion.py"):
            with self.subTest(strategy=filename):
                semantic, dimension = candidates()
                result = apply_fusion(
                    semantic, dimension, top_k=2, query="alpha", strategy_file=filename,
                )
                self.assertEqual(result["fusion_strategy"]["file"], filename)
                self.assertEqual(len(result["fusion_results"]), 2)

    def test_other_strategies_reject_unsupported_custom_weights(self):
        for filename in ("online_adaptive.py", "recommended_fusion.py"):
            with self.subTest(strategy=filename):
                semantic, dimension = candidates()
                with self.assertRaisesRegex(ValueError, "dimension_score"):
                    apply_fusion(
                        semantic, dimension, top_k=2, query="alpha",
                        strategy_file=filename, dimension_weight=0.0,
                    )


class FusionWeightForwardingTests(unittest.TestCase):
    def test_retriever_forwards_request_weights_to_fusion(self):
        retriever = Retriever.__new__(Retriever)
        retriever.settings = SimpleNamespace(
            top_k=2, semantic_pool=2, dimension_pool=2,
            fact_anchor_rerank_enabled=False,
        )
        retriever._parse_query = Mock(return_value={"constraints": {}, "spot_names": []})
        retriever._fact_index_matches = Mock(return_value={})
        retriever._dimension_policy = Mock(return_value={})
        retriever.embeddings = Mock()
        retriever.embeddings.encode.return_value = [[1.0, 0.0]]
        retriever.vector_retriever = Mock(strategy_file="original_subqueries_centroid_w2.py")
        retriever.vector_retriever.query_texts.return_value = ["alpha"]
        retriever.vector_retriever.search.return_value = []
        retriever.tags_by_dim = {}
        retriever._entity_anchor_candidates = Mock(return_value=[])
        retriever._fact_index_candidates = Mock(return_value=[])
        retriever._no_candidate_lexical_fallback = Mock(return_value=[])
        for name in (
            "_rerank_main_dimension_coverage", "_rerank_verified_fact_evidence",
            "_rerank_answer_evidence", "_rerank_answerability", "_rerank_slot_coverage",
        ):
            setattr(retriever, name, Mock(side_effect=lambda rows, *args: rows))
        fused = {
            "fusion_strategy": {"file": "dimension_score.py"},
            "fusion_candidates": [], "fusion_results": [],
        }
        weights = dict(semantic_weight=0.0, dimension_weight=0.4, lexical_weight=0.7)
        with patch("rag_core.retrieval.apply_fusion", return_value=fused) as fusion:
            retriever.search("alpha", top_k=2, **weights)
        for key, value in weights.items():
            self.assertEqual(fusion.call_args.kwargs[key], value)

    def test_qa_service_forwards_request_weights_to_retriever(self):
        service = QAService.__new__(QAService)
        service.settings = SimpleNamespace(top_k=5, mock=True)
        service.prompts = Mock()
        service.prompts.expand.return_value = {"sub_queries": ["alpha", "beta"]}
        service.prompts.load_module.return_value = {}
        service.retriever = Mock()
        service.retriever.search.return_value = {"top_chunks": []}
        service.cluster_router = SimpleNamespace(available=False)
        service._last_answer_lock = threading.Lock()
        weights = dict(semantic_weight=0.0, dimension_weight=0.4, lexical_weight=0.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            service.run_dir = Path(temp_dir)
            service.answer("original", top_k=3, **weights)
        service.retriever.search.assert_called_once_with(
            "alpha | beta", top_k=3, original_query="original",
            subqueries=["alpha", "beta"], **weights,
        )


if __name__ == "__main__":
    unittest.main()
