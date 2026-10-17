import json
from pathlib import Path
import tempfile
import unittest

from rag_core.adaptive_prompting import adaptive_optimize
from rag_core.cluster_prompting import ClusterPromptPipeline, ClusterPromptRouter
from rag_core.embedding import EmbeddingModel
from rag_core.llm import LLMClient
from rag_core.prompt_policy import MODULE_FIELDS, NOISE_REQUIREMENT, RERANK_REQUIREMENT, validate_module, request_module
from rag_core.prompting import PromptOptimizer
from rag_core.scenic_ablation import prepare_shared


def plain():
    result = {key: "基于给定上下文核验事实并完整回答问题。" for key in MODULE_FIELDS}
    return result


class FakePlainClient:
    api_key = "fake-no-network"
    mock = False
    model = "Qwen-unit-test"

    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        self.calls.append((system, user))
        if "评测器" in system:
            return {key: {"score": 80, "reasoning": "test"} for key in ("answer_quality", "accuracy", "completeness")}
        return plain()

    def complete(self, system, user, **kwargs):
        return "test answer"


class ScenicAblationTests(unittest.TestCase):
    def test_plain_schema_does_not_inject_policy(self):
        self.assertEqual(validate_module(plain(), evidence_policy=False), plain())
        normalized = PromptOptimizer.normalize_module(plain(), evidence_policy=False)
        self.assertNotIn(NOISE_REQUIREMENT, str(normalized))
        self.assertNotIn(RERANK_REQUIREMENT, str(normalized))
        with self.assertRaises(ValueError):
            validate_module(plain(), evidence_policy=True)

    def test_control_rejects_spontaneous_policy_reintroduction(self):
        for phrase in (NOISE_REQUIREMENT, RERANK_REQUIREMENT, "先筛选证据", "对上下文进行重排序", "rerank documents"):
            result = plain()
            result["instruction"] += phrase
            with self.assertRaises(ValueError):
                validate_module(result, evidence_policy=False)

    def test_control_request_does_not_append_canonical_requirements(self):
        client = FakePlainClient()
        result = request_module(client, "meta", "case", evidence_policy=False)
        self.assertEqual(result, plain())
        self.assertNotIn(NOISE_REQUIREMENT, client.calls[0][1])
        self.assertNotIn(RERANK_REQUIREMENT, client.calls[0][1])

    def test_control_adaptive_initial_and_rewrite_stay_plain(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakePlainClient()
            meta = Path(__file__).resolve().parent.parent / "rag_core/prompts/ablation_initial_meta.txt"
            optimizer = PromptOptimizer(client=client, run_dir=directory, evidence_policy=False, meta_prompt_path=meta)
            row = {"question": "q", "context": "gold", "context_kind": "golden", "reference_answer": "a"}
            result = adaptive_optimize(optimizer, [row])
            self.assertFalse(result["evidence_policy"])
            validate_module(result["initial_prompt"], evidence_policy=False)
            validate_module(result, evidence_policy=False)
            for step in result["history"]:
                validate_module(step["prompt"], evidence_policy=False)

    def test_control_collective_synthesis_stays_plain(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakePlainClient()
            pipeline = ClusterPromptPipeline(client=client, embeddings=EmbeddingModel(mock=True),
                run_dir=directory, evidence_policy=False)
            result = pipeline._cluster_prompt(0, [{"question": "q", "final_prompt": plain(), "history": []}])
            validate_module(result["prompt"], evidence_policy=False)
            self.assertNotIn(NOISE_REQUIREMENT, client.calls[0][1])

    def test_router_auto_reads_false_mode_without_legacy_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = EmbeddingModel(mock=True, dimension=8)
            center = embeddings.encode(["q"])[0]
            (root / "cluster_prompting.json").write_text(json.dumps({"config": {"evidence_policy": False},
                "clusters": [{"cluster_id": 0, "center": center, "prompt": plain()}]}), encoding="utf-8")
            router = ClusterPromptRouter(run_dir=root, embeddings=embeddings, enabled=True)
            self.assertFalse(router.evidence_policy)
            validate_module(router.route("q")[0]["prompt"], evidence_policy=False)
            with self.assertRaises(ValueError):
                ClusterPromptRouter(run_dir=root, embeddings=embeddings, enabled=True, evidence_policy=True)

    def test_prepare_excludes_training_and_short_contexts_and_uses_only_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [{"question": "train", "answer": "a", "source": ["gold"]}]
            test = [{"question_id": "a", "question": "test", "answer": "ref"},
                    {"question_id": "b", "question": "short", "answer": "ref2"}]
            hit = {"chunk_id": "doc", "chunk_text": "text", "is_gold": True}
            retrieval = [{"query_id": "1", "question": "train", "routes": {"fusion": [hit] * 10}},
                {"query_id": "2", "question": "test", "routes": {"fusion": [hit] * 5}},
                {"query_id": "3", "question": "short", "routes": {"fusion": [hit] * 4}}]
            for name, data in (("train.json", train), ("eval.json", test), ("retrieval.json", retrieval)):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            training, cases, manifest = prepare_shared(root / "retrieval.json", root / "train.json",
                                                      root / "eval.json", root / "output")
            self.assertEqual(len(training), 1)
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(cases[0]["retrieved_contexts"]), 5)
            self.assertNotIn("is_gold", str(cases[0]["retrieved_contexts"]))
            self.assertEqual(len(manifest["excluded"]), 2)


if __name__ == "__main__":
    unittest.main()
