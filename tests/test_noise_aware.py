import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_core.adaptive_prompting import adaptive_optimize, parse_judgement, stopping_reason
from rag_core.cached_context import prepare_data, render_context, select_contexts
from rag_core.cluster_prompting import ClusterPromptPipeline, normalize_examples
from rag_core.case_collective_experiment import run_inference
from rag_core.embedding import EmbeddingModel
from rag_core.llm import LLMClient
from rag_core.prompt_policy import MODULE_FIELDS, POLICY_TEXT, validate_module
from rag_core.prompting import PromptOptimizer
from rag_core.settings import Settings


def module(name="generated"):
    result = {key: name + "_" + key for key in MODULE_FIELDS}
    result["context_strategy"] += "\n" + POLICY_TEXT
    return result


def judgement(a=80, b=80, c=80):
    return {key: {"score": value, "reasoning": "test reason"}
            for key, value in zip(("answer_quality", "accuracy", "completeness"), (a, b, c))}


class FakeClient:
    mock = False
    api_key = "fake_no_network"
    model = "unit_test_fake"

    def __init__(self, judges, modules=None):
        self.judges = iter(judges)
        self.modules = iter(modules) if modules is not None else None
        self.json_calls = []
        self.answer_calls = []
        self.module_index = 0

    def complete_json(self, system, user, **kwargs):
        self.json_calls.append((system, user))
        if "评测器" in system:
            return next(self.judges)
        self.module_index += 1
        return next(self.modules) if self.modules is not None else module(f"generated_{self.module_index}")

    def complete(self, system, user, **kwargs):
        self.answer_calls.append((system, user))
        return f"candidate_{len(self.answer_calls)}"


class NoiseAwareTests(unittest.TestCase):
    def example(self):
        return {"id": "case1", "question": "test question", "context": "golden evidence FULL",
                "context_kind": "golden", "reference_answer": "SECRET_REFERENCE_FOR_JUDGE_ONLY"}

    def optimize(self, client, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = PromptOptimizer(client=client, run_dir=directory)
            return adaptive_optimize(optimizer, [self.example()], **kwargs)

    def test_dimension_decrease_rolls_back_even_if_total_increases(self):
        result = self.optimize(FakeClient([judgement(), judgement(95, 79, 95)]))
        self.assertEqual(result["selected_iteration"], 0)
        self.assertEqual(result["system_prompt"], "generated_1_system_prompt")
        self.assertIn("accuracy", result["stop_reason"])
        self.assertFalse(result["history"][-1]["accepted"])

    def test_small_gain_rolls_back(self):
        result = self.optimize(FakeClient([judgement(), judgement(80.5, 80.5, 80.5)]))
        self.assertEqual(result["stop_reason"], "total_gain_below_threshold")
        self.assertEqual(result["selected_iteration"], 0)

    def test_exact_threshold_accepted_and_next_drop_keeps_previous(self):
        result = self.optimize(FakeClient([judgement(), judgement(81, 81, 81), judgement(80, 80, 80)]))
        self.assertEqual(result["selected_iteration"], 1)
        self.assertEqual(result["system_prompt"], "generated_2_system_prompt")
        self.assertEqual(result["selected_answers"][0]["answer"], "candidate_2")

    def test_safety_cap(self):
        result = self.optimize(FakeClient([judgement(), judgement(82, 82, 82)]), max_rounds=1)
        self.assertEqual(result["stop_reason"], "max_rounds")
        self.assertEqual(result["selected_iteration"], 1)

    def test_no_reference_in_meta_or_answer_input(self):
        client = FakeClient([judgement(), judgement()])
        self.optimize(client)
        self.assertNotIn("SECRET_REFERENCE_FOR_JUDGE_ONLY", client.json_calls[0][1])
        self.assertIn("golden evidence FULL", client.json_calls[0][1])
        for _, user in client.answer_calls:
            self.assertNotIn("SECRET_REFERENCE_FOR_JUDGE_ONLY", user)
            self.assertIn(POLICY_TEXT, user)
            self.assertIn("golden evidence FULL", user)

    def test_no_silent_6000_character_truncation(self):
        client = FakeClient([judgement(), judgement()])
        with tempfile.TemporaryDirectory() as directory:
            example = self.example()
            example["context"] = "X" * 6500 + "GOLD_END"
            adaptive_optimize(PromptOptimizer(client=client, run_dir=directory), [example])
        self.assertIn("GOLD_END", client.answer_calls[0][1])

    def test_retrieval_context_cannot_be_training_golden(self):
        normalized = normalize_examples([{"question": "q", "retrieved_context": "NOT_GOLD", "answer": "a"}])[0]
        self.assertEqual(normalized["context"], "")
        self.assertEqual(normalized["context_kind"], "missing_golden")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "golden"):
                adaptive_optimize(PromptOptimizer(client=FakeClient([]), run_dir=directory), [normalized])

    def test_golden_preferred_over_plain_context(self):
        normalized = normalize_examples([{"question": "q", "context": "noise", "source": ["gold"], "answer": "a"}])[0]
        self.assertEqual(normalized["context"], "gold")

    def test_missing_policy_rejected_and_previous_kept(self):
        invalid = module("invalid")
        invalid["context_strategy"] = "rerank only, no immutable policies"
        result = self.optimize(FakeClient([judgement()], modules=[module("initial"), invalid, invalid]))
        self.assertEqual(result["stop_reason"], "invalid_prompt")
        self.assertEqual(result["system_prompt"], "initial_system_prompt")

    def test_initial_generation_cannot_fall_back_to_default(self):
        invalid = {key: "no evidence policy" for key in MODULE_FIELDS}
        with self.assertRaisesRegex(ValueError, "validation"):
            self.optimize(FakeClient([], modules=[invalid, invalid]))

    def test_batch_cannot_silently_share_one_initial_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = PromptOptimizer(client=FakeClient([]), run_dir=directory)
            with self.assertRaisesRegex(ValueError, "exactly one case"):
                adaptive_optimize(optimizer, [self.example(), {**self.example(), "id": "other"}])

    def test_invalid_judgement_is_not_a_zero_score(self):
        for invalid in ({"score": 100}, judgement(float("nan")), judgement(101)):
            with self.assertRaises((ValueError, KeyError)):
                parse_judgement(invalid)

    def test_select_top5_is_exact_prefix_without_golden(self):
        documents = [{"kind": "retrieved", "text": f"retrieval {i}"} for i in range(10)]
        selected = select_contexts({"retrieved_top10": documents, "gold_context": "SECRET_GOLD"}, 5)
        self.assertEqual(selected, documents[:5])
        self.assertEqual(len(selected), 5)
        self.assertNotIn("SECRET_GOLD", render_context(selected))
        self.assertNotIn("retrieval 5", render_context(selected))

    def test_invalid_context_budget(self):
        for budget in (0, 11):
            with self.assertRaises(ValueError):
                select_contexts({"retrieved_top10": []}, budget)

    def test_prepare_excludes_training_overlap_and_oracle_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [{"question": "training", "answer": "answer", "source": ["gold"]},
                     {"question": " TＥＳＴ ? ", "answer": "test answer", "source": ["leak"]}]
            evaluation = [{"id": "test1", "question": "test", "answer": "reference"}]
            hits = [{"rank": i, "chunk_id": str(i), "chunk_text": f"text {i}", "is_gold": True,
                     "text_truncated": False} for i in range(1, 16)]
            retrieval = [{"id": "r1", "question": "test", "gold": {"evidence_texts": ["SECRET"]},
                          "retrieval": {"fusion_results": hits}}]
            for name, data in (("train.json", train), ("eval.json", evaluation), ("retrieval.json", retrieval)):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            result = prepare_data(root / "train.json", root / "eval.json", root / "retrieval.json", root / "out")
            self.assertEqual(result["training_count"], 1)
            self.assertEqual(result["question_overlap_count"], 0)
            self.assertEqual(result["excluded_training"][0]["reason"], "overlaps_evaluation")
            output = json.loads((root / "out/inference_top10.json").read_text())
            self.assertEqual(len(output[0]["retrieved_top10"]), 10)
            self.assertNotIn("is_gold", output[0]["retrieved_top10"][0])
            self.assertNotIn("SECRET", json.dumps(output))

    def test_cluster_synthesis_contains_both_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([])
            pipeline = ClusterPromptPipeline(client=client, embeddings=EmbeddingModel(mock=True), run_dir=directory)
            result = pipeline._cluster_prompt(0, [{"question": "q", "final_prompt": module(), "history": []}])
            validate_module(result["prompt"])
            self.assertIn(POLICY_TEXT, client.json_calls[0][1])
            self.assertEqual(result["method"], "llm_collective_synthesis")

    def test_five_modules_generated_separately_for_each_case(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([judgement(), judgement(), judgement(), judgement()])
            pipeline = ClusterPromptPipeline(client=client, embeddings=EmbeddingModel(mock=True), run_dir=directory)
            a = pipeline._case_iteration(self.example(), 30)
            bcase = {**self.example(), "id": "case2", "question": "other question"}
            b = pipeline._case_iteration(bcase, 30)
            self.assertNotEqual(a["initial_prompt"]["system_prompt"], b["initial_prompt"]["system_prompt"])

    def test_completed_case_resumes_without_new_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([judgement(), judgement()])
            pipeline = ClusterPromptPipeline(client=client, embeddings=EmbeddingModel(mock=True), run_dir=directory)
            first = pipeline._case_iteration(self.example(), 30)
            calls = len(client.json_calls)
            second = pipeline._case_iteration(self.example(), 30)
            self.assertEqual(len(client.json_calls), calls)
            self.assertEqual(first["final_prompt"], second["final_prompt"])

    def test_mock_end_to_end_is_separate_and_real_rejects_fallback(self):
        with tempfile.TemporaryDirectory(prefix="rag_mock_") as directory:
            root = Path(directory)
            examples = [{**self.example(), "source": ["golden evidence FULL"]}]
            ClusterPromptPipeline(client=LLMClient(mock=True), embeddings=EmbeddingModel(mock=True), run_dir=root).run(
                examples, cluster_count=1)
            data = [{"id": "test", "question": "q", "reference_answer": "SECRET_REFERENCE",
                     "retrieved_top10": [{"kind": "retrieved", "text": f"context {i}"} for i in range(10)]}]
            path = root / "data.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            settings = Settings(run_dir=root, mock=True)
            result = run_inference(settings, path, budget=5, workers=1)
            self.assertEqual(result["count"], 1)
            answer = json.loads(Path(result["output"]).read_text(encoding="utf-8").strip())
            self.assertEqual(len(answer["context_documents"]), 5)
            self.assertNotIn("SECRET_REFERENCE", answer["full_user_prompt"])
            self.assertIn("MOCK_ONLY", result["output"])
            # Same command resumes rather than appends duplicate rows.
            run_inference(settings, path, budget=5, workers=1)
            self.assertEqual(len(Path(result["output"]).read_text(encoding="utf-8").splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "mock/fallback"):
                run_inference(Settings(run_dir=root, mock=False), path, budget=5)

    def test_llm_payload_disables_qwen_thinking(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}).encode()
        client = LLMClient(api_key="fake", base_url="http://localhost", model="Qwen3-8B")
        with patch("urllib.request.urlopen", return_value=Response()) as mocked:
            self.assertEqual(client.complete("system", "user"), "answer")
            payload = json.loads(mocked.call_args.args[0].data)
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertNotIn("extra_body", payload)


if __name__ == "__main__":
    unittest.main()
