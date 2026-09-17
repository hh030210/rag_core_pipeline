import json
import tempfile
import unittest
from pathlib import Path

from rag_core.dimension_stage import default_schema, build_inverted_index, build_tags
from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import evaluate_row, load_gold_records, map_gold_to_chunks, route_metrics
from rag_core.prompting import PromptOptimizer
from rag_core.prompt_module import PromptOptimizationModule
from rag_core.settings import Settings
from rag_core.storage import make_payload
from rag_core.schema_v2 import load_schema
from rag_core.llm import LLMClient


class CorePipelineTests(unittest.TestCase):
    def test_schema_is_two_level_and_valid(self):
        schema = load_schema(default_schema(), allow_legacy=False, strict=True)
        self.assertTrue(all(node["level"] in (1, 2) for node in schema["dimensions"]))
        self.assertTrue(all(node["indexable"] is (node["level"] == 2) for node in schema["dimensions"]))

    def test_multilabel_tags_keep_arrays(self):
        schema = default_schema()
        records = [{"doc_id": "d1", "doc_text": "儿童和青少年可在北京景点参观，上午开放，门票免费。"}]
        settings = Settings(run_dir=Path("/tmp/rag_core_test"), backend="local", mock=True)
        output = build_tags(records, schema, settings)
        tags = output["documents"]["d1"]["tags"]
        self.assertIn("儿童", tags["content.entity"])
        self.assertIn("青少年", tags["content.entity"])
        self.assertIsInstance(tags["content.entity"], list)
        index = build_inverted_index(output, schema)
        self.assertIn("儿童", index["postings"]["content.entity"])
        payload = make_payload({"chunk_id": "d1", "doc_id": "d1", "doc_text": records[0]["doc_text"]},
                               output["documents"]["d1"], schema)
        self.assertIsInstance(payload["dim_content.entity"], list)

    def test_mock_embedding_is_deterministic(self):
        model = EmbeddingModel(dimension=8, mock=True)
        self.assertEqual(model.encode(["a"])[0], model.encode(["a"])[0])

    def test_prompt_expansion_falls_back_without_llm(self):
        manager = PromptOptimizer(client=LLMClient(mock=True), run_dir=Path("/tmp/rag_core_test"))
        result = manager.expand("测试问题")
        self.assertEqual(result["sub_queries"], ["测试问题"])
        self.assertTrue(manager.load_module()["system_prompt"])

    def test_prompt_module_has_explicit_json_input_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "prompt_input.json"
            output_path = root / "prompt_output.json"
            input_path.write_text(json.dumps({
                "examples": [{
                    "question": "景区几点开放？",
                    "context": "景区每日八点开放。",
                    "reference_answer": "景区每日八点开放。",
                }],
                "config": {"iterations": 2},
            }, ensure_ascii=False), encoding="utf-8")
            module = PromptOptimizationModule(
                client=LLMClient(mock=True),
                work_dir=root / "work",
            )
            result = module.run_file(input_path, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["module_version"], "1.0")
            self.assertEqual(result["io"]["example_count"], 1)
            self.assertEqual(result["io"]["iterations"], 2)
            self.assertEqual(saved["method"], "no_examples_or_mock")
            self.assertTrue(saved["system_prompt"])

    def test_route_metrics_report_first_gold_and_cutoffs(self):
        result = route_metrics(
            [{"chunk_id": "noise"}, {"chunk_id": "gold"}, {"chunk_id": "other"}],
            ["gold"],
            ks=(1, 3),
        )
        self.assertEqual(result["first_gold_rank"], 2)
        self.assertEqual(result["rr"], 0.5)
        self.assertFalse(result["hit_at_1"])
        self.assertTrue(result["hit_at_3"])
        self.assertEqual(result["rr_at_1"], 0.0)
        self.assertEqual(result["rr_at_3"], 0.5)

    def test_gold_aliases_and_evidence_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold.json"
            path.write_text(json.dumps([{
                "id": "q1",
                "query": "几点开放？",
                "source": ["景区每日八点开放。"],
                "attraction": "测试景区",
            }], ensure_ascii=False), encoding="utf-8")
            record = load_gold_records(path)[0]
            self.assertEqual(record["question"], "几点开放？")
            self.assertEqual(record["spot"], "测试景区")
            mapped = map_gold_to_chunks(record, [{
                "chunk_id": "c1",
                "source_file": "测试景区",
                "chunk_text_full": "介绍：景区每日八点开放。",
            }])
            self.assertEqual(mapped["gold_chunk_ids"], ["c1"])
            self.assertEqual(mapped["gold_resolution"]["match_basis"], "evidence_text")

    def test_evaluation_keeps_fusion_metrics_and_diagnostics_separate(self):
        row = evaluate_row(
            {"id": "q1", "question": "测试", "gold_chunk_ids": ["gold"]},
            {
                "semantic_candidates": [{"chunk_id": "noise"}, {"chunk_id": "gold"}],
                "dimension_candidates": [{"chunk_id": "gold"}],
                "fusion_candidates": [{"chunk_id": "noise"}, {"chunk_id": "gold"}],
            },
            ks=(1, 2),
            eval_depth=2,
        )
        self.assertEqual(row["retrieval_metrics"]["fusion"]["first_gold_rank"], 2)
        self.assertIn("fusion_diagnostics", row["retrieval_metrics"])
        self.assertEqual(row["retrieval_metrics"]["fusion_diagnostics"]["rank_gain"], -1)


if __name__ == "__main__":
    unittest.main()
