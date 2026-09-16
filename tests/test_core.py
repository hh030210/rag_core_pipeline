import unittest
from pathlib import Path

from rag_core.dimension_stage import default_schema, build_inverted_index, build_tags
from rag_core.embedding import EmbeddingModel
from rag_core.prompting import PromptOptimizer
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


if __name__ == "__main__":
    unittest.main()
