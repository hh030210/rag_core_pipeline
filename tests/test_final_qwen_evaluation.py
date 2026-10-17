import json
from pathlib import Path
import tempfile
import unittest

from rag_core.final_qwen_evaluation import DIMENSIONS, MACHINE_METRICS, evaluate


class FakeProtocol:
    @staticmethod
    def score_answer(answer, reference):
        return {key: float(answer == reference) for key in MACHINE_METRICS}

    @staticmethod
    def parse_generated_answer(answer):
        return answer

    @staticmethod
    def build_llm_judge_prompt(row):
        return json.dumps(row, ensure_ascii=False)

    @staticmethod
    def parse_llm_judge_scores(raw):
        return raw


class FakeQwen:
    mock = False
    model = "Qwen3-8B-test"
    base_url = "http://unit-test-no-network"
    calls = 0

    def complete_json(self, *args, **kwargs):
        self.calls += 1
        return {key: {"score": 5, "reasoning": "test"} for key in DIMENSIONS}


class FinalEvaluationTests(unittest.TestCase):
    def test_mechanical_and_three_qwen_scores_complete_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "answers.jsonl"
            source.write_text(json.dumps({"id": "1", "query": "question", "reference_answer": "answer",
                "generated_answer": "answer", "context_budget": 5, "mock": False}) + "\n", encoding="utf-8")
            client = FakeQwen()
            summary = evaluate(source, root / "eval", client=client, protocol=FakeProtocol,
                               protocol_path=__file__, workers=80)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["machine_metrics"]["f1"], 1.0)
            self.assertEqual(summary["qwen_scores_1_to_5"]["answer_quality"], 5)
            calls = client.calls
            evaluate(source, root / "eval", client=client, protocol=FakeProtocol,
                     protocol_path=__file__, workers=80)
            self.assertEqual(client.calls, calls)
            self.assertTrue((root / "eval/question_scores.csv").exists())

    def test_mock_results_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "answers.jsonl"
            source.write_text(json.dumps({"id": "1", "generated_answer": "mock", "mock": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "mock"):
                evaluate(source, root / "eval", client=FakeQwen(), protocol=FakeProtocol,
                         protocol_path=__file__)


if __name__ == "__main__":
    unittest.main()
