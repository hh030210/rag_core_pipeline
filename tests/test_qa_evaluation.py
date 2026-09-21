import unittest

from rag_core.qa_evaluation import compute_mechanical_metrics


class QaMechanicalMetricsTests(unittest.TestCase):
    def test_exact_answer_and_evidence_are_scored_deterministically(self):
        row = {
            "answer": "长陵建于1409年，位于北京。",
            "reference_answer": "长陵建于1409年，位于北京。",
            "retrieval_top5": {
                "fusion_top5": [{"chunk_text_full": "长陵建于1409年，位于北京。"}]
            },
        }
        metrics = compute_mechanical_metrics(row)
        self.assertEqual(metrics["normalized_exact_match"], 1)
        self.assertEqual(metrics["char_f1"], 1.0)
        self.assertEqual(metrics["rouge_l_f1"], 1.0)
        self.assertEqual(metrics["numeric_recall"], 1.0)
        self.assertEqual(metrics["missing_reference_numbers"], [])
        self.assertEqual(metrics["extra_answer_numbers"], [])
        self.assertEqual(metrics["answer_evidence_char_precision"], 1.0)
        self.assertEqual(metrics["reference_evidence_char_recall"], 1.0)

    def test_numeric_metrics_expose_missing_and_extra_facts(self):
        row = {
            "answer": "长陵建于1408年。",
            "reference_answer": "长陵建于1409年。",
        }
        metrics = compute_mechanical_metrics(row)
        self.assertEqual(metrics["numeric_recall"], 0.0)
        self.assertEqual(metrics["numeric_f1"], 0.0)
        self.assertEqual(metrics["missing_reference_numbers"], ["1409"])
        self.assertEqual(metrics["extra_answer_numbers"], ["1408"])


if __name__ == "__main__":
    unittest.main()
