"""Read-only consistency checks for a completed top5/Qwen experiment."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .cached_context import question_key
from .prompt_policy import MODULE_FIELDS, validate_module


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(directory):
    directory = Path(directory)
    train = read(directory / "prepared/train_golden.json")["examples"]
    inference_data = read(directory / "prepared/inference_top10.json")
    cached = {str(row["id"]): row for row in inference_data}
    answers = rows(directory / "answers_top5.jsonl")
    scored = rows(directory / "evaluation_qwen/question_scores.jsonl")
    summary = read(directory / "evaluation_qwen/summary.json")
    manifest = read(directory / "prepared/data_manifest.json")
    artifact = read(directory / "cluster_prompting.json")
    assert len(answers) == len(scored) == len(cached), "Incomplete question coverage"
    assert {str(row["id"]) for row in answers} == set(cached), "Answer ID mismatch"
    assert {str(row["id"]) for row in scored} == set(cached), "Evaluation ID mismatch"
    assert manifest["question_overlap_count"] == 0
    assert not ({question_key(row["question"]) for row in train} &
                {question_key(row["question"]) for row in inference_data})
    for row in answers:
        assert row["context_budget"] == 5 and not row["mock"]
        assert row["context_documents"] == cached[str(row["id"])]["retrieved_top10"][:5]
        assert all(doc["kind"] == "retrieved" for doc in row["context_documents"])
        validate_module(row["prompt_module"])
    for cluster in artifact["clusters"]:
        assert cluster["prompt_method"] == "llm_collective_synthesis"
        validate_module(cluster["prompt"])
    cases = [read(path) for path in (directory / "cluster_prompting/case_iterations").glob("*.json")]
    assert len(cases) == len(train)
    for case in cases:
        assert case["context_kind"] == "golden"
        assert case["method"] == "adaptive_meta_case_optimization"
        assert case["stop_reason"] not in {"in_progress", "invalid_prompt", "mock_only_no_real_optimization"}
        validate_module(case["initial_prompt"])
        validate_module(case)
        selected = next(row for row in case["history"] if row["iteration"] == case["selected_iteration"])
        assert selected["accepted"]
        assert all(case[key] == selected["prompt"][key] for key in MODULE_FIELDS)
    for row in scored:
        assert row["status"] == "ok"
        assert "qwen" in row["judge_model"].casefold()
        assert set(row["qwen_scores"]) == {"answer_quality", "accuracy", "completeness"}
        assert all(1 <= item["score"] <= 5 for item in row["qwen_scores"].values())
    for metric, value in summary["machine_metrics"].items():
        expected = sum(row["machine_metrics"][metric] for row in scored) / len(scored)
        assert math.isclose(value, expected, abs_tol=1e-12)
    assert summary["complete"] and summary["expected_count"] == len(cached)
    assert read(directory / "inference_top5_errors.json") == []
    assert read(directory / "evaluation_qwen/errors.json") == []
    result = {"audit_passed": True, "training_count": len(train), "inference_count": len(answers),
              "scored_count": len(scored), "clusters": len(artifact["clusters"]), "context_budget": 5,
              "training_context": "golden", "inference_context": "retrieved_top10_prefix5",
              "test_training_overlap": 0, "case_stop_reasons": dict(Counter(case["stop_reason"] for case in cases))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
