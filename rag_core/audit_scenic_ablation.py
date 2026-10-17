"""Audit paired coverage, contexts, clustering, policy separation and means."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import unicodedata

from .prompt_policy import MODULE_FIELDS, validate_module


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def audit(directory):
    directory = Path(directory)
    manifest = read(directory / "shared/data_manifest.json")
    tests = read(directory / "shared/test_top5.json")
    training = read(directory / "shared/train_golden.json")["examples"]
    expected = {str(row["id"]): row for row in tests}
    assert not ({key(row["question"]) for row in training} & {key(row["question"]) for row in tests})
    artifacts, configs, answers_by_variant, summaries, reason_counts = {}, {}, {}, {}, {}
    for name, policy in (("with_policy", True), ("without_policy", False)):
        target = directory / name
        artifacts[name] = read(target / "cluster_prompting.json")
        configs[name] = read(target / "training_config.json")
        answers = rows(target / "answers_top5.jsonl")
        scores = rows(target / "evaluation_qwen/question_scores.jsonl")
        summary = read(target / "evaluation_qwen/summary.json")
        summaries[name] = summary
        assert len(answers) == len(scores) == len(expected)
        assert {str(row["id"]) for row in answers} == {str(row["id"]) for row in scores} == set(expected)
        assert summary["complete"] and summary["scored_count"] == len(expected)
        assert read(target / "inference_errors.json") == []
        assert read(target / "evaluation_qwen/errors.json") == []
        answers_by_variant[name] = {str(row["id"]): row for row in answers}
        for row in answers:
            assert not row["mock"] and row["context_budget"] == 5 and row["evidence_policy"] == policy
            assert row["context_documents"] == expected[str(row["id"])]["retrieved_contexts"][:5]
            assert row["reference_answer"] == expected[str(row["id"])]["reference_answer"]
            assert all(doc["kind"] == "retrieved" and "is_gold" not in doc for doc in row["context_documents"])
            validate_module(row["prompt_module"], evidence_policy=policy)
        cases = [read(path) for path in (target / "cluster_prompting/case_iterations").glob("*.json")]
        assert len(cases) == len(training)
        for case in cases:
            assert case["evidence_policy"] == policy and case["context_kind"] == "golden"
            assert case["stop_reason"] not in {"in_progress", "invalid_prompt", "mock_only_no_real_optimization"}
            validate_module(case["initial_prompt"], evidence_policy=policy)
            validate_module(case, evidence_policy=policy)
            for step in case["history"]:
                if "prompt" in step:
                    validate_module(step["prompt"], evidence_policy=policy)
            chosen = next(step for step in case["history"] if step["iteration"] == case["selected_iteration"])
            assert chosen["accepted"]
            assert all(case[field] == chosen["prompt"][field] for field in MODULE_FIELDS)
        reason_counts[name] = dict(Counter(case["stop_reason"] for case in cases))
        for cluster in artifacts[name]["clusters"]:
            assert cluster["prompt_method"] == "llm_collective_synthesis"
            validate_module(cluster["prompt"], evidence_policy=policy)
        for row in scores:
            assert row["status"] == "ok" and "qwen" in row["judge_model"].casefold()
            assert all(1 <= item["score"] <= 5 for item in row["qwen_scores"].values())
        for metric, value in summary["machine_metrics"].items():
            mean = sum(row["machine_metrics"][metric] for row in scores) / len(scores)
            assert math.isclose(mean, value, abs_tol=1e-12)
        for metric, value in summary["qwen_scores_1_to_5"].items():
            mean = sum(row["qwen_scores"][metric]["score"] for row in scores) / len(scores)
            assert math.isclose(mean, value, abs_tol=1e-12)
    a, b = (artifacts[name] for name in ("with_policy", "without_policy"))
    assert a["question_cluster_mapping"] == b["question_cluster_mapping"]
    assert [cluster["center"] for cluster in a["clusters"]] == [cluster["center"] for cluster in b["clusters"]]
    for case_id in expected:
        left, right = (answers_by_variant[name][case_id] for name in ("with_policy", "without_policy"))
        assert left["context_documents"] == right["context_documents"]
        assert left["cluster_id"] == right["cluster_id"]
    common_left = {k: v for k, v in configs["with_policy"].items() if k != "evidence_policy"}
    common_right = {k: v for k, v in configs["without_policy"].items() if k != "evidence_policy"}
    assert common_left == common_right
    evaluation_configs = [read(directory / name / "evaluation_qwen/evaluation_config.json")
                          for name in ("with_policy", "without_policy")]
    assert evaluation_configs[0]["protocol_sha256"] == evaluation_configs[1]["protocol_sha256"]
    assert evaluation_configs[0]["judge_model"] == evaluation_configs[1]["judge_model"]
    result = {"audit_passed": True, "source_count": manifest["source_question_count"],
              "test_count_per_variant": len(expected), "golden_training_count": len(training),
              "same_contexts": True, "same_references": True, "same_qa_clusters_and_routes": True,
              "same_meta_template_and_hyperparameters": True, "same_evaluation_protocol": True,
              "policy_separation_validated_at_all_prompt_stages": True, "case_stop_reasons": reason_counts}
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
