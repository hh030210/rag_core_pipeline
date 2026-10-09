"""Validate a simple subquery-length gate between one-search centroid variants."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .experiment import DEFAULT_RUN, LAB_DIR, load_jsonl, query_parts, score_metrics, unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--variant-rankings", type=Path,
                        default=LAB_DIR / "outputs/centroid_variants_refined_20261008/per_query_rankings.jsonl")
    parser.add_argument("--max-threshold", type=float, default=None,
                        help="Optionally cap the training-fold threshold candidates.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    records = load_jsonl(args.variant_rankings)
    if len(rows) != len(records):
        raise RuntimeError(f"mapped rows={len(rows)} != ranking rows={len(records)}")
    output = args.output_dir or LAB_DIR / "outputs" / ("char_gate_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    features = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        parts = [part for part in unique(query_parts(row)) if part != question]
        features.append(float(sum(len(part) for part in parts)))

    strategy_names = ["current_question_weight_2_centroid", "fixed_question_mass_32",
                      "fixed_question_mass_33", "fixed_question_mass_50", "fixed_question_mass_60"]
    rankings = {name: [record["rankings"][name][:20] for record in records] for name in strategy_names}
    fixed_metrics = {name: score_metrics(rows, value) for name, value in rankings.items()}

    folds = [int(hashlib.sha256(str(row.get("retrieval_query") or row.get("question") or "").encode("utf-8")).hexdigest()[:8], 16) % 5
             for row in rows]
    predictions: List[List[str] | None] = [None] * len(rows)
    selected_by_fold: Dict[str, Dict[str, Any]] = {}
    policy_name_by_fold: Dict[str, str] = {}
    cv_thresholds: Dict[str, List[float]] = {}

    for fold in range(5):
        train = [i for i, assigned in enumerate(folds) if assigned != fold]
        test = [i for i, assigned in enumerate(folds) if assigned == fold]
        candidates: List[Dict[str, Any]] = [
            {"type": "fixed", "strategy": name} for name in strategy_names
        ]
        train_values = sorted(set(features[i] for i in train))
        thresholds = [(left + right) / 2.0 for left, right in zip(train_values, train_values[1:])]
        if args.max_threshold is not None:
            thresholds = [value for value in thresholds if value <= args.max_threshold]
        cv_thresholds[str(fold)] = thresholds
        for threshold in thresholds:
            for low_strategy, high_strategy in (("fixed_question_mass_60", "fixed_question_mass_32"),
                                                ("fixed_question_mass_32", "fixed_question_mass_60")):
                candidates.append({
                    "type": "gate", "threshold": threshold,
                    "low_strategy": low_strategy, "high_strategy": high_strategy,
                })

        def strategy_for(policy: Dict[str, Any], index: int) -> str:
            if policy["type"] == "fixed":
                return policy["strategy"]
            return policy["low_strategy"] if features[index] <= policy["threshold"] else policy["high_strategy"]

        def objective(policy: Dict[str, Any]) -> Tuple[int, int, float, float]:
            hit15 = hit5 = mrr = ndcg = 0.0
            for index in train:
                ranking = rankings[strategy_for(policy, index)][index]
                gold = set((rows[index].get("gold") or {}).get("chunk_ids") or [])
                first = next((rank for rank, chunk_id in enumerate(ranking, 1) if chunk_id in gold), None)
                hit15 += int(first is not None and first <= 15)
                hit5 += int(first is not None and first <= 5)
                mrr += 1.0 / first if first is not None and first <= 10 else 0.0
                dcg = sum(1.0 / __import__("math").log2(rank + 1)
                          for rank, chunk_id in enumerate(ranking[:5], 1) if chunk_id in gold)
                ideal = sum(1.0 / __import__("math").log2(rank + 1)
                            for rank in range(1, min(5, len(gold)) + 1))
                ndcg += dcg / ideal if ideal else 0.0
            return int(hit15), int(hit5), mrr, ndcg

        chosen = max(candidates, key=objective)
        name = (chosen["strategy"] if chosen["type"] == "fixed" else
                f"q60_if_subquery_chars<={chosen['threshold']:.1f}_else_q32"
                if chosen["low_strategy"] == "fixed_question_mass_60" else
                f"q32_if_subquery_chars<={chosen['threshold']:.1f}_else_q60")
        selected_by_fold[str(fold)] = chosen
        policy_name_by_fold[str(fold)] = name
        for index in test:
            predictions[index] = rankings[strategy_for(chosen, index)][index]
        print(f"[char-gate] fold={fold} selected={name}", flush=True)

    cv_metrics = score_metrics(rows, [ranking or [] for ranking in predictions])
    summary = {
        "experiment": "single_search_subquery_length_gate",
        "mapped_queries": len(rows), "feature": "sum of characters across unique subqueries, excluding exact copies of the original question",
        "grouped_5fold_cv_selected_policy_by_fold": selected_by_fold,
        "grouped_5fold_cv_policy_name_by_fold": policy_name_by_fold,
        "train_threshold_candidates_by_fold": cv_thresholds,
        "grouped_5fold_cv_metrics": cv_metrics,
        "fixed_strategy_metrics": fixed_metrics,
        "fold_rule": "SHA256(retrieval_query) modulo five",
        "search_cost": "The gate selects one centroid vector before retrieval; one Qdrant search per query.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                "id": row.get("id"), "retrieval_query": row.get("retrieval_query"),
                "subquery_char_count": features[index],
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "cv_selected_ranking": predictions[index],
            }, ensure_ascii=False) + "\n")
    print(f"[char-gate] results={output}")
    print(json.dumps({"cv_metrics": cv_metrics, "policies": policy_name_by_fold}, ensure_ascii=False))


if __name__ == "__main__":
    main()
