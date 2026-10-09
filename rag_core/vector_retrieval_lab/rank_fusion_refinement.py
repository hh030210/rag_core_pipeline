"""Offline two-search rank-fusion sweep over saved one-search candidate rankings."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

from .experiment import DEFAULT_RUN, LAB_DIR, load_jsonl, score_metrics

STRATEGIES = [
    "current_question_weight_2_centroid",
    "fixed_question_mass_33",
    "fixed_question_mass_40",
    "fixed_question_mass_50",
    "fixed_question_mass_60",
]


def fuse_rrf(left: Sequence[str], right: Sequence[str], k: int, ratio: float) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    first: Dict[str, int] = {}
    order = 0
    for ranking, weight in ((left, ratio), (right, 1.0)):
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id not in first:
                first[chunk_id] = order
                order += 1
            scores[chunk_id] += weight / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], first[item]))


def fuse_borda(left: Sequence[str], right: Sequence[str], ratio: float) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    first: Dict[str, int] = {}
    order = 0
    depth = max(len(left), len(right), 1)
    for ranking, weight in ((left, ratio), (right, 1.0)):
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id not in first:
                first[chunk_id] = order
                order += 1
            scores[chunk_id] += weight * (depth - rank + 1) / depth
    return sorted(scores, key=lambda item: (-scores[item], first[item]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--variant-rankings", type=Path,
                        default=LAB_DIR / "outputs/centroid_variants_full_20261008/per_query_rankings.jsonl")
    parser.add_argument("--input-depth", type=int, default=20)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    records = load_jsonl(args.variant_rankings)
    if len(rows) != len(records):
        raise RuntimeError(f"mapped rows={len(rows)} != ranking rows={len(records)}")
    output = args.output_dir or LAB_DIR / "outputs" / ("rank_fusion_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    results: Dict[str, List[List[str]]] = defaultdict(list)
    for record in records:
        for name in args.strategies:
            results[name].append(record["rankings"][name][:20])

    pairs = list(combinations(args.strategies, 2))
    for left_name, right_name in pairs:
        for k in (0, 5, 20, 60):
            for ratio in (0.5, 1.0, 2.0):
                name = f"dual_{left_name}__{right_name}_rrf_k{k}_left{ratio:g}"
                for record in records:
                    ranking = fuse_rrf(record["rankings"][left_name][:args.input_depth],
                                       record["rankings"][right_name][:args.input_depth], k, ratio)
                    results[name].append(ranking[:20])
        for ratio in (0.5, 1.0, 2.0):
            name = f"dual_{left_name}__{right_name}_borda_left{ratio:g}"
            for record in records:
                ranking = fuse_borda(record["rankings"][left_name][:args.input_depth],
                                      record["rankings"][right_name][:args.input_depth], ratio)
                results[name].append(ranking[:20])

    metrics = {name: score_metrics(rows, rankings) for name, rankings in results.items()}
    folds = [int(hashlib.sha256(str(row.get("retrieval_query") or row.get("question") or "").encode("utf-8")).hexdigest()[:8], 16) % 5
             for row in rows]
    candidates = [name for name in results if name.startswith("dual_")]
    predictions: List[List[str] | None] = [None] * len(rows)
    selected: Dict[str, str] = {}
    for fold in range(5):
        train = [i for i, assigned in enumerate(folds) if assigned != fold]
        test = [i for i, assigned in enumerate(folds) if assigned == fold]
        train_metrics = {
            name: score_metrics([rows[i] for i in train], [results[name][i] for i in train])
            for name in candidates
        }
        chosen = max(candidates, key=lambda name: (
            train_metrics[name]["hit@15"], train_metrics[name]["mrr@10"], train_metrics[name]["ndcg@5"]))
        selected[str(fold)] = chosen
        for i in test:
            predictions[i] = results[chosen][i]
    cv_metrics = score_metrics(rows, [ranking or [] for ranking in predictions])
    ordered = sorted(candidates, key=lambda name: (
        metrics[name]["hit@15"], metrics[name]["mrr@10"], metrics[name]["ndcg@5"]), reverse=True)
    summary = {
        "experiment": "offline_two_search_rank_fusion_refinement",
        "mapped_queries": len(rows), "input_rank_depth": args.input_depth,
        "candidate_pairs": [list(pair) for pair in pairs],
        "fusion_methods": ["RRF k=0,5,20,60 with left/right weight ratios 0.5,1,2", "weighted Borda with ratios 0.5,1,2"],
        "strategy_metrics": metrics, "ranking_by_hit15_mrr_ndcg": ordered,
        "best_full_set_strategy": ordered[0],
        "grouped_5fold_cv_selected_strategy_by_fold": selected,
        "grouped_5fold_cv_metrics": cv_metrics,
        "fold_rule": "SHA256(retrieval_query) modulo five",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({
                "id": row.get("id"), "retrieval_query": row.get("retrieval_query"),
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "rankings": {name: ranking[i] for name, ranking in results.items()},
                "cv_selected_ranking": predictions[i],
            }, ensure_ascii=False) + "\n")

    lines = [
        "# Offline two-search rank-fusion refinement", "",
        f"- Queries: {len(rows)}; fused inputs contain the top {args.input_depth} candidates from each retrieval.",
        f"- Grouped 5-fold selected metrics: Hit@15 {cv_metrics['hit@15']:.2%}; MRR@10 {cv_metrics['mrr@10']:.4f}; nDCG@5 {cv_metrics['ndcg@5']:.4f}",
        f"- CV selected by fold: {selected}", "",
        "| Searches | Strategy | Hit@15 | MRR@10 | nDCG@5 |",
        "|---:|---|---:|---:|---:|",
    ]
    for name in ordered[:20]:
        metric = metrics[name]
        lines.append(f"| 2 | {name} | {metric['hit@15']:.2%} | {metric['mrr@10']:.4f} | {metric['ndcg@5']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rank-fusion] results={output}")
    print((output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
