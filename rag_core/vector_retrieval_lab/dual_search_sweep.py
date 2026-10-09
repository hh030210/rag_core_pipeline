"""Find high-quality strategies with at most two Qdrant searches per query."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from .experiment import (DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, filtered_hits,
                         ids, load_jsonl, normalized_score_fusion, query_parts,
                         reciprocal_rank_fusion, score_metrics, spot_names,
                         unique, vector_centroid)
from .vector_search import DenseVectorSearch, l2_normalize


def cosine(left: List[float], right: List[float]) -> float:
    a, b = l2_normalize(left), l2_normalize(right)
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--saved-rankings", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    saved_path = args.saved_rankings or LAB_DIR / "outputs/full_20261008/per_query_rankings.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    saved = load_jsonl(saved_path)
    if len(saved) != len(rows):
        raise RuntimeError(f"Saved rankings count {len(saved)} != mapped baseline rows {len(rows)}")
    if args.limit:
        rows, saved = rows[:args.limit], saved[:args.limit]
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir or LAB_DIR / "outputs" / ("dual_search_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    data, all_texts = [], []
    for row in rows:
        question = str(row.get("question") or "").strip()
        joined = str(row.get("retrieval_query") or question).strip()
        parts = query_parts(row)
        texts = unique([question, joined, *parts])
        data.append({"question": question, "joined": joined, "parts": parts,
                     "texts": texts, "spots": spot_names(row)})
        all_texts.extend(texts)
    all_texts = unique(all_texts)
    print(f"[dual-search] queries={len(rows)} unique_texts={len(all_texts)} pool={args.pool}", flush=True)
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(all_texts)
    vectors = {text: vector for text, vector in zip(all_texts, encoded)}
    raw_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    results: Dict[str, List[List[str]]] = defaultdict(list)
    for i, (row, query) in enumerate(zip(rows, data), 1):
        previous = saved[i - 1]["rankings"]
        baseline_ids = previous["saved_baseline"][:args.depth]
        centroid_ids = previous["question_plus_subquery_vector_centroid"][:args.depth]
        results["saved_baseline_one_search"].append(baseline_ids)
        results["centroid_one_search"].append(centroid_ids)
        spots = query["spots"]
        question, joined, parts = query["question"], query["joined"], query["parts"]
        center_texts = unique([question, *parts])
        center_weights = [2.0 if text == question else 1.0 for text in center_texts]
        center_vector = vector_centroid(center_texts, center_weights, vectors)
        center50 = searcher.search_vector(center_vector, args.pool)
        center50 = [hit for hit in center50 if searcher.matches_spot(hit["payload"], spots)]
        center50_ids = ids(center50, args.depth)
        center_name = f"centroid_pool{args.pool}_one_search"
        results[center_name].append(center50_ids)

        if parts:
            novel = min(parts, key=lambda part: cosine(vectors[part], center_vector))
        else:
            novel = question
        secondary_by_label = {"joined": joined, "question": question, "novel": novel}
        secondary = unique(list(secondary_by_label.values()))
        secondary_lists = {
            text: filtered_hits(searcher, text, args.pool, spots, vectors, raw_cache)
            for text in secondary
        }
        # Keep a ranking slot for every named strategy on every query, even when
        # two labels happen to resolve to the same text.
        for label, text in secondary_by_label.items():
            single_name = f"single_{label}_pool{args.pool}"
            results[single_name].append(ids(secondary_lists[text], args.depth))

        for label, second in secondary_by_label.items():
            right = secondary_lists[second]
            fusions = {
                "rrf_k20_c2s1": reciprocal_rank_fusion([center50, right], [2.0, 1.0], 20),
                "rrf_k60_c2s1": reciprocal_rank_fusion([center50, right], [2.0, 1.0], 60),
                "rrf_k20_equal": reciprocal_rank_fusion([center50, right], [1.0, 1.0], 20),
                "rrf_k60_equal": reciprocal_rank_fusion([center50, right], [1.0, 1.0], 60),
                "rrf_k60_s2": reciprocal_rank_fusion([center50, right], [1.0, 2.0], 60),
                "minmax_equal": normalized_score_fusion([center50, right]),
            }
            for fusion_name, fused in fusions.items():
                name = f"dual_centroid_{label}_pool{args.pool}_{fusion_name}"
                results[name].append(ids(fused, args.depth))
        if i % 50 == 0 or i == len(rows):
            print(f"[dual-search] queries={i}/{len(rows)} cached_text_searches={len(raw_cache)}", flush=True)

    metrics = {name: score_metrics(rows, ranked) for name, ranked in results.items()}
    # Group identical final retrieval queries so duplicates cannot cross folds.
    folds = [int(hashlib.sha256(str(row.get("retrieval_query") or row.get("question") or "").encode("utf-8")).hexdigest()[:8], 16) % 5 for row in rows]
    predictions: List[List[str] | None] = [None] * len(rows)
    selected_by_fold: Dict[str, str] = {}
    for fold in range(5):
        train = [i for i, assigned in enumerate(folds) if assigned != fold]
        test = [i for i, assigned in enumerate(folds) if assigned == fold]
        train_metrics = {name: score_metrics([rows[i] for i in train], [results[name][i] for i in train]) for name in results}
        chosen = max(results, key=lambda name: (
            train_metrics[name].get("hit@15", 0.0), train_metrics[name].get("mrr@10", 0.0),
            train_metrics[name].get("ndcg@5", 0.0)))
        selected_by_fold[str(fold)] = chosen
        for i in test:
            predictions[i] = results[chosen][i]
    cv_metrics = score_metrics(rows, [ranked or [] for ranked in predictions])
    ordered = sorted(metrics, key=lambda name: (
        metrics[name].get("hit@15", 0.0), metrics[name].get("mrr@10", 0.0),
        metrics[name].get("ndcg@5", 0.0)), reverse=True)
    costs = {name: (2 if name.startswith("dual_") else 1) for name in results}
    winner = ordered[0]
    baseline_rankings = results["saved_baseline_one_search"]
    winner_rankings = results[winner]
    changes = {"new_hit15": 0, "lost_hit15": 0, "same_hit15": 0}
    for row, baseline_ids, winner_ids in zip(rows, baseline_rankings, winner_rankings):
        gold = set((row.get("gold") or {}).get("chunk_ids") or [])
        base_hit, winner_hit = bool(gold.intersection(baseline_ids[:15])), bool(gold.intersection(winner_ids[:15]))
        if not base_hit and winner_hit:
            changes["new_hit15"] += 1
        elif base_hit and not winner_hit:
            changes["lost_hit15"] += 1
        else:
            changes["same_hit15"] += 1

    summary = {
        "experiment": "one_or_two_search_dense_query_fusion_sweep",
        "source_commit": "932a3f7", "baseline_results": str(baseline_path),
        "saved_rankings": str(saved_path), "collection": manifest["collection"],
        "qdrant_url": manifest.get("qdrant_url"), "model_path": str(args.model_path),
        "device": args.device, "mapped_queries": len(rows), "candidate_pool_per_search": args.pool,
        "max_searches_per_strategy_per_query": 2,
        "searches_per_query": costs, "saved_baseline_vs_current_centroid_1search": {
            "baseline_metrics": metrics["saved_baseline_one_search"],
            "centroid_metrics": metrics["centroid_one_search"],
        },
        "strategy_metrics": metrics, "ranking_by_hit15_mrr_ndcg": ordered,
        "best_full_set_strategy": winner, "best_full_set_search_cost": costs[winner],
        "best_full_set_hit_changes_vs_baseline": changes,
        "grouped_5fold_cv_selected_strategy_per_fold": selected_by_fold,
        "grouped_5fold_cv_metrics": cv_metrics,
        "fold_rule": "SHA256(retrieval_query) modulo five; repeated queries stay in the same fold.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            record = {
                "id": row.get("id"), "question": row.get("question"),
                "retrieval_query": row.get("retrieval_query"),
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "rankings": {name: ranked[i] for name, ranked in results.items()},
                "cv_selected_ranking": predictions[i],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# One/two-search dense vector strategies", "",
        f"- Queries: {len(rows)} mapped; per-search candidate pool: {args.pool}",
        "- Baseline reproduction and single-centroid results are copied from the validated first sweep.",
        f"- Grouped 5-fold CV selected metrics: Hit@15 {cv_metrics['hit@15']:.2%}; MRR@10 {cv_metrics['mrr@10']:.4f}; nDCG@5 {cv_metrics['ndcg@5']:.4f}",
        f"- CV selected methods by fold: {selected_by_fold}",
        "", "| Search count | Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        m = metrics[name]
        lines.append(f"| {costs[name]} | {name} | {m['hit@5']:.2%} | {m['hit@10']:.2%} | {m['hit@15']:.2%} | "
                     f"{m['gold_recall@15']:.2%} | {m['mrr@10']:.4f} | {m['ndcg@5']:.4f} |")
    lines.extend(["", f"Best full-set strategy: {winner} ({costs[winner]} searches). Hit@15 changes vs baseline: {changes}.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[dual-search] results={output}")
    print((output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
