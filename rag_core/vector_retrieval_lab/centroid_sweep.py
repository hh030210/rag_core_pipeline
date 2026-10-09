"""Tune query-vector centroid weights and evaluate with grouped 5-fold CV."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from .experiment import DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, load_jsonl, query_parts, score_metrics, spot_names, unique
from .vector_search import DenseVectorSearch, l2_normalize

WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def centroid(query_texts: List[str], query_weight: float, vectors: Dict[str, List[float]]) -> List[float]:
    terms = unique(query_texts)
    normalized = [l2_normalize(vectors[text]) for text in terms]
    part_set = set(query_texts[1:])
    weights = [query_weight * int(text == query_texts[0]) + int(text in part_set) for text in terms]
    total = sum(weights) or 1.0
    result = [sum(w * vec[i] for vec, w in zip(normalized, weights)) / total for i in range(len(normalized[0]))]
    return l2_normalize(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(baseline) if (row.get("gold") or {}).get("chunk_ids")]
    if args.limit:
        rows = rows[:args.limit]
    output = args.output_dir or LAB_DIR / "outputs" / ("centroid_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    query_data = []
    all_texts = []
    for row in rows:
        parts = query_parts(row)
        question = str(row.get("question") or row.get("retrieval_query") or "").strip()
        texts = unique([question, *parts])
        query_data.append({"question": question, "parts": parts, "texts": texts,
                           "spots": spot_names(row)})
        all_texts.extend(texts)
    texts = unique(all_texts)
    print(f"[centroid-sweep] queries={len(rows)} unique_texts={len(texts)} weights={WEIGHTS}", flush=True)
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(texts)
    vectors = {text: vector for text, vector in zip(texts, encoded)}
    rankings: Dict[float, List[List[str]]] = {weight: [] for weight in WEIGHTS}

    for index, (row, query) in enumerate(zip(rows, query_data), 1):
        for weight in WEIGHTS:
            vector = centroid(query["texts"], weight, vectors)
            hits = searcher.search_vector(vector, args.pool)
            hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
            rankings[weight].append([str(hit["chunk_id"]) for hit in hits[:args.depth]])
        if index % 50 == 0 or index == len(rows):
            print(f"[centroid-sweep] queries={index}/{len(rows)}", flush=True)

    metrics = {str(weight): score_metrics(rows, result) for weight, result in rankings.items()}
    fold_for_row = []
    for row in rows:
        key = str(row.get("retrieval_query") or row.get("question") or "")
        fold_for_row.append(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 5)
    cv_predictions: List[List[str] | None] = [None] * len(rows)
    selected_by_fold: Dict[str, float] = {}
    fold_scores = {}
    for fold in range(5):
        train = [i for i, value in enumerate(fold_for_row) if value != fold]
        test = [i for i, value in enumerate(fold_for_row) if value == fold]
        train_metrics = {
            weight: score_metrics([rows[i] for i in train], [rankings[weight][i] for i in train])
            for weight in WEIGHTS
        }
        chosen = max(WEIGHTS, key=lambda weight: (
            train_metrics[weight].get("hit@15", 0.0),
            train_metrics[weight].get("mrr@10", 0.0),
            train_metrics[weight].get("ndcg@5", 0.0),
        ))
        for i in test:
            cv_predictions[i] = rankings[chosen][i]
        selected_by_fold[str(fold)] = chosen
        fold_scores[str(fold)] = score_metrics([rows[i] for i in test], [rankings[chosen][i] for i in test])
    cv = score_metrics(rows, [value or [] for value in cv_predictions])
    selected_weight_counts = Counter(selected_by_fold.values())
    fixed_winner = max(WEIGHTS, key=lambda weight: (
        metrics[str(weight)].get("hit@15", 0.0), metrics[str(weight)].get("mrr@10", 0.0),
        metrics[str(weight)].get("ndcg@5", 0.0)))
    baseline_rankings = [[str(item.get("chunk_id", "")) for item in (row.get("retrieval") or {}).get("semantic_results", [])]
                         for row in rows]
    baseline_metrics = score_metrics(rows, baseline_rankings)
    winner_rankings = rankings[fixed_winner]
    per_query = []
    for i, row in enumerate(rows):
        gold = set((row.get("gold") or {}).get("chunk_ids") or [])
        base_hit = bool(gold.intersection(baseline_rankings[i][:15]))
        best_hit = bool(gold.intersection(winner_rankings[i][:15]))
        per_query.append({
            "id": row.get("id"), "question": row.get("question"),
            "gold_chunk_ids": sorted(gold), "baseline_hit15": base_hit,
            "winner_hit15": best_hit, "winner_ranked_ids": winner_rankings[i],
            "cv_selected_ranked_ids": cv_predictions[i],
        })
    net = {
        "new_hit_queries": sum(not item["baseline_hit15"] and item["winner_hit15"] for item in per_query),
        "lost_hit_queries": sum(item["baseline_hit15"] and not item["winner_hit15"] for item in per_query),
        "unchanged_hit_queries": sum(item["baseline_hit15"] == item["winner_hit15"] for item in per_query),
    }
    summary = {
        "experiment": "query_centroid_weight_sweep_with_grouped_5fold_cv",
        "source_commit": "932a3f7", "baseline_results": str(baseline),
        "collection": manifest["collection"], "model_path": str(args.model_path),
        "device": args.device, "query_count": len(rows), "pool": args.pool,
        "depth": args.depth, "weights_tested": list(WEIGHTS),
        "weight_metrics_full_set": metrics,
        "saved_baseline_metrics": baseline_metrics,
        "cross_validated_selected_weight_metrics": cv,
        "selected_weight_by_fold": selected_by_fold,
        "selected_weight_vote_count": {str(k): v for k, v in selected_weight_counts.items()},
        "heldout_metrics_by_fold": fold_scores,
        "full_set_winner_by_hit15_mrr_ndcg": fixed_winner,
        "full_set_winner_net_hit_changes_vs_baseline": net,
        "split_rule": "SHA256(retrieval_query) modulo five; duplicate retrieval queries remain in the same fold.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for item in per_query:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    lines = [
        "# Query vector centroid weight sweep", "",
        f"- Mapped queries: {len(rows)}; Qdrant pool: {args.pool}; folds: 5",
        f"- Saved baseline: Hit@15 {baseline_metrics['hit@15']:.2%}, MRR@10 {baseline_metrics['mrr@10']:.4f}, nDCG@5 {baseline_metrics['ndcg@5']:.4f}",
        f"- CV weight selections by fold: {selected_by_fold}",
        f"- CV selected-weight score: Hit@15 {cv['hit@15']:.2%}, MRR@10 {cv['mrr@10']:.4f}, nDCG@5 {cv['ndcg@5']:.4f}",
        "", "| Original-query weight | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weight in sorted(WEIGHTS):
        m = metrics[str(weight)]
        lines.append(f"| {weight:g} | {m['hit@5']:.2%} | {m['hit@10']:.2%} | {m['hit@15']:.2%} | "
                     f"{m['gold_recall@15']:.2%} | {m['mrr@10']:.4f} | {m['ndcg@5']:.4f} |")
    lines.extend(["", f"Best full-set weight: {fixed_winner:g}; new Hit@15 cases: {net['new_hit_queries']}; regressed cases: {net['lost_hit_queries']}.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[centroid-sweep] results={output}")
    print((output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
