"""Sweep one-search query-vector aggregation strategies with grouped CV."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .experiment import DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, ids, load_jsonl, query_parts, score_metrics, spot_names, unique
from .vector_search import DenseVectorSearch, l2_normalize


def normalize_weights(values: List[float]) -> List[float]:
    values = [max(0.0, float(value)) for value in values]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def weighted_vector(question_vector: List[float], sub_vectors: List[List[float]],
                    question_mass: float, sub_weights: List[float]) -> List[float]:
    components = [(l2_normalize(question_vector), question_mass)]
    components.extend((l2_normalize(vector), weight) for vector, weight in zip(sub_vectors, sub_weights))
    result = [sum(vector[i] * weight for vector, weight in components)
              for i in range(len(question_vector))]
    return l2_normalize(result)


def cosine(left: List[float], right: List[float]) -> float:
    a, b = l2_normalize(left), l2_normalize(right)
    return sum(x * y for x, y in zip(a, b))


def softmax(values: List[float], temperature: float) -> List[float]:
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    return normalize_weights(exps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--saved-rankings", type=Path,
                        default=LAB_DIR / "outputs/full_20261008/per_query_rankings.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    saved = load_jsonl(args.saved_rankings)
    if len(rows) != len(saved):
        raise RuntimeError(f"mapped rows={len(rows)} but saved rankings={len(saved)}")
    if args.limit:
        rows, saved = rows[:args.limit], saved[:args.limit]
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir or LAB_DIR / "outputs" / ("centroid_variants_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    data: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        parts = [part for part in unique(query_parts(row)) if part != question]
        if not parts:
            parts = [question]
        data.append({"question": question, "parts": parts, "spots": spot_names(row)})
        all_texts.extend([question, *parts])
    all_texts = unique(all_texts)

    print(f"[centroid-variants] queries={len(rows)} unique_texts={len(all_texts)} pool={args.pool}", flush=True)
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(all_texts)
    vectors = {text: vector for text, vector in zip(all_texts, encoded)}

    alpha_values = (0.15, 0.25, 0.30, 0.32, 0.33, 0.34, 0.36, 0.38, 0.40, 0.50, 0.60, 0.70, 0.80)
    names = [f"fixed_question_mass_{int(round(alpha * 100)):02d}" for alpha in alpha_values]
    names += ["question50_subquery_cosine_softmax_t10",
              "question50_subquery_cosine_softmax_t25",
              "question50_subquery_diversity_weighted"]
    results: Dict[str, List[List[str]]] = defaultdict(list)

    for index, (row, query, old) in enumerate(zip(rows, data, saved), 1):
        question = query["question"]
        qvec = l2_normalize(vectors[question])
        subqueries = query["parts"]
        svecs = [l2_normalize(vectors[text]) for text in subqueries]
        n = len(svecs)
        base_weights = normalize_weights([1.0] * n)

        # Preserve concatenation and the current optimized centroid as controls.
        results["saved_concatenated_baseline"].append(old["rankings"]["saved_baseline"][:args.depth])
        results["current_question_weight_2_centroid"].append(
            old["rankings"]["question_plus_subquery_vector_centroid"][:args.depth])

        for alpha, name in zip(alpha_values, names[:len(alpha_values)]):
            sub_weights = [(1.0 - alpha) / n] * n
            vector = weighted_vector(qvec, svecs, alpha, sub_weights)
            hits = searcher.search_vector(vector, args.pool)
            hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
            results[name].append(ids(hits, args.depth))

        sims = [cosine(qvec, vector) for vector in svecs]
        for temperature, name in ((0.10, names[-3]), (0.25, names[-2])):
            proportions = softmax(sims, temperature)
            sub_weights = [(1.0 - 0.5) * value for value in proportions]
            vector = weighted_vector(qvec, svecs, 0.5, sub_weights)
            hits = searcher.search_vector(vector, args.pool)
            hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
            results[name].append(ids(hits, args.depth))

        if n > 1:
            novelty = []
            for i, vector in enumerate(svecs):
                nearest_other = max(cosine(vector, other) for j, other in enumerate(svecs) if i != j)
                novelty.append(max(0.05, 1.0 - nearest_other))
            proportions = normalize_weights(novelty)
        else:
            proportions = base_weights
        sub_weights = [(1.0 - 0.5) * value for value in proportions]
        vector = weighted_vector(qvec, svecs, 0.5, sub_weights)
        hits = searcher.search_vector(vector, args.pool)
        hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
        results[names[-1]].append(ids(hits, args.depth))

        if index % 50 == 0 or index == len(rows):
            print(f"[centroid-variants] queries={index}/{len(rows)}", flush=True)

    metrics = {name: score_metrics(rows, rankings) for name, rankings in results.items()}
    folds = [int(hashlib.sha256(str(row.get("retrieval_query") or row.get("question") or "").encode("utf-8")).hexdigest()[:8], 16) % 5
             for row in rows]
    predictions: List[List[str] | None] = [None] * len(rows)
    selected: Dict[str, str] = {}
    candidate_names = names
    for fold in range(5):
        train = [i for i, assigned in enumerate(folds) if assigned != fold]
        test = [i for i, assigned in enumerate(folds) if assigned == fold]
        train_metrics = {
            name: score_metrics([rows[i] for i in train], [results[name][i] for i in train])
            for name in candidate_names
        }
        chosen = max(candidate_names, key=lambda name: (
            train_metrics[name]["hit@15"], train_metrics[name]["mrr@10"], train_metrics[name]["ndcg@5"]))
        selected[str(fold)] = chosen
        for i in test:
            predictions[i] = results[chosen][i]
    cv_metrics = score_metrics(rows, [ranking or [] for ranking in predictions])
    ordered = sorted(candidate_names, key=lambda name: (
        metrics[name]["hit@15"], metrics[name]["mrr@10"], metrics[name]["ndcg@5"]), reverse=True)

    summary = {
        "experiment": "one_search_centroid_aggregation_variants",
        "baseline_results": str(baseline_path), "saved_rankings": str(args.saved_rankings),
        "collection": manifest["collection"], "model_path": str(args.model_path),
        "mapped_queries": len(rows), "candidate_pool": args.pool, "depth": args.depth,
        "strategies": metrics, "ranked_strategies": ordered,
        "best_full_set_strategy": ordered[0],
        "grouped_5fold_cv_selected_strategy_by_fold": selected,
        "grouped_5fold_cv_metrics": cv_metrics,
        "fold_rule": "SHA256(retrieval_query) modulo five",
        "note": "Full-set best is exploratory; CV results choose only among new aggregation strategies on each training fold.",
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
        "# One-search centroid aggregation sweep", "",
        f"- Queries: {len(rows)}; Qdrant pool: {args.pool}; output depth: {args.depth}",
        f"- Grouped 5-fold selected metrics: Hit@15 {cv_metrics['hit@15']:.2%}; MRR@10 {cv_metrics['mrr@10']:.4f}; nDCG@5 {cv_metrics['ndcg@5']:.4f}",
        f"- CV selected by fold: {selected}", "",
        "| Strategy | Hit@15 | MRR@10 | nDCG@5 |",
        "|---|---:|---:|---:|",
    ]
    for name in ordered:
        metric = metrics[name]
        lines.append(f"| {name} | {metric['hit@15']:.2%} | {metric['mrr@10']:.4f} | {metric['ndcg@5']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[centroid-variants] results={output}")
    print((output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
