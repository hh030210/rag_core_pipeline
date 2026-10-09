"""Replay the final adaptive one-search centroid against the mapped evaluation set."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from .adaptive_centroid import choose_question_mass, compose_centroid
from .experiment import DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, ids, load_jsonl, query_parts, score_metrics, spot_names, unique
from .vector_search import DenseVectorSearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--reference-rankings", type=Path,
                        default=LAB_DIR / "outputs/centroid_variants_refined_20261008/per_query_rankings.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--threshold-chars", type=int, default=39)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    reference = load_jsonl(args.reference_rankings)
    if len(rows) != len(reference):
        raise RuntimeError(f"mapped rows={len(rows)} != reference rankings={len(reference)}")
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir or LAB_DIR / "outputs" / ("adaptive_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    data: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        parts = query_parts(row)
        question_mass, clean_parts, char_count = choose_question_mass(question, parts, args.threshold_chars)
        data.append({
            "question": question, "parts": clean_parts, "question_mass": question_mass,
            "subquery_char_count": char_count, "spots": spot_names(row),
        })
        all_texts.extend([question, *clean_parts])
    all_texts = unique(all_texts)
    print(f"[adaptive-replay] queries={len(rows)} unique_texts={len(all_texts)} pool={args.pool} threshold={args.threshold_chars}", flush=True)
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(all_texts)
    vectors = {text: vector for text, vector in zip(all_texts, encoded)}
    results: Dict[str, List[List[str]]] = defaultdict(list)
    details: List[Dict[str, Any]] = []
    parity = {"exact_order_vs_composed_centroid": 0, "total": len(rows)}

    for index, (row, query, old) in enumerate(zip(rows, data, reference), 1):
        encoded_query = [vectors[query["question"]], *[vectors[part] for part in query["parts"]]]
        vector = compose_centroid(encoded_query[0], encoded_query[1:], query["question_mass"])
        hits = searcher.search_vector(vector, args.pool)
        hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
        ranking = ids(hits, args.depth)
        results["adaptive_threshold_39"].append(ranking)
        results["saved_concatenated_baseline"].append(old["rankings"]["saved_concatenated_baseline"][:args.depth])
        results["current_question_weight_2_centroid"].append(old["rankings"]["current_question_weight_2_centroid"][:args.depth])

        expected_name = "fixed_question_mass_60" if query["question_mass"] == 0.60 else "fixed_question_mass_32"
        expected = old["rankings"][expected_name][:args.depth]
        parity["exact_order_vs_composed_centroid"] += int(ranking == expected)
        details.append({
            "id": row.get("id"), "retrieval_query": row.get("retrieval_query"),
            "question_mass": query["question_mass"],
            "subquery_char_count": query["subquery_char_count"],
            "ranking": ranking, "expected_component_strategy": expected_name,
        })
        if index % 50 == 0 or index == len(rows):
            print(f"[adaptive-replay] queries={index}/{len(rows)}", flush=True)

    metrics = {name: score_metrics(rows, ranking) for name, ranking in results.items()}
    summary = {
        "experiment": "adaptive_centroid_threshold_39_exact_replay",
        "collection": manifest["collection"], "mapped_queries": len(rows),
        "candidate_pool": args.pool, "output_depth": args.depth,
        "threshold_chars": args.threshold_chars,
        "rule": "question mass 0.60 if sum of unique subquery characters <= 39; otherwise 0.32; remaining mass shared equally by subquery vectors.",
        "searches_per_query": 1,
        "parity_vs_precomputed_component_searches": parity,
        "strategy_metrics": metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    lines = [
        "# Adaptive centroid exact replay", "",
        f"- Queries: {len(rows)}; Qdrant searches/query: 1; pool: {args.pool}; top-{args.depth}",
        f"- Exact order parity with precomputed component retrievals: {parity['exact_order_vs_composed_centroid']}/{parity['total']}",
        "", "| Strategy | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        lines.append(f"| {name} | {metric['hit@1']:.2%} | {metric['hit@5']:.2%} | {metric['hit@10']:.2%} | {metric['hit@15']:.2%} | {metric['gold_recall@15']:.2%} | {metric['mrr@10']:.4f} | {metric['ndcg@5']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[adaptive-replay] results={output}")
    print((output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
