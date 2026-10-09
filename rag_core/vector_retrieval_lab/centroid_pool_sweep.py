"""Retrieve a deeper candidate pool for the best one-search centroid variants."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from .experiment import DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, ids, load_jsonl, query_parts, score_metrics, spot_names, unique
from .vector_search import DenseVectorSearch, l2_normalize


def combine(question: List[float], parts: List[List[float]], qmass: float) -> List[float]:
    n = max(1, len(parts))
    weights = [qmass] + [(1.0 - qmass) / n] * len(parts)
    vectors = [l2_normalize(question)] + [l2_normalize(vector) for vector in parts]
    total = [sum(weight * vector[i] for weight, vector in zip(weights, vectors))
             for i in range(len(question))]
    return l2_normalize(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    if args.limit:
        rows = rows[:args.limit]
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir or LAB_DIR / "outputs" / ("centroid_pool_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
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
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(all_texts)
    vectors = {text: vector for text, vector in zip(all_texts, encoded)}
    names = ["current_question_weight_2_centroid", "fixed_question_mass_33",
             "fixed_question_mass_50", "fixed_question_mass_60"]
    results: Dict[str, List[List[str]]] = defaultdict(list)
    print(f"[centroid-pool] queries={len(rows)} pool={args.pool}", flush=True)
    for index, (row, query) in enumerate(zip(rows, data), 1):
        question = query["question"]
        qvec = l2_normalize(vectors[question])
        subvecs = [l2_normalize(vectors[text]) for text in query["parts"]]
        if question in query["parts"]:
            # The current formula de-duplicates identical question/subquery text.
            subvecs = [l2_normalize(vectors[text]) for text in query["parts"] if text != question] or [qvec]
        current_weights = [2.0] + [1.0] * len(subvecs)
        current_vectors = [qvec] + subvecs
        current = l2_normalize([
            sum(weight * vector[i] for weight, vector in zip(current_weights, current_vectors))
            for i in range(len(qvec))
        ])
        vectors_to_search = [
            current,
            combine(qvec, subvecs, 0.33),
            combine(qvec, subvecs, 0.50),
            combine(qvec, subvecs, 0.60),
        ]
        for name, vector in zip(names, vectors_to_search):
            hits = searcher.search_vector(vector, args.pool)
            hits = [hit for hit in hits if searcher.matches_spot(hit["payload"], query["spots"])]
            results[name].append(ids(hits, args.depth))
        if index % 50 == 0 or index == len(rows):
            print(f"[centroid-pool] queries={index}/{len(rows)}", flush=True)

    metrics = {name: score_metrics(rows, ranking) for name, ranking in results.items()}
    summary = {
        "experiment": "deeper_candidate_pool_for_centroid_variants",
        "collection": manifest["collection"], "mapped_queries": len(rows),
        "candidate_pool": args.pool, "stored_depth": args.depth, "strategy_metrics": metrics,
        "strategies": names,
        "note": "Each listed strategy uses exactly one Qdrant vector search per query.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({
                "id": row.get("id"), "retrieval_query": row.get("retrieval_query"),
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "rankings": {name: ranking[i] for name, ranking in results.items()},
            }, ensure_ascii=False) + "\n")
    print(f"[centroid-pool] results={output}")
    for name in names:
        print(name, metrics[name])


if __name__ == "__main__":
    main()
