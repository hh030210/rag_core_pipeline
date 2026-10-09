"""Compare isolated dense-query strategies with the saved main experiment."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from .vector_search import DenseVectorSearch, l2_normalize

DEFAULT_RUN = Path("/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1")
DEFAULT_MODEL = Path("/home/humq/rag_db_silm/model/bge-m3")
LAB_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def query_parts(record: Dict[str, Any]) -> List[str]:
    text = str(record.get("retrieval_query") or record.get("question") or "").strip()
    return unique(piece for piece in text.split("|") if piece.strip()) or [text]


def spot_names(record: Dict[str, Any]) -> List[str]:
    return unique((record.get("query_analysis") or {}).get("spot_names") or [])


def score_metrics(rows: Sequence[Dict[str, Any]], rankings: Sequence[Sequence[str]]) -> Dict[str, float]:
    count = len(rows)
    result: Dict[str, float] = {"query_count": count}
    first_ranks: List[int | None] = []
    for row, ranked in zip(rows, rankings):
        gold = set((row.get("gold") or {}).get("chunk_ids") or [])
        first_ranks.append(next((i for i, cid in enumerate(ranked, 1) if cid in gold), None))
    for k in (1, 5, 10, 15, 20):
        hits, recall_sum = 0, 0.0
        for row, ranked in zip(rows, rankings):
            gold = set((row.get("gold") or {}).get("chunk_ids") or [])
            found = len(gold.intersection(ranked[:k]))
            hits += int(found > 0)
            recall_sum += found / len(gold) if gold else 0.0
        result[f"hit@{k}"] = hits / count if count else 0.0
        result[f"gold_recall@{k}"] = recall_sum / count if count else 0.0
    result["mrr@10"] = sum(1.0 / r for r in first_ranks if r and r <= 10) / count if count else 0.0
    ndcg_sum = 0.0
    for row, ranked in zip(rows, rankings):
        gold = set((row.get("gold") or {}).get("chunk_ids") or [])
        dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:5]) if cid in gold)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold))))
        ndcg_sum += dcg / ideal if ideal else 0.0
    result["ndcg@5"] = ndcg_sum / count if count else 0.0
    hit_ranks = [r for r in first_ranks if r]
    result["mean_first_gold_rank_on_hits"] = sum(hit_ranks) / len(hit_ranks) if hit_ranks else 0.0
    return result


def filtered_hits(searcher: DenseVectorSearch, text: str, limit: int, spots: List[str],
                  vectors: Dict[str, List[float]], raw_cache: Dict[Tuple[str, int], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    key = (text, limit)
    if key not in raw_cache:
        raw_cache[key] = searcher.search_vector(vectors[text], limit)
    matched = [hit for hit in raw_cache[key] if searcher.matches_spot(hit["payload"], spots)]
    return [dict(hit, rank=rank) for rank, hit in enumerate(matched, 1)]


def reciprocal_rank_fusion(lists: Sequence[Sequence[Dict[str, Any]]], weights: Sequence[float], k: int) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = defaultdict(float)
    order: Dict[str, int] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    sequence = 0
    for hits, weight in zip(lists, weights):
        for rank, hit in enumerate(hits, 1):
            cid = str(hit["chunk_id"])
            if cid not in order:
                order[cid] = sequence
                sequence += 1
            scores[cid] += float(weight) / (k + rank)
            payloads[cid] = hit.get("payload", {})
    ranked = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))
    return [{"chunk_id": cid, "score": scores[cid], "payload": payloads[cid]} for cid in ranked]


def normalized_score_fusion(lists: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = defaultdict(float)
    order: Dict[str, int] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    sequence = 0
    if not lists:
        return []
    for hits in lists:
        vals = [float(hit["score"]) for hit in hits]
        low, high = (min(vals), max(vals)) if vals else (0.0, 0.0)
        for hit in hits:
            cid = str(hit["chunk_id"])
            if cid not in order:
                order[cid] = sequence
                sequence += 1
            scores[cid] += ((float(hit["score"]) - low) / (high - low) if high > low else 1.0) / len(lists)
            payloads[cid] = hit.get("payload", {})
    ranked = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))
    return [{"chunk_id": cid, "score": scores[cid], "payload": payloads[cid]} for cid in ranked]


def vector_centroid(texts: Sequence[str], weights: Sequence[float], vectors: Dict[str, List[float]]) -> List[float]:
    normalized = [l2_normalize(vectors[text]) for text in texts]
    total = sum(weights) or 1.0
    center = [sum(w * vec[i] for vec, w in zip(normalized, weights)) / total for i in range(len(normalized[0]))]
    return l2_normalize(center)


def ids(hits: Sequence[Dict[str, Any]], depth: int) -> List[str]:
    return [str(hit["chunk_id"]) for hit in hits[:depth]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-pool", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-run only; 0 evaluates all mapped queries.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError(f"No mapped evaluation rows in {baseline_path}")
    output_dir = args.output_dir or LAB_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"], model_path=str(args.model_path), device=args.device,
    )
    records: List[Dict[str, Any]] = []
    query_texts: List[str] = []
    for row in rows:
        joined = str(row.get("retrieval_query") or row.get("question") or "").strip()
        question = str(row.get("question") or joined).strip()
        parts = query_parts(row)
        comma_joined = "，".join(parts)
        query_texts.extend([joined, question, comma_joined, *parts])
        records.append({"joined": joined, "question": question, "parts": parts,
                        "comma_joined": comma_joined, "spots": spot_names(row)})
    texts = unique(query_texts)
    print(f"[vector-lab] mapped_queries={len(rows)} unique_query_texts={len(texts)}", flush=True)
    encoded = searcher.encode(texts)
    vector_by_text = {text: vector for text, vector in zip(texts, encoded)}
    raw_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    rankings: Dict[str, List[List[str]]] = defaultdict(list)
    parity = {"exact_order": 0, "same_top20_set": 0, "total": 0}

    for index, (row, query) in enumerate(zip(rows, records), 1):
        joined, question = query["joined"], query["question"]
        parts, spots = query["parts"], query["spots"]
        saved = (row.get("retrieval") or {}).get("semantic_results") or []
        saved_ids = [str(item.get("chunk_id", "")) for item in saved if item.get("chunk_id")]
        rankings["saved_baseline"].append(saved_ids[:args.depth])

        joined_hits = filtered_hits(searcher, joined, args.baseline_pool, spots, vector_by_text, raw_cache)
        rankings["concat_current"].append(ids(joined_hits, args.depth))
        question_hits = filtered_hits(searcher, question, args.baseline_pool, spots, vector_by_text, raw_cache)
        rankings["question_only"].append(ids(question_hits, args.depth))
        comma_hits = filtered_hits(searcher, query["comma_joined"], args.baseline_pool, spots, vector_by_text, raw_cache)
        rankings["subqueries_comma_joined"].append(ids(comma_hits, args.depth))

        part_lists = [filtered_hits(searcher, part, args.candidate_pool, spots, vector_by_text, raw_cache)
                      for part in parts]
        for rrf_k in (20, 60, 100):
            name = f"subquery_rrf_k{rrf_k}_pool{args.candidate_pool}"
            rankings[name].append(ids(reciprocal_rank_fusion(part_lists, [1.0] * len(part_lists), rrf_k), args.depth))

        weighted_texts = unique([question, *parts])
        weighted_lists = [filtered_hits(searcher, text, args.candidate_pool, spots, vector_by_text, raw_cache)
                          for text in weighted_texts]
        weights = [2.0 if text == question else 1.0 for text in weighted_texts]
        rankings[f"question_plus_subquery_rrf_k60_pool{args.candidate_pool}"].append(
            ids(reciprocal_rank_fusion(weighted_lists, weights, 60), args.depth))
        rankings[f"question_plus_subquery_rrf_k20_pool{args.candidate_pool}"].append(
            ids(reciprocal_rank_fusion(weighted_lists, weights, 20), args.depth))
        rankings[f"question_plus_subquery_minmax_pool{args.candidate_pool}"].append(
            ids(normalized_score_fusion(weighted_lists), args.depth))

        joined_texts = unique([joined, *parts])
        joined_lists = [filtered_hits(searcher, text, args.candidate_pool, spots, vector_by_text, raw_cache)
                        for text in joined_texts]
        joined_weights = [2.0 if text == joined else 1.0 for text in joined_texts]
        rankings[f"joined_plus_subquery_rrf_k60_pool{args.candidate_pool}"].append(
            ids(reciprocal_rank_fusion(joined_lists, joined_weights, 60), args.depth))

        center = vector_centroid(parts, [1.0] * len(parts), vector_by_text)
        center_hits = searcher.search_vector(center, args.baseline_pool)
        center_hits = [hit for hit in center_hits if searcher.matches_spot(hit["payload"], spots)]
        rankings["subquery_vector_centroid"].append(ids(center_hits, args.depth))
        center_texts = unique([question, *parts])
        center_weights = [2.0 if text == question else 1.0 for text in center_texts]
        weighted_center = vector_centroid(center_texts, center_weights, vector_by_text)
        weighted_hits = searcher.search_vector(weighted_center, args.baseline_pool)
        weighted_hits = [hit for hit in weighted_hits if searcher.matches_spot(hit["payload"], spots)]
        rankings["question_plus_subquery_vector_centroid"].append(ids(weighted_hits, args.depth))

        if saved_ids:
            parity["total"] += 1
            current_ids = ids(joined_hits, args.depth)
            parity["exact_order"] += int(current_ids == saved_ids[:args.depth])
            parity["same_top20_set"] += int(set(current_ids) == set(saved_ids[:args.depth]))
        if index % 50 == 0 or index == len(rows):
            print(f"[vector-lab] queries={index}/{len(rows)} qdrant_searches={len(raw_cache)}", flush=True)

    metrics = {name: score_metrics(rows, ranked) for name, ranked in rankings.items()}
    ordered = sorted(metrics, key=lambda name: (
        metrics[name].get("hit@15", 0.0), metrics[name].get("mrr@10", 0.0),
        metrics[name].get("ndcg@5", 0.0)), reverse=True)
    parity_summary = {key: (value / parity["total"] if key != "total" and parity["total"] else value)
                      for key, value in parity.items()}
    summary = {
        "experiment": "dense_vector_query_strategy_comparison",
        "source_commit": "932a3f7",
        "baseline_results": str(baseline_path), "run_dir": str(args.run_dir),
        "collection": manifest["collection"], "qdrant_url": manifest.get("qdrant_url"),
        "embedding_model": str(args.model_path), "embedding_device": args.device,
        "mapped_query_count": len(rows), "baseline_pool": args.baseline_pool,
        "candidate_pool_for_multiqueury": args.candidate_pool, "evaluation_depth": args.depth,
        "saved_baseline_vs_fresh_concat_parity": parity_summary,
        "strategy_metrics": metrics, "ranking_by_hit15_then_mrr10_then_ndcg5": ordered,
        "selection_note": "Exploratory comparison on the saved mapped set; validate the selected strategy on a fresh holdout before production adoption.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            item = {
                "id": row.get("id"), "question": row.get("question"),
                "retrieval_query": row.get("retrieval_query"),
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "rankings": {name: ranked[row_index] for name, ranked in rankings.items()},
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Dense vector query strategy comparison", "",
        f"- Mapped queries: {len(rows)} / saved main evaluation",
        f"- Collection: {manifest['collection']}",
        f"- Embedding: {args.model_path} on {args.device}",
        f"- Baseline pool: {args.baseline_pool}; multi-query candidate pool: {args.candidate_pool}; reported depth: {args.depth}",
        f"- Saved-baseline vs fresh concatenation exact order parity: {parity['exact_order']}/{parity['total']}",
        "", "| Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        m = metrics[name]
        lines.append(f"| {name} | {m['hit@5']:.2%} | {m['hit@10']:.2%} | {m['hit@15']:.2%} | "
                     f"{m['gold_recall@15']:.2%} | {m['mrr@10']:.4f} | {m['ndcg@5']:.4f} |")
    lines.extend(["", "Per-query top rankings are in per_query_rankings.jsonl.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[vector-lab] results={output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
