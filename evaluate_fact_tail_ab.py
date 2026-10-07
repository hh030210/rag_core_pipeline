"""Paired, single-process A/B for verified fact tail insertion.

Using one Retriever instance prevents parser-cache or process-level variation
from being misread as a fact-index gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import aggregate_summary, evaluate_row, load_gold_records
from rag_core.retrieval import Retriever
from rag_core.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--query-fact-cache", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tail-start", default=10, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw_cache = json.loads(args.query_fact_cache.read_text(encoding="utf-8"))
    query_facts = {
        str(item.get("query", "")): item.get("facts", [])
        for item in raw_cache.get("queries", []) if isinstance(item, dict)
    }
    raw_by_id = {item["id"]: item for item in load_gold_records(args.dataset)}
    baseline = args.run_dir / "evaluation" / "results.jsonl"
    records = []
    for line in baseline.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        raw = raw_by_id.get(str(row["id"]), {})
        gold = row.get("gold", {})
        records.append({
            "id": row["id"], "question": row.get("question", ""),
            "gold_chunk_ids": gold.get("chunk_ids", []),
            "gold_evidence_texts": gold.get("evidence_texts", []),
            "gold_relevance": {}, "answerable": True,
            "retrieval_query": raw.get("rewritten_query", "") or row.get("question", ""),
        })
    records = [item for item in records if item["gold_chunk_ids"]]
    settings = Settings.from_env(
        args.run_dir, backend="local", vector_dim=1024, dimension_pool=100,
        model_path=args.model_path, embedding_device="cpu",
    )
    settings.poi_scope_enabled = True
    settings.entity_anchor_candidate_recall_enabled = True
    settings.high_precision_entity_anchor_enabled = True
    settings.no_candidate_lexical_fallback_enabled = True
    settings.fact_index_tail_insertion_rank = max(0, args.tail_start)
    retriever = Retriever(
        run_dir=args.run_dir, settings=settings,
        embeddings=EmbeddingModel(args.model_path, "cpu", dimension=1024), llm=None,
    )
    variants = {}
    rows_by_variant = {}
    for name, enabled in (("control", False), (f"fact_tail_{args.tail_start}", True)):
        settings.fact_index_candidate_recall_enabled = enabled
        rows = []
        for index, record in enumerate(records, 1):
            query = record["retrieval_query"]
            result = retriever.search(
                query, top_k=5, dimension_pool=100, dimension_only=True,
                query_facts=query_facts.get(query, []) if enabled else [],
            )
            rows.append(evaluate_row(record, result, (1, 3, 5, 10, 15, 20), 20, 800))
            if index % 100 == 0:
                print(f"[{name}] {index}/{len(records)}", flush=True)
        variants[name] = aggregate_summary(rows, (1, 3, 5, 10, 15, 20))["routes"]["dimension"]
        rows_by_variant[name] = rows
    names = list(variants)
    keys = ("hit_rate@15", "gold_recall@15", "mrr@15", "ndcg@15", "miss_count")
    report = {
        "evaluated_with_mapped_golden": len(records), "tail_start": args.tail_start,
        "control": {key: variants[names[0]][key] for key in keys},
        "fact_tail": {key: variants[names[1]][key] for key in keys},
        "delta": {key: round(variants[names[1]][key] - variants[names[0]][key], 8) for key in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
