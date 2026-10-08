"""Evaluate only the dimension route; no semantic model or vector scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import (
    aggregate_summary,
    evaluate_row,
    load_chunks,
    load_gold_records,
    map_gold_to_chunks,
)
from rag_core.retrieval import Retriever
from rag_core.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--threshold-profile", default="global_058",
                        choices=("global_058", "typed_moderate"))
    parser.add_argument("--disable-poi-registry", action="store_true",
                        help="Run the direct-string subject filter baseline.")
    parser.add_argument("--disable-fact-anchor-rerank", action="store_true",
                        help="Disable query literal / POI fact tie-breaking bonus.")
    parser.add_argument("--enable-entity-anchor-candidate-recall", action="store_true",
                        help="Union exact POI / rare-fact chunks within the resolved scenic scope.")
    parser.add_argument("--enable-auto-entity-registry", action="store_true",
                        help="Derive landmark-to-scenic mappings from indexed chunk text.")
    parser.add_argument("--enable-high-precision-entity-anchor", action="store_true",
                        help="Use validated rare POI/building/person/artifact/organization anchors only.")
    parser.add_argument("--include-unmapped-golden", action="store_true",
                        help="Keep samples with no validated gold-to-chunk mapping in metric denominators.")
    parser.add_argument("--use-rewritten-query", action="store_true",
                        help="Use dataset-provided context-resolved queries when available (multi-turn evaluation only).")
    parser.add_argument("--enable-no-candidate-lexical-fallback", action="store_true",
                        help="Use constrained BM25-style fact recall only when the dimension route is empty.")
    fact_index_group = parser.add_mutually_exclusive_group()
    fact_index_group.add_argument("--enable-verified-fact-index", dest="verified_fact_index",
                                  action="store_true",
                                  help="Enable exact subject+relation matches from fact_index_v1.json (default).")
    fact_index_group.add_argument("--disable-verified-fact-index", dest="verified_fact_index",
                                  action="store_false",
                                  help="Disable verified fact candidate recall for this evaluation.")
    parser.set_defaults(verified_fact_index=None)
    parser.add_argument("--query-fact-cache", type=Path,
                        help="JSON facts pre-parsed from evaluation queries; required to activate fact matching.")
    parser.add_argument("--verified-fact-tail-start", default=None, type=int,
                        help="Insert verified fact candidates after this rank (default: 10; 0 mixes by score).")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env(
        args.run_dir, backend="local", mock=args.mock, vector_dim=1024,
        dimension_pool=100, model_path=args.model_path, embedding_device=args.device,
    )
    settings.tag_vector_threshold_profile = args.threshold_profile
    settings.poi_scope_enabled = not args.disable_poi_registry
    settings.fact_anchor_rerank_enabled = not args.disable_fact_anchor_rerank
    settings.entity_anchor_candidate_recall_enabled = args.enable_entity_anchor_candidate_recall
    settings.auto_entity_registry_enabled = args.enable_auto_entity_registry
    settings.high_precision_entity_anchor_enabled = args.enable_high_precision_entity_anchor
    settings.no_candidate_lexical_fallback_enabled = args.enable_no_candidate_lexical_fallback
    if args.verified_fact_index is not None:
        settings.fact_index_candidate_recall_enabled = args.verified_fact_index
    if args.verified_fact_tail_start is not None:
        settings.fact_index_tail_insertion_rank = max(0, args.verified_fact_tail_start)
    query_fact_cache = {}
    if args.query_fact_cache:
        cache = json.loads(args.query_fact_cache.read_text(encoding="utf-8"))
        entries = cache.get("queries", []) if isinstance(cache, dict) else cache
        for item in entries if isinstance(entries, list) else []:
            if isinstance(item, dict) and str(item.get("query", "")).strip():
                query_fact_cache[str(item["query"])] = item.get("facts", [])
    fact_index_path = args.run_dir / "fact_index_v1.json"
    if settings.fact_index_candidate_recall_enabled:
        if not fact_index_path.exists():
            print(f"[验证事实索引] 未找到 {fact_index_path}，本次不会召回事实候选。", flush=True)
        if not query_fact_cache:
            print("[验证事实索引] 查询事实缓存为空；请传入 --query-fact-cache，事实匹配才会生效。", flush=True)
    retriever = Retriever(
        run_dir=args.run_dir,
        settings=settings,
        embeddings=EmbeddingModel(
            args.model_path, args.device, dimension=1024, mock=args.mock,
        ),
        llm=None,
    )
    # Reuse the exact gold-to-chunk mapping from the baseline when available;
    # this makes the comparison independent of later fuzzy-mapping changes.
    baseline_results = args.run_dir / "evaluation" / "results.jsonl"
    if baseline_results.exists():
        dataset_by_id = {item["id"]: item for item in load_gold_records(args.dataset)}
        records = []
        for line in baseline_results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            raw_record = dataset_by_id.get(str(row["id"]), {})
            records.append({
                "id": row["id"],
                "question": row.get("question", ""),
                "gold_chunk_ids": row.get("gold", {}).get("chunk_ids", []),
                "gold_evidence_texts": row.get("gold", {}).get("evidence_texts", []),
                "gold_resolution": row.get("gold", {}).get("gold_resolution", {}),
                "gold_relevance": {},
                "reference_answer": row.get("reference_answer", ""),
                "question_type": row.get("question_type", ""),
                "spot": row.get("spot", ""),
                "answerable": row.get("answerable", True),
                "rewritten_query": raw_record.get("rewritten_query", ""),
                "multi_turn_context": raw_record.get("multi_turn_context", ""),
            })
    else:
        chunks = load_chunks(args.run_dir)
        records = [
            map_gold_to_chunks(record, chunks)
            for record in load_gold_records(args.dataset)
        ]
    dataset_total = len(records)
    unmapped_records = [record for record in records if not record.get("gold_chunk_ids")]
    if not args.include_unmapped_golden:
        records = [record for record in records if record.get("gold_chunk_ids")]
    if not records:
        raise ValueError("No records with a validated gold-to-chunk mapping are available for evaluation")
    ks = (1, 3, 5, 10, 15, 20)
    rows = []
    empty_candidates = 0
    for index, record in enumerate(records, 1):
        retrieval_query = record.get("rewritten_query", "") if args.use_rewritten_query else ""
        retrieval_query = retrieval_query or record["question"]
        retrieval = retriever.search(
            retrieval_query, top_k=5, dimension_pool=100,
            dimension_only=True, query_facts=query_fact_cache.get(retrieval_query, []),
        )
        if not retrieval["dimension_candidates"]:
            empty_candidates += 1
        rows.append(evaluate_row(record, retrieval, ks, 20, 800))
        if index % 100 == 0:
            print(f"[维度评测] {index}/{len(records)}", flush=True)

    summary = aggregate_summary(rows, ks)
    summary["dataset_total"] = dataset_total
    summary["excluded_unmapped_golden"] = 0 if args.include_unmapped_golden else len(unmapped_records)
    summary["evaluated_with_mapped_golden"] = len(records)
    summary["dimension_diagnostics"] = {
        "empty_candidate_queries": empty_candidates,
        "empty_candidate_rate": round(empty_candidates / len(rows), 8) if rows else 0.0,
        "policy": "wide recall; main-dimension rerank; canonical labels and parent coverage bonus",
        "scope": "parent_document",
        "embedding_mode": "mock" if args.mock else (args.model_path or "configured_default"),
        "tag_vector_threshold_profile": args.threshold_profile,
        "poi_registry_enabled": settings.poi_scope_enabled,
        "fact_anchor_rerank_enabled": settings.fact_anchor_rerank_enabled,
        "entity_anchor_candidate_recall_enabled": settings.entity_anchor_candidate_recall_enabled,
        "auto_entity_registry_enabled": settings.auto_entity_registry_enabled,
        "high_precision_entity_anchor_enabled": settings.high_precision_entity_anchor_enabled,
        "no_candidate_lexical_fallback_enabled": settings.no_candidate_lexical_fallback_enabled,
        "verified_fact_index_enabled": settings.fact_index_candidate_recall_enabled,
        "verified_fact_index_file_exists": fact_index_path.exists(),
        "query_fact_cache": str(args.query_fact_cache) if args.query_fact_cache else "",
        "query_fact_cache_query_count": len(query_fact_cache),
        "verified_fact_tail_start": settings.fact_index_tail_insertion_rank,
        "use_rewritten_query": args.use_rewritten_query,
    }
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.output / "unmapped_golden_audit.json").write_text(
        json.dumps([{
            "id": record.get("id", ""),
            "question": record.get("question", ""),
            "gold_resolution": record.get("gold_resolution", {}),
        } for record in unmapped_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "total": summary["total"],
        "gold_mapped": summary["gold_mapped"],
        "excluded_unmapped_golden": summary["excluded_unmapped_golden"],
        "dimension": summary["routes"]["dimension"],
        "dimension_diagnostics": summary["dimension_diagnostics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
