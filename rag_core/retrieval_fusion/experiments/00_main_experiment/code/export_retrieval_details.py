#!/usr/bin/env python3
"""Freshly retrieve all raw QA questions and save three aligned routes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))

from rag_core.retrieval_fusion.experiments.common.live_runtime import (  # noqa: E402
    LiveRetrievalRuntime,
    add_live_arguments,
    parse_ks,
)

from rag_core.retrieval_fusion.experiments.common.latest_output import stage_latest  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "output" / "runs"
MAX_K = 20
ROUTES = {
    "vector": ("semantic_candidates", "向量检索（用户查询语义）"),
    "dimension": ("dimension_candidates", "维度检索（关键信息匹配）"),
    "hybrid": ("fusion_candidates", "混合检索（融合排序）"),
}
DETAIL_FILES = {
    "vector": "vector_retrieval_details.jsonl",
    "dimension": "dimension_retrieval_details.jsonl",
    "hybrid": "hybrid_retrieval_details.jsonl",
}


def _document_key(value) -> str:
    value = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if value.lower().endswith((".md", ".txt", ".json")):
        value = value.rsplit(".", 1)[0]
    return value


def _chunk_document(chunk) -> str:
    return _document_key(
        chunk.get("source_file") or chunk.get("parent_doc_id") or chunk.get("doc_id")
        or chunk.get("doc_title")
    )


def _gold_documents(record, chunks_by_id) -> list[str]:
    documents = {
        _chunk_document(chunks_by_id.get(str(chunk_id), {}))
        for chunk_id in record.get("gold_chunk_ids", [])
    }
    documents.update(
        _document_key(evidence.get("source_doc_id"))
        for evidence in record.get("gold_evidence", [])
        if isinstance(evidence, dict) and evidence.get("source_doc_id")
    )
    return sorted(value for value in documents if value)


def _retrieved(candidates, gold_ids, gold_documents):
    ordered = sorted(candidates, key=lambda item: int(item.get("rank") or 10**9))
    rows = []
    for rank, candidate in enumerate(ordered[:MAX_K], start=1):
        chunk_id = str(candidate.get("chunk_id", ""))
        document = _chunk_document(candidate)
        rows.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "score": float(candidate.get("score", 0.0)),
            "source": candidate.get("source", ""),
            "source_file": candidate.get("source_file", ""),
            "doc_id": candidate.get("doc_id", ""),
            "parent_doc_id": candidate.get("parent_doc_id", ""),
            "chunk_text": str(candidate.get("chunk_text_full") or candidate.get("chunk_text") or "")[:500],
            "is_golden_chunk": chunk_id in gold_ids,
            "is_golden_doc": document in gold_documents if document else False,
        })
    return rows


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(parser, default_top_k=MAX_K, default_semantic_pool=100, default_dimension_pool=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    ks = parse_ks(args.ks)
    if max(ks) > MAX_K:
        parser.error(f"--ks cannot exceed the saved result depth {MAX_K}")
    if args.top_k < MAX_K:
        parser.error("--top-k must be at least 20 so every configured metric cutoff is available")
    if args.semantic_pool < args.top_k or args.dimension_pool < args.top_k:
        parser.error("--semantic-pool and --dimension-pool must be at least --top-k")

    with LiveRetrievalRuntime(args) as runtime:
        output_dir = args.output_root.expanduser().resolve() / "latest"
        detail_rows = {route: [] for route in ROUTES}
        for position, record in enumerate(runtime.records, start=1):
            question = record["question"]
            parts = runtime.query_parts(question)
            retrieval_query = " | ".join(parts)
            result = runtime.retrieve(
                question,
                parts,
                top_k=args.top_k,
                semantic_pool=args.semantic_pool,
                dimension_pool=args.dimension_pool,
            )
            gold_ids = set(map(str, record.get("gold_chunk_ids") or []))
            gold_documents = _gold_documents(record, runtime.chunks_by_id)
            for route, (result_key, label) in ROUTES.items():
                candidates = result.get(result_key) or []
                detail_rows[route].append({
                    "run_id": runtime.run_id,
                    "id": record["id"],
                    "query": question,
                    "retrieval_query": retrieval_query,
                    "subqueries": parts,
                    "route": route,
                    "route_label": label,
                    "strategy": (
                        result.get("fusion_strategy") if route == "hybrid"
                        else runtime.retriever.vector_retriever.strategy_file if route == "vector"
                        else "dimension_retrieval"
                    ),
                    "candidate_count": len(candidates),
                    "answerable": record.get("answerable", True),
                    "gold_mapped": bool(gold_ids),
                    "gold_resolution": record.get("gold_resolution", {}),
                    "gold_chunk_ids": sorted(gold_ids),
                    "gold_documents": gold_documents,
                    "query_analysis": result.get("query_analysis", {}),
                    "retrieved": _retrieved(candidates, gold_ids, gold_documents),
                })
            if position % 25 == 0 or position == len(runtime.records):
                print(f"live retrieval: {position}/{len(runtime.records)}", flush=True)

        metadata = runtime.metadata()
        metadata.update({
            "experiment": "00_main_experiment",
            "output_dir": str(output_dir),
            "metric_ks": list(ks),
            "saved_top_k": MAX_K,
            "fusion_strategy": (
                detail_rows["hybrid"][0].get("strategy") if detail_rows["hybrid"] else None
            ),
            "files": DETAIL_FILES,
        })
        manifest = {
            **metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with stage_latest(output_dir) as staging:
            for route, filename in DETAIL_FILES.items():
                _write_jsonl(staging / filename, detail_rows[route])
            (staging / "details_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
