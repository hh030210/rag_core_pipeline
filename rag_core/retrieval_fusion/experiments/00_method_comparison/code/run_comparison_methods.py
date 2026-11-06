#!/usr/bin/env python3
"""Run all fusion strategies on fresh routes retrieved from the live store."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))

from rag_core.evaluation import route_metrics  # noqa: E402
from rag_core.retrieval_fusion.experiments.common.live_runtime import (  # noqa: E402
    LiveRetrievalRuntime,
    add_live_arguments,
    parse_ks,
)

HERE = Path(__file__).resolve().parents[1]
CODE_DIR = HERE / "code"
DEFAULT_OUTPUT_DIR = HERE / "output" / "runs"
METHODS = tuple(sorted(
    path.stem for path in CODE_DIR.glob("*.py")
    if path.name != "run_comparison_methods.py"
))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _public_hit(hit, rank, gold_ids, gold_documents):
    document = _chunk_document(hit)
    return {
        "chunk_id": str(hit.get("chunk_id", "")),
        "rank": rank,
        "score": float(hit.get("score", 0.0)),
        "source": str(hit.get("source", "")),
        "source_file": str(hit.get("source_file", "")),
        "doc_id": str(hit.get("doc_id", "")),
        "parent_doc_id": str(hit.get("parent_doc_id", "")),
        "is_golden_chunk": str(hit.get("chunk_id", "")) in gold_ids,
        "is_golden_doc": document in gold_documents if document else False,
    }


def _average_precision(hits, gold_ids, k):
    gold_ids = set(map(str, gold_ids))
    if not gold_ids:
        return 0.0
    found = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if str(hit.get("chunk_id", "")) in gold_ids:
            found += 1
            precision_sum += found / rank
    return precision_sum / min(len(gold_ids), k)


def _summarize(rows, ks):
    query_count = len(rows)
    summary = {
        "query_count": query_count,
        "mapped_gold_query_count": sum(bool(row["gold_chunk_ids"]) for row in rows),
        "unmapped_gold_query_count": sum(not bool(row["gold_chunk_ids"]) for row in rows),
        "metrics_by_k": {},
    }
    for k in ks:
        per_query = [route_metrics(row["results"], row["gold_chunk_ids"], (k,)) for row in rows]
        gold_count = sum(len(set(row["gold_chunk_ids"])) for row in rows)
        found_count = sum(
            len(set(hit["chunk_id"] for hit in row["results"][:k]) & set(row["gold_chunk_ids"]))
            for row in rows
        )
        retrieved_count = sum(len(row["results"][:k]) for row in rows)
        doc_relevant = sum(
            int(hit["is_golden_doc"])
            for row in rows for hit in row["results"][:k]
        )
        summary["metrics_by_k"][str(k)] = {
            "hit_at_k": sum(bool(item[f"hit_at_{k}"]) for item in per_query) / query_count if query_count else 0.0,
            "recall_at_k_macro": sum(item[f"gold_recall_at_{k}"] for item in per_query) / query_count if query_count else 0.0,
            "recall_at_k_micro": found_count / gold_count if gold_count else 0.0,
            "mrr_at_k": sum(item[f"rr_at_{k}"] for item in per_query) / query_count if query_count else 0.0,
            "map_at_k": sum(_average_precision(row["results"], row["gold_chunk_ids"], k) for row in rows) / query_count if query_count else 0.0,
            "ndcg_at_k": sum(item[f"ndcg_at_{k}"] for item in per_query) / query_count if query_count else 0.0,
            "doc_precision_at_k_micro": doc_relevant / retrieved_count if retrieved_count else 0.0,
            "gold_chunk_count": gold_count,
            "returned_chunk_count": retrieved_count,
            "gold_document_chunk_count": doc_relevant,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(parser, default_top_k=20, default_semantic_pool=100, default_dimension_pool=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    ks = parse_ks(args.ks)
    if max(ks) > args.top_k:
        parser.error(f"--top-k must be at least max(--ks), which is {max(ks)}")
    if args.semantic_pool < args.top_k or args.dimension_pool < args.top_k:
        parser.error("candidate pools must be at least --top-k")

    modules = {
        method: import_module(
            "rag_core.retrieval_fusion.experiments.00_method_comparison.code." + method
        )
        for method in args.methods
    }
    with LiveRetrievalRuntime(args) as runtime:
        run_id = runtime.run_id
        run_dir = args.output_dir.expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_metadata = runtime.metadata()
        run_metadata.update({
            "experiment": "fusion_method_comparison",
            "methods": list(args.methods),
            "metric_ks": list(ks),
            "output_dir": str(run_dir),
        })
        rows_by_method = {method: [] for method in args.methods}
        total = len(runtime.records)
        started = time.perf_counter()

        for index, record in enumerate(runtime.records, start=1):
            question = record["question"]
            parts = runtime.query_parts(question)
            retrieval_query = " | ".join(parts)
            # search() freshly encodes the query and executes both semantic and
            # dimension retrieval against the configured current collection.
            live_result = runtime.retrieve(
                question,
                parts,
                top_k=args.top_k,
                semantic_pool=args.semantic_pool,
                dimension_pool=args.dimension_pool,
            )
            query_analysis = live_result.get("query_analysis") or {}
            semantic = live_result.get("semantic_candidates") or []
            dimension = live_result.get("dimension_candidates") or []
            gold_ids = set(map(str, record.get("gold_chunk_ids") or []))
            gold_documents = _gold_documents(record, runtime.chunks_by_id)

            for method, module in modules.items():
                fusion_result = module.fuse(
                    copy.deepcopy(semantic),
                    copy.deepcopy(dimension),
                    top_k=args.top_k,
                    query=retrieval_query,
                    original_query=question,
                    query_analysis=copy.deepcopy(query_analysis),
                )
                candidates = fusion_result.get("fusion_candidates") or fusion_result.get("fusion_results") or []
                results = [
                    _public_hit(candidate, rank, gold_ids, gold_documents)
                    for rank, candidate in enumerate(candidates[:max(ks)], start=1)
                ]
                metrics = route_metrics(candidates, gold_ids, ks)
                for k in ks:
                    metrics[f"average_precision_at_{k}"] = _average_precision(candidates, gold_ids, k)
                rows_by_method[method].append({
                    "id": record["id"],
                    "query": question,
                    "retrieval_query": retrieval_query,
                    "subqueries": parts,
                    "query_analysis": query_analysis,
                    "answerable": record.get("answerable", True),
                    "gold_mapped": bool(gold_ids),
                    "gold_resolution": record.get("gold_resolution", {}),
                    "gold_chunk_ids": sorted(gold_ids),
                    "gold_documents": gold_documents,
                    "semantic_candidate_count": len(semantic),
                    "dimension_candidate_count": len(dimension),
                    "fusion_strategy": fusion_result.get("fusion_strategy", {}),
                    "results": results,
                    "metrics": metrics,
                })
            if index % 25 == 0 or index == total:
                elapsed = round(time.perf_counter() - started, 1)
                print(f"live fusion retrieval: {index}/{total} ({elapsed}s)", flush=True)

        method_outputs = {}
        for method, rows in rows_by_method.items():
            method_dir = run_dir / method
            _write_jsonl(method_dir / "queries.jsonl", rows)
            _write_json(method_dir / "summary.json", {
                "metadata": {**run_metadata, "method": method},
                "metrics": _summarize(rows, ks),
            })
            method_outputs[method] = {
                "detail_file": f"{method}/queries.jsonl",
                "summary_file": f"{method}/summary.json",
                "query_count": len(rows),
            }

        manifest = {
            **run_metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method_outputs": method_outputs,
        }
        _write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
