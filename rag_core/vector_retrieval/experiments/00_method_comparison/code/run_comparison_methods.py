#!/usr/bin/env python3
"""Run every vector strategy against the live collection and raw QA dataset."""
from __future__ import annotations

import argparse
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


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


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


def _summarize(method_rows, ks):
    all_count = len(method_rows)
    mapped_count = sum(bool(row["gold_chunk_ids"]) for row in method_rows)
    result = {
        "query_count": all_count,
        "mapped_gold_query_count": mapped_count,
        "unmapped_gold_query_count": all_count - mapped_count,
        "metrics_by_k": {},
    }
    for k in ks:
        query_metrics = [route_metrics(row["results"], row["gold_chunk_ids"], (k,)) for row in method_rows]
        gold_count = sum(len(set(row["gold_chunk_ids"])) for row in method_rows)
        found_count = sum(metric[f"gold_recall_at_{k}"] * len(set(row["gold_chunk_ids"]))
                          for metric, row in zip(query_metrics, method_rows))
        retrieved_count = sum(min(k, len(row["results"])) for row in method_rows)
        golden_doc_hits = sum(
            bool(hit["is_golden_doc"])
            for row in method_rows for hit in row["results"][:k]
        )
        result["metrics_by_k"][str(k)] = {
            "hit_at_k": sum(bool(metric[f"hit_at_{k}"]) for metric in query_metrics) / all_count if all_count else 0.0,
            "recall_at_k_macro": sum(metric[f"gold_recall_at_{k}"] for metric in query_metrics) / all_count if all_count else 0.0,
            "recall_at_k_micro": found_count / gold_count if gold_count else 0.0,
            "mrr_at_k": sum(metric[f"rr_at_{k}"] for metric in query_metrics) / all_count if all_count else 0.0,
            "map_at_k": sum(_average_precision(row["results"], row["gold_chunk_ids"], k) for row in method_rows) / all_count if all_count else 0.0,
            "ndcg_at_k": sum(metric[f"ndcg_at_{k}"] for metric in query_metrics) / all_count if all_count else 0.0,
            "doc_precision_at_k_micro": golden_doc_hits / retrieved_count if retrieved_count else 0.0,
            "gold_chunk_count": gold_count,
            "returned_chunk_count": retrieved_count,
            "gold_document_chunk_count": golden_doc_hits,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(
        parser,
        default_top_k=20,
        default_semantic_pool=20,
        default_dimension_pool=20,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    ks = parse_ks(args.ks)
    pool = int(args.semantic_pool)
    if pool < max(ks):
        parser.error(f"--semantic-pool must be at least max(--ks), which is {max(ks)}")
    if args.top_k > pool:
        parser.error("--top-k must not exceed --semantic-pool")

    args.top_k = min(args.top_k, pool)
    args.dimension_pool = max(args.dimension_pool, args.top_k)
    with LiveRetrievalRuntime(args) as runtime:
        run_id = runtime.run_id
        run_dir = args.output_dir.expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_metadata = runtime.metadata()
        run_metadata.update({
            "experiment": "vector_method_comparison",
            "methods": list(args.methods),
            "metric_ks": list(ks),
            "candidate_pool": pool,
            "output_dir": str(run_dir),
        })
        method_rows = {method: [] for method in args.methods}
        total = len(runtime.records)
        started = time.perf_counter()

        for index, record in enumerate(runtime.records, start=1):
            question = record["question"]
            parts = runtime.query_parts(question)
            retrieval_query = " | ".join(parts)
            query_analysis = runtime.parse_query(retrieval_query)
            spot_names = query_analysis.get("spot_names") or []
            gold_ids = set(map(str, record.get("gold_chunk_ids") or []))
            gold_documents = _gold_documents(record, runtime.chunks_by_id)
            for method in args.methods:
                runtime.set_vector_strategy(method + ".py")
                vector_retriever = runtime.retriever.vector_retriever
                hits = vector_retriever.search(
                    parts,
                    pool,
                    original_query=question,
                    spot_names=spot_names,
                )
                selected = [
                    _public_hit(hit, rank, gold_ids, gold_documents)
                    for rank, hit in enumerate(hits[:max(ks)], start=1)
                ]
                metrics = route_metrics(hits, gold_ids, ks)
                for k in ks:
                    metrics[f"average_precision_at_{k}"] = _average_precision(hits, gold_ids, k)
                method_rows[method].append({
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
                    "results": selected,
                    "metrics": metrics,
                })
            if index % 25 == 0 or index == total:
                elapsed = round(time.perf_counter() - started, 1)
                print(f"live vector retrieval: {index}/{total} ({elapsed}s)", flush=True)

        methods_manifest = {}
        for method, rows in method_rows.items():
            method_dir = run_dir / method
            _write_jsonl(method_dir / "queries.jsonl", rows)
            summary = _summarize(rows, ks)
            _write_json(method_dir / "summary.json", {
                "metadata": {**run_metadata, "method": method},
                "metrics": summary,
            })
            methods_manifest[method] = {
                "detail_file": f"{method}/queries.jsonl",
                "summary_file": f"{method}/summary.json",
                "query_count": len(rows),
            }

        manifest = {
            **run_metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method_outputs": methods_manifest,
        }
        _write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
