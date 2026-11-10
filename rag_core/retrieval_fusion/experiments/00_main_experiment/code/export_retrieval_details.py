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
    DEFAULT_RUN_DIR,
    add_live_arguments,
    parse_ks,
)

from rag_core.retrieval_fusion.experiments.common.dataset_profiles import (  # noqa: E402
    MODEL_PATH, PROFILES, profile_output_root,
)
from rag_core.retrieval_fusion.experiments.common.latest_output import stage_latest  # noqa: E402
from rag_core.retrieval_fusion.experiments.common.scenic_query_replay import (  # noqa: E402
    ScenicRetrievalRuntime,
)

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


def build_parser(*, output_root=DEFAULT_OUTPUT_ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(parser, default_top_k=MAX_K, default_query_parser_mode="llm")
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument(
        "--replay-query-analysis", action="store_true",
        help="Use archived LLM analyses without calling the API",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--dataset-profile", choices=PROFILES, default="scenic",
                        help="Select the dataset and matching index (default: scenic)")
    parser.set_defaults(run_dir=None, query_parser_mode=None)
    args = parser.parse_args(argv)
    profile = PROFILES[args.dataset_profile]
    use_profile_index = args.run_dir is None and profile.run_dir is not None
    args.run_dir = args.run_dir or profile.run_dir or DEFAULT_RUN_DIR
    args.dataset = args.dataset or profile.dataset
    args.query_parser_mode = args.query_parser_mode or profile.parser_mode
    if use_profile_index:
        args.backend = args.backend or "qdrant"
        args.collection = args.collection or profile.collection
    if profile.run_dir is not None:
        args.model_path = args.model_path or MODEL_PATH
    if args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = profile_output_root(DEFAULT_OUTPUT_ROOT, args.dataset_profile)
    return export(parser, args)


def export(parser, args, *, runtime_class=ScenicRetrievalRuntime, experiment=EXPERIMENT_DIR.name):
    if args.replay_query_analysis and (args.query_parser_mode != "llm" or args.query_expansion):
        parser.error("--replay-query-analysis requires llm parser mode and query expansion disabled")
    ks = parse_ks(args.ks)
    if max(ks) > MAX_K:
        parser.error(f"--ks cannot exceed the saved result depth {MAX_K}")
    if args.top_k < MAX_K:
        parser.error("--top-k must be at least 20 so every configured metric cutoff is available")
    if args.semantic_pool < args.top_k or args.dimension_pool < args.top_k:
        parser.error("--semantic-pool and --dimension-pool must be at least --top-k")

    with runtime_class(args) as runtime:
        if getattr(args, "dataset_profile", None) == "duretrieval_mteb" and not args.query_expansion:
            print("Preparing corpus frequencies for DuRetrieval queries", flush=True)
            runtime.retriever.prepare_query_anchor_frequencies(record["question"] for record in runtime.records)
        output_dir = args.output_root.expanduser().resolve() / "latest"
        detail_rows = {route: [] for route in ROUTES}
        for position, record in enumerate(runtime.records, start=1):
            question = record["question"]
            parts = runtime.query_parts(question)
            retrieval_query = " | ".join(parts)
            result = runtime.retrieve_record(record, parts)
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
                        else result.get("dimension_strategy", "dimension_retrieval")
                    ),
                    "candidate_count": len(candidates),
                    **({"dimension_retrieval_query": result["dimension_retrieval_query"]}
                       if route == "dimension" and "dimension_retrieval_query" in result else {}),
                    "answerable": record.get("answerable", True),
                    "gold_mapped": bool(gold_ids),
                    "gold_resolution": record.get("gold_resolution", {}),
                    "gold_chunk_ids": sorted(gold_ids),
                    "gold_documents": gold_documents,
                    "query_parser_source": runtime.retriever.parser_source,
                    "query_expansion_source": runtime.expansion_source,
                    "query_analysis": result.get("query_analysis", {}),
                    "retrieved": _retrieved(candidates, gold_ids, gold_documents),
                })
            if position % 25 == 0 or position == len(runtime.records):
                print(f"live retrieval: {position}/{len(runtime.records)}", flush=True)

        metadata = runtime.metadata()
        metadata.update({
            "experiment": experiment,
            "dataset_profile": getattr(args, "dataset_profile", "scenic"),
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
        with stage_latest(
            output_dir, experiment=experiment, dataset_profile=getattr(args, "dataset_profile", None),
        ) as staging:
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
