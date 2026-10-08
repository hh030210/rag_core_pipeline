#!/usr/bin/env python3
"""Run dimension extraction and vector ingestion for prepared DuRetrieval chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.dimension_stage import (
    build_inverted_index,
    default_schema,
    induce_dataset_schema,
    run_dimensions,
)
from code_jyx.index_builder import IndexBuilderV2
from rag_core.embedding import EmbeddingModel
from rag_core.settings import Settings
from rag_core.storage import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backend", default="qdrant", choices=("qdrant", "local"))
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dimension-batch-size", type=int, default=None)
    parser.add_argument(
        "--dimension-mode",
        choices=("llm", "empty"),
        default="llm",
        help="LLM labels or explicit no-dimension mode for datasets without facet annotations",
    )
    parser.add_argument("--resume-dimensions", action="store_true")
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="induce and save a fresh dataset schema, then stop before tag extraction and ingestion",
    )
    parser.add_argument(
        "--strict-dataset-schema",
        action="store_true",
        help="freshly induce dimensions from this dataset's chunks; fail instead of using a fallback schema",
    )
    parser.add_argument(
        "--resume-tag-extraction",
        action="store_true",
        help="resume from tags_checkpoint.jsonl in this run directory",
    )
    args = parser.parse_args()
    if args.strict_dataset_schema and (args.resume_dimensions or args.dimension_mode == "empty"):
        parser.error("--strict-dataset-schema 不能与 --resume-dimensions 或 --dimension-mode empty 同时使用")
    if args.schema_only and not args.strict_dataset_schema:
        parser.error("--schema-only 必须与 --strict-dataset-schema 一起使用")
    if args.schema_only and args.schema_path:
        parser.error("--schema-only 不能再指定 --schema-path")
    if args.schema_path and args.resume_dimensions:
        parser.error("--schema-path 不能与 --resume-dimensions 同时使用")
    if args.resume_tag_extraction and (
        not args.strict_dataset_schema or not args.schema_path or args.resume_dimensions or args.schema_only
    ):
        parser.error("--resume-tag-extraction 要求 strict schema + --schema-path，且不能处于 schema-only/resume-dimensions")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.prepared_root / "chunks.json"
    records = json.loads(chunks_path.read_text(encoding="utf-8"))
    (args.run_dir / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    settings = Settings.from_env(
        args.run_dir,
        backend=args.backend,
        qdrant_url=args.qdrant_url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        collection=args.collection,
        model_path=args.model_path or os.getenv("BGE_MODEL_PATH", ""),
    )
    if args.dimension_batch_size is not None:
        settings.dimension_batch_size = max(1, args.dimension_batch_size)
    settings.ensure_dirs()

    if args.schema_only:
        schema = induce_dataset_schema(records, settings)
        for filename in ("V_cand_v2.json", "V_core_v2.json"):
            (args.run_dir / filename).write_text(
                json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        schema_manifest = {
            "dataset": "DuRetrieval/MTEB dev",
            "prepared_root": str(args.prepared_root),
            "run_dir": str(args.run_dir),
            "generation_method": schema["generation_method"],
            "source_record_count": schema["schema_source_record_count"],
            "source_fingerprint": schema["schema_source_fingerprint"],
            "parent_count": sum(node.get("level") == 1 for node in schema["dimensions"]),
            "leaf_count": sum(node.get("level") == 2 for node in schema["dimensions"]),
            "schema_file": str(args.run_dir / "V_core_v2.json"),
        }
        (args.run_dir / "schema_generation_manifest.json").write_text(
            json.dumps(schema_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(schema_manifest, ensure_ascii=False, indent=2), flush=True)
        return 0

    tags_path = args.run_dir / "tags_output_v2.json"
    schema_path = args.run_dir / "V_core_v2.json"
    if args.dimension_mode == "empty":
        schema = default_schema()
        empty_documents = {
            str(record["doc_id"]): {
                "tags": {},
                "tag_details": {},
                "dimension_paths": [],
                "schema_version": "2.0",
            }
            for record in records
        }
        tag_output = {"schema_version": "2.0", "documents": empty_documents}
        index = build_inverted_index(tag_output, schema)
        (args.run_dir / "V_cand_v2.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        tags_path.write_text(json.dumps(tag_output, ensure_ascii=False), encoding="utf-8")
        IndexBuilderV2(schema, output_dir=args.run_dir).save(index, args.run_dir)
        dimension_result = {"tags": tag_output, "schema": schema}
        print("[维度] 当前数据集没有维度标注，本次显式使用 empty dimension mode", flush=True)
    elif args.resume_dimensions and tags_path.exists() and schema_path.exists():
        dimension_result = {
            "tags": json.loads(tags_path.read_text(encoding="utf-8")),
            "schema": json.loads(schema_path.read_text(encoding="utf-8")),
        }
        print("[维度] 使用已有 tags_output_v2.json 和 V_core_v2.json", flush=True)
    else:
        dimension_result = run_dimensions(
            records,
            settings,
            schema_path=args.schema_path or "",
            strict_dataset_schema=args.strict_dataset_schema,
            resume_tags=args.resume_tag_extraction,
        )

    embeddings = EmbeddingModel(
        settings.model_path,
        settings.embedding_device,
        settings.vector_dim,
        settings.mock,
    )
    store = VectorStore(
        run_dir=args.run_dir,
        backend=settings.backend,
        qdrant_url=settings.qdrant_url,
        collection=settings.collection,
        vector_dim=settings.vector_dim,
    )
    store_result = store.upsert(
        records,
        dimension_result["tags"],
        dimension_result["schema"],
        embeddings,
    )
    manifest = {
        "project": "rag_core_pipeline",
        "dataset": "DuRetrieval/MTEB dev",
        "prepared_root": str(args.prepared_root),
        "run_dir": str(args.run_dir),
        "backend": store_result["backend"],
        "collection": store_result["collection"],
        "point_count": store_result["point_count"],
        "experiment_config": {
            "semantic_pool": settings.semantic_pool,
            "dimension_pool": settings.dimension_pool,
            "top_k": settings.top_k,
            "dim_alpha": settings.dim_alpha,
            "dimension_batch_size": settings.dimension_batch_size,
            "dimension_workers": settings.dimension_workers,
            "dimension_mode": args.dimension_mode,
            "dataset_schema_mode": (
                "fresh_llm_induced_strict" if args.strict_dataset_schema else "standard"
            ),
            "schema_generation_method": dimension_result["schema"].get("generation_method", "unknown"),
            "schema_source_record_count": dimension_result["schema"].get(
                "schema_source_record_count", len(records)
            ),
            "schema_parent_count": sum(
                node.get("level") == 1 for node in dimension_result["schema"].get("dimensions", [])
            ),
            "schema_leaf_count": sum(
                node.get("level") == 2 for node in dimension_result["schema"].get("dimensions", [])
            ),
            "embedding_model": settings.model_path,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
        },
        "artifacts": {
            "chunks": str(args.run_dir / "chunks.json"),
            "schema": str(args.run_dir / "V_core_v2.json"),
            "tags": str(args.run_dir / "tags_output_v2.json"),
            "tags_checkpoint": str(args.run_dir / "tags_checkpoint.jsonl"),
            "inverted_index": str(args.run_dir / "inverted_index_v2.json"),
        },
    }
    (args.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
