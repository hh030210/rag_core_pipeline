#!/usr/bin/env python3
"""Run dimension extraction and vector ingestion for prepared DuRetrieval chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.dimension_stage import run_dimensions
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
    parser.add_argument("--resume-dimensions", action="store_true")
    args = parser.parse_args()

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

    tags_path = args.run_dir / "tags_output_v2.json"
    schema_path = args.run_dir / "V_core_v2.json"
    if args.resume_dimensions and tags_path.exists() and schema_path.exists():
        dimension_result = {
            "tags": json.loads(tags_path.read_text(encoding="utf-8")),
            "schema": json.loads(schema_path.read_text(encoding="utf-8")),
        }
        print("[维度] 使用已有 tags_output_v2.json 和 V_core_v2.json", flush=True)
    else:
        dimension_result = run_dimensions(records, settings)

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
            "embedding_model": settings.model_path,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
        },
        "artifacts": {
            "chunks": str(args.run_dir / "chunks.json"),
            "schema": str(args.run_dir / "V_core_v2.json"),
            "tags": str(args.run_dir / "tags_output_v2.json"),
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
