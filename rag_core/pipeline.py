"""完整非交互流水线：分片 → 维度抽取 → Qdrant/本地入库 → Prompt 优化 → 问答。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .chunk_stage import run_chunking
from .dimension_stage import run_dimensions
from .embedding import EmbeddingModel
from .prompt_module import run_prompt_module
from .settings import Settings
from .storage import VectorStore


def run_ingest(settings: Settings, schema_path: str = "") -> Dict[str, Any]:
    settings.apply_runtime_environment()
    if not settings.input_path:
        raise ValueError("ingest/run 必须指定 --input")
    settings.ensure_dirs()
    chunk_result = run_chunking(settings.input_path, settings.run_dir / "chunks", settings)
    dimension_result = run_dimensions(chunk_result["records"], settings, schema_path=schema_path)
    embeddings = EmbeddingModel(settings.model_path, settings.embedding_device,
                                settings.vector_dim, settings.mock)
    store = VectorStore(run_dir=settings.run_dir, backend="local" if settings.mock else settings.backend,
                        qdrant_url=settings.qdrant_url, collection=settings.collection,
                        vector_dim=settings.vector_dim)
    store_result = store.upsert(chunk_result["records"], dimension_result["tags"],
                                dimension_result["schema"], embeddings)
    manifest = {
        "project": "rag_core_pipeline", "schema_version": "2.0",
        "backend": store_result["backend"], "collection": store_result["collection"],
        "input": str(settings.input_path), "run_dir": str(settings.run_dir),
        "experiment_config": {
            "semantic_pool": int(getattr(settings, "semantic_pool", 20)),
            "dimension_pool": int(getattr(settings, "dimension_pool", 100)),
            "top_k": int(settings.top_k),
            "dim_alpha": float(settings.dim_alpha),
            "embedding_model": settings.model_path or "configured_default",
            "denoise_method": settings.denoise_method,
        },
        "chunk_summary": chunk_result["summary"], "store": store_result,
        "artifacts": {
            "chunks": str(settings.run_dir / "chunks.json"),
            "schema": str(settings.run_dir / "V_core_v2.json"),
            "tags": str(settings.run_dir / "tags_output_v2.json"),
            "inverted_index": str(settings.run_dir / "inverted_index_v2.json"),
        },
    }
    (settings.run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def optimize_prompt(settings: Settings, examples_path: str, iterations: int) -> Dict[str, Any]:
    return run_prompt_module(
        examples_path,
        settings.run_dir / "optimized_prompt.json",
        settings,
        iterations=iterations,
    )
