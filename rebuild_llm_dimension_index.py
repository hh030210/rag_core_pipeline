"""Full, isolated LLM re-tagging and index rebuild for an existing chunk run.

The source chunks and validated hierarchical schema are reused verbatim.  Only
document-side dimension labels and retrieval entities are regenerated, so this
script makes a directly comparable dimension-retrieval experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rag_core.dimension_stage import run_dimensions
from rag_core.embedding import EmbeddingModel
from rag_core.settings import Settings
from rag_core.storage import VectorStore


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM full dimension-index rebuild")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8910/v1")
    parser.add_argument("--llm-model", default="DeepSeek-V3.2")
    parser.add_argument("--llm-interval", type=float, default=0.15)
    parser.add_argument("--dimensions-per-request", type=int, default=15)
    parser.add_argument("--tag-workers", type=int, default=1,
                        help="document-level LLM extraction workers; default is conservative single-threading")
    args = parser.parse_args()

    source, target = args.source_run, args.output_run
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output run must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    chunk_path = source / "chunks.json"
    if not chunk_path.exists():
        chunk_path = source / "chunks" / "chunks.json"
    schema_path = source / "V_core_v2.json"
    records = _load_json(chunk_path)
    if not records:
        raise ValueError("source run contains no chunks")

    # The SDU local gateway is OpenAI-compatible but intentionally does not
    # require a real API key; a nonempty sentinel satisfies the common client.
    settings = Settings.from_env(
        target,
        backend="local",
        model_path=args.model_path,
        embedding_device=args.device,
        vector_dim=1024,
        llm_api_key="sdu-local",
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_openai_compat=True,
        llm_interval=args.llm_interval,
        max_dimensions_per_request=args.dimensions_per_request,
        mock=False,
    )
    settings.apply_runtime_environment()
    # Read by TagGeneratorV2.  Kept as an explicit process setting so normal
    # pipeline runs retain their conservative sequential behavior by default.
    import os
    os.environ["TAG_EXTRACTION_WORKERS"] = str(max(1, args.tag_workers))
    os.environ["TAG_EXTRACTION_PROGRESS_PATH"] = str(target / "tagging_progress.json")
    dimension_result = run_dimensions(records, settings, schema_path=str(schema_path))

    embedding = EmbeddingModel(args.model_path, args.device, dimension=1024, mock=False)
    store = VectorStore(
        run_dir=target, backend="local", qdrant_url=settings.qdrant_url,
        collection=settings.collection, vector_dim=settings.vector_dim,
    )
    store_result = store.upsert(
        records, dimension_result["tags"], dimension_result["schema"], embedding,
    )
    for name in ("merged_7_rag_test_set_filtered.json", "dimension_label_ontology.json"):
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)

    manifest = {
        "source_run": str(source),
        "rebuild": "llm_dimension_and_entity_retag",
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "chunk_count": len(records),
        "schema_path": str(schema_path),
        "store": store_result,
    }
    (target / "llm_rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
