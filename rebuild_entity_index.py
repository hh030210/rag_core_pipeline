"""Rebuild a local index with entity sidecars without touching the source run.

Use this for migrations: it preserves existing dimension tags, derives generic
landmark entities for every chunk, writes ``entity_index_v2.json``, and creates
new local vector points in a separate run directory.  A later LLM re-tagging
run can replace the derived sidecars with structured ``entity_mentions`` while
using the same index and retrieval interfaces.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from code_jyx.index_builder import IndexBuilderV2
from rag_core.embedding import EmbeddingModel
from rag_core.entity_registry import extract_landmark_mentions
from rag_core.settings import Settings
from rag_core.storage import VectorStore


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = args.source_run
    target = args.output_run
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"输出目录必须为空：{target}")
    target.mkdir(parents=True, exist_ok=True)
    schema = load_json(source / "V_core_v2.json")
    chunks_path = source / "chunks.json"
    if not chunks_path.exists():
        chunks_path = source / "chunks" / "chunks.json"
    records = load_json(chunks_path)
    tags = load_json(source / "tags_output_v2.json")
    record_by_id = {str(item["chunk_id"]): item for item in records}

    entity_total = 0
    for chunk_id, document in (tags.get("documents", {}) or {}).items():
        record = record_by_id.get(str(chunk_id), {})
        mentions = extract_landmark_mentions(
            record.get("doc_title", ""), record.get("chunk_gen_title", ""),
            record.get("chunk_text_full", record.get("doc_text", "")),
        )
        document["entity_mentions"] = [
            {"name": name, "type": "auto_landmark", "aliases": [], "evidence": name}
            for name in mentions
        ]
        entity_total += len(mentions)

    (target / "V_core_v2.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "chunks.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "tags_output_v2.json").write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding="utf-8")

    index = IndexBuilderV2(schema, output_dir=target).build(tags)
    index_files = IndexBuilderV2(schema, output_dir=target).save(index, target)
    embeddings = EmbeddingModel(args.model_path, args.device, dimension=1024, mock=False)
    settings = Settings.from_env(target, backend="local", model_path=args.model_path,
                                 embedding_device=args.device, vector_dim=1024)
    store = VectorStore(run_dir=target, backend="local", qdrant_url=settings.qdrant_url,
                        collection=settings.collection, vector_dim=settings.vector_dim)
    store_result = store.upsert(records, tags, schema, embeddings)
    for name in ("merged_7_rag_test_set_filtered.json", "dimension_label_ontology.json"):
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, target / name)
    manifest = {
        "source_run": str(source), "rebuild": "entity_sidecar_offline",
        "chunk_count": len(records), "derived_entity_mentions": entity_total,
        "index_files": index_files, "store": store_result,
    }
    (target / "entity_rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
