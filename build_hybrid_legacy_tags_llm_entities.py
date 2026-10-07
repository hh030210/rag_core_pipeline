"""Build a comparable hybrid run: legacy dimension tags plus LLM entities.

The source run remains the authority for every dimension label and hierarchy.
Only structured entity_mentions are copied from an independently LLM-tagged
run, then entity postings and local payloads are rebuilt.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from code_jyx.index_builder import IndexBuilderV2
from rag_core.embedding import EmbeddingModel
from rag_core.settings import Settings
from rag_core.storage import VectorStore


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-run", required=True, type=Path)
    parser.add_argument("--entity-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    legacy, entity_run, target = args.legacy_run, args.entity_run, args.output_run
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output run must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    chunk_path = legacy / "chunks.json"
    if not chunk_path.exists():
        chunk_path = legacy / "chunks" / "chunks.json"
    schema = load_json(legacy / "V_core_v2.json")
    records = load_json(chunk_path)
    legacy_tags = load_json(legacy / "tags_output_v2.json")
    entity_tags = load_json(entity_run / "tags_output_v2.json").get("documents", {})

    copied_entities = 0
    zero_entity_docs = 0
    for doc_id, document in legacy_tags.get("documents", {}).items():
        candidate = entity_tags.get(str(doc_id), {})
        mentions = candidate.get("entity_mentions", []) if isinstance(candidate, dict) else []
        mentions = [item for item in mentions if isinstance(item, dict) and str(item.get("name", "")).strip()]
        document["entity_mentions"] = mentions
        copied_entities += len(mentions)
        zero_entity_docs += int(not mentions)

    (target / "V_core_v2.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "chunks.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "tags_output_v2.json").write_text(json.dumps(legacy_tags, ensure_ascii=False, indent=2), encoding="utf-8")
    index = IndexBuilderV2(schema, output_dir=target).build(legacy_tags)
    index_files = IndexBuilderV2(schema, output_dir=target).save(index, target)

    embeddings = EmbeddingModel(args.model_path, args.device, dimension=1024, mock=False)
    settings = Settings.from_env(target, backend="local", model_path=args.model_path,
                                 embedding_device=args.device, vector_dim=1024)
    store = VectorStore(run_dir=target, backend="local", qdrant_url=settings.qdrant_url,
                        collection=settings.collection, vector_dim=settings.vector_dim)
    store_result = store.upsert(records, legacy_tags, schema, embeddings)
    for name in ("merged_7_rag_test_set_filtered.json", "dimension_label_ontology.json"):
        path = legacy / name
        if path.exists():
            shutil.copy2(path, target / name)
    baseline_results = legacy / "evaluation" / "results.jsonl"
    if baseline_results.exists():
        (target / "evaluation").mkdir(exist_ok=True)
        shutil.copy2(baseline_results, target / "evaluation" / "results.jsonl")

    manifest = {
        "legacy_dimension_tag_source": str(legacy),
        "llm_entity_source": str(entity_run),
        "chunk_count": len(records),
        "copied_llm_entities": copied_entities,
        "zero_entity_docs": zero_entity_docs,
        "index_files": index_files,
        "store": store_result,
    }
    (target / "hybrid_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
