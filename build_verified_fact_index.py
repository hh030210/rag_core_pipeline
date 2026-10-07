"""Create an evidence-grounded fact sidecar without altering dimension tags.

The script is resumable: ``facts_output_v1.json`` is updated after every chunk
and can safely be reused after a gateway restart.  It copies the selected
legacy/hybrid run first, preserving its dimension index as the primary index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from code_jyx.llm_service import DimensionMiningWithQwen
from rag_core.fact_index import build_fact_index


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--max-text-chars", default=1800, type=int)
    parser.add_argument("--limit", default=0, type=int,
                        help="Optional smoke-test limit; 0 means all chunks.")
    args = parser.parse_args()

    source, target = args.source_run, args.output_run
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        shutil.copytree(source, target)
    elif not (target / "tags_output_v2.json").exists():
        raise FileExistsError(f"output run is not a resumable copied run: {target}")

    chunk_path = target / "chunks.json"
    if not chunk_path.exists():
        chunk_path = target / "chunks" / "chunks.json"
    records = load_json(chunk_path)
    tags_path = target / "tags_output_v2.json"
    tags = load_json(tags_path)
    facts_path = target / "facts_output_v1.json"
    facts_output = load_json(facts_path) if facts_path.exists() else {
        "schema_version": "fact_output_v1", "documents": {},
    }
    documents = facts_output.setdefault("documents", {})
    miner = DimensionMiningWithQwen()
    selected = records[:args.limit] if args.limit else records
    total = len(selected)
    for completed, record in enumerate(selected, 1):
        chunk_id = str(record.get("chunk_id", ""))
        if chunk_id not in documents:
            text = str(record.get("chunk_text_full", record.get("doc_text", "")) or "")
            documents[chunk_id] = {"facts": miner.extract_facts_v2(text, max_text_chars=args.max_text_chars)}
            save_json(facts_path, facts_output)
        progress = {
            "stage": "verified_fact_extraction", "completed": completed, "total": total,
            "percent": round(100.0 * completed / total, 2) if total else 100.0,
        }
        save_json(target / "fact_extraction_progress.json", progress)
        print(f"[事实索引] {completed}/{total}", flush=True)

    # Only the processed records are indexed in a smoke test.  A full run
    # always contains every chunk and is the artifact used for evaluation.
    index = build_fact_index(selected, documents)
    save_json(target / "fact_index_v1.json", index)
    tag_documents = tags.get("documents", {})
    for chunk_id, document in documents.items():
        if chunk_id in tag_documents and isinstance(tag_documents[chunk_id], dict):
            tag_documents[chunk_id]["facts"] = document.get("facts", [])
    save_json(tags_path, tags)

    # The local payload is used by offline evaluation.  Preserve embeddings
    # and all labels, adding facts only as inspectable result metadata.
    points_path = target / "local_points.json"
    if points_path.exists():
        points = load_json(points_path)
        for point in points:
            payload = point.get("payload", {})
            chunk_id = str(payload.get("chunk_id", ""))
            payload["facts"] = documents.get(chunk_id, {}).get("facts", [])
        save_json(points_path, points)
    manifest = {
        "source_run": str(source), "fact_index": "fact_index_v1.json",
        "fact_count": index["fact_count"], "fact_chunk_count": index["chunk_count"],
        "processed_chunks": len(selected), "full_corpus": len(selected) == len(records),
    }
    save_json(target / "fact_index_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
