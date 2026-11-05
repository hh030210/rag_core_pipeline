#!/usr/bin/env python3
"""Export vector, dimension, and current fusion rankings for unified evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))
from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY, apply_fusion  # noqa: E402

DEFAULT_DATASET = Path(
    "/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/qa_dataset_435.json"
)
DEFAULT_ROUTE_DIR = (
    REPOSITORY / "rag_core/retrieval_fusion/experiments/00_method_comparison"
    / "output/dimension_score"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
MAX_K = 20
ROUTES = {
    "vector": ("semantic", "向量检索（用户查询语义）"),
    "dimension": ("dimension", "维度检索（关键信息匹配）"),
    "hybrid": ("hybrid", "混合检索（融合排序）"),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query(value):
    return " ".join(str(value or "").split())


def canonical_doc(value):
    value = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if value.lower().endswith((".md", ".txt", ".json")):
        value = value.rsplit(".", 1)[0]
    return value


def infer_doc_from_chunk_id(chunk_id):
    parent = str(chunk_id).split("::", 1)[0]
    return re.sub(r"_\d{3}$", "", parent)


def sorted_candidates(candidates):
    return sorted(
        candidates,
        key=lambda item: int(item.get("rank") or 10**9),
    )


def build_retrieved(candidates, source_by_chunk, gold_ids, gold_docs):
    rows = []
    for rank, candidate in enumerate(sorted_candidates(candidates)[:MAX_K], start=1):
        chunk_id = str(candidate.get("chunk_id", ""))
        source_file = canonical_doc(candidate.get("source_file"))
        if not source_file:
            source_file = source_by_chunk.get(chunk_id) or infer_doc_from_chunk_id(chunk_id)
        rows.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "is_golden_chunk": chunk_id in gold_ids,
            "is_golden_doc": source_file in gold_docs,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--strategy", default=FUSION_STRATEGY)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    manifest = load_json(args.route_dir / "index.json")
    refs = manifest.get("items", [])
    if len(dataset) != len(refs):
        raise RuntimeError(
            f"Dataset/route count mismatch: {len(dataset)} vs {len(refs)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = {route: [] for route in ROUTES}
    no_gold = 0
    unmapped = 0
    doc_fallbacks = 0

    for position, (question, ref) in enumerate(zip(dataset, refs), start=1):
        if normalize_query(question.get("query")) != normalize_query(ref.get("query")):
            raise RuntimeError(f"Query order mismatch at position {position}")
        detail = load_json(args.route_dir / ref["name"])
        if normalize_query(detail.get("query")) != normalize_query(question.get("query")):
            raise RuntimeError(f"Snapshot query mismatch in {ref['name']}")

        gold = detail.get("gold") or {}
        resolution = gold.get("gold_resolution") or {}
        gold_ids = set(map(str, gold.get("chunk_ids") or []))
        gold_mapped = bool(resolution.get("gold_mapped", bool(gold_ids)))
        if not gold_ids:
            no_gold += 1
        elif not gold_mapped:
            unmapped += 1

        routes = detail.get("routes") or {}
        semantic = copy.deepcopy(routes.get("semantic") or [])
        dimension = copy.deepcopy(routes.get("dimension") or [])
        chunks = detail.get("chunks") or {}

        # The compact route snapshot omits text; restore saved payloads before fusion.
        for candidate in semantic + dimension:
            payload = chunks.get(str(candidate.get("chunk_id", "")), {})
            if payload:
                candidate["chunk_text_full"] = payload.get("text", "")
                candidate.setdefault("source_file", payload.get("source_file", ""))
                candidate.setdefault("doc_title", payload.get("title", ""))

        source_by_chunk = {}
        known_candidates = (
            semantic + dimension + (detail.get("fusion_candidates") or [])
        )
        for candidate in known_candidates:
            chunk_id = str(candidate.get("chunk_id", ""))
            if chunk_id:
                source_by_chunk[chunk_id] = canonical_doc(candidate.get("source_file"))
        for chunk_id, payload in chunks.items():
            source_by_chunk.setdefault(
                str(chunk_id), canonical_doc(payload.get("source_file"))
            )

        gold_docs = set()
        for chunk_id in gold_ids:
            doc = source_by_chunk.get(chunk_id)
            if not doc:
                doc = infer_doc_from_chunk_id(chunk_id)
                doc_fallbacks += 1
            if doc:
                gold_docs.add(doc)

        fused = apply_fusion(
            semantic,
            dimension,
            top_k=MAX_K,
            query=detail.get("retrieval_query") or question["query"],
            original_query=question["query"],
            query_analysis=detail.get("query_analysis"),
            strategy_file=args.strategy,
        )
        hybrid_candidates = (
            fused.get("fusion_candidates")
            or fused.get("fusion_results")
            or []
        )
        route_candidates = {
            "vector": semantic,
            "dimension": dimension,
            "hybrid": hybrid_candidates,
        }

        for route, (source_key, label) in ROUTES.items():
            candidates = route_candidates[route]
            retrieved = build_retrieved(
                candidates, source_by_chunk, gold_ids, gold_docs
            )
            detail_rows[route].append({
                "id": question.get("id"),
                "query": question.get("query"),
                "route": route,
                "route_label": label,
                "strategy": args.strategy if route == "hybrid" else source_key,
                "source_batch_id": manifest.get("batch_id"),
                "top_k_saved": MAX_K,
                "candidate_count": len(candidates),
                "answerable": bool(gold_ids),
                "gold_mapped": gold_mapped,
                "gold_chunk_ids": sorted(gold_ids),
                "gold_documents": sorted(gold_docs),
                "retrieved": retrieved,
            })

    files = {}
    for route in ROUTES:
        name = {
            "vector": "vector_retrieval_details.jsonl",
            "dimension": "dimension_retrieval_details.jsonl",
            "hybrid": "hybrid_retrieval_details.jsonl",
        }[route]
        path = args.output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in detail_rows[route]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[route] = name

    index = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(args.dataset.resolve()),
        "route_snapshot_dir": str(args.route_dir.resolve()),
        "source_batch_id": manifest.get("batch_id"),
        "source_snapshot_count": len(refs),
        "top_k_saved": MAX_K,
        "fusion_strategy": args.strategy,
        "files": files,
        "questions_without_golden_chunks": no_gold,
        "questions_with_unmapped_golden": unmapped,
        "gold_document_id_fallbacks": doc_fallbacks,
    }
    (args.output_dir / "details_manifest.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

