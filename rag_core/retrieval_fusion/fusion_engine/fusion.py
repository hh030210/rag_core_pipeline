"""Score normalization and weighted fusion for retrieval routes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


_COMMON_CANDIDATE_FIELDS = (
    "chunk_id", "score", "rank", "dimension_paths",
)
_ROUTE_CANDIDATE_FIELDS = {
    "semantic": ("normalized_semantic_score",),
    "dimension": (
        "matched_dimensions", "matches", "fact_anchor_bonus",
        "precise_entity_bonus", "normalized_dimension_score",
    ),
    "fusion": (
        "semantic_score", "dimension_score", "normalized_semantic_score",
        "normalized_dimension_score", "sem_rank", "dim_rank",
        "fusion_branch", "matched_dimensions", "matches",
    ),
}


def _compact_candidates(items: List[Dict[str, Any]], route: str) -> List[Dict[str, Any]]:
    """Keep only route scores and annotations needed for replay and viewer."""
    fields = _COMMON_CANDIDATE_FIELDS + _ROUTE_CANDIDATE_FIELDS[route]
    compacted = []
    for item in items:
        saved = {key: item[key] for key in fields if key in item}
        if isinstance(saved.get("matches"), list):
            saved["matches"] = [
                {key: match[key] for key in ("dimension_id", "dimension_path", "label", "similarity")
                 if key in match}
                for match in saved["matches"] if isinstance(match, dict)
            ]
        compacted.append(saved)
    return compacted


def _compact_chunks(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Store display text once per chunk instead of repeating it per route."""
    chunks: Dict[str, Dict[str, str]] = {}
    for item in items:
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id or chunk_id in chunks:
            continue
        text = str(item.get("chunk_text_full") or item.get("chunk_text") or "")
        chunks[chunk_id] = {
            "title": str(item.get("chunk_gen_title") or item.get("doc_title") or ""),
            "source_file": str(item.get("source_file") or ""),
            "text": text,
        }
    return chunks


def _save_fusion_snapshot(
    semantic: List[Dict[str, Any]],
    dimension: List[Dict[str, Any]],
    fusion_candidates: List[Dict[str, Any]],
    *,
    dim_alpha: float,
    top_k: int,
    query: str | None = None,
    original_query: str | None = None,
    query_analysis: Dict[str, Any] | None = None,
) -> Path:
    """Save one complete retrieval/fusion result under the project output dir."""
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f%z")
    snapshot = {
        "created_at": created_at.isoformat(timespec="microseconds"),
        "dim_alpha": dim_alpha,
        "top_k": top_k,
        "semantic_candidates": _compact_candidates(semantic, "semantic"),
        "dimension_candidates": _compact_candidates(dimension, "dimension"),
        "fusion_candidates": _compact_candidates(fusion_candidates, "fusion"),
        "chunks": _compact_chunks(fusion_candidates),
    }
    if query is not None:
        snapshot["query"] = query
        # query remains for compatibility; retrieval_query explicitly names
        # the exact text that Retriever.search encoded for semantic search.
        snapshot["retrieval_query"] = query
    if original_query is not None:
        snapshot["original_query"] = original_query
    if query_analysis is not None:
        # The offline fusion strategy does not need parser diagnostics. The
        # viewer only uses constraints, so retain that small, useful subset.
        snapshot["query_analysis"] = {
            "constraints": {
                dimension: {
                    key: value[key]
                    for key in ("labels", "intent_terms", "role")
                    if key in value
                }
                for dimension, value in query_analysis.get("constraints", {}).items()
                if isinstance(value, dict)
            }
        }
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2)

    suffix = 0
    while True:
        suffix_text = f"_{suffix}" if suffix else ""
        path = output_dir / f"retrieval_fusion_{timestamp}{suffix_text}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
            return path
        except FileExistsError:
            suffix += 1


def _normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Min-max normalize one route's scores, keyed by chunk ID."""
    if not results:
        return {}
    scores = [float(item.get("score", 0.0)) for item in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {str(item["chunk_id"]): 1.0 for item in results}
    return {
        str(item["chunk_id"]): (float(item.get("score", 0.0)) - low) / (high - low)
        for item in results
    }


def fuse_retrieval_results(
    semantic: List[Dict[str, Any]],
    dimension: List[Dict[str, Any]],
    *,
    dim_alpha: float,
    top_k: int,
    query: str | None = None,
    original_query: str | None = None,
    query_analysis: Dict[str, Any] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize two route result lists and return their weighted fusion.

    Route candidates are annotated with their original and normalized scores
    for auditability. The fused candidates are a union by ``chunk_id``; a
    candidate missing from one route receives a zero score for that route.
    """
    semantic_norm = _normalize_scores(semantic)
    dimension_norm = _normalize_scores(dimension)

    for item in semantic:
        item["semantic_score"] = item.get("score")
        item["normalized_semantic_score"] = round(
            semantic_norm.get(str(item["chunk_id"]), 0.0), 8
        )
    for item in dimension:
        item["dimension_score"] = item.get("score")
        item["normalized_dimension_score"] = round(
            dimension_norm.get(str(item["chunk_id"]), 0.0), 8
        )

    semantic_by_id = {item["chunk_id"]: item for item in semantic}
    dimension_by_id = {item["chunk_id"]: item for item in dimension}
    by_id: Dict[Any, Dict[str, Any]] = {}
    for chunk_id in dict.fromkeys(
        [item["chunk_id"] for item in semantic + dimension]
    ):
        item = dict(dimension_by_id.get(chunk_id) or semantic_by_id[chunk_id])
        if chunk_id in semantic_by_id:
            item["sem_rank"] = semantic_by_id[chunk_id].get("rank")
            item["semantic_score"] = semantic_by_id[chunk_id].get("score")
            item["normalized_semantic_score"] = semantic_norm.get(chunk_id, 0.0)
        if chunk_id in dimension_by_id:
            item["dim_rank"] = dimension_by_id[chunk_id].get("rank")
            item["dimension_score"] = dimension_by_id[chunk_id].get("score")
            item["normalized_dimension_score"] = dimension_norm.get(chunk_id, 0.0)
        item.setdefault("normalized_semantic_score", 0.0)
        item.setdefault("normalized_dimension_score", 0.0)
        by_id[chunk_id] = item

    fused_scores = {
        chunk_id: dim_alpha * dimension_norm.get(chunk_id, 0.0)
        + (1 - dim_alpha) * semantic_norm.get(chunk_id, 0.0)
        for chunk_id in by_id
    }
    fusion_candidates = []
    for chunk_id, score in sorted(
        fused_scores.items(), key=lambda pair: pair[1], reverse=True
    ):
        item = by_id[chunk_id]
        item["score"] = score
        item["final_score"] = score
        item["fused_score"] = score
        item["source"] = (
            ("dimension" if chunk_id in dimension_norm else "")
            + ("+semantic" if chunk_id in semantic_norm else "")
        )
        fusion_candidates.append(item)

    result = {
        "semantic_candidates": semantic,
        "dimension_candidates": dimension,
        "fusion_candidates": fusion_candidates,
        "fusion_results": fusion_candidates[:top_k],
    }
    _save_fusion_snapshot(
        result["semantic_candidates"],
        result["dimension_candidates"],
        result["fusion_candidates"],
        dim_alpha=dim_alpha,
        top_k=top_k,
        query=query,
        original_query=original_query,
        query_analysis=query_analysis,
    )
    return result
