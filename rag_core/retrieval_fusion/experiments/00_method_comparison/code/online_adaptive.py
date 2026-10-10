"""Retained adaptive online baseline."""
from importlib import import_module
_shared = import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.runtime')
_normalize_scores = _shared._normalize_scores

def adaptive_fusion(semantic, dimension, *, top_k):
    """Current online adaptive_report_v1 strategy, extracted without rank changes."""
    sem_norm = _normalize_scores(semantic)
    dim_norm = _normalize_scores(dimension)
    # Keep the route-local normalized scores on the candidates themselves.
    # Fusion is computed from these values, so retaining them makes every
    # ranking decision auditable instead of exposing only the final score.
    for item in semantic:
        item["semantic_score"] = item.get("score")
        item["normalized_semantic_score"] = round(
            sem_norm.get(str(item["chunk_id"]), 0.0), 8
        )
    for item in dimension:
        item["dimension_score"] = item.get("score")
        item["normalized_dimension_score"] = round(
            dim_norm.get(str(item["chunk_id"]), 0.0), 8
        )
    semantic_by_id = {item["chunk_id"]: item for item in semantic}
    dimension_by_id = {item["chunk_id"]: item for item in dimension}
    # The fusion strategy is deliberately applied only after both route
    # searches are complete.  This keeps semantic/dimension retrieval
    # unchanged and makes the candidate order an auditable tie-breaker.
    candidate_order = list(dict.fromkeys([item["chunk_id"] for item in semantic + dimension]))
    by_id = {}
    for cid in candidate_order:
        item = dict(dimension_by_id.get(cid) or semantic_by_id[cid])
        if cid in semantic_by_id:
            item["sem_rank"] = semantic_by_id[cid].get("rank")
            item["semantic_score"] = semantic_by_id[cid].get("score")
            item["normalized_semantic_score"] = sem_norm.get(cid, 0.0)
        if cid in dimension_by_id:
            item["dim_rank"] = dimension_by_id[cid].get("rank")
            item["dimension_score"] = dimension_by_id[cid].get("score")
            item["normalized_dimension_score"] = dim_norm.get(cid, 0.0)
        # A candidate absent from one route contributes zero from that
        # route to the weighted fusion score.
        item.setdefault("normalized_semantic_score", 0.0)
        item.setdefault("normalized_dimension_score", 0.0)
        by_id[cid] = item
    semantic_count = len(semantic)
    dimension_count = len(dimension)
    overlap5 = len(
        set(item["chunk_id"] for item in semantic[:5])
        & set(item["chunk_id"] for item in dimension[:5])
    )
    use_rank_aware = (
        (semantic_count >= 40 and not (dimension_count >= 20 and overlap5 == 0))
        or (
            semantic_count < 40
            and (
                (dimension_count == 10 and overlap5 <= 2)
                or (dimension_count >= 20 and overlap5 == 0)
            )
        )
    )
    fusion_branch = "dimension_rank_aware" if use_rank_aware else "score_weighted"
    fused_scores = {}
    for cid in candidate_order:
        if use_rank_aware:
            dim_rank = dimension_by_id.get(cid, {}).get("rank")
            dim_rank_term = 0.6 / (int(dim_rank) + 1) if dim_rank else 0.0
            fused_scores[cid] = 0.4 * sem_norm.get(cid, 0.0) + dim_rank_term
        else:
            fused_scores[cid] = 0.77 * sem_norm.get(cid, 0.0) + 0.23 * dim_norm.get(cid, 0.0)
    fusion_candidates = []
    for cid, score in sorted(
        fused_scores.items(), key=lambda pair: pair[1], reverse=True
    ):
        item = by_id[cid]
        item["score"] = score
        item["final_score"] = score
        item["fused_score"] = score
        item["fusion_branch"] = fusion_branch
        item["source"] = ("dimension" if cid in dim_norm else "") + ("+semantic" if cid in sem_norm else "")
        fusion_candidates.append(item)
    fusion = fusion_candidates[:top_k]

    return {
        "fusion_candidates": fusion_candidates, "fusion_results": fusion,
        "fusion_strategy": {
            "name": "adaptive_report_v1", "branch": fusion_branch,
            "semantic_count": semantic_count, "dimension_count": dimension_count,
            "overlap5": overlap5,
            "score_weighted": {"semantic": 0.77, "dimension": 0.23},
            "rank_aware": {"semantic": 0.4, "dimension_rank": 0.6},
            "candidate_order": "semantic_then_dimension_union",
        },
    }


def fuse(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
    return adaptive_fusion(semantic, dimension, top_k=top_k)
