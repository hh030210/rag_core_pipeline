"""Shared route preparation and payload preservation; no strategy selection."""
from typing import Any, Dict, List

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



def make_snapshot(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
    from .code.online_adaptive import adaptive_fusion
    baseline = adaptive_fusion(semantic, dimension, top_k=top_k)
    candidates = baseline['fusion_candidates']
    chunks = {str(c['chunk_id']): {'text':str(c.get('chunk_text_full') or c.get('chunk_text') or ''), 'title':c.get('doc_title') or c.get('chunk_gen_title') or '', 'source_file':c.get('source_file') or ''} for c in candidates}
    payloads = candidates
    candidates = [{k:v for k,v in c.items() if k != 'chunk_text'} for c in candidates]
    return dict(payloads=payloads, query=query or original_query or '', retrieval_query=query or original_query or '', original_query=original_query, query_analysis=query_analysis or {}, top_k=top_k, semantic_candidates=semantic, dimension_candidates=dimension, fusion_candidates=candidates, chunks=chunks)

def restore_payloads(result, snapshot, filename):
    originals = {str(c['chunk_id']): c for c in snapshot.get('payloads', snapshot['fusion_candidates'])}
    candidates = [{**originals[str(c['chunk_id'])], **c, 'final_score':c['score'], 'fused_score':c['score']} for c in result['fusion_candidates']]
    result['fusion_candidates'] = candidates
    result['fusion_results'] = candidates[:snapshot['top_k']]
    result['fusion_strategy'] = dict(name=filename[:-3], file=filename, configuration=result.get('configuration', {}))
    return result
