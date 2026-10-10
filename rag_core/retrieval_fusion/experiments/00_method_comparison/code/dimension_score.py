"""Dimension-score baseline; same scorer used online and in replay."""
import math,re
from collections import Counter
DIMENSION_WEIGHT=0.05
LEXICAL_WEIGHT=0.10
MISSING_SEMANTIC_PENALTY=0.025

def grams(text):
    result = set()
    for token in re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', text.lower()):
        for width in (2, 3, 4):
            result.update(token[index:index + width]
                          for index in range(max(0, len(token) - width + 1)))
    return result


def score_snapshot(snapshot):
    semantic_route = snapshot['semantic_candidates']
    dimension_route = snapshot['dimension_candidates']
    old = {str(item['chunk_id']): item for item in snapshot['fusion_candidates']}
    chunks = snapshot.get('chunks') or {}
    semantic = {str(item['chunk_id']): item for item in semantic_route}
    dimension = {str(item['chunk_id']): item for item in dimension_route}
    semantic_scores = [float(item.get('score', 0.0)) for item in semantic_route]
    semantic_floor = max(0.0, min(semantic_scores) - MISSING_SEMANTIC_PENALTY) if semantic_scores else 0.0
    query = snapshot.get('retrieval_query') or snapshot.get('query') or ''
    query_grams = grams(query)
    bags = [grams(str(item.get('chunk_text') or
                      chunks.get(str(item['chunk_id']), {}).get('text', ''))[:4000])
            for item in snapshot['fusion_candidates']]
    document_frequency = Counter(
        gram for bag in bags for gram in bag if gram in query_grams
    )
    weights = {gram: math.log((len(bags) + 1) / (document_frequency[gram] + 0.5))
               for gram in query_grams}
    total_weight = sum(weights.values()) or 1.0
    lexical_scores = [sum(weights[gram] for gram in query_grams & bag) / total_weight
                      for bag in bags]
    max_lexical = max(lexical_scores, default=0.0)
    ranked = []
    for old_rank, (old_item, raw_lexical) in enumerate(zip(snapshot['fusion_candidates'], lexical_scores), 1):
        chunk_id = str(old_item['chunk_id'])
        sem = semantic.get(chunk_id, {})
        dim = dimension.get(chunk_id, {})
        effective_semantic = float(sem.get('score', semantic_floor))
        dimension_norm = float(dim.get('normalized_dimension_score', 0.0))
        lexical_score = raw_lexical / max_lexical if max_lexical else 0.0
        components = {
            'semantic': effective_semantic,
            'dimension': DIMENSION_WEIGHT * dimension_norm,
            'lexical': LEXICAL_WEIGHT * lexical_score,
        }
        score = round(sum(components.values()), 10)
        chunk = chunks.get(chunk_id, {})
        ranked.append({
            'chunk_id': chunk_id, 'score': score,
            'semantic_rank': sem.get('rank'), 'dimension_rank': dim.get('rank'),
            'semantic_score': sem.get('score'), 'dimension_score': dim.get('score'),
            'normalized_semantic_score': sem.get('normalized_semantic_score', 0.0),
            'normalized_dimension_score': dimension_norm,
            'effective_semantic_score': effective_semantic,
            'semantic_score_imputed': chunk_id not in semantic,
            'lexical_score': lexical_score, 'score_components': components,
            'original_fusion_rank': old_rank,
            'original_fusion_score': old_item.get('score'),
            'doc_title': old_item.get('doc_title') or chunk.get('title'),
            'source_file': old_item.get('source_file') or chunk.get('source_file'),
            'chunk_text': old_item.get('chunk_text') or chunk.get('text', '')[:500],
        })
    ranked.sort(key=lambda item: (-item['score'], item['original_fusion_rank']))
    for rank, item in enumerate(ranked, 1):
        item['rank'] = rank
    top_k = int(snapshot.get('top_k', 10))
    return {
        'query': query, 'retrieval_query': query,
        'original_query': snapshot.get('original_query'),
        'strategy': 'normalized_dimension_score_blend_v1',
        'configuration': {
            'dimension_weight': DIMENSION_WEIGHT,
            'lexical_weight': LEXICAL_WEIGHT,
            'missing_semantic_penalty': MISSING_SEMANTIC_PENALTY,
        },
        'top_k': top_k, 'fusion_candidates': ranked,
        'fusion_results': ranked[:top_k],
    }




def fuse(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
    from importlib import import_module
    _shared = import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.runtime')
    make_snapshot = _shared.make_snapshot
    restore_payloads = _shared.restore_payloads
    snapshot = make_snapshot(semantic, dimension, top_k=top_k, query=query, original_query=original_query, query_analysis=query_analysis)
    result = score_snapshot(snapshot)
    return restore_payloads(result, snapshot, "dimension_score.py")
