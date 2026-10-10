"""Dimension-role calibrated fusion; identical online and replay scorer."""
import math
from collections import Counter
from importlib import import_module
_shared = import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.code.dimension_score')
score_snapshot = _shared.score_snapshot
grams = _shared.grams

LABEL = '推荐融合（维度角色校准）'
CONFIG = dict(dimension_weight=0.0375, lexical_weight=0.075,
              shared_gap_relative=0.02, shared_gap_prominence=5.0, shared_rank_mix=0.625,
              dimension_scale=0.85, window_size=320, lexical_blend=0.75,
              semantic_rank_mix=0.15, role_semantic_gap=0.01,
              topic_dimension_scale=0.25, topic_lexical_scale=0.5, anchor_weight=0.015)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def gap_profile(route):
    head = route[:5]
    scores = [float(c['score']) for c in head]
    gap = max(0.0, scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    relative = gap / max(abs(scores[1]), 1e-8) if len(scores) >= 2 else 0.0
    tail_step = max(0.0, (scores[1] - scores[-1]) / (len(scores) - 2)) if len(scores) >= 3 else 0.0
    return dict(count=len(head), gap=gap, relative_gap=relative, tail_step=tail_step,
                prominence=gap / max(tail_step, 1e-6), top1=head[0]['chunk_id'] if head else None)


def rank(detail):
    """Use route scores, text and query analysis, without reading Gold."""
    routes = dict(detail['routes'])
    from .online_adaptive import adaptive_fusion
    routes['old'] = adaptive_fusion(routes['semantic'], routes['dimension'], top_k=10)['fusion_candidates']
    sem_route, dim_route = routes['semantic'], routes['dimension']
    query = detail.get('retrieval_query') or detail['query']
    baseline = score_snapshot(dict(query=query, semantic_candidates=sem_route,
                         dimension_candidates=dim_route, fusion_candidates=routes['old'],
                         chunks=detail['chunks']))['fusion_candidates']
    baseline.sort(key=lambda c: c['original_fusion_rank'])
    gaps = {key: gap_profile(routes[key]) for key in ('semantic', 'dimension')}
    shared = gaps['semantic']['top1'] is not None and gaps['semantic']['top1'] == gaps['dimension']['top1']
    shared = shared and any(p['count'] >= 3 and p['relative_gap'] >= CONFIG['shared_gap_relative']
                            and p['prominence'] >= CONFIG['shared_gap_prominence'] for p in gaps.values())
    beta = CONFIG['shared_rank_mix'] if shared else 0.0
    positions = {key: {c['chunk_id']: i for i, c in enumerate(routes[key], 1)} for key in gaps}
    sem_scores = [float(c['score']) for c in sem_route]
    low, high = min(sem_scores, default=0.0), max(sem_scores, default=0.0)
    dim_scores = {c['chunk_id']: float(c['score']) for c in dim_route}
    dim_max = max(dim_scores.values(), default=0.0)
    texts = [detail['chunks'].get(c['chunk_id'], {}).get('text', '')[:4000] for c in baseline]
    query_grams = grams(query)
    bags = [grams(text) for text in texts]
    df = Counter(g for bag in bags for g in query_grams if g in bag)
    idf = {g: math.log(1 + (len(bags) - df[g] + 0.5) / (df[g] + 0.5)) for g in query_grams}
    width = CONFIG['window_size']
    windows = [max((sum(idf[g] for g in query_grams & grams(text[start:start + width]))
                    for start in range(0, max(1, len(text)), width // 2)), default=0.0) for text in texts]
    window_max = max(windows, default=0.0)
    analysis = detail.get('query_analysis') or {}
    constraints = analysis.get('constraints') or {}
    precise_poi = any(x.get('entity_type') == 'poi' for x in analysis.get('resolved_poi_entities', []) if isinstance(x, dict))
    topic_only = set(constraints).issubset({'entity_name', 'geo_admin', 'geo_position'}) and not precise_poi
    active = topic_only and gaps['semantic']['relative_gap'] >= CONFIG['role_semantic_gap']
    anchors = {str(t).strip() for t in analysis.get('fact_anchor_terms', []) if len(str(t).strip()) >= 2}
    for value in constraints.values():
        if isinstance(value, dict): anchors.update(str(t).strip() for t in value.get('intent_terms', []) if len(str(t).strip()) >= 2)
    anchor_df = Counter(t for t in anchors for text in texts if t in text)
    anchor_idf = {t: math.log(1 + (len(texts) - anchor_df[t] + 0.5) / (anchor_df[t] + 0.5)) for t in anchors}
    anchor_scores = [sum(anchor_idf[t] for t in anchors if t in text) for text in texts]
    anchor_max = max(anchor_scores, default=0.0)
    dimension_scale = CONFIG['dimension_scale'] * (CONFIG['topic_dimension_scale'] if active else 1.0)
    lexical_scale = CONFIG['topic_lexical_scale'] if active else 1.0
    ranked = []
    for index, c in enumerate(baseline):
        cid = c['chunk_id']; sem_pos = positions['semantic'].get(cid); dim_pos = positions['dimension'].get(cid)
        sem = max(0.0, low) if c['semantic_score_imputed'] else c['effective_semantic_score']
        raw_dim = max(0.0, dim_scores.get(cid, 0.0)) / dim_max if dim_max > 0 else 0.0
        sem_percentile = (len(sem_route) - sem_pos) / max(1, len(sem_route) - 1) if sem_pos is not None else 0.0
        dim_percentile = (len(dim_route) - dim_pos) / max(1, len(dim_route) - 1) if dim_pos is not None else 0.0
        sem_rank_score = low + (high - low) * sem_percentile
        sem = (1 - beta) * sem + beta * sem_rank_score if sem_pos is not None else sem
        dim = (1 - beta) * raw_dim + beta * dim_percentile
        lexical_weight = (1 - beta) * CONFIG['lexical_weight']
        components = dict(semantic=sem, dimension=(1 - beta) * CONFIG['dimension_weight'] * dim,
                          lexical=lexical_weight * c['lexical_score'])
        base_score = round(sum(components.values()), 10)
        window = windows[index] / window_max if window_max else 0.0
        anchor = anchor_scores[index] / anchor_max if anchor_max else 0.0
        adjustments = dict(dimension_rescale=(dimension_scale - 1) * components['dimension'],
                           lexical_rescale=(lexical_scale - 1) * components['lexical'],
                           window_replacement=CONFIG['lexical_blend'] * lexical_scale * lexical_weight * (window - c['lexical_score']),
                           semantic_rank_adjustment=CONFIG['semantic_rank_mix'] * (sem_rank_score - sem) if sem_pos is not None else 0.0)
        if active: adjustments['intent_anchor_bonus'] = CONFIG['anchor_weight'] * anchor
        ranked.append(dict(chunk_id=cid, score=round(base_score + sum(adjustments.values()), 10),
                           score_components={**components, **adjustments}, semantic_score=c['semantic_score'],
                           dimension_score=c['dimension_score'], effective_semantic_score=sem,
                           effective_dimension_score=dim, lexical_score=c['lexical_score'],
                           semantic_score_imputed=c['semantic_score_imputed'],
                           lexical_evidence=dict(window=window, intent_anchor=anchor,
                                                 matched_anchors=sorted(t for t in anchors if t in texts[index])),
                           original_fusion_rank=c['original_fusion_rank']))
    ranked.sort(key=lambda c: (-c['score'], c['original_fusion_rank']))
    for i, c in enumerate(ranked, 1): c['rank'] = i
    return dict(candidates=ranked, diagnostics=dict(gaps=gaps, shared_gap_active=shared,
                topic_only=topic_only, role_calibration_active=active,
                dimension_scale=dimension_scale, lexical_scale=lexical_scale))




def fuse(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
    from importlib import import_module
    _shared = import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.runtime')
    make_snapshot = _shared.make_snapshot
    restore_payloads = _shared.restore_payloads
    snapshot = make_snapshot(semantic, dimension, top_k=top_k, query=query, original_query=original_query, query_analysis=query_analysis)
    detail = {"query":snapshot["query"], "retrieval_query":snapshot["retrieval_query"], "query_analysis":query_analysis or {}, "chunks":snapshot["chunks"], "routes":{"semantic":semantic,"dimension":dimension,"old":snapshot["fusion_candidates"]}}
    scored = rank(detail)
    return restore_payloads({"fusion_candidates":scored["candidates"],"configuration":CONFIG,"diagnostics":scored["diagnostics"]}, snapshot, "recommended_fusion.py")
