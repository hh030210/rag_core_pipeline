"""Label-free raw-score fusion and reusable features for offline replay."""
from __future__ import annotations

import math
import re
from collections import Counter


def grams(text):
    result = set()
    for token in re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', text.lower()):
        for width in (2, 3, 4):
            result.update(token[i:i+width] for i in range(max(0, len(token)-width+1)))
    return result


def prepare_features(snapshot):
    """Use retrieval scores/text only. Labels are deliberately not an argument."""
    sem = snapshot['semantic_candidates']
    dim = snapshot['dimension_candidates']
    old = snapshot['fusion_candidates']
    sm = {str(x['chunk_id']): x for x in sem}
    dm = {str(x['chunk_id']): x for x in dim}
    qgrams = grams(snapshot.get('query', '') or '')
    bags = [grams(str(x.get('chunk_text', ''))[:4000]) for x in old]
    df = Counter(g for bag in bags for g in bag if g in qgrams)
    weights = {g: math.log((len(old)+1)/(df[g]+0.5)) for g in qgrams}
    total = sum(weights.values()) or 1.0
    lexical = [sum(weights[g] for g in qgrams & bag)/total for bag in bags]
    best = max(lexical, default=0.0)
    mean_length = sum(len(bag) for bag in bags)/len(bags) if bags else 1.0
    adjusted = {}
    for b in (0, .25, .5, .75):
        values = [lex/(1-b+b*len(bag)/(mean_length or 1.0)) for lex, bag in zip(lexical, bags)]
        max_value = max(values, default=0.0)
        adjusted[str(b)] = [v/max_value if max_value else 0.0 for v in values]
    candidates = []
    for index, (item, lex) in enumerate(zip(old, lexical)):
        cid = str(item['chunk_id'])
        s, d = sm.get(cid, {}), dm.get(cid, {})
        candidates.append({
            'chunk_id': cid, 's': float(s.get('score', 0)), 'd': float(d.get('score', 0)),
            'has_s': cid in sm, 'has_d': cid in dm,
            'a': float(d.get('fact_anchor_bonus', 0))+float(d.get('precise_entity_bonus', 0)),
            'sn': float(s.get('normalized_semantic_score', 0)),
            'dn': float(d.get('normalized_dimension_score', 0)),
            'l': lex/best if best else 0.0, 'lc': lex,
            'lb': {b: adjusted[b][index] for b in adjusted},
            'sr': s.get('rank'), 'dr': d.get('rank'), 'old_rank': index+1,
            'text': str(item.get('chunk_text', ''))[:500],
        })
    scores = sorted((float(x['score']) for x in sem), reverse=True)
    if any(not math.isfinite(c[k]) for c in candidates for k in ('s', 'd', 'a')):
        raise ValueError('Retrieval scores must be finite')
    return {'query': snapshot.get('query'), 'candidates': candidates, 'scores': scores,
            'gaps': [scores[i]-scores[i+1] for i in range(min(len(scores)-1, 9))]}


def rank_features(row, *, mode='raw_fixed_imputed', beta=0, lexical=.1,
                  head=4, threshold=.05, boost=0, missing_penalty=.025,
                  dimension_rank_constant=None, lexical_length_weight=0,
                  lexical_raw=False, anchor_weight=0, lexical_boost=0):
    """Raw route scores are used unchanged; missing semantic scores are imputed.

    ``head`` counts adjacent gaps, so head=4 inspects Top 1 through Top 5.
    Multi-position uncertainty can control both dimension and lexical weights.
    Optional rank, lexical-length and anchor terms support recorded ablations.
    """
    if threshold <= 0 or head < 1:
        raise ValueError('threshold and head must be positive')
    gap = row['gaps'][0] if row['gaps'] else 0.0
    if mode in ('raw_top1', 'raw_top1_imputed', 'raw_lex_top1'):
        confidence_gap = gap
    elif mode in ('raw_head', 'raw_head_imputed', 'raw_rank_head', 'raw_lex_head'):
        confidence_gap = max(row['gaps'][:head], default=0.0)
    elif mode == 'raw_margin':
        tail = row['scores'][1:head+1]
        confidence_gap = row['scores'][0]-sum(tail)/len(tail) if tail else 0.0
    elif mode == 'raw_entropy_imputed':
        values = row['scores'][:head+1]
        probs = [math.exp((s-values[0])/threshold) for s in values]
        total = sum(probs) or 1.0
        probs = [p/total for p in probs]
        entropy = -sum(p*math.log(p) for p in probs)/math.log(len(probs)) if len(probs)>1 else 1.0
        confidence_gap = threshold*(1-entropy)
    elif mode in ('raw_fixed', 'raw_fixed_imputed', 'raw_rank_fixed', 'raw_length_fixed', 'raw_coverage_fixed'):
        confidence_gap = threshold
    else:
        raise ValueError(f'Unknown raw fusion mode: {mode}')
    uncertainty = max(0.0, 1.0-confidence_gap/threshold)
    dimension_weight = beta+boost*uncertainty
    lexical_weight = lexical+lexical_boost*uncertainty
    floor = max(0.0, row['scores'][-1]-missing_penalty) if missing_penalty is not None and row['scores'] else 0.0
    ranked = []
    for c in row['candidates']:
        semantic = c['s'] if c['has_s'] else floor
        dimension = c['d']
        if dimension_rank_constant is not None:
            dimension = dimension/(dimension_rank_constant+c['dr']) if c['dr'] else 0.0
        lex = c['lc'] if lexical_raw else c['lb'][f'{lexical_length_weight:g}']
        components = {'semantic': semantic, 'dimension': dimension_weight*dimension,
                      'lexical': lexical_weight*lex, 'anchor': anchor_weight*c['a']}
        score = round(sum(components.values()), 10)
        ranked.append({**c, 'score': score, 'components': components,
                       'effective_lexical_score': lex, 'semantic_imputed': not c['has_s']})
    ranked.sort(key=lambda c: (-c['score'], c['old_rank']))
    head_gaps = row['gaps'][:head]
    diagnostics = {'semantic_head_scores': row['scores'][:head+1],
                   'semantic_top1_gap': gap, 'semantic_head_gap': max(head_gaps, default=0.0),
                   'semantic_head_gap_position': head_gaps.index(max(head_gaps))+1 if head_gaps else None,
                   'confidence_gap': confidence_gap, 'uncertainty': uncertainty,
                   'effective_dimension_weight': dimension_weight,
                   'effective_lexical_weight': lexical_weight, 'semantic_missing_estimate': floor}
    return ranked, diagnostics


def fuse_raw(snapshot, config):
    features = prepare_features(snapshot)
    ranked, diagnostics = rank_features(features, **config)
    old = {str(x['chunk_id']): x for x in snapshot['fusion_candidates']}
    results = []
    for rank, c in enumerate(ranked, 1):
        item = old[c['chunk_id']]
        results.append({
            'chunk_id': c['chunk_id'], 'rank': rank, 'score': c['score'],
            'semantic_rank': c['sr'], 'dimension_rank': c['dr'],
            'semantic_score': c['s'] if c['has_s'] else None,
            'dimension_score': c['d'] if c['has_d'] else None,
            'normalized_semantic_score': c['sn'], 'normalized_dimension_score': c['dn'],
            'effective_semantic_score': c['components']['semantic'],
            'semantic_score_imputed': c['semantic_imputed'],
            'lexical_score': c['effective_lexical_score'], 'anchor_score': c['a'],
            'score_components': c['components'],
            'original_fusion_rank': c['old_rank'], 'original_fusion_score': item.get('score'),
            'doc_title': item.get('doc_title'), 'source_file': item.get('source_file'),
            'chunk_text': item.get('chunk_text', ''),
        })
    top_k = int(snapshot.get('top_k', 10))
    return {'query': snapshot.get('query'), 'strategy': 'raw_head_lexical',
            'configuration': config, 'diagnostics': diagnostics, 'top_k': top_k,
            'fusion_candidates': results, 'fusion_results': results[:top_k]}
