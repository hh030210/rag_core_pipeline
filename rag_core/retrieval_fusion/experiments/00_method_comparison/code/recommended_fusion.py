"""Self-contained dimension-role calibrated fusion."""
import math,re,json
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

def make_snapshot(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
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
    snapshot = make_snapshot(semantic, dimension, top_k=top_k, query=query, original_query=original_query, query_analysis=query_analysis)
    detail = {"query":snapshot["query"], "retrieval_query":snapshot["retrieval_query"], "query_analysis":query_analysis or {}, "chunks":snapshot["chunks"], "routes":{"semantic":semantic,"dimension":dimension,"old":snapshot["fusion_candidates"]}}
    scored = rank(detail)
    return restore_payloads({"fusion_candidates":scored["candidates"],"configuration":CONFIG,"diagnostics":scored["diagnostics"]}, snapshot, "recommended_fusion.py")


def run_comparison(input_dir, output_dir=None, *, top_k=10):
    """Replay this method from saved route inputs; atomically replace its flat JSON output."""
    import copy, json, shutil, tempfile
    from pathlib import Path
    from datetime import datetime, timezone
    from uuid import uuid4
    source = Path(input_dir).resolve()
    method = Path(__file__).stem
    comparison = Path(__file__).resolve().parents[1]
    target = Path(output_dir or comparison / 'output' / method).absolute()
    if target.is_symlink():
        raise ValueError('Output must not be a symlink')
    manifest = json.loads((source / 'index.json').read_text())
    rows = []
    for name, relative in manifest['detail_files'].items():
        file = (source / relative).resolve()
        if Path(name).name != name or not file.is_relative_to(source):
            raise ValueError('Unsafe detail path')
        detail = json.loads(file.read_text())
        if detail.get('batch_id') != manifest.get('batch_id'):
            raise ValueError('Mixed source batches')
        rows.append((name, detail))
    target.parent.mkdir(parents=True, exist_ok=True)
    batch = uuid4().hex
    index = dict(schema_version=2, batch_id=batch, input_batch_id=manifest['batch_id'],
                 generated_at=datetime.now(timezone.utc).isoformat(), strategy_file=method+'.py',
                 source='run_comparison_methods', items=[], detail_files={})
    mapped = hits5 = 0
    with tempfile.TemporaryDirectory(prefix='.comparison-', dir=target.parent) as temporary:
        stage = Path(temporary) / 'current'
        previous = Path(temporary) / 'previous'
        stage.mkdir()
        for name, row in rows:
            semantic, dimension = copy.deepcopy(row['routes']['semantic']), copy.deepcopy(row['routes']['dimension'])
            for candidate in semantic + dimension:
                candidate['chunk_text_full'] = row.get('chunks', {}).get(str(candidate['chunk_id']), {}).get('text', '')
            result = fuse(semantic, dimension, top_k=top_k,
                          query=row.get('retrieval_query') or row.get('query', ''),
                          original_query=row.get('query'), query_analysis=row.get('query_analysis'))
            result['fusion_strategy']['file'] = method+'.py'
            candidates = result['fusion_candidates']
            detail = {**copy.deepcopy(row), **result, 'name':name, 'batch_id':batch, 'strategy_file':method+'.py'}
            detail['routes'] = {**row['routes'], 'old':adaptive_fusion(copy.deepcopy(semantic), copy.deepcopy(dimension), top_k=top_k)['fusion_candidates']}
            if method == 'dimension_score': detail['routes']['new'] = candidates
            if method == 'recommended_fusion': detail['candidates'] = candidates
            (stage/name).write_text(json.dumps(detail, ensure_ascii=False)+'\n')
            gold = set(detail['gold']['gold_chunk_ids'])
            mapped += bool(gold)
            hits5 += bool(gold & {c['chunk_id'] for c in candidates[:5]})
            index['items'].append(dict(name=name, query=detail['query'], gold_count=len(gold)))
            index['detail_files'][name] = name
        (stage/'index.json').write_text(json.dumps(index, ensure_ascii=False)+'\n')
        had_previous = target.exists()
        if had_previous: target.rename(previous)
        try: stage.rename(target)
        except BaseException:
            if had_previous: previous.rename(target)
            raise
    print(json.dumps(dict(method=method, queries=len(rows), mapped=mapped, hit5=hits5,
                          hit5_percent=round(hits5/(mapped or 1)*100, 4), output=str(target)), ensure_ascii=False), flush=True)
    return index


def main(argv=None):
    import argparse
    from pathlib import Path
    from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY
    comparison = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', type=Path, default=comparison/'output'/Path(FUSION_STRATEGY).stem)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_k <= 0: parser.error('--top-k must be positive')
    run_comparison(args.input_dataset, args.output_dir, top_k=args.top_k)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    if not __package__: sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    main()
