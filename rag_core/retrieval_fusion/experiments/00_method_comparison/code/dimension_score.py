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

def fuse(semantic, dimension, *, top_k, query=None, original_query=None, query_analysis=None):
    snapshot = make_snapshot(semantic, dimension, top_k=top_k, query=query, original_query=original_query, query_analysis=query_analysis)
    result = score_snapshot(snapshot)
    return restore_payloads(result, snapshot, "dimension_score.py")


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
