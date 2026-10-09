"""Replay the single offline strategy: normalized dimension-score fusion."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from uuid import uuid4
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[3]
ONLINE_DATASET = ENGINE / 'experiments' / '01_online_snapshots' / 'output' / 'dataset'
DIMENSION_WEIGHT = 0.05
LEXICAL_WEIGHT = 0.10
MISSING_SEMANTIC_PENALTY = 0.025


def grams(text):
    result = set()
    for token in re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', text.lower()):
        for width in (2, 3, 4):
            result.update(token[index:index + width]
                          for index in range(max(0, len(token) - width + 1)))
    return result


def load_dataset_snapshots(dataset_dir: Path):
    """Load the labeled cohort owned by the online baseline experiment."""
    manifest = json.loads((dataset_dir / 'index.json').read_text(encoding='utf-8'))
    snapshots = []
    for name, relative in manifest['detail_files'].items():
        detail = json.loads((dataset_dir / relative).read_text(encoding='utf-8'))
        routes = detail['routes']
        snapshots.append((name, {
            'query': detail.get('retrieval_query') or detail.get('query') or '',
            'retrieval_query': detail.get('retrieval_query') or detail.get('query') or '',
            'original_query': detail.get('query') or '', 'top_k': 10,
            'semantic_candidates': routes.get('semantic') or [],
            'dimension_candidates': routes.get('dimension') or [],
            'fusion_candidates': routes.get('old') or [],
            'chunks': detail.get('chunks') or {},
            'gold_chunk_ids': detail['gold']['gold_chunk_ids'],
            'view_detail': detail,
        }))
    return snapshots


def fuse(snapshot):
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


def read_labels(path: Path | None):
    labels = {}
    if path is None:
        return labels
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            labels[row['retrieval_query']] = set(row['gold']['chunk_ids'])
    return labels


def first_rank(items, gold):
    return next((rank for rank, item in enumerate(items, 1)
                 if item.get('chunk_id') in gold), None)


def ndcg5(items, gold):
    if not gold:
        return 0.0
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), 5) + 1))
    actual = sum(1 / math.log2(rank + 1) for rank, item in enumerate(items[:5], 1)
                 if item.get('chunk_id') in gold)
    return actual / ideal


def summarize(rows):
    count = len(rows)
    mapped = sum(bool(row['gold_count']) for row in rows)
    report = {'labeled_queries': count, 'gold_mapped': mapped}
    for name, rank_key, ndcg_key in (
        ('online', 'online_rank', 'online_ndcg5'),
        ('dimension_score_blend', 'new_rank', 'new_ndcg5'),
    ):
        ranks = [row[rank_key] for row in rows]
        report[name] = {
            'hit_at_1': sum(rank is not None and rank <= 1 for rank in ranks),
            'hit_at_5': sum(rank is not None and rank <= 5 for rank in ranks),
            'hit_at_10': sum(rank is not None and rank <= 10 for rank in ranks),
            'mrr_at_10': round(sum(1 / rank for rank in ranks if rank and rank <= 10) / (count or 1), 6),
            'ndcg_at_5': round(sum(row[ndcg_key] for row in rows) / (count or 1), 6),
        }
    return report


def replace_output(output_dir, writer):
    """Publish a complete successful replay; preserve the previous run on failure."""
    output_dir = Path(output_dir).absolute()
    if output_dir.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.replay-', dir=output_dir.parent))
    staged_output = stage / 'output'
    backup = stage / 'previous'
    try:
        writer(staged_output)
        manifest = json.loads((staged_output / 'dataset' / 'index.json').read_text(encoding='utf-8'))
        for relative in manifest['detail_files'].values():
            detail = (staged_output / 'dataset' / relative).resolve()
            if not detail.is_relative_to(staged_output.resolve()) or not detail.is_file():
                raise ValueError('Incomplete replay dataset')
        if output_dir.exists():
            output_dir.rename(backup)
        try:
            staged_output.rename(output_dir)
        except BaseException:
            if backup.exists():
                backup.rename(output_dir)
            raise
    finally:
        shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        help='Use retrieval snapshot JSON files instead of the online baseline dataset')
    parser.add_argument('--input-dataset', type=Path, default=ONLINE_DATASET)
    parser.add_argument('--output-dir', type=Path, default=ENGINE / 'experiments' / '06_dimension_score' / 'output')
    parser.add_argument('--evaluation-results', type=Path, help='Optional Golden results for --input-dir; dataset mode uses its own Golden')
    args = parser.parse_args(argv)
    def write(staged_output):
        args.output_dir = staged_output
        _run(args, parser)
    replace_output(args.output_dir, write)


def _run(args, parser):
    if args.input_dir:
        snapshots = [(path.name, json.loads(path.read_text(encoding='utf-8')))
                     for path in sorted(args.input_dir.glob('retrieval_fusion_*.json'))]
    else:
        if not (args.input_dataset / 'index.json').is_file():
            parser.error('Missing online evaluation dataset; run run.py evaluate first')
        snapshots = load_dataset_snapshots(args.input_dataset)
    if not snapshots:
        parser.error('No retrieval snapshots found')
    labels = read_labels(args.evaluation_results if args.evaluation_results and args.evaluation_results.is_file() else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = json.loads((args.input_dataset / "index.json").read_text()) if not args.input_dir and (args.input_dataset / "index.json").is_file() else {}
    batch_id = uuid4().hex
    generated_at = datetime.now(timezone.utc).isoformat()
    comparisons = []
    dataset_dir = args.output_dir / "dataset"
    detail_dir = dataset_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "items": [], "detail_files": {}, "batch_id": batch_id, "input_batch_id": input_manifest.get("batch_id"), "generated_at": generated_at, "source": "06_dimension_score independent replay"}
    for name, snapshot in snapshots:
        result = fuse(snapshot)
        result['source_snapshot'] = name
        result['batch_id'] = batch_id
        detail = snapshot.get('view_detail') or {
            'query': snapshot.get('original_query') or snapshot['query'],
            'retrieval_query': snapshot.get('retrieval_query') or snapshot['query'],
            'query_analysis': snapshot.get('query_analysis', {}),
            'chunks': snapshot.get('chunks', {}),
            'gold': {'gold_chunk_ids': snapshot.get('gold_chunk_ids', [])},
            'routes': {'semantic': snapshot['semantic_candidates'], 'dimension': snapshot['dimension_candidates'], 'old': snapshot['fusion_candidates']},
        }
        result.update({k: detail[k] for k in ['query', 'retrieval_query', 'query_analysis', 'chunks', 'gold']})
        result['name'] = name
        result['routes'] = {**detail['routes'], 'new': result['fusion_candidates']}
        query = result['retrieval_query']
        gold = set(snapshot['gold_chunk_ids']) if 'gold_chunk_ids' in snapshot else labels.get(query, set())
        result['gold'] = {**result['gold'], 'gold_chunk_ids': sorted(gold)}
        (detail_dir / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        query = result['retrieval_query']
        gold = set(snapshot['gold_chunk_ids']) if 'gold_chunk_ids' in snapshot else labels.get(query, set())
        manifest['items'].append({'name': name, 'query': snapshot.get('original_query') or query, 'gold_count': len(gold)})
        manifest['detail_files'][name] = 'details/' + name
        online = snapshot.get('fusion_candidates') or []
        comparisons.append({
            'query': query, 'source_snapshot': name,
            'gold_count': len(gold), 'gold_chunk_ids': sorted(gold),
            'semantic_rank': first_rank(snapshot['semantic_candidates'], gold),
            'dimension_rank': first_rank(snapshot['dimension_candidates'], gold),
            'old_rank': first_rank(online, gold), 'new_rank': first_rank(result['fusion_candidates'], gold),
            'online_rank': first_rank(online, gold),
            'old_ndcg5': ndcg5(online, gold),
            'new_ndcg5': ndcg5(result['fusion_candidates'], gold),
            'online_ndcg5': ndcg5(online, gold),
        })
    (dataset_dir / 'index.json').write_text(json.dumps(manifest, ensure_ascii=False) + '\n', encoding='utf-8')
    report = {'snapshots': len(snapshots), 'strategy': 'normalized_dimension_score_blend_v1',
              'metrics': summarize(comparisons) if comparisons else {}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
