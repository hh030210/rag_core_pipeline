"""Replay the latest offline fusion strategy on saved retrieval snapshots.
Gold labels are used only for reporting after ranking.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from strategies import fuse_raw

ENGINE = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).with_name('selected_config.json')


def fuse(snapshot, config=None):
    if config is None:
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
    return fuse_raw(snapshot, config)


def first_rank(items, gold):
    return next((i for i, item in enumerate(items, 1) if item['chunk_id'] in gold), None)


def read_labels(path):
    labels = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            labels[row['retrieval_query']] = set(row['gold']['chunk_ids'])
    return labels


def summarize(comparisons):
    n = len(comparisons)
    report = {'labeled_queries': n, 'gold_mapped': sum(bool(row['gold_count']) for row in comparisons)}
    for name, key in [('baseline', 'old'), ('fusion_new', 'new')]:
        ranks = [row[key+'_rank'] for row in comparisons]
        report[name] = {
            'hit_at_1': sum(r is not None and r <= 1 for r in ranks),
            'hit_at_5': sum(r is not None and r <= 5 for r in ranks),
            'hit_at_10': sum(r is not None and r <= 10 for r in ranks),
            'mrr_at_10': round(sum(1/r for r in ranks if r is not None and r <= 10)/n, 6),
            'ndcg_at_5': round(sum(row[key+'_ndcg5'] for row in comparisons)/n, 6),
        }
    report['rescued_at_5'] = sum((r['old_rank'] is None or r['old_rank'] > 5) and
                                 r['new_rank'] is not None and r['new_rank'] <= 5 for r in comparisons)
    report['harmed_at_5'] = sum(r['old_rank'] is not None and r['old_rank'] <= 5 and
                                (r['new_rank'] is None or r['new_rank'] > 5) for r in comparisons)
    for name, key in [('baseline', 'old'), ('fusion_new', 'new')]:
        report[name]['single_route_top5_lost'] = sum(
            ((r['semantic_rank'] is not None and r['semantic_rank'] <= 5) or
             (r['dimension_rank'] is not None and r['dimension_rank'] <= 5)) and
            (r[key+'_rank'] is None or r[key+'_rank'] > 5) for r in comparisons)
    return report


def ndcg5(items, gold):
    if not gold:
        return 0.0
    ideal = sum(1/math.log2(i+2) for i in range(min(len(gold), 5)))
    actual = sum(1/math.log2(i+1) for i, x in enumerate(items[:5], 1) if x['chunk_id'] in gold)
    return actual/ideal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ENGINE/'output')
    parser.add_argument('--output-dir', type=Path, default=ENGINE/'output_new')
    parser.add_argument('--config', type=Path, default=CONFIG,
                        help='Parameters for the latest offline strategy')
    parser.add_argument('--evaluation-results', type=Path,
                        help='Optional labels for reporting; never used in ranking')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    paths = sorted(args.input_dir.glob('retrieval_fusion_*.json'))
    if not paths:
        parser.error(f'No snapshots in {args.input_dir}')
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error('Input and output must differ')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.evaluation_results) if args.evaluation_results else {}
    comparisons = []
    badcases = []
    for path in paths:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        result = fuse(snapshot, config)
        result['source_snapshot'] = str(path.resolve())
        (args.output_dir/path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        gold = labels.get(result['query'])
        if gold is None:
            continue
        old = snapshot['fusion_candidates']
        new = result['fusion_candidates']
        old_rank, new_rank = first_rank(old, gold), first_rank(new, gold)
        comparison = {
            'query': result['query'], 'source_snapshot': path.name,
            'gold_count': len(gold), 'gold_chunk_ids': sorted(gold),
            'semantic_rank': first_rank(snapshot['semantic_candidates'], gold),
            'dimension_rank': first_rank(snapshot['dimension_candidates'], gold),
            'old_rank': old_rank, 'new_rank': new_rank,
            'old_ndcg5': ndcg5(old, gold), 'new_ndcg5': ndcg5(new, gold),
            'semantic_top1_gap': result['diagnostics']['semantic_top1_gap'],
            'semantic_head_gap': result['diagnostics']['semantic_head_gap'],
        }
        comparisons.append(comparison)
        old_hit = old_rank is not None and old_rank <= 5
        new_hit = new_rank is not None and new_rank <= 5
        if not old_hit or not new_hit:
            category = ('rescued' if new_hit else 'harmed' if old_hit else
                        'unmapped_gold' if not gold else 'retrieval_miss' if new_rank is None else 'ranking_miss')
            def brief(item):
                return {key:item.get(key) for key in ('chunk_id', 'rank', 'score', 'semantic_score',
                        'dimension_score', 'effective_semantic_score', 'semantic_score_imputed', 'score_components')} | {
                        'is_gold':item['chunk_id'] in gold, 'text':item.get('chunk_text', '')[:350]}
            badcases.append({**comparison, 'category':category,
                             'diagnostics':result['diagnostics'],
                             'gold_candidates':[brief(item) for item in new if item['chunk_id'] in gold],
                             'fusion_new_top5':[brief(item) for item in new[:5]]})
    report = {'snapshots': len(paths), 'strategy': 'raw_head_lexical', 'configuration':config}
    if comparisons:
        report.update(summarize(comparisons))
    for filename, records in [('comparison.jsonl', comparisons), ('badcase_analysis.jsonl', badcases)]:
        output = args.output_dir/filename
        if records:
            output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), encoding='utf-8')
        else:
            output.unlink(missing_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
