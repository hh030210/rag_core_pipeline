"""Static gap-aware lexical fusion for saved retrieval snapshots.
Gold labels are read only for reporting, after ranking.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent


def grams(text):
    result = set()
    for token in re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', text.lower()):
        for width in (2, 3, 4):
            result.update(token[i:i+width] for i in range(max(0, len(token)-width+1)))
    return result


def first_rank(items, gold):
    return next((i for i, item in enumerate(items, 1) if item['chunk_id'] in gold), None)


def fuse(snapshot):
    sem = snapshot['semantic_candidates']
    dim = snapshot['dimension_candidates']
    old = snapshot['fusion_candidates']
    gap = float(sem[0]['score'])-float(sem[1]['score']) if len(sem)>1 else 0.0
    alpha = min(0.7, 0.2+0.1*max(0.0, 1.0-gap/0.05))
    sm = {str(x['chunk_id']): x for x in sem}
    dm = {str(x['chunk_id']): x for x in dim}
    qgrams = grams(snapshot.get('query', ''))
    bags = [grams(str(x.get('chunk_text', ''))[:4000]) for x in old]
    count = len(old)
    df = Counter(g for bag in bags for g in bag if g in qgrams)
    weights = {g: math.log((count+1)/(df[g]+0.5)) for g in qgrams}
    total = sum(weights.values()) or 1.0
    lexical = [sum(weights[g] for g in qgrams & bag)/total for bag in bags]
    max_lexical = max(lexical, default=0.0)
    ranked = []
    for old_rank, (item, lex) in enumerate(zip(old, lexical), 1):
        cid = str(item['chunk_id'])
        s, d = sm.get(cid, {}), dm.get(cid, {})
        sn = float(s.get('normalized_semantic_score', 0.0))
        dn = float(d.get('normalized_dimension_score', 0.0))
        ln = lex/max_lexical if max_lexical else 0.0
        score = (1-alpha)*sn+alpha*dn+0.3*ln
        ranked.append({'chunk_id': cid, 'score': round(score, 10),
                       'semantic_rank': s.get('rank'), 'dimension_rank': d.get('rank'),
                       'semantic_score': s.get('score'), 'dimension_score': d.get('score'),
                       'normalized_semantic_score': round(sn, 8),
                       'normalized_dimension_score': round(dn, 8),
                       'lexical_score': round(ln, 8),
                       'original_fusion_rank': old_rank,
                       'original_fusion_score': item.get('score'),
                       'doc_title': item.get('doc_title'),
                       'source_file': item.get('source_file'),
                       'chunk_text': item.get('chunk_text', '')})
    ranked.sort(key=lambda x: (-x['score'], x['original_fusion_rank']))
    for rank, item in enumerate(ranked, 1):
        item['rank'] = rank
    top_k = int(snapshot.get('top_k', 10))
    return {'query': snapshot.get('query'), 'strategy': 'gap_aware_lexical_v1',
            'semantic_top1_gap': round(gap, 8),
            'effective_dimension_weight': round(alpha, 8),
            'lexical_weight': 0.3, 'top_k': top_k,
            'fusion_candidates': ranked, 'fusion_results': ranked[:top_k]}


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
    parser.add_argument('--evaluation-results', type=Path,
                        help='Optional labels for reporting; never used in ranking')
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob('retrieval_fusion_*.json'))
    if not paths:
        parser.error(f'No snapshots in {args.input_dir}')
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error('Input and output must differ')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.evaluation_results) if args.evaluation_results else {}
    comparisons = []
    for path in paths:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        result = fuse(snapshot)
        result['source_snapshot'] = str(path.resolve())
        (args.output_dir/path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        gold = labels.get(result['query'])
        if gold is None:
            continue
        old = snapshot['fusion_candidates']
        new = result['fusion_candidates']
        comparisons.append({
            'query': result['query'], 'source_snapshot': path.name,
            'gold_count': len(gold),
            'semantic_rank': first_rank(snapshot['semantic_candidates'], gold),
            'dimension_rank': first_rank(snapshot['dimension_candidates'], gold),
            'old_rank': first_rank(old, gold), 'new_rank': first_rank(new, gold),
            'old_ndcg5': ndcg5(old, gold), 'new_ndcg5': ndcg5(new, gold),
            'semantic_top1_gap': result['semantic_top1_gap']})
    report = {'snapshots': len(paths)}
    if comparisons:
        report.update(summarize(comparisons))
        (args.output_dir/'comparison.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in comparisons), encoding='utf-8')
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
