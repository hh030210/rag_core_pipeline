"""Replay the selected fusion strategy; labels are used only for reporting."""
from __future__ import annotations

import argparse
import json
import math
from uuid import uuid4
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from importlib import import_module
_dimension_score = import_module("rag_core.retrieval_fusion.experiments.00_method_comparison.replay_dimension")
ENGINE, ONLINE_DATASET = _dimension_score.ENGINE, _dimension_score.ONLINE_DATASET
fuse, grams = _dimension_score.fuse, _dimension_score.grams
first_rank, ndcg5 = _dimension_score.first_rank, _dimension_score.ndcg5

from importlib import import_module
_shared = import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.code.recommended_fusion')
rank = _shared.rank
LABEL = _shared.LABEL
CONFIG = _shared.CONFIG

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', type=Path, help='Optional input cohort; defaults to the dimension-score experiment saved inputs')
    parser.add_argument('--baseline-dataset', type=Path, default=_dimension_score.COMPARISON / 'output' / 'dimension_score' / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=_dimension_score.COMPARISON / 'output' / 'recommended_fusion')
    args = parser.parse_args(argv)
    def write(staged_output):
        args.output_dir = staged_output
        _run(args, parser)
    _dimension_score.replace_output(args.output_dir, write)


def _run(args, parser):
    args.input_dataset = args.input_dataset or args.baseline_dataset
    manifest = read_json(args.input_dataset / 'index.json')
    baseline = read_json(args.baseline_dataset / 'index.json')
    if set(manifest['detail_files']) != set(baseline['detail_files']):
        parser.error('Online and dimension-score datasets must contain the same query cohort')
    if args.input_dataset.resolve() != args.baseline_dataset.resolve() and baseline.get('input_batch_id') != manifest.get('batch_id'):
        parser.error('Baseline and online datasets must have the same batch_id')
    batch_id = uuid4().hex
    generated_at = datetime.now(timezone.utc).isoformat()
    index = dict(batch_id=batch_id, input_batch_id=manifest.get('batch_id'), baseline_batch_id=baseline.get('batch_id'), generated_at=generated_at, source='00_method_comparison recommended_fusion independent replay', label=LABEL, configuration=CONFIG, items={}, detail_files={}, metrics=[])
    rankings, golds, badcases = [], [], []
    counts = dict(rescued=0, harmed=0)
    dataset_dir = args.output_dir / 'dataset'
    details = dataset_dir / 'details'; details.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(manifest['items']):
        name = row['name']; detail = read_json(args.input_dataset / manifest['detail_files'][name])
        detail['routes']['new'] = read_json(args.baseline_dataset / baseline['detail_files'][name])['fusion_candidates']
        payload = rank(detail); candidates = payload['candidates']; gold = set(detail.get('gold', {}).get('gold_chunk_ids', []))
        new_rank = first_rank(candidates, gold)
        index['items'][name] = dict(recommended_rank=new_rank)
        payload['configuration'] = CONFIG
        payload['source_snapshot'] = name
        payload['batch_id'] = batch_id
        payload['name'] = name
        payload.update({k: detail[k] for k in ['query', 'retrieval_query', 'query_analysis', 'chunks', 'gold', 'routes']})
        relative = 'details/' + name; index['detail_files'][name] = relative
        (dataset_dir / relative).write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        old_hit = bool(gold) and any(c['chunk_id'] in gold for c in detail['routes']['new'][:5])
        hit = new_rank is not None and new_rank <= 5
        if gold:
            counts['rescued'] += hit and not old_hit; counts['harmed'] += old_hit and not hit
            if not hit: badcases.append(dict(name=name, query=detail['query'], rank=new_rank, candidate_pool_missing=new_rank is None))
        rankings.append(candidates); golds.append(gold)
    ranks = [first_rank(r, g) for r, g in zip(rankings, golds)]
    total = len(ranks); mapped = sum(bool(g) for g in golds)
    m = dict(hit_at_1=sum(r is not None and r <= 1 for r in ranks),
             hit_at_5=sum(r is not None and r <= 5 for r in ranks),
             hit_at_10=sum(r is not None and r <= 10 for r in ranks),
             mrr_at_10=sum(1 / r for r in ranks if r and r <= 10) / (total or 1),
             ndcg_at_5=sum(ndcg5(r, g) for r, g in zip(rankings, golds)) / (total or 1),
             mapped_queries=mapped, total_queries=total)
    index['report'] = dict(metrics=m, changes_vs_baseline=counts, remaining_badcases=len(badcases),
                           candidate_pool_missing=sum(b['candidate_pool_missing'] for b in badcases))
    index['metrics'] = [dict(key='recommended', label=LABEL,
         hits={'Hit@1': m['hit_at_1'], 'Hit@5': m['hit_at_5'], 'Hit@10': m['hit_at_10']},
         ranking_scores={'MRR@10': m['mrr_at_10'], 'nDCG@5': m['ndcg_at_5']})]
    index.pop('report', None)
    index.pop('metrics', None)
    (dataset_dir / 'index.json').write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps({'metrics': m, 'changes_vs_baseline': counts, 'remaining_badcases': len(badcases)}, ensure_ascii=False))


if __name__ == '__main__': main()
