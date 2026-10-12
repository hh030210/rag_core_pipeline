#!/usr/bin/env python3
"""Run the main run.py evaluate command and publish only its selected strategy output."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

FUSION_DIR = Path(__file__).resolve().parent
REPOSITORY = FUSION_DIR.parents[1]
if not __package__:
    sys.path.insert(0, str(REPOSITORY))
DEFAULT_RUN_DIR = Path('/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1')
DEFAULT_MODEL = '/home/humq/rag_db_silm/model/bge-m3'


def output_dataset(name):
    return FUSION_DIR / 'experiments' / '00_method_comparison' / 'output' / name


def load_manifest(dataset):
    manifest = json.loads((dataset / 'index.json').read_text(encoding='utf-8'))
    if not manifest.get('batch_id') or not manifest.get('detail_files'):
        raise RuntimeError(f'Incomplete dataset: {dataset}')
    for relative in manifest['detail_files'].values():
        detail_path = (dataset / relative).resolve()
        if not detail_path.is_relative_to(dataset.resolve()):
            raise RuntimeError(f'Detail path escapes dataset: {relative}')
        detail = json.loads(detail_path.read_text(encoding='utf-8'))
        if detail.get('batch_id') != manifest['batch_id']:
            raise RuntimeError(f'Mixed batches in {detail_path}')
    return manifest


def build_steps(args):
    run_dir = args.run_dir.resolve()
    dataset = (args.dataset or run_dir / 'qa_dataset_435.json').resolve()
    sm_path = run_dir / 'store_manifest.json'
    store = json.loads(sm_path.read_text(encoding='utf-8')) if sm_path.exists() else {}
    from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY
    online = output_dataset(Path(FUSION_STRATEGY).stem)
    evaluate = [args.python, str(REPOSITORY / 'run.py'), 'evaluate',
                '--run-dir', str(run_dir), '--dataset', str(dataset),
                '--backend', 'qdrant', '--qdrant-url', args.qdrant_url or store.get('qdrant_url', 'http://127.0.0.1:6333'),
                '--collection', args.collection or store.get('collection', 'rag_core_sdu_v4pro_full_20260921_run1'),
                '--model-path', args.model_path, '--top-k', str(args.top_k),
                '--eval-depth', str(args.eval_depth), '--ks', args.ks,
                '--query-parser-mode', args.query_parser_mode]
    if args.no_query_expansion:
        evaluate.append('--no-query-expansion')
    return [('evaluate', evaluate, online)]



def run_steps(steps, dry_run=False):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    started = time.perf_counter()
    for position, (name, command, dataset) in enumerate(steps, 1):
        print(f'\n[{position}/{len(steps)}] {name}\n{shlex.join(command)}', flush=True)
        if dry_run:
            continue
        old_path = dataset / 'index.json'
        old_batch = json.loads(old_path.read_text(encoding='utf-8')).get('batch_id') if old_path.exists() else None
        subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)
        current = load_manifest(dataset)
        if current['batch_id'] == old_batch:
            raise RuntimeError(f'{name} did not publish a fresh batch; stopping')
        print(f'[{position}/{len(steps)}] completed: queries={len(current["detail_files"])} batch={current["batch_id"]}', flush=True)
    if not dry_run:
        print(f'\nMainline evaluation updated in {time.perf_counter()-started:.1f}s. Refresh visualization to load the new data.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--dataset', type=Path, default=None, help='Default: <run-dir>/qa_dataset_435.json')
    parser.add_argument('--model-path', default=DEFAULT_MODEL)
    parser.add_argument('--python', default=sys.executable, help='Python interpreter used for evaluate')
    parser.add_argument('--qdrant-url', default=None, help='Default: store_manifest.json value')
    parser.add_argument('--collection', default=None, help='Default: store_manifest.json value')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--eval-depth', type=int, default=20)
    parser.add_argument('--ks', default='1,5,10,15')
    parser.add_argument('--query-parser-mode', choices=('llm', 'deterministic'), default='deterministic')
    parser.add_argument('--no-query-expansion', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the evaluate command without updating data')
    args = parser.parse_args(argv)
    if args.top_k <= 0 or args.eval_depth < args.top_k:
        parser.error('Require eval-depth >= top-k > 0')
    try:
        run_steps(build_steps(args), dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(f'Evaluation failed (exit {exc.returncode}); no fresh batch was confirmed.', file=sys.stderr, flush=True)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Workflow stopped: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
