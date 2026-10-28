"""Portable fresh-run wrapper for run34; no credentials or archived answer reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
NAME = 'scenic_ten_point_four_train_six_final_deepseek_v4_pro_20261027_run34'
ARCHIVE = HERE.parents[1] / 'results' / NAME / 'audit_archive.zip'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def init(args):
    assert not args.run.exists(), 'Use a new directory; never overwrite an experiment'
    assert Path(args.embedding).is_dir(), 'An existing local embedding model directory is required'
    cfg = read(HERE / 'config.original.json')
    assert cfg['protocol'] == 'ten_point_four_train_six_final_v1'
    with zipfile.ZipFile(ARCHIVE) as archive:
        data = {name: archive.read('shared/' + name) for name in ('train_top5.json', 'test_top5.json')}
    train, test = (json.loads(data[name]) for name in ('train_top5.json', 'test_top5.json'))
    assert len(train) == 23 and len(test) == 378
    assert all(len(row['context_documents']) == 5 for row in train + test)
    assert not {row['question'].strip() for row in train} & {row['question'].strip() for row in test}
    args.run.mkdir(parents=True)
    runtime = args.run / 'runtime'
    runtime.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.txt', '.md') and path.name != 'launch_ten_point.py':
            shutil.copyfile(path, runtime / path.name)
    cfg.update(base_url=args.base_url.rstrip('/'), embedding_model=str(Path(args.embedding).resolve()),
               machine_python=sys.executable, library_code=str(runtime),
               data_source='bundled_shared_inputs', workers=args.workers)
    save(runtime / 'config.json', cfg)
    shared = args.run / 'shared'
    shared.mkdir()
    for name, content in data.items():
        (shared / name).write_bytes(content)
    save(args.run / 'frozen_manifest.json', {
        'config': cfg,
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in runtime.iterdir() if path.is_file()},
        'input_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in shared.iterdir()},
        'created_at': time.time(),
        'publication_wrapper': 'Fresh inputs only; no archived answers/scores/stages/approvals copied'})
    print('Initialized fresh run34 protocol; run and inspect each semantic review gate before approving.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'run', 'status', 'approve'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--embedding')
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8910/v1'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gate', choices=('stage', 'cluster', 'feedback_stage', 'feedback'))
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.command == 'init':
        assert args.embedding and args.workers > 0
        return init(args)
    assert (args.run / 'runtime/config.json').exists(), 'Initialize a fresh run first'
    if args.command == 'status':
        path = args.run / 'job_status.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'initialized; not started')
    elif args.command == 'approve':
        assert args.gate and args.reviewed, 'Inspect artifacts, then specify --gate and --reviewed'
        review = read(args.run / (args.gate + '_review.json'))
        save(args.run / (args.gate + '_approval.json'), {
            'artifact_sha256': review['artifact_sha256'], 'approved_at': time.time(),
            'checked_by': 'operator_semantic_review', 'no_performance_selection': True})
        print('Approved exact artifact hash. Run again to continue.')
    else:
        env = dict(os.environ, RESTORED_RUN=str(args.run), PYTHONIOENCODING='utf-8')
        os.execve(sys.executable, [sys.executable, str(args.run / 'runtime/ten_point_workflow.py')], env)


if __name__ == '__main__':
    main()
