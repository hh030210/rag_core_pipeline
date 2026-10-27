"""Portable init/review/resume wrapper around byte-identical run33 sources.

No SSH, credentials, scheduler, automatic approvals or archived answer reuse.
"""
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
ARCHIVE = HERE.parents[1] / 'results/scenic_context_first_knowledge_deepseek_v4_pro_20261026_run33/audit_archive.zip'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def digest(value):
    return hashlib.sha256((value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()


def init(args):
    assert not args.run.exists(), 'Use a new directory; archived results must not be overwritten'
    assert Path(args.embedding).is_dir(), 'Embedding model directory is required'
    args.run.mkdir(parents=True)
    runtime = args.run / 'runtime'
    runtime.mkdir()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.txt','.md') and p.name!='launch_collective.py':
            shutil.copyfile(p,runtime/p.name)
    cfg = read(HERE/'config.original.json')
    cfg.update(base_url=args.base_url.rstrip('/'), embedding_model=str(Path(args.embedding).resolve()),
               machine_python=sys.executable, library_code=str(runtime), data_source='bundled_shared_inputs')
    # Keep model, budgets, metas, stopping, clustering and judging rules unchanged.
    cfg['workers'] = args.workers
    save(runtime/'config.json',cfg)
    shared=args.run/'shared';shared.mkdir()
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in ('train_top5.json','test_top5.json'):
            (shared/name).write_bytes(z.read('shared/'+name))
    train,test=read(shared/'train_top5.json'),read(shared/'test_top5.json')
    assert len(train)==23 and len(test)==378
    assert not {r['question'] for r in train}&{r['question'] for r in test}
    save(args.run/'frozen_manifest.json',{'config':cfg,
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.iterdir() if p.is_file()},
         'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in shared.iterdir()},
         'created_at':time.time(),'publication_wrapper':'portable init; no archived answers/scores/prompts copied'})
    print('Initialized fresh run. Next: run; inspect each review gate before approve.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('init','run','status','approve'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--embedding')
    parser.add_argument('--base-url',default=os.getenv('OPENAI_BASE_URL','http://127.0.0.1:8910/v1'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--gate',choices=('stage','cluster','feedback_stage','feedback'))
    parser.add_argument('--reviewed',action='store_true')
    args=parser.parse_args();args.run=args.run.resolve()
    if args.command=='init':
        assert args.embedding and args.workers>0
        return init(args)
    assert (args.run/'runtime/config.json').exists(), 'Initialize a fresh run first'
    if args.command=='status':
        p=args.run/'job_status.json'
        print(p.read_text(encoding='utf-8') if p.exists() else 'initialized; not started')
    elif args.command=='approve':
        assert args.gate and args.reviewed, 'Explicit --gate and --reviewed are required after inspecting prompts'
        review=read(args.run/(args.gate+'_review.json'))
        save(args.run/(args.gate+'_approval.json'),{'artifact_sha256':review['artifact_sha256'],
             'approved_at':time.time(),'checked_by':'operator_semantic_review','no_performance_selection':True})
        print('Approved exact artifact hash; run again to continue.')
    else:
        env=dict(os.environ,RESTORED_RUN=str(args.run),PYTHONIOENCODING='utf-8')
        os.execve(sys.executable,[sys.executable,str(args.run/'runtime/knowledge_fallback_workflow.py')],env)


if __name__=='__main__':
    main()
