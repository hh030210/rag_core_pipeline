"""Replay all fusion methods from one saved input batch, without running evaluate."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
from importlib import import_module

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
COMPARISON = Path(__file__).resolve().parents[1]
PACKAGE = 'rag_core.retrieval_fusion.experiments.00_method_comparison.code'


def available_methods():
    return sorted(p.stem for p in Path(__file__).parent.glob('*.py') if p.name != 'run_comparison_methods.py')


def main(argv=None):
    from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', type=Path, default=COMPARISON/'output'/Path(FUSION_STRATEGY).stem)
    parser.add_argument('--output-dir', type=Path, default=COMPARISON/'output')
    parser.add_argument('--methods', nargs='+', choices=available_methods(), default=available_methods())
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_k <= 0: parser.error('--top-k must be positive')
    source = args.input_dataset.resolve()
    manifest = json.loads((source/'index.json').read_text())
    # Freeze inputs before any method replaces its own output (possibly the source).
    with tempfile.TemporaryDirectory(prefix='fusion-comparison-input-') as temporary:
        frozen = Path(temporary)
        index = {**manifest, 'detail_files': {}}
        for name, relative in manifest['detail_files'].items():
            file = (source/relative).resolve()
            if Path(name).name != name or not file.is_relative_to(source): raise ValueError('Unsafe input path')
            shutil.copy2(file, frozen/name)
            index['detail_files'][name] = name
        (frozen/'index.json').write_text(json.dumps(index, ensure_ascii=False)+'\n')
        for method in args.methods:
            module = import_module(PACKAGE+'.'+method)
            module.run_comparison(frozen, args.output_dir/method, top_k=args.top_k)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
