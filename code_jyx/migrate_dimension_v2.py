"""Non-destructive CLI for v2 schema/tag/index dry-runs."""

from __future__ import annotations

import argparse
import json

try:
    from .v2_pipeline import run_dry_run
except ImportError:  # 支持 python code_jyx/migrate_dimension_v2.py 直接执行
    from v2_pipeline import run_dry_run


def main():
    parser = argparse.ArgumentParser(description="Run a non-destructive dimension-index v2 dry-run")
    parser.add_argument("--source", default="", help="text directory/file used for mock extraction")
    parser.add_argument("--schema", dest="schema_path", default="", help="v2 or legacy schema JSON")
    parser.add_argument("--tags", dest="tags_path", default="", help="v2 or legacy tags JSON")
    parser.add_argument("--output-dir", default="./experiment_data/dimension_v2_dry_run")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    report = run_dry_run(
        source=args.source or None,
        schema_path=args.schema_path or None,
        tags_path=args.tags_path or None,
        output_dir=args.output_dir,
        limit=args.limit,
        mock=not bool(args.tags_path),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
