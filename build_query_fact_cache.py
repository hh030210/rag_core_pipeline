"""Pre-parse evaluation queries into controlled fact constraints.

Keeping this cache separate makes retrieval deterministic and avoids calling an
LLM inside the evaluator.  The cache is keyed by final retrieval query rather
than data-set id, because the evaluation set contains duplicate ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_jyx.llm_service import DimensionMiningWithQwen
from rag_core.evaluation import load_gold_records


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--use-rewritten-query", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {"queries": []}
    entries = existing.get("queries", []) if isinstance(existing, dict) else []
    cache = {str(item.get("query", "")): item for item in entries if isinstance(item, dict)}
    queries = []
    for record in load_gold_records(args.dataset):
        query = record.get("rewritten_query", "") if args.use_rewritten_query else ""
        query = str(query or record.get("question", "")).strip()
        if query and query not in cache and query not in queries:
            queries.append(query)
    if args.limit:
        queries = queries[:args.limit]
    miner = DimensionMiningWithQwen()
    total = len(queries)
    for completed, query in enumerate(queries, 1):
        cache[query] = {"query": query, "facts": miner.extract_query_facts_v2(query)}
        output = {"schema_version": "query_fact_cache_v1", "queries": list(cache.values())}
        save_json(args.output, output)
        print(f"[查询事实] {completed}/{total}", flush=True)
    save_json(args.output, {"schema_version": "query_fact_cache_v1", "queries": list(cache.values())})


if __name__ == "__main__":
    main()
