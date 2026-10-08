#!/usr/bin/env python3
"""Gold-free RRF reranking of previously judged candidates.

The ranking stage never reads gold labels.  Gold is only consumed by the
evaluation summary after the final ranking has been produced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("base_rerank", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def retrieval_rrf(item: dict[str, Any], k: int) -> float:
    semantic_rank = int(item.get("semantic_rank") or 999)
    dimension_rank = int(item.get("dimension_rank") or 999)
    score = 0.0
    if semantic_rank < 999:
        score += 1.0 / (k + semantic_rank)
    if dimension_rank < 999:
        score += 1.0 / (k + dimension_rank)
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(Path(__file__).with_name("rerank_ablation_v3_20260922.py"))
    rows = base.load_rows(args.input)
    output = []
    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        candidates = [dict(x) for x in row.get("rerank_candidates", [])]
        # These two rankings use only the candidate's model quality score and
        # the original semantic/dimension ranks.  No gold field is accessed.
        quality_order = sorted(candidates, key=lambda x: (float(x.get("quality_score", 0.0)), float(x.get("base_score", 0.0))), reverse=True)
        retrieval_order = sorted(candidates, key=lambda x: (retrieval_rrf(x, args.rrf_k), float(x.get("base_score", 0.0))), reverse=True)
        quality_rank = {str(x.get("chunk_id")): i + 1 for i, x in enumerate(quality_order)}
        retrieval_rank = {str(x.get("chunk_id")): i + 1 for i, x in enumerate(retrieval_order)}
        for item in candidates:
            cid = str(item.get("chunk_id"))
            item["quality_rank"] = quality_rank[cid]
            item["retrieval_rrf"] = retrieval_rrf(item, args.rrf_k)
            item["retrieval_rank"] = retrieval_rank[cid]
            # Standard, fixed RRF over the two rank lists.  No score weight is
            # tuned on the evaluation golden labels.
            item["gold_free_rrf_score"] = 1.0 / (args.rrf_k + item["quality_rank"]) + 1.0 / (args.rrf_k + item["retrieval_rank"])
        candidates.sort(key=lambda x: (float(x.get("gold_free_rrf_score", 0.0)), float(x.get("base_score", 0.0))), reverse=True)
        row["rerank_strategy"] = "gold_free_rrf_v1"
        row["rerank_candidates"] = candidates
        row["rerank_results"] = candidates[:5]
        output.append(row)
    summary = base.summarize_final(output, "rerank_results")
    summary.update({
        "strategy": "gold_free_rrf_v1",
        "status": "completed",
        "total": len(output),
        "rrf_k": args.rrf_k,
        "gold_used_by_ranking": False,
        "source": str(args.input),
    })
    (args.output_dir / "gold_free_rrf.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in output) + "\n", encoding="utf-8")
    (args.output_dir / "gold_free_rrf.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
