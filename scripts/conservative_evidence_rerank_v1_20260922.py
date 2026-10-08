#!/usr/bin/env python3
"""Offline conservative rerank using already-computed answerability scores."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quality-weight", type=float, default=0.50)
    parser.add_argument("--base-weight", type=float, default=0.30)
    parser.add_argument("--route-weight", type=float, default=0.20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(Path(__file__).with_name("rerank_ablation_v3_20260922.py"))
    rows = base.load_rows(args.input)
    output = []
    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        candidates = []
        for item in row.get("rerank_candidates", []):
            copy = dict(item)
            route_top5 = int((copy.get("semantic_rank") or 999) <= 5 or (copy.get("dimension_rank") or 999) <= 5)
            copy["route_top5_evidence"] = route_top5
            copy["conservative_rerank_score"] = (
                args.quality_weight * float(copy.get("quality_score", 0.0))
                + args.base_weight * float(copy.get("base_score", 0.0))
                + args.route_weight * route_top5
            )
            candidates.append(copy)
        candidates.sort(key=lambda x: float(x.get("conservative_rerank_score", 0.0)), reverse=True)
        row["rerank_strategy"] = "conservative_evidence_rerank_v1"
        row["rerank_candidates"] = candidates
        row["rerank_results"] = candidates[:5]
        output.append(row)
    summary = base.summarize_final(output, "rerank_results")
    summary.update({
        "strategy": "conservative_evidence_rerank_v1",
        "status": "completed",
        "total": len(output),
        "score_formula": f"{args.quality_weight:.2f}*quality + {args.base_weight:.2f}*base + {args.route_weight:.2f}*route_top5_evidence",
        "source": str(args.input),
    })
    (args.output_dir / "conservative_evidence_rerank.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in output) + "\n", encoding="utf-8")
    (args.output_dir / "conservative_evidence_rerank.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
