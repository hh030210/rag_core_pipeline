"""Export the best dimension-only candidates for downstream fusion tests.

The public candidate file intentionally excludes every Golden-related field.
The paired evaluation file is for offline metric calculation only and must not
be used as input to a live fusion ranker.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


CANDIDATE_FIELDS = (
    "chunk_id", "rank", "score", "final_score", "fused_score",
    "dimension_score", "normalized_dimension_score", "source", "dim_rank",
    "doc_id", "parent_doc_id", "doc_title", "chunk_gen_title", "source_file",
    "spot_name", "dimension_paths", "matched_dimensions", "matches",
    "fact_anchor_bonus", "fact_anchor_matches", "chunk_text", "chunk_text_full",
    "text_truncated", "base_dimension_score", "main_dimension_hit",
    "direct_parent_coverage", "main_coverage_rerank_bonus",
    "verified_fact_evidence_matches", "verified_fact_evidence_count",
    "verified_fact_evidence_rerank_bonus", "answer_evidence_matches",
    "answer_evidence_rerank_bonus", "answerability_entity_matches",
    "answerability_attribute_matches", "answerability_relation_count",
    "answerability_rerank_bonus", "slot_coverage_total", "slot_coverage_count",
    "slot_coverage_ratio", "slot_coverage_entities", "slot_coverage_attributes",
    "slot_coverage_relations", "slot_coverage_rerank_bonus",
)


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep fusion-relevant retrieval fields and drop evaluation leakage."""
    result = {name: candidate[name] for name in CANDIDATE_FIELDS if name in candidate}
    # Provide one stable text field even if historical output uses chunk_text.
    if "chunk_text_full" not in result and "chunk_text" in result:
        result["chunk_text_full"] = result["chunk_text"]
    return result


def export(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    no_gold_path = output_dir / "dimension_candidates_top20_for_fusion.jsonl"
    eval_path = output_dir / "dimension_candidates_top20_with_gold_for_eval.jsonl"
    count = 0
    with source.open("r", encoding="utf-8") as reader, \
            no_gold_path.open("w", encoding="utf-8") as no_gold, \
            eval_path.open("w", encoding="utf-8") as with_gold:
        for line in reader:
            row = json.loads(line)
            candidates = [
                compact_candidate(item)
                for item in (row.get("retrieval", {}).get("dimension_results", []) or [])[:20]
            ]
            payload = {
                "id": row.get("id"),
                "question": row.get("question", ""),
                "retrieval_query": row.get("retrieval_query", row.get("question", "")),
                "query_analysis": row.get("query_analysis", {}),
                "dimension_candidate_count": len(candidates),
                "dimension_candidates": candidates,
            }
            no_gold.write(json.dumps(payload, ensure_ascii=False) + "\n")
            evaluation_payload = {
                **payload,
                "gold": row.get("gold", {}),
                "reference_answer": row.get("reference_answer", ""),
            }
            with_gold.write(json.dumps(evaluation_payload, ensure_ascii=False) + "\n")
            count += 1

    summary_source = source.parent / "summary.json"
    shutil.copy2(summary_source, output_dir / "best_dimension_retrieval_summary.json")
    manifest = {
        "format_version": "dimension-fusion-export-v1",
        "source_result": str(source),
        "candidate_route": "dimension",
        "top_k": 20,
        "query_count": count,
        "no_gold_file": no_gold_path.name,
        "evaluation_file": eval_path.name,
        "warning": "Do not use the evaluation file, which contains Golden labels, as fusion-model input.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.output_dir)


if __name__ == "__main__":
    main()
