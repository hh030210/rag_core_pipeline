"""Export reviewable retrieval results without embedding full chunk text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "result" / "evaluation_verified_fact_index_tail10_bge" / "results.jsonl"
PAIRED_SUMMARY = ROOT / "result" / "evaluation_verified_fact_index_tail10_paired_bge" / "summary.json"
FACT_MANIFEST = ROOT / "result" / "hybrid_legacy_tags_sdu_entities_20261002_verified_facts" / "fact_index_manifest.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for source, target in (
        (PAIRED_SUMMARY, OUTPUT_DIR / "fact_tail10_paired_summary.json"),
        (FACT_MANIFEST, OUTPUT_DIR / "fact_index_manifest.json"),
    ):
        target.write_text(json.dumps(read_json(source), ensure_ascii=False, indent=2), encoding="utf-8")

    compact = []
    for line in SOURCE_RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metric = row.get("retrieval_metrics", {}).get("dimension", {})
        analysis = row.get("query_analysis", {})
        candidates = row.get("retrieval", {}).get("dimension_results", [])[:15]
        compact.append({
            "id": row.get("id"),
            "question": row.get("question", ""),
            "retrieval_query": row.get("retrieval_query", ""),
            "gold_chunk_ids": row.get("gold", {}).get("chunk_ids", []),
            "first_gold_rank": metric.get("first_gold_rank"),
            "hit_at_15": metric.get("hit_at_15", False),
            "gold_recall_at_15": metric.get("gold_recall_at_15", 0.0),
            "query_fact_count": analysis.get("query_fact_count", 0),
            "verified_fact_match_count": analysis.get("verified_fact_match_count", 0),
            "verified_fact_index_candidate_count": analysis.get("verified_fact_index_candidate_count", 0),
            "top15": [
                {
                    "rank": item.get("rank"), "chunk_id": item.get("chunk_id"),
                    "source": item.get("source"), "score": item.get("score"),
                    "is_gold": item.get("is_gold", False),
                }
                for item in candidates
            ],
        })
    output = OUTPUT_DIR / "top15_results_compact.jsonl"
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in compact), encoding="utf-8"
    )
    print(f"Exported {len(compact)} retrieval rows to {output}")


if __name__ == "__main__":
    main()
