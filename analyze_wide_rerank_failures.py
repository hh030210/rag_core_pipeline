"""Diagnose Top@5 misses from the wide-recall main-dimension reranker."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rag_core.dimension_labels import CanonicalLabelResolver


def _values(payload, dimension_id):
    value = payload.get(f"dim_{dimension_id}", [])
    return value if isinstance(value, list) else [value]


def _subject_matches(payload, spots):
    if not spots:
        return True
    explicit = str(payload.get("spot_name", "")).strip()
    source = str(payload.get("source_file", "")).split("-", 1)[0].strip()
    return explicit in spots or source in spots


def _compact(row, category, main_state):
    results = row.get("retrieval", {}).get("dimension_results", [])[:3]
    return {
        "id": row.get("id"),
        "question": row.get("question", ""),
        "category": category,
        "main_dimension": row.get("query_analysis", {}).get("routing", {}).get("main_dimension", ""),
        "main_state": main_state,
        "gold": row.get("gold", {}).get("chunk_ids", []),
        "first_gold_rank": row.get("retrieval_metrics", {}).get("dimension", {}).get("first_gold_rank"),
        "top3": [
            {
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "matched_dimensions": item.get("matched_dimensions", []),
                "doc_title": item.get("doc_title", ""),
            }
            for item in results
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in (args.evaluation_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    points = json.loads((args.run_dir / "local_points.json").read_text(encoding="utf-8"))
    payloads = {str(item.get("payload", {}).get("chunk_id", item.get("id"))): item.get("payload", {}) for item in points}
    by_parent = defaultdict(list)
    vocabulary = defaultdict(set)
    for payload in payloads.values():
        parent = str(payload.get("parent_doc_id") or payload.get("doc_id") or payload.get("chunk_id"))
        by_parent[parent].append(payload)
        for key, value in payload.items():
            if key.startswith("dim_"):
                vocabulary[key[4:]].update(str(item) for item in (value if isinstance(value, list) else [value]))
    resolver = CanonicalLabelResolver(args.run_dir, vocabulary)

    buckets = Counter()
    main_state_counts = Counter()
    examples = defaultdict(list)
    all_misses = 0
    mapped_misses = 0
    for row in rows:
        metrics = row.get("retrieval_metrics", {}).get("dimension", {})
        if metrics.get("hit_at_5"):
            continue
        all_misses += 1
        analysis = row.get("query_analysis", {})
        constraints = analysis.get("constraints", {}) or {}
        main = analysis.get("routing", {}).get("main_dimension", "")
        gold_ids = [str(value) for value in row.get("gold", {}).get("chunk_ids", [])]
        if not gold_ids:
            category = "Z_gold_unmapped"
            main_state = "gold_unmapped"
            buckets[category] += 1
            main_state_counts[main_state] += 1
            if len(examples[category]) < 4:
                examples[category].append(_compact(row, category, main_state))
            continue
        mapped_misses += 1
        gold_payloads = [payloads[value] for value in gold_ids if value in payloads]
        parents = {
            str(payload.get("parent_doc_id") or payload.get("doc_id") or payload.get("chunk_id"))
            for payload in gold_payloads
        }
        parent_payloads = [payload for parent in parents for payload in by_parent[parent]]
        q_constraint = constraints.get(main, {}) if main else {}
        q_labels = q_constraint.get("labels", [])
        parent_main_values = [value for payload in parent_payloads for value in _values(payload, main)] if main else []
        main_present = bool(parent_main_values)
        q_concepts = resolver.concepts(main, q_labels) if main else set()
        parent_concepts = resolver.concepts(main, parent_main_values) if main else set()
        label_overlap = bool(q_concepts & parent_concepts) if q_concepts else None
        if not constraints:
            main_state = "no_query_constraint"
        elif not main:
            main_state = "no_main_dimension"
        elif not main_present:
            main_state = "gold_parent_missing_main_tag"
        elif label_overlap is False:
            main_state = "same_dimension_label_mismatch"
        else:
            main_state = "main_dimension_available"
        main_state_counts[main_state] += 1

        rank = metrics.get("first_gold_rank")
        spot_excluded = bool(gold_payloads and analysis.get("spot_names") and not any(
            _subject_matches(payload, analysis["spot_names"]) for payload in gold_payloads
        ))
        if not constraints:
            category = "A_no_query_constraint"
        elif spot_excluded:
            category = "B_subject_filter_excludes_gold"
        elif rank is not None and rank <= 20:
            category = "C_rank_6_to_20"
        elif rank is not None:
            category = "D_rank_21_to_100"
        elif main_state == "gold_parent_missing_main_tag":
            category = "E_gold_parent_missing_main_tag"
        elif main_state == "same_dimension_label_mismatch":
            category = "F_same_dimension_label_mismatch"
        else:
            category = "G_candidate_pool_miss_other"
        buckets[category] += 1
        if len(examples[category]) < 4:
            examples[category].append(_compact(row, category, main_state))

    report = {
        "questions": len(rows),
        "top5_misses": all_misses,
        "mapped_top5_misses": mapped_misses,
        "primary_buckets": dict(buckets),
        "main_dimension_state_on_misses": dict(main_state_counts),
        "examples": dict(examples),
    }
    (args.evaluation_dir / "failure_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Wide Recall Main-Rerank Failure Analysis", "", f"- Top@5 misses: {all_misses}", "", "## Primary buckets", ""]
    for key, value in buckets.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Main-dimension state", ""])
    for key, value in main_state_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Examples", ""])
    for key, values in examples.items():
        lines.append(f"### {key}")
        lines.append("")
        for item in values:
            lines.append(f"- [{item['id']}] {item['question']}")
            lines.append(f"  - main={item['main_dimension'] or '(none)'}, state={item['main_state']}, rank={item['first_gold_rank']}")
            lines.append(f"  - gold={', '.join(item['gold'])}")
        lines.append("")
    (args.evaluation_dir / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
