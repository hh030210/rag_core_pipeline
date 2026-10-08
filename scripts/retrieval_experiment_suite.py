#!/usr/bin/env python3
"""Run offline retrieval experiments from a saved wide-recall JSONL export.

The script deliberately does not call an LLM.  It reuses the saved semantic
and dimension rankings to compare K, candidate-pool prefixes, fusion rules,
adaptive K policies, truncation loss, question types, and no-hit cases.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


K_VALUES = (5, 10, 15, 20, 30, 50, 75, 96, 100)
SEMANTIC_POOLS = (20, 50, 100)
DIMENSION_POOLS = (5, 10, 20)
ADAPTIVE_KS = (5, 10, 20, 50, 96)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def item_id(item: Any) -> str:
    return str(item.get("chunk_id", "") if isinstance(item, dict) else item)


def result_score(item: Any) -> float:
    if not isinstance(item, dict):
        return 0.0
    for key in ("score", "fused_score", "final_score"):
        try:
            return float(item.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    return 0.0


def route_items(row: dict[str, Any], route: str) -> list[dict[str, Any]]:
    return list(row.get(f"{route}_results") or [])


def gold(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("gold_chunk_ids", []) if str(value)}


def metrics(rows: Iterable[dict[str, Any]], rankings: Iterable[list[str]], k: int) -> dict[str, Any]:
    rows = list(rows)
    rankings = list(rankings)
    valid = [row for row in rows if gold(row)]
    any_hits = 0
    gold_hits = 0
    gold_total = sum(len(gold(row)) for row in valid)
    reciprocal = []
    for row, ranking in zip(rows, rankings):
        expected = gold(row)
        if not expected:
            continue
        top = ranking[:k]
        top_set = set(top)
        hit = expected & top_set
        any_hits += int(bool(hit))
        gold_hits += len(hit)
        first = next((index + 1 for index, value in enumerate(top) if value in expected), None)
        reciprocal.append(1.0 / first if first else 0.0)
    denominator = len(valid)
    return {
        "valid_questions": denominator,
        "hit_count": any_hits,
        "hit_rate": round(any_hits / denominator, 8) if denominator else 0.0,
        "gold_hits": gold_hits,
        "gold_total": gold_total,
        "gold_recall": round(gold_hits / gold_total, 8) if gold_total else 0.0,
        "mrr": round(sum(reciprocal) / len(reciprocal), 8) if reciprocal else 0.0,
    }


def normalize(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    scores = [result_score(item) for item in items]
    low, high = min(scores), max(scores)
    if high == low:
        return {item_id(item): 1.0 for item in items}
    return {item_id(item): (result_score(item) - low) / (high - low) for item in items}


def weighted_fusion(semantic: list[dict[str, Any]], dimension: list[dict[str, Any]], alpha: float = 0.2) -> list[str]:
    sem_norm = normalize(semantic)
    dim_norm = normalize(dimension)
    ids = set(sem_norm) | set(dim_norm)
    return [
        chunk_id
        for chunk_id, _ in sorted(
            ((chunk_id, (1 - alpha) * sem_norm.get(chunk_id, 0.0) + alpha * dim_norm.get(chunk_id, 0.0))
             for chunk_id in ids),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def rrf_fusion(semantic: list[dict[str, Any]], dimension: list[dict[str, Any]], constant: int = 60) -> list[str]:
    sem_rank = {item_id(item): index + 1 for index, item in enumerate(semantic)}
    dim_rank = {item_id(item): index + 1 for index, item in enumerate(dimension)}
    ids = set(sem_rank) | set(dim_rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(
            ((chunk_id, 1 / (constant + sem_rank.get(chunk_id, constant + 10000))
             + 1 / (constant + dim_rank.get(chunk_id, constant + 10000))) for chunk_id in ids),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def interleave_fusion(semantic: list[dict[str, Any]], dimension: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(semantic), len(dimension))):
        for route in (semantic, dimension):
            if index >= len(route):
                continue
            chunk_id = item_id(route[index])
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                output.append(chunk_id)
    return output


def dataset_meta(dataset_path: Path | None) -> dict[str, dict[str, Any]]:
    if not dataset_path or not dataset_path.exists():
        return {}
    raw = read_json(dataset_path)
    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("data", raw.get("items", [])))
    output = {}
    for index, item in enumerate(raw if isinstance(raw, list) else [], 1):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id", item.get("question_id", item.get("qid", index))))
        output[key] = item
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-jsonl", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.retrieval_jsonl))
    metadata = dataset_meta(Path(args.dataset) if args.dataset else None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    routes = {route: [
        [item_id(item) for item in route_items(row, route)] for row in rows
    ] for route in ("semantic", "dimension", "fusion")}
    k_sweep = {
        str(k): {route: metrics(rows, rankings, k) for route, rankings in routes.items()}
        for k in K_VALUES
    }

    pool_sweep: dict[str, Any] = {}
    for sem_pool in SEMANTIC_POOLS:
        for dim_pool in DIMENSION_POOLS:
            rankings = []
            for row in rows:
                semantic = route_items(row, "semantic")[:sem_pool]
                dimension = route_items(row, "dimension")[:dim_pool]
                rankings.append(weighted_fusion(semantic, dimension))
            key = f"semantic_{sem_pool}_dimension_{dim_pool}"
            pool_sweep[key] = metrics(rows, rankings, 5)

    algorithm_rankings = {
        "semantic_only": routes["semantic"],
        "dimension_only": routes["dimension"],
        "weighted_score_alpha_0.2": [],
        "rrf_k60": [],
        "interleave_union": [],
    }
    for row in rows:
        semantic = route_items(row, "semantic")
        dimension = route_items(row, "dimension")
        algorithm_rankings["weighted_score_alpha_0.2"].append(weighted_fusion(semantic, dimension, 0.2))
        algorithm_rankings["rrf_k60"].append(rrf_fusion(semantic, dimension))
        algorithm_rankings["interleave_union"].append(interleave_fusion(semantic, dimension))
    algorithm_comparison = {
        name: metrics(rows, ranking, 5) for name, ranking in algorithm_rankings.items()
    }

    adaptive: dict[str, Any] = {}
    for threshold in (0.01, 0.03, 0.05, 0.10):
        chosen_rankings = []
        chosen_ks = []
        for row in rows:
            items = route_items(row, "fusion")
            scores = [result_score(item) for item in items]
            chosen = ADAPTIVE_KS[-1]
            for candidate in ADAPTIVE_KS[:-1]:
                if len(scores) <= candidate:
                    chosen = candidate
                    break
                gap = scores[candidate - 1] - scores[candidate]
                if gap >= threshold:
                    chosen = candidate
                    break
            chosen_ks.append(chosen)
            chosen_rankings.append([item_id(item) for item in items])
        valid_rows = [row for row in rows if gold(row)]
        hit_count = gold_hits = 0
        gold_total = sum(len(gold(row)) for row in valid_rows)
        reciprocal = []
        for row, ranking, chosen in zip(rows, chosen_rankings, chosen_ks):
            expected = gold(row)
            if not expected:
                continue
            top = ranking[:chosen]
            hit = expected & set(top)
            hit_count += int(bool(hit))
            gold_hits += len(hit)
            first = next((index + 1 for index, value in enumerate(top) if value in expected), None)
            reciprocal.append(1.0 / first if first else 0.0)
        hit_metrics = {
            "valid_questions": len(valid_rows),
            "hit_count": hit_count,
            "hit_rate": round(hit_count / len(valid_rows), 8) if valid_rows else 0.0,
            "gold_hits": gold_hits,
            "gold_total": gold_total,
            "gold_recall": round(gold_hits / gold_total, 8) if gold_total else 0.0,
            "mrr": round(sum(reciprocal) / len(reciprocal), 8) if reciprocal else 0.0,
        }
        adaptive[f"margin_{threshold:.2f}"] = {
            **hit_metrics,
            "average_k": round(sum(chosen_ks) / len(chosen_ks), 4) if chosen_ks else 0.0,
            "k_distribution": dict(sorted(Counter(chosen_ks).items())),
        }

    truncation_counts = Counter()
    truncation_examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected = gold(row)
        if not expected:
            continue
        semantic = set(routes["semantic"][index])
        dimension = set(routes["dimension"][index])
        union = semantic | dimension
        fusion_top5 = set(routes["fusion"][index][:5])
        truncated = (expected & union) - fusion_top5
        if truncated:
            truncation_counts["gold_in_union_but_not_fusion_top5"] += len(truncated)
            if len(truncation_examples) < 50:
                truncation_examples.append({
                    "id": row.get("id"),
                    "question": row.get("question", ""),
                    "gold_chunk_ids": sorted(expected),
                    "truncated_gold_chunk_ids": sorted(truncated),
                    "semantic_ranks": {chunk_id: routes["semantic"][index].index(chunk_id) + 1
                                       for chunk_id in truncated if chunk_id in routes["semantic"][index]},
                    "dimension_ranks": {chunk_id: routes["dimension"][index].index(chunk_id) + 1
                                        for chunk_id in truncated if chunk_id in routes["dimension"][index]},
                    "fusion_top5": routes["fusion"][index][:5],
                })
        truncation_counts["questions_with_truncation"] += int(bool(truncated))

    type_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        item = metadata.get(str(row.get("id")), {})
        question_type = str(item.get("question_type") or row.get("question_type") or "(unknown)")
        type_groups[question_type].append(index)
    question_types = {}
    for question_type, indexes in sorted(type_groups.items()):
        subset_rows = [rows[index] for index in indexes]
        subset_rankings = [routes["fusion"][index] for index in indexes]
        question_types[question_type] = {
            "count": len(subset_rows),
            "fusion_top5": metrics(subset_rows, subset_rankings, 5),
        }

    no_hit = []
    for index, row in enumerate(rows):
        expected = gold(row)
        if expected and not (expected & (set(routes["semantic"][index]) | set(routes["dimension"][index]))):
            no_hit.append({"id": row.get("id"), "question": row.get("question", ""),
                           "gold_chunk_ids": sorted(expected)})

    report = {
        "source": str(args.retrieval_jsonl),
        "row_count": len(rows),
        "gold_question_count": sum(bool(gold(row)) for row in rows),
        "gold_chunk_count": sum(len(gold(row)) for row in rows),
        "k_sweep": k_sweep,
        "candidate_pool_sweep_top5": pool_sweep,
        "fusion_algorithm_top5": algorithm_comparison,
        "adaptive_k": adaptive,
        "fusion_truncation": {
            **dict(truncation_counts),
            "examples": truncation_examples,
        },
        "question_type_top5": question_types,
        "unreachable_no_hit": {
            "count": len(no_hit),
            "examples": no_hit[:100],
        },
    }
    (output_dir / "retrieval_experiment_suite.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "retrieval_experiment_suite.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 离线检索实验汇总", "",
        f"- 样本数：{report['row_count']}",
        f"- 有 golden 的问题数：{report['gold_question_count']}",
        f"- golden chunk 数：{report['gold_chunk_count']}", "",
        "## K 扫描（至少命中一个 golden chunk）", "",
        "| K | 语义命中率 | 维度命中率 | 融合命中率 |", "|---:|---:|---:|---:|",
    ]
    for k, item in report["k_sweep"].items():
        lines.append("| {} | {:.2%} | {:.2%} | {:.2%} |".format(
            k, item["semantic"]["hit_rate"], item["dimension"]["hit_rate"], item["fusion"]["hit_rate"]))
    lines.extend(["", "## 融合算法 Top-5", "", "| 方法 | Hit Rate | Golden Recall | MRR |", "|---|---:|---:|---:|"])
    for name, item in report["fusion_algorithm_top5"].items():
        lines.append("| {} | {:.2%} | {:.2%} | {:.4f} |".format(
            name, item["hit_rate"], item["gold_recall"], item["mrr"]))
    truncation = report["fusion_truncation"]
    lines.extend(["", "## 融合截断", "",
                  f"- 有 golden 在两路候选并集但未进入融合 Top-5 的问题数：{truncation.get('questions_with_truncation', 0)}",
                  f"- 被截断的 golden chunk 数：{truncation.get('gold_in_union_but_not_fusion_top5', 0)}",
                  "", "## 未命中", "",
                  f"- 两路完整候选并集都未命中的问题数：{report['unreachable_no_hit']['count']}", ""])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
