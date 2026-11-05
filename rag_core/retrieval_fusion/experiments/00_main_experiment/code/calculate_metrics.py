#!/usr/bin/env python3
"""Calculate unified chunk and document metrics from saved retrieval details."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"
K_VALUES = (1, 5, 10, 15, 20)
ROUTES = {
    "vector": "向量检索（用户查询语义）",
    "dimension": "维度检索（关键信息匹配）",
    "hybrid": "混合检索（融合排序）",
}
DETAIL_FILES = {
    "vector": "vector_retrieval_details.jsonl",
    "dimension": "dimension_retrieval_details.jsonl",
    "hybrid": "hybrid_retrieval_details.jsonl",
}
THRESHOLDS = {"recall_at_k_macro": 0.90, "doc_precision_at_k_micro": 0.85}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_query(record, k):
    gold_ids = set(map(str, record.get("gold_chunk_ids") or []))
    gold_docs = set(record.get("gold_documents") or [])
    retrieved = (record.get("retrieved") or [])[:k]
    relevant_ranks = [
        rank for rank, item in enumerate(retrieved, start=1)
        if str(item.get("chunk_id", "")) in gold_ids
    ]
    hit_count = len(relevant_ranks)
    recall = hit_count / len(gold_ids) if gold_ids else 0.0
    first_rank = relevant_ranks[0] if relevant_ranks else None
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg if idcg else 0.0
    doc_relevant = sum(
        1 for item in retrieved if item.get("source_file", "") in gold_docs
    )
    doc_precision = doc_relevant / len(retrieved) if retrieved else 0.0
    return {
        "hit_at_k": bool(relevant_ranks),
        "chunk_hits": hit_count,
        "gold_chunk_count": len(gold_ids),
        "recall_at_k": recall,
        "first_golden_rank_at_k": first_rank,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "doc_relevant_chunks": doc_relevant,
        "gold_doc_count": len(gold_docs),
        "retrieved_count": len(retrieved),
        "doc_precision_at_k": doc_precision,
        "all_golden_chunks_hit": hit_count == len(gold_ids) if gold_ids else False,
    }


def aggregate(rows):
    query_count = len(rows)
    gold_chunks = sum(row["gold_chunk_count"] for row in rows)
    chunk_hits = sum(row["chunk_hits"] for row in rows)
    gold_docs = sum(row["gold_doc_count"] for row in rows)
    retrieved = sum(row["retrieved_count"] for row in rows)
    doc_relevant = sum(row["doc_relevant_chunks"] for row in rows)
    return {
        "eligible_queries": query_count,
        "hit_at_k": (
            sum(row["hit_at_k"] for row in rows) / query_count if query_count else 0.0
        ),
        "recall_at_k_macro": (
            statistics.fmean(row["recall_at_k"] for row in rows) if rows else 0.0
        ),
        "recall_at_k_micro": chunk_hits / gold_chunks if gold_chunks else 0.0,
        "mrr_at_k": (
            statistics.fmean(row["mrr_at_k"] for row in rows) if rows else 0.0
        ),
        "ndcg_at_k": (
            statistics.fmean(row["ndcg_at_k"] for row in rows) if rows else 0.0
        ),
        "doc_precision_at_k_micro": (
            doc_relevant / retrieved if retrieved else 0.0
        ),
        "doc_precision_at_k_macro": (
            statistics.fmean(row["doc_precision_at_k"] for row in rows)
            if rows else 0.0
        ),
        "complete_chunk_query_rate": (
            sum(row["all_golden_chunks_hit"] for row in rows) / query_count
            if query_count else 0.0
        ),
        "gold_chunks": gold_chunks,
        "gold_documents": gold_docs,
        "retrieved_chunks": retrieved,
        "doc_relevant_chunks": doc_relevant,
        "mean_returned_per_query": (
            statistics.fmean(row["retrieved_count"] for row in rows) if rows else 0.0
        ),
    }


def percentage(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = read_json(args.output_dir / "details_manifest.json")
    route_data = {
        route: read_jsonl(args.output_dir / filename)
        for route, filename in DETAIL_FILES.items()
    }
    counts = {len(rows) for rows in route_data.values()}
    if len(counts) != 1 or counts != {manifest["source_snapshot_count"]}:
        raise RuntimeError("Three retrieval detail files have different row counts")

    reference = route_data["vector"]
    for route, rows in route_data.items():
        for index, (expected, actual) in enumerate(zip(reference, rows), start=1):
            if (
                expected.get("id") != actual.get("id")
                or expected.get("query") != actual.get("query")
            ):
                raise RuntimeError(f"Route detail alignment failed: {route} row {index}")

    eligible_indices = [
        index for index, row in enumerate(reference)
        if row.get("answerable") and row.get("gold_mapped")
        and row.get("gold_chunk_ids")
    ]
    excluded_no_gold = sum(not row.get("gold_chunk_ids") for row in reference)
    excluded_unmapped = sum(
        bool(row.get("gold_chunk_ids")) and not row.get("gold_mapped")
        for row in reference
    )

    all_metrics = {}
    per_query_rows = []
    for route in ROUTES:
        per_k = {}
        for k in K_VALUES:
            rows = [
                evaluate_query(route_data[route][index], k)
                for index in eligible_indices
            ]
            metrics = aggregate(rows)
            metrics["targets_met"] = {
                name: metrics[name] >= threshold
                for name, threshold in THRESHOLDS.items()
            }
            per_k[str(k)] = metrics
        all_metrics[route] = per_k

    for index, base in enumerate(reference):
        item = {
            "id": base.get("id"),
            "query": base.get("query"),
            "answerable": base.get("answerable"),
            "gold_chunk_ids": base.get("gold_chunk_ids", []),
            "gold_documents": base.get("gold_documents", []),
            "metrics": {},
        }
        if index in eligible_indices:
            for route in ROUTES:
                item["metrics"][route] = {
                    str(k): evaluate_query(route_data[route][index], k)
                    for k in K_VALUES
                }
        per_query_rows.append(item)

    summary = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_path": manifest["dataset_path"],
            "source_snapshot_dir": manifest["route_snapshot_dir"],
            "source_batch_id": manifest["source_batch_id"],
            "fusion_strategy": manifest.get("fusion_strategy"),
            "source_snapshot_count": manifest["source_snapshot_count"],
            "evaluated_queries": len(eligible_indices),
            "excluded_no_golden_chunks": excluded_no_gold,
            "excluded_unmapped_golden": excluded_unmapped,
            "top_k_values": list(K_VALUES),
            "route_detail_files": DETAIL_FILES,
            "thresholds": THRESHOLDS,
            "doc_precision_denominator": (
                "actual number returned among the first k chunks; routes returning "
                "fewer than k candidates use their actual returned count"
            ),
        },
        "metrics_by_route": all_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in per_query_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "evaluation_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_report(summary):
    meta = summary["metadata"]
    metrics = summary["metrics_by_route"]
    lines = [
        "# 景区数据集三路检索统一评测",
        "",
        f"- 测试集：{meta['dataset_path']}（共 {meta['source_snapshot_count']} 条）",
        f"- 有效评测问题：{meta['evaluated_queries']} 条；无 golden chunk 排除 {meta['excluded_no_golden_chunks']} 条；gold 映射失败排除 {meta['excluded_unmapped_golden']} 条。",
        f"- 路由候选批次：{meta['source_batch_id']}",
        f"- 混合检索策略：{meta['fusion_strategy'] if 'fusion_strategy' in meta else '见 details_manifest.json'}",
        f"- 评测时间：{meta['created_at_utc']}",
        "",
        "## 评测范围与数据",
        "",
        "向量检索使用 semantic 路由候选，衡量用户查询语义；维度检索使用 dimension 路由候选，衡量关键信息匹配；混合检索将同一问题的 semantic 和 dimension 候选交给当前 fusion.py 的 apply_fusion 排序。三路候选详情分别保存在 output 下的 JSONL 文件，之后由同一指标脚本统一计算。",
        "",
        "本次使用已保存的景区数据集路由候选回放，未重新调用向量库或 query parser。Golden chunks 取自测试集 source 证据文本的现有映射；golden 文档优先取 source_file，缺失时由 chunk ID 的来源前缀推断。",
        "",
        "## 指标定义",
        "",
        "所有指标在有 golden chunk 且映射成功的问题上计算。chunk 相关性采用二元判断：chunk_id 属于该题 golden_chunk_ids 即相关。",
        "",
        "1. **Hit@k**：Top-k 中至少命中一个 golden chunk 的问题比例。",
        "2. **Recall@k**：单题命中 golden chunk 数 / 该题 golden chunk 总数；报告 macro（逐题平均）和 micro（全体 golden chunks 汇总）。",
        "3. **MRR@k**：每题首个 golden chunk 的倒数排名，再对问题取平均；未命中的题计 0。",
        "4. **nDCG@k**：按 chunk 二元相关性计算折损累计增益，并用理想排序归一化。",
        "5. **Doc Precision@k**：Top-k 实际返回 chunk 中，source_file 属于该题 golden 文档的比例；报告 micro 与逐题 macro。若某路返回少于 k 个 chunk，分母使用实际返回数。",
        "",
        "原目标阈值继续用于诊断：Recall@k macro ≥ 90%，Doc Precision@k micro ≥ 85%。Chunk 完整覆盖率作为补充项，表示 Top-k 命中该题全部 golden chunks 的问题比例。",
        "",
        "## 结果",
        "",
        "| 检索方式 | k | Hit@k | Recall@k macro / micro | MRR@k | nDCG@k | Doc Precision@k micro / macro | Chunk 完整覆盖率 | 阈值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route, label in ROUTES.items():
        for k in K_VALUES:
            row = metrics[route][str(k)]
            passed = all(row["targets_met"].values())
            status = "通过" if passed else "未达标"
            lines.append(
                f"| {label} | {k} | {percentage(row['hit_at_k'])} "
                f"| {percentage(row['recall_at_k_macro'])} / {percentage(row['recall_at_k_micro'])} "
                f"| {percentage(row['mrr_at_k'])} | {percentage(row['ndcg_at_k'])} "
                f"| {percentage(row['doc_precision_at_k_micro'])} / {percentage(row['doc_precision_at_k_macro'])} "
                f"| {percentage(row['complete_chunk_query_rate'])} | {status} |"
            )
    lines += [
        "",
        "## 文件",
        "",
        "- vector_retrieval_details.jsonl：向量检索 Top-20 结果及逐 chunk 文档归属、gold 标签。",
        "- dimension_retrieval_details.jsonl：维度检索 Top-20 结果及逐 chunk 文档归属、gold 标签。",
        "- hybrid_retrieval_details.jsonl：当前融合策略 Top-20 结果及逐 chunk 文档归属、gold 标签。",
        "- per_query_metrics.jsonl：三种检索逐题、逐 k 的统一指标。",
        "- summary.json：汇总数据、数据源批次和阈值判断。",
        "",
        "复现命令：",
        "",
        "    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py",
        "    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

