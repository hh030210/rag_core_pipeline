#!/usr/bin/env python3
"""Calculate unified metrics from three aligned details of one live run."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))

from rag_core.retrieval_fusion.experiments.common.latest_output import lock_output  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "output" / "runs"
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_query(record, k):
    gold_ids = set(map(str, record.get("gold_chunk_ids") or []))
    gold_docs = set(map(str, record.get("gold_documents") or []))
    retrieved = (record.get("retrieved") or [])[:k]
    relevant_ranks = [
        rank for rank, item in enumerate(retrieved, start=1)
        if str(item.get("chunk_id", "")) in gold_ids
    ]
    hit_count = len(relevant_ranks)
    recall = hit_count / len(gold_ids) if gold_ids else 0.0
    first_rank = relevant_ranks[0] if relevant_ranks else None
    mrr = 1.0 / first_rank if first_rank else 0.0
    average_precision = (
        sum(index / rank for index, rank in enumerate(relevant_ranks, start=1))
        / min(len(gold_ids), k)
        if gold_ids else 0.0
    )
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg if idcg else 0.0
    doc_relevant = sum(
        1 for item in retrieved
        if item.get("is_golden_doc") is True
        or str(item.get("source_file", "")) in gold_docs
        or str(item.get("parent_doc_id", "")) in gold_docs
        or str(item.get("doc_id", "")) in gold_docs
    )
    doc_precision = doc_relevant / len(retrieved) if retrieved else 0.0
    return {
        "hit_at_k": bool(relevant_ranks),
        "chunk_hits": hit_count,
        "gold_chunk_count": len(gold_ids),
        "recall_at_k": recall,
        "first_golden_rank_at_k": first_rank,
        "mrr_at_k": mrr,
        "average_precision_at_k": average_precision,
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
        "query_count": query_count,
        "mapped_gold_query_count": sum(bool(row["gold_chunk_count"]) for row in rows),
        "unmapped_gold_query_count": sum(not bool(row["gold_chunk_count"]) for row in rows),
        "gold_document_query_count": sum(bool(row["gold_doc_count"]) for row in rows),
        "hit_at_k": sum(row["hit_at_k"] for row in rows) / query_count if query_count else 0.0,
        # Macro includes every raw QA row. Questions whose Golden cannot be
        # mapped to the active collection receive zero instead of disappearing.
        "recall_at_k_macro": statistics.fmean(row["recall_at_k"] for row in rows) if rows else 0.0,
        # Micro is chunk-weighted and therefore uses only known Golden chunks.
        "recall_at_k_micro": chunk_hits / gold_chunks if gold_chunks else 0.0,
        "mrr_at_k": statistics.fmean(row["mrr_at_k"] for row in rows) if rows else 0.0,
        "map_at_k": statistics.fmean(row["average_precision_at_k"] for row in rows) if rows else 0.0,
        "ndcg_at_k": statistics.fmean(row["ndcg_at_k"] for row in rows) if rows else 0.0,
        # Every returned chunk is in the denominator; if a question has no
        # mapped Golden document, none of its results can count as relevant.
        "doc_precision_at_k_micro": doc_relevant / retrieved if retrieved else 0.0,
        "doc_precision_at_k_macro": statistics.fmean(row["doc_precision_at_k"] for row in rows) if rows else 0.0,
        "complete_chunk_query_rate": (
            sum(row["all_golden_chunks_hit"] for row in rows) / query_count if query_count else 0.0
        ),
        "gold_chunks": gold_chunks,
        "gold_documents": gold_docs,
        "retrieved_chunks": retrieved,
        "doc_relevant_chunks": doc_relevant,
        "mean_returned_per_query": statistics.fmean(row["retrieved_count"] for row in rows) if rows else 0.0,
    }


def percentage(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "latest",
        help="Directory containing a completed export (default: output/runs/latest)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.name == "latest":
        with lock_output(output_dir):
            calculate_metrics(output_dir)
    else:
        calculate_metrics(output_dir)


def calculate_metrics(output_dir: Path) -> None:
    manifest = read_json(output_dir / "details_manifest.json")
    if manifest.get("retrieval_mode") != "live_end_to_end" or not manifest.get("run_id"):
        raise RuntimeError(
            "Refusing to evaluate this directory: expected a live_end_to_end manifest with run_id"
        )
    route_data = {
        route: read_jsonl(output_dir / manifest.get("files", {}).get(route, filename))
        for route, filename in DETAIL_FILES.items()
    }
    counts = {len(rows) for rows in route_data.values()}
    if len(counts) != 1 or counts != {int(manifest.get("dataset_count", -1))}:
        raise RuntimeError("Three retrieval detail files must match the raw dataset row count")

    reference = route_data["vector"]
    run_id = manifest["run_id"]
    for route, rows in route_data.items():
        for index, (expected, actual) in enumerate(zip(reference, rows), start=1):
            if actual.get("run_id") != run_id:
                raise RuntimeError(f"{route} row {index} belongs to a different run")
            if (
                expected.get("id") != actual.get("id")
                or expected.get("query") != actual.get("query")
                or expected.get("gold_chunk_ids") != actual.get("gold_chunk_ids")
            ):
                raise RuntimeError(f"Route detail alignment failed: {route} row {index}")

    all_metrics = {}
    per_query_rows = []
    for route in ROUTES:
        per_k = {}
        for k in K_VALUES:
            rows = [evaluate_query(record, k) for record in route_data[route]]
            metrics = aggregate(rows)
            metrics["targets_met"] = {
                name: metrics[name] >= threshold
                for name, threshold in THRESHOLDS.items()
            }
            per_k[str(k)] = metrics
        all_metrics[route] = per_k

    for index, base in enumerate(reference):
        item = {
            "run_id": run_id,
            "id": base.get("id"),
            "query": base.get("query"),
            "answerable": base.get("answerable"),
            "gold_mapped": base.get("gold_mapped"),
            "gold_chunk_ids": base.get("gold_chunk_ids", []),
            "gold_documents": base.get("gold_documents", []),
            "metrics": {
                route: {
                    str(k): evaluate_query(route_data[route][index], k)
                    for k in K_VALUES
                }
                for route in ROUTES
            },
        }
        per_query_rows.append(item)

    summary = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "retrieval_mode": manifest["retrieval_mode"],
            "run_id": run_id,
            "dataset_path": manifest["dataset_path"],
            "dataset_sha256": manifest.get("dataset_sha256"),
            "run_dir": manifest["run_dir"],
            "collection": manifest.get("collection"),
            "source_commit": manifest.get("source_commit"),
            "fusion_strategy": manifest.get("fusion_strategy"),
            "dataset_count": manifest["dataset_count"],
            "evaluated_queries": len(reference),
            "mapped_golden_queries": manifest.get("gold_mapped_count"),
            "unmapped_golden_queries": manifest.get("gold_unmapped_count"),
            "unmapped_questions_are_macro_misses": True,
            "top_k_values": list(K_VALUES),
            "route_detail_files": manifest.get("files", DETAIL_FILES),
            "thresholds": THRESHOLDS,
            "doc_precision_denominator": (
                "actual number of returned chunks among the first k; all rows are included, "
                "so questions without a mapped Golden document contribute zero relevant chunks"
            ),
        },
        "metrics_by_route": all_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = output_dir / "evaluation_report.md"
    report = render_report(summary, output_dir=output_dir)
    report_path.write_text(report, encoding="utf-8")
    if output_dir == DEFAULT_OUTPUT_ROOT.resolve() / "latest":
        published_report = DEFAULT_OUTPUT_ROOT.parent / report_path.name
        with TemporaryDirectory(prefix=".report-", dir=published_report.parent) as temporary:
            staged_report = Path(temporary) / report_path.name
            staged_report.write_text(report, encoding="utf-8")
            staged_report.replace(published_report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_report(summary, *, output_dir: Path):
    meta = summary["metadata"]
    metrics = summary["metrics_by_route"]
    lines = [
        "# 景区数据集三路实时检索评测",
        "",
        f"- 测试集：{meta['dataset_path']}（共 {meta['dataset_count']} 条）",
        f"- Golden 映射：{meta['mapped_golden_queries']} 条成功，{meta['unmapped_golden_queries']} 条未映射；未映射问题在 macro 指标中按 miss 计入。",
        f"- Run ID：`{meta['run_id']}`；检索模式：`{meta['retrieval_mode']}`",
        f"- 当前集合：`{meta['collection']}`；融合策略：`{meta['fusion_strategy']}`",
        f"- 代码提交：`{meta['source_commit']}`；评测时间：{meta['created_at_utc']}",
        "",
        "## 评测口径",
        "",
        "export_retrieval_details.py 每轮直接读取原始 QA JSON/JSONL，使用当前 run_dir、当前 Qdrant/本地集合和当前向量检索策略完成查询解析、向量检索、维度检索与 fusion.py 融合。查询解析缓存限定在本轮临时目录。三路结果由同一个 run_id 标识；本报告只接受三份同轮产物，不读取旧实验目录、route snapshot 或先前查询解析结果。",
        "",
        "向量检索衡量用户查询语义，使用本轮 semantic candidates；维度检索衡量关键信息匹配，使用 dimension candidates；混合检索使用 fusion.py 当前策略生成的 fusion candidates。Golden 证据按当前集合中的 chunk payload 重新映射。",
        "",
        "## 指标定义",
        "",
        "chunk 相关性采用二元判断：检索 chunk_id 属于该问题映射后的 golden_chunk_ids 即相关。所有指标按原始测试集问题计算；未映射到当前集合的 Golden 在 Hit、Recall macro、MRR、nDCG 和完整覆盖率中按 miss 计入。Recall micro 以可映射 golden chunk 总数为分母。",
        "",
        "1. **Hit@k**：Top-k 命中至少一个 golden chunk 的问题占全部问题的比例。",
        "2. **Recall@k macro**：逐题计算命中 golden chunk 数 / 该题 golden chunk 数，再对全部题目取平均；Golden 未映射题计 0。**Recall@k micro**：全体命中 golden chunk 数 / 全体已映射 golden chunk 总数。",
        "3. **MRR@k**：每题第一个 golden chunk 的倒数排名，再对全部题目取平均；未命中计 0。",
        "4. **nDCG@k**：以 chunk 二元相关性计算折损累计增益，并以理想排序归一化；未命中或无已映射 Golden 计 0。",
        "5. **Doc Precision@k**：Top-k 返回 chunk 中，来源文档属于该题 golden 文档的比例。报告 micro（汇总相关 chunk / 汇总返回 chunk）和 macro（逐题平均）；分母使用实际返回数量。Golden 文档未映射的问题，其返回 chunk 均不计为相关。",
        "6. **Chunk 完整覆盖率**：Top-k 命中该题全部已映射 golden chunks 的问题比例；无映射 Golden 题计 0。",
        "",
        "Recall@k macro ≥ 90%、Doc Precision@k micro ≥ 85% 为原目标阈值；报告逐个 k 的判断。",
        "",
        "## 结果",
        "",
        "| 检索方式 | k | Hit@k | Recall@k macro / micro | MRR@k | MAP@k | nDCG@k | Doc Precision@k micro / macro | Chunk 完整覆盖率 | 阈值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route, label in ROUTES.items():
        for k in K_VALUES:
            row = metrics[route][str(k)]
            passed = all(row["targets_met"].values())
            status = "通过" if passed else "未达标"
            lines.append(
                f"| {label} | {k} | {percentage(row['hit_at_k'])} "
                f"| {percentage(row['recall_at_k_macro'])} / {percentage(row['recall_at_k_micro'])} "
                f"| {percentage(row['mrr_at_k'])} | {percentage(row['map_at_k'])} "
                f"| {percentage(row['ndcg_at_k'])} "
                f"| {percentage(row['doc_precision_at_k_micro'])} / {percentage(row['doc_precision_at_k_macro'])} "
                f"| {percentage(row['complete_chunk_query_rate'])} | {status} |"
            )
    lines += [
        "",
        "## 文件",
        "",
        f"- 本轮完整输出目录：`{output_dir}`",
        "- vector_retrieval_details.jsonl：本轮向量检索 Top-20 详情。",
        "- dimension_retrieval_details.jsonl：本轮维度检索 Top-20 详情。",
        "- hybrid_retrieval_details.jsonl：本轮融合检索 Top-20 详情。",
        "- per_query_metrics.jsonl：逐题、逐路由、逐 k 指标。",
        "- summary.json：汇总指标及运行元数据。",
        "",
        "## 重跑命令",
        "",
        "    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py",
        "    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py --output-dir <export 命令打印的 output_dir>",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
