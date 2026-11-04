#!/usr/bin/env python3
"""Replay the configured fusion strategy on saved scenic-dataset route candidates."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))

from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY, apply_fusion  # noqa: E402

DEFAULT_DATASET = Path(
    "/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/qa_dataset_435.json"
)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_DIR = (
    REPOSITORY / "rag_core/retrieval_fusion/experiments/00_method_comparison"
    / "output/dimension_score"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_doc(value: str | None) -> str:
    value = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if value.lower().endswith((".md", ".txt", ".json")):
        value = value.rsplit(".", 1)[0]
    return value


def infer_doc_from_chunk_id(chunk_id: str) -> str:
    parent = str(chunk_id).split("::", 1)[0]
    return re.sub(r"_\d{3}$", "", parent)


def metric_row(gold_chunk_ids, gold_docs, retrieved, k, full_ranked=None):
    retrieved_ids = [str(item.get("chunk_id", "")) for item in retrieved]
    retrieved_set = set(retrieved_ids)
    retrieved_docs = {
        canonical_doc(item.get("source_file"))
        or infer_doc_from_chunk_id(str(item.get("chunk_id", "")))
        for item in retrieved
    }
    retrieved_docs.discard("")
    gold_chunk_ids = set(map(str, gold_chunk_ids))
    gold_docs = set(filter(None, map(canonical_doc, gold_docs)))
    chunk_hits = len(gold_chunk_ids & retrieved_set)
    doc_hits = len(gold_docs & retrieved_docs)
    relevant_count = sum(
        1 for item in retrieved
        if (
            canonical_doc(item.get("source_file"))
            or infer_doc_from_chunk_id(str(item.get("chunk_id", "")))
        ) in gold_docs
    )
    gold_ranks = [
        rank for rank, cid in enumerate(retrieved_ids, start=1)
        if cid in gold_chunk_ids
    ]
    full_ranked = full_ranked if full_ranked is not None else retrieved
    full_gold_ranks = [
        rank for rank, item in enumerate(full_ranked, start=1)
        if str(item.get("chunk_id", "")) in gold_chunk_ids
    ]
    first_gold_rank = min(full_gold_ranks) if full_gold_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in gold_ranks)
    ideal_count = min(len(gold_chunk_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    chunk_recall = chunk_hits / len(gold_chunk_ids) if gold_chunk_ids else 0.0
    return {
        "hit_at_k": bool(gold_ranks),
        "chunk_hits": chunk_hits,
        "gold_chunk_count": len(gold_chunk_ids),
        "gold_recall_at_k": chunk_recall,
        "chunk_recall": chunk_recall,
        "precision_at_k": chunk_hits / k if k else 0.0,
        "mrr_at_k": 1.0 / gold_ranks[0] if gold_ranks else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
        "first_gold_rank": first_gold_rank,
        "miss": first_gold_rank is None,
        "rr": 1.0 / first_gold_rank if first_gold_rank else 0.0,
        "result_count": len(full_ranked),
        "doc_hits": doc_hits,
        "gold_doc_count": len(gold_docs),
        "doc_recall": doc_hits / len(gold_docs) if gold_docs else 0.0,
        "doc_complete": doc_hits == len(gold_docs) if gold_docs else False,
        "chunk_complete": chunk_hits == len(gold_chunk_ids) if gold_chunk_ids else False,
        "same_doc_relevant_chunks": relevant_count,
        "retrieved_count": len(retrieved),
        "accuracy": relevant_count / len(retrieved) if retrieved else 0.0,
    }


def aggregate(rows, k):
    gold_chunks = sum(row["gold_chunk_count"] for row in rows)
    gold_docs = sum(row["gold_doc_count"] for row in rows)
    retrieved = sum(row["retrieved_count"] for row in rows)
    relevant = sum(row["same_doc_relevant_chunks"] for row in rows)
    chunk_hits = sum(row["chunk_hits"] for row in rows)
    doc_hits = sum(row["doc_hits"] for row in rows)
    found_ranks = [row["first_gold_rank"] for row in rows if row["first_gold_rank"]]
    query_count = len(rows)
    return {
        "answerable_queries": query_count,
        "hit_at_k": sum(row["hit_at_k"] for row in rows) / query_count if query_count else 0.0,
        "gold_recall_at_k_micro": chunk_hits / gold_chunks if gold_chunks else 0.0,
        "gold_recall_at_k_macro": statistics.fmean(r["gold_recall_at_k"] for r in rows) if rows else 0.0,
        "chunk_recall_micro": chunk_hits / gold_chunks if gold_chunks else 0.0,
        "chunk_recall_macro": statistics.fmean(r["chunk_recall"] for r in rows) if rows else 0.0,
        "precision_at_k": chunk_hits / (k * query_count) if k and query_count else 0.0,
        "mrr_at_k": statistics.fmean(r["mrr_at_k"] for r in rows) if rows else 0.0,
        "ndcg_at_k": statistics.fmean(r["ndcg_at_k"] for r in rows) if rows else 0.0,
        "mrr": statistics.fmean(r["rr"] for r in rows) if rows else 0.0,
        "miss_rate": sum(r["miss"] for r in rows) / query_count if query_count else 0.0,
        "first_gold_rank_mean_when_found": statistics.fmean(found_ranks) if found_ranks else None,
        "chunk_complete_query_rate": (
            sum(r["chunk_complete"] for r in rows) / query_count if query_count else 0.0
        ),
        "doc_recall_micro": doc_hits / gold_docs if gold_docs else 0.0,
        "doc_recall_macro": statistics.fmean(r["doc_recall"] for r in rows) if rows else 0.0,
        "doc_complete_query_rate": (
            sum(r["doc_complete"] for r in rows) / query_count if query_count else 0.0
        ),
        "accuracy_micro": relevant / retrieved if retrieved else 0.0,
        "accuracy_macro": statistics.fmean(r["accuracy"] for r in rows) if rows else 0.0,
        "gold_chunks": gold_chunks,
        "gold_documents": gold_docs,
        "retrieved_chunks": retrieved,
        "same_doc_relevant_chunks": relevant,
        "mean_result_count": statistics.fmean(r["result_count"] for r in rows) if rows else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated cutoffs")
    parser.add_argument("--strategy", default=FUSION_STRATEGY)
    return parser.parse_args()


def main():
    args = parse_args()
    ks = sorted(set(int(value.strip()) for value in args.ks.split(",") if value.strip()))
    if not ks or ks[0] <= 0:
        raise SystemExit("--ks must contain positive integers")
    max_k = max(ks)
    dataset = load_json(args.dataset)
    manifest = load_json(args.route_dir / "index.json")
    items = manifest.get("items", [])
    if len(dataset) != len(items):
        raise RuntimeError(
            f"Dataset/route count mismatch: {len(dataset)} questions vs {len(items)} snapshots"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query = []
    totals = {k: [] for k in ks}
    skipped_unanswerable = 0
    skipped_unmapped = 0
    mapping_fallbacks = 0

    for position, (question, ref) in enumerate(zip(dataset, items), start=1):
        if " ".join(str(question.get("query", "")).split()) != " ".join(str(ref.get("query", "")).split()):
            raise RuntimeError(f"Query order mismatch at position {position}")
        detail = load_json(args.route_dir / ref["name"])
        if " ".join(str(detail.get("query", "")).split()) != " ".join(str(question.get("query", "")).split()):
            raise RuntimeError(f"Snapshot query mismatch in {ref['name']}")

        gold = detail.get("gold") or {}
        resolution = gold.get("gold_resolution") or {}
        gold_ids = sorted(set(map(str, gold.get("chunk_ids") or [])))
        if not gold_ids:
            skipped_unanswerable += 1
            continue
        if resolution and not resolution.get("gold_mapped", True):
            skipped_unmapped += 1
            continue

        routes = detail.get("routes") or {}
        semantic = copy.deepcopy(routes.get("semantic") or [])
        dimension = copy.deepcopy(routes.get("dimension") or [])
        if not semantic and not dimension:
            raise RuntimeError(f"Missing route candidates in {ref['name']}")

        # Route snapshots omit text to keep them compact; restore the saved chunk
        # payloads before running strategies that use lexical text features.
        chunks = detail.get("chunks") or {}
        for candidate in semantic + dimension:
            chunk = chunks.get(str(candidate.get("chunk_id", "")), {})
            if chunk:
                candidate["chunk_text_full"] = chunk.get("text", "")
                candidate.setdefault("source_file", chunk.get("source_file", ""))
                candidate.setdefault("doc_title", chunk.get("title", ""))

        source_by_chunk = {}
        for candidate in semantic + dimension + (detail.get("fusion_candidates") or []):
            cid = str(candidate.get("chunk_id", ""))
            if cid:
                source_by_chunk[cid] = canonical_doc(candidate.get("source_file"))
        for cid, chunk in (detail.get("chunks") or {}).items():
            source_by_chunk.setdefault(str(cid), canonical_doc(chunk.get("source_file")))

        gold_docs = []
        for cid in gold_ids:
            doc = source_by_chunk.get(cid) or infer_doc_from_chunk_id(cid)
            if not source_by_chunk.get(cid):
                mapping_fallbacks += 1
            gold_docs.append(doc)

        fused = apply_fusion(
            semantic,
            dimension,
            top_k=max_k,
            query=detail.get("retrieval_query") or question["query"],
            original_query=question["query"],
            query_analysis=detail.get("query_analysis"),
            strategy_file=args.strategy,
        )
        full_candidates = fused.get("fusion_candidates") or fused.get("fusion_results") or []
        candidates = fused.get("fusion_results") or full_candidates
        results = []
        for item in candidates[:max_k]:
            row = dict(item)
            cid = str(row.get("chunk_id", ""))
            row["source_file"] = canonical_doc(row.get("source_file")) or (
                source_by_chunk.get(cid) or infer_doc_from_chunk_id(cid)
            )
            results.append(row)

        query_result = {
            "id": question.get("id"),
            "query": question.get("query"),
            "gold_chunk_ids": gold_ids,
            "gold_documents": sorted(set(filter(None, map(canonical_doc, gold_docs)))),
            "strategy": args.strategy,
            "retrieved": [
                {
                    "rank": rank,
                    "chunk_id": str(item.get("chunk_id", "")),
                    "source_file": item.get("source_file", ""),
                    "is_golden_chunk": str(item.get("chunk_id", "")) in gold_ids,
                }
                for rank, item in enumerate(results, start=1)
            ],
            "metrics": {},
        }
        for k in ks:
            row = metric_row(gold_ids, gold_docs, results[:k], k, full_candidates)
            totals[k].append(row)
            query_result["metrics"][str(k)] = row
        per_query.append(query_result)

    if not per_query:
        raise RuntimeError("No answerable, evidence-mapped questions were evaluated")

    result_by_k = {str(k): aggregate(totals[k], k) for k in ks}
    thresholds = {"chunk_recall_micro": 0.90, "doc_recall_micro": 0.90, "accuracy_micro": 0.85}
    for metrics in result_by_k.values():
        metrics["targets_met"] = {
            name: metrics[name] >= target for name, target in thresholds.items()
        }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "fusion_entrypoint": "rag_core.retrieval_fusion.fusion.apply_fusion",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_questions": len(dataset),
        "route_snapshot_dir": str(args.route_dir.resolve()),
        "route_batch_id": manifest.get("batch_id"),
        "route_snapshot_count": len(items),
        "evaluated_answerable_questions": len(per_query),
        "excluded_no_golden_chunks": skipped_unanswerable,
        "excluded_unmapped_evidence": skipped_unmapped,
        "gold_document_mapping_fallbacks": mapping_fallbacks,
        "ks": ks,
        "thresholds": thresholds,
        "evaluation_mode": (
            "Offline fusion replay: saved semantic and dimension route candidates "
            "were passed through the current fusion.py strategy; vector retrieval was not rerun."
        ),
    }
    summary = {"metadata": metadata, "metrics_by_k": result_by_k}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "evaluation_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def pct(value):
    return f"{value * 100:.2f}%"


def render_report(summary):
    meta = summary["metadata"]
    results = summary["metrics_by_k"]
    lines = [
        "# 景区数据集多路检索融合评测",
        "",
        f"- 评测时间：{meta['created_at_utc']}",
        f"- 融合入口：{meta['fusion_entrypoint']}",
        f"- 当前策略：{meta['strategy']}",
        f"- 测试集：{meta['dataset_path']}（{meta['dataset_questions']} 条）",
        f"- 路由候选来源：{meta['route_snapshot_dir']}，批次 {meta['route_batch_id']}",
        f"- 有效评测问题：{meta['evaluated_answerable_questions']} 条；无 golden chunk 排除 {meta['excluded_no_golden_chunks']} 条；证据映射失败排除 {meta['excluded_unmapped_evidence']} 条。",
        f"- Golden 文档映射回退次数：{meta['gold_document_mapping_fallbacks']}（从 chunk ID 的来源文件前缀推断）。",
        "",
        "## 评测范围",
        "",
        "程序读取现有景区 QA 测试集对应的语义路由和维度路由候选，并调用当前 fusion.py 的 apply_fusion 重新执行融合排序。该实验评估融合排序本身；没有重新调用向量库、embedding 或 query parser。路由候选批次 ID 保存在 summary.json，便于复现和审计。",
        "",
        "Golden chunk 使用路由快照中由 QA source 证据文本映射出的 gold.chunk_ids。Golden 文档优先取候选记录的 source_file；当 golden chunk 没有候选元数据时，从其 chunk ID 的来源文件前缀推断。",
        "",
        "## 指标口径",
        "",
        "所有指标只在有 golden chunk 且证据映射成功的问题上计算。k 表示融合排序 Top-k；若实际返回少于 k 条，则准确率分母为实际返回条数。",
        "",
        "1. **Hit@k**：每题 Top-k 至少命中一个 golden chunk 的问题比例。",
        "2. **Gold Recall@k**：每题命中的 golden chunk 数 / golden chunk 总数；报告 micro 汇总值与 macro 逐题平均值。",
        "3. **Precision@k**：Top-k 中精确命中的 golden chunk 数 / k，保留原有 chunk 级精确率口径。",
        "4. **MRR@k**：每题首个 golden chunk 排名倒数的平均值；同时保留全融合候选集的 RR、first_gold_rank、miss 和 result_count。",
        "5. **NDCG@k**：按 golden chunk 二元相关性计算折损累计增益并按理想排序归一化。",
        "6. **Chunk 完整覆盖率**：Top-k 命中该题全部 golden chunks 的问题数 / 有效问题数，用于检查关键证据是否齐全。",
        "7. **Doc 召回率**：Top-k 命中的 golden 来源文档数 / golden 来源文档总数；报告 micro、macro 和逐题完整覆盖率。",
        "8. **同源文档准确率**：Top-k 中属于该题 golden 来源文档的 chunk 数 / Top-k 实际返回 chunk 数；按用户要求，同一 golden 文档中的 chunk 均计为有效。",
        "",
        "验收阈值按汇总 micro 指标判断：Chunk 召回率 ≥ 90%，Doc 召回率 ≥ 90%，准确率 ≥ 85%。逐题完整覆盖率作为额外诊断项展示。",
        "",
        "## 结果",
        "",
        "| K | Hit@k | Gold Recall@k micro / macro | Precision@k | MRR@k | NDCG@k | Chunk 完整覆盖率 | Doc Recall micro | Doc 完整覆盖率 | 同源文档准确率 micro | 阈值结果 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for k, m in results.items():
        status = "通过" if all(m["targets_met"].values()) else "未全部达标"
        lines.append(
            f"| {k} | {pct(m['hit_at_k'])} "
            f"| {pct(m['gold_recall_at_k_micro'])} / {pct(m['gold_recall_at_k_macro'])} "
            f"| {pct(m['precision_at_k'])} | {pct(m['mrr_at_k'])} | {pct(m['ndcg_at_k'])} "
            f"| {pct(m['chunk_complete_query_rate'])} | {pct(m['doc_recall_micro'])} "
            f"| {pct(m['doc_complete_query_rate'])} | {pct(m['accuracy_micro'])} | {status} |"
        )
    final_k = max(results, key=lambda value: int(value))
    final_metrics = results[final_k]
    passed = [name for name, ok in final_metrics["targets_met"].items() if ok]
    missed = [name for name, ok in final_metrics["targets_met"].items() if not ok]
    lines += [
        "",
        "## 达标结论",
        "",
        f"Top-{final_k} 汇总：Chunk 召回率 {pct(final_metrics['chunk_recall_micro'])}，Doc 召回率 {pct(final_metrics['doc_recall_micro'])}，准确率 {pct(final_metrics['accuracy_micro'])}。",
        f"达到阈值：{', '.join(passed) if passed else '无'}；未达到阈值：{', '.join(missed) if missed else '无'}。",
        f"Top-{final_k} Hit 率为 {pct(final_metrics['hit_at_k'])}；这表示至少命中一块 golden chunk，不表示每题所有证据均齐全。",
        f"Top-{final_k} 每题完整召回全部 golden chunks 的比例为 {pct(final_metrics['chunk_complete_query_rate'])}。",
        f"全候选集 MRR 为 {pct(final_metrics['mrr'])}，miss 率为 {pct(final_metrics['miss_rate'])}，首个 golden chunk 的平均排名（命中问题）为 {final_metrics['first_gold_rank_mean_when_found']:.2f}，平均候选数为 {final_metrics['mean_result_count']:.2f}。",
        "",
        "## 结果解释",
        "",
        "Chunk 级指标判断标准证据分块是否进入 Top-k；Doc 级指标判断这些证据所在的来源文档是否被覆盖。准确率依照本实验约定，统计 Top-k 中属于 golden 来源文档的所有 chunk，因此不要求 chunk 本身必须等于 golden chunk。",
        "",
        "逐题明细见 per_query.jsonl，汇总指标和阈值布尔值见 summary.json。重跑命令：",
        "",
        "    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/evaluate_main.py --ks 1,5,10",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

