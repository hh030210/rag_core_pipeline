#!/usr/bin/env python3
"""Run matched Top-10/Top-15 context QA experiments from saved rerank candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_top20_module() -> Any:
    path = ROOT / "scripts" / "run_top20_context_qa_20260923.py"
    spec = importlib.util.spec_from_file_location("topk_qa_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有问答实验逻辑：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_topk(module: Any, rows: list[dict[str, Any]], output_dir: Path,
                  model: str, base_url: str, api_key: str, workers: int) -> dict[str, Any]:
    from rag_core.llm import LLMClient

    eval_input = output_dir / f"qa_eval_input_top{rows[0]['context_stats']['context_k']}.jsonl"
    eval_rows = []
    for row in rows:
        copy = dict(row)
        retrieval = dict(copy.get("retrieval_top5") or {})
        context = copy.get("retrieval_context") or {}
        retrieval["fusion_top5"] = context.get("fusion_topk") or []
        copy["retrieval_top5"] = retrieval
        eval_rows.append(copy)
    module.write_jsonl(eval_input, eval_rows)

    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        openai_compat=True,
        interval=0.0,
    )
    evaluation_dir = output_dir / "qa_evaluation"
    summary = module.evaluate_qa(
        input_path=eval_input,
        output_dir=evaluation_dir,
        client=client,
        concurrency=max(1, workers),
    )
    for retry_round in range(2):
        if not summary.get("failed"):
            break
        print(f"[eval] {output_dir.name}: {summary['failed']} 条失败，重试 {retry_round + 1}", flush=True)
        summary = module.evaluate_qa(
            input_path=eval_input,
            output_dir=evaluation_dir,
            client=client,
            concurrency=max(1, workers),
        )
    return summary


def mean_retrieval_metrics(base: Any, source_rows: list[dict[str, Any]],
                           candidates: list[list[dict[str, Any]]], k: int) -> dict[str, Any]:
    selected = []
    for row, ranked in zip(source_rows, candidates):
        gold = set(str(x) for x in ((row.get("gold") or {}).get("chunk_ids") or []))
        if gold:
            selected.append(base.metric(ranked[:k], gold, k))
    if not selected:
        return {"valid_questions": 0}
    return {
        "valid_questions": len(selected),
        **{
            name: round(sum(float(item[name]) for item in selected) / len(selected), 6)
            for name in ("hit", "recall", "mrr", "ndcg")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8911/v1")
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--api-key", default="sdu-cookie")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    source_dir = args.source_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_top20_module()
    base = module.load_base_module()
    candidate_pools, candidate_meta = module.candidate_sets(source_dir, base)
    source_rows = module.read_jsonl(
        source_dir / "rerank_ablation_v3_20260922" / "deepseek_listwise.jsonl"
    )
    baseline_rows = module.read_jsonl(source_dir / "qa_results.jsonl")
    if len(source_rows) != 435 or len(baseline_rows) != len(source_rows):
        raise RuntimeError(f"样本数不符：source={len(source_rows)}, baseline={len(baseline_rows)}")
    for index, (source, baseline) in enumerate(zip(source_rows, baseline_rows), 1):
        if source.get("question") != baseline.get("question"):
            raise RuntimeError(f"第 {index} 题与 Top-5 基线错位")

    manifest_path = output_dir / "experiment_manifest.json"
    manifest = {
        "experiment": "matched_top10_top15_context_qa",
        "started_at_unix": time.time(),
        "source_run_dir": str(source_dir),
        "model": args.model,
        "api_key": "redacted; server-local compatibility proxy",
        "context_k_values": [10, 15],
        "rankers": ["listwise", "answerability", "gold_free_rrf_k60"],
        "prompt": "reuse each question's prompt from the existing Top-5 QA result",
        "candidate_metadata": candidate_meta,
        "workers": args.workers,
        "eval_workers": args.eval_workers,
        "max_retries": args.retries,
        "status": "running",
    }
    write_json(manifest_path, manifest)

    generation_summaries: dict[str, dict[str, Any]] = {}
    evaluation_summaries: dict[str, dict[str, Any]] = {}
    retrieval_summaries: dict[str, dict[str, Any]] = {}

    for k in (10, 15):
        for strategy in ("listwise", "answerability", "gold_free_rrf_k60"):
            label = f"{strategy}_top{k}"
            strategy_dir = output_dir / label
            chosen = [pool[:k] for pool in candidate_pools[strategy]]
            qa_rows, qa_summary = module.generate_strategy(
                strategy=label,
                candidates_by_row=chosen,
                source_rows=source_rows,
                baseline_rows=baseline_rows,
                output_dir=strategy_dir,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                workers=args.workers,
                retries=args.retries,
            )
            for row in qa_rows:
                row["qa_strategy"] = label
                retrieval_context = row.setdefault("retrieval_context", {})
                retrieval_context["context_k"] = k
                if "fusion_top20" in retrieval_context:
                    retrieval_context["fusion_topk"] = retrieval_context.pop("fusion_top20")
                row["context_stats"]["context_k"] = k
                row["context_stats"]["mode"] = f"full_top{k}"
            module.write_jsonl(strategy_dir / "qa_results.jsonl", qa_rows)
            qa_summary["strategy"] = label
            qa_summary["context_k"] = k
            write_json(strategy_dir / "qa_summary.json", qa_summary)
            generation_summaries[label] = qa_summary
            print(f"[generation] {label}: {qa_summary['completed']}/{qa_summary['total']}, "
                  f"tokens={qa_summary['total_tokens']:,}", flush=True)

            evaluation_summaries[label] = evaluate_topk(
                module, qa_rows, strategy_dir, args.model, args.base_url,
                args.api_key, args.eval_workers,
            )
            retrieval_summaries[label] = mean_retrieval_metrics(
                base, source_rows, candidate_pools[strategy], k
            )
            print(f"[evaluation] {label}: evaluated="
                  f"{evaluation_summaries[label].get('evaluated', 0)}, "
                  f"failed={evaluation_summaries[label].get('failed', 0)}", flush=True)
            manifest.update({
                "status": f"completed_{label}",
                "updated_at_unix": time.time(),
                "generation_summaries": generation_summaries,
                "evaluation_summaries": evaluation_summaries,
                "retrieval_summaries": retrieval_summaries,
            })
            write_json(manifest_path, manifest)

    # Align all conditions on the same dataset rows, including the historical Top-5 baseline
    # and the already completed Top-20 results.
    evaluation_paths: dict[str, Path] = {
        "Top-5 加权融合基线": source_dir / "qa_evaluation" / "qa_answer_evaluation.jsonl",
    }
    top20_dir = source_dir / "top20_context_qa_20260923_run1"
    for key, label in (
        ("listwise", "Listwise Top-20"),
        ("answerability", "Answerability Top-20"),
        ("gold_free_rrf_k60", "Gold-free RRF Top-20"),
    ):
        evaluation_paths[label] = top20_dir / key / "qa_evaluation" / "qa_answer_evaluation.jsonl"
    for k in (10, 15):
        for strategy, label in (
            ("listwise", "Listwise"),
            ("answerability", "Answerability"),
            ("gold_free_rrf_k60", "Gold-free RRF"),
        ):
            evaluation_paths[f"{label} Top-{k}"] = (
                output_dir / f"{strategy}_top{k}" / "qa_evaluation" / "qa_answer_evaluation.jsonl"
            )

    evaluations = {label: module.valid_eval_rows(path) for label, path in evaluation_paths.items()}
    common = set.intersection(*(set(rows) for rows in evaluations.values()))
    qa_metrics = {label: module.aggregate(rows, common) for label, rows in evaluations.items()}

    lines = [
        "# Top-10 / Top-15 问答上下文实验",
        "",
        f"- 评测集：同一份 435 题问答集；问答模型：{args.model}。",
        "- K=10、15 使用已保存的三种排序候选，只改变提供给问答模型的上下文条数；未重跑切片、维度抽取、入库或 rerank。",
        "- 每题复用原 Top-5 问答记录中的 Prompt；并与已有 Top-5、Top-20 结果按共同成功评价的问题对齐。",
        f"- 所有条件共同评价样本：{len(common)}。",
        "",
        "## 问答质量",
        "",
        "| 条件 | N | Pass Rate | 正确性/5 | 完整性/5 | 相关性/5 | 证据支撑/5 | 无依据率 | 字符 F1 | ROUGE-L F1 | 数字 F1 | 参考证据覆盖 Recall | 生成 Token | 秒/题 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["Top-5 加权融合基线", "Listwise Top-20", "Listwise Top-15", "Listwise Top-10",
             "Answerability Top-20", "Answerability Top-15", "Answerability Top-10",
             "Gold-free RRF Top-20", "Gold-free RRF Top-15", "Gold-free RRF Top-10"]
    summary_by_label = {
        "Listwise Top-20": top20_dir / "listwise" / "qa_summary.json",
        "Answerability Top-20": top20_dir / "answerability" / "qa_summary.json",
        "Gold-free RRF Top-20": top20_dir / "gold_free_rrf_k60" / "qa_summary.json",
    }
    for label in order:
        metric = qa_metrics.get(label, {"n": 0})
        summary_path = summary_by_label.get(label)
        if summary_path is None:
            for k in (10, 15):
                for strategy, name in (("listwise", "Listwise"),
                                       ("answerability", "Answerability"),
                                       ("gold_free_rrf_k60", "Gold-free RRF")):
                    if label == f"{name} Top-{k}":
                        summary_path = output_dir / f"{strategy}_top{k}" / "qa_summary.json"
                        break
        summary = read_json(summary_path, {}) if summary_path else {}
        if label.startswith("Top-5"):
            summary = {}
        if not metric.get("n"):
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {metric['n']} | {metric['pass_rate']:.2%} | "
            f"{metric['correctness']:.2f} | {metric['completeness']:.2f} | "
            f"{metric['relevance']:.2f} | {metric['groundedness']:.2f} | "
            f"{metric['unsupported_claim_rate']:.2%} | {metric['char_f1']:.4f} | "
            f"{metric['rouge_l_f1']:.4f} | {metric['numeric_f1']:.4f} | "
            f"{metric['reference_evidence_char_recall']:.4f} | "
            f"{summary.get('total_tokens', '历史未记录')} | "
            f"{summary.get('average_latency_seconds', '历史未记录')} |"
        )

    lines.extend([
        "",
        "## 检索质量（按每档 K 截断后评估，只统计有 golden 的问题）",
        "",
        "| 排序方案 | K | 有效题数 | Hit Rate | Golden Recall | MRR | nDCG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for k in (10, 15):
        for strategy, label in (("listwise", "DeepSeek Listwise"),
                                ("answerability", "Answerability"),
                                ("gold_free_rrf_k60", "Gold-free RRF")):
            key = f"{strategy}_top{k}"
            item = retrieval_summaries[key]
            lines.append(
                f"| {label} | {k} | {item.get('valid_questions', 0)} | "
                f"{item.get('hit', 0):.2%} | {item.get('recall', 0):.2%} | "
                f"{item.get('mrr', 0):.4f} | {item.get('ndcg', 0):.4f} |"
            )

    lines.extend([
        "",
        "> 说明：模型评审分数和 Pass Rate 来自 LLM-as-a-judge；字符 F1、ROUGE-L、数字 F1、证据覆盖属于机械/文本重合指标，不能单独等同于事实正确。",
        "",
    ])
    report_path = output_dir / "comparison_top10_top15.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "common_evaluation_n": len(common),
        "qa_metrics_on_common_questions": qa_metrics,
        "retrieval_metrics": retrieval_summaries,
        "generation_summaries": generation_summaries,
        "evaluation_summaries": evaluation_summaries,
        "report": str(report_path),
    }
    write_json(output_dir / "comparison_top10_top15.json", result)
    manifest["status"] = "completed"
    manifest["completed_at_unix"] = time.time()
    manifest["report"] = str(report_path)
    write_json(manifest_path, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
