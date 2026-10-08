#!/usr/bin/env python3
"""Evidence-grounded QA evaluation for DuRetrieval, which has qrels but no answers."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.llm import LLMClient


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def overlap(answer: str, evidence: str) -> float:
    left, right = normalize(answer), normalize(evidence)
    if not left or not right:
        return 0.0
    counts = Counter(right)
    matched = sum(min(count, counts[ch]) for ch, count in Counter(left).items())
    return round(matched / len(left), 6)


def judge_one(row, client: LLMClient):
    evidence = "\n\n".join(
        str(item.get("evidence_text") or "")[:3000]
        for item in row.get("gold_evidence", [])
        if isinstance(item, dict)
    )
    answer = str(row.get("answer") or "")
    prompt = (
        "请评价一个检索增强问答系统的答案。参考证据来自公开检索数据集的相关段落，"
        "没有人工标准答案；只能依据问题和参考证据判断。只输出合法 JSON："
        "{\"correctness\":0到2,\"completeness\":0到2,\"relevance\":0到2,"
        "\"groundedness\":0到2,\"unsupported_claims\":0到2,\"comment\":\"...\"}。"
        "correctness 表示事实是否与证据一致；completeness 表示是否覆盖问题要点；"
        "groundedness 表示答案是否都能在证据中找到依据；unsupported_claims 越高表示无依据陈述越多。\n"
        f"问题：{row.get('question', '')}\n"
        f"参考证据：\n{evidence}\n"
        f"系统答案：\n{answer}"
    )
    try:
        judged = client.complete_json(
            "你是严格的证据型 QA 评测器，只输出 JSON。",
            prompt,
            temperature=0.0,
            max_tokens=512,
        )
        if not isinstance(judged, dict):
            raise ValueError("judge result is not object")
        result = {
            key: float(judged.get(key, 0) or 0)
            for key in ("correctness", "completeness", "relevance", "groundedness", "unsupported_claims")
        }
        result["comment"] = str(judged.get("comment", ""))
    except Exception as exc:
        result = {key: None for key in ("correctness", "completeness", "relevance", "groundedness", "unsupported_claims")}
        result["comment"] = f"评测失败：{type(exc).__name__}: {exc}"
    result["answer_chars"] = len(answer)
    result["evidence_overlap"] = overlap(answer, evidence)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dataset", required=True, type=Path)
    parser.add_argument("--qa-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    gold = {str(row["id"]): row for row in json.loads(args.qa_dataset.read_text(encoding="utf-8"))}
    rows = read_jsonl(args.qa_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient(api_key=args.api_key, base_url=args.base_url, model=args.model, interval=0)
    output = {}

    def one(row):
        return str(row.get("id")), judge_one({**gold.get(str(row.get("id")), {}), **row}, client)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(one, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            key, value = future.result()
            output[key] = value
            if index % 50 == 0 or index == len(futures):
                print(f"[QA评测] {index}/{len(futures)}", flush=True)
    detail = []
    for row in rows:
        item = {"id": str(row.get("id")), "question": row.get("question", ""), "answer": row.get("answer", "")}
        item["metrics"] = output.get(item["id"], {})
        detail.append(item)
    detail_path = args.output_dir / "qa_evidence_evaluation.jsonl"
    detail_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detail), encoding="utf-8")
    metric_keys = ("correctness", "completeness", "relevance", "groundedness", "unsupported_claims", "evidence_overlap")
    summary = {"dataset": str(args.qa_dataset), "qa_results": str(args.qa_results), "total": len(detail), "metrics": {}}
    for key in metric_keys:
        values = [row["metrics"].get(key) for row in detail if isinstance(row["metrics"].get(key), (int, float))]
        summary["metrics"][key] = round(sum(values) / len(values), 6) if values else None
    summary["judge_failures"] = sum("评测失败" in str(row["metrics"].get("comment", "")) for row in detail)
    (args.output_dir / "qa_evidence_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "qa_evidence_evaluation_summary.md").write_text(
        "# DuRetrieval 证据型问答评测\n\n"
        "该数据集没有人工参考答案，因此指标是依据 qrels 相关段落的证据型 LLM 评价，不等同于人工答案准确率。\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in summary["metrics"].items())
        + f"\n- 评测失败数: {summary['judge_failures']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
