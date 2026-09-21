"""对批量问答结果执行答案质量评价。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from .llm import LLMClient


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _clamp_score(value: Any) -> float:
    try:
        return round(max(0.0, min(5.0, float(value))), 2)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是", "有"}
    return bool(value)


def _context_text(row: Dict[str, Any]) -> str:
    top5 = (row.get("retrieval_top5") or {}).get("fusion_top5") or []
    parts = []
    for index, item in enumerate(top5, 1):
        text = str(item.get("chunk_text_full") or item.get("chunk_text") or "")
        parts.append(f"[证据{index}] chunk_id={item.get('chunk_id', '')}\n{text[:3000]}")
    return "\n\n".join(parts)


def _judge(client: LLMClient, row: Dict[str, Any]) -> Dict[str, Any]:
    system = "你是严格的中文 RAG 问答质量评测器，只输出合法 JSON，不输出 Markdown。"
    user = (
        "请评价候选答案相对于参考答案和检索证据的质量。评分均为 0 到 5 的整数或一位小数：\n"
        "- correctness：事实是否正确；\n"
        "- completeness：是否覆盖参考答案的关键要点；\n"
        "- relevance：是否直接回答问题、无明显废话；\n"
        "- groundedness：答案中的事实是否能被给定证据支持；\n"
        "- unsupported_claims：是否包含证据中没有依据的事实。\n"
        "请输出字段：correctness、completeness、relevance、groundedness、"
        "unsupported_claims、verdict、reason。其中 verdict 只能是 pass、partial、fail。\n\n"
        f"问题：{row.get('question', '')}\n"
        f"参考答案：{row.get('reference_answer', '')}\n"
        f"候选答案：{row.get('answer', '')}\n"
        f"检索证据：\n{_context_text(row)}"
    )
    try:
        result = client.complete_json(
            system,
            user,
            temperature=0.0,
            max_tokens=512,
            repair_prompt="请严格根据原始评测请求，修复为包含指定字段的 JSON。",
        )
        if not isinstance(result, dict):
            raise ValueError("评测结果不是对象")
        return {
            "correctness": _clamp_score(result.get("correctness")),
            "completeness": _clamp_score(result.get("completeness")),
            "relevance": _clamp_score(result.get("relevance")),
            "groundedness": _clamp_score(result.get("groundedness")),
            "unsupported_claims": _as_bool(result.get("unsupported_claims", False)),
            "verdict": (
                str(result.get("verdict", "partial")).lower()
                if str(result.get("verdict", "partial")).lower() in {"pass", "partial", "fail"}
                else "partial"
            ),
            "reason": str(result.get("reason", ""))[:1000],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def evaluate_qa(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    client: LLMClient,
    retrieval_results: str | Path | None = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    rows = _read_jsonl(input_path)
    retrieval_rows = _read_jsonl(retrieval_results) if retrieval_results else []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qa_answer_evaluation.jsonl"

    completed: Dict[int, Dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("dataset_index") is not None and not item.get("judgment", {}).get("error"):
                    completed[int(item["dataset_index"])] = item

    pending = [(index, row) for index, row in enumerate(rows, 1) if index not in completed]

    def evaluate_one(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        judgment = _judge(client, row)
        result = {
            "dataset_index": row.get("dataset_index", index),
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "judgment": judgment,
        }
        if index <= len(retrieval_rows):
            metrics = (retrieval_rows[index - 1].get("retrieval_metrics") or {}).get("fusion") or {}
            result["retrieval_fusion_metrics"] = {
                key: metrics.get(key)
                for key in ("hit_at_5", "gold_recall_at_5", "rr_at_5", "ndcg_at_5")
            }
        return result

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = {executor.submit(evaluate_one, index, row): index for index, row in pending}
        for future in as_completed(futures):
            item = future.result()
            completed[futures[future]] = item
            ordered = [completed[index] for index in sorted(completed)]
            output_path.write_text(
                "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in ordered),
                encoding="utf-8",
            )

    ordered = [completed[index] for index in sorted(completed)]
    output_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in ordered),
        encoding="utf-8",
    )
    valid = [item for item in ordered if "error" not in item.get("judgment", {})]

    def mean(key: str) -> float:
        values = [float(item["judgment"].get(key, 0.0)) for item in valid]
        return round(sum(values) / len(values), 4) if values else 0.0

    pass_count = sum(item["judgment"].get("verdict") == "pass" for item in valid)
    partial_count = sum(item["judgment"].get("verdict") == "partial" for item in valid)
    fail_count = sum(item["judgment"].get("verdict") == "fail" for item in valid)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(rows),
        "evaluated": len(valid),
        "failed": len(rows) - len(valid),
        "verdict_counts": {"pass": pass_count, "partial": partial_count, "fail": fail_count},
        "pass_rate": round(pass_count / len(valid), 4) if valid else 0.0,
        "average_scores": {
            key: mean(key) for key in ("correctness", "completeness", "relevance", "groundedness")
        },
        "unsupported_claim_rate": round(
            sum(bool(item["judgment"].get("unsupported_claims")) for item in valid) / len(valid), 4
        ) if valid else 0.0,
        "retrieval_metrics_source": str(retrieval_results or ""),
    }
    (output_dir / "qa_answer_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "qa_answer_evaluation_summary.md").write_text(
        "# 问答答案质量评价\n\n"
        f"- 总数：{summary['total']}\n"
        f"- 已评价：{summary['evaluated']}\n"
        f"- 评价失败：{summary['failed']}\n"
        f"- verdict：pass {pass_count} / partial {partial_count} / fail {fail_count}\n"
        f"- Pass Rate：{summary['pass_rate']:.2%}\n"
        f"- 平均正确性：{summary['average_scores']['correctness']:.2f}/5\n"
        f"- 平均完整性：{summary['average_scores']['completeness']:.2f}/5\n"
        f"- 平均相关性：{summary['average_scores']['relevance']:.2f}/5\n"
        f"- 平均证据支撑：{summary['average_scores']['groundedness']:.2f}/5\n"
        f"- 无依据陈述率：{summary['unsupported_claim_rate']:.2%}\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="评价批量问答答案质量")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-results", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        openai_compat=True,
        interval=args.interval,
    )
    print(json.dumps(evaluate_qa(
        input_path=args.input,
        output_dir=args.output_dir,
        client=client,
        retrieval_results=args.retrieval_results or None,
        concurrency=args.concurrency,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
