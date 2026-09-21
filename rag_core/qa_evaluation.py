"""对批量问答结果执行答案质量评价。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
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


def _metric_text(value: Any) -> str:
    """为机械指标统一文本格式，去掉 Markdown、空白和来源标记。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"\[(?:来源|证据|source|evidence)[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\s+", "", text)
    return "".join(char for char in text if char.isalnum())


def _counter_prf(prediction: str, reference: str) -> Dict[str, float]:
    """按字符多重集合计算 Precision/Recall/F1。"""
    predicted = Counter(prediction)
    expected = Counter(reference)
    overlap = sum((predicted & expected).values())
    precision = overlap / len(prediction) if prediction else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _lcs_length(left: str, right: str) -> int:
    """计算两个字符序列的最长公共子序列长度。"""
    if not left or not right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, 1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    lcs = _lcs_length(prediction, reference)
    precision = lcs / len(prediction) if prediction else 0.0
    recall = lcs / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _numeric_metrics(prediction: str, reference: str) -> Dict[str, Any]:
    """比较答案和参考答案中的阿拉伯数字，避免数字事实被文字相似度掩盖。"""
    predicted = _NUMBER_PATTERN.findall(prediction)
    expected = _NUMBER_PATTERN.findall(reference)
    predicted_counter = Counter(predicted)
    expected_counter = Counter(expected)
    overlap_counter = predicted_counter & expected_counter
    overlap = sum(overlap_counter.values())
    missing = list((expected_counter - predicted_counter).elements())
    extra = list((predicted_counter - expected_counter).elements())
    if not expected:
        precision = 0.0 if predicted else None
        recall = None
        f1 = None
    else:
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(expected)
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "missing_reference_numbers": missing,
        "extra_answer_numbers": extra,
    }


def compute_mechanical_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """计算不调用模型的答案质量指标。

    字符重叠、ROUGE-L 和证据覆盖是 lexical proxy，只用于稳定对比，不能替代事实判断。
    """
    answer = _metric_text(row.get("answer", ""))
    reference = _metric_text(row.get("reference_answer", ""))
    overlap = _counter_prf(answer, reference)
    rouge_l = _rouge_l(answer, reference)
    numeric = _numeric_metrics(answer, reference)
    evidence_parts = []
    for item in ((row.get("retrieval_top5") or {}).get("fusion_top5") or []):
        evidence_parts.append(_metric_text(item.get("chunk_text_full") or item.get("chunk_text") or ""))
    evidence = "".join(evidence_parts)
    answer_evidence = _counter_prf(answer, evidence)
    # 交换参数，使 recall 的分母为参考答案长度，表示“参考答案有多少字符能在证据中找到”。
    evidence_reference = _counter_prf(evidence, reference)
    return {
        "normalized_exact_match": int(bool(answer) and answer == reference),
        "answer_length_chars": len(answer),
        "reference_length_chars": len(reference),
        "length_ratio": round(len(answer) / len(reference), 4) if reference else None,
        "char_precision": overlap["precision"],
        "char_recall": overlap["recall"],
        "char_f1": overlap["f1"],
        "rouge_l_precision": rouge_l["precision"],
        "rouge_l_recall": rouge_l["recall"],
        "rouge_l_f1": rouge_l["f1"],
        "numeric_precision": numeric["precision"],
        "numeric_recall": numeric["recall"],
        "numeric_f1": numeric["f1"],
        "missing_reference_numbers": numeric["missing_reference_numbers"],
        "extra_answer_numbers": numeric["extra_answer_numbers"],
        "answer_evidence_char_precision": answer_evidence["precision"],
        "reference_evidence_char_recall": evidence_reference["recall"],
    }


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

    # 旧版本结果没有机械指标；对已完成的 LLM 评价重新补算，避免为了新增指标重复调用 API。
    for index, item in completed.items():
        if 1 <= index <= len(rows):
            item["mechanical_metrics"] = compute_mechanical_metrics(rows[index - 1])

    pending = [(index, row) for index, row in enumerate(rows, 1) if index not in completed]

    def evaluate_one(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        judgment = _judge(client, row)
        result = {
            "dataset_index": row.get("dataset_index", index),
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "judgment": judgment,
            "mechanical_metrics": compute_mechanical_metrics(row),
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

    def mean_mechanical(key: str) -> float:
        values = [
            item.get("mechanical_metrics", {}).get(key)
            for item in valid
        ]
        values = [float(value) for value in values if value is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    mechanical = [item.get("mechanical_metrics", {}) for item in valid]
    numeric_evaluable = sum(
        item.get("numeric_recall") is not None for item in mechanical
    )
    summary["mechanical_metrics"] = {
        "normalized_exact_match_rate": mean_mechanical("normalized_exact_match"),
        "mean_length_ratio": mean_mechanical("length_ratio"),
        "mean_char_precision": mean_mechanical("char_precision"),
        "mean_char_recall": mean_mechanical("char_recall"),
        "mean_char_f1": mean_mechanical("char_f1"),
        "mean_rouge_l_f1": mean_mechanical("rouge_l_f1"),
        "numeric_evaluable_count": numeric_evaluable,
        "mean_numeric_precision": mean_mechanical("numeric_precision"),
        "mean_numeric_recall": mean_mechanical("numeric_recall"),
        "mean_numeric_f1": mean_mechanical("numeric_f1"),
        "mean_missing_reference_number_count": round(
            sum(len(item.get("missing_reference_numbers", [])) for item in mechanical) / len(mechanical), 4
        ) if mechanical else 0.0,
        "mean_extra_answer_number_count": round(
            sum(len(item.get("extra_answer_numbers", [])) for item in mechanical) / len(mechanical), 4
        ) if mechanical else 0.0,
        "mean_answer_evidence_char_precision": mean_mechanical("answer_evidence_char_precision"),
        "mean_reference_evidence_char_recall": mean_mechanical("reference_evidence_char_recall"),
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
        f"- 无依据陈述率：{summary['unsupported_claim_rate']:.2%}\n\n"
        "## 机械计算指标\n\n"
        "> 以下指标不调用模型，基于归一化后的字符、数字和融合 Top5 证据做确定性计算；字符重叠和证据覆盖属于 lexical proxy，不能单独等同于事实正确。\n\n"
        f"- 归一化 Exact Match：{summary['mechanical_metrics']['normalized_exact_match_rate']:.2%}\n"
        f"- 平均答案/参考答案长度比：{summary['mechanical_metrics']['mean_length_ratio']:.4f}\n"
        f"- 字符级 Precision：{summary['mechanical_metrics']['mean_char_precision']:.4f}\n"
        f"- 字符级 Recall：{summary['mechanical_metrics']['mean_char_recall']:.4f}\n"
        f"- 字符级 F1：{summary['mechanical_metrics']['mean_char_f1']:.4f}\n"
        f"- ROUGE-L F1：{summary['mechanical_metrics']['mean_rouge_l_f1']:.4f}\n"
        f"- 数字事实可评价样本：{summary['mechanical_metrics']['numeric_evaluable_count']}\n"
        f"- 数字 Precision：{summary['mechanical_metrics']['mean_numeric_precision']:.4f}\n"
        f"- 数字 Recall：{summary['mechanical_metrics']['mean_numeric_recall']:.4f}\n"
        f"- 数字 F1：{summary['mechanical_metrics']['mean_numeric_f1']:.4f}\n"
        f"- 平均遗漏参考数字数：{summary['mechanical_metrics']['mean_missing_reference_number_count']:.4f}\n"
        f"- 平均多答数字数：{summary['mechanical_metrics']['mean_extra_answer_number_count']:.4f}\n"
        f"- 答案字符在融合证据中的覆盖 Precision：{summary['mechanical_metrics']['mean_answer_evidence_char_precision']:.4f}\n"
        f"- 参考答案字符在融合证据中的覆盖 Recall：{summary['mechanical_metrics']['mean_reference_evidence_char_recall']:.4f}\n",
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
