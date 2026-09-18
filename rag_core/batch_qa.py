"""批量生成式问答：复用 QAService，并保留三路 Top-5 完整证据。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .qa import QAService


def _read_json_or_jsonl(path: str | Path) -> Any:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def load_qa_records(path: str | Path) -> List[Dict[str, Any]]:
    """统一 query/question、answer/reference_answer 字段并跳过空问题。"""
    raw = _read_json_or_jsonl(path)
    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("data", raw.get("items", [])))
    if not isinstance(raw, list):
        raise ValueError("问答数据集必须是数组、JSONL，或包含 records/data/items 数组的对象")

    records = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("question", item.get("query", item.get("prompt", ""))) or "").strip()
        if not query:
            continue
        reference = item.get("reference_answer", item.get("answer", item.get("gold_answer", "")))
        sample_id = str(item.get("id", item.get("question_id", item.get("qid", index))))
        records.append({
            **item,
            "id": sample_id,
            "question": query,
            "reference_answer": str(reference or ""),
        })
    return records


def _result_row(record: Dict[str, Any], result: Dict[str, Any], index: int) -> Dict[str, Any]:
    """只保留问答实验所需字段，三路 Top-5 中保留完整 chunk payload。"""
    return {
        "dataset_index": index,
        "id": record["id"],
        "question": record["question"],
        "reference_answer": record.get("reference_answer", ""),
        "answer": result.get("answer", ""),
        "retrieval_query": result.get("retrieval_query", ""),
        "expansion": result.get("expansion", {}),
        "retrieval_top5": result.get("retrieval_top5", {}),
        "context_stats": result.get("context_stats", {}),
        "prompt": result.get("prompt", {}),
        "prompt_mode": result.get("prompt_mode", "global"),
        "prompt_modules_used": result.get("prompt_modules_used", []),
        "cluster_routing": result.get("cluster_routing", {}),
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_batch_qa(
    *,
    run_dir: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    settings,
    top_k: int = 5,
    context_chars: int = 0,
    concurrency: int = 1,
    resume: bool = True,
) -> Dict[str, Any]:
    """批量问答并支持从已有 JSONL 结果中恢复。"""
    records = load_qa_records(dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 测试集中的 id 可能跨来源重复，断点键必须使用数据集行号。
    completed: Dict[int, Dict[str, Any]] = {}
    if resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("dataset_index") is not None and not row.get("error"):
                completed[int(row["dataset_index"])] = row

    pending = [
        (index, record) for index, record in enumerate(records, 1)
        if index not in completed
    ]
    workers = max(1, int(concurrency))
    service = QAService(run_dir=run_dir, settings=settings)

    def answer_one(index: int, record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = service.answer(
                record["question"], top_k=top_k, context_chars=context_chars
            )
            return _result_row(record, result, index)
        except Exception as exc:
            return {
                "dataset_index": index,
                "id": record["id"],
                "question": record["question"],
                "reference_answer": record.get("reference_answer", ""),
                "error": f"{type(exc).__name__}: {exc}",
            }

    print(f"[问答] 总数 {len(records)}，已完成 {len(completed)}，待处理 {len(pending)}，并发 {workers}")
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(answer_one, index, record): (index, record)
                for index, record in pending
            }
            for done_index, future in enumerate(as_completed(futures), 1):
                index, record = futures[future]
                row = future.result()
                if row.get("error"):
                    print(f"[问答] {done_index}/{len(pending)} 失败 id={record['id']}: {row['error']}")
                else:
                    completed[index] = row
                    print(f"[问答] {done_index}/{len(pending)} 完成 id={record['id']}")

                # 每完成一条就保存，允许中断后继续。
                ordered = [completed[index] for index, _ in enumerate(records, 1) if index in completed]
                _write_jsonl(output, ordered)

    ordered = [completed[index] for index, _ in enumerate(records, 1) if index in completed]
    _write_jsonl(output, ordered)
    failed = len(records) - len(ordered)
    summary = {
        "dataset": str(dataset_path),
        "output": str(output),
        "total": len(records),
        "completed": len(ordered),
        "failed": failed,
        "concurrency": workers,
        "prompt_path": str(Path(run_dir) / "optimized_prompt.json"),
    }
    (output.parent / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary
