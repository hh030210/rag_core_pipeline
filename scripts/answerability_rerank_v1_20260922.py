#!/usr/bin/env python3
"""Answerability-aware reranking on a frozen candidate set.

This is deliberately a reranker, not a retriever.  It receives the same
semantic Top-100 / dimension Top-20 candidates as the existing experiment,
asks DeepSeek to score every candidate on answerability-related dimensions,
and only then orders the candidates.  An incomplete model response causes an
all-or-nothing fallback to the original candidate order for that question.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("rerank_ablation_v3_20260922.py")
spec = importlib.util.spec_from_file_location("base_rerank", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def build_prompt(row: dict[str, Any], pool: list[dict[str, Any]]) -> str:
    facets = base.extract_facets(row)
    payload = [
        {"index": i, "chunk_id": item.get("chunk_id"), "text": str(item.get("chunk_text", ""))[:3000]}
        for i, item in enumerate(pool)
    ]
    return (
        "你是检索候选重排器。候选已经由语义检索和维度检索召回，你不能重新检索，也不能使用golden标签。"
        "请逐个判断每个候选片段能否直接、完整地回答问题。必须评价全部候选，不能遗漏任何index，不能输出解释文字。\n"
        f"问题：{row.get('question', '')}\n"
        f"问题要点：{json.dumps(facets, ensure_ascii=False)}\n"
        "对每个候选输出：answerability（能否用于回答，0到1）、directness（是否直接回答，0到1）、"
        "fact_coverage（覆盖问题要点程度，0到1）、completeness（上下文是否完整，0到1）、noise（无关或干扰程度，0到1）。"
        "只返回严格JSON：{\"items\":[{\"index\":0,\"answerability\":0.0,\"directness\":0.0,"
        "\"fact_coverage\":0.0,\"completeness\":0.0,\"noise\":0.0}]}。"
        f"候选数量={len(pool)}，index必须覆盖0到{len(pool)-1}且每个只出现一次。\n"
        f"候选片段：{json.dumps(payload, ensure_ascii=False)}"
    )


def cache_key(row: dict[str, Any], pool: list[dict[str, Any]]) -> str:
    payload = [{"chunk_id": x.get("chunk_id"), "text": str(x.get("chunk_text", ""))[:3000]} for x in pool]
    raw = "answerability_v1\n" + str(row.get("question")) + json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def parse_scores(content: str, expected: int) -> dict[int, dict[str, float]] | None:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return None
    scores: dict[int, dict[str, float]] = {}
    fields = ("answerability", "directness", "fact_coverage", "completeness", "noise")
    for item in items:
        if not isinstance(item, dict):
            return None
        try:
            index = int(item["index"])
            values = {field: max(0.0, min(1.0, float(item[field]))) for field in fields}
        except (KeyError, TypeError, ValueError):
            return None
        if index in scores or index < 0 or index >= expected:
            return None
        scores[index] = values
    if set(scores) != set(range(expected)):
        return None
    return scores


def quality_score(values: dict[str, float]) -> float:
    # Answerability is primary; the other dimensions make the judgment
    # explicitly about usable context rather than lexical similarity alone.
    value = (
        0.45 * values["answerability"]
        + 0.20 * values["directness"]
        + 0.20 * values["fact_coverage"]
        + 0.15 * values["completeness"]
        - 0.15 * values["noise"]
    )
    return max(0.0, min(1.0, value))


def score_row(row: dict[str, Any], client: Any, model_name: str, cache: dict[str, Any], lock: threading.Lock, retries: int) -> tuple[dict[str, Any], bool, int, str | None, str]:
    pool = base.candidate_pool(row, limit=20)
    key = cache_key(row, pool)
    cached = cache.get(key)
    scores = None
    attempts = 0
    error = None
    if isinstance(cached, dict):
        scores = parse_scores(json.dumps(cached, ensure_ascii=False), len(pool))
    while scores is None and attempts <= retries:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": build_prompt(row, pool)}],
                temperature=0,
                max_tokens=3200,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            scores = parse_scores(content, len(pool))
            if scores is not None:
                with lock:
                    cache[key] = {"items": [{"index": i, **scores[i]} for i in range(len(pool))]}
            else:
                error = "incomplete_or_invalid_json"
        except Exception as exc:
            error = repr(exc)
        if scores is None and attempts <= retries:
            time.sleep(0.5)
    row_copy = json.loads(json.dumps(row, ensure_ascii=False))
    if scores is None:
        # All-or-nothing fallback.  Never mix an incomplete LLM ranking with
        # base scores from unjudged candidates.
        candidates = [dict(x, rerank_score=float(x.get("base_score", 0.0)), rerank_status="fallback") for x in pool]
        valid = False
    else:
        candidates = []
        for i, item in enumerate(pool):
            copy = dict(item)
            copy.update(scores[i])
            copy["quality_score"] = quality_score(scores[i])
            # Keep the retrieval score as a prior, but let answerability drive
            # the order.  Both are normalized to [0, 1].
            copy["rerank_score"] = 0.70 * copy["quality_score"] + 0.30 * float(copy.get("base_score", 0.0))
            copy["rerank_status"] = "scored"
            candidates.append(copy)
        valid = True
    candidates.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    row_copy["rerank_strategy"] = "deepseek_answerability_v1"
    row_copy["rerank_candidates"] = candidates
    row_copy["rerank_results"] = candidates[:5]
    return row_copy, valid, attempts, error, key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8911/v1")
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--api-key", default="sdu-cookie")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = base.load_rows(args.input)
    cache_path = args.output_dir / "answerability_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    lock = threading.Lock()
    from openai import OpenAI

    def one(index: int, original: dict[str, Any]):
        client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=90.0, max_retries=0)
        result = score_row(original, client, args.model, cache, lock, args.retries)
        if args.interval > 0:
            time.sleep(args.interval)
        return index, result

    output_by_index: dict[int, dict[str, Any]] = {}
    invalid = 0
    retries_used = 0
    errors: list[str] = []
    workers = max(1, min(8, args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, i, row) for i, row in enumerate(rows)]
        for done, future in enumerate(as_completed(futures), 1):
            index, (row, valid, attempts, error, _) = future.result()
            output_by_index[index] = row
            retries_used += max(0, attempts - 1)
            if not valid:
                invalid += 1
                if error:
                    errors.append(error)
            if done % 25 == 0:
                print(f"[answerability] {done}/{len(rows)} invalid={invalid} retries={retries_used} cache={len(cache)}", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    output = [output_by_index[i] for i in range(len(rows))]
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    summary = base.summarize_final(output, "rerank_results")
    summary.update({"strategy": "deepseek_answerability_v1", "status": "completed", "model": args.model, "total": len(rows), "invalid_fallback_questions": invalid, "retries_used": retries_used, "cache_size": len(cache), "workers": workers, "score_formula": "0.70*quality_score + 0.30*base_score"})
    (args.output_dir / "answerability_rerank.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in output) + "\n", encoding="utf-8")
    (args.output_dir / "answerability_rerank.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "answerability_cache.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
