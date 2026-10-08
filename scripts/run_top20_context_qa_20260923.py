#!/usr/bin/env python3
"""Run matched QA experiments using the full Top-20 pool from three rerankers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.llm import LLMClient  # noqa: E402
from rag_core.prompting import PromptOptimizer  # noqa: E402
from rag_core.qa_evaluation import evaluate_qa  # noqa: E402


STRATEGY_FILES = {
    "listwise": ("rerank_ablation_v3_20260922", "deepseek_listwise.jsonl"),
    "answerability": ("answerability_rerank_v1_20260922", "answerability_rerank.jsonl"),
    "gold_free_rrf_k60": ("gold_free_rrf_v1_20260922", "gold_free_rrf.jsonl"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_base_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "rerank_ablation_v3_20260922.py"
    spec = importlib.util.spec_from_file_location("top20_rerank_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载候选构造脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def listwise_cache_key(row: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = [
        {
            "index": index,
            "chunk_id": item.get("chunk_id"),
            "text": str(item.get("chunk_text", ""))[:3000],
        }
        for index, item in enumerate(candidates)
    ]
    raw = str(row.get("question")) + json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class NoUnexpectedListwiseCall:
    """Fail closed if a supposedly completed Listwise row has no cached ranking."""

    @staticmethod
    def _unexpected_call(**_kwargs: Any) -> None:
        raise RuntimeError("Listwise 缓存缺行；本实验不会隐式重跑 rerank")

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._unexpected_call))


class OpenAICompatClient:
    """Small stdlib client for the server's OpenAI-compatible DeepSeek proxy."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 180.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]], temperature: float,
                max_tokens: int) -> Any:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 返回空 choices")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=message.get("content")))],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


def candidate_sets(source_dir: Path, base: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    listwise_dir = source_dir / "rerank_ablation_v3_20260922"
    listwise_rows = read_jsonl(listwise_dir / "deepseek_listwise.jsonl")
    cache_path = listwise_dir / "deepseek_listwise_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if len(listwise_rows) != 435:
        raise RuntimeError(f"Listwise 结果应为 435 行，实际 {len(listwise_rows)} 行")

    listwise_candidates: list[dict[str, Any]] = []
    cache_hits = 0
    for index, row in enumerate(listwise_rows, 1):
        pool = base.candidate_pool(row, limit=20)
        key = listwise_cache_key(row, pool)
        if key not in cache:
            raise RuntimeError(f"Listwise 第 {index} 行没有对应 rerank 缓存，拒绝用未重排池替代")
        cache_hits += 1
        ranked = base.deepseek_rerank(row, NoUnexpectedListwiseCall(), "DeepSeek-V4-Pro", cache)
        saved_top5 = [str(x.get("chunk_id")) for x in row.get("rerank_results", [])[:5]]
        rebuilt_top5 = [str(x.get("chunk_id")) for x in ranked[:5]]
        if saved_top5 != rebuilt_top5:
            raise RuntimeError(f"Listwise 第 {index} 行无法与已保存 Top-5 对齐")
        listwise_candidates.append(ranked[:20])
    loaded["listwise"] = listwise_candidates
    metadata["listwise_cache_hits"] = cache_hits

    for strategy, (directory, filename) in STRATEGY_FILES.items():
        if strategy == "listwise":
            continue
        rows = read_jsonl(source_dir / directory / filename)
        if len(rows) != 435:
            raise RuntimeError(f"{strategy} 结果应为 435 行，实际 {len(rows)} 行")
        pools = []
        for index, row in enumerate(rows, 1):
            if row.get("question") != listwise_rows[index - 1].get("question"):
                raise RuntimeError(f"{strategy} 第 {index} 行与 Listwise 数据集错位")
            candidates = row.get("rerank_candidates") or []
            saved_top5 = [str(x.get("chunk_id")) for x in row.get("rerank_results", [])[:5]]
            pool_top5 = [str(x.get("chunk_id")) for x in candidates[:5]]
            if saved_top5 != pool_top5:
                raise RuntimeError(f"{strategy} 第 {index} 行候选顺序与已保存 Top-5 不一致")
            pools.append([dict(item) for item in candidates[:20]])
        loaded[strategy] = pools

    all_rows = listwise_rows
    for strategy, pools in loaded.items():
        lengths = [len(pool) for pool in pools]
        if not pools or any(length == 0 or length > 20 for length in lengths):
            raise RuntimeError(f"{strategy} 的 Top-20 候选池有空池或超过 20 条的情况")
        missing_text = sum(
            not full_chunk_text(item)
            for pool in pools
            for item in pool
        )
        if missing_text:
            raise RuntimeError(f"{strategy} 有 {missing_text} 个候选缺少 chunk 正文")
        metadata[strategy] = {
            "rows": len(pools),
            "min_candidates": min(lengths),
            "median_candidates": sorted(lengths)[len(lengths) // 2],
            "max_candidates": max(lengths),
            "exact_20_rows": sum(length == 20 for length in lengths),
            **({"listwise_cache_hits": cache_hits} if strategy == "listwise" else {}),
        }
    metadata["source_rows"] = len(all_rows)
    return loaded, metadata


def candidate_signature(candidates: list[dict[str, Any]]) -> str:
    payload = [
        [str(item.get("chunk_id", "")), str(item.get("chunk_text_full") or item.get("chunk_text") or "")]
        for item in candidates
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def prompt_routes(baseline_row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    global_prompt = PromptOptimizer.normalize_module(baseline_row.get("prompt"))
    routes = []
    for item in baseline_row.get("prompt_modules_used") or []:
        module = item.get("prompt") if isinstance(item, dict) else None
        if isinstance(module, dict) and module.get("system_prompt"):
            routes.append({
                "cluster_id": item.get("cluster_id"),
                "prompt": PromptOptimizer.normalize_module(module),
            })
    if not routes:
        routes = [{"cluster_id": None, "prompt": global_prompt}]
    return routes, global_prompt


def full_chunk_text(item: dict[str, Any]) -> str:
    return str(item.get("chunk_text_full") or item.get("chunk_text") or item.get("doc_text") or "")


def make_context(candidates: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], int]:
    parts = []
    stats = []
    for index, chunk in enumerate(candidates, 1):
        text = full_chunk_text(chunk)
        chunk_id = str(chunk.get("chunk_id", ""))
        title = str(chunk.get("doc_title", chunk.get("chunk_gen_title", "")) or "")
        parts.append(f"[来源 {index}] {title} ({chunk_id})：\n{text}")
        stats.append({"chunk_id": chunk_id, "source_chars": len(text), "provided_chars": len(text)})
    context = "\n\n".join(parts)
    return context, stats, len(context)


def call_answer(client: Any, model: str, system: str, user: str, retries: int) -> tuple[str, dict[str, int], float]:
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=1200,
            )
            answer = str(response.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("模型返回空答案")
            usage = getattr(response, "usage", None)
            token_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            return answer, token_usage, round(time.monotonic() - started, 3)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}") from last_error


def generate_strategy(
    *, strategy: str, candidates_by_row: list[list[dict[str, Any]]], source_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]], output_dir: Path, model: str, base_url: str,
    api_key: str, workers: int, retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_output = output_dir / "qa_results.jsonl"
    progress_path = output_dir / "qa_results.progress.jsonl"
    error_path = output_dir / "qa_generation_errors.jsonl"
    done: dict[int, dict[str, Any]] = {}
    for path in (main_output, progress_path):
        for row in read_jsonl(path):
            index = int(row.get("dataset_index", 0) or 0)
            if index and not row.get("error"):
                done[index] = row

    tasks = []
    for index, (source_row, baseline_row, candidates) in enumerate(
        zip(source_rows, baseline_rows, candidates_by_row), 1
    ):
        expected_signature = candidate_signature(candidates)
        existing = done.get(index)
        if existing and existing.get("candidate_signature") == expected_signature:
            continue
        question = str(source_row.get("question", ""))
        if question != str(baseline_row.get("question", "")):
            raise RuntimeError(f"{strategy} 第 {index} 行问题与 Top-5 基线不一致")
        context, context_stats, context_chars = make_context(candidates)
        routes, global_prompt = prompt_routes(baseline_row)
        tasks.append((index, source_row, baseline_row, candidates, context, context_stats,
                      context_chars, routes, global_prompt, expected_signature))

    print(f"[{strategy}] total={len(source_rows)} completed={len(done)} pending={len(tasks)} workers={workers}", flush=True)
    errors: list[dict[str, Any]] = []

    def one(task: tuple[Any, ...]) -> tuple[int, dict[str, Any]]:
        (index, source_row, baseline_row, candidates, context, context_stats,
         context_chars, routes, global_prompt, signature) = task
        client = OpenAICompatClient(api_key=api_key, base_url=base_url)
        answer_candidates = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        elapsed = 0.0
        for route in routes:
            module = route["prompt"]
            user = PromptOptimizer._qa_user(str(source_row.get("question", "")), context, module)
            answer, usage, seconds = call_answer(
                client, model, module["system_prompt"], user, retries
            )
            answer_candidates.append({
                "cluster_id": route.get("cluster_id"),
                "answer": answer,
            })
            for key, value in usage.items():
                usage_total[key] += value
            elapsed += seconds
        if len(answer_candidates) == 1:
            final_answer = answer_candidates[0]["answer"]
        else:
            fusion_input = "\n\n".join(
                f"候选答案 {i}（聚类 {item['cluster_id']}）：\n{item['answer']}"
                for i, item in enumerate(answer_candidates, 1)
            )
            fusion_user = (
                "请只依据给定候选答案和原问题，去重并合并为一个准确、完整的答案。"
                "不得添加候选答案之外的事实，不要输出分析过程。\n"
                f"用户问题：{source_row.get('question', '')}\n{fusion_input}"
            )
            final_answer, usage, seconds = call_answer(
                client, model, global_prompt["system_prompt"], fusion_user, retries
            )
            for key, value in usage.items():
                usage_total[key] += value
            elapsed += seconds

        retrieval = source_row.get("retrieval") or {}
        row = {
            "dataset_index": index,
            "id": str(source_row.get("id", baseline_row.get("id", index))),
            "question": str(source_row.get("question", "")),
            "reference_answer": str(source_row.get("reference_answer", "") or ""),
            "answer": final_answer,
            "answer_candidates": answer_candidates,
            "qa_strategy": strategy,
            "retrieval_query": source_row.get("retrieval_query", ""),
            "retrieval_top5": {
                "semantic_top5": (retrieval.get("semantic_results") or [])[:5],
                "dimension_top5": (retrieval.get("dimension_results") or [])[:5],
                "fusion_top5": candidates[:5],
            },
            "retrieval_context": {
                "fusion_top20": candidates,
                "candidate_count": len(candidates),
                "candidate_signature": signature,
            },
            "candidate_signature": signature,
            "context_stats": {
                "mode": "full_top20",
                "chunks": context_stats,
                "chunk_count": len(candidates),
                "provided_chars": context_chars,
            },
            "prompt": global_prompt,
            "prompt_mode": baseline_row.get("prompt_mode", "global"),
            "prompt_modules_used": routes,
            "cluster_routing": baseline_row.get("cluster_routing", {}),
            "model": model,
            "usage": usage_total,
            "generation_latency_seconds": round(elapsed, 3),
        }
        return index, row

    pending_errors: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(one, task): task[0] for task in tasks}
        for completed_count, future in enumerate(as_completed(future_map), 1):
            index = future_map[future]
            try:
                _, row = future.result()
                done[index] = row
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                pending_errors[index] = {
                    "dataset_index": index,
                    "id": source_rows[index - 1].get("id", index),
                    "question": source_rows[index - 1].get("question", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(
                    f"[{strategy}] {completed_count}/{len(tasks)} newly_processed "
                    f"success={len(done)} errors={len(pending_errors)}",
                    flush=True,
                )

    ordered = [done[index] for index in sorted(done)]
    write_jsonl(main_output, ordered)
    write_jsonl(error_path, [pending_errors[index] for index in sorted(pending_errors)])
    summary = {
        "strategy": strategy,
        "model": model,
        "total": len(source_rows),
        "completed": len(ordered),
        "failed": len(source_rows) - len(ordered),
        "context_k": 20,
        "context_candidate_count_min": min((x["context_stats"]["chunk_count"] for x in ordered), default=0),
        "context_candidate_count_max": max((x["context_stats"]["chunk_count"] for x in ordered), default=0),
        "average_context_chars": round(
            sum(x["context_stats"]["provided_chars"] for x in ordered) / len(ordered), 2
        ) if ordered else 0,
        "total_prompt_tokens": sum(x["usage"]["prompt_tokens"] for x in ordered),
        "total_completion_tokens": sum(x["usage"]["completion_tokens"] for x in ordered),
        "total_tokens": sum(x["usage"]["total_tokens"] for x in ordered),
        "average_latency_seconds": round(
            sum(x["generation_latency_seconds"] for x in ordered) / len(ordered), 3
        ) if ordered else 0,
        "output": str(main_output),
    }
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ordered, summary


def evaluate_strategy(
    *, rows: list[dict[str, Any]], output_dir: Path, model: str,
    base_url: str, api_key: str, workers: int,
) -> dict[str, Any]:
    eval_input = output_dir / "qa_eval_input_top20.jsonl"
    evaluation_dir = output_dir / "qa_evaluation"
    eval_rows = []
    for row in rows:
        copy = dict(row)
        top5_view = dict(copy.get("retrieval_top5") or {})
        top5_view["fusion_top5"] = (copy.get("retrieval_context") or {}).get("fusion_top20", [])
        copy["retrieval_top5"] = top5_view
        eval_rows.append(copy)
    write_jsonl(eval_input, eval_rows)

    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        openai_compat=True,
        interval=0.0,
    )
    summary = evaluate_qa(
        input_path=eval_input,
        output_dir=evaluation_dir,
        client=client,
        concurrency=max(1, workers),
    )
    (output_dir / "qa_summary.json").write_text(
        json.dumps({**json.loads((output_dir / "qa_summary.json").read_text(encoding="utf-8")),
                    "evaluation": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def valid_eval_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    output = {}
    for index, row in enumerate(rows, 1):
        dataset_index = int(row.get("dataset_index", index) or index)
        if not (row.get("judgment") or {}).get("error"):
            output[dataset_index] = row
    return output


def aggregate(eval_rows: dict[int, dict[str, Any]], indices: set[int]) -> dict[str, Any]:
    chosen = [eval_rows[index] for index in sorted(indices) if index in eval_rows]
    if not chosen:
        return {"n": 0}
    judgments = [row.get("judgment") or {} for row in chosen]
    mechanics = [row.get("mechanical_metrics") or {} for row in chosen]

    def mean(items: list[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "n": len(chosen),
        "pass_rate": round(sum(x.get("verdict") == "pass" for x in judgments) / len(chosen), 4),
        "unsupported_claim_rate": round(sum(bool(x.get("unsupported_claims")) for x in judgments) / len(chosen), 4),
        "correctness": mean(judgments, "correctness"),
        "completeness": mean(judgments, "completeness"),
        "relevance": mean(judgments, "relevance"),
        "groundedness": mean(judgments, "groundedness"),
        "char_f1": mean(mechanics, "char_f1"),
        "rouge_l_f1": mean(mechanics, "rouge_l_f1"),
        "numeric_f1": mean(mechanics, "numeric_f1"),
        "reference_evidence_char_recall": mean(mechanics, "reference_evidence_char_recall"),
    }


def retrieval_at5(base: Any, rows: list[dict[str, Any]], candidate_lists: list[list[dict[str, Any]]]) -> dict[str, float]:
    values = []
    for row, candidates in zip(rows, candidate_lists):
        gold = set(str(x) for x in (row.get("gold") or {}).get("chunk_ids", []))
        values.append(base.metric(candidates, gold, 5))
    if not values:
        return {}
    return {key: round(sum(item[key] for item in values) / len(values), 4)
            for key in ("hit", "recall", "mrr", "ndcg")}


def create_report(
    *, source_dir: Path, output_dir: Path, baseline_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]], candidate_lists: dict[str, list[list[dict[str, Any]]]],
    qa_summaries: dict[str, dict[str, Any]], evaluation_summaries: dict[str, dict[str, Any]],
    base: Any,
) -> dict[str, Any]:
    eval_paths = {
        "Top-5 基线": source_dir / "qa_evaluation" / "qa_answer_evaluation.jsonl",
        "Listwise Top-20": output_dir / "listwise" / "qa_evaluation" / "qa_answer_evaluation.jsonl",
        "Answerability Top-20": output_dir / "answerability" / "qa_evaluation" / "qa_answer_evaluation.jsonl",
        "Gold-free RRF Top-20": output_dir / "gold_free_rrf_k60" / "qa_evaluation" / "qa_answer_evaluation.jsonl",
    }
    evaluated = {name: valid_eval_rows(path) for name, path in eval_paths.items()}
    common = set.intersection(*(set(x) for x in evaluated.values())) if evaluated else set()
    comparison = {name: aggregate(rows, common) for name, rows in evaluated.items()}
    baseline_by_index = {int(row.get("dataset_index", i)): row for i, row in enumerate(baseline_rows, 1)}
    baseline_retrieval = []
    for index, source_row in enumerate(source_rows):
        baseline_retrieval.append(baseline_by_index[index + 1].get("retrieval_top5", {}).get("fusion_top5", []))
    retrieval_metrics = {
        "Top-5 基线": retrieval_at5(base, source_rows, baseline_retrieval),
        "Listwise Top-20": retrieval_at5(base, source_rows, candidate_lists["listwise"]),
        "Answerability Top-20": retrieval_at5(base, source_rows, candidate_lists["answerability"]),
        "Gold-free RRF Top-20": retrieval_at5(base, source_rows, candidate_lists["gold_free_rrf_k60"]),
    }
    retrieval_labels = {"hit": "Hit@5", "recall": "Gold Recall@5", "mrr": "MRR@5", "ndcg": "nDCG@5"}

    lines = [
        "# Top-20 上下文问答实验与 Top-5 基线对比",
        "",
        f"- 评测集：{len(source_rows)} 条同一问答集；QA 模型：DeepSeek-V4-Pro。",
        "- 三个 Top-20 方案分别使用其已生成的 20 个候选及各自排序作为完整上下文；没有重跑检索或 rerank。",
        "- QA Prompt 按题复用原 Top-5 实验记录中的 Prompt；问答评价使用同一套 QA 评价器。",
        f"- 四组评价共有样本数：{len(common)}。Top-5 基线历史评价若有单条失败，会从共同样本统计中排除。",
        "",
        "## 问答质量（共同评价样本）",
        "",
        "| 实验 | 上下文候选数 | N | Pass Rate | 正确性 /5 | 完整性 /5 | 相关性 /5 | 证据支撑 /5 | 无依据陈述率 | 字符 F1 | ROUGE-L F1 | 数字 F1 | 参考答案证据覆盖 Recall | Token（问答生成） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_order = ["Top-5 基线", "Listwise Top-20", "Answerability Top-20", "Gold-free RRF Top-20"]
    for name in table_order:
        item = comparison.get(name, {"n": 0})
        if name == "Top-5 基线":
            context_k = 5
            tokens = "—（历史未记录）"
        else:
            key = {"Listwise Top-20": "listwise", "Answerability Top-20": "answerability",
                   "Gold-free RRF Top-20": "gold_free_rrf_k60"}[name]
            context_k = qa_summaries.get(key, {}).get("context_k", 20)
            tokens = f"{qa_summaries.get(key, {}).get('total_tokens', 0):,}"
        if not item.get("n"):
            lines.append(f"| {name} | {context_k} | 0 | — | — | — | — | — | — | — | — | — | — | {tokens} |")
            continue
        lines.append(
            f"| {name} | {context_k} | {item['n']} | {item['pass_rate']:.2%} | "
            f"{item['correctness']:.2f} | {item['completeness']:.2f} | {item['relevance']:.2f} | "
            f"{item['groundedness']:.2f} | {item['unsupported_claim_rate']:.2%} | "
            f"{item['char_f1']:.4f} | {item['rouge_l_f1']:.4f} | "
            f"{item['numeric_f1']:.4f} | {item['reference_evidence_char_recall']:.4f} | {tokens} |"
        )

    lines.extend(["", "## 检索 Top-5 指标（用于核对 rerank 质量）", "",
                  "| 排序方案 | Hit@5 | Gold Recall@5 | MRR@5 | nDCG@5 |", "|---|---:|---:|---:|---:|"])
    for name in table_order:
        values = retrieval_metrics.get(name, {})
        if values:
            lines.append(f"| {name} | {values['hit']:.2%} | {values['recall']:.2%} | {values['mrr']:.4f} | {values['ndcg']:.4f} |")

    lines.extend([
        "",
        "## 运行量与耗时",
        "",
        "| 方案 | 成功生成 / 435 | 平均上下文字符数 | 生成平均秒数 / 题 | 生成总 Token | 评价成功 / 生成成功 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for strategy, label in (("listwise", "Listwise Top-20"), ("answerability", "Answerability Top-20"),
                            ("gold_free_rrf_k60", "Gold-free RRF Top-20")):
        qa = qa_summaries.get(strategy, {})
        ev = evaluation_summaries.get(strategy, {})
        lines.append(
            f"| {label} | {qa.get('completed', 0)}/{qa.get('total', 435)} | "
            f"{qa.get('average_context_chars', 0):.0f} | {qa.get('average_latency_seconds', 0):.2f} | "
            f"{qa.get('total_tokens', 0):,} | {ev.get('evaluated', 0)}/{qa.get('completed', 0)} |"
        )
    lines.extend([
        "",
        "> 说明：Pass Rate 和四项模型评分来自 LLM-as-a-judge；字符 F1、ROUGE-L、数字 F1 和证据覆盖是机械指标/文本重合代理，不能单独等同于事实正确。Top-20 评价时，评价器看到的也是完整 20 条上下文。",
        "",
    ])
    report_path = output_dir / "comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "common_evaluation_n": len(common),
        "qa_metrics_on_common_questions": comparison,
        "retrieval_metrics_at_5": retrieval_metrics,
        "evaluation_summaries": evaluation_summaries,
        "qa_generation_summaries": qa_summaries,
        "report": str(report_path),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, type=Path,
                        help="包含三个 rerank 输出和原 Top-5 QA 基线的远端 run 目录")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="新的独立实验输出目录，不覆盖来源实验")
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
    base = load_base_module()
    candidates, candidate_meta = candidate_sets(source_dir, base)
    source_rows = read_jsonl(source_dir / "rerank_ablation_v3_20260922" / "deepseek_listwise.jsonl")
    baseline_rows = read_jsonl(source_dir / "qa_results.jsonl")
    if len(baseline_rows) != len(source_rows):
        raise RuntimeError(f"Top-5 基线行数 {len(baseline_rows)} 与候选集 {len(source_rows)} 不一致")
    for index, (source_row, baseline_row) in enumerate(zip(source_rows, baseline_rows), 1):
        if source_row.get("question") != baseline_row.get("question"):
            raise RuntimeError(f"Top-5 基线与候选集第 {index} 行问题错位")

    manifest_path = output_dir / "experiment_manifest.json"
    manifest = {
        "experiment": "top20_context_qa_vs_existing_top5",
        "started_at_unix": time.time(),
        "source_run_dir": str(source_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "api_key": "redacted; server-local compatibility proxy",
        "qa_prompt": "reuse per-question prompt_modules_used/prompt from existing Top-5 QA result",
        "context_policy": "all up to 20 ranked rerank_candidates; Listwise pool rebuilt from cached scores",
        "candidate_metadata": candidate_meta,
        "workers": args.workers,
        "eval_workers": args.eval_workers,
        "max_retries": args.retries,
        "status": "running",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_run_dir") != str(source_dir) or existing.get("model") != args.model:
            raise RuntimeError(f"输出目录已有不匹配的 manifest：{manifest_path}")
        manifest["resumed"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preflight": candidate_meta, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)

    qa_summaries: dict[str, dict[str, Any]] = {}
    evaluation_summaries: dict[str, dict[str, Any]] = {}
    labels = {
        "listwise": "Listwise Top-20",
        "answerability": "Answerability Top-20",
        "gold_free_rrf_k60": "Gold-free RRF Top-20",
    }
    for strategy in ("listwise", "answerability", "gold_free_rrf_k60"):
        strategy_dir = output_dir / strategy
        qa_rows, qa_summary = generate_strategy(
            strategy=strategy,
            candidates_by_row=candidates[strategy],
            source_rows=source_rows,
            baseline_rows=baseline_rows,
            output_dir=strategy_dir,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            workers=args.workers,
            retries=args.retries,
        )
        qa_summaries[strategy] = qa_summary
        if qa_rows:
            print(f"[eval] 开始评价 {labels[strategy]}，上下文为完整 Top-20", flush=True)
            eval_summary = evaluate_strategy(
                rows=qa_rows,
                output_dir=strategy_dir,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                workers=args.eval_workers,
            )
            for retry_round in range(2):
                if not eval_summary.get("failed"):
                    break
                print(f"[eval] {labels[strategy]} 有 {eval_summary['failed']} 条评价失败，重试轮次 {retry_round + 1}", flush=True)
                eval_summary = evaluate_strategy(
                    rows=qa_rows,
                    output_dir=strategy_dir,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    workers=args.eval_workers,
                )
            evaluation_summaries[strategy] = eval_summary
        else:
            evaluation_summaries[strategy] = {"evaluated": 0, "failed": len(source_rows)}
        manifest["status"] = f"completed_{strategy}"
        manifest["last_updated_unix"] = time.time()
        manifest["qa_summaries"] = qa_summaries
        manifest["evaluation_summaries"] = evaluation_summaries
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = create_report(
        source_dir=source_dir,
        output_dir=output_dir,
        baseline_rows=baseline_rows,
        source_rows=source_rows,
        candidate_lists=candidates,
        qa_summaries=qa_summaries,
        evaluation_summaries=evaluation_summaries,
        base=base,
    )
    manifest["status"] = "completed"
    manifest["completed_at_unix"] = time.time()
    manifest["report"] = report["report"]
    manifest["qa_summaries"] = qa_summaries
    manifest["evaluation_summaries"] = evaluation_summaries
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
