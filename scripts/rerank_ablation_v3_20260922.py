#!/usr/bin/env python3
"""Compare three reranking schemes on one frozen retrieval result set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any


FACET_RULES = [
    ("time", ["什么时候", "何时", "哪年", "年份", "年代", "始建", "建于", "开始修建", "历经"]),
    ("person", ["谁", "哪位", "人物", "皇帝", "作者", "由谁", "和谁"]),
    ("place", ["哪里", "在哪", "位于", "地点", "怎么去", "路线", "公交", "停车"]),
    ("quantity", ["多少", "几个", "几处", "数量", "多大", "多长", "历时"]),
    ("comparison", ["区别", "不同", "相比", "比较", "差异", "哪个好"]),
    ("operation", ["开放", "预约", "门票", "票价", "寄存", "充电", "营业", "优惠"]),
    ("cause", ["为什么", "原因", "因何", "缘由"]),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def extract_facets(row: dict[str, Any]) -> list[dict[str, Any]]:
    question = norm_text(row.get("question"))
    anchors = [norm_text(x) for x in row.get("query_analysis", {}).get("fact_anchor_terms", []) if norm_text(x)]
    facets: list[dict[str, Any]] = []
    for facet_id, words in FACET_RULES:
        hits = [word for word in words if word in question]
        if hits:
            facets.append({"id": facet_id, "anchors": hits})
    if not facets:
        facets.append({"id": "general", "anchors": anchors[:8]})
    if anchors:
        facets[0]["anchors"] = list(dict.fromkeys(facets[0]["anchors"] + anchors[:8]))
    return facets


def facet_coverage(candidate: dict[str, Any], facets: list[dict[str, Any]]) -> dict[str, float]:
    text = norm_text(candidate.get("chunk_text"))
    dimension_text = norm_text(" ".join(str(x) for x in candidate.get("matched_dimensions", [])))
    matches: dict[str, float] = {}
    for facet in facets:
        anchors = [norm_text(x) for x in facet.get("anchors", []) if norm_text(x)]
        score = sum(1 for anchor in anchors if anchor and anchor in text) / len(anchors) if anchors else 0.0
        if facet["id"] in dimension_text:
            score = max(score, 0.75)
        matches[facet["id"]] = min(1.0, score)
    return matches


def candidate_pool(row: dict[str, Any], limit: int = 30) -> list[dict[str, Any]]:
    semantic = row.get("retrieval", {}).get("semantic_results", [])[:100]
    dimension = row.get("retrieval", {}).get("dimension_results", [])[:20]
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source, items in (("semantic", semantic), ("dimension", dimension)):
        for item in items:
            cid = str(item.get("chunk_id"))
            if cid not in by_id:
                by_id[cid] = dict(item)
                order.append(cid)
            target = by_id[cid]
            if source == "semantic":
                target["semantic_rank"] = item.get("rank")
                target["normalized_semantic_score"] = float(item.get("normalized_semantic_score") or 0.0)
            else:
                target["dimension_rank"] = item.get("rank")
                target["normalized_dimension_score"] = float(item.get("normalized_dimension_score") or 0.0)
            target["source_routes"] = sorted(set(target.get("source_routes", [])) | {source})
    for cid in order:
        item = by_id[cid]
        item.setdefault("normalized_semantic_score", 0.0)
        item.setdefault("normalized_dimension_score", 0.0)
        item["base_score"] = 0.8 * item["normalized_semantic_score"] + 0.2 * item["normalized_dimension_score"]
        item["agreement"] = int(item.get("semantic_rank", 999) <= 5 and item.get("dimension_rank", 999) <= 5)
    protected_ids = [str(x.get("chunk_id")) for x in semantic[:5] + dimension[:5]]
    ranked = sorted(order, key=lambda cid: by_id[cid]["base_score"], reverse=True)
    selected_ids = list(dict.fromkeys(protected_ids + ranked))[:limit]
    return [by_id[cid] for cid in selected_ids]


def metric(items: list[dict[str, Any]], gold: set[str], k: int) -> dict[str, float]:
    if not gold:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    ids = [str(x.get("chunk_id")) for x in items[:k]]
    hits = [x for x in ids if x in gold]
    first = next((i + 1 for i, x in enumerate(ids) if x in gold), None)
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ids) if x in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"hit": float(bool(hits)), "recall": len(set(hits)) / len(gold), "mrr": 1.0 / first if first else 0.0, "ndcg": dcg / ideal if ideal else 0.0}


def loss_counts(row: dict[str, Any], final: list[dict[str, Any]]) -> dict[str, int]:
    gold = set(str(x) for x in row.get("gold", {}).get("chunk_ids", []))
    sem_gold = {str(x.get("chunk_id")) for x in row.get("retrieval", {}).get("semantic_results", [])[:5]} & gold
    dim_gold = {str(x.get("chunk_id")) for x in row.get("retrieval", {}).get("dimension_results", [])[:5]} & gold
    final_gold = {str(x.get("chunk_id")) for x in final[:5]} & gold
    return {"semantic_lost_chunks": len(sem_gold - final_gold), "dimension_lost_chunks": len(dim_gold - final_gold), "any_route_hit_final_miss": int(bool(sem_gold or dim_gold) and not final_gold)}


def summarize_final(rows: list[dict[str, Any]], result_key: str | None = None) -> dict[str, Any]:
    metric_rows = {k: [] for k in (1, 3, 5, 10, 20)}
    loss = {"semantic_lost_chunks": 0, "dimension_lost_chunks": 0, "any_route_hit_final_miss": 0}
    for row in rows:
        final = row.get(result_key, []) if result_key else row.get("retrieval", {}).get("fusion_results", [])
        gold = set(str(x) for x in row.get("gold", {}).get("chunk_ids", []))
        for k in metric_rows:
            metric_rows[k].append(metric(final, gold, k))
        current_loss = loss_counts(row, final)
        for key in loss:
            loss[key] += current_loss[key]
    return {"total": len(rows), "metrics": {str(k): {name: sum(x[name] for x in values) / len(values) for name in ("hit", "recall", "mrr", "ndcg")} for k, values in metric_rows.items()}, "loss": loss}


def select_plain(candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: float(x.get("rerank_score", x.get("base_score", 0.0))), reverse=True)[:top_k]


def add_facet_scores(row: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    facets = extract_facets(row)
    for item in candidates:
        item["facet_scores"] = facet_coverage(item, facets)


def prepare_features(row: dict[str, Any], item: dict[str, Any]) -> list[float]:
    question = norm_text(row.get("question"))
    text = norm_text(item.get("chunk_text"))
    q_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", question))
    overlap = sum(1 for term in q_terms if term in text) / max(1, len(q_terms))
    anchors = [norm_text(x) for x in row.get("query_analysis", {}).get("fact_anchor_terms", []) if norm_text(x)]
    anchor_overlap = sum(1 for term in anchors if term in text) / max(1, len(anchors))
    facets = item.get("facet_scores", {})
    semantic_rank = float(item.get("semantic_rank") or 999)
    dimension_rank = float(item.get("dimension_rank") or 999)
    return [
        float(item.get("normalized_semantic_score") or 0.0), float(item.get("normalized_dimension_score") or 0.0), float(item.get("base_score") or 0.0),
        1.0 / (semantic_rank + 1.0) if semantic_rank < 999 else 0.0, 1.0 / (dimension_rank + 1.0) if dimension_rank < 999 else 0.0,
        float(item.get("agreement") or 0), float(semantic_rank <= 5), float(dimension_rank <= 5), float(len(item.get("source_routes", [])) >= 2),
        math.log1p(len(str(item.get("chunk_text", "")))), float(len(item.get("matched_dimensions", []) or [])), overlap, anchor_overlap,
        float(sum(v >= 0.5 for v in facets.values())), float(max(facets.values()) if facets else 0.0),
    ]


def train_lambdamart(rows: list[dict[str, Any]], seed: int = 42) -> dict[int, list[float]]:
    """Small LambdaMART-style pairwise model with question-group CV.

    LightGBM is not installed on the target server.  Regression trees fitted
    to within-question LambdaRank pseudo-gradients provide the same boosting
    structure without pretending that a pointwise classifier is LambdaMART.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeRegressor

    pools, x_groups, y_groups = [], [], []
    for row in rows:
        pool = candidate_pool(row, limit=30)
        add_facet_scores(row, pool)
        pools.append(pool)
        x_groups.append(np.asarray([prepare_features(row, x) for x in pool], dtype=np.float32))
        gold = set(str(x) for x in row.get("gold", {}).get("chunk_ids", []))
        y_groups.append(np.asarray([1.0 if str(x.get("chunk_id")) in gold else 0.0 for x in pool], dtype=np.float32))
    x = np.concatenate(x_groups, axis=0)
    y = np.concatenate(y_groups, axis=0)
    groups = np.concatenate([np.full(len(pool), i, dtype=np.int32) for i, pool in enumerate(pools)])
    predictions = {i: [] for i in range(len(rows))}
    splitter = GroupKFold(n_splits=5)
    rng = np.random.RandomState(seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), 1):
        model_score = np.zeros(len(train_idx), dtype=np.float64)
        train_groups = groups[train_idx]
        trees: list[tuple[DecisionTreeRegressor, float]] = []
        for _ in range(60):
            lambdas = np.zeros(len(train_idx), dtype=np.float64)
            for group_id in np.unique(train_groups):
                positions = np.where(train_groups == group_id)[0]
                positive = positions[y[train_idx][positions] > 0.5]
                negative = positions[y[train_idx][positions] <= 0.5]
                if not len(positive) or not len(negative):
                    continue
                for p in positive:
                    for n in negative:
                        diff = model_score[p] - model_score[n]
                        probability = 1.0 / (1.0 + math.exp(min(50.0, max(-50.0, diff))))
                        weight = probability * (1.0 - probability)
                        lambdas[p] += probability * weight
                        lambdas[n] -= probability * weight
            tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=8, random_state=int(rng.randint(0, 2**31 - 1)))
            tree.fit(x[train_idx], lambdas)
            model_score += 0.08 * tree.predict(x[train_idx])
            trees.append((tree, 0.08))
        valid_score = np.zeros(len(valid_idx), dtype=np.float64)
        for tree, rate in trees:
            valid_score += rate * tree.predict(x[valid_idx])
        for flat_idx, score in zip(valid_idx, valid_score):
            predictions[int(groups[flat_idx])].append(float(score))
        print(f"[lambdamart] fold {fold}/5 train={len(train_idx)} valid={len(valid_idx)}", flush=True)
    return {group_id: scores for group_id, scores in predictions.items()}


def run_lambdamart(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = train_lambdamart(rows)
    output: list[dict[str, Any]] = []
    for idx, original in enumerate(rows):
        row = json.loads(json.dumps(original, ensure_ascii=False))
        pool = candidate_pool(row, limit=30)
        add_facet_scores(row, pool)
        scores = predictions[idx]
        if len(scores) != len(pool):
            raise RuntimeError(f"LambdaMART prediction size mismatch for row {idx}: {len(scores)} vs {len(pool)}")
        for item, score in zip(pool, scores):
            item["rerank_score"] = float(score)
        row["rerank_strategy"] = "lambdamart"
        row["rerank_results"] = select_plain(pool)
        output.append(row)
    return summarize_final(output, "rerank_results") | {"strategy": "lambdamart", "status": "completed", "cv": "5-group-fold"}, output


def load_cross_encoder(model_path: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_path, max_length=512, local_files_only=True)


def run_cross_encoder(rows: list[dict[str, Any]], model_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        model = load_cross_encoder(model_path)
    except Exception as exc:
        return {"strategy": "cross_encoder", "status": "unavailable", "model": model_path, "error": repr(exc)}, []
    output: list[dict[str, Any]] = []
    for n, original in enumerate(rows, 1):
        row = json.loads(json.dumps(original, ensure_ascii=False))
        pool = candidate_pool(row, limit=30)
        pairs = [(str(row.get("question", "")), str(item.get("chunk_text", ""))[:4000]) for item in pool]
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        for item, score in zip(pool, scores):
            item["rerank_score"] = float(score)
        row["rerank_strategy"] = "cross_encoder"
        row["rerank_results"] = select_plain(pool)
        output.append(row)
        if n % 50 == 0:
            print(f"[cross_encoder] {n}/{len(rows)}", flush=True)
    return summarize_final(output, "rerank_results") | {"strategy": "cross_encoder", "status": "completed", "model": model_path}, output


def deepseek_rerank(row: dict[str, Any], client: Any, model_name: str, cache: dict[str, Any]) -> list[dict[str, Any]]:
    pool = candidate_pool(row, limit=20)
    facets = extract_facets(row)
    payload = [{"index": i, "chunk_id": item.get("chunk_id"), "text": str(item.get("chunk_text", ""))[:3000]} for i, item in enumerate(pool)]
    cache_key = hashlib.sha256((str(row.get("question")) + json.dumps(payload, ensure_ascii=False)).encode()).hexdigest()
    if cache_key in cache:
        result = cache[cache_key]
    else:
        prompt = ("你是检索重排器。只根据问题和候选片段判断相关性，不要使用任何golden标签。\n"
                  f"问题：{row.get('question', '')}\n问题要点：{json.dumps(facets, ensure_ascii=False)}\n"
                  "请返回JSON：{\"ranking\":[{\"index\":整数,\"relevance\":0到1,\"covered_facets\":[要点id]}]}。"
                  "按最相关到最不相关排序，片段必须直接支持问题答案。\n"
                  f"候选：{json.dumps(payload, ensure_ascii=False)}")
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1800)
        content = response.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", content, re.S)
        result = json.loads(match.group(0) if match else "{}")
        cache[cache_key] = result
    by_index = {int(x.get("index")): x for x in result.get("ranking", []) if str(x.get("index", "")).isdigit()}
    output = []
    for i, item in enumerate(pool):
        copy = dict(item)
        judged = by_index.get(i, {})
        copy["rerank_score"] = float(judged.get("relevance", copy.get("base_score", 0.0)) or 0.0)
        copy["facet_scores"] = {str(x): 1.0 for x in judged.get("covered_facets", [])}
        output.append(copy)
    return sorted(output, key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)


def run_deepseek(rows: list[dict[str, Any]], base_url: str, model_name: str, api_key: str, cache_path: Path, interval: float, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def one(index: int, original: dict[str, Any]) -> tuple[int, dict[str, Any], bool, str | None]:
        # A client per worker avoids sharing a synchronous httpx connection
        # between threads while the local FastAPI gateway handles concurrency.
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)
        row = json.loads(json.dumps(original, ensure_ascii=False))
        failed = False
        error_text = None
        try:
            candidates = deepseek_rerank(row, client, model_name, cache)
        except Exception as exc:
            failed = True
            error_text = repr(exc)
            candidates = candidate_pool(row, limit=20)
            for item in candidates:
                item["rerank_score"] = float(item.get("base_score", 0.0))
        add_facet_scores(row, candidates)
        row["rerank_strategy"] = "deepseek_listwise"
        row["rerank_results"] = select_plain(candidates)
        if interval > 0:
            time.sleep(interval)
        return index, row, failed, error_text

    output_by_index: dict[int, dict[str, Any]] = {}
    failures = 0
    completed = 0
    worker_count = max(1, min(int(workers), 8))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(one, index, original) for index, original in enumerate(rows)]
        for future in as_completed(futures):
            index, row, failed, error_text = future.result()
            output_by_index[index] = row
            completed += 1
            if failed:
                failures += 1
                print(f"[deepseek_listwise] fallback row={index + 1} error={error_text}", flush=True)
            if completed % 25 == 0:
                print(f"[deepseek_listwise] {completed}/{len(rows)} cache={len(cache)} failures={failures} workers={worker_count}", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    output = [output_by_index[i] for i in range(len(rows))]
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    summary = summarize_final(output, "rerank_results") | {"strategy": "deepseek_listwise", "status": "completed", "model": model_name, "failures": failures, "cache_size": len(cache), "workers": worker_count}
    return summary, output, cache


def write_result(output_dir: Path, name: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if rows:
        (output_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    (output_dir / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cross-encoder-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--deepseek-base-url", default="http://127.0.0.1:8911/v1")
    parser.add_argument("--deepseek-model", default="DeepSeek-V4-Pro")
    parser.add_argument("--deepseek-api-key", default="sdu-cookie")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    summaries: dict[str, Any] = {"current_fusion": summarize_final(rows)}
    print(json.dumps({"current_fusion": summaries["current_fusion"]}, ensure_ascii=False, indent=2), flush=True)
    cross_summary, cross_rows = run_cross_encoder(rows, args.cross_encoder_model)
    summaries["cross_encoder"] = cross_summary
    write_result(args.output_dir, "cross_encoder", cross_summary, cross_rows)
    print(f"[cross_encoder] {cross_summary.get('status')}: {cross_summary.get('error', '')}", flush=True)
    lm_summary, lm_rows = run_lambdamart(rows)
    summaries["lambdamart"] = lm_summary
    write_result(args.output_dir, "lambdamart", lm_summary, lm_rows)
    ds_summary, ds_rows, cache = run_deepseek(rows, args.deepseek_base_url, args.deepseek_model, args.deepseek_api_key, args.output_dir / "deepseek_listwise_cache.json", args.interval, args.workers)
    summaries["deepseek_listwise"] = ds_summary
    write_result(args.output_dir, "deepseek_listwise", ds_summary, ds_rows)
    (args.output_dir / "deepseek_listwise_cache.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
