"""离线检索评测、Golden 映射和 baseline/candidate 对比。

该模块只评估检索，不自动生成答案。它接受带 Golden 证据的 JSON/JSONL
测试集，分别计算语义、维度和融合三路的排名指标，并输出逐问题结果。
"""

from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompting import PromptOptimizer
from .retrieval import Retriever


ROUTES = ("semantic", "dimension", "fusion")
DEFAULT_KS = (1, 3, 5, 10, 20)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def _string_list(value: Any) -> List[str]:
    result = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            item = _first_value(item, ("evidence_text", "text", "source", "content", "value"), "")
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _read_records(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"评测集不存在: {file_path}")
    if file_path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL 第 {line_number} 行必须是对象")
            records.append(dict(value))
        return records

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"不是合法 JSON: {file_path}: {exc}") from exc
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        for key in ("examples", "records", "questions", "data", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if "question" in raw or "query" in raw:
            return [dict(raw)]
    raise ValueError("评测集必须是对象数组、JSONL，或包含 examples/records/questions/data 的对象")


def normalize_gold_record(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """兼容文档规范和现有 fair_qa/tourist 数据的字段别名。"""
    nested = raw.get("gold") if isinstance(raw.get("gold"), Mapping) else {}
    record_id = _first_value(raw, ("id", "question_id", "qid"), f"q_{index:04d}")
    question = _first_value(raw, ("question", "query", "prompt"), "")
    reference_answer = _first_value(
        raw, ("reference_answer", "answer", "standard_answer", "gold_answer"), ""
    )
    if not reference_answer:
        reference_answer = _first_value(nested, ("reference_answer", "answer"), "")

    chunk_ids = _first_value(raw, ("gold_chunk_ids", "gold_ids", "gold_chunks"), None)
    if chunk_ids is None:
        chunk_ids = _first_value(nested, ("chunk_ids", "gold_chunk_ids"), [])
    chunk_ids = [str(value) for value in _as_list(chunk_ids) if str(value).strip()]

    evidence_objects = _first_value(raw, ("gold_evidence", "gold_evidences", "evidence"), None)
    if evidence_objects is None:
        evidence_objects = _first_value(nested, ("evidence", "gold_evidence"), [])
    evidence: List[Dict[str, Any]] = []
    evidence_texts = _string_list(
        _first_value(raw, ("gold_evidence_texts", "evidence_texts", "gold_evidence_text"), [])
    )
    if not evidence_texts:
        evidence_texts = _string_list(_first_value(nested, ("evidence_texts", "gold_evidence_texts"), []))
    if not evidence_texts:
        # 现有 Tourist/fair_qa 数据使用 source 保存证据片段。
        evidence_texts = _string_list(raw.get("source"))

    for item in _as_list(evidence_objects):
        if isinstance(item, Mapping):
            text = str(_first_value(item, ("evidence_text", "text", "source", "content"), "") or "").strip()
            entry = {
                "source_doc_id": str(_first_value(item, ("source_doc_id", "doc_id", "source_file"), "") or ""),
                "evidence_text": text,
                "start_offset": _first_value(item, ("start_offset", "start"), None),
                "end_offset": _first_value(item, ("end_offset", "end"), None),
            }
            if text or entry["source_doc_id"]:
                evidence.append(entry)
        elif str(item).strip():
            evidence.append({"source_doc_id": "", "evidence_text": str(item).strip()})
    existing_texts = {item.get("evidence_text", "") for item in evidence}
    for text in evidence_texts:
        if text not in existing_texts:
            evidence.append({"source_doc_id": "", "evidence_text": text})

    relevance = _first_value(raw, ("gold_relevance", "relevance"), {})
    if not isinstance(relevance, Mapping):
        relevance = {}

    return {
        "id": str(record_id),
        "question": str(question or "").strip(),
        "gold_chunk_ids": list(dict.fromkeys(chunk_ids)),
        "gold_evidence": evidence,
        "gold_evidence_texts": [item["evidence_text"] for item in evidence if item.get("evidence_text")],
        "gold_relevance": {str(key): float(value) for key, value in relevance.items()
                           if isinstance(value, (int, float))},
        "reference_answer": str(reference_answer or ""),
        "question_type": str(_first_value(raw, ("question_type", "type", "category"), "") or ""),
        "spot": str(_first_value(raw, ("spot", "attraction", "scenic_spot"), "") or ""),
        "answerable": bool(raw.get("answerable", True)),
    }


def load_gold_records(path: str | Path) -> List[Dict[str, Any]]:
    records = [normalize_gold_record(item, index) for index, item in enumerate(_read_records(path), 1)]
    invalid = [item["id"] for item in records if not item["question"]]
    if invalid:
        raise ValueError(f"以下评测样本缺少 question/query: {invalid[:5]}")
    return records


def load_chunks(run_dir: str | Path) -> List[Dict[str, Any]]:
    root = Path(run_dir)
    candidates = (root / "chunks.json", root / "chunks" / "chunks.json")
    for path in candidates:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _normalize_text(value: Any) -> str:
    # 统一空白、标点和大小写，避免证据末尾的句号/逗号差异阻断映射。
    return "".join(
        char for char in str(value or "").lower()
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("chunk_text_full") or chunk.get("chunk_text") or chunk.get("doc_text") or "")


def _source_matches(chunk: Mapping[str, Any], source_doc_id: str) -> bool:
    if not source_doc_id:
        return True
    target = str(source_doc_id)
    values = [chunk.get(key, "") for key in ("doc_id", "parent_doc_id", "source_file", "doc_title")]
    return any(target == str(value) or target in str(value) for value in values if value)


def _offset_match(chunk: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    start = evidence.get("start_offset")
    end = evidence.get("end_offset")
    chunk_start = _first_value(chunk, ("start_offset", "source_start", "char_start"), None)
    chunk_end = _first_value(chunk, ("end_offset", "source_end", "char_end"), None)
    if not all(isinstance(value, (int, float)) for value in (start, end, chunk_start, chunk_end)):
        return False
    return float(chunk_start) < float(end) and float(chunk_end) > float(start)


def map_gold_to_chunks(record: Mapping[str, Any], chunks: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """把证据文本/offset 映射为当前索引版本的 chunk ID。"""
    output = dict(record)
    chunks = [dict(chunk) for chunk in chunks]
    mapped_ids = list(dict.fromkeys(str(value) for value in record.get("gold_chunk_ids", []) if str(value)))
    evidence_matches = []
    unmatched = []
    basis = "chunk_id" if mapped_ids else "unmatched"

    for evidence in record.get("gold_evidence", []) or []:
        if not isinstance(evidence, Mapping):
            continue
        text = str(evidence.get("evidence_text") or "").strip()
        source_doc_id = str(evidence.get("source_doc_id") or "")
        candidates = [chunk for chunk in chunks if _source_matches(chunk, source_doc_id)]
        matches = []
        match_basis = ""
        if evidence.get("start_offset") is not None and evidence.get("end_offset") is not None:
            matches = [chunk for chunk in candidates if _offset_match(chunk, evidence)]
            if matches:
                match_basis = "offset"
        if not matches and text:
            matches = [chunk for chunk in candidates if text in _chunk_text(chunk)]
            if matches:
                match_basis = "evidence_text"
        if not matches and text:
            normalized = _normalize_text(text)
            matches = [chunk for chunk in candidates if normalized and normalized in _normalize_text(_chunk_text(chunk))]
            if matches:
                match_basis = "normalized_text"
        if not matches and text and len(text) >= 12 and candidates:
            scored = sorted(
                ((difflib.SequenceMatcher(None, _normalize_text(text), _normalize_text(_chunk_text(chunk))).ratio(), chunk)
                 for chunk in candidates),
                key=lambda pair: pair[0], reverse=True,
            )
            if scored and scored[0][0] >= 0.72:
                matches = [scored[0][1]]
                match_basis = "approximate"
        match_ids = [str(chunk.get("chunk_id") or chunk.get("doc_id") or "") for chunk in matches]
        match_ids = [value for value in match_ids if value]
        if match_ids:
            mapped_ids.extend(match_ids)
            evidence_matches.append({"evidence_text": text, "chunk_ids": match_ids, "match_basis": match_basis})
            if basis in {"unmatched", ""}:
                basis = match_basis
        elif text:
            unmatched.append(text)

    mapped_ids = list(dict.fromkeys(mapped_ids))
    if basis == "chunk_id" and evidence_matches:
        basis = "chunk_id+evidence"
    if not basis:
        basis = "unmatched"
    output["gold_chunk_ids"] = mapped_ids
    output["gold_resolution"] = {
        "match_basis": basis,
        "mapped_chunk_count": len(mapped_ids),
        "unmatched_evidence_texts": unmatched,
        "evidence_matches": evidence_matches,
        "gold_mapped": bool(mapped_ids),
    }
    return output


def _result_id(item: Mapping[str, Any]) -> str:
    return str(item.get("chunk_id") or item.get("id") or "")


def _ordered_ids(results: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
    ordered = []
    ranks = {}
    for position, item in enumerate(results, 1):
        chunk_id = _result_id(item)
        if not chunk_id or chunk_id in ranks:
            continue
        ordered.append(chunk_id)
        ranks[chunk_id] = position
    return ordered, ranks


def _ndcg(ordered: Sequence[str], gold_ids: set[str], relevance: Mapping[str, float], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for position, chunk_id in enumerate(ordered[:k], 1):
        rel = float(relevance.get(chunk_id, 1.0 if chunk_id in gold_ids else 0.0))
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(position + 1)
    ideal_rels = sorted(
        [float(relevance.get(chunk_id, 1.0)) for chunk_id in gold_ids], reverse=True
    )[:k]
    idcg = sum((2 ** rel - 1) / math.log2(position + 1) for position, rel in enumerate(ideal_rels, 1))
    return dcg / idcg if idcg else 0.0


def route_metrics(
    results: Sequence[Mapping[str, Any]],
    gold_ids: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    relevance: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    gold = {str(value) for value in gold_ids if str(value)}
    relevance = relevance or {}
    ordered, ranks = _ordered_ids(results)
    gold_ranks = [ranks[chunk_id] for chunk_id in gold if chunk_id in ranks]
    first_rank = min(gold_ranks) if gold_ranks else None
    output: Dict[str, Any] = {
        "first_gold_rank": first_rank,
        "miss": first_rank is None,
        "rr": round(1.0 / first_rank, 8) if first_rank else 0.0,
        "result_count": len(ordered),
    }
    for k in sorted({int(value) for value in ks if int(value) > 0}):
        found = set(ordered[:k]) & gold
        hit = bool(found)
        recall = len(found) / len(gold) if gold else 0.0
        output[f"hit_at_{k}"] = hit
        output[f"gold_recall_at_{k}"] = round(recall, 8)
        output[f"precision_at_{k}"] = round(len(found) / k, 8)
        output[f"ndcg_at_{k}"] = round(_ndcg(ordered, gold, relevance, k), 8)
        output[f"rr_at_{k}"] = round(1.0 / first_rank, 8) if first_rank and first_rank <= k else 0.0
    return output


def _compact_result(item: Mapping[str, Any], gold_ids: set[str], text_chars: int) -> Dict[str, Any]:
    result = {}
    for key in (
        "chunk_id", "rank", "score", "final_score", "source", "sem_rank", "dim_rank",
        "doc_id", "parent_doc_id", "doc_title", "chunk_gen_title", "source_file",
        "spot_name", "dimension_paths", "matched_dimensions", "matches",
    ):
        if key in item:
            result[key] = item[key]
    chunk_id = _result_id(item)
    result["chunk_id"] = chunk_id
    text = str(item.get("chunk_text_full") or item.get("chunk_text") or "")
    if text_chars > 0 and len(text) > text_chars:
        result["chunk_text"] = text[:text_chars]
        result["text_truncated"] = True
    else:
        result["chunk_text"] = text
        result["text_truncated"] = False
    result["is_gold"] = chunk_id in gold_ids
    return result


def evaluate_row(
    record: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    ks: Sequence[int] = DEFAULT_KS,
    eval_depth: int = 20,
    result_text_chars: int = 800,
) -> Dict[str, Any]:
    gold_ids = {str(value) for value in record.get("gold_chunk_ids", []) if str(value)}
    routes = {
        "semantic": retrieval.get("semantic_candidates") or retrieval.get("semantic_results") or [],
        "dimension": retrieval.get("dimension_candidates") or retrieval.get("dimension_results") or [],
        "fusion": retrieval.get("fusion_candidates") or retrieval.get("fusion_results") or [],
    }
    metrics = {
        route: route_metrics(items, gold_ids, ks, record.get("gold_relevance", {}))
        for route, items in routes.items()
    }
    max_k = max([int(value) for value in ks if int(value) > 0] or [5])
    semantic_hit = bool(metrics["semantic"].get(f"hit_at_{max_k}", False))
    dimension_hit = bool(metrics["dimension"].get(f"hit_at_{max_k}", False))
    fusion_hit = bool(metrics["fusion"].get(f"hit_at_{max_k}", False))
    route_class = (
        "both_hit" if semantic_hit and dimension_hit else
        "semantic_only_hit" if semantic_hit else
        "dimension_only_hit" if dimension_hit else
        "neither_hit"
    )
    penalty = max(int(eval_depth), max_k) + 1
    best_branch_rank = min(
        metrics["semantic"].get("first_gold_rank") or penalty,
        metrics["dimension"].get("first_gold_rank") or penalty,
    )
    fusion_rank = metrics["fusion"].get("first_gold_rank") or penalty
    fusion_rank_gain = best_branch_rank - fusion_rank
    semantic_top = {_result_id(item) for item in routes["semantic"][:max_k] if _result_id(item)}
    dimension_top = {_result_id(item) for item in routes["dimension"][:max_k] if _result_id(item)}
    union = semantic_top | dimension_top
    jaccard = len(semantic_top & dimension_top) / len(union) if union else 0.0
    fusion_fields = {
        "best_branch_rank": None if best_branch_rank == penalty else best_branch_rank,
        "fusion_rank": None if fusion_rank == penalty else fusion_rank,
        "rank_gain": fusion_rank_gain,
        "rescue_at_k": fusion_hit and not (semantic_hit and dimension_hit),
        "rescued_from_semantic_miss": fusion_hit and not semantic_hit,
        "rescued_from_dimension_miss": fusion_hit and not dimension_hit,
        "harm_at_k": (semantic_hit or dimension_hit) and not fusion_hit,
        "route_jaccard_at_k": round(jaccard, 8),
    }
    for k in sorted({int(value) for value in ks if int(value) > 0}):
        sem_hit_k = bool(metrics["semantic"].get(f"hit_at_{k}", False))
        dim_hit_k = bool(metrics["dimension"].get(f"hit_at_{k}", False))
        fusion_hit_k = bool(metrics["fusion"].get(f"hit_at_{k}", False))
        fusion_fields[f"rescue_at_{k}"] = fusion_hit_k and not (sem_hit_k and dim_hit_k)
        fusion_fields[f"harm_at_{k}"] = (sem_hit_k or dim_hit_k) and not fusion_hit_k
    labels = []
    if not record.get("answerable", True):
        labels.append("unanswerable")
    if not record.get("gold_resolution", {}).get("gold_mapped", bool(gold_ids)):
        labels.append("gold_unmapped")
    if route_class == "neither_hit":
        labels.append("both_branch_miss")
    if fusion_fields["harm_at_k"]:
        labels.append("fusion_harm")
    if fusion_fields["rescue_at_k"]:
        labels.append("fusion_rescue")
    if not fusion_hit:
        labels.append("fusion_miss")
    return {
        "id": str(record["id"]),
        "question": record.get("question", ""),
        "gold": {
            "chunk_ids": sorted(gold_ids),
            "evidence_texts": record.get("gold_evidence_texts", []),
            "match_basis": record.get("gold_resolution", {}).get("match_basis", ""),
            "gold_resolution": record.get("gold_resolution", {}),
        },
        "reference_answer": record.get("reference_answer", ""),
        "question_type": record.get("question_type", ""),
        "spot": record.get("spot", ""),
        "answerable": bool(record.get("answerable", True)),
        "retrieval_query": retrieval.get("query", record.get("question", "")),
        "query_analysis": retrieval.get("query_analysis", {}),
        "retrieval_metrics": {
            **metrics,
            "route_class": route_class,
            "fusion_diagnostics": fusion_fields,
        },
        "retrieval": {
            f"{route}_results": [_compact_result(item, gold_ids, result_text_chars) for item in items]
            for route, items in routes.items()
        },
        "badcase": {
            "is_badcase": bool(fusion_fields["harm_at_k"] or not fusion_hit),
            "labels": list(dict.fromkeys(labels)),
        },
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numbers) / len(numbers), 8) if numbers else None


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if row.get("answerable", True)]
    output: Dict[str, Any] = {
        "total": len(rows),
        "answerable": len(valid_rows),
        "gold_mapped": sum(bool(row.get("gold", {}).get("chunk_ids")) for row in valid_rows),
        "badcase_count": sum(bool(row.get("badcase", {}).get("is_badcase")) for row in valid_rows),
        "routes": {},
        "route_class_counts": {},
        "fusion": {},
    }
    for route in ROUTES:
        route_rows = [row.get("retrieval_metrics", {}).get(route, {}) for row in valid_rows]
        route_summary = {
            "first_gold_rank_mean_on_hits": _mean(item.get("first_gold_rank") for item in route_rows),
            "miss_count": sum(bool(item.get("miss", True)) for item in route_rows),
        }
        for k in sorted({int(value) for value in ks if int(value) > 0}):
            route_summary[f"mrr@{k}"] = _mean(item.get(f"rr_at_{k}", 0.0) for item in route_rows) or 0.0
            route_summary[f"hit_rate@{k}"] = _mean(
                1.0 if item.get(f"hit_at_{k}", False) else 0.0 for item in route_rows
            ) or 0.0
            route_summary[f"gold_recall@{k}"] = _mean(item.get(f"gold_recall_at_{k}", 0.0) for item in route_rows) or 0.0
            route_summary[f"precision@{k}"] = _mean(item.get(f"precision_at_{k}", 0.0) for item in route_rows) or 0.0
            route_summary[f"ndcg@{k}"] = _mean(item.get(f"ndcg_at_{k}", 0.0) for item in route_rows) or 0.0
        output["routes"][route] = route_summary
    for row in valid_rows:
        key = row.get("retrieval_metrics", {}).get("route_class", "unknown")
        output["route_class_counts"][key] = output["route_class_counts"].get(key, 0) + 1
    fusion_rows = [row.get("retrieval_metrics", {}).get("fusion_diagnostics", {}) for row in valid_rows]
    for k in sorted({int(value) for value in ks if int(value) > 0}):
        output["fusion"][f"rescue_rate@{k}"] = _mean(
            1.0 if item.get(f"rescue_at_{k}", False) else 0.0 for item in fusion_rows
        ) or 0.0
        output["fusion"][f"harm_rate@{k}"] = _mean(
            1.0 if item.get(f"harm_at_{k}", False) else 0.0 for item in fusion_rows
        ) or 0.0
    output["fusion"]["rank_gain_mean"] = _mean(item.get("rank_gain") for item in fusion_rows)
    return output


def aggregate_summary(rows: Sequence[Mapping[str, Any]], ks: Sequence[int] = DEFAULT_KS) -> Dict[str, Any]:
    """生成全量及 question_type/spot 分组汇总。"""
    summary = _aggregate_rows(rows, ks)
    for field in ("question_type", "spot"):
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            value = str(row.get(field) or "(empty)")
            groups.setdefault(value, []).append(row)
        summary[f"by_{field}"] = {key: _aggregate_rows(value, ks) for key, value in sorted(groups.items())}
    return summary


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Answerable: {summary.get('answerable', 0)}",
        f"- Gold mapped: {summary.get('gold_mapped', 0)}",
        f"- Fusion badcases: {summary.get('badcase_count', 0)}",
        "",
        "| Route | MRR@5 | HitRate@5 | GoldenRecall@5 | nDCG@5 | Misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        item = summary.get("routes", {}).get(route, {})
        lines.append(
            f"| {route} | {item.get('mrr@5', 0):.4f} | {item.get('hit_rate@5', 0):.4f} | "
            f"{item.get('gold_recall@5', 0):.4f} | {item.get('ndcg@5', 0):.4f} | {item.get('miss_count', 0)} |"
        )
    lines.extend(["", "## Route Classes", ""])
    for key, value in sorted((summary.get("route_class_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_ks(values: Sequence[int] | str) -> Tuple[int, ...]:
    if isinstance(values, str):
        values = [int(item.strip()) for item in values.split(",") if item.strip()]
    result = tuple(sorted({int(value) for value in values if int(value) > 0}))
    return result or DEFAULT_KS


def run_evaluation(
    *,
    run_dir: str | Path,
    dataset_path: str | Path,
    settings,
    output_dir: str | Path,
    top_k: int = 10,
    eval_depth: int = 20,
    ks: Sequence[int] = DEFAULT_KS,
    result_text_chars: int = 800,
    query_expansion: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = _parse_ks(ks)
    records = load_gold_records(dataset_path)
    chunks = load_chunks(run_dir)
    records = [map_gold_to_chunks(record, chunks) for record in records]
    embeddings = EmbeddingModel(
        settings.model_path, settings.embedding_device, settings.vector_dim, settings.mock
    )
    prompt_manager = None
    if query_expansion:
        client = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            openai_compat=settings.llm_openai_compat,
            interval=settings.llm_interval,
            mock=settings.mock,
        )
        prompt_manager = PromptOptimizer(client=client, run_dir=run_dir)
    retriever = Retriever(run_dir=run_dir, settings=settings, embeddings=embeddings, llm=None)
    rows = []
    for index, record in enumerate(records, 1):
        if prompt_manager:
            expansion = prompt_manager.expand(record["question"])
            retrieval_query = " | ".join(expansion.get("sub_queries") or [record["question"]])
        else:
            retrieval_query = record["question"]
        retrieval = retriever.search(
            retrieval_query,
            top_k=top_k,
            semantic_pool=max(int(eval_depth), int(settings.semantic_pool)),
            dimension_pool=max(int(eval_depth), int(settings.dimension_pool)),
        )
        retrieval["query"] = retrieval_query
        row = evaluate_row(record, retrieval, ks, eval_depth, result_text_chars)
        row["query_expansion"] = retrieval_query != record["question"]
        rows.append(row)
        if index % 100 == 0:
            print(f"[评测] {index}/{len(records)}")

    summary = aggregate_summary(rows, ks)
    config = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "top_k": int(top_k),
        "eval_depth": int(eval_depth),
        "ks": list(ks),
        "backend": settings.backend,
        "collection": settings.collection,
        "embedding_model": settings.model_path or "configured_default",
        "query_expansion": bool(query_expansion),
    }
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    _write_jsonl(results_path, rows)
    summary_path.write_text(json.dumps({"config": config, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return {
        "config": config,
        "summary": summary,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
    }


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], route: str, key: str) -> Optional[float]:
    left = candidate.get("retrieval_metrics", {}).get(route, {}).get(key)
    right = baseline.get("retrieval_metrics", {}).get(route, {}).get(key)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(left) - float(right), 8)


def _diagnostic_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> Optional[float]:
    left = candidate.get("retrieval_metrics", {}).get("fusion_diagnostics", {}).get(key)
    right = baseline.get("retrieval_metrics", {}).get("fusion_diagnostics", {}).get(key)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(left) - float(right), 8)


def compare_evaluations(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = _parse_ks(ks)
    baseline_rows = {str(row.get("id")): row for row in _read_jsonl(baseline_path)}
    candidate_rows = {str(row.get("id")): row for row in _read_jsonl(candidate_path)}
    common_ids = sorted(set(baseline_rows) & set(candidate_rows))
    changes = []
    improved = regressed = unchanged = 0
    max_k = max(ks)
    for record_id in common_ids:
        baseline = baseline_rows[record_id]
        candidate = candidate_rows[record_id]
        b_hit = bool(baseline.get("retrieval_metrics", {}).get("fusion", {}).get(f"hit_at_{max_k}", False))
        c_hit = bool(candidate.get("retrieval_metrics", {}).get("fusion", {}).get(f"hit_at_{max_k}", False))
        b_rr = float(baseline.get("retrieval_metrics", {}).get("fusion", {}).get(f"rr_at_{max_k}", 0.0) or 0.0)
        c_rr = float(candidate.get("retrieval_metrics", {}).get("fusion", {}).get(f"rr_at_{max_k}", 0.0) or 0.0)
        if c_hit and not b_hit or c_hit == b_hit and c_rr > b_rr:
            status = "improved"
            improved += 1
        elif b_hit and not c_hit or c_hit == b_hit and c_rr < b_rr:
            status = "regressed"
            regressed += 1
        else:
            status = "unchanged"
            unchanged += 1
        changes.append({
            "id": record_id,
            "question": candidate.get("question", baseline.get("question", "")),
            "status": status,
            "fusion_hit_change": int(c_hit) - int(b_hit),
            "fusion_rr_change": round(c_rr - b_rr, 8),
            "semantic_first_rank_change": _metric_delta(candidate, baseline, "semantic", "first_gold_rank"),
            "dimension_first_rank_change": _metric_delta(candidate, baseline, "dimension", "first_gold_rank"),
            "fusion_rank_gain_change": _diagnostic_delta(candidate, baseline, "rank_gain"),
            "baseline_labels": baseline.get("badcase", {}).get("labels", []),
            "candidate_labels": candidate.get("badcase", {}).get("labels", []),
        })
    baseline_common = [baseline_rows[key] for key in common_ids]
    candidate_common = [candidate_rows[key] for key in common_ids]
    baseline_summary = aggregate_summary(baseline_common, ks)
    candidate_summary = aggregate_summary(candidate_common, ks)
    summary = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "common_count": len(common_ids),
        "baseline_only": sorted(set(baseline_rows) - set(candidate_rows)),
        "candidate_only": sorted(set(candidate_rows) - set(baseline_rows)),
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_count": unchanged,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "route_metric_deltas": {
            route: {
                key: round(
                    float(candidate_summary.get("routes", {}).get(route, {}).get(key, 0.0) or 0.0)
                    - float(baseline_summary.get("routes", {}).get(route, {}).get(key, 0.0) or 0.0),
                    8,
                )
                for key in (f"mrr@{max_k}", f"hit_rate@{max_k}", f"gold_recall@{max_k}", f"ndcg@{max_k}")
            }
            for route in ROUTES
        },
    }
    (output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(output_dir / "comparison.jsonl", changes)
    lines = [
        "# Baseline vs Candidate",
        "",
        f"- Common questions: {len(common_ids)}",
        f"- Improved: {improved}",
        f"- Regressed: {regressed}",
        f"- Unchanged: {unchanged}",
        "",
        "| Route | MRR delta | HitRate delta | GoldenRecall delta | nDCG delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for route, values in summary["route_metric_deltas"].items():
        lines.append(
            f"| {route} | {values[f'mrr@{max_k}']:.4f} | {values[f'hit_rate@{max_k}']:.4f} | "
            f"{values[f'gold_recall@{max_k}']:.4f} | {values[f'ndcg@{max_k}']:.4f} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "changes_path": str(output_dir / "comparison.jsonl")}
