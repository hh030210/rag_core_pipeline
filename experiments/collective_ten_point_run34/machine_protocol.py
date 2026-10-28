#!/usr/bin/env python3
"""Run and evaluate the standalone TouristQA experiment matrix.

The bundle contains 412 held-out evaluation questions, eight collective prompts,
saved BGE routes, and external fusion retrieval results.  No chapter3 import is used.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EXPERIMENTS: Tuple[str, ...] = (
    "q_only",
    "crowd_q",
    "crowd_q_gold",
    "crowd_q_gold_top2",
    "crowd_q_top3",
)

# Kept separate so the original ``--experiments all`` remains unchanged.
TOPK_EXPERIMENTS: Tuple[str, ...] = (
    "crowd_q_gold_top4",
    "crowd_q_top5",
    "crowd_q_gold_top9",
    "crowd_q_top10",
    "crowd_q_gold_top14",
    "crowd_q_top15",
)

EXPERIMENT_LABELS = {
    "q_only": "Q only",
    "crowd_q": "Crowd Prompt + Q",
    "crowd_q_gold": "Crowd Prompt + Q + Golden",
    "crowd_q_gold_top2": "Crowd Prompt + Q + Golden + Retrieval Top-2",
    "crowd_q_top3": "Crowd Prompt + Q + Retrieval Top-3",
    "crowd_q_gold_top4": "Crowd Prompt + Q + Golden + Retrieval Top-4",
    "crowd_q_top5": "Crowd Prompt + Q + Retrieval Top-5",
    "crowd_q_gold_top9": "Crowd Prompt + Q + Golden + Retrieval Top-9",
    "crowd_q_top10": "Crowd Prompt + Q + Retrieval Top-10",
    "crowd_q_gold_top14": "Crowd Prompt + Q + Golden + Retrieval Top-14",
    "crowd_q_top15": "Crowd Prompt + Q + Retrieval Top-15",
}

METRIC_NAMES = (
    "bleu-avg",
    "bleu-1",
    "bleu-2",
    "bleu-3",
    "bleu-4",
    "rouge-L",
    "precision",
    "recall",
    "f1",
    "length",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def load_jsonl_latest(path: Path, key: str = "case_id") -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return output
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            item_key = str(value.get(key, ""))
            if item_key:
                output[item_key] = value
    return output


def load_assets(assets_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cases_path = assets_dir / "evaluation_cases.jsonl"
    prompts_path = assets_dir / "crowd_prompts.json"
    centers_path = assets_dir / "cluster_centers.json"
    manifest_path = assets_dir / "manifest.json"
    for path in (cases_path, prompts_path, centers_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing asset: {path}")

    cases = list(load_jsonl_latest(cases_path).values())
    prompts = read_json(prompts_path)
    centers = read_json(centers_path)
    manifest = read_json(manifest_path)

    expected_hashes = manifest.get("asset_sha256", {})
    for name, path in (
        ("evaluation_cases.jsonl", cases_path),
        ("crowd_prompts.json", prompts_path),
        ("cluster_centers.json", centers_path),
    ):
        expected = str(expected_hashes.get(name, ""))
        actual = sha256_file(path)
        if expected and actual != expected:
            raise ValueError(f"Asset hash mismatch for {name}: expected {expected}, got {actual}")

    prompt_clusters = prompts.get("clusters", {})
    center_clusters = centers.get("clusters", [])
    if len(cases) != 412:
        raise ValueError(f"Expected 412 held-out cases, got {len(cases)}")
    if sorted(map(int, prompt_clusters)) != list(range(8)):
        raise ValueError("Crowd prompts must cover clusters 0..7")
    if sorted(int(item["cluster_id"]) for item in center_clusters) != list(range(8)):
        raise ValueError("Cluster centers must cover clusters 0..7")

    for case in cases:
        case_id = str(case.get("case_id", ""))
        query = str(case.get("query", "")).strip()
        reference = str(case.get("reference_answer", "")).strip()
        golden = case.get("golden_context", [])
        top3 = case.get("retrieved_top3", [])
        cluster_id = str(int(case.get("route", {}).get("cluster_id", -1)))
        if not case_id or not query or not reference:
            raise ValueError(f"Incomplete case: {case_id or '<missing id>'}")
        if not isinstance(golden, list) or not golden:
            raise ValueError(f"Missing Golden context for {case_id}")
        if not isinstance(top3, list) or len(top3) != 3 or any(not str(item.get("text", "")).strip() for item in top3):
            raise ValueError(f"Expected exactly three retrieval hits for {case_id}")
        if cluster_id not in prompt_clusters:
            raise ValueError(f"No prompt for {case_id}, cluster {cluster_id}")
    return cases, prompts, centers, manifest


def _normalize_match_text(text: Any, *, loose: bool = False) -> str:
    """Normalize text used to match evaluation cases to qa_results records."""
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = " ".join(value.split()).strip()
    if not loose:
        return value
    return "".join(
        char
        for char in value.casefold()
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "S"))
    )


def _iter_qa_result_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream records from the external qa_results JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def _record_index(records: Sequence[Mapping[str, Any]], key_builder: Any) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for index_number, record in enumerate(records):
        key = key_builder(record)
        if key:
            index.setdefault(key, []).append(index_number)
    return index


def apply_external_fusion_retrieval(
    cases: Sequence[Dict[str, Any]],
    qa_results_path: Path,
) -> Dict[str, Any]:
    """Replace only retrieval contexts with ``fusion_top5`` from qa_results.

    Matching deliberately does not use ``dataset_question_id`` or transformed
    numeric IDs.  It first matches normalized question/reference text.  The
    current export is missing one held-out question, so a final fallback accepts
    only a unique, high-overlap reference answer and records that fallback in
    the run metadata.
    """
    if not qa_results_path.is_file():
        raise FileNotFoundError(f"Missing external retrieval file: {qa_results_path}")

    records = list(_iter_qa_result_records(qa_results_path))
    if not records:
        raise ValueError(f"No records found in {qa_results_path}")

    def strict_question(value: Mapping[str, Any]) -> str:
        return _normalize_match_text(value.get("query", value.get("question", "")))

    def loose_question(value: Mapping[str, Any]) -> str:
        return _normalize_match_text(value.get("query", value.get("question", "")), loose=True)

    def strict_reference(value: Mapping[str, Any]) -> str:
        return _normalize_match_text(value.get("reference_answer", ""))

    def loose_reference(value: Mapping[str, Any]) -> str:
        return _normalize_match_text(value.get("reference_answer", ""), loose=True)

    record_by_pair = _record_index(
        records,
        lambda record: f"{strict_question(record)}\x00{strict_reference(record)}",
    )
    record_by_question = _record_index(records, strict_question)
    record_by_loose_pair = _record_index(
        records,
        lambda record: f"{loose_question(record)}\x00{loose_reference(record)}",
    )
    record_by_loose_question = _record_index(records, loose_question)
    record_by_reference = _record_index(records, strict_reference)
    record_by_loose_reference = _record_index(records, loose_reference)

    def unique_candidate(index: Dict[str, List[int]], key: str) -> Optional[int]:
        candidates = index.get(key, [])
        return candidates[0] if len(candidates) == 1 else None

    def convert_fusion(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        retrieval_top5 = record.get("retrieval_top5", {})
        fusion_top5 = retrieval_top5.get("fusion_top5", []) if isinstance(retrieval_top5, Mapping) else []
        if not isinstance(fusion_top5, list) or len(fusion_top5) < 3:
            raise ValueError(
                f"Question has fewer than 3 fusion_top5 hits: {record.get('question', '')}"
            )

        converted: List[Dict[str, Any]] = []
        for position, hit in enumerate(fusion_top5[:5], 1):
            if not isinstance(hit, Mapping):
                raise ValueError(f"Invalid fusion_top5 item for question: {record.get('question', '')}")
            text = str(hit.get("chunk_text", "")).strip()
            if not text:
                text = str(hit.get("chunk_text_full", hit.get("text", ""))).strip()
            if not text:
                raise ValueError(
                    f"Empty fusion context at rank {position} for question: {record.get('question', '')}"
                )
            try:
                score = float(hit.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            converted.append(
                {
                    "rank": position,
                    "index": position - 1,
                    "score": score,
                    "text": text,
                    "chunk_id": str(hit.get("chunk_id", "")),
                    "source_file": str(hit.get("source_file", "")),
                    "external_source": "qa_results.retrieval_top5.fusion_top5",
                }
            )
        return converted

    matched: Dict[str, Tuple[int, str, float]] = {}
    used_record_indices = set()

    # Strong matches are one-to-one and do not reuse an external record.
    for case in cases:
        case_id = str(case.get("case_id", ""))
        question = strict_question(case)
        question_loose = loose_question(case)
        reference = strict_reference(case)
        reference_loose = loose_reference(case)
        candidate_methods = (
            (unique_candidate(record_by_pair, f"{question}\x00{reference}"), "question+reference(strict)"),
            (unique_candidate(record_by_question, question), "question(strict)"),
            (unique_candidate(record_by_loose_pair, f"{question_loose}\x00{reference_loose}"), "question+reference(loose)"),
            (unique_candidate(record_by_loose_question, question_loose), "question(loose)"),
            (unique_candidate(record_by_reference, reference), "reference(strict)"),
            (unique_candidate(record_by_loose_reference, reference_loose), "reference(loose)"),
        )
        for record_index, method in candidate_methods:
            if record_index is not None and record_index not in used_record_indices:
                matched[case_id] = (record_index, method, 1.0)
                used_record_indices.add(record_index)
                break

    # One QA export record is missing for the current 412-case asset.  Use a
    # deterministic high-overlap answer fallback only when it is unambiguous.
    fallback_threshold = 0.80
    fallback_margin = 0.10
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if case_id in matched:
            continue
        target_reference = loose_reference(case)
        scored = sorted(
            (
                difflib.SequenceMatcher(
                    None,
                    target_reference,
                    loose_reference(record),
                ).ratio(),
                record_index,
            )
            for record_index, record in enumerate(records)
            if target_reference and loose_reference(record)
        )
        if not scored:
            continue
        best_score, best_index = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else 0.0
        if best_score < fallback_threshold or best_score - second_score < fallback_margin:
            continue
        matched[case_id] = (best_index, "reference_similarity_fallback", best_score)

    missing_cases = [case for case in cases if str(case.get("case_id", "")) not in matched]
    if missing_cases:
        examples = [
            {
                "case_id": str(case.get("case_id", "")),
                "original_id": str(case.get("original_id", "")),
                "query": str(case.get("query", "")),
                "reference_answer": str(case.get("reference_answer", ""))[:160],
            }
            for case in missing_cases[:10]
        ]
        raise ValueError(
            f"qa_results.jsonl could not match {len(missing_cases)} evaluation cases. "
            "Matching uses question/reference text, not dataset_question_id or transformed IDs. "
            f"Unmatched examples: {json.dumps(examples, ensure_ascii=False)}"
        )

    method_counts: Dict[str, int] = {}
    fallback_details: List[Dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        record_index, method, score = matched[case_id]
        external_top5 = convert_fusion(records[record_index])
        case["retrieved_top3_original"] = case.get("retrieved_top3", [])
        case["external_fusion_top5"] = external_top5
        case["retrieved_top3"] = external_top5[:3]
        case["external_retrieval_match_method"] = method
        case["external_retrieval_match_score"] = score
        method_counts[method] = method_counts.get(method, 0) + 1
        if method == "reference_similarity_fallback":
            fallback_details.append(
                {
                    "case_id": case_id,
                    "qa_question": str(records[record_index].get("question", "")),
                    "similarity": score,
                }
            )

    return {
        "matched_cases": len(matched),
        "qa_results_path": str(qa_results_path),
        "retrieval_field": "retrieval_top5.fusion_top5",
        "text_field": "chunk_text",
        "match_methods": method_counts,
        "fallback_details": fallback_details,
        "golden_unchanged": True,
        "golden_plus_top2": "fusion_top5[:2]",
        "top3": "fusion_top5[:3]",
    }


def parse_experiments(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(EXPERIMENTS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    supported = EXPERIMENTS + TOPK_EXPERIMENTS
    unknown = [item for item in requested if item not in supported]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; supported: {', '.join(supported)}")
    if not requested:
        raise ValueError("At least one experiment is required")
    return list(dict.fromkeys(requested))


def runtime_routes(
    cases: Sequence[Dict[str, Any]],
    centers: Dict[str, Any],
    *,
    reroute: bool,
    embedding_model: str,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    if not reroute:
        return {str(case["case_id"]): dict(case["route"]) for case in cases}
    if not embedding_model:
        raise ValueError("--reroute requires --embedding-model")
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Fresh BGE routing requires numpy, torch, and transformers. "
            "Install requirements-routing.txt or use the default saved routes."
        ) from exc

    print(f"Loading routing encoder: {embedding_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    queries = [str(case["query"]) for case in cases]
    embeddings = []
    for start in range(0, len(queries), max(1, batch_size)):
        batch = queries[start : start + max(1, batch_size)]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            vectors = model(**encoded).last_hidden_state[:, 0].cpu().numpy()
        embeddings.extend(vectors)

    cluster_items = sorted(centers["clusters"], key=lambda item: int(item["cluster_id"]))
    cluster_ids = [int(item["cluster_id"]) for item in cluster_items]
    cluster_vectors = [np.asarray(item["center"], dtype=np.float32) for item in cluster_items]
    output: Dict[str, Dict[str, Any]] = {}
    for case, vector in zip(cases, embeddings):
        similarities = [
            float(np.dot(vector, center) / ((np.linalg.norm(vector) * np.linalg.norm(center)) + 1e-8))
            for center in cluster_vectors
        ]
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        output[str(case["case_id"])] = {
            "method": "runtime_bge_cosine_top1",
            "embedding_model": embedding_model,
            "cluster_id": cluster_ids[best_index],
            "cluster_similarity": similarities[best_index],
        }
    return output


def golden_document(case: Mapping[str, Any]) -> str:
    return "\n\n".join(str(item).strip() for item in case.get("golden_context", []) if str(item).strip())


def context_documents(case: Mapping[str, Any], experiment: str) -> List[Dict[str, Any]]:
    top3 = list(case.get("retrieved_top3", []))
    if experiment in {"q_only", "crowd_q"}:
        return []
    if experiment == "crowd_q_gold":
        return [{"position": 1, "kind": "golden", "text": golden_document(case)}]
    if experiment == "crowd_q_gold_top2":
        documents = [{"position": 1, "kind": "golden", "text": golden_document(case)}]
        for position, hit in enumerate(top3[:2], 2):
            documents.append(
                {
                    "position": position,
                    "kind": "retrieved",
                    "retrieval_rank": int(hit["rank"]),
                    "retrieval_index": int(hit["index"]),
                    "retrieval_score": float(hit["score"]),
                    "text": str(hit["text"]),
                }
            )
        return documents
    if experiment == "crowd_q_top3":
        return [
            {
                "position": position,
                "kind": "retrieved",
                "retrieval_rank": int(hit["rank"]),
                "retrieval_index": int(hit["index"]),
                "retrieval_score": float(hit["score"]),
                "text": str(hit["text"]),
            }
            for position, hit in enumerate(top3, 1)
        ]
    raise ValueError(experiment)


def build_crowd_prompt(module: Mapping[str, Any], query: str, documents: Sequence[Mapping[str, Any]]) -> str:
    context_text = (
        "\n\n".join(f"[文档 {index}]\n{str(document['text'])}" for index, document in enumerate(documents, 1))
        if documents
        else "无相关上下文"
    )
    return f"""{module.get('P_sys', '你是一个专业的旅游咨询助手。')}

{module.get('I_t', '请回答用户的问题。')}

{module.get('C_t', '请参考提供的上下文信息。')}

{module.get('F_t', '请以JSON格式返回。')}

{module.get('U_t', '如果不确定，请说明。')}

---

**上下文信息**：
{context_text}

---

**用户问题**：{query}

请根据上下文信息回答上述问题。"""


def build_request(
    case: Mapping[str, Any],
    experiment: str,
    prompts: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    query = str(case["query"])
    documents = context_documents(case, experiment)
    if experiment == "q_only":
        return [{"role": "user", "content": query}], documents, None, query

    cluster_id = str(int(route["cluster_id"]))
    selected = prompts["clusters"][cluster_id]
    module = dict(selected["prompt_module"])
    full_prompt = build_crowd_prompt(module, query, documents)
    prompt_used = {
        "cluster_id": int(cluster_id),
        "cluster_similarity": float(route["cluster_similarity"]),
        "route_method": str(route["method"]),
        "embedding_model": str(route["embedding_model"]),
        "prompt_module": module,
        "based_questions_count": int(selected.get("based_questions_count", 0)),
    }
    # Keep the original chapter3 behavior: the complete modular prompt is one
    # user message.  P_sys is embedded in the prompt rather than sent separately.
    return [{"role": "user", "content": full_prompt}], documents, prompt_used, full_prompt


class VLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        max_retries: int,
        retry_wait: float,
        request_interval: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_wait = max(0.0, retry_wait)
        self.request_interval = max(0.0, request_interval)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _throttle(self) -> None:
        if self.request_interval <= 0:
            return
        with self._lock:
            wait = self.request_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def discover_model(self) -> str:
        request = urllib.request.Request(f"{self.base_url}/models", headers=self._headers(), method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = body.get("data") or []
        if not models or not models[0].get("id"):
            raise RuntimeError(f"No model returned by {self.base_url}/models: {body}")
        return str(models[0]["id"])

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            self._throttle()
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError(f"Empty choices: {body}")
                first = choices[0]
                content = (first.get("message") or {}).get("content")
                if isinstance(content, list):
                    content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
                text = str(content or "").strip()
                if not text:
                    raise RuntimeError(f"Empty assistant content: {body}")
                return text, {
                    "finish_reason": first.get("finish_reason"),
                    "usage": body.get("usage") or {},
                    "response_id": body.get("id"),
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_wait * (2**attempt))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", str(text), flags=re.IGNORECASE).strip()


def parse_generated_answer(raw: str) -> str:
    text = strip_thinking(raw)
    fenced = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text
    candidates = [candidate]
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        candidates.insert(0, candidate[start : end + 1])
    for value in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, dict) and str(parsed.get("answer", "")).strip():
                return str(parsed["answer"]).strip()
    return text


_METRIC_DEPENDENCIES: Any = None


def ensure_metric_dependencies() -> Tuple[Any, Any, Any, Any]:
    global _METRIC_DEPENDENCIES
    if _METRIC_DEPENDENCIES is not None:
        return _METRIC_DEPENDENCIES
    try:
        import jieba  # type: ignore
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
        from rouge_chinese import Rouge  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing metric dependency. Run: python -m pip install -r requirements.txt") from exc
    jieba.setLogLevel(logging.WARNING)
    _METRIC_DEPENDENCIES = (jieba, sentence_bleu, SmoothingFunction, Rouge())
    return _METRIC_DEPENDENCIES


def chinese_tokenize(text: str) -> List[str]:
    """Match chapter3's TouristQA tokenizer for BLEU and set-overlap metrics."""
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", str(text or ""))
    result: List[str] = []
    for token in (item for item in cleaned.split() if item):
        if re.match(r"[\u4e00-\u9fa5]", token):
            result.extend(list(token))
        else:
            result.append(token.lower())
    return result


def score_answer(generated: str, reference: str) -> Dict[str, Any]:
    jieba, sentence_bleu, SmoothingFunction, rouge = ensure_metric_dependencies()
    generated_tokens = chinese_tokenize(generated)
    reference_tokens = chinese_tokenize(reference)
    bleu_scores = [0.0, 0.0, 0.0, 0.0]
    if generated_tokens and reference_tokens:
        smoothing = SmoothingFunction().method1
        weights = (
            (1.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0),
            (0.33, 0.33, 0.34, 0.0),
            (0.25, 0.25, 0.25, 0.25),
        )
        bleu_scores = [
            float(sentence_bleu([reference_tokens], generated_tokens, weights=item, smoothing_function=smoothing))
            for item in weights
        ]
    bleu_1, bleu_2, bleu_3, bleu_4 = bleu_scores
    bleu_avg = sum(bleu_scores) / 4

    if generated_tokens and reference_tokens:
        generated_set, reference_set = set(generated_tokens), set(reference_tokens)
        overlap = generated_set & reference_set
        precision = len(overlap) / len(generated_set) if generated_set else 0.0
        recall = len(overlap) / len(reference_set) if reference_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        precision = recall = f1 = 0.0

    rouge_l = rouge_l_precision = rouge_l_recall = 0.0
    if str(generated).strip() and str(reference).strip():
        try:
            generated_rouge = " ".join(jieba.cut(str(generated)))
            reference_rouge = " ".join(jieba.cut(str(reference)))
            rouge_score = rouge.get_scores(generated_rouge, reference_rouge)[0]["rouge-l"]
            rouge_l = float(rouge_score["f"])
            rouge_l_precision = float(rouge_score["p"])
            rouge_l_recall = float(rouge_score["r"])
        except Exception:
            pass
    return {
        "bleu-avg": bleu_avg,
        "bleu-1": bleu_1,
        "bleu-2": bleu_2,
        "bleu-3": bleu_3,
        "bleu-4": bleu_4,
        "rouge-L": rouge_l,
        "rouge-L-precision": rouge_l_precision,
        "rouge-L-recall": rouge_l_recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "length": len(generated),
        "valid": bool(generated.strip()),
    }


def run_one(
    *,
    case: Dict[str, Any],
    experiment: str,
    prompts: Dict[str, Any],
    route: Dict[str, Any],
    client: VLLMClient,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    request_builder: Callable[..., Tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[Dict[str, Any]], str]] = build_request,
) -> Dict[str, Any]:
    started = time.monotonic()
    messages, documents, prompt_used, full_prompt = request_builder(case, experiment, prompts, route)
    base = {
        "case_id": case["case_id"],
        "original_id": case["original_id"],
        "dataset_question_id": case.get("dataset_question_id", ""),
        "attraction": case.get("attraction", ""),
        "experiment": experiment,
        "experiment_label": EXPERIMENT_LABELS[experiment],
        "query": case["query"],
        "reference_answer": case["reference_answer"],
        "context_documents": documents,
        "context_document_count": len(documents),
        "prompt_used": prompt_used,
        "full_prompt": full_prompt,
        "messages": messages,
        "created_at": now_iso(),
    }
    try:
        raw, response_meta = client.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        answer = parse_generated_answer(raw)
        base.update(
            {
                "status": "ok",
                "raw_model_output": raw,
                "generated_answer": answer,
                "metrics": score_answer(answer, str(case["reference_answer"])),
                "response_meta": response_meta,
            }
        )
    except Exception as exc:
        base.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    base["elapsed_seconds"] = time.monotonic() - started
    return base


def run_experiment(
    *,
    experiment: str,
    cases: Sequence[Dict[str, Any]],
    prompts: Dict[str, Any],
    routes: Dict[str, Dict[str, Any]],
    output_dir: Path,
    client: VLLMClient,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    workers: int,
    request_builder: Callable[..., Tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[Dict[str, Any]], str]] = build_request,
) -> None:
    result_path = output_dir / f"{experiment}.jsonl"
    existing = load_jsonl_latest(result_path)
    pending = [case for case in cases if existing.get(str(case["case_id"]), {}).get("status") != "ok"]
    print(
        f"[{experiment}] completed={len(cases) - len(pending)}/{len(cases)}, pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    successful = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                run_one,
                case=case,
                experiment=experiment,
                prompts=prompts,
                route=routes[str(case["case_id"])],
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                request_builder=request_builder,
            ): case
            for case in pending
        }
        for completed_index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            append_jsonl(result_path, record)
            if record["status"] == "ok":
                successful += 1
            else:
                failed += 1
            print(
                f"[{experiment}] {completed_index}/{len(pending)} "
                f"ok={successful} error={failed} case={record['case_id']}",
                flush=True,
            )


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(differences: Sequence[float], draws: int, seed: int) -> Tuple[float, float, float]:
    if not differences:
        return 0.0, 0.0, 0.0
    generator = random.Random(seed)
    count = len(differences)
    means = [sum(differences[generator.randrange(count)] for _ in range(count)) / count for _ in range(draws)]
    return sum(differences) / count, percentile(means, 0.025), percentile(means, 0.975)


def sign_test_pvalue(differences: Sequence[float]) -> float:
    nonzero = [value for value in differences if value != 0]
    count = len(nonzero)
    if not count:
        return 1.0
    smaller = min(sum(value > 0 for value in nonzero), sum(value < 0 for value in nonzero))
    tail = sum(math.comb(count, index) for index in range(smaller + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def load_judges(path: Path) -> Dict[str, Dict[str, Any]]:
    return load_jsonl_latest(path, key="judge_key")


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Judge response is not a JSON object")
    return value


LLM_JUDGE_DIMENSIONS: Tuple[str, ...] = ("answer_quality", "accuracy", "completeness")


def judge_context_text(result: Mapping[str, Any]) -> str:
    documents = result.get("context_documents", [])
    if not isinstance(documents, list) or not documents:
        return "无提供上下文（该实验条件未向回答模型提供上下文）。"
    parts: List[str] = []
    for index, document in enumerate(documents, 1):
        if isinstance(document, Mapping):
            text = str(document.get("text", "")).strip()
            kind = str(document.get("kind", "context")).strip() or "context"
        else:
            text = str(document).strip()
            kind = "context"
        if text:
            parts.append(f"[上下文 {index} | {kind}]\n{text}")
    return "\n\n".join(parts) if parts else "无提供上下文（上下文内容为空）。"


def build_llm_judge_prompt(result: Mapping[str, Any]) -> str:
    return f'''你是一名严格、稳定的问答评测器。请根据“用户问题、参考答案、候选答案、回答时可用的上下文”对候选答案分别评分。

评分必须彼此独立，均使用 1-5 分整数：

1. 答案质量（Answer Quality, AQ）
   综合评价答案的整体质量，包括语言表达是否清晰、连贯、自然、相关，以及答案在准确性和完整性层面的总体表现。
   5分：表达清晰专业，紧扣问题，整体准确且完整；
   4分：整体很好，仅有轻微表达、准确性或覆盖问题；
   3分：基本可用，但存在明显表达问题、部分不准确或部分缺失；
   2分：质量较差，存在较多错误、遗漏或表达混乱；
   1分：基本不可用、严重偏题、错误或难以理解。

2. 准确性（Accuracy）
   衡量答案中的事实、数据及关键信息是否正确，是否与参考答案及上下文证据一致，是否包含错误、矛盾或无依据的信息。
   5分：关键信息全部正确，无实质性错误；
   4分：核心正确，仅有轻微、不影响结论的小问题；
   3分：核心大体正确，但有若干明显错误或不严谨之处；
   2分：存在重要事实错误，影响主要结论；
   1分：大部分错误、与参考/证据明显冲突或完全无法支持。

3. 完整性（Completeness）
   衡量答案是否覆盖问题要求的各个方面，是否包含参考答案中的关键内容，以及是否存在重要信息遗漏。
   5分：关键内容和问题要求均完整覆盖；
   4分：覆盖主要内容，仅遗漏次要信息；
   3分：覆盖部分关键内容，但存在明显遗漏；
   2分：遗漏多个重要方面，只回答了问题的一小部分；
   1分：几乎没有覆盖问题所需的关键内容。

判分规则：
- 参考答案是主要对照标准；上下文用于核验候选答案中的事实依据。
- 不因为候选答案比参考答案措辞不同而扣分，只看语义、事实与覆盖程度。
- 若上下文与参考答案存在冲突，优先指出冲突并依据可验证证据谨慎评分。
- 不要给出总分，不要把三个维度合并成一个分数。
- 只输出合法 JSON，不要输出 Markdown，不要输出思考过程。

输出格式必须严格为：
{{
  "answer_quality": {{"score": 1, "reasoning": "简要理由"}},
  "accuracy": {{"score": 1, "reasoning": "简要理由"}},
  "completeness": {{"score": 1, "reasoning": "简要理由"}}
}}

用户问题：
{str(result.get('query', ''))}

参考答案：
{str(result.get('reference_answer', ''))}

候选答案：
{str(result.get('generated_answer', ''))}

回答时可用的上下文：
{judge_context_text(result)}'''


def parse_llm_judge_scores(parsed: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for dimension in LLM_JUDGE_DIMENSIONS:
        value = parsed.get(dimension)
        if not isinstance(value, Mapping):
            raise ValueError(f"Judge response missing object: {dimension}")
        raw_score = value.get("score")
        score_float = float(raw_score)
        if not math.isfinite(score_float):
            raise ValueError(f"Non-finite judge score for {dimension}: {raw_score!r}")
        score = int(round(score_float))
        score = max(1, min(5, score))
        output[dimension] = {
            "score": score,
            "reasoning": str(value.get("reasoning", "")).strip(),
        }
    return output


def judge_record_complete(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "ok":
        return False
    scores = record.get("scores")
    if not isinstance(scores, Mapping):
        return False
    return all(
        isinstance(scores.get(dimension), Mapping)
        and scores[dimension].get("score") is not None
        for dimension in LLM_JUDGE_DIMENSIONS
    )


def run_llm_judge(
    *,
    groups: Sequence[str],
    output_dir: Path,
    client: VLLMClient,
    model: str,
    seed: int,
    group_field: str = "experiment",
    max_tokens: int = 768,
    workers: int = 1,
) -> None:
    '''Run one LLM call per answer and return AQ/Accuracy/Completeness together.

    Results are stored separately in llm_judge.jsonl so answer-generation results
    remain reusable. Existing legacy one-score judge records are automatically
    re-evaluated because they do not satisfy judge_record_complete().
    '''
    judge_path = output_dir / "llm_judge.jsonl"
    completed = load_judges(judge_path)
    pending: List[Tuple[str, str, Dict[str, Any]]] = []
    for group in groups:
        results = load_jsonl_latest(output_dir / f"{group}.jsonl")
        for case_id, result in results.items():
            if result.get("status") != "ok":
                continue
            judge_key = f"{group}:{case_id}"
            if judge_record_complete(completed.get(judge_key, {})):
                continue
            pending.append((group, case_id, result))

    if not pending:
        return

    def judge_one(group: str, case_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        judge_key = f"{group}:{case_id}"
        record: Dict[str, Any] = {
            "judge_key": judge_key,
            group_field: group,
            "case_id": case_id,
            "judge_model": model,
            "created_at": now_iso(),
        }
        try:
            raw, response_meta = client.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严格的问答质量评测器。只输出符合用户指定结构的合法JSON，不输出思考过程。",
                    },
                    {"role": "user", "content": build_llm_judge_prompt(result)},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                seed=seed,
            )
            parsed = parse_json_object(raw)
            scores = parse_llm_judge_scores(parsed)
            record.update(
                {
                    "status": "ok",
                    "scores": scores,
                    "raw": raw,
                    "response_meta": response_meta,
                }
            )
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(judge_one, group, case_id, result): (group, case_id)
            for group, case_id, result in pending
        }
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(judge_path, record)
            completed[str(record["judge_key"])] = record
            if record.get("status") == "ok":
                score_text = ", ".join(
                    f"{dimension}={record['scores'][dimension]['score']}"
                    for dimension in LLM_JUDGE_DIMENSIONS
                )
                print(f"[judge] {record['judge_key']}: ok ({score_text})", flush=True)
            else:
                print(f"[judge] {record['judge_key']}: error", flush=True)

def summarize_results(
    cases: Sequence[Dict[str, Any]],
    result_map: Mapping[str, Dict[str, Any]],
    judges: Mapping[str, Dict[str, Any]],
    experiment: str,
) -> Dict[str, Any]:
    ok = [result_map[str(case["case_id"])] for case in cases if result_map.get(str(case["case_id"]), {}).get("status") == "ok"]
    errors = [result_map[str(case["case_id"])] for case in cases if result_map.get(str(case["case_id"]), {}).get("status") == "error"]
    means = {
        metric: (sum(float(item.get("metrics", {}).get(metric, 0.0)) for item in ok) / len(ok) if ok else 0.0)
        for metric in METRIC_NAMES
    }

    judge_records = [
        judges[f"{experiment}:{item['case_id']}"]
        for item in ok
        if judge_record_complete(judges.get(f"{experiment}:{item['case_id']}", {}))
    ]
    judge_summary: Dict[str, Any] = {"completed": len(judge_records)}
    for dimension in LLM_JUDGE_DIMENSIONS:
        values = [float(record["scores"][dimension]["score"]) for record in judge_records]
        judge_summary[dimension] = {
            "mean_score_1_to_5": sum(values) / len(values) if values else None,
        }

    return {
        "label": EXPERIMENT_LABELS[experiment],
        "expected": len(cases),
        "completed": len(ok),
        "errors_latest": len(errors),
        "missing": len(cases) - len(ok),
        "metrics": means,
        "llm_judge": judge_summary,
    }

def pairwise_comparison(
    *,
    cases: Sequence[Dict[str, Any]],
    left_name: str,
    right_name: str,
    left: Mapping[str, Dict[str, Any]],
    right: Mapping[str, Dict[str, Any]],
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    paired_ids = [
        str(case["case_id"])
        for case in cases
        if left.get(str(case["case_id"]), {}).get("status") == "ok"
        and right.get(str(case["case_id"]), {}).get("status") == "ok"
    ]
    output: Dict[str, Any] = {
        "direction": f"{right_name} minus {left_name}",
        "n_paired": len(paired_ids),
        "metrics": {},
    }
    for metric_index, metric in enumerate(("bleu-avg", "rouge-L", "f1")):
        differences = [
            float(right[case_id]["metrics"].get(metric, 0.0))
            - float(left[case_id]["metrics"].get(metric, 0.0))
            for case_id in paired_ids
        ]
        mean, low, high = paired_bootstrap(differences, draws, seed + metric_index)
        output["metrics"][metric] = {
            "mean_difference": mean,
            "bootstrap_ci95": [low, high],
            "sign_test_p": sign_test_pvalue(differences),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
    return output


def build_report(
    *,
    cases: Sequence[Dict[str, Any]],
    output_dir: Path,
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    draws: int,
) -> Dict[str, Any]:
    result_maps = {
        experiment: load_jsonl_latest(output_dir / f"{experiment}.jsonl")
        for experiment in EXPERIMENTS
        if (output_dir / f"{experiment}.jsonl").exists()
    }
    judges = load_judges(output_dir / "llm_judge.jsonl")
    summaries = {
        experiment: summarize_results(cases, result_maps.get(experiment, {}), judges, experiment)
        for experiment in EXPERIMENTS
    }
    comparisons = {}
    for pair_index, (left_name, right_name) in enumerate(itertools.combinations(EXPERIMENTS, 2)):
        if left_name not in result_maps or right_name not in result_maps:
            continue
        comparisons[f"{right_name}_minus_{left_name}"] = pairwise_comparison(
            cases=cases,
            left_name=left_name,
            right_name=right_name,
            left=result_maps[left_name],
            right=result_maps[right_name],
            draws=draws,
            seed=int(config.get("seed", 42)) + pair_index * 10,
        )
    return {
        "generated_at": now_iso(),
        "dataset": {
            "held_out_cases": len(cases),
            "prompt_optimization_cases": manifest.get("split", {}).get("prompt_optimization_questions", 24),
            "asset_sha256": manifest.get("asset_sha256", {}),
        },
        "configuration": config,
        "metric_definition": {
            "tokenizer": "chapter3 TouristQA character/alphanumeric tokenizer for BLEU and P/R/F1; jieba for ROUGE-L",
            "bleu": "NLTK cumulative BLEU-1..4 with method1 smoothing and brevity penalty; BLEU-avg is their arithmetic mean",
            "rouge-L": "rouge-chinese ROUGE-L F1 over jieba-tokenized text",
            "precision_recall_f1": "set overlap over chapter3 TouristQA character/alphanumeric tokens",
            "length": "generated-answer character count",
            "llm_judge": "optional LLM-as-a-Judge: Answer Quality, Accuracy, and Completeness, each scored independently from 1 to 5 using question, reference answer, candidate answer, and available context",
        },
        "experiments": summaries,
        "pairwise_comparisons": comparisons,
    }


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TouristQA 实验报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 汇总",
        "",
        "| 实验 | 完成/总数 | BLEU-avg | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Precision | Recall | F1 | AQ | Accuracy | Completeness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        summary = report["experiments"][experiment]
        metrics = summary["metrics"]
        aq = summary["llm_judge"]["answer_quality"]["mean_score_1_to_5"]
        accuracy = summary["llm_judge"]["accuracy"]["mean_score_1_to_5"]
        completeness = summary["llm_judge"]["completeness"]["mean_score_1_to_5"]
        aq_text = "-" if aq is None else f"{aq:.4f}"
        accuracy_text = "-" if accuracy is None else f"{accuracy:.4f}"
        completeness_text = "-" if completeness is None else f"{completeness:.4f}"
        lines.append(
            f"| {summary['label']} | {summary['completed']}/{summary['expected']} | "
            f"{metrics['bleu-avg']:.6f} | {metrics['bleu-1']:.6f} | {metrics['bleu-2']:.6f} | "
            f"{metrics['bleu-3']:.6f} | {metrics['bleu-4']:.6f} | {metrics['rouge-L']:.6f} | "
            f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | "
            f"{aq_text} | {accuracy_text} | {completeness_text} |"
        )
    lines.extend(
        [
            "",
            "## 条件说明",
            "",
            "- `Q only` 严格只发送一条原始问题消息。",
            "- `Crowd Prompt + Q` 使用BGE路由的群智提示，但上下文为“无相关上下文”。",
            "- Golden由数据集完整 `source` 合并为一个逻辑文档。",
            "- Golden + Top-2 一共三个上下文文档，不对Golden和检索结果去重。",
            "- Top-3使用 `qa_results.jsonl` 的 `fusion_top5` 前三条，不含Golden；Golden + Top-2使用前两条。",
            "- BLEU、ROUGE和词重叠指标是词面指标，应结合逐题结果与误差分析解释。",
            "",
            "## 配对比较",
            "",
        ]
    )
    for name, comparison in report.get("pairwise_comparisons", {}).items():
        lines.append(f"### {comparison['direction']}")
        lines.append("")
        lines.append(f"配对题数：{comparison['n_paired']}")
        lines.append("")
        lines.append("| 指标 | 均值差 | 95% Bootstrap CI | Sign test p | 胜/平/负 |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric, value in comparison["metrics"].items():
            ci = value["bootstrap_ci95"]
            lines.append(
                f"| {metric} | {value['mean_difference']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | "
                f"{value['sign_test_p']:.6g} | {value['wins']}/{value['ties']}/{value['losses']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=script_dir / "assets")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results" / "qwen3_8b")
    parser.add_argument("--experiments", default="all", help="all or comma-separated experiment ids")
    parser.add_argument(
        "--qa-results",
        type=Path,
        default=script_dir / "qa_results.jsonl",
        help="external retrieval results; uses retrieval_top5.fusion_top5",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 412 cases")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--llm-judge", action="store_true", help="run resumable LLM-as-a-Judge for AQ/Accuracy/Completeness")
    parser.add_argument("--judge-max-tokens", type=int, default=768, help="max tokens for one three-dimension judge response")
    parser.add_argument("--judge-workers", type=int, default=0, help="parallel LLM Judge workers; 0 means use --workers")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reroute", action="store_true", help="recompute BGE-to-cluster routes instead of saved routes")
    parser.add_argument("--embedding-model", default="", help="local path or model id for --reroute")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiments = parse_experiments(args.experiments)
    if args.limit < 0 or args.max_tokens < 1 or args.judge_max_tokens < 1 or args.workers < 1 or args.judge_workers < 0 or args.bootstrap_draws < 1:
        raise ValueError("limit must be >=0; max-tokens, judge-max-tokens, workers, and bootstrap-draws must be positive; judge-workers must be >=0")
    judge_workers = args.judge_workers or args.workers

    assets_dir = args.assets_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    qa_results_path = args.qa_results.expanduser().resolve()
    cases, prompts, centers, manifest = load_assets(assets_dir)
    if args.limit:
        cases = cases[: args.limit]
    retrieval_override = apply_external_fusion_retrieval(cases, qa_results_path)
    print(
        f"External retrieval loaded: {retrieval_override['matched_cases']} cases from {qa_results_path}; "
        "Golden unchanged; Golden+Top2=fusion[:2], Top3=fusion[:3]",
        flush=True,
    )
    if retrieval_override["fallback_details"]:
        print(
            "Retrieval matching fallback used for: "
            + json.dumps(retrieval_override["fallback_details"], ensure_ascii=False),
            flush=True,
        )
    routes = runtime_routes(
        cases,
        centers,
        reroute=args.reroute,
        embedding_model=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )

    print(f"Asset validation passed: {len(cases)} cases, 8 crowd-prompt clusters", flush=True)
    if args.validate_only:
        for experiment in EXPERIMENTS:
            messages, documents, prompt_used, _ = build_request(cases[0], experiment, prompts, routes[str(cases[0]["case_id"])])
            print(
                f"{experiment}: messages={len(messages)}, context_documents={len(documents)}, "
                f"crowd_prompt={'yes' if prompt_used else 'no'}",
                flush=True,
            )
        return 0

    # Generation records include deterministic metrics, so fail before sending
    # requests if the tokenizer dependency is absent. Asset-only validation does
    # not need third-party packages.
    ensure_metric_dependencies()

    client: Optional[VLLMClient] = None
    model = args.model
    if not args.evaluate_only or args.llm_judge:
        client = VLLMClient(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            request_interval=args.request_interval,
        )
        if model.lower() == "auto":
            model = client.discover_model()

    config_path = output_dir / "run_config.json"
    existing_config = read_json(config_path) if config_path.exists() else {}
    if args.evaluate_only and model.lower() == "auto" and existing_config.get("model"):
        model = str(existing_config["model"])
    config = {
        "created_at": existing_config.get("created_at", now_iso()),
        "updated_at": now_iso(),
        "assets_dir": str(assets_dir),
        "qa_results": str(qa_results_path),
        "external_retrieval": retrieval_override,
        "asset_sha256": manifest.get("asset_sha256", {}),
        "evaluation_cases": len(cases),
        "base_url": args.base_url.rstrip("/"),
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "llm_judge_enabled": bool(args.llm_judge),
        "judge_max_tokens": args.judge_max_tokens,
        "seed": args.seed,
        "enable_thinking": False,
        "workers": args.workers,
        "route_mode": "runtime_bge" if args.reroute else "saved_bge_routes",
        "embedding_model": args.embedding_model if args.reroute else "BAAI/bge-large-zh-v1.5 (saved routes)",
        "experiments_requested": experiments,
        "retrieval": "qa_results.jsonl retrieval_top5.fusion_top5; Golden unchanged; Golden+Top-2 uses first 2 fusion hits; Top-3 uses first 3 fusion hits",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)

    if not args.evaluate_only:
        assert client is not None
        for experiment in experiments:
            run_experiment(
                experiment=experiment,
                cases=cases,
                prompts=prompts,
                routes=routes,
                output_dir=output_dir,
                client=client,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                workers=args.workers,
            )

    if args.llm_judge:
        assert client is not None
        run_llm_judge(
            groups=experiments,
            output_dir=output_dir,
            client=client,
            model=model,
            seed=args.seed,
            group_field="experiment",
            max_tokens=args.judge_max_tokens,
            workers=judge_workers,
        )

    report = build_report(
        cases=cases,
        output_dir=output_dir,
        manifest=manifest,
        config=config,
        draws=args.bootstrap_draws,
    )
    write_json(output_dir / "report.json", report)
    write_text(output_dir / "report.md", report_markdown(report))
    print(f"JSON report: {output_dir / 'report.json'}", flush=True)
    print(f"Markdown report: {output_dir / 'report.md'}", flush=True)

    incomplete = [
        experiment
        for experiment in experiments
        if report["experiments"][experiment]["completed"] < len(cases)
    ]
    return 2 if incomplete and not args.evaluate_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
