"""QA 聚类、聚类级 Prompt 优化和新查询路由。

这个模块把原项目中的 Step 2--Step 5 收敛成一个可复用的实现：

1. 对每条 QA 样本执行案例级 Prompt 迭代；
2. 用问题和答案的向量对 QA 样本做确定性 KMeans 聚类；
3. 汇总同一聚类中的多条迭代记录，由 LLM 生成一个聚类级 Prompt；
4. 新问题与聚类中心做余弦匹配，并将对应 Prompt 交给问答流程。

这里的“群智优化”表示聚类内多条 QA 经验的集体归纳，不依赖 sklearn，
也不引入多份重复的 LLM 客户端。所有产物均写入当前 run-dir。
"""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompting import PromptOptimizer
from .prompt_policy import POLICY_TEXT, META_PROMPT_PATH, request_module, validate_module


CLUSTER_PROMPT_VERSION = "1.0"
ARTIFACT_NAME = "cluster_prompting.json"


def _first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def _text_list(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if isinstance(item, Mapping):
            item = _first(item, ("text", "content", "evidence_text", "source"), "")
        item = str(item or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def normalize_examples(raw_examples: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """统一 prompt 样本和评测样本字段，保留原始字段便于审计。"""
    examples = []
    for index, raw in enumerate(raw_examples, 1):
        if not isinstance(raw, Mapping):
            continue
        question = str(_first(raw, ("question", "query", "prompt"), "") or "").strip()
        if not question:
            continue
        # Golden-only offline optimization. Never silently substitute retrieval.
        context = str(_first(raw, ("gold_context", "golden_context"), "") or "").strip()
        if not context:
            context = "\n".join(_text_list(_first(raw, ("gold_evidence_texts", "source"), [])))
        if not context and isinstance(raw.get("gold"), Mapping):
            context = "\n".join(_text_list(raw["gold"].get("evidence_texts", [])))
        if not context and raw.get("context_kind") == "golden":
            context = str(raw.get("context", "")).strip()
        reference = str(_first(
            raw, ("reference_answer", "answer", "standard_answer", "gold_answer"), ""
        ) or "").strip()
        sample_id = str(_first(raw, ("id", "question_id", "qid"), f"qa_{index:04d}"))
        examples.append({
            **dict(raw),
            "id": sample_id,
            "question": question,
            "context": context,
            "context_kind": "golden" if context else "missing_golden",
            "reference_answer": reference,
        })
    return examples


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = _unit(left)
    right_array = _unit(right)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    size = min(left_array.size, right_array.size)
    return float(np.dot(left_array[:size], right_array[:size]))


def _kmeans(vectors: Sequence[Sequence[float]], cluster_count: int) -> tuple[np.ndarray, np.ndarray]:
    """小型确定性 KMeans，使用余弦等价的归一化欧氏距离。"""
    data = np.asarray([_unit(vector) for vector in vectors], dtype=float)
    if data.ndim != 2 or not len(data):
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    count = min(max(1, int(cluster_count)), len(data))
    indices = np.linspace(0, len(data) - 1, count, dtype=int)
    centers = data[indices].copy()

    for _ in range(60):
        distances = 1.0 - np.clip(data @ centers.T, -1.0, 1.0)
        labels = np.argmin(distances, axis=1)
        updated = np.zeros_like(centers)
        for cluster_id in range(count):
            members = data[labels == cluster_id]
            if len(members):
                updated[cluster_id] = _unit(np.mean(members, axis=0))
            else:
                # 空簇取当前最不相似的样本，保证每个输出簇都有定义。
                farthest = int(np.argmax(np.min(1.0 - np.clip(data @ centers.T, -1.0, 1.0), axis=1)))
                updated[cluster_id] = data[farthest]
        if np.allclose(updated, centers, atol=1e-7):
            centers = updated
            break
        centers = updated
    labels = np.argmax(data @ centers.T, axis=1)
    return centers, labels


def _silhouette(vectors: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    data = np.asarray([_unit(vector) for vector in vectors], dtype=float)
    labels = np.asarray(labels, dtype=int)
    unique = sorted(set(labels.tolist())) if len(labels) else []
    if len(data) < 2 or len(unique) < 2:
        return 0.0
    distances = 1.0 - np.clip(data @ data.T, -1.0, 1.0)
    scores = []
    for index, label in enumerate(labels):
        own = labels == label
        own[index] = False
        a = float(np.mean(distances[index, own])) if np.any(own) else 0.0
        other_means = [float(np.mean(distances[index, labels == other]))
                       for other in unique if other != label and np.any(labels == other)]
        b = min(other_means) if other_means else 0.0
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def _safe_module(value: Any, evidence_policy=True) -> Dict[str, str]:
    return PromptOptimizer.normalize_module(value if isinstance(value, Mapping) else None, evidence_policy=evidence_policy)


class ClusterPromptPipeline:
    """把 QA 迭代、聚类和群智 Prompt 生成串成一个可审计流水线。"""

    def __init__(self, *, client: LLMClient, embeddings: EmbeddingModel,
                 run_dir: str | Path, evidence_policy=True, meta_prompt_path=None):
        self.client = client
        self.embeddings = embeddings
        self.run_dir = Path(run_dir)
        self.evidence_policy = bool(evidence_policy)
        self.meta_prompt_path = Path(meta_prompt_path) if meta_prompt_path else META_PROMPT_PATH
        self.case_dir = self.run_dir / "cluster_prompting" / "case_iterations"
        self.prompt_dir = self.run_dir / "cluster_prompting" / "cluster_prompts"

    def _case_iteration(self, example: Dict[str, Any], iterations: int,
                        min_gain: float = 1.0, initial_module=None) -> Dict[str, Any]:
        """复用统一 PromptOptimizer，避免从旧项目复制第二套 LLM 逻辑。"""
        case_id = str(example["id"])
        output_path = self.case_dir / f"{case_id}.json"
        if output_path.resolve().parent != self.case_dir.resolve():
            raise ValueError("Unsafe case id")
        signature = hashlib.sha256(json.dumps({"example": example, "max_rounds": iterations,
            "min_gain": min_gain, "initial_guidance": initial_module,
            "meta_prompt": self.meta_prompt_path.read_text(encoding="utf-8"),
            "evidence_policy": self.evidence_policy,
            "model": getattr(self.client, "model", ""), "seed": getattr(self.client, "seed", None),
            "base_url": getattr(self.client, "base_url", ""), "mock": self.client.mock},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        try:
            optimizer = PromptOptimizer(client=self.client, run_dir=self.case_dir,
                evidence_policy=self.evidence_policy, meta_prompt_path=self.meta_prompt_path)
            result = None
            if output_path.exists():
                previous = json.loads(output_path.read_text(encoding="utf-8"))
                if previous.get("case_signature") == signature and previous.get("stop_reason") not in {None, "in_progress"}:
                    validate_module(previous, evidence_policy=self.evidence_policy)
                    result = previous
                elif previous.get("case_signature") not in {None, signature}:
                    raise ValueError("Case input/config changed; use a new run directory")
            if result is None:
                result = optimizer.optimize(
                    [example], iterations=int(iterations), min_gain=min_gain,
                    initial_module=initial_module, output_path=output_path,
                )
                result["case_signature"] = signature
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "id": case_id,
                "question": example["question"],
                "answer": example.get("reference_answer", ""),
                "context": example.get("context", ""),
                "final_prompt": _safe_module(result, evidence_policy=self.evidence_policy),
                "history": result.get("history", []),
                "generated_answer": self._last_generated_answer([{"answers": result.get("selected_answers", [])}]),
                "initial_prompt": result.get("initial_prompt", {}),
                "stop_reason": result.get("stop_reason", ""),
                "selected_iteration": result.get("selected_iteration"),
                "selected_scores": result.get("selected_scores", {}),
            }
        except Exception as exc:
            if not self.client.mock:
                raise RuntimeError(f"Case {case_id} failed; no silent default prompt accepted: {exc}") from exc
            # 单个样本失败不阻断其余样本；失败信息会进入审计文件。
            fallback = {
                "id": case_id,
                "question": example["question"],
                "answer": example.get("reference_answer", ""),
                "context": example.get("context", ""),
                "final_prompt": PromptOptimizer.default_module(self.evidence_policy),
                "history": [],
                "generated_answer": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.case_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8")
            return fallback

    @staticmethod
    def _last_generated_answer(history: Any) -> str:
        if not isinstance(history, list):
            return ""
        for round_result in reversed(history):
            if not isinstance(round_result, Mapping):
                continue
            answers = round_result.get("answers", [])
            if isinstance(answers, list) and answers:
                item = answers[-1]
                if isinstance(item, Mapping) and item.get("answer"):
                    return str(item["answer"])
        return ""

    def _cluster_prompt(self, cluster_id: int, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        fallback = PromptOptimizer.default_module(self.evidence_policy)
        if self.client.mock or not self.client.api_key:
            return {"cluster_id": cluster_id, "prompt": fallback,
                    "source_case_count": len(cases), "method": "fallback"}

        summaries = []
        for case in cases:
            summaries.append({
                "question": case.get("question", ""),
                "reference_answer": case.get("answer", ""),
                "final_prompt": case.get("final_prompt", {}),
                "iteration_history": [
                    {key: item.get(key) for key in ("iteration", "dimensions", "total", "feedback", "accepted", "reason")}
                    for item in case.get("history", [])
                ],
            })
        user = (
            "请根据同一问题簇中的多条 QA 迭代记录，生成这个问题簇通用的问答 Prompt。\n"
            "要求：只依据检索上下文；完整回答问题中的多个要点；保留实体、时间、数量；"
            "证据不足时明确说明，不得编造；只输出 JSON，字段必须是："
            "system_prompt、instruction、context_strategy、format_requirement、uncertainty_handling。\n"
            f"问题簇编号：{cluster_id}\n"
            "不得将具体案例答案硬编码到通用提示中。\n"
            + (f"context_strategy 必须逐字保留以下两条约束：\n{POLICY_TEXT}\n" if self.evidence_policy else "")
            +
            f"QA 迭代记录：{json.dumps(summaries, ensure_ascii=False)}"
        )
        try:
            result = request_module(self.client,
                "你是聚类级 Prompt 群智优化专家，只输出合法 JSON。",
                user,
                max_tokens=2400,
                evidence_policy=self.evidence_policy,
            )
            module = _safe_module(result, evidence_policy=self.evidence_policy)
            method = "llm_collective_synthesis"
        except Exception as exc:
            if not self.client.mock:
                raise RuntimeError(f"Cluster {cluster_id} synthesis failed: {exc}") from exc
            module = fallback
            method = f"fallback:{type(exc).__name__}"
        return {"cluster_id": cluster_id, "prompt": module,
                "source_case_count": len(cases), "method": method}

    def run(self, examples: Iterable[Mapping[str, Any]], *, cluster_count: int = 6,
            iterations: int = 30, min_gain: float = 1.0, initial_module=None, workers: int = 1) -> Dict[str, Any]:
        examples = normalize_examples(examples)
        if not examples:
            raise ValueError("聚类 Prompt 流程至少需要一条有效 QA 样本")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        if not self.client.mock and any(example.get("context_kind") != "golden" for example in examples):
            raise ValueError("All training examples must contain explicit golden evidence")
        if workers < 1:
            raise ValueError("workers must be positive")
        cases = [None] * len(examples)
        with ThreadPoolExecutor(max_workers=min(workers, len(examples))) as executor:
            futures = {executor.submit(self._case_iteration, example, iterations, min_gain, initial_module): index
                       for index, example in enumerate(examples)}
            for count, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                cases[index] = future.result()
                print(f"case optimization: {count}/{len(examples)} id={cases[index]['id']} "
                      f"stop={cases[index].get('stop_reason')} selected={cases[index].get('selected_iteration')}", flush=True)

        # 与原流程保持一致：聚类文本由“问题 + 答案”构成；答案缺失时只使用问题。
        cluster_texts = [f"{case['question']}\n{case.get('answer') or case.get('generated_answer', '')}".strip()
                         for case in cases]
        vectors = self.embeddings.encode(cluster_texts)
        centers, labels = _kmeans(vectors, cluster_count)
        groups: Dict[int, List[Dict[str, Any]]] = {cluster_id: [] for cluster_id in range(len(centers))}
        for case, label in zip(cases, labels.tolist()):
            groups.setdefault(int(label), []).append(case)

        clusters = []
        mapping = []
        for cluster_id in range(len(centers)):
            members = groups.get(cluster_id, [])
            prompt_result = self._cluster_prompt(cluster_id, members)
            prompt_path = self.prompt_dir / f"cluster_{cluster_id}.json"
            self.prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(json.dumps(prompt_result, ensure_ascii=False, indent=2), encoding="utf-8")
            for rank, case in enumerate(members, 1):
                mapping.append({"id": case["id"], "cluster_id": cluster_id, "cluster_rank": rank})
            clusters.append({
                "cluster_id": cluster_id,
                "size": len(members),
                "center": [round(float(value), 8) for value in centers[cluster_id].tolist()],
                "example_ids": [case["id"] for case in members],
                "prompt": prompt_result["prompt"],
                "prompt_method": prompt_result["method"],
                "prompt_path": str(prompt_path),
            })

        artifact = {
            "version": CLUSTER_PROMPT_VERSION,
            "method": "question_answer_kmeans_collective_prompt",
            "config": {"cluster_count_requested": int(cluster_count),
                       "cluster_count_actual": len(clusters),
                       "prompt_iterations": int(iterations),
                       "iteration_mode": "adaptive", "min_total_gain": float(min_gain),
                       "training_context": "golden",
                       "evidence_policy": self.evidence_policy,
                       "embedding_text": "question + reference_answer"},
            "metrics": {"sample_count": len(cases),
                        "cluster_count_actual": len(clusters),
                        "silhouette": round(_silhouette(vectors, labels), 6)},
            "clusters": clusters,
            "question_cluster_mapping": mapping,
            "case_iteration_dir": str(self.case_dir),
        }
        artifact_path = self.run_dir / ARTIFACT_NAME
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**artifact, "artifact_path": str(artifact_path)}


class ClusterPromptRouter:
    """加载聚类产物并为新问题选择 Top-K 聚类 Prompt。"""

    def __init__(self, *, run_dir: str | Path, embeddings: EmbeddingModel,
                 enabled: bool = False, top_k: int = 1, evidence_policy=None):
        self.run_dir = Path(run_dir)
        self.embeddings = embeddings
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))
        self.artifact_path = self.run_dir / ARTIFACT_NAME
        self.artifact: Dict[str, Any] = {}
        if self.enabled and self.artifact_path.exists():
            try:
                raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("clusters"), list):
                    self.artifact = raw
            except (OSError, json.JSONDecodeError):
                self.artifact = {}
        saved_policy = self.artifact.get("config", {}).get("evidence_policy", True)
        if evidence_policy is not None and bool(evidence_policy) != saved_policy:
            raise ValueError("Router policy mode must match cluster artifact")
        self.evidence_policy = saved_policy if evidence_policy is None else bool(evidence_policy)

    @property
    def available(self) -> bool:
        return bool(self.artifact.get("clusters"))

    def route(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not self.enabled or not self.available:
            return []
        query_vector = self.embeddings.encode([query])[0]
        candidates = []
        for cluster in self.artifact.get("clusters", []):
            center = cluster.get("center", [])
            if not center:
                continue
            candidates.append({
                "cluster_id": int(cluster.get("cluster_id", 0)),
                "score": round(_cosine(query_vector, center), 6),
                "size": int(cluster.get("size", 0)),
                "prompt": _safe_module(cluster.get("prompt"), evidence_policy=self.evidence_policy),
                "prompt_path": cluster.get("prompt_path", ""),
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(candidates, 1):
            item["rank"] = rank
        return candidates[:max(1, int(top_k or self.top_k))]
