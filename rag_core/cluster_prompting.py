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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompting import PromptOptimizer


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
        context = str(_first(raw, ("context", "retrieved_context", "gold_context"), "") or "").strip()
        if not context:
            context = "\n".join(_text_list(_first(raw, ("gold_evidence_texts", "evidence_texts", "source"), [])))
        reference = str(_first(
            raw, ("reference_answer", "answer", "standard_answer", "gold_answer"), ""
        ) or "").strip()
        sample_id = str(_first(raw, ("id", "question_id", "qid"), f"qa_{index:04d}"))
        examples.append({
            **dict(raw),
            "id": sample_id,
            "question": question,
            "context": context,
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


def _safe_module(value: Any) -> Dict[str, str]:
    return PromptOptimizer.normalize_module(value if isinstance(value, Mapping) else None)


class ClusterPromptPipeline:
    """把 QA 迭代、聚类和群智 Prompt 生成串成一个可审计流水线。"""

    def __init__(self, *, client: LLMClient, embeddings: EmbeddingModel,
                 run_dir: str | Path):
        self.client = client
        self.embeddings = embeddings
        self.run_dir = Path(run_dir)
        self.case_dir = self.run_dir / "cluster_prompting" / "case_iterations"
        self.prompt_dir = self.run_dir / "cluster_prompting" / "cluster_prompts"

    def _case_iteration(self, example: Dict[str, Any], iterations: int) -> Dict[str, Any]:
        """复用统一 PromptOptimizer，避免从旧项目复制第二套 LLM 逻辑。"""
        case_id = str(example["id"])
        output_path = self.case_dir / f"{case_id}.json"
        try:
            optimizer = PromptOptimizer(client=self.client, run_dir=self.case_dir)
            result = optimizer.optimize(
                [example], iterations=max(0, int(iterations)),
                output_path=output_path,
            )
            return {
                "id": case_id,
                "question": example["question"],
                "answer": example.get("reference_answer", ""),
                "context": example.get("context", ""),
                "final_prompt": _safe_module(result),
                "history": result.get("history", []),
                "generated_answer": self._last_generated_answer(result.get("history", [])),
            }
        except Exception as exc:
            # 单个样本失败不阻断其余样本；失败信息会进入审计文件。
            fallback = {
                "id": case_id,
                "question": example["question"],
                "answer": example.get("reference_answer", ""),
                "context": example.get("context", ""),
                "final_prompt": PromptOptimizer.default_module(),
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
        fallback = PromptOptimizer.default_module()
        if self.client.mock or not self.client.api_key:
            return {"cluster_id": cluster_id, "prompt": fallback,
                    "source_case_count": len(cases), "method": "fallback"}

        summaries = []
        for case in cases[:20]:
            summaries.append({
                "question": case.get("question", ""),
                "reference_answer": case.get("answer", ""),
                "final_prompt": case.get("final_prompt", {}),
                "iteration_history": case.get("history", []),
            })
        user = (
            "请根据同一问题簇中的多条 QA 迭代记录，生成这个问题簇通用的问答 Prompt。\n"
            "要求：只依据检索上下文；完整回答问题中的多个要点；保留实体、时间、数量；"
            "证据不足时明确说明，不得编造；只输出 JSON，字段必须是："
            "system_prompt、instruction、context_strategy、format_requirement、uncertainty_handling。\n"
            f"问题簇编号：{cluster_id}\n"
            f"QA 迭代记录：{json.dumps(summaries, ensure_ascii=False)}"
        )
        try:
            result = self.client.complete_json(
                "你是聚类级 Prompt 群智优化专家，只输出合法 JSON。",
                user,
                temperature=0.1,
                max_tokens=1400,
                repair_prompt="把原始输出修复为包含五个指定字符串字段的合法 JSON。",
            )
            module = _safe_module(result)
            method = "llm_collective_synthesis"
        except Exception as exc:
            module = fallback
            method = f"fallback:{type(exc).__name__}"
        return {"cluster_id": cluster_id, "prompt": module,
                "source_case_count": len(cases), "method": method}

    def run(self, examples: Iterable[Mapping[str, Any]], *, cluster_count: int = 6,
            iterations: int = 3) -> Dict[str, Any]:
        examples = normalize_examples(examples)
        if not examples:
            raise ValueError("聚类 Prompt 流程至少需要一条有效 QA 样本")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        cases = [self._case_iteration(example, iterations) for example in examples]

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
                 enabled: bool = False, top_k: int = 1):
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
                "prompt": _safe_module(cluster.get("prompt")),
                "prompt_path": cluster.get("prompt_path", ""),
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(candidates, 1):
            item["rank"] = rank
        return candidates[:max(1, int(top_k or self.top_k))]
