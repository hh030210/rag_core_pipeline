"""Prompt 扩展、案例级迭代优化和最终问答 Prompt。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .llm import LLMClient


DEFAULT_SYSTEM_PROMPT = """你是知识库问答助手。你只能依据给定的检索上下文回答问题。
请直接给出准确、完整、清晰的答案，保留上下文中的人名、地名、时间和数量。
如果上下文没有足够证据，请明确说明“文中未说明”，不要使用外部知识或编造信息。
不要输出思考过程。"""


class PromptOptimizer:
    """保留原项目的三步在线扩展，并提供可审计的案例级迭代。"""

    def __init__(self, *, client: LLMClient, run_dir: str | Path):
        self.client = client
        self.run_dir = Path(run_dir)
        self.prompt_path = self.run_dir / "optimized_prompt.json"
        self.cache_path = self.run_dir / "query_expansion_cache.json"
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    @staticmethod
    def default_module() -> Dict[str, str]:
        return {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "instruction": "先定位问题所需事实，再逐项回答。",
            "context_strategy": "优先使用与问题实体、属性和时间条件直接相关的片段。",
            "format_requirement": "使用自然语言回答；问题包含多个要点时分点回答。",
            "uncertainty_handling": "证据不足时明确指出缺失信息，不补写外部事实。",
        }

    def load_module(self) -> Dict[str, str]:
        if self.prompt_path.exists():
            try:
                data = json.loads(self.prompt_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("system_prompt"):
                    return self.normalize_module(data)
            except Exception:
                pass
        return self.default_module()

    @classmethod
    def normalize_module(cls, module: Dict[str, Any] | None) -> Dict[str, str]:
        source = module if isinstance(module, dict) else {}
        default = cls.default_module()
        return {key: str(source.get(key, default[key])) for key in default}

    def expand(self, query: str) -> Dict[str, Any]:
        if query in self.cache:
            sub_queries = self.cache[query].get("sub_queries", [query])
            entity_terms = self.cache[query].get("entity_terms", [])
            return {"original_query": query, "sub_queries": sub_queries,
                    "entity_terms": entity_terms, "cached": True}
        sub_queries = [query]
        entity_terms: List[str] = []
        if not self.client.mock and self.client.api_key:
            try:
                raw = self.client.complete(
                    "你是检索查询分析器，只输出查询列表。",
                    f"将下面的问题拆为 2 至 4 个互相独立的检索子查询，每行一条，不要序号或解释：\n{query}",
                    temperature=0.2, max_tokens=256,
                )
                candidates = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
                sub_queries = [line for line in candidates if 4 < len(line) < 120][:4] or [query]
            except Exception as exc:
                print(f"[提示] 子查询扩展失败，使用原问题: {exc}")
            try:
                raw = self.client.complete(
                    "你是实体术语提取器，只输出用 | 分隔的术语。",
                    f"从问题中提取实体和关键术语（最多10个），不要解释：\n{query}",
                    temperature=0.1, max_tokens=96,
                )
                entity_terms = [term.strip() for term in raw.split("|") if term.strip()][:10]
            except Exception as exc:
                print(f"[提示] 实体提取失败，保留空列表: {exc}")
        self.cache[query] = {"sub_queries": sub_queries, "entity_terms": entity_terms}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"original_query": query, "sub_queries": sub_queries,
                "entity_terms": entity_terms, "cached": False}

    def optimize(self, examples: List[Dict[str, Any]], iterations: int = 3, *,
                 initial_module: Dict[str, Any] | None = None,
                 output_path: str | Path | None = None) -> Dict[str, Any]:
        current = self.normalize_module(initial_module) if initial_module is not None else self.load_module()
        history = []
        examples = [item for item in examples if isinstance(item, dict) and item.get("question")][:20]
        if not examples or self.client.mock or not self.client.api_key:
            return self._save(current, history, "no_examples_or_mock", output_path=output_path)
        for round_index in range(max(1, int(iterations))):
            feedback = []
            scores = []
            for example in examples:
                context = str(example.get("context", example.get("retrieved_context", "")))[:6000]
                reference = str(example.get("reference_answer", example.get("answer", "")))[:2000]
                answer = self.client.complete(
                    current["system_prompt"],
                    self._qa_user(example["question"], context, current),
                    temperature=0.1, max_tokens=800,
                )
                try:
                    judged = self.client.complete_json(
                        "你是严格的 RAG 答案评测器，只输出 JSON。",
                        "评估候选答案与参考答案的一致性，返回 {\"score\":0到100,\"feedback\":\"具体问题\"}。\n"
                        f"问题：{example['question']}\n参考答案：{reference}\n候选答案：{answer}\n检索上下文：{context}",
                        temperature=0.0, max_tokens=256,
                    )
                    scores.append(float(judged.get("score", 0)))
                    feedback.append(str(judged.get("feedback", "")))
                except Exception as exc:
                    feedback.append(f"评测失败：{exc}")
            rewrite_user = (
                "根据当前 Prompt 和多条评测反馈，改写一个更适合本知识库问答的 Prompt。"
                "必须保留上下文约束、证据不足时拒答、完整回答多个要点的要求。只输出 JSON，字段为 "
                "system_prompt、instruction、context_strategy、format_requirement、uncertainty_handling。\n"
                f"当前 Prompt：{json.dumps(current, ensure_ascii=False)}\n"
                f"本轮平均分：{sum(scores)/len(scores) if scores else 0:.2f}\n"
                f"评测反馈：{json.dumps(feedback, ensure_ascii=False)}"
            )
            try:
                updated = self.client.complete_json("你是 Prompt 优化专家，只输出合法 JSON。", rewrite_user,
                                                    temperature=0.1, max_tokens=1000)
                if isinstance(updated, dict) and updated.get("system_prompt"):
                    current = {key: str(updated.get(key, current.get(key, "")))
                               for key in ("system_prompt", "instruction", "context_strategy",
                                           "format_requirement", "uncertainty_handling")}
            except Exception as exc:
                feedback.append(f"Prompt 改写失败：{exc}")
            history.append({"iteration": round_index + 1, "scores": scores, "feedback": feedback})
        return self._save(current, history, "iterative_optimization", output_path=output_path)

    @staticmethod
    def _qa_user(question: str, context: str, module: Dict[str, str]) -> str:
        return (f"<instruction>{module.get('instruction', '')}</instruction>\n"
                f"<context_strategy>{module.get('context_strategy', '')}</context_strategy>\n"
                f"<format_requirement>{module.get('format_requirement', '')}</format_requirement>\n"
                f"<uncertainty_handling>{module.get('uncertainty_handling', '')}</uncertainty_handling>\n\n"
                f"检索上下文：\n{context}\n\n用户问题：\n{question}\n\n请给出最终答案：")

    def _save(self, module: Dict[str, str], history: List[Dict[str, Any]], method: str,
              *, output_path: str | Path | None = None) -> Dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        result = {**module, "method": method, "history": history}
        path = Path(output_path) if output_path else self.prompt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
