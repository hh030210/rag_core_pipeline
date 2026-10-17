"""Prompt 扩展、案例级迭代优化和最终问答 Prompt。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

from .llm import LLMClient
from .prompt_policy import protect_module


DEFAULT_SYSTEM_PROMPT = """你是知识库问答助手。你只能依据给定的检索上下文回答问题。
请直接给出准确、完整、清晰的答案，保留上下文中的人名、地名、时间和数量。
如果上下文没有足够证据，请明确说明“文中未说明”，不要使用外部知识或编造信息。
不要输出思考过程。"""


class PromptOptimizer:
    """保留原项目的三步在线扩展，并提供可审计的案例级迭代。"""

    def __init__(self, *, client: LLMClient, run_dir: str | Path, evidence_policy=True,
                 meta_prompt_path=None):
        self.client = client
        self.run_dir = Path(run_dir)
        self.evidence_policy = bool(evidence_policy)
        self.meta_prompt_path = Path(meta_prompt_path) if meta_prompt_path else None
        self.prompt_path = self.run_dir / "optimized_prompt.json"
        self.cache_path = self.run_dir / "query_expansion_cache.json"
        self._cache_lock = threading.Lock()
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    @staticmethod
    def default_module(evidence_policy=True) -> Dict[str, str]:
        return protect_module({
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "instruction": "先定位问题所需事实，再逐项回答。",
            "context_strategy": "优先使用与问题实体、属性和时间条件直接相关的片段。",
            "format_requirement": "使用自然语言回答；问题包含多个要点时分点回答。",
            "uncertainty_handling": "证据不足时明确指出缺失信息，不补写外部事实。",
        }, evidence_policy=evidence_policy)

    def load_module(self) -> Dict[str, str]:
        if self.prompt_path.exists():
            try:
                data = json.loads(self.prompt_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("system_prompt"):
                    return self.normalize_module(data, evidence_policy=self.evidence_policy)
            except Exception:
                pass
        return self.default_module(self.evidence_policy)

    @classmethod
    def normalize_module(cls, module: Dict[str, Any] | None, evidence_policy=True) -> Dict[str, str]:
        source = module if isinstance(module, dict) else {}
        default = cls.default_module(evidence_policy)
        return protect_module({key: str(source.get(key, default[key])) for key in default}, evidence_policy=evidence_policy)

    def expand(self, query: str) -> Dict[str, Any]:
        with self._cache_lock:
            cached = self.cache.get(query)
        if cached:
            sub_queries = cached.get("sub_queries", [query])
            entity_terms = cached.get("entity_terms", [])
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
        with self._cache_lock:
            self.cache[query] = {"sub_queries": sub_queries, "entity_terms": entity_terms}
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return {"original_query": query, "sub_queries": sub_queries,
                "entity_terms": entity_terms, "cached": False}

    def optimize(self, examples: List[Dict[str, Any]], iterations: int = 30, *,
                 initial_module: Dict[str, Any] | None = None,
                 output_path: str | Path | None = None,
                 min_gain: float = 1.0) -> Dict[str, Any]:
        examples = [item for item in examples if isinstance(item, dict) and item.get("question")]
        if self.client.mock:
            return self._save(self.default_module(self.evidence_policy), [], "no_examples_or_mock", output_path=output_path,
                              metadata={"mock": True, "stop_reason": "mock_only_no_real_optimization"})
        from .adaptive_prompting import adaptive_optimize
        return adaptive_optimize(self, examples, initial_module=initial_module,
                                 max_rounds=int(iterations), min_gain=float(min_gain), output_path=output_path)

    @staticmethod
    def _qa_user(question: str, context: str, module: Dict[str, str]) -> str:
        return (f"<instruction>{module.get('instruction', '')}</instruction>\n"
                f"<context_strategy>{module.get('context_strategy', '')}</context_strategy>\n"
                f"<format_requirement>{module.get('format_requirement', '')}</format_requirement>\n"
                f"<uncertainty_handling>{module.get('uncertainty_handling', '')}</uncertainty_handling>\n\n"
                f"检索上下文：\n{context}\n\n用户问题：\n{question}\n\n请给出最终答案：")

    def _save(self, module: Dict[str, str], history: List[Dict[str, Any]], method: str,
              *, output_path: str | Path | None = None,
              metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        result = {**module, "method": method, "history": history, **(metadata or {})}
        path = Path(output_path) if output_path else self.prompt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
