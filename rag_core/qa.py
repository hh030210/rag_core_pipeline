"""非交互问答入口：查询扩展、融合检索、完整证据上下文和答案生成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompting import PromptOptimizer
from .retrieval import Retriever


class QAService:
    def __init__(self, *, run_dir: str | Path, settings):
        self.run_dir = Path(run_dir)
        self.settings = settings
        settings.apply_runtime_environment()
        self.llm = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                             model=settings.llm_model, openai_compat=settings.llm_openai_compat,
                             interval=settings.llm_interval, mock=settings.mock)
        self.embeddings = EmbeddingModel(settings.model_path, settings.embedding_device,
                                         settings.vector_dim, settings.mock)
        self.prompts = PromptOptimizer(client=self.llm, run_dir=self.run_dir)
        self.retriever = Retriever(run_dir=self.run_dir, settings=settings,
                                   embeddings=self.embeddings,
                                   llm=None if settings.mock else self._dimension_miner())

    def _dimension_miner(self):
        if not self.settings.llm_api_key:
            return None
        try:
            from .llm_service import DimensionMiningWithQwen
            return DimensionMiningWithQwen(api_key=self.settings.llm_api_key,
                                           model_name=self.settings.llm_model,
                                           base_url=self.settings.llm_base_url)
        except Exception as exc:
            print(f"[提示] 查询维度解析器不可用，使用确定性解析: {exc}")
            return None

    def answer(self, query: str, *, top_k: int | None = None, context_chars: int = 0) -> Dict[str, Any]:
        expansion = self.prompts.expand(query)
        retrieval_query = " | ".join(expansion["sub_queries"])
        retrieval = self.retriever.search(retrieval_query, top_k=top_k or self.settings.top_k)
        chunks = retrieval.get("top_chunks", [])
        context_parts = []
        context_stats = []
        for index, chunk in enumerate(chunks):
            text = str(chunk.get("chunk_text_full") or chunk.get("chunk_text") or "")
            if context_chars > 0 and len(text) > context_chars:
                text = text[:context_chars]
            context_parts.append(f"[来源 {index + 1}] {chunk.get('doc_title', '')} "
                                 f"({chunk.get('chunk_id', '')})：\n{text}")
            context_stats.append({"chunk_id": chunk.get("chunk_id", ""),
                                  "source_chars": len(str(chunk.get("chunk_text_full") or "")),
                                  "provided_chars": len(text)})
        context = "\n\n".join(context_parts)
        if not context:
            answer = "检索不到足够的知识库证据，无法回答。"
        elif self.settings.mock:
            answer = str(chunks[0].get("chunk_text_full") or chunks[0].get("chunk_text") or "")
        else:
            module = self.prompts.load_module()
            answer = self.llm.complete(module["system_prompt"], self.prompts._qa_user(query, context, module),
                                       temperature=0.1, max_tokens=1200)
        result = {
            "query": query, "retrieval_query": retrieval_query,
            "expansion": expansion, "retrieval": retrieval,
            "answer": answer.strip(),
            "context_stats": {"mode": "full" if context_chars <= 0 else "bounded",
                               "chunks": context_stats,
                               "provided_chars": sum(item["provided_chars"] for item in context_stats)},
            "prompt": self.prompts.load_module(),
        }
        (self.run_dir / "last_answer.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
