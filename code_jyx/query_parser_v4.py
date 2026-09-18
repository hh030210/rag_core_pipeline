"""隔离的 v4 查询解析器。

v4 的目标不是把所有问题强行转成标签，而是把查询拆成两种可审计的信号：

* ``labels``：问题中出现、可以进入叶子维度倒排索引的规范/原子标签；
* ``intent_terms``：没有规范标签字面值，但能说明查询意图的短语，例如“几点开门”。

两者都保留在约束中，但只有前者走精确标签匹配，后者走同叶子维度的语义召回。
当 LLM 失败或返回空约束时，继续使用 QueryParserV2 中的 schema 约束确定性恢复。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from .query_parser import QueryParserV2, _as_list, _clean_terms
from .schema_v2 import SCHEMA_VERSION, normalize_label, normalize_text


QUERY_PARSER_V4_VERSION = "4.0"


class QueryParserV4(QueryParserV2):
    """Schema-constrained parser with intent-slot preservation and diagnostics."""

    def _cache_key(self, qid: Any, query_text: str) -> str:
        digest = hashlib.sha1(normalize_text(query_text).encode("utf-8")).hexdigest()
        return (
            f"{SCHEMA_VERSION}:query_parser_{QUERY_PARSER_V4_VERSION}:"
            f"{self.vocabulary_signature}:{qid}:{digest}"
        )

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if (
                isinstance(value, dict)
                and value.get("schema_version") == SCHEMA_VERSION
                and value.get("parser_version") == QUERY_PARSER_V4_VERSION
                and value.get("vocabulary_signature") == self.vocabulary_signature
            ):
                self.cache = value.get("items", {})
        except (OSError, json.JSONDecodeError):
            self.cache = {}

    def _save_cache(self) -> None:
        parent = os.path.dirname(self.cache_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "parser_version": QUERY_PARSER_V4_VERSION,
                    "vocabulary_signature": self.vocabulary_signature,
                    "items": self.cache,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _allowed_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Only expose indexable leaves to the LLM and exact parser."""

        allowed: Dict[str, Dict[str, Any]] = {}
        for node_id in self.maps["leaves"]:
            node = self.maps["nodes"][node_id]
            allowed[str(node_id)] = node
            allowed[normalize_text(node.get("name", ""))] = node
            for alias in _as_list(node.get("aliases")):
                alias = normalize_text(alias)
                if alias:
                    allowed[alias] = node
        return allowed

    @staticmethod
    def _slot(item: Mapping[str, Any], index: int) -> Dict[str, Any]:
        labels = list(item.get("labels", []) or [])
        intent_terms = list(item.get("intent_terms", []) or [])
        return {
            "slot_id": f"slot_{index}",
            "dimension_id": str(item.get("dimension_id", "")),
            "labels": labels,
            "intent_terms": intent_terms,
            "required": True,
        }

    def parse(self, qid: Any, query_text: str) -> Dict[str, Any]:
        key = self._cache_key(qid, query_text)
        if key in self.cache:
            return self.cache[key]

        # Passing leaves only makes it impossible for a model to invent a
        # parent as a retrievable tag. Parent expansion remains a retrieval
        # operation when an explicitly stored parent constraint is encountered.
        nodes = [self.maps["nodes"][node_id] for node_id in self.maps["leaves"]]
        llm_error = ""
        try:
            parser_method = getattr(self.miner, "parse_query_intent_v4", None)
            if parser_method is None:
                parsed = self.miner.parse_query_intent_v2(query_text, nodes)
            else:
                parsed = parser_method(query_text, nodes)
        except Exception as exc:
            parsed = {}
            llm_error = f"{type(exc).__name__}: {exc}"

        llm_constraints = self._clean_llm_constraints(parsed)
        constraints = self._recover_constraints(query_text, llm_constraints)
        raw_diagnostics = parsed.get("diagnostics", {}) if isinstance(parsed, dict) else {}
        if not isinstance(raw_diagnostics, dict):
            raw_diagnostics = {}
        sources = sorted({str(item.get("source", "unknown")) for item in constraints})
        result = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": QUERY_PARSER_V4_VERSION,
            "constraints": constraints,
            "required_slots": [self._slot(item, index) for index, item in enumerate(constraints, 1)],
            "diagnostics": {
                "llm_constraint_count": int(raw_diagnostics.get("llm_constraint_count", len(llm_constraints)) or 0),
                "llm_intent_only_count": int(
                    raw_diagnostics.get(
                        "llm_intent_only_count",
                        sum(bool(item.get("intent_terms")) and not item.get("labels") for item in llm_constraints),
                    )
                    or 0
                ),
                "recovered_constraint_count": len(constraints),
                # ``fallback_used`` answers whether the LLM supplied no usable
                # constraint. Deterministic cue/lexical enrichment is exposed
                # separately so a valid LLM result is not mislabeled as a
                # fallback merely because it was augmented.
                "fallback_used": not bool(llm_constraints),
                "deterministic_enrichment_used": any(
                    item.get("source") != "llm_v4" for item in constraints
                ),
                "recovery_sources": sources,
                "llm_attempts": int(raw_diagnostics.get("llm_attempts", 1) or 1),
                "llm_error": str(raw_diagnostics.get("llm_error", "") or llm_error),
            },
        }
        self.cache[key] = result
        self._save_cache()
        return result
