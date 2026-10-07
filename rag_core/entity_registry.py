"""Corpus-derived landmark entities for scoped retrieval.

The curated POI table is useful for high-confidence aliases, but it cannot
cover every pavilion, hall, cave, bridge, or stele added with new documents.
This module builds a conservative sidecar index from chunk text at indexing
time.  Only landmark-shaped, corpus-unique mentions are admitted, so it is not
a general n-gram matcher.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping


LANDMARK_SUFFIXES = frozenset("阁殿寺庙陵洞塔桥堤坊门院祠亭岩寨山湖园馆宫台关道壁碑楼")
ENTITY_STOPWORDS = {
    "这个", "那个", "这里", "那里", "景区", "景点", "建筑", "地方", "里面", "外面",
    "大门", "山洞", "寺庙", "公园", "园区", "道路", "门票", "时间", "文化", "历史",
}


def extract_landmark_mentions(*texts: Any, max_length: int = 6) -> List[str]:
    """Extract conservative CJK landmark names such as ``思鲁阁`` or ``药方洞``.

    We use suffix-shaped spans rather than a broad N-gram vocabulary. The
    result intentionally favors precision: entities not matching this pattern
    can still be supplied by the LLM tagging stage as ``entity_mentions``.
    """
    found = set()
    for text in texts:
        value = str(text or "")
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", value):
            for end, char in enumerate(segment, 1):
                if char not in LANDMARK_SUFFIXES:
                    continue
                lower = max(0, end - max_length)
                for start in range(lower, end - 1):
                    term = segment[start:end]
                    if len(term) < 3 or any(token in term for token in "的和与及或在是有将把") or term in ENTITY_STOPWORDS:
                        continue
                    found.add(term)
    return sorted(found, key=lambda item: (-len(item), item))


class CorpusEntityRegistry:
    """Map corpus entities to scenic scopes with optional precision controls."""

    def __init__(self, points: Iterable[Mapping[str, Any]], *, max_document_frequency: int = 10,
                 structured_only: bool = False):
        scopes_by_term: Dict[str, set[str]] = defaultdict(set)
        df_by_term: Dict[str, set[str]] = defaultdict(set)
        chunk_ids_by_term: Dict[str, set[str]] = defaultdict(set)
        for point in points:
            payload = point.get("payload", {}) if isinstance(point, Mapping) else {}
            scope = str(payload.get("source_file", "")).split("-", 1)[0].strip()
            chunk_id = str(payload.get("chunk_id") or point.get("id") or "")
            if not scope or not chunk_id:
                continue
            supplied = payload.get("entity_mentions", [])
            if not isinstance(supplied, list):
                supplied = [supplied]
            text = "\n".join(str(payload.get(key, "")) for key in (
                "doc_title", "chunk_gen_title", "chunk_text_full",
            ))
            mentions = set()
            for item in supplied:
                if not isinstance(item, Mapping):
                    continue
                entity_type = str(item.get("type", "")).strip().lower()
                if structured_only and entity_type not in {
                    "poi", "building", "person", "artifact", "organization",
                }:
                    continue
                for value in [item.get("name", ""), *(item.get("aliases", []) or [])]:
                    value = str(value).strip()
                    if value and value in text:
                        mentions.add(value)
            if not structured_only:
                # Backward compatibility: legacy payloads have no sidecar field.
                mentions.update(extract_landmark_mentions(
                    payload.get("doc_title", ""), payload.get("chunk_gen_title", ""),
                    payload.get("chunk_text_full", ""),
                ))
            for term in mentions:
                if len(term) < 2 or len(term) > 12 or term in ENTITY_STOPWORDS:
                    continue
                scopes_by_term[term].add(scope)
                df_by_term[term].add(chunk_id)
                chunk_ids_by_term[term].add(chunk_id)
        self._scopes_by_term = {
            term: sorted(scopes)
            for term, scopes in scopes_by_term.items()
            if len(df_by_term[term]) <= max_document_frequency
        }
        self._chunk_ids_by_term = {
            term: sorted(chunk_ids_by_term[term]) for term in self._scopes_by_term
        }

    def resolve(self, query: str) -> List[Dict[str, str]]:
        result = []
        for term in sorted(self._scopes_by_term, key=len, reverse=True):
            if term not in query:
                continue
            for scope in self._scopes_by_term[term]:
                result.append({
                    "mention": term,
                    "canonical_name": term,
                    "scenic_area": scope,
                    "entity_type": "auto_landmark",
                })
        return result

    def scenic_scopes(self, query: str) -> List[str]:
        return list(dict.fromkeys(item["scenic_area"] for item in self.resolve(query)))

    def candidate_chunk_ids(self, query: str) -> List[str]:
        """Direct entity inverted lookup for already indexed entity mentions."""
        ids = []
        for term in sorted(self._chunk_ids_by_term, key=len, reverse=True):
            if term in query:
                ids.extend(self._chunk_ids_by_term[term])
        return list(dict.fromkeys(ids))
