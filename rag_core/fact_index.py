"""Evidence-grounded entity--relation--attribute fact index.

The index is intentionally independent from dimension labels.  It contains
only triples whose two endpoints and evidence were verified against the chunk,
so fact retrieval can expand recall without turning generated text into a new
uncontrolled tag vocabulary.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_fact_text(value: Any) -> str:
    return re.sub(r"[\s\u3000，。！？；：、（）()【】\[\]{}‘’“”\"'·_\-]+", "", str(value or "").lower())


def validate_fact(item: Any, chunk_text: str = "") -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    subject = str(item.get("subject", "")).strip()
    predicate = str(item.get("predicate", "")).strip().lower()
    obj = str(item.get("object", "")).strip()
    evidence = str(item.get("evidence", "")).strip()
    allowed = {
        "built_by", "built_time", "identity", "located_in", "feature",
        "service_rule", "price", "transport_route",
    }
    if len(subject) < 2 or not obj or predicate not in allowed or not evidence:
        return None
    if chunk_text and (subject not in chunk_text or obj not in chunk_text or evidence not in chunk_text):
        return None
    result = {
        "subject": subject, "subject_type": str(item.get("subject_type", "")).strip().lower(),
        "predicate": predicate, "object": obj, "evidence": evidence,
    }
    try:
        confidence = float(item.get("confidence", 0.0))
        if 0.0 <= confidence <= 1.0:
            result["confidence"] = confidence
    except (TypeError, ValueError):
        pass
    return result


def build_fact_index(records: Iterable[Dict[str, Any]], documents: Dict[str, Any]) -> Dict[str, Any]:
    """Build a serializable exact-match posting index from verified facts."""
    by_chunk: Dict[str, List[Dict[str, Any]]] = {}
    subjects, predicates = defaultdict(set), defaultdict(set)
    objects = defaultdict(set)
    fact_count = 0
    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        text = str(record.get("chunk_text_full", record.get("doc_text", "")) or "")
        document = documents.get(chunk_id, {}) if isinstance(documents, dict) else {}
        raw_facts = document.get("facts", []) if isinstance(document, dict) else []
        if not isinstance(raw_facts, list):
            raw_facts = []
        validated, seen = [], set()
        for raw in raw_facts:
            fact = validate_fact(raw, text)
            if not fact:
                continue
            key = (normalize_fact_text(fact["subject"]), fact["predicate"], normalize_fact_text(fact["object"]))
            if key in seen:
                continue
            seen.add(key)
            validated.append(fact)
            subjects[key[0]].add(chunk_id)
            predicates[fact["predicate"]].add(chunk_id)
            objects[key[2]].add(chunk_id)
        if validated:
            by_chunk[chunk_id] = validated
            fact_count += len(validated)
    return {
        "schema_version": "fact_index_v1",
        "fact_count": fact_count,
        "chunk_count": len(by_chunk),
        "facts_by_chunk": by_chunk,
        "subject_postings": {key: sorted(value) for key, value in subjects.items()},
        "predicate_postings": {key: sorted(value) for key, value in predicates.items()},
        "object_postings": {key: sorted(value) for key, value in objects.items()},
    }


class FactIndex:
    def __init__(self, path: str | Path):
        source = Path(path)
        self.data = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
        self.facts_by_chunk = self.data.get("facts_by_chunk", {}) if isinstance(self.data, dict) else {}
        self.subject_postings = self.data.get("subject_postings", {}) if isinstance(self.data, dict) else {}

    def matches(self, query_facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Return exact-subject, controlled-predicate matches by chunk.

        Object is optional in a question.  If present it acts as an additional
        literal constraint, not a semantic guess.
        """
        result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for query_fact in query_facts or []:
            subject = normalize_fact_text(query_fact.get("subject", ""))
            predicate = str(query_fact.get("predicate", "")).strip().lower()
            obj = normalize_fact_text(query_fact.get("object", ""))
            if not subject or not predicate:
                continue
            for chunk_id in self.subject_postings.get(subject, []):
                for fact in self.facts_by_chunk.get(str(chunk_id), []):
                    if str(fact.get("predicate", "")) != predicate:
                        continue
                    fact_obj = normalize_fact_text(fact.get("object", ""))
                    evidence = normalize_fact_text(fact.get("evidence", ""))
                    if obj and obj not in fact_obj and obj not in evidence:
                        continue
                    result[str(chunk_id)].append({
                        "query_fact": {"subject": query_fact.get("subject", ""), "predicate": predicate,
                                       "object": query_fact.get("object", "")},
                        "fact": fact,
                    })
        return dict(result)
