"""维度与语义候选召回及打分。"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:  # pragma: no cover - lightweight local environments
    np = None

from .dimension_labels import CanonicalLabelResolver
from .entity_registry import CorpusEntityRegistry
from .fact_index import FactIndex
from .retrieval_fusion.fusion import _save_fusion_snapshot, adaptive_fusion
from .poi_registry import PoiRegistry
from .schema_v2 import load_schema, normalize_label, normalize_text, schema_maps
from .storage import VectorStore
from .vector_retrieval import VectorRetriever


TAG_VECTOR_THRESHOLD_PROFILES = {
    "global_058": {},
    # Open, descriptive dimensions need more recall; short operational and
    # entity labels need more precision to avoid accidental semantic matches.
    "typed_moderate": {
        "historical_event": 0.54,
        "cultural_connotation": 0.54,
        "artistic_feature": 0.55,
        "landscape_composition": 0.54,
        "visual_feature": 0.55,
        "entity_type": 0.66,
        "person_role": 0.66,
        "location_hierarchy": 0.68,
        "opening_period": 0.70,
        "visitor_service": 0.70,
        "ticket_type": 0.70,
        "ticket_policy": 0.70,
        "transportation_mode": 0.70,
    },
}

# Function words and broad tourism nouns must never become fact anchors.  The
# remaining low-document-frequency CJK fragments act as lightweight POI,
# person, building, or named-fact signals without requiring an external NER.
FACT_ANCHOR_STOPWORDS = {
    "什么", "怎么", "为什么", "哪些", "哪个", "哪里", "多少", "如何", "可以", "有没有",
    "一个", "这个", "那个", "里面", "现在", "以前", "分别", "还有", "一下", "一下子",
    "景区", "景点", "建筑", "地方", "内容", "特点", "历史", "文化", "旅游", "参观",
    "皇帝", "陵墓", "门票", "优惠", "服务", "时间", "问题", "介绍", "推荐", "值得",
}


def _unit_vector(values: List[float]) -> List[float]:
    """Return a plain, normalized list for repeated tag comparisons."""
    if np is not None:
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector
    numeric = [float(value) for value in values]
    norm = math.sqrt(math.fsum(value * value for value in numeric))
    return [value / norm for value in numeric] if norm > 0 else numeric


def _dot_unit_vectors(left: List[float], right: List[float]) -> float:
    """One-pass dot product for vectors normalized at cache time."""
    if np is not None:
        return float(np.dot(left, right))
    total = 0.0
    for index in range(min(len(left), len(right))):
        total += left[index] * right[index]
    return total


class _DeterministicQueryMiner:
    """No-op LLM adapter used to run QueryParserV4 recovery offline."""

    @staticmethod
    def parse_query_intent_v4(query_text, nodes):
        return {
            "schema_version": "4.0",
            "constraints": [],
            "diagnostics": {"mode": "deterministic", "llm_attempts": 0},
        }


class Retriever:
    def __init__(self, *, run_dir: str | Path, settings, embeddings, llm=None):
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.embeddings = embeddings
        self.llm = llm
        self.schema = load_schema(self.run_dir / "V_core_v2.json", allow_legacy=True, strict=False)
        self.maps = schema_maps(self.schema)
        self.dimension_paths = {
            leaf_id: " / ".join(
                self.maps["nodes"][part]["name"]
                for part in self.maps["paths"].get(leaf_id, [leaf_id])
                if part in self.maps["nodes"]
            )
            for leaf_id in self.maps["leaves"]
        }
        self.store = VectorStore(run_dir=self.run_dir, backend=settings.backend,
                                 qdrant_url=settings.qdrant_url,
                                 collection=settings.collection, vector_dim=settings.vector_dim)
        self.vector_retriever = VectorRetriever(
            embeddings=self.embeddings,
            vector_store=self.store,
            vector_name=getattr(settings, "vector_name", "chunk_text_vec"),
        )
        raw_index = self.run_dir / "inverted_index_v2.json"
        self.postings = json.loads(raw_index.read_text(encoding="utf-8")) if raw_index.exists() else {}
        self.points = self.store.scroll()
        self.fact_index = FactIndex(self.run_dir / "fact_index_v1.json")
        self.tags_by_dim = {leaf_id: set((self.postings.get(leaf_id) or {}).keys())
                            for leaf_id in self.maps["leaves"]}
        # Dimension scoring can only produce a positive score for payloads
        # carrying at least one extracted leaf tag.  Keep the full point list
        # for semantic retrieval, but avoid rescanning untagged chunks for
        # every query during the dimension route.
        self.dimension_points_by_leaf = {leaf_id: [] for leaf_id in self.maps["leaves"]}
        self.dimension_points = []
        for point in self.points:
            payload = point.get("payload", {})
            has_dimension_tags = False
            for leaf_id in self.maps["leaves"]:
                if payload.get(f"dim_{leaf_id}"):
                    self.dimension_points_by_leaf[leaf_id].append(point)
                    has_dimension_tags = True
            if has_dimension_tags:
                self.dimension_points.append(point)
        self.source_spot_names = sorted({
            str(point.get("payload", {}).get("source_file", "")).split("-", 1)[0].strip()
            for point in self.points
            if str(point.get("payload", {}).get("source_file", "")).strip()
        }, key=len, reverse=True)
        # ``spot_name`` is empty in the current corpus, while questions often
        # name a child POI.  Resolve those aliases to the source-file-level
        # scenic area before candidate filtering.
        self.poi_registry = PoiRegistry(self.source_spot_names)
        self.auto_entity_registry = CorpusEntityRegistry(self.points)
        self.precise_entity_registry = CorpusEntityRegistry(
            self.points, max_document_frequency=6, structured_only=True,
        )
        self._anchor_df_cache: Dict[str, int] = {}
        self.label_resolver = CanonicalLabelResolver(self.run_dir, self.tags_by_dim)
        self.concept_df: Dict[str, Dict[str, int]] = {}
        tag_vector_keys, tag_vector_texts, seen_tag_inputs = [], [], set()
        for point in self.points:
            payload = point.get("payload", {})
            for leaf_id in self.maps["leaves"]:
                values = payload.get(f"dim_{leaf_id}", [])
                if not isinstance(values, list):
                    values = [values]
                concepts = self.label_resolver.concepts(leaf_id, values)
                counter = self.concept_df.setdefault(leaf_id, {})
                for concept in concepts:
                    counter[concept] = counter.get(concept, 0) + 1
                for value in values:
                    if not str(value).strip():
                        continue
                    evidence = self._tag_evidence(payload, leaf_id, value)
                    cache_key = self._tag_vector_key(leaf_id, value, evidence)
                    if cache_key not in seen_tag_inputs:
                        tag_vector_keys.append(cache_key)
                        tag_vector_texts.append(
                            self._tag_embedding_text(leaf_id, str(value), evidence)
                        )
                        seen_tag_inputs.add(cache_key)
        self.tag_vector_cache = {}
        cache_identity = "|".join((
            str(getattr(self.embeddings, "model_path", "")),
            str(getattr(self.embeddings, "dimension", "")),
            str(bool(getattr(self.embeddings, "mock", False))),
            str(getattr(self.embeddings, "_mode", "")),
        ))
        cache_digest = hashlib.sha256(cache_identity.encode("utf-8"))
        for text in tag_vector_texts:
            cache_digest.update(text.encode("utf-8"))
            cache_digest.update(b"\0")
        cache_signature = cache_digest.hexdigest()
        # Evidence-aware tag vectors are intentionally stored separately from
        # the older label-only cache.  The cache signature includes every
        # encoded evidence string, so stale vectors can never be reused.
        vector_cache_path = self.run_dir / "dimension_tag_evidence_vectors_v2.npy"
        vector_cache_meta_path = self.run_dir / "dimension_tag_evidence_vectors_v2.json"
        matrix = None
        cache_loaded = False
        if np is not None and vector_cache_path.exists() and vector_cache_meta_path.exists():
            try:
                metadata = json.loads(vector_cache_meta_path.read_text(encoding="utf-8"))
                if metadata.get("signature") == cache_signature and metadata.get("count") == len(tag_vector_keys):
                    matrix = np.load(vector_cache_path, mmap_mode="r")
                    if matrix.ndim != 2 or matrix.shape[0] != len(tag_vector_keys):
                        matrix = None
            except Exception as exc:
                print(f"[维度检索] 标签向量缓存不可复用，将重新编码: {exc}", flush=True)
                matrix = None
        if matrix is not None:
            cache_loaded = True
            self.tag_vector_cache = {key: matrix[index] for index, key in enumerate(tag_vector_keys)}
            print(f"[维度检索] 复用标签向量缓存 {len(tag_vector_keys)}/{len(tag_vector_keys)}", flush=True)
        try:
            cache_batch_size = max(1, int(os.getenv("TAG_VECTOR_CACHE_BATCH_SIZE", "4096")))
        except ValueError:
            cache_batch_size = 4096
        for start in range(0, len(tag_vector_keys) if not cache_loaded else 0, cache_batch_size):
            end = min(start + cache_batch_size, len(tag_vector_keys))
            encode_compact = getattr(self.embeddings, "encode_compact", self.embeddings.encode)
            vectors = encode_compact(tag_vector_texts[start:end])
            if np is not None and vectors is not None and len(vectors) > 0:
                matrix = np.asarray(vectors, dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                np.divide(matrix, norms, out=matrix, where=norms > 0)
                for offset, key in enumerate(tag_vector_keys[start:end]):
                    self.tag_vector_cache[key] = matrix[offset]
            else:
                for key, vector in zip(tag_vector_keys[start:end], vectors):
                    self.tag_vector_cache[key] = _unit_vector(vector)
            if end == len(tag_vector_keys) or end % (cache_batch_size * 4) == 0:
                print(f"[维度检索] 标签向量缓存 {end}/{len(tag_vector_keys)}", flush=True)
        if np is not None and not cache_loaded and tag_vector_keys:
            try:
                matrix = np.stack([self.tag_vector_cache[key] for key in tag_vector_keys]).astype(
                    np.float32, copy=False
                )
                np.save(vector_cache_path, matrix)
                temp_meta_path = vector_cache_meta_path.with_suffix(".json.tmp")
                temp_meta_path.write_text(json.dumps({
                    "signature": cache_signature,
                    "count": len(tag_vector_keys),
                    "dimension": int(matrix.shape[1]),
                }), encoding="utf-8")
                temp_meta_path.replace(vector_cache_meta_path)
                matrix = np.load(vector_cache_path, mmap_mode="r")
                self.tag_vector_cache = {key: matrix[index] for index, key in enumerate(tag_vector_keys)}
                print(f"[维度检索] 标签向量缓存已持久化 {len(tag_vector_keys)} 条", flush=True)
            except Exception as exc:
                print(f"[维度检索] 持久化标签向量缓存失败，继续使用内存缓存: {exc}", flush=True)
        self.tag_vector_matrix = matrix
        self.tag_vector_index = (
            {key: index for index, key in enumerate(tag_vector_keys)}
            if np is not None and matrix is not None else {}
        )
        self.query_parser = None
        try:
            from code_jyx.query_parser_v4 import QueryParserV4

            use_llm_parser = bool(self.llm and not self.settings.mock)
            self.query_parser = QueryParserV4(
                self.schema,
                cache_file=str(
                    self.run_dir
                    / ("query_cache_v4.json" if use_llm_parser else "query_cache_v4_deterministic.json")
                ),
                miner=self.llm if use_llm_parser else _DeterministicQueryMiner(),
                label_vocabulary=self.postings,
            )
        except Exception as exc:
            print(f"[提示] 原始 code_jyx QueryParserV4 不可用，使用内置确定性解析: {exc}")

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"[\s\u3000，。！？；：、（）()【】\[\]{}‘’“”\"'·_\-]+", "", str(value or "").lower())

    def _parse_query(self, query: str) -> Dict[str, Any]:
        constraints: Dict[str, Dict[str, Any]] = {}
        parser_diagnostics = {}
        parsed: Dict[str, Any] = {}

        # QueryParserV4 is authoritative: it separates the property being
        # asked from nouns merely mentioned as the query subject. Running a
        # second unconditional vocabulary scan here would reintroduce the
        # broad entity/type constraints that the strict parser removed.
        if self.query_parser:
            try:
                parsed = self.query_parser.parse("online", query)
                parser_diagnostics = parsed.get("diagnostics", {}) if isinstance(parsed, dict) else {}
                for item in parsed.get("constraints", []) if isinstance(parsed, dict) else []:
                    leaf_id = str(item.get("dimension_id", ""))
                    if leaf_id not in self.maps["leaves"]:
                        continue
                    target = constraints.setdefault(
                        leaf_id, {"labels": [], "intent_terms": [], "match": "ANY", "role": "auxiliary"}
                    )
                    role = str(item.get("role", "auxiliary"))
                    if role == "main" or (role == "secondary" and target.get("role") != "main"):
                        target["role"] = role
                    for key in ("labels", "intent_terms"):
                        for value in item.get(key, []) or []:
                            value = normalize_text(value)
                            if value and value not in target[key]:
                                target[key].append(value)
            except Exception as exc:
                parser_diagnostics = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[提示] 原始 code_jyx 查询解析失败，使用确定性解析: {exc}")

        # 查询父维度时展开所有叶子，但在打分时按父维度取最大值，避免子节点数量带来额外分数。
        for parent_id, children in self.maps["children"].items():
            parent = self.maps["nodes"].get(parent_id, {})
            aliases = [parent.get("name", "")] + list(parent.get("aliases", []) or [])
            if any(alias and alias in query for alias in aliases):
                for child_id in children:
                    constraints.setdefault(child_id, {"labels": [], "intent_terms": [], "match": "ANY", "role": "auxiliary"})

        # Long-tail ontology aliases are only injected into dimensions already
        # selected by the parser.  This makes phrases such as "学生优惠" or
        # "世界文化遗产价值" comparable with their normalized document tags
        # without treating arbitrary POI names as dimension labels.
        for leaf_id, constraint in constraints.items():
            for label in self.label_resolver.labels_in_text(leaf_id, query):
                if label not in constraint["labels"]:
                    constraint["labels"].append(label)

        explicit_spot_names = sorted({str(point.get("payload", {}).get("spot_name", "")).strip()
                                      for point in self.points if str(point.get("payload", {}).get("spot_name", "")).strip()}, key=len, reverse=True)
        direct_names = list(dict.fromkeys(explicit_spot_names + self.source_spot_names))
        poi_scope_enabled = bool(getattr(self.settings, "poi_scope_enabled", True))
        resolved_entities = self.poi_registry.resolve(query) if poi_scope_enabled else []
        auto_entity_enabled = bool(getattr(self.settings, "auto_entity_registry_enabled", False))
        auto_entities = self.auto_entity_registry.resolve(query) if auto_entity_enabled else []
        precise_entity_enabled = bool(getattr(self.settings, "high_precision_entity_anchor_enabled", False))
        precise_entities = self.precise_entity_registry.resolve(query) if precise_entity_enabled else []
        resolved_entities = list({
            (item["mention"], item["canonical_name"], item["scenic_area"]): item
            for item in [*resolved_entities, *auto_entities, *precise_entities]
        }.values())
        # Preserve direct matching for new scenic areas not yet listed in the
        # registry; aliases and child POIs contribute their parent scope.
        matched_spots = list(dict.fromkeys(
            (self.poi_registry.scenic_scopes(query) if poi_scope_enabled else [])
            + (self.auto_entity_registry.scenic_scopes(query) if auto_entity_enabled else [])
            + (self.precise_entity_registry.scenic_scopes(query) if precise_entity_enabled else [])
            + [spot for spot in direct_names if spot in query]
        ))
        return {
            "constraints": constraints,
            "active_dimensions": list(constraints),
            "spot_names": matched_spots,
            "resolved_poi_entities": resolved_entities,
            "precise_entity_terms": [str(item.get("mention", "")) for item in precise_entities
                                     if str(item.get("mention", ""))],
            "confidence": min(1.0, 0.25 + 0.15 * len(constraints)) if constraints else 0.0,
            "routing": parsed.get("routing", {}) if self.query_parser and isinstance(parsed, dict) else {},
            "parser_diagnostics": parser_diagnostics,
        }

    def _dimension_context_text(self, leaf_id: str) -> str:
        node = self.maps["nodes"].get(leaf_id, {})
        name_path = self.dimension_paths.get(leaf_id, str(node.get("name", leaf_id)))
        description = str(node.get("description", "") or "").strip()
        aliases = node.get("aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases_text = "、".join(str(alias).strip() for alias in aliases if str(alias).strip())
        parts = [f"目标信息维度：{name_path}"]
        if description:
            parts.append(f"维度定义：{description}")
        if aliases_text:
            parts.append(f"维度常见表达：{aliases_text}")
        return "。".join(parts)

    def _tag_embedding_text(self, leaf_id: str, tag: str, evidence: str = "") -> str:
        """Encode the specific tagged fact, not repeated long schema boilerplate."""
        node = self.maps["nodes"].get(leaf_id, {})
        dimension_name = str(node.get("name", leaf_id)).strip() or leaf_id
        parts = [f"信息维度：{dimension_name}", f"事实标签：{tag}"]
        if evidence:
            parts.append(f"对应原文证据：{evidence[:500]}")
        return "。".join(parts)

    @staticmethod
    def _tag_vector_key(leaf_id: str, tag: Any, evidence: str = "") -> str:
        normalized_evidence = normalize_text(evidence)
        evidence_digest = (
            hashlib.sha1(normalized_evidence.encode("utf-8")).hexdigest()[:16]
            if normalized_evidence else "no-evidence"
        )
        return f"{leaf_id}:{normalize_label(tag)}:{evidence_digest}"

    @classmethod
    def _tag_evidence(cls, payload: Dict[str, Any], leaf_id: str, tag: Any) -> str:
        """Return the evidence attached to this leaf/tag pair, if available."""
        target = normalize_label(tag)
        if not target:
            return ""
        evidence = []
        for detail in cls._details(payload, leaf_id):
            if not isinstance(detail, dict):
                continue
            if normalize_label(detail.get("label", "")) != target:
                continue
            text = str(detail.get("evidence", "") or "").strip()
            if text and text not in evidence:
                evidence.append(text)
        return "。".join(evidence)[:500]

    def _dimension_query_text(
        self, query: str, leaf_id: str, constraint: Dict[str, Any]
    ) -> str:
        terms = list(constraint.get("labels", []) or []) + list(
            constraint.get("intent_terms", []) or []
        )
        terms = list(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))
        target = "、".join(terms[:8]) or query
        node = self.maps["nodes"].get(leaf_id, {})
        dimension_name = str(node.get("name", leaf_id)).strip() or leaf_id
        return (
            f"信息维度：{dimension_name}。"
            f"用户问题：{query}。"
            f"用户要查找的信息：{target}"
        )

    @staticmethod
    def _matches_spot(payload: Dict[str, Any], spot_names: List[str]) -> bool:
        if not spot_names:
            return True
        explicit = str(payload.get("spot_name", "")).strip()
        source_spot = str(payload.get("source_file", "")).split("-", 1)[0].strip()
        return explicit in spot_names or source_spot in spot_names

    def _anchor_document_frequency(self, term: str) -> int:
        if term not in self._anchor_df_cache:
            self._anchor_df_cache[term] = sum(
                term in str(point.get("payload", {}).get("doc_title", ""))
                or term in str(point.get("payload", {}).get("chunk_gen_title", ""))
                or term in str(point.get("payload", {}).get("chunk_text_full", ""))
                for point in self.points
            )
        return self._anchor_df_cache[term]

    def _fact_anchor_terms(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Extract query literals that are discriminative in this corpus."""
        candidates = {
            str(item.get("mention", ""))
            for item in analysis.get("resolved_poi_entities", [])
            if str(item.get("mention", ""))
        }
        # Names are often not present in the POI registry (for example people
        # or inscriptions).  CJK 2--6 grams let corpus frequency identify them
        # without using a brittle, domain-specific NER model.
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", query):
            for size in range(2, min(6, len(segment)) + 1):
                for start in range(0, len(segment) - size + 1):
                    candidates.add(segment[start:start + size])
        max_df = max(12, math.ceil(len(self.points) * 0.08))
        terms = []
        for term in candidates:
            if term in FACT_ANCHOR_STOPWORDS or len(term) < 2:
                continue
            df = self._anchor_document_frequency(term)
            if 0 < df <= max_df:
                terms.append(term)
        return sorted(terms, key=lambda value: (-len(value), value))[:12]

    def _fact_anchor_bonus(self, payload: Dict[str, Any], terms: List[str]) -> tuple[float, List[Dict[str, Any]]]:
        """Give a bounded tie-breaking bonus to literal named-fact matches."""
        title = (
            str(payload.get("doc_title", ""))
            + "\n" + str(payload.get("chunk_gen_title", ""))
        )
        body = str(payload.get("chunk_text_full", ""))
        matches: List[Dict[str, Any]] = []
        for term in terms:
            if term not in body and term not in title:
                continue
            df = self._anchor_document_frequency(term)
            rarity = math.log((len(self.points) + 1) / (df + 1)) / math.log(len(self.points) + 1)
            bonus = 0.10 + 0.26 * rarity
            in_title = term in title
            if in_title:
                bonus += 0.08
            matches.append({
                "term": term, "document_frequency": df,
                "in_title": in_title, "bonus": round(bonus, 6),
            })
        matches.sort(key=lambda item: item["bonus"], reverse=True)
        # At most two distinct rare facts contribute: the signal should break
        # dimension-score ties, not replace dimensional relevance.
        return min(0.55, sum(item["bonus"] for item in matches[:2])), matches[:4]

    def _precise_entity_bonus(self, payload: Dict[str, Any], terms: List[str]) -> tuple[float, List[Dict[str, Any]]]:
        """Small reranking bonus for validated exact named-entity matches.

        Terms originate only from the strict entity registry, whose entity type,
        textual provenance, and corpus frequency were checked at load time.
        This branch never expands an arbitrary n-gram into a named entity.
        """
        if not terms:
            return 0.0, []
        title = str(payload.get("doc_title", "")) + "\n" + str(payload.get("chunk_gen_title", ""))
        body = str(payload.get("chunk_text_full", ""))
        matches = []
        for term in terms:
            if term not in title and term not in body:
                continue
            df = self._anchor_document_frequency(term)
            rarity = math.log((len(self.points) + 1) / (df + 1)) / math.log(len(self.points) + 1)
            in_title = term in title
            bonus = 0.08 + 0.10 * rarity + (0.10 if in_title else 0.0)
            matches.append({"term": term, "document_frequency": df, "in_title": in_title,
                            "bonus": round(bonus, 6)})
        matches.sort(key=lambda item: item["bonus"], reverse=True)
        return min(0.28, sum(item["bonus"] for item in matches[:2])), matches[:3]

    def _entity_anchor_candidates(self, analysis: Dict[str, Any], fact_terms: List[str],
                                  existing_ids: set[str], entity_index_ids: set[str],
                                  limit: int = 6) -> List[Dict[str, Any]]:
        """Add a small number of exact, entity-scoped candidates.

        Dimension retrieval can omit a correct chunk when its extracted tag is
        incomplete.  This branch is deliberately narrower than generic lexical
        retrieval: it only operates inside a resolved scenic-area scope and
        requires an exact child-POI or rare named-fact match in the chunk.
        """
        if not bool(getattr(self.settings, "entity_anchor_candidate_recall_enabled", False)):
            return []
        scopes = analysis.get("spot_names", []) or []
        if not scopes:
            return []

        terms: Dict[str, str] = {}
        for entity in analysis.get("resolved_poi_entities", []) or []:
            # A scenic-area name is already applied as the scope filter. A
            # child POI, however, is discriminative enough to retrieve chunks.
            if str(entity.get("entity_type", "poi")) == "scenic_area":
                continue
            for value in (entity.get("mention", ""), entity.get("canonical_name", "")):
                value = str(value).strip()
                if len(value) >= 2:
                    terms[value] = "poi"
        # N-gram fact anchors are admitted only when both rare and at least
        # three characters long; this prevents broad phrases from opening a
        # lexical recall path across all chunks in the scenic area.
        for term in fact_terms:
            if len(term) >= 3 and self._anchor_document_frequency(term) <= 6:
                terms.setdefault(term, "fact")
        if not terms:
            return []

        candidates = []
        for point in self.points:
            payload = point.get("payload", {})
            chunk_id = str(payload.get("chunk_id") or point.get("id") or "")
            if not chunk_id or chunk_id in existing_ids or not self._matches_spot(payload, scopes):
                continue
            title = str(payload.get("doc_title", "")) + "\n" + str(payload.get("chunk_gen_title", ""))
            body = str(payload.get("chunk_text_full", ""))
            matched = []
            for term, kind in terms.items():
                if term not in title and term not in body:
                    continue
                df = self._anchor_document_frequency(term)
                rarity = math.log((len(self.points) + 1) / (df + 1)) / math.log(len(self.points) + 1)
                in_title = term in title
                matched.append({
                    "term": term, "kind": kind, "document_frequency": df,
                    "in_title": in_title,
                    "bonus": round(0.10 + 0.26 * rarity + (0.08 if in_title else 0.0), 6),
                })
            if chunk_id in entity_index_ids:
                # A structured entity/alias match is valid even when the
                # query uses an alias whose exact surface form is not repeated
                # in this chunk. The entity inverted index supplies the link.
                matched.append({
                    "term": "entity_index", "kind": "entity_index", "document_frequency": 1,
                    "in_title": False, "bonus": 0.34,
                })
            if not matched:
                continue
            matched.sort(key=lambda item: (item["kind"] == "poi", item["in_title"], item["bonus"]), reverse=True)
            # Keep the score below a clear main-dimension match while making a
            # precise POI/fact candidate competitive with weak auxiliary-only
            # matches. This is a candidate-union score, not a new dimension.
            best = matched[0]
            strict_entity_mode = bool(getattr(self.settings, "high_precision_entity_anchor_enabled", False))
            # In strict mode the entity route is a recall safety net, not a
            # replacement for an already well-ranked dimension candidate.
            score = (0.58 + min(0.18, best["bonus"])) if strict_entity_mode else (
                0.85 + best["bonus"] + (0.15 if best["kind"] in {"poi", "entity_index"} else 0.0)
            )
            candidates.append({
                "chunk_id": chunk_id, "score": round(score, 6),
                "source": "entity_anchor", **payload,
                "matched_dimensions": [], "matches": [],
                "entity_anchor_matches": matched[:4],
                "entity_anchor_recall_bonus": round(score, 6),
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def _fact_index_matches(self, query_facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if not bool(getattr(self.settings, "fact_index_candidate_recall_enabled", True)):
            return {}
        return self.fact_index.matches(query_facts)

    def _fact_index_candidates(self, analysis: Dict[str, Any], fact_matches: Dict[str, List[Dict[str, Any]]],
                               existing_ids: set[str], limit: int = 8) -> List[Dict[str, Any]]:
        """Union verified fact hits as a bounded, low-weight recall route."""
        if not fact_matches:
            return []
        candidates = []
        point_by_chunk = {
            str(point.get("payload", {}).get("chunk_id") or point.get("id")): point
            for point in self.points
        }
        for chunk_id, matches in fact_matches.items():
            if chunk_id in existing_ids:
                continue
            point = point_by_chunk.get(str(chunk_id))
            if not point:
                continue
            payload = point.get("payload", {})
            if not self._matches_spot(payload, analysis.get("spot_names", []) or []):
                continue
            # Exact subject+relation is the admission criterion.  Confidence
            # only breaks ties between already validated source facts.
            confidence = max((float(item.get("fact", {}).get("confidence", 0.0))
                              for item in matches), default=0.0)
            score = 0.61 + min(0.11, 0.11 * max(0.0, min(confidence, 1.0)))
            candidates.append({
                "chunk_id": str(chunk_id), "score": round(score, 6), "source": "verified_fact_index",
                **payload, "matched_dimensions": [], "matches": [],
                "verified_fact_matches": matches[:3],
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    @staticmethod
    def _dimension_policy(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wide recall: use main/secondary dimensions for ranking, not filtering."""
        constraints = analysis.get("constraints", {}) or {}
        main = next((leaf_id for leaf_id, item in constraints.items() if item.get("role") == "main"), "")
        secondary = sorted(leaf_id for leaf_id, item in constraints.items() if item.get("role") == "secondary")
        return {
            "mode": "wide_recall_main_dimension_rerank",
            "main_dimension": main,
            "secondary_dimensions": secondary,
            "required_dimensions": [],
            "required_minimum": 0,
            "scope": "parent_document",
        }

    @staticmethod
    def _passes_dimension_policy(policy: Dict[str, Any], matched_dims: set) -> bool:
        required = set(policy.get("required_dimensions", []))
        return not required or required.issubset(matched_dims)

    @staticmethod
    def _details(payload: Dict[str, Any], leaf_id: str) -> List[Dict[str, Any]]:
        details = (payload.get("tag_details") or {}).get(leaf_id, [])
        return details if isinstance(details, list) else []

    def _tag_vector_threshold(self, leaf_id: str) -> float:
        profile = str(getattr(self.settings, "tag_vector_threshold_profile", "global_058"))
        return float(TAG_VECTOR_THRESHOLD_PROFILES.get(profile, {}).get(leaf_id, 0.58))

    def _dimension_score(self, query: str, query_vec: List[float], payload: Dict[str, Any],
                         analysis: Dict[str, Any],
                         dimension_query_vecs: Dict[str, List[float]] | None = None,
                         tag_similarity_cache: Dict[str, float] | None = None,
                         tag_vector_inputs: Dict[tuple[int, str], List[tuple[str, str]]] | None = None
                         ) -> tuple[float, List[Dict[str, Any]], set]:
        by_parent: Dict[str, List[float]] = {}
        matches: List[Dict[str, Any]] = []
        matched_dims = set()
        tags_by_dim = {}
        for leaf_id in analysis["constraints"]:
            values = payload.get(f"dim_{leaf_id}", [])
            if not isinstance(values, list):
                values = [values]
            tags_by_dim[leaf_id] = [str(value) for value in values if str(value).strip()]

        for leaf_id, constraint in analysis["constraints"].items():
            doc_tags = tags_by_dim.get(leaf_id, [])
            if not doc_tags:
                continue
            q_labels = constraint.get("labels", [])
            intent_terms = constraint.get("intent_terms", [])
            query_concepts = self.label_resolver.concepts(leaf_id, q_labels)
            doc_concepts = self.label_resolver.concepts(leaf_id, doc_tags)
            overlap = query_concepts.intersection(doc_concepts)
            match_mode = str(constraint.get("match", "ANY")).upper()
            labels_match = (
                bool(overlap) if match_mode != "ALL"
                else bool(query_concepts) and query_concepts.issubset(doc_concepts)
            )
            intent = [term for term in intent_terms if term and term in payload.get("chunk_text_full", "")]
            best = 1.0 if labels_match else (0.78 if intent else 0.0)
            best_hits = [
                label for label in doc_tags
                if self.label_resolver.canonical(leaf_id, label) in overlap
            ] if labels_match else (intent or [])
            role = str(constraint.get("role", "auxiliary"))
            # Vector matching is reserved for the explicitly routed property;
            # using it on every auxiliary noun is both noisy and expensive.
            routed_query_vec = (dimension_query_vecs or {}).get(leaf_id, query_vec)
            if best <= 0 and role in {"main", "secondary"} and routed_query_vec is not None and (q_labels or intent_terms):
                cached_inputs = (tag_vector_inputs or {}).get((id(payload), leaf_id))
                tag_inputs = cached_inputs or [
                    (tag, self._tag_vector_key(
                        leaf_id, tag, self._tag_evidence(payload, leaf_id, tag)
                    ))
                    for tag in doc_tags
                ]
                scores = []
                scored_tags = []
                for tag, text in tag_inputs:
                    if text not in self.tag_vector_cache:
                        continue
                    if tag_similarity_cache is not None and text in tag_similarity_cache:
                        score = tag_similarity_cache[text]
                    else:
                        score = _dot_unit_vectors(routed_query_vec, self.tag_vector_cache[text])
                        if tag_similarity_cache is not None:
                            tag_similarity_cache[text] = score
                    scores.append(score)
                    scored_tags.append(tag)
                if scores and max(scores) >= self._tag_vector_threshold(leaf_id):
                    best = max(scores)
                    best_hits = [scored_tags[scores.index(best)]]
            role_weight = {"main": 3.0, "secondary": 1.25, "auxiliary": 0.3}.get(role, 0.3)
            # Set iteration order is randomized across Python processes.  A
            # deterministic concept choice is required for reproducible
            # evaluation and for fair A/B comparisons of rerankers.
            concept_for_idf = sorted(overlap)[0] if overlap else ""
            if not concept_for_idf and doc_concepts:
                concept_for_idf = sorted(doc_concepts)[0]
            df = self.concept_df.get(leaf_id, {}).get(concept_for_idf, len(self.points))
            idf_bonus = 1.0 + 0.25 * math.log((len(self.points) + 1) / (df + 1)) / math.log(len(self.points) + 1)
            weighted = best * role_weight * idf_bonus
            if best > 0:
                matched_dims.add(leaf_id)
                parent_id = self.maps["nodes"][leaf_id].get("parent_id") or leaf_id
                by_parent.setdefault(parent_id, []).append(weighted)
                detail_map = {normalize_label(item.get("label", "")): item for item in self._details(payload, leaf_id)}
                for hit_label in dict.fromkeys(best_hits):
                    detail = detail_map.get(normalize_label(hit_label), {})
                    matches.append({
                        "dimension_id": leaf_id,
                        "dimension_path": self.dimension_paths.get(leaf_id, leaf_id),
                        "label": hit_label,
                        "evidence": detail.get("evidence", ""),
                        "similarity": round(best, 6),
                        "role": role,
                        "idf_bonus": round(idf_bonus, 6),
                        "source": "canonical_label" if labels_match else ("intent_text" if intent else "dimension_vector"),
                    })
        # 同一父维度只取最大叶子分数；不同父维度可以累加。
        score = sum(max(values) for values in by_parent.values())
        return score, matches, matched_dims

    def _batch_tag_similarities(
        self,
        candidate_points: List[Dict[str, Any]],
        constraints: Dict[str, Dict[str, Any]],
        dimension_query_vecs: Dict[str, List[float]],
        query_vec: List[float],
        spot_names: List[str],
    ) -> tuple[Dict[str, float], Dict[tuple[int, str], List[tuple[str, str]]]]:
        """Compute routed tag similarities in a few matrix operations per query.

        The previous scalar path called ``numpy.dot`` once per tag.  On a
        corpus with hundreds of thousands of evidence-tag vectors, that Python
        call overhead dominated retrieval.  Gather each query's candidate tag
        rows, perform one matrix-vector product per routed leaf, then let the
        normal scorer consume the same exact cosine values.
        """
        if np is None or self.tag_vector_matrix is None or not self.tag_vector_index:
            return {}, {}

        vectors_by_leaf: Dict[str, Dict[str, int]] = {}
        inputs_by_payload: Dict[tuple[int, str], List[tuple[str, str]]] = {}
        query_vectors = {}
        for leaf_id, constraint in constraints.items():
            if constraint.get("role") not in {"main", "secondary"}:
                continue
            if not (constraint.get("labels") or constraint.get("intent_terms")):
                continue
            routed_vec = dimension_query_vecs.get(leaf_id, query_vec)
            if routed_vec is None:
                continue
            query_vectors[leaf_id] = np.asarray(routed_vec, dtype=np.float32)
            vectors_by_leaf[leaf_id] = {}

        if not vectors_by_leaf:
            return {}, inputs_by_payload

        for point in candidate_points:
            payload = point.get("payload", {})
            if not self._matches_spot(payload, spot_names):
                continue
            for leaf_id, keys_to_rows in vectors_by_leaf.items():
                values = payload.get(f"dim_{leaf_id}", [])
                if not isinstance(values, list):
                    values = [values]
                tag_inputs = []
                for value in values:
                    tag = str(value)
                    if not tag.strip():
                        continue
                    cache_key = self._tag_vector_key(
                        leaf_id, tag, self._tag_evidence(payload, leaf_id, tag)
                    )
                    tag_inputs.append((tag, cache_key))
                    row_index = self.tag_vector_index.get(cache_key)
                    if row_index is not None:
                        keys_to_rows.setdefault(cache_key, row_index)
                if tag_inputs:
                    inputs_by_payload[(id(payload), leaf_id)] = tag_inputs

        similarities: Dict[str, float] = {}
        for leaf_id, keys_to_rows in vectors_by_leaf.items():
            if not keys_to_rows:
                continue
            keys = list(keys_to_rows)
            row_indices = [keys_to_rows[key] for key in keys]
            matrix = np.asarray(self.tag_vector_matrix[row_indices], dtype=np.float32)
            scores = matrix @ query_vectors[leaf_id]
            similarities.update(
                (key, float(score)) for key, score in zip(keys, scores)
            )
        return similarities, inputs_by_payload

    @staticmethod
    def _normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [float(item.get("score", 0.0)) for item in results]
        low, high = min(scores), max(scores)
        if high == low:
            return {str(item["chunk_id"]): 1.0 for item in results}
        return {str(item["chunk_id"]): (float(item.get("score", 0.0)) - low) / (high - low)
                for item in results}

    def _lexical_fallback_terms(self, query: str) -> List[str]:
        """Return conservative literal terms for the no-candidate route only."""
        stop_chars = set("的了是在有和与或把被给对从到去来能会想问我你他她它们这那哪什么怎么吗呢吧啊")
        terms = set()
        for segment in re.findall(r"[\u4e00-\u9fff]{1,}", query):
            for size in range(1, min(5, len(segment)) + 1):
                for start in range(0, len(segment) - size + 1):
                    value = segment[start:start + size]
                    if len(value) == 1 and value in stop_chars:
                        continue
                    terms.add(value)
        max_df = max(15, math.ceil(len(self.points) * 0.15))
        return sorted(
            (term for term in terms if 0 < self._anchor_document_frequency(term) <= max_df),
            key=lambda term: (-len(term), self._anchor_document_frequency(term), term),
        )[:30]

    def _no_candidate_lexical_fallback(self, query: str, analysis: Dict[str, Any],
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """Bounded BM25-style literal recall for context-poor empty queries.

        It is deliberately unavailable when the dimension route has candidates,
        preventing sparse lexical signals from disturbing normal dimension
        ranking.  When a scenic scope is resolved, the same scope constraint is
        retained; otherwise only a few high-IDF multi-term matches are allowed.
        """
        if not bool(getattr(self.settings, "no_candidate_lexical_fallback_enabled", False)):
            return []
        terms = self._lexical_fallback_terms(query)
        if not terms:
            return []
        scopes = analysis.get("spot_names", []) or []
        candidates = []
        for point in self.points:
            payload = point.get("payload", {})
            if not self._matches_spot(payload, scopes):
                continue
            title = str(payload.get("doc_title", "")) + "\n" + str(payload.get("chunk_gen_title", ""))
            body = str(payload.get("chunk_text_full", ""))
            text = title + "\n" + body
            matched = []
            score = 0.0
            for term in terms:
                count = text.count(term)
                if not count:
                    continue
                df = self._anchor_document_frequency(term)
                idf = math.log((len(self.points) + 1) / (df + 1))
                weight = (1.0 + 0.35 * min(len(term), 4)) * idf * min(count, 3)
                if term in title:
                    weight *= 1.25
                score += weight
                matched.append({"term": term, "document_frequency": df,
                                "in_title": term in title, "count": count})
            # One generic character is never sufficient. Require either two
            # independent signals or a rare phrase of at least two characters.
            if len(matched) < 2 and not any(len(item["term"]) >= 2 and item["document_frequency"] <= 4 for item in matched):
                continue
            if not matched:
                continue
            matched.sort(key=lambda item: (-len(item["term"]), item["document_frequency"], item["term"]))
            candidates.append({
                "chunk_id": payload.get("chunk_id", point.get("id")),
                "score": score, "source": "no_candidate_lexical_fallback", **payload,
                "matched_dimensions": [], "matches": [],
                "lexical_fallback_matches": matched[:8],
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            return []
        top = candidates[0]["score"] or 1.0
        for item in candidates:
            # Keep fallback scores visibly below a regular dimension match.
            item["score"] = round(0.42 + 0.24 * (item["score"] / top), 6)
        return candidates[:limit]

    def search(self, query: str, top_k: int | None = None,
               semantic_pool: int | None = None,
               dimension_pool: int | None = None,
               dimension_only: bool = False,
               query_facts: List[Dict[str, Any]] | None = None,
               original_query: str | None = None,
               subqueries: List[str] | None = None,
               save_snapshot: bool = False) -> Dict[str, Any]:
        vector_subqueries = [
            str(item) for item in (subqueries or []) if str(item).strip()
        ]
        if vector_subqueries:
            query = " | ".join(vector_subqueries)
        else:
            vector_subqueries = [query]
        top_k = top_k or self.settings.top_k
        semantic_pool = max(top_k, int(semantic_pool or getattr(self.settings, "semantic_pool", 20)))
        dimension_pool = max(top_k, int(dimension_pool or getattr(self.settings, "dimension_pool", 100)))
        analysis = self._parse_query(query)
        verified_fact_matches = self._fact_index_matches(query_facts or [])
        analysis["query_fact_count"] = len(query_facts or [])
        analysis["verified_fact_match_count"] = sum(
            len(value) for value in verified_fact_matches.values()
        )
        dimension_query_texts = {
            leaf_id: self._dimension_query_text(query, leaf_id, constraint)
            for leaf_id, constraint in analysis["constraints"].items()
            if constraint.get("role") in {"main", "secondary"}
            and (constraint.get("labels") or constraint.get("intent_terms"))
        }
        vector_inputs = [query] + list(dimension_query_texts.values())
        encoded_vectors = self.embeddings.encode(vector_inputs)
        query_vec = _unit_vector(encoded_vectors[0])
        dimension_query_vecs = {
            leaf_id: _unit_vector(vector)
            for (leaf_id, _), vector in zip(
                dimension_query_texts.items(), encoded_vectors[1:]
            )
        }
        dimension_policy = self._dimension_policy(analysis)
        fact_anchor_terms = (
            self._fact_anchor_terms(query, analysis)
            if bool(getattr(self.settings, "fact_anchor_rerank_enabled", True)) else []
        )
        analysis["fact_anchor_terms"] = fact_anchor_terms
        semantic = [] if dimension_only else self.vector_retriever.search(
            vector_subqueries,
            semantic_pool,
            original_query=original_query or query,
            query_vector=query_vec,
            spot_names=analysis["spot_names"],
        )

        dimension = []
        # DuRetrieval has no dimension/facet annotations in this run.  When
        # every posting list is empty, scanning every point for every query is
        # both wasteful and misleading: it cannot produce a dimension hit.
        # Keep the route explicitly empty and let fusion reduce to semantic
        # candidates instead of spending O(num_queries * num_points) time.
        if any(self.tags_by_dim.values()):
            scored_points = []
            group_dimensions: Dict[str, set] = {}
            candidate_points = {}
            for leaf_id in analysis["constraints"]:
                for point in self.dimension_points_by_leaf.get(leaf_id, []):
                    candidate_points.setdefault(str(point.get("id", id(point))), point)
            tag_similarity_cache, tag_vector_inputs = self._batch_tag_similarities(
                list(candidate_points.values()), analysis["constraints"],
                dimension_query_vecs, query_vec, analysis["spot_names"]
            )
            for point in candidate_points.values():
                payload = point.get("payload", {})
                if not self._matches_spot(payload, analysis["spot_names"]):
                    continue
                score, matches, matched_dims = self._dimension_score(
                    query, query_vec, payload, analysis,
                    dimension_query_vecs, tag_similarity_cache, tag_vector_inputs
                )
                if score <= 0:
                    continue
                fact_anchor_bonus, fact_anchor_matches = self._fact_anchor_bonus(payload, fact_anchor_terms)
                entity_anchor_bonus, entity_anchor_matches = self._precise_entity_bonus(payload, analysis.get("precise_entity_terms", []))
                group_id = str(payload.get("parent_doc_id") or payload.get("doc_id") or
                               payload.get("chunk_id") or point.get("id"))
                group_dimensions.setdefault(group_id, set()).update(matched_dims)
                scored_points.append((group_id, point, score + fact_anchor_bonus + entity_anchor_bonus, matches, matched_dims,
                                      fact_anchor_bonus, fact_anchor_matches, entity_anchor_bonus, entity_anchor_matches))
            for group_id, point, score, matches, matched_dims, fact_anchor_bonus, fact_anchor_matches, entity_anchor_bonus, entity_anchor_matches in scored_points:
                payload = point.get("payload", {})
                group_matched = group_dimensions[group_id]
                main = str(dimension_policy.get("main_dimension", ""))
                secondary = set(dimension_policy.get("secondary_dimensions", []))
                # Parent-level evidence is a ranking bonus only. It allows sibling
                # chunks to reinforce a fact without excluding a candidate whose
                # tag was missed by the extractor.
                coverage_bonus = 0.20 * int(bool(main and main in group_matched))
                coverage_bonus += 0.06 * len(secondary.intersection(group_matched))
                coverage_bonus += 0.01 * len(group_matched - {main} - secondary)
                dimension.append({"chunk_id": payload.get("chunk_id", point.get("id")),
                                  "score": score + coverage_bonus, "source": "dimension", **payload,
                                  "matched_dimensions": sorted(matched_dims),
                                  "match_group": group_id, "matches": matches})
                dimension[-1]["fact_anchor_bonus"] = round(fact_anchor_bonus, 6)
                dimension[-1]["fact_anchor_matches"] = fact_anchor_matches
                dimension[-1]["precise_entity_bonus"] = round(entity_anchor_bonus, 6)
                dimension[-1]["precise_entity_matches"] = entity_anchor_matches
            dimension.sort(key=lambda item: item["score"], reverse=True)
            dimension = dimension[:dimension_pool]
        indexed_entity_ids = set(self.auto_entity_registry.candidate_chunk_ids(query)) if bool(
            getattr(self.settings, "auto_entity_registry_enabled", False)
        ) else set()
        if bool(getattr(self.settings, "high_precision_entity_anchor_enabled", False)):
            indexed_entity_ids.update(self.precise_entity_registry.candidate_chunk_ids(query))
        anchor_candidates = self._entity_anchor_candidates(
            analysis, fact_anchor_terms,
            {str(item.get("chunk_id", "")) for item in dimension},
            indexed_entity_ids,
        )
        if anchor_candidates:
            dimension.extend(anchor_candidates)
            dimension.sort(key=lambda item: item["score"], reverse=True)
            dimension = dimension[:dimension_pool]
        analysis["entity_anchor_candidate_count"] = len(anchor_candidates)
        fact_index_candidates = self._fact_index_candidates(
            analysis, verified_fact_matches,
            {str(item.get("chunk_id", "")) for item in dimension},
        )
        if fact_index_candidates:
            tail_start = max(0, int(getattr(self.settings, "fact_index_tail_insertion_rank", 10)))
            if tail_start:
                # Preserve high-confidence dimension ranks.  Verified facts
                # are a recall supplement, so they are deliberately placed
                # after the base head rather than allowed to displace it.
                dimension = (
                    dimension[:tail_start]
                    + fact_index_candidates
                    + dimension[tail_start:]
                )
            else:
                dimension.extend(fact_index_candidates)
                dimension.sort(key=lambda item: item["score"], reverse=True)
            dimension = dimension[:dimension_pool]
        analysis["verified_fact_index_candidate_count"] = len(fact_index_candidates)
        lexical_fallback_candidates = []
        if not dimension:
            lexical_fallback_candidates = self._no_candidate_lexical_fallback(query, analysis)
            dimension = lexical_fallback_candidates
        analysis["no_candidate_lexical_fallback_count"] = len(lexical_fallback_candidates)
        for rank, item in enumerate(dimension, 1):
            item["rank"] = rank

        fused = adaptive_fusion(semantic, dimension, top_k=top_k)
        fusion_candidates = fused["fusion_candidates"]
        fusion = fused["fusion_results"]
        if save_snapshot:
            _save_fusion_snapshot(
                semantic,
                dimension,
                fusion_candidates,
                dim_alpha=self.settings.dim_alpha,
                top_k=top_k,
                query=query,
                original_query=original_query or query,
                query_analysis=analysis,
            )
        return {
            "query": query, "query_analysis": analysis,
            "dimension_policy": dimension_policy,
            "constraints": analysis["constraints"],
            "fact_anchor_terms": fact_anchor_terms,
            "query_facts": query_facts or [],
            "semantic_candidates": semantic,
            "dimension_candidates": dimension,
            "fusion_candidates": fusion_candidates,
            "fusion_strategy": fused["fusion_strategy"],
            "semantic_results": semantic[:top_k],
            "dimension_results": dimension[:top_k],
            "fusion_results": fusion, "top_chunks": fusion,
        }
