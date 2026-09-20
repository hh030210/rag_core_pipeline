"""维度检索、语义检索和归一化融合。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from .dimension_labels import CanonicalLabelResolver
from .poi_registry import PoiRegistry
from .schema_v2 import load_schema, normalize_label, normalize_text, schema_maps
from .storage import VectorStore


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
    numeric = [float(value) for value in values]
    norm = math.sqrt(math.fsum(value * value for value in numeric))
    return [value / norm for value in numeric] if norm > 0 else numeric


def _dot_unit_vectors(left: List[float], right: List[float]) -> float:
    """One-pass dot product for vectors normalized at cache time."""
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
        raw_index = self.run_dir / "inverted_index_v2.json"
        self.postings = json.loads(raw_index.read_text(encoding="utf-8")) if raw_index.exists() else {}
        self.points = self.store.scroll()
        self.tags_by_dim = {leaf_id: set((self.postings.get(leaf_id) or {}).keys())
                            for leaf_id in self.maps["leaves"]}
        self.source_spot_names = sorted({
            str(point.get("payload", {}).get("source_file", "")).split("-", 1)[0].strip()
            for point in self.points
            if str(point.get("payload", {}).get("source_file", "")).strip()
        }, key=len, reverse=True)
        # ``spot_name`` is empty in the current corpus, while questions often
        # name a child POI.  Resolve those aliases to the source-file-level
        # scenic area before candidate filtering.
        self.poi_registry = PoiRegistry(self.source_spot_names)
        self._anchor_df_cache: Dict[str, int] = {}
        self.label_resolver = CanonicalLabelResolver(self.run_dir, self.tags_by_dim)
        self.concept_df: Dict[str, Dict[str, int]] = {}
        tag_inputs, seen_tag_inputs = [], set()
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
                    text = f"{leaf_id}:{value}"
                    if str(value).strip() and text not in seen_tag_inputs:
                        tag_inputs.append(text)
                        seen_tag_inputs.add(text)
        tag_vectors = self.embeddings.encode(tag_inputs) if tag_inputs else []
        self.tag_vector_cache = {
            text: _unit_vector(vector) for text, vector in zip(tag_inputs, tag_vectors)
        }
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
        # Preserve direct matching for new scenic areas not yet listed in the
        # registry; aliases and child POIs contribute their parent scope.
        matched_spots = list(dict.fromkeys(
            (self.poi_registry.scenic_scopes(query) if poi_scope_enabled else [])
            + [spot for spot in direct_names if spot in query]
        ))
        return {
            "constraints": constraints,
            "active_dimensions": list(constraints),
            "spot_names": matched_spots,
            "resolved_poi_entities": resolved_entities,
            "confidence": min(1.0, 0.25 + 0.15 * len(constraints)) if constraints else 0.0,
            "routing": parsed.get("routing", {}) if self.query_parser and isinstance(parsed, dict) else {},
            "parser_diagnostics": parser_diagnostics,
        }

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
                term in (
                    str(point.get("payload", {}).get("doc_title", ""))
                    + "\n" + str(point.get("payload", {}).get("chunk_gen_title", ""))
                    + "\n" + str(point.get("payload", {}).get("chunk_text_full", ""))
                )
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
                         tag_similarity_cache: Dict[str, float] | None = None) -> tuple[float, List[Dict[str, Any]], set]:
        by_parent: Dict[str, List[float]] = {}
        matches: List[Dict[str, Any]] = []
        matched_dims = set()
        tags_by_dim = {}
        for leaf_id in self.maps["leaves"]:
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
            if best <= 0 and role in {"main", "secondary"} and query_vec and (q_labels or intent_terms):
                tag_inputs = [f"{leaf_id}:{tag}" for tag in doc_tags]
                scores = []
                for text in tag_inputs:
                    if text not in self.tag_vector_cache:
                        continue
                    if tag_similarity_cache is not None and text in tag_similarity_cache:
                        score = tag_similarity_cache[text]
                    else:
                        score = _dot_unit_vectors(query_vec, self.tag_vector_cache[text])
                        if tag_similarity_cache is not None:
                            tag_similarity_cache[text] = score
                    scores.append(score)
                if scores and max(scores) >= self._tag_vector_threshold(leaf_id):
                    best = max(scores)
                    best_hits = [doc_tags[scores.index(best)]]
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

    def search(self, query: str, top_k: int | None = None,
               semantic_pool: int | None = None,
               dimension_pool: int | None = None,
               dimension_only: bool = False) -> Dict[str, Any]:
        top_k = top_k or self.settings.top_k
        semantic_pool = max(top_k, int(semantic_pool or getattr(self.settings, "semantic_pool", 20)))
        dimension_pool = max(top_k, int(dimension_pool or getattr(self.settings, "dimension_pool", 100)))
        analysis = self._parse_query(query)
        # Dimension-only evaluation skips semantic *search*, but still encodes
        # the query for the same-dimension tag-vector fallback.
        query_vec = _unit_vector(self.embeddings.encode([query])[0]) if analysis["constraints"] else []
        dimension_policy = self._dimension_policy(analysis)
        fact_anchor_terms = (
            self._fact_anchor_terms(query, analysis)
            if bool(getattr(self.settings, "fact_anchor_rerank_enabled", True)) else []
        )
        analysis["fact_anchor_terms"] = fact_anchor_terms
        semantic_hits = [] if dimension_only else self.store.semantic_search(query_vec, semantic_pool)
        semantic = []
        for hit in semantic_hits:
            payload = hit["payload"]
            if not self._matches_spot(payload, analysis["spot_names"]):
                continue
            semantic.append({"chunk_id": payload.get("chunk_id", hit["id"]),
                             "score": hit["score"], "source": "semantic", **payload})
        for rank, item in enumerate(semantic, 1):
            item["rank"] = rank

        scored_points = []
        group_dimensions: Dict[str, set] = {}
        tag_similarity_cache: Dict[str, float] = {}
        for point in self.points:
            payload = point.get("payload", {})
            if not self._matches_spot(payload, analysis["spot_names"]):
                continue
            score, matches, matched_dims = self._dimension_score(
                query, query_vec, payload, analysis, tag_similarity_cache
            )
            if score <= 0:
                continue
            fact_anchor_bonus, fact_anchor_matches = self._fact_anchor_bonus(payload, fact_anchor_terms)
            group_id = str(payload.get("parent_doc_id") or payload.get("doc_id") or
                           payload.get("chunk_id") or point.get("id"))
            group_dimensions.setdefault(group_id, set()).update(matched_dims)
            scored_points.append((group_id, point, score + fact_anchor_bonus, matches, matched_dims,
                                  fact_anchor_bonus, fact_anchor_matches))
        dimension = []
        for group_id, point, score, matches, matched_dims, fact_anchor_bonus, fact_anchor_matches in scored_points:
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
        dimension.sort(key=lambda item: item["score"], reverse=True)
        dimension = dimension[:dimension_pool]
        for rank, item in enumerate(dimension, 1):
            item["rank"] = rank

        sem_norm = self._normalize_scores(semantic)
        dim_norm = self._normalize_scores(dimension)
        semantic_by_id = {item["chunk_id"]: item for item in semantic}
        dimension_by_id = {item["chunk_id"]: item for item in dimension}
        by_id = {}
        for cid in dict.fromkeys([item["chunk_id"] for item in semantic + dimension]):
            item = dict(dimension_by_id.get(cid) or semantic_by_id[cid])
            if cid in semantic_by_id:
                item["sem_rank"] = semantic_by_id[cid].get("rank")
                item["semantic_score"] = semantic_by_id[cid].get("score")
            if cid in dimension_by_id:
                item["dim_rank"] = dimension_by_id[cid].get("rank")
                item["dimension_score"] = dimension_by_id[cid].get("score")
            by_id[cid] = item
        fused_scores = {}
        for cid in by_id:
            fused_scores[cid] = self.settings.dim_alpha * dim_norm.get(cid, 0.0) + (1 - self.settings.dim_alpha) * sem_norm.get(cid, 0.0)
        fusion_candidates = []
        for cid, score in sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True):
            item = by_id[cid]
            item["score"] = score
            item["final_score"] = score
            item["source"] = ("dimension" if cid in dim_norm else "") + ("+semantic" if cid in sem_norm else "")
            fusion_candidates.append(item)
        fusion = fusion_candidates[:top_k]
        return {
            "query": query, "query_analysis": analysis,
            "dimension_policy": dimension_policy,
            "constraints": analysis["constraints"],
            "fact_anchor_terms": fact_anchor_terms,
            "semantic_candidates": semantic,
            "dimension_candidates": dimension,
            "fusion_candidates": fusion_candidates,
            "semantic_results": semantic[:top_k],
            "dimension_results": dimension[:top_k],
            "fusion_results": fusion, "top_chunks": fusion,
        }
