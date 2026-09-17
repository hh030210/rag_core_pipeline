"""维度检索、语义检索和归一化融合。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .schema_v2 import dimension_path, load_schema, normalize_label, normalize_text, schema_maps
from .storage import VectorStore


class Retriever:
    def __init__(self, *, run_dir: str | Path, settings, embeddings, llm=None):
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.embeddings = embeddings
        self.llm = llm
        self.schema = load_schema(self.run_dir / "V_core_v2.json", allow_legacy=True, strict=False)
        self.maps = schema_maps(self.schema)
        self.store = VectorStore(run_dir=self.run_dir, backend=settings.backend,
                                 qdrant_url=settings.qdrant_url,
                                 collection=settings.collection, vector_dim=settings.vector_dim)
        raw_index = self.run_dir / "inverted_index_v2.json"
        self.postings = json.loads(raw_index.read_text(encoding="utf-8")) if raw_index.exists() else {}
        self.points = self.store.scroll()
        self.tags_by_dim = {leaf_id: set((self.postings.get(leaf_id) or {}).keys())
                            for leaf_id in self.maps["leaves"]}

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"[\s\u3000，。！？；：、（）()【】\[\]{}‘’“”\"'·_\-]+", "", str(value or "").lower())

    def _parse_query(self, query: str) -> Dict[str, Any]:
        constraints: Dict[str, Dict[str, Any]] = {}
        normalized_query = self._norm(query)

        # 先用已有标签词表做严格词匹配，最长标签优先。
        for leaf_id in self.maps["leaves"]:
            labels = sorted(self.tags_by_dim.get(leaf_id, set()), key=len, reverse=True)
            for label in labels:
                if len(normalize_label(label)) >= 2 and self._norm(label) in normalized_query:
                    item = constraints.setdefault(leaf_id, {"labels": [], "intent_terms": [], "match": "ANY"})
                    if label not in item["labels"]:
                        item["labels"].append(label)
        # 再让 v2 LLM parser 识别自然语言意图；失败时保留严格匹配结果。
        if self.llm and not self.settings.mock:
            try:
                leaves = [self.maps["nodes"][leaf_id] for leaf_id in self.maps["leaves"]]
                parsed = self.llm.parse_query_intent_v4(query, leaves)
                for item in parsed.get("constraints", []) if isinstance(parsed, dict) else []:
                    leaf_id = str(item.get("dimension_id", ""))
                    if leaf_id not in self.maps["leaves"]:
                        continue
                    target = constraints.setdefault(leaf_id, {"labels": [], "intent_terms": [], "match": "ANY"})
                    for key in ("labels", "intent_terms"):
                        for value in item.get(key, []) or []:
                            value = normalize_text(value)
                            if value and value not in target[key]:
                                target[key].append(value)
            except Exception as exc:
                print(f"[提示] 查询解析失败，使用确定性标签解析: {exc}")

        # 维度名/别名触发意图槽，避免“几点开门、怎么预约”等问题没有标签时为空。
        cue_map = {
            "opening": "开放 开门 关门 营业 几点 时间".split(),
            "ticket": "门票 票价 预约 预订 购票 收费 免费".split(),
            "ticketing": "门票 票价 预约 预订 购票 收费 免费".split(),
            "location": "位置 位于 哪里 地址 地理".split(),
            "history": "历史 始建 修建 建造 朝代 年代 由来 多久".split(),
            "route": "交通 路线 公交 地铁 自驾 怎么去 到达".split(),
            "service": "服务 设施 厕所 餐厅 讲解 寄存".split(),
            "seasonal": "季节 春季 夏季 秋季 冬季 最佳时间".split(),
        }
        for leaf_id in self.maps["leaves"]:
            if leaf_id in constraints:
                continue
            leaf_name = self.maps["nodes"][leaf_id]["name"]
            suffix = leaf_id.split(".")[-1]
            cues = cue_map.get(suffix, [])
            if any(cue in query for cue in cues) or leaf_name in query:
                terms = [cue for cue in cues if cue in query]
                constraints[leaf_id] = {"labels": [], "intent_terms": terms, "match": "ANY"}

        # 查询父维度时展开所有叶子，但在打分时按父维度取最大值，避免子节点数量带来额外分数。
        for parent_id, children in self.maps["children"].items():
            parent = self.maps["nodes"].get(parent_id, {})
            aliases = [parent.get("name", "")] + list(parent.get("aliases", []) or [])
            if any(alias and alias in query for alias in aliases):
                for child_id in children:
                    constraints.setdefault(child_id, {"labels": [], "intent_terms": [], "match": "ANY"})

        spot_names = sorted({str(point.get("payload", {}).get("spot_name", "")).strip()
                             for point in self.points if str(point.get("payload", {}).get("spot_name", "")).strip()}, key=len, reverse=True)
        matched_spots = [spot for spot in spot_names if spot in query]
        return {
            "constraints": constraints,
            "active_dimensions": list(constraints),
            "spot_names": matched_spots,
            "confidence": min(1.0, 0.25 + 0.15 * len(constraints)) if constraints else 0.0,
        }

    @staticmethod
    def _details(payload: Dict[str, Any], leaf_id: str) -> List[Dict[str, Any]]:
        details = (payload.get("tag_details") or {}).get(leaf_id, [])
        return details if isinstance(details, list) else []

    def _dimension_score(self, query: str, query_vec: List[float], payload: Dict[str, Any],
                         analysis: Dict[str, Any]) -> tuple[float, List[Dict[str, Any]], set]:
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
            best = 0.0
            best_hits = []
            for doc_label in doc_tags:
                doc_norm = normalize_label(doc_label)
                exact = [label for label in q_labels if normalize_label(label) == doc_norm or
                         normalize_label(label) in doc_norm or doc_norm in normalize_label(label)]
                intent = [term for term in intent_terms if term and term in payload.get("chunk_text_full", "")]
                score = 1.0 if exact else (0.78 if intent else 0.0)
                if score > best:
                    best = score
                    best_hits = exact or intent or [doc_label]
                elif score == best and score > 0:
                    best_hits.extend(exact or intent or [doc_label])
            if best <= 0 and (q_labels or intent_terms):
                # 同维度语义兜底：使用 query 与标签短语的向量相似度，且不跨维度。
                tag_vectors = self.embeddings.encode([f"{leaf_id}:{tag}" for tag in doc_tags])
                scores = [self.store._cosine(query_vec, vector) for vector in tag_vectors]
                if scores and max(scores) >= 0.58:
                    best = max(scores)
                    best_hits = [doc_tags[scores.index(best)]]
            if best > 0:
                matched_dims.add(leaf_id)
                parent_id = self.maps["nodes"][leaf_id].get("parent_id") or leaf_id
                by_parent.setdefault(parent_id, []).append(best)
                detail_map = {normalize_label(item.get("label", "")): item for item in self._details(payload, leaf_id)}
                for hit_label in dict.fromkeys(best_hits):
                    detail = detail_map.get(normalize_label(hit_label), {})
                    matches.append({
                        "dimension_id": leaf_id,
                        "dimension_path": dimension_path(self.schema, leaf_id),
                        "label": hit_label,
                        "evidence": detail.get("evidence", ""),
                        "similarity": round(best, 6),
                        "source": "exact" if q_labels and best >= 1.0 else "intent_or_vector",
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
               dimension_pool: int | None = None) -> Dict[str, Any]:
        top_k = top_k or self.settings.top_k
        semantic_pool = max(top_k, int(semantic_pool or getattr(self.settings, "semantic_pool", 20)))
        dimension_pool = max(top_k, int(dimension_pool or getattr(self.settings, "dimension_pool", 100)))
        analysis = self._parse_query(query)
        query_vec = self.embeddings.encode([query])[0]
        semantic_hits = self.store.semantic_search(query_vec, semantic_pool)
        semantic = []
        for hit in semantic_hits:
            payload = hit["payload"]
            if analysis["spot_names"] and payload.get("spot_name") not in analysis["spot_names"]:
                continue
            semantic.append({"chunk_id": payload.get("chunk_id", hit["id"]),
                             "score": hit["score"], "source": "semantic", **payload})
        for rank, item in enumerate(semantic, 1):
            item["rank"] = rank

        dimension = []
        for point in self.points:
            payload = point.get("payload", {})
            if analysis["spot_names"] and payload.get("spot_name") not in analysis["spot_names"]:
                continue
            score, matches, matched_dims = self._dimension_score(query, query_vec, payload, analysis)
            if score <= 0:
                continue
            dimension.append({"chunk_id": payload.get("chunk_id", point.get("id")),
                              "score": score, "source": "dimension", **payload,
                              "matched_dimensions": sorted(matched_dims), "matches": matches})
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
            "constraints": analysis["constraints"],
            "semantic_candidates": semantic,
            "dimension_candidates": dimension,
            "fusion_candidates": fusion_candidates,
            "semantic_results": semantic[:top_k],
            "dimension_results": dimension[:top_k],
            "fusion_results": fusion, "top_chunks": fusion,
        }
