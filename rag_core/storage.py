"""Qdrant 入库及本地文件后端。默认 Qdrant；local 仅用于验收和单机调试。"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from code_jyx.index_builder import build_qdrant_payload_v2
from .schema_v2 import SCHEMA_VERSION, schema_maps
from .entity_registry import extract_landmark_mentions


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def make_payload(record: Dict[str, Any], document_tags: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    # Qdrant 的 v2 数组标签、dimension_paths 和 tag_details 统一使用原始
    # code_jyx IndexBuilderV2 的 payload 实现；这里仅补充新项目 chunk 元数据。
    payload = build_qdrant_payload_v2(str(record["doc_id"]), document_tags, schema,
                                      include_empty=True)
    payload.update({
        "chunk_id": str(record["chunk_id"]),
        "doc_id": str(record["doc_id"]),
        "parent_doc_id": str(record.get("parent_doc_id", "")),
        "doc_title": str(record.get("doc_title", "")),
        "chunk_gen_title": str(record.get("chunk_gen_title", "")),
        "source_file": str(record.get("source_file", "")),
        "chunk_text": str(record.get("chunk_text", record.get("doc_text", ""))),
        "chunk_text_full": str(record.get("chunk_text_full", record.get("doc_text", ""))),
        "chunk_len": int(record.get("chunk_len", len(record.get("doc_text", "")))),
        "spot_name": str(record.get("spot_name", "")),
        "schema_version": SCHEMA_VERSION,
    })
    # Persist a lightweight entity sidecar alongside dimension labels. This is
    # derived from source text, not from evaluation cases, and lets new POIs
    # become queryable without manually editing a registry.
    inferred_entities = extract_landmark_mentions(
        payload.get("doc_title", ""), payload.get("chunk_gen_title", ""),
        payload.get("chunk_text_full", ""),
    )
    supplied_entities = payload.get("entity_mentions", [])
    if not isinstance(supplied_entities, list):
        supplied_entities = []
    # Retain the structured LLM entities; append rule-derived landmarks only
    # when they are not already represented by a canonical entity name.
    supplied_names = {str(item.get("name", "")) for item in supplied_entities if isinstance(item, dict)}
    payload["entity_mentions"] = [*supplied_entities, *[
        {"name": name, "type": "auto_landmark", "aliases": [], "evidence": name}
        for name in inferred_entities if name not in supplied_names
    ]]
    return payload


class VectorStore:
    def __init__(self, *, run_dir: str | Path, backend: str, qdrant_url: str,
                 collection: str, vector_dim: int):
        self.run_dir = Path(run_dir)
        self.backend = backend
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.vector_dim = int(vector_dim)
        self.client = None
        if backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Qdrant 后端需要安装 qdrant-client；本地验收请使用 --backend local") from exc
            self.client = QdrantClient(url=qdrant_url, timeout=60, prefer_grpc=False,
                                       check_compatibility=False)

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models
        try:
            self.client.get_collection(self.collection)
            return
        except Exception:
            pass
        vectors = {
            "chunk_text_vec": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            "doc_title_vec": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            "chunk_title_vec": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        }
        self.client.create_collection(collection_name=self.collection, vectors_config=vectors)

    def upsert(self, records: Iterable[Dict[str, Any]], tag_output: Dict[str, Any],
               schema: Dict[str, Any], embeddings) -> Dict[str, Any]:
        records = list(records)
        tag_documents = tag_output.get("documents", {})
        batch_size = max(16, int(os.getenv("RAG_UPSERT_BATCH_SIZE", "256")))
        all_points = []
        point_count = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [record.get("chunk_text_full", record.get("doc_text", "")) for record in batch]
            titles = [record.get("doc_title", "") for record in batch]
            chunk_titles = [record.get("chunk_gen_title", "") for record in batch]
            vectors_text = embeddings.encode(texts)
            vectors_doc = embeddings.encode(titles)
            vectors_title = embeddings.encode(chunk_titles)
            if vectors_text and self.vector_dim != len(vectors_text[0]):
                self.vector_dim = len(vectors_text[0])
            points = []
            for index, record in enumerate(batch):
                cid = str(record["chunk_id"])
                points.append({
                    "id": _point_id(cid),
                    "vector": {
                        "chunk_text_vec": vectors_text[index],
                        "doc_title_vec": vectors_doc[index],
                        "chunk_title_vec": vectors_title[index],
                    },
                    "payload": make_payload(record, tag_documents.get(cid, {}), schema),
                })

            if self.backend == "local":
                all_points.extend(points)
            else:
                from qdrant_client import models
                self._ensure_collection(self.vector_dim)
                qdrant_points = [models.PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
                                 for item in points]
                self.client.upsert(collection_name=self.collection,
                                   points=qdrant_points, wait=True)
            point_count += len(points)

        if self.backend == "local":
            path = self.run_dir / "local_points.json"
            path.write_text(json.dumps(all_points, ensure_ascii=False), encoding="utf-8")
        manifest = {
            "backend": self.backend, "qdrant_url": self.qdrant_url,
            "collection": self.collection, "vector_dim": self.vector_dim,
            "point_count": point_count,
            "local_points": str(self.run_dir / "local_points.json") if self.backend == "local" else "",
        }
        (self.run_dir / "store_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def load_local_points(self) -> List[Dict[str, Any]]:
        path = self.run_dir / "local_points.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def scroll(self) -> List[Dict[str, Any]]:
        if self.backend == "local":
            return self.load_local_points()
        result, offset = [], None
        while True:
            points, offset = self.client.scroll(collection_name=self.collection, limit=256,
                                                offset=offset, with_payload=True, with_vectors=False)
            result.extend({"id": point.id, "payload": point.payload or {}} for point in points)
            if offset is None:
                break
        return result

    @staticmethod
    def _cosine(left: List[float], right: List[float]) -> float:
        size = min(len(left), len(right))
        if not size:
            return 0.0
        dot = sum(float(left[i]) * float(right[i]) for i in range(size))
        nl = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
        nr = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
        return dot / (nl * nr or 1.0)

    def semantic_search(self, vector: List[float], top_k: int,
                        vector_name: str = "chunk_text_vec") -> List[Dict[str, Any]]:
        if self.backend == "local":
            hits = []
            for point in self.load_local_points():
                score = self._cosine(vector, point["vector"][vector_name])
                hits.append({"id": point["id"], "score": score, "payload": point.get("payload", {})})
            return sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]
        try:
            result = self.client.query_points(collection_name=self.collection, query=vector,
                                              using=vector_name, limit=top_k,
                                              with_payload=True, with_vectors=False)
            points = getattr(result, "points", result)
        except Exception:
            points = self.client.search(collection_name=self.collection,
                                        query_vector=(vector_name, vector), limit=top_k,
                                        with_payload=True, with_vectors=False)
        return [{"id": point.id, "score": float(point.score), "payload": point.payload or {}}
                for point in points]
