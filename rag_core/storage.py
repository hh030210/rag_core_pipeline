"""Qdrant 入库及本地文件后端。默认 Qdrant；local 仅用于验收和单机调试。"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema_v2 import SCHEMA_VERSION, schema_maps


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def make_payload(record: Dict[str, Any], document_tags: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    maps = schema_maps(schema)
    payload = {
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
        "dimension_paths": list(document_tags.get("dimension_paths", []) or []),
        "tag_details": document_tags.get("tag_details", {}) or {},
    }
    tags = document_tags.get("tags", {}) or {}
    for leaf_id in maps["leaves"]:
        values = tags.get(leaf_id, [])
        if not isinstance(values, list):
            values = [values]
        # Qdrant keyword array 必须保持数组语义，禁止拼接成字符串。
        payload[f"dim_{leaf_id}"] = [str(value) for value in values if str(value).strip()]
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
        texts = [record.get("chunk_text_full", record.get("doc_text", "")) for record in records]
        titles = [record.get("doc_title", "") for record in records]
        chunk_titles = [record.get("chunk_gen_title", "") for record in records]
        vectors_text = embeddings.encode(texts)
        vectors_doc = embeddings.encode(titles)
        vectors_title = embeddings.encode(chunk_titles)
        if vectors_text:
            self.vector_dim = len(vectors_text[0])
        points = []
        for index, record in enumerate(records):
            cid = str(record["chunk_id"])
            payload = make_payload(record, tag_documents.get(cid, {}), schema)
            points.append({
                "id": _point_id(cid),
                "vector": {
                    "chunk_text_vec": vectors_text[index],
                    "doc_title_vec": vectors_doc[index],
                    "chunk_title_vec": vectors_title[index],
                },
                "payload": payload,
            })

        if self.backend == "local":
            path = self.run_dir / "local_points.json"
            path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
        else:
            from qdrant_client import models
            self._ensure_collection(self.vector_dim)
            qdrant_points = [models.PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
                             for item in points]
            for start in range(0, len(qdrant_points), 128):
                self.client.upsert(collection_name=self.collection,
                                   points=qdrant_points[start:start + 128], wait=True)
        manifest = {
            "backend": self.backend, "qdrant_url": self.qdrant_url,
            "collection": self.collection, "vector_dim": self.vector_dim,
            "point_count": len(points),
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

    def semantic_search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        if self.backend == "local":
            hits = []
            for point in self.load_local_points():
                score = self._cosine(vector, point["vector"]["chunk_text_vec"])
                hits.append({"id": point["id"], "score": score, "payload": point.get("payload", {})})
            return sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]
        try:
            result = self.client.query_points(collection_name=self.collection, query=vector,
                                              using="chunk_text_vec", limit=top_k,
                                              with_payload=True, with_vectors=False)
            points = getattr(result, "points", result)
        except Exception:
            points = self.client.search(collection_name=self.collection,
                                        query_vector=("chunk_text_vec", vector), limit=top_k,
                                        with_payload=True, with_vectors=False)
        return [{"id": point.id, "score": float(point.score), "payload": point.payload or {}}
                for point in points]

