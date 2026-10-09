"""Isolated copy of the current dense-vector route.

The behavior mirrors Retriever.search's semantic branch plus
VectorStore.semantic_search: BGE-M3 dense query embedding, Qdrant
chunk_text_vec search, then the original scenic-spot post-filter.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from qdrant_client import QdrantClient

from .embedding import EmbeddingModel


def l2_normalize(vector: Iterable[float]) -> List[float]:
    values = [float(value) for value in vector]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values


class DenseVectorSearch:
    def __init__(self, *, qdrant_url: str, collection: str, model_path: str,
                 device: str = "cpu", timeout: int = 60):
        self.collection = collection
        self.client = QdrantClient(
            url=qdrant_url, timeout=timeout, prefer_grpc=False,
            check_compatibility=False,
        )
        self.embedder = EmbeddingModel(model_path=model_path, device=device,
                                       dimension=1024, mock=False)

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        return self.embedder.encode(texts)

    def search_vector(self, vector: Iterable[float], limit: int) -> List[Dict[str, Any]]:
        query = [float(value) for value in vector]
        try:
            result = self.client.query_points(
                collection_name=self.collection,
                query=query,
                using="chunk_text_vec",
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(result, "points", result)
        except Exception:
            points = self.client.search(
                collection_name=self.collection,
                query_vector=("chunk_text_vec", query),
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
            )
        return [
            {
                "chunk_id": str((point.payload or {}).get("chunk_id", point.id)),
                "score": float(point.score),
                "payload": point.payload or {},
            }
            for point in points
        ]

    @staticmethod
    def matches_spot(payload: Dict[str, Any], spot_names: List[str]) -> bool:
        """Exact copy of the existing semantic route's scenic-spot filter."""
        if not spot_names:
            return True
        explicit = str(payload.get("spot_name", "")).strip()
        source_spot = str(payload.get("source_file", "")).split("-", 1)[0].strip()
        return explicit in spot_names or source_spot in spot_names

    def search_text(self, text: str, limit: int) -> List[Dict[str, Any]]:
        vector = self.encode([text])[0]
        return self.search_vector(vector, limit)
