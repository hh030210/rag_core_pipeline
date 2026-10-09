"""Semantic vector retrieval, kept separate from dimension search and fusion."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - lightweight local environments
    np = None


class VectorRetriever:
    """Encode a retrieval query and retrieve semantically similar chunks."""

    def __init__(self, *, embeddings, vector_store, vector_name: str = "chunk_text_vec"):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.vector_name = str(vector_name or "chunk_text_vec")

    @staticmethod
    def _unit_vector(values):
        if np is not None:
            vector = np.asarray(values, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm > 0 else vector
        numeric = [float(value) for value in values]
        norm = math.sqrt(math.fsum(value * value for value in numeric))
        return [value / norm for value in numeric] if norm > 0 else numeric

    @staticmethod
    def _matches_spot(payload: Dict[str, Any], spot_names: Sequence[str]) -> bool:
        if not spot_names:
            return True
        allowed = set(spot_names)
        explicit = str(payload.get("spot_name", "")).strip()
        source_spot = str(payload.get("source_file", "")).split("-", 1)[0].strip()
        return explicit in allowed or source_spot in allowed

    def search(
        self,
        subqueries: Sequence[str],
        semantic_pool: int,
        *,
        original_query: str | None = None,
        query_vector: Sequence[float] | None = None,
        spot_names: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return ranked semantic candidates in the legacy Retriever format.

        The caller may supply a pre-encoded vector to preserve batched encoding
        with dimension-query vectors. Otherwise, this method encodes the joined
        subqueries itself. original_query is used only as a fallback when no
        usable subqueries were supplied.
        """
        normalized_subqueries = [
            str(item) for item in (subqueries or []) if str(item).strip()
        ]
        if not normalized_subqueries and original_query:
            normalized_subqueries = [str(original_query)]
        if not normalized_subqueries:
            return []

        retrieval_query = " | ".join(normalized_subqueries)
        if query_vector is None:
            encoded = self.embeddings.encode([retrieval_query])
            if encoded is None or len(encoded) == 0:
                return []
            query_vector = self._unit_vector(encoded[0])

        limit = max(0, int(semantic_pool))
        if limit == 0:
            return []
        if self.vector_name == "chunk_text_vec":
            hits = self.vector_store.semantic_search(query_vector, limit)
        else:
            hits = self.vector_store.semantic_search(
                query_vector, limit, vector_name=self.vector_name
            )

        results = []
        for hit in hits:
            payload = hit.get("payload", {})
            if not self._matches_spot(payload, spot_names or []):
                continue
            results.append({
                "chunk_id": payload.get("chunk_id", hit["id"]),
                "score": hit["score"],
                "source": "semantic",
                **payload,
            })
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
        return results
