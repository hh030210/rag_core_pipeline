"""Select a semantic retrieval strategy by its Python filename."""
from __future__ import annotations

# Choose any .py file directly in experiments/00_method_comparison/code/.
VECTOR_RETRIEVAL_STRATEGY = "original_subqueries_centroid_w2.py"

import math
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Sequence
try:
    import numpy as np
except ImportError:
    np = None

STRATEGY_DIRECTORY = Path(__file__).resolve().parent / 'experiments/00_method_comparison/code'

class VectorRetriever:
    def __init__(self, *, embeddings, vector_store, vector_name='chunk_text_vec',
                 strategy_file=None, strategy_data_dir=None):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.vector_name = str(vector_name or 'chunk_text_vec')
        self.strategy_data_dir = strategy_data_dir
        self.strategy_file = strategy_file or VECTOR_RETRIEVAL_STRATEGY
        filename = Path(self.strategy_file)
        choices = sorted(p.name for p in STRATEGY_DIRECTORY.glob('*.py'))
        if filename.name != self.strategy_file or filename.suffix != '.py' or self.strategy_file not in choices:
            raise ValueError(f'Unknown vector strategy {self.strategy_file!r}; choose one of {choices}')
        module = import_module('rag_core.vector_retrieval.experiments.00_method_comparison.code.' + filename.stem)
        if not hasattr(module, 'Strategy'):
            raise TypeError(f'{self.strategy_file} must define Strategy')
        self.strategy = module.Strategy(self)

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
        return (str(payload.get('spot_name', '')).strip() in allowed or
                str(payload.get('source_file', '')).split('-', 1)[0].strip() in allowed)

    @staticmethod
    def _queries(subqueries, original_query):
        parts = [str(item) for item in (subqueries or []) if str(item).strip()]
        original = str(original_query or '').strip()
        if not parts and original:
            parts = [original]
        if not original:
            original = ' | '.join(parts)
        return parts, original

    def query_texts(self, subqueries, original_query=None):
        parts, original = self._queries(subqueries, original_query)
        return self.strategy.query_texts(parts, original) if parts else []

    def _search_store(self, vector, limit, spot_names):
        if self.vector_name == 'chunk_text_vec':
            hits = self.vector_store.semantic_search(vector, limit)
        else:
            hits = self.vector_store.semantic_search(vector, limit, vector_name=self.vector_name)
        results = []
        for hit in hits:
            payload = hit.get('payload', {})
            if self._matches_spot(payload, spot_names):
                results.append({'chunk_id': payload.get('chunk_id', hit['id']),
                                'score': hit['score'], 'source': 'semantic', **payload})
        for rank, item in enumerate(results, 1):
            item['rank'] = rank
        return results

    def search(self, subqueries, semantic_pool, *, original_query=None,
               query_vector=None, query_vector_text=None, query_vectors=None, spot_names=None):
        parts, original = self._queries(subqueries, original_query)
        limit = max(0, int(semantic_pool))
        if not parts or not limit:
            return []
        texts = self.strategy.query_texts(parts, original)
        vectors = {t: v for t, v in (query_vectors or {}).items() if t in texts}
        # Existing callers supplied the joined-subquery vector. Never reuse it
        # as an original-query or centroid vector without a matching text.
        supplied_text = query_vector_text if query_vector_text is not None else ' | '.join(parts)
        if query_vector is not None and supplied_text in texts:
            vectors.setdefault(supplied_text, query_vector)
        missing = [t for t in texts if t not in vectors]
        if missing:
            encoded = self.embeddings.encode(missing)
            if encoded is None or len(encoded) != len(missing):
                raise ValueError('Encoder did not return one vector per required strategy text')
            vectors.update(zip(missing, encoded))
        return self.strategy.search(parts, original, vectors, limit, spot_names or [])
