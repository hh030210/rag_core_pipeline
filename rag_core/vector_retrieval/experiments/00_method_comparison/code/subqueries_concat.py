"""Vector retrieval strategy: subqueries_concat."""
import math
from typing import Any, Dict, List, Sequence
try:
    import numpy as np
except ImportError:
    np=None

def _unit_vector(values):
    if np is not None:
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector
    numeric = [float(value) for value in values]
    norm = math.sqrt(math.fsum(value * value for value in numeric))
    return [value / norm for value in numeric] if norm > 0 else numeric

def _matches_spot(payload: Dict[str, Any], spot_names: Sequence[str]) -> bool:
    if not spot_names:
        return True
    allowed = set(spot_names)
    return (str(payload.get('spot_name', '')).strip() in allowed or
            str(payload.get('source_file', '')).split('-', 1)[0].strip() in allowed)

def _search_store(retriever, vector, limit, spot_names):
    if retriever.vector_name == 'chunk_text_vec':
        hits = retriever.vector_store.semantic_search(vector, limit)
    else:
        hits = retriever.vector_store.semantic_search(vector, limit, vector_name=retriever.vector_name)
    results = []
    for hit in hits:
        payload = hit.get('payload', {})
        if _matches_spot(payload, spot_names):
            results.append({'chunk_id': payload.get('chunk_id', hit['id']),
                            'score': hit['score'], 'source': 'semantic', **payload})
    for rank, item in enumerate(results, 1):
        item['rank'] = rank
    return results

class Strategy:
    def __init__(self, retriever):
        self.retriever = retriever
    def query_texts(self, parts, original):
        return [' | '.join(parts)]
    def search(self, parts, original, vectors, limit, spot_names):
        text = self.query_texts(parts, original)[0]
        vector = _unit_vector(vectors[text])
        return _search_store(self.retriever, vector, limit, spot_names)
