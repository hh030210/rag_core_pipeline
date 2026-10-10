"""Original-query weight 2, subquery weight 1; deduplicate texts before averaging."""
import math
class Strategy:
    def __init__(self, retriever):
        self.retriever = retriever
    def query_texts(self, parts, original):
        return list(dict.fromkeys([original, *parts]))
    def search(self, parts, original, vectors, limit, spot_names):
        texts = self.query_texts(parts, original)
        normalized = [self.retriever._unit_vector(vectors[t]) for t in texts]
        weights = [2.0 if t == original else 1.0 for t in texts]
        vector = self.retriever._unit_vector([
            math.fsum(float(v[i]) * w for v, w in zip(normalized, weights))
            for i in range(len(normalized[0]))
        ])
        return self.retriever._search_store(vector, limit, spot_names)
