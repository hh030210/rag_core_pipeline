"""Vector retrieval strategy: original_only."""
class Strategy:
    def __init__(self, retriever):
        self.retriever = retriever
    def query_texts(self, parts, original):
        return [original]
    def search(self, parts, original, vectors, limit, spot_names):
        text = self.query_texts(parts, original)[0]
        vector = self.retriever._unit_vector(vectors[text])
        return self.retriever._search_store(vector, limit, spot_names)
