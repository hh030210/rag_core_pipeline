"""Read live chunk vectors and encode short passages per search, without disk indexes."""
import re
import time
import numpy as np


def normalized(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def windows(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return [text[i:i+240] for i in range(0, max(1, len(text)), 160) if text[i:i+240]]


class DenseContextRetriever:
    def __init__(self, embeddings, vector_store):
        self.embeddings = embeddings
        self.vector_store = vector_store

    def _live_points(self):
        store = self.vector_store
        if getattr(store, 'backend', None) == 'local':
            return store.load_local_points(), 0
        points, offset, calls = [], None, 0
        while True:
            batch, offset = store.client.scroll(
                collection_name=store.collection, limit=256, offset=offset,
                with_vectors=['chunk_text_vec'], with_payload=True)
            calls += 1
            points.extend({'id': p.id, 'payload': p.payload or {}, 'vector': p.vector} for p in batch)
            if offset is None:
                return points, calls

    def search(self, query, top_k=20, spot_names=None, query_vector=None):
        started = time.perf_counter()
        needs_embedding = query_vector is None
        if needs_embedding:
            query_vector = self.embeddings.encode([query])[0]
        q = normalized(query_vector)
        points, calls = self._live_points()
        corpus = [dict(p['payload'], chunk_id=str(p['payload'].get('chunk_id', p['id']))) for p in points]
        self.last_cost = {'query_embeddings': int(needs_embedding), 'dense_indexes': 3,
                          'online_qdrant_calls': calls, 'candidate_text_encodings': 0,
                          'vectors_scored': 0, 'persistent_vector_cache': False}
        if not corpus or top_k <= 0:
            self.last_cost['scoring_seconds'] = time.perf_counter() - started
            return []
        ids = [c['chunk_id'] for c in corpus]
        by_id = {cid: i for i, cid in enumerate(ids)}
        docs = [c.get('chunk_text_full') or c.get('chunk_text', '') for c in corpus]
        chunk_vectors = normalized([p['vector']['chunk_text_vec'] for p in points])
        if chunk_vectors.shape[1] != q.shape[0]:
            raise ValueError('Query and database vector dimensions differ')
        features = []
        for contextual in (False, True):
            texts, owners = [], []
            for i, (cid, text) in enumerate(zip(ids, docs)):
                if contextual:
                    prefix, sep, suffix = cid.rpartition('::chunk_')
                    previous = by_id.get(prefix + sep + f'{int(suffix)-1:04d}') if sep and suffix.isdigit() else None
                    context = docs[previous][-160:] if previous is not None else ''
                    text = corpus[i].get('doc_title', '') + ' ' + context + ' ' + text
                parts = windows(text)
                texts.extend(parts)
                owners.extend([i] * len(parts))
            maximum = np.full(len(ids), -1., dtype=np.float32)
            if texts:
                vectors = normalized(self.embeddings.encode_compact(texts))
                np.maximum.at(maximum, np.asarray(owners), vectors @ q)
                self.last_cost['candidate_text_encodings'] += len(texts)
                del vectors
            features.append(maximum)
        scores = .25 * (chunk_vectors @ q) + .375 * features[0] + .375 * features[1]
        spots = spot_names or []
        allowed = np.asarray([not spots or c.get('spot_name') in spots or c.get('source_file', '').split('-', 1)[0] in spots for c in corpus])
        selected = np.flatnonzero(allowed)
        order = selected[np.argsort(-scores[selected], kind='stable')[:top_k]]
        self.last_cost['vectors_scored'] = len(ids) + self.last_cost['candidate_text_encodings']
        self.last_cost['scoring_seconds'] = time.perf_counter() - started
        return [{**corpus[i], 'chunk_id': ids[i], 'score': float(scores[i]), 'rank': rank, 'source': 'semantic'}
                for rank, i in enumerate(order, 1)]


class Strategy:
    def __init__(self, retriever):
        self.retriever = retriever
        self._scorer = DenseContextRetriever(retriever.embeddings, retriever.vector_store)

    def query_texts(self, parts, original):
        return [original]

    def search(self, parts, original, vectors, limit, spot_names):
        if self.retriever.vector_name != 'chunk_text_vec':
            raise ValueError('dense_ensemble.py supports only chunk_text_vec')
        hits = self._scorer.search(original, top_k=limit, spot_names=spot_names, query_vector=vectors[original])
        self.retriever.last_cost = self._scorer.last_cost
        return hits
