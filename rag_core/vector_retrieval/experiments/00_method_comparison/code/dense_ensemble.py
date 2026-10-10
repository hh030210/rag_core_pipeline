"""Deployable experimental pure-dense scorer. No evaluation labels or cached query ranks."""
from pathlib import Path
import json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'output/dataset'
class DenseContextRetriever:
 def __init__(self,embeddings=None,root=ROOT):
  self.embeddings=embeddings;root=Path(root)
  source=root
  self.corpus=json.loads((source/'corpus.json').read_text());self.ids=[c['chunk_id'] for c in self.corpus];index={c:i for i,c in enumerate(self.ids)}
  self.chunk_vectors=np.load(source/'chunk_vectors.npy')
  self.passage=[]
  for vector_file,metadata_file in (('passage_vectors.npy','passages.json'),('context_vectors.npy','context_passages.json')):
   v=np.load(source/vector_file);meta=json.loads((source/metadata_file).read_text());owner=np.asarray([index[x['chunk_id']] for x in meta]);self.passage.append((v,owner))
 def search(self,query,top_k=20,spot_names=None,query_vector=None):
  started=time.perf_counter();needs_embedding=query_vector is None
  if query_vector is None:
   if self.embeddings is None:raise ValueError('Supply encoder or query_vector')
   query_vector=self.embeddings.encode([query])[0]
  q=np.asarray(query_vector,dtype=np.float32);q=q/max(float(np.linalg.norm(q)),1e-12)
  chunk=self.chunk_vectors@q;features=[]
  for v,owner in self.passage:
   maximum=np.full(len(self.ids),-1.,dtype=np.float32);np.maximum.at(maximum,owner,v@q);features.append(maximum)
  scores=.375*features[0]+.375*features[1]+.25*chunk
  spots=spot_names or [];allowed=np.asarray([not spots or c.get('spot_name') in spots or c.get('source_file','').split('-',1)[0] in spots for c in self.corpus]);sel=np.flatnonzero(allowed);order=sel[np.argsort(-scores[sel],kind='stable')[:top_k]]
  self.last_cost={'query_embeddings':int(needs_embedding),'dense_indexes':3,'vectors_scored':len(self.ids)+sum(len(v) for v,_ in self.passage),'online_qdrant_calls':0,'scoring_seconds':time.perf_counter()-started}
  return [{**self.corpus[i],'chunk_id':self.ids[i],'score':float(scores[i]),'rank':r,'source':'semantic'} for r,i in enumerate(order,1)]


class Strategy:
 def __init__(self,retriever):
  self.retriever=retriever;self._index=None
 def query_texts(self,parts,original):
  return [original]
 def search(self,parts,original,vectors,limit,spot_names):
  if self.retriever.vector_name != 'chunk_text_vec':
   raise ValueError('dense_ensemble.py supports only chunk_text_vec')
  if self._index is None:
   root=self.retriever.strategy_data_dir or ROOT
   root=Path(root);manifest=json.loads((root/'index.json').read_text())
   expected=manifest.get('configuration',{}).get('collection')
   actual=getattr(self.retriever.vector_store,'collection',None)
   if expected and actual and expected != actual:
    raise ValueError(f'Dense index collection {expected} differs from active collection {actual}; rebuild the experiment index')
   self._index=DenseContextRetriever(root=root)
  hits=self._index.search(original,top_k=limit,spot_names=spot_names,query_vector=vectors[original])
  self.retriever.last_cost=self._index.last_cost
  return hits
