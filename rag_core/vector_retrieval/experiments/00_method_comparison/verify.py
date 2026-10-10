"""Recompute the five retained metrics and verify standalone dense inference."""
import argparse,json
from importlib import import_module
import numpy as np
b=import_module(__package__+'.run')
def main():
 p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true',help='Also encode and retrieve three representative queries');a=p.parse_args();ds=b.HERE/'output/dataset';m=b.read(ds/'index.json');rows=[b.read(ds/f) for f in m['detail_files'].values()];assert tuple(m['methods'])==b.METHODS
 assert len(rows)==len(m['items']) and any(r['gold_chunk_ids'] for r in rows)
 for r in rows:
  assert set(r['strategies'])==set(b.METHODS)
  for method,s in r['strategies'].items():
   ranks=[x['chunk_id'] for x in s['results']];assert len(ranks)==len(set(ranks))
   for k in b.KS:assert bool(s['metrics'][f'hit_at_{k}'])==bool(set(r['gold_chunk_ids']) & set(ranks[:k]))
 b.report(rows)
 if a.fresh:
  from rag_core.embedding import EmbeddingModel
  from rag_core.storage import VectorStore
  from rag_core.vector_retrieval import VectorRetriever
  chosen=[rows[0],rows[len(rows)//2],rows[-1]];e=EmbeddingModel('/home/humq/rag_db_silm/model/bge-m3',device='cuda:0');texts=list(dict.fromkeys(t for r in chosen for t in [*b.prepare(r)[3].values(),*b.prepare(r)[2]]));vectors=dict(zip(texts,e.encode(texts)))
  sm=b.read('/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/store_manifest.json');store=VectorStore(run_dir='/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1',backend='qdrant',qdrant_url=sm['qdrant_url'],collection=sm['collection'],vector_dim=sm['vector_dim']);baseline=VectorRetriever(embeddings=e,vector_store=store);dense=b.DenseContextRetriever()
  for r in chosen:
   actual=b.evaluate(r,vectors,baseline,dense)
   for method,hits in actual.items():
    saved=r['strategies'][method]['results'];assert [str(x['chunk_id']) for x in hits]==[str(x['chunk_id']) for x in saved],(r['name'],method)
  print('FRESH RETRIEVAL PARITY 3 queries x 5 methods',flush=True)
 print('DATASET CHECK PASSED',len(rows),flush=True)
if __name__=='__main__':main()
