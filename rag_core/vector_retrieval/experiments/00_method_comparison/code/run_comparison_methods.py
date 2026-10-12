"""Evaluate only the five retained vector retrieval methods."""
import argparse,json,shutil,time
from pathlib import Path
from uuid import uuid4
from importlib import import_module
import sys
if not __package__: sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from rag_core.embedding import EmbeddingModel
from rag_core.storage import VectorStore
from rag_core.vector_retrieval import VectorRetriever
from rag_core.evaluation import route_metrics
DenseContextRetriever=import_module('rag_core.vector_retrieval.experiments.00_method_comparison.code.dense_ensemble').DenseContextRetriever
HERE=Path(__file__).resolve().parents[1]
METHODS=tuple(sorted(p.stem for p in (HERE/'code').glob('*.py') if p.name!='run_comparison_methods.py'))
KS=(1,5,10,15,20)
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def prepare(r):
 orig=r.get('original_query') or r.get('query');parts=r.get('subqueries')
 if parts is None:parts=[s.strip() for s in r['retrieval_query'].split('|') if s.strip()]
 parts=parts or [orig];comp=list(dict.fromkeys([orig,*parts]));return orig,parts,comp,{'subqueries_concat':' | '.join(parts),'original_only':orig,'original_plus_subqueries_concat':' | '.join([orig,*parts])}
def evaluate(r,vectors,retriever,ensemble,pool=20):
 orig,parts,comp,texts=prepare(r);spots=r.get('query_analysis',{}).get('spot_names') or []
 if not hasattr(retriever,'_comparison_strategies'):
  retriever._comparison_strategies={method:VectorRetriever(embeddings=retriever.embeddings,vector_store=retriever.vector_store,strategy_file=method+'.py') for method in METHODS}
  retriever._comparison_strategies['dense_ensemble'].strategy._scorer=ensemble
 return {method:selected.search(parts,100 if method=='dense_ensemble' else pool,original_query=orig,query_vectors=vectors,spot_names=spots) for method,selected in retriever._comparison_strategies.items()}

def report(rows):
 mapped=[r for r in rows if r['gold_chunk_ids']]
 for method in METHODS:
  counts={k:sum(bool(set(r['gold_chunk_ids']) & {h['chunk_id'] for h in r['strategies'][method]['results'][:k]}) for r in mapped) for k in (5,10,15)}
  print(method,{'mapped_count':len(mapped),'all_count':len(rows),'hit_counts':counts,'mapped_hit_pct':{k:round(n/(len(mapped) or 1)*100,4) for k,n in counts.items()},'all_hit_pct':{k:round(n/(len(rows) or 1)*100,4) for k,n in counts.items()}},flush=True)
def load_rows(source=None):
 source=Path(source or HERE/'output/original_only');manifest=read(source/'index.json')
 rows=[read(source/f) for f in manifest['detail_files'].values()]
 return manifest,rows

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input-dataset',type=Path,default=HERE/'output/original_only')
 p.add_argument('--run-dir',type=Path,default=Path('/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1'))
 p.add_argument('--model-path',default='/home/humq/rag_db_silm/model/bge-m3');p.add_argument('--device',default='cuda:0')
 p.add_argument('--pool',type=int,default=20);args=p.parse_args(argv)
 if args.pool<20:p.error('--pool must be >=20')
 manifest,rows=load_rows(args.input_dataset)
 for r in rows:
  r.setdefault('original_query',r.get('query'));r.setdefault('gold_chunk_ids',(r.get('gold') or {}).get('chunk_ids',[]));r.setdefault('gold_relevance',(r.get('gold') or {}).get('relevance',{}))
 emb=EmbeddingModel(args.model_path,device=args.device)
 texts=list(dict.fromkeys(t for r in rows for t in [*prepare(r)[3].values(),*prepare(r)[2]]));vectors=dict(zip(texts,emb.encode(texts)))
 sm=read(args.run_dir/'store_manifest.json');store=VectorStore(run_dir=args.run_dir,backend='qdrant',qdrant_url=sm['qdrant_url'],collection=sm['collection'],vector_dim=sm['vector_dim']);retriever=VectorRetriever(embeddings=emb,vector_store=store)
 target=HERE/'output';staged=HERE/('.output-stage-'+uuid4().hex);backup=HERE/('.output-backup-'+uuid4().hex)
 try:
  staged.mkdir();ensemble=DenseContextRetriever(emb,store)
  idx={'schema_version':2,'batch_id':uuid4().hex,'input_batch_id':manifest['batch_id'],'methods':METHODS,'items':[],'detail_files':{},'configuration':{'input_count':len(rows),'baseline_pool':args.pool,'ensemble_pool':100,'model':args.model_path,'collection':sm['collection'],'ks':KS}}
  for method in METHODS:(staged/method).mkdir()
  published=[];start=time.perf_counter()
  for n,r in enumerate(rows):
   ranking=evaluate(r,vectors,retriever,ensemble,args.pool);orig,parts,comp,_=prepare(r);name=f'query_{n:06d}.json'
   item={'name':name,'batch_id':idx['batch_id'],'original_query':orig,'subqueries':parts,'component_texts':comp,'query_analysis':r.get('query_analysis',{}),'gold_chunk_ids':r['gold_chunk_ids'],'gold_relevance':r['gold_relevance'],'strategies':{},'cost':{}}
   for method in METHODS:
    hits=ranking[method];result={'results':[{'chunk_id':str(h['chunk_id']),'rank':i,'score':float(h['score'])} for i,h in enumerate(hits,1)],'metrics':route_metrics(hits,item['gold_chunk_ids'],KS,item['gold_relevance'])}
    cost={'logical_dense_searches':3 if method=='dense_ensemble' else 1,'query_embedding_texts':len(comp) if method=='original_subqueries_centroid_w2' else 1,'online_qdrant_calls':ensemble.last_cost['online_qdrant_calls'] if method=='dense_ensemble' else 1, 'candidate_text_encodings':ensemble.last_cost['candidate_text_encodings'] if method=='dense_ensemble' else 0}
    item['strategies'][method]=result;item['cost'][method]=cost
    write(staged/method/name,{**item,'strategy_file':method+'.py','strategies':{method:result},'cost':{method:cost}})
   published.append(item);idx['items'].append({'name':name,'query':orig});idx['detail_files'][name]=name
   if n%50==0:print('comparison',n+1,'/',len(rows),round(time.perf_counter()-start,1),'s',flush=True)
  for method in METHODS:write(staged/method/'index.json',{**idx,'method':method,'strategy_file':method+'.py'})
  report(published)
  had_previous=target.exists()
  if had_previous:target.rename(backup)
  try:staged.rename(target)
  except BaseException:
   if had_previous:backup.rename(target)
   raise
  if had_previous:shutil.rmtree(backup)
 finally:
  for temporary in (staged,):
   if temporary.exists():shutil.rmtree(temporary)
if __name__=='__main__':main()
