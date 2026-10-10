"""Regression checks for strategy dispatch, vector ownership, and fusion payload compatibility."""
import unittest
from pathlib import Path
from importlib import import_module
from unittest.mock import patch
import numpy as np
from rag_core.vector_retrieval import VectorRetriever
search_module=import_module('rag_core.vector_retrieval.search')
class Encoder:
 def __init__(self):self.calls=[]
 def encode(self,texts):
  self.calls.append(list(texts));return [{'original':[1.,0.],'part':[0.,1.],'other':[0.,-1.],'part | other':[-1.,0.],'original | part | other':[1.,1.]}[t] for t in texts]
class Store:
 def __init__(self):self.calls=[]
 def semantic_search(self,vector,limit,**kwargs):
  self.calls.append((np.asarray(vector),limit,kwargs));return [{'id':'a','score':.8,'payload':{'chunk_id':'a','source_file':'spot-document','chunk_text_full':'text'}}]
class SwitchingTests(unittest.TestCase):
 def test_all_code_files_are_selectable(self):
  files=sorted(p.name for p in search_module.STRATEGY_DIRECTORY.glob('*.py'));self.assertEqual(len(files),5)
  for filename in files:
   with patch.object(search_module,'VECTOR_RETRIEVAL_STRATEGY',filename):
    r=VectorRetriever(embeddings=Encoder(),vector_store=Store());self.assertEqual(r.strategy_file,filename);self.assertEqual(r.search([],0),[])
 def test_joined_vector_is_not_reused_for_original(self):
  e=Encoder();s=Store();r=VectorRetriever(embeddings=e,vector_store=s,strategy_file='original_only.py');r.search(['part','other'],20,original_query='original',query_vector=[-1.,0.]);self.assertEqual(e.calls,[['original']]);np.testing.assert_allclose(s.calls[0][0],[1.,0.])
 def test_centroid_deduplicates_and_batches_missing_texts(self):
  e=Encoder();s=Store();r=VectorRetriever(embeddings=e,vector_store=s,strategy_file='original_subqueries_centroid_w2.py');r.search(['original','part','part'],20,original_query='original',query_vectors={'original':[1.,0.]});self.assertEqual(e.calls,[['part']]);np.testing.assert_allclose(s.calls[0][0],np.asarray([2.,1.])/np.sqrt(5),atol=1e-6);self.assertEqual(len(s.calls),1)
 def test_legacy_joined_vector_reuse_and_named_field(self):
  e=Encoder();s=Store();r=VectorRetriever(embeddings=e,vector_store=s,vector_name='custom',strategy_file='subqueries_concat.py');hits=r.search(['part','other'],20,original_query='original',query_vector=[-1.,0.],spot_names=['spot']);self.assertEqual(e.calls,[]);self.assertEqual(s.calls[0][2],{'vector_name':'custom'});self.assertEqual(hits[0]['chunk_text_full'],'text');self.assertEqual(hits[0]['rank'],1)
 def test_invalid_filename_is_rejected(self):
  for name in ('../search.py','run.py','original_only'):
   with self.assertRaises(ValueError):VectorRetriever(embeddings=Encoder(),vector_store=Store(),strategy_file=name)
 def test_dense_returns_payload_without_database_calls(self):
  s=Store();r=VectorRetriever(embeddings=Encoder(),vector_store=s,strategy_file='dense_ensemble.py');q=np.ones(1024,dtype=np.float32);hits=r.search(['part','other'],5,original_query='original',query_vectors={'original':q});self.assertEqual(len(hits),5);self.assertEqual(s.calls,[]);self.assertEqual(hits[0]['source'],'semantic');self.assertIn('source_file',hits[0]);self.assertIn('chunk_text_full',hits[0]);self.assertEqual(r.last_cost['dense_indexes'],3)
 def test_main_flow_batches_strategy_texts_with_dimensions(self):
  from types import SimpleNamespace
  from rag_core.retrieval import Retriever
  for method,expected in [('subqueries_concat.py',[]),('original_only.py',['original']),('original_subqueries_centroid_w2.py',['original','part','other'])]:
   e=Encoder();old=e.encode
   def encode(texts):
    ordinary=[t for t in texts if t!='dimension'];values=old(ordinary);lookup=dict(zip(ordinary,values));lookup['dimension']=[.3,.7];e.calls[-1]=list(texts);return [lookup[t] for t in texts]
   e.encode=encode;store=Store();r=Retriever.__new__(Retriever);r.settings=SimpleNamespace(top_k=5,dim_alpha=.5,fact_anchor_rerank_enabled=False);r.embeddings=e;r.vector_retriever=VectorRetriever(embeddings=e,vector_store=store,strategy_file=method);r.tags_by_dim={};r._parse_query=lambda q:{'constraints':{'leaf':{'role':'main','labels':['label']}},'spot_names':[]};r._dimension_query_text=lambda *args:'dimension';r._dimension_policy=lambda a:{};r._fact_index_matches=lambda facts:{};r._entity_anchor_candidates=lambda *args:[];r._fact_index_candidates=lambda *args:[];r._no_candidate_lexical_fallback=lambda *args:[]
   result=r.search('part | other',original_query='original',subqueries=['part','other']);self.assertEqual(e.calls,[['part | other','dimension',*expected]]);self.assertEqual(result['query_analysis']['vector_retrieval_strategy'],method);self.assertEqual(len(store.calls),1)
if __name__=='__main__':unittest.main()
