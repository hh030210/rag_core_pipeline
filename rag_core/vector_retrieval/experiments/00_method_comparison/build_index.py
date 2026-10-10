"""Build the three vector indexes required by the retained pure-dense ensemble."""
import json,re
from pathlib import Path
import numpy as np

def normalized(v):
 v=np.asarray(v,dtype=np.float32);return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def windows(text):
 text=re.sub(r'\s+',' ',text).strip();return [text[i:i+240] for i in range(0,max(1,len(text)),160) if text[i:i+240]]
def build(store,embeddings,dataset):
 dataset=Path(dataset);points=[];offset=None
 while True:
  batch,offset=store.client.scroll(collection_name=store.collection,limit=256,offset=offset,with_vectors=['chunk_text_vec'],with_payload=True);points.extend(batch)
  if offset is None:break
 corpus=[dict(p.payload,chunk_id=str(p.payload['chunk_id'])) for p in points];ids=[c['chunk_id'] for c in corpus];byid={c:i for i,c in enumerate(ids)};docs=[c.get('chunk_text_full') or c.get('chunk_text','') for c in corpus]
 (dataset/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False));np.save(dataset/'chunk_vectors.npy',normalized([p.vector['chunk_text_vec'] for p in points]))
 for contextual in (False,True):
  texts=[];metadata=[]
  for i,(cid,doc) in enumerate(zip(ids,docs)):
   if contextual:
    prefix,sep,suffix=cid.rpartition('::chunk_');previous=byid.get(prefix+sep+f'{int(suffix)-1:04d}') if sep and suffix.isdigit() else None;context=docs[previous][-160:] if previous is not None else '';doc=corpus[i].get('doc_title','')+' '+context+' '+doc
   for p in windows(doc):texts.append(p);metadata.append({'chunk_id':cid,'text':p})
  print('build contextual' if contextual else 'build standalone','passages',len(texts),flush=True);vectors=normalized(embeddings.encode_compact(texts));np.save(dataset/('context_vectors.npy' if contextual else 'passage_vectors.npy'),vectors);(dataset/('context_passages.json' if contextual else 'passages.json')).write_text(json.dumps(metadata,ensure_ascii=False))
