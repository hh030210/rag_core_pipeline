"""Verify all retained ranks, dispatch, payloads and missing-route behavior."""
from pathlib import Path
import json
import copy
from rag_core.retrieval_fusion.fusion import apply_fusion, FUSION_STRATEGY
ROOT=Path(__file__).resolve().parent/'output'
def main():
    online=ROOT/'online/dataset'; index=json.loads((online/'index.json').read_text())
    names=list(index['detail_files']);count=0
    for name in names:
        detail=json.loads((online/index['detail_files'][name]).read_text())
        routes=detail['routes']
        selected=detail.get('query_analysis',{}).get('fusion_strategy') or 'online_adaptive.py'
        for file,folder,key in [(selected,'online','old'),('online_adaptive.py','online_adaptive','old'),('dimension_score.py','dimension_score','new'),('recommended_fusion.py','recommended_fusion','candidates')]:
            expected_detail=json.loads((ROOT/folder/'dataset/details'/name).read_text())
            expected=expected_detail[key] if key=='candidates' else expected_detail['routes'][key]
            sem,dim=copy.deepcopy(routes['semantic']),copy.deepcopy(routes['dimension'])
            for c in sem+dim:
                c['chunk_text_full']=detail['chunks'].get(c['chunk_id'],{}).get('text','')
                c['payload_marker']='preserved'
            result=apply_fusion(sem,dim,top_k=10,query=detail.get('retrieval_query') or detail['query'],original_query=detail['query'],query_analysis=detail.get('query_analysis'),strategy_file=file)
            actual=result['fusion_candidates']
            assert [c['chunk_id'] for c in actual]==[c['chunk_id'] for c in expected],(name,file,'ranking changed')
            assert all(abs(a['score']-b['score'])<1e-8 for a,b in zip(actual,expected)),(name,file,'scores changed')
            assert all(c['payload_marker']=='preserved' for c in actual)
            assert result['fusion_strategy']['file']==file
            assert result['fusion_results']==actual[:10]
            count+=1
    for file in ['online_adaptive.py','dimension_score.py','recommended_fusion.py']:
        for sem,dim in [([],[]),([{'chunk_id':'a','score':0.8,'rank':1,'chunk_text_full':'示例景区'}],[]),([],[{'chunk_id':'a','score':0.8,'rank':1,'chunk_text_full':'示例景区'}])]:
            result=apply_fusion(sem,dim,top_k=3,query='示例景区',strategy_file=file)
            assert len(result['fusion_candidates'])==bool(sem or dim)
    for file in ['../dimension_score.py','missing.py']:
        try:apply_fusion([],[],top_k=1,strategy_file=file)
        except ValueError:pass
        else:raise AssertionError('Invalid file accepted')
    assert FUSION_STRATEGY in {'online_adaptive.py','dimension_score.py','recommended_fusion.py'}
    assert apply_fusion([],[],top_k=1)['fusion_strategy']['file']==FUSION_STRATEGY
    print(f'PASS: {count} saved query/strategy rankings, payload preservation, empty/missing routes, filename validation and default dispatch')
if __name__=='__main__':main()
