"""Verify extracted online fusion against all archived fresh-run snapshots."""
import copy,json,math
from pathlib import Path
from rag_core.retrieval_fusion.fusion import adaptive_fusion, _save_fusion_snapshot
from rag_core.retrieval import Retriever
F=Path(__file__).resolve().parents[3]
ROOT=F.parents[1]
snapshots=F/'experiments/01_online_snapshots/output'
files=sorted(snapshots.glob('retrieval_fusion_*.json'))
assert files, 'No archived online snapshots found'
branches={}
for path in files:
    data=json.loads(path.read_text())
    result=adaptive_fusion(copy.deepcopy(data['semantic_candidates']),
                           copy.deepcopy(data['dimension_candidates']),top_k=data['top_k'])
    got=result['fusion_candidates'];expected=data['fusion_candidates']
    assert [x['chunk_id'] for x in got]==[x['chunk_id'] for x in expected],path.name
    assert all(math.isclose(a['score'],b['score'],rel_tol=1e-12,abs_tol=1e-12)
               for a,b in zip(got,expected)),path.name
    branch=result['fusion_strategy']['branch'];branches[branch]=branches.get(branch,0)+1
out=Path(__file__).resolve().parent.parent/'output/validation.json'
out.write_text(json.dumps({'snapshots':len(files),'ranking_and_scores_equal':len(files),
                           'branches':branches,'retriever_import':True},indent=2))
print(f'PASS: {len(files)}/{len(files)} archived online fusion orders and scores unchanged; Retriever imports successfully.')

