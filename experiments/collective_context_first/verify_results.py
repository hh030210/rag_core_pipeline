"""Read-only integrity and aggregate-score verification; no model calls."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def verify():
    repo=Path(__file__).resolve().parents[2]
    folder=repo/'results/scenic_context_first_knowledge_deepseek_v4_pro_20261026_run33'
    inventory=json.loads((folder/'archive_inventory.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(folder/'audit_archive.zip') as z:
        assert set(z.namelist())==set(inventory)
        for name,record in inventory.items():
            data=z.read(name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
        load=lambda name:json.loads(z.read(name))
        frozen=load('frozen_manifest.json')
        for name,sha in frozen['source_sha256'].items():
            assert hashlib.sha256(z.read('code_snapshot/'+name)).hexdigest()==sha,name
        for name,sha in frozen['input_sha256'].items():
            assert hashlib.sha256(z.read('shared/'+name)).hexdigest()==sha,name
        rows=load('reports/全量逐题对照.json')['records']
        result=load('results.json')
        assert result==json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert len(rows)==378 and len(load('shared/train_top5.json'))==23
        assert load('clustering.json')['k']==8
        assert sum(n.startswith('api_calls/') and n.endswith('.json') for n in z.namelist())==1834
        groups=[row['groups']['collective'] for row in rows]
        answered=[v for v in groups if v['answer']]
        complete=[v for v in groups if v['status']=='complete']
        assert len(answered)==result['valid_answers']==377
        assert len(complete)==result['valid_evaluations']==376
        for key,entry in result['llm_scores'].items():
            scores=[v['llm_scores'][key] for v in complete if v['llm_scores'][key] is not None]
            assert len(scores)==entry['n'] and abs(statistics.mean(scores)-entry['mean'])<1e-10,key
        for key,mean in result['machine_metrics'].items():
            assert abs(statistics.mean(v['machine_metrics'][key] for v in answered)-mean)<1e-10,key
        for cid in range(8):
            final=load(f'collective_feedback/final_clusters/cluster_{cid}.json')
            assert final['sha256']==load(f'clusters/cluster_{cid}.json')['sha256']
        for p in (folder/'reports').iterdir():
            if p.is_file():
                assert p.read_bytes()==z.read('reports/'+p.name),p.name
    print(json.dumps({'passed':True,'archive_files':len(inventory),'raw_calls':1834,
         'valid_answers':377,'complete_evaluations':376,'clusters':8,
         'accepted_feedback_updates':0,'aggregate_scores_verified':True,
         'scope':'Integrity/aggregation, not certification of model judgments'},ensure_ascii=False))


if __name__=='__main__':
    verify()
