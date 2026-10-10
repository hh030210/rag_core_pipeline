"""Save the adaptive strategy independently of the configured online strategy."""
from pathlib import Path
from importlib import import_module
import argparse,copy,json
from uuid import uuid4
from datetime import datetime,timezone
from rag_core.retrieval_fusion.fusion import apply_fusion
_shared=import_module('rag_core.retrieval_fusion.experiments.00_method_comparison.replay_dimension')
ROOT=Path(__file__).resolve().parent

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset',type=Path,default=ROOT/'output/online/dataset')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/online_adaptive')
    args=parser.parse_args(argv)
    source=json.loads((args.input_dataset/'index.json').read_text())
    batch=uuid4().hex
    methods=sorted(p.name for p in (ROOT/'code').glob('*.py'))
    def write(output):
        dataset=output/'dataset';details=dataset/'details';details.mkdir(parents=True)
        index=dict(schema_version=1,batch_id=batch,input_batch_id=source['batch_id'],generated_at=datetime.now(timezone.utc).isoformat(),strategy_file='online_adaptive.py',methods=methods,items=[],detail_files={})
        for name,relative in source['detail_files'].items():
            detail=json.loads((args.input_dataset/relative).read_text())
            if detail['batch_id']!=source['batch_id']:raise ValueError('Mixed source batches')
            result=apply_fusion(copy.deepcopy(detail['routes']['semantic']),copy.deepcopy(detail['routes']['dimension']),top_k=10,strategy_file='online_adaptive.py')
            detail['routes']['old']=result['fusion_candidates'];detail['fusion_strategy']=result['fusion_strategy']
            detail['batch_id']=batch
            (details/name).write_text(json.dumps(detail,ensure_ascii=False)+'\n')
            index['items'].append(dict(name=name,query=detail['query'],gold_count=len(detail['gold']['gold_chunk_ids'])))
            index['detail_files'][name]='details/'+name
        (dataset/'index.json').write_text(json.dumps(index,ensure_ascii=False)+'\n')
    _shared.replace_output(args.output_dir,write)
    print(f'Adaptive strategy updated: {len(source["detail_files"])} queries, batch={batch}')
if __name__=='__main__':main()
