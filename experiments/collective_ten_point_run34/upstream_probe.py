"""One isolated request through the existing school's wrapper, no proxy edits."""
import ast
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


async def main():
    run=Path(sys.argv[1]);root=Path('/pub/yanglw/tourist_three_experiments/sdu_ds')
    out=run/'technical_repairs/upstream_readonly_probe_v1';out.mkdir(exist_ok=False)
    tree=ast.parse((root/'run_server.py').read_text())
    model_config=None
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='model_config' for n in node.targets):model_config=ast.literal_eval(node.value)
    assert model_config and model_config['DeepSeek-V4-Pro']==(73,'本科生',2,'DeepSeek-V4-Pro',2)
    os.chdir(root);sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('owned_school_wrapper',root/'sduwrap.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import httpx
    key='final:bare:970cc795f02a17c85895b68091c0fc68:answer'
    candidates=[(p,json.loads(p.read_text())) for p in (run/'api_calls').glob('*.json')]
    source,previous=next((p,r) for p,r in candidates if r['key']==key and r.get('error')=='ValueError: Empty response')
    request=previous['request'];config=module.ChatConfig()
    for name,value in zip(('compose_id','auth_tag','deep_search','model_name','internet_search'),model_config[request['model']]):setattr(config,name,value)
    content=request['messages'][-1]['content'];history=request['messages'][:-1]
    expected=module.make_chat_request(content,history,config)
    record={'original_nonstream_request':str(source),'logical_request':request,'expected_upstream_form':expected,
            'scope':'one failed exact input via existing wrapper in own isolated process, proxy unmodified','started_at':time.time()}
    async with httpx.AsyncClient() as client:
        class RecordingClient:
            async def post(self,*args,**kwargs):
                assert kwargs.get('data')==expected,'Effective model input/profile changed'
                response=await client.post(*args,**kwargs)
                record.update(http_status=response.status_code,content_type=response.headers.get('content-type'),raw_upstream_text=response.text)
                (out/'raw_response.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
                return response
        pieces=[]
        async for chunk in module.chat(content,history,config,RecordingClient()):pieces.append(chunk)
    record.update(wrapper_answer=''.join(pieces),seconds=time.time()-record['started_at'])
    (out/'raw_response.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    schemas=[]
    for line in record.get('raw_upstream_text','').splitlines():
        if line.startswith('data:'):
            text=line[5:].strip()
            try:
                value=json.loads(text);schemas.append({'type':type(value).__name__,'keys':list(value) if isinstance(value,dict) else None})
            except ValueError:schemas.append({'json_parse_error':True,'length':len(text)})
    print(json.dumps({'http_status':record.get('http_status'),'content_type':record.get('content_type'),
                      'raw_length':len(record.get('raw_upstream_text','')),'wrapper_answer_length':len(record['wrapper_answer']),'schemas':schemas[:8]},ensure_ascii=False))


if __name__=='__main__':asyncio.run(main())
