"""Offline wire-equivalence and transport recovery checks; no billed requests."""
from pathlib import Path
import json,os,tempfile,types
from unittest.mock import patch
import pilot
import siliconflow_rate_call as paced

class Response:
    def __init__(self,status,body):
        self.status_code=status;self.body=body;self.ok=status==200
        self.headers={'content-type':'application/json'};self.text=json.dumps(body)
    def json(self):return self.body
    def close(self):pass
class Pacer:
    def __init__(self):self.releases=[];self.starts=[]
    def acquire(self,model,tokens):
        self.starts.append((model,tokens));return {'ticket':str(len(self.starts)),'model':model}
    def release(self,ticket,**values):self.releases.append(values)
raw={'id':'offline-response','choices':[{'message':{'content':'{"relations":[]}'},'finish_reason':'stop'}],
     'usage':{'prompt_tokens':15,'completion_tokens':4,'total_tokens':19}}
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);key=root/'test_key';key.write_text('offline-test-key')
    environment={'SILICONFLOW_KEY_FILE':str(key),'FEEDBACK_API_STREAM':'0',
      'FEEDBACK_API_RETRIES':'2','FEEDBACK_API_TEMPERATURE':'.7','FEEDBACK_API_TOP_P':'.8',
      'FEEDBACK_API_TOP_K':'20','FEEDBACK_API_MIN_P':'0','FEEDBACK_API_FREQUENCY_PENALTY':'0',
      'FEEDBACK_MAX_ERRORS':'none','FEEDBACK_MAX_OUTPUT_TOKENS':'8192'}
    received=[]
    def success(url,**kwargs):received.append((url,kwargs['json']));return Response(200,raw)
    proxy=types.SimpleNamespace(post=success,RequestException=pilot.requests.RequestException)
    pacer=Pacer()
    with patch.dict(os.environ,environment),patch.object(pilot,'ROOT',root),patch.object(paced,'ROOT',root), \
         patch.object(pilot,'requests',proxy),patch.object(paced,'requests',proxy),patch.object(paced,'PACER',pacer):
        (root/'logs').mkdir();pilot.SERVICE_BLOCKED=paced.SERVICE_BLOCKED=False
        a=pilot.call('test-model','prompt',root/'original.json')
        b=paced.call('test-model','prompt',root/'paced.json')
        assert received[0]==received[1]
        for field in ['relations','usage','model','endpoint','prompt_sha256','finish_reason',
                      'temperature','max_output_tokens','top_p','request_sampling','raw_content']:
            assert a[field]==b[field],field
        sequence=iter([Response(429,{'code':50602,'message':'TPM limit reached.'}),Response(200,raw)])
        proxy.post=lambda *args,**kwargs:next(sequence)
        with patch.object(paced.time,'sleep',lambda _:None):
            c=paced.call('test-model','same prompt',root/'recovered.json')
        assert not c.get('error') and c['relations']==[]
        assert pacer.releases[-2]['rate_failure'] and not pacer.releases[-1]['rate_failure']
        failure=json.loads((root/'recovered.http_failure0.json').read_text())
        assert failure['status_code']==429 and 'TPM' in failure['body']
        assert 'offline-test-key' not in json.dumps(failure)
    print('PASS: frozen payload/output equivalence, admission before each attempt, preserved 429 body, cooldown signal and usage reconciliation')
