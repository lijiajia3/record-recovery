"""Owned fake HTTPS and shared FIFO SOURCE checks; never an API request."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,pathlib,sys,tempfile,traceback,types
from unittest.mock import patch

def need(value,message):
    if not value:raise AssertionError(message)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];out=pathlib.Path(args.output_dir).absolute()
    need('..' not in pathlib.Path(args.output_dir).parts and out.is_relative_to(root/'research') and not out.exists(),'fresh research output')
    out.mkdir(parents=True);client=(root/'src/sccomics_api_transport_v1.py').read_bytes()
    (out/'client.before.py.txt').write_bytes(client)
    spec=importlib.util.spec_from_file_location('_INVENTED_transport_http',root/'src/sccomics_api_transport_v1.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    rows=[]
    def check(name,fn):
        try:rows.append({'id':name,'passed':True,'detail':fn()})
        except BaseException as e:rows.append({'id':name,'passed':False,'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
    class Sock:
        def __init__(self):self.timeouts=[]
        def settimeout(self,value):self.timeouts.append(value)
    class Response:
        status=200
        def __init__(self):self.blocks=[b'INVENTED_FULL_',b'RESPONSE',b'']
        def getheaders(self):return [('Content-Type','application/json'),('Authorization','INVENTED_HEADER_NOT_SAVED')]
        def read1(self,n):need(n==65536,'explicit read block');return self.blocks.pop(0)
    created=[]
    class HTTPS:
        def __init__(self,host,**kwargs):self.host=host;self.kwargs=kwargs;self.sock=Sock();self.closed=False;self.requests=[];created.append(self)
        def request(self,*args,**kwargs):self.requests.append((args,kwargs))
        def getresponse(self):return Response()
        def close(self):self.closed=True
    options={'idle_timeout_s':2,'wall_timeout_s':10}
    def exact_route():
        chunks=[]
        with patch.object(api.http.client,'HTTPSConnection',HTTPS),patch.object(api.ssl,'create_default_context',lambda:'INVENTED_SSL'):
            result=api.https_exchange(b'INVENTED_BODY','INVENTED_KEY',options,chunks.append)
        c=created[-1];need(c.host=='api.siliconflow.cn' and c.kwargs['context']=='INVENTED_SSL','fixed TLS host/context')
        args,kw=c.requests[0];need(args==('POST','/v1/chat/completions') and kw['body']==b'INVENTED_BODY','exact method/path/bytes')
        need(kw['headers']['Authorization']=='Bearer INVENTED_KEY','key only transient transport header')
        need(c.closed and b''.join(chunks)==b'INVENTED_FULL_RESPONSE' and len(c.requests)==1,'one attempt complete bytes close')
        need(result['safe_response_headers']=={'content-type':'application/json'},'no auth headers returned')
        return {'exact_fixed_endpoint':True,'actual_HTTP':False}
    check('HTTPS_exact_route_payload_single_attempt_and_header_policy',exact_route)
    def timeout():
        class Failure(HTTPS):
            def getresponse(self):raise TimeoutError('INVENTED_TIMEOUT')
        with patch.object(api.http.client,'HTTPSConnection',Failure),patch.object(api.ssl,'create_default_context',lambda:None):
            try:api.https_exchange(b'X','INVENTED_KEY',options,lambda b:None)
            except TimeoutError:pass
            else:raise AssertionError('timeout must propagate')
        need(created[-1].closed and len(created[-1].requests)==1,'timeout closed/no retry');return True
    check('HTTPS_timeout_closed_no_retry',timeout)
    def byte_abort():
        with patch.object(api.http.client,'HTTPSConnection',HTTPS),patch.object(api.ssl,'create_default_context',lambda:None):
            try:api.https_exchange(b'X','INVENTED_KEY',options,lambda b:(_ for _ in ()).throw(api.ProbeError('INVENTED_BUDGET')))
            except api.ProbeError:pass
            else:raise AssertionError('sink failure must propagate')
        need(created[-1].closed,'sink failure closed');return True
    check('HTTPS_response_budget_callback_closed',byte_abort)
    def wall_timeout():
        ticks=iter([0,11])
        with patch.object(api.http.client,'HTTPSConnection',HTTPS),patch.object(api.ssl,'create_default_context',lambda:None),patch.object(api.time,'monotonic',lambda:next(ticks)):
            try:api.https_exchange(b'X','INVENTED_KEY',options,lambda b:None)
            except api.ProbeError:pass
            else:raise AssertionError('wall timeout must propagate')
        need(created[-1].closed,'wall failure closed');return True
    check('HTTPS_explicit_wall_boundary',wall_timeout)
    with tempfile.TemporaryDirectory(prefix='INVENTED_FIFO_') as temporary:
        owned=pathlib.Path(temporary).resolve();(owned/'src').mkdir();old={n:sys.modules.get(n) for n in ('siliconflow_pacing','siliconflow_fair_pacing')}
        pins={}
        try:
            for name in api.PACING_PATHS:
                raw=(root/name).read_bytes();pins[name]=api.digest(raw);(owned/name).write_bytes(raw)
                module_name=pathlib.Path(name).stem;m=types.ModuleType(module_name);m.__file__=str(owned/name);sys.modules[module_name]=m;exec(compile(raw,str(owned/name),'exec'),m.__dict__)
            p=api.actual_pacer(owned,api.MODELS[0],pins);native=sys.modules['siliconflow_pacing'];fair=sys.modules['siliconflow_fair_pacing'];tickets=[]
            def two():
                tickets.extend(p.acquire(api.MODELS[0],1000) for _ in range(2))
                with p.locked(api.MODELS[0]) as state:
                    d=native.admission_decision(state,api.time.time(),1000,p.config);need(not d['fits_now'] and len(state['active'])==2,'shared max two')
                return {'two_shared_tickets':True,'third_not_admitted':True,'no_actual_provider_quota_claim':True}
            check('shared_original_FIFO_two_active_concurrency',two)
            def fifo():
                state={'client_tpm':60000,'entries':[],'active':{},'queue':[{'ticket':'first','last_seen':10,'alive':True},{'ticket':'second','last_seen':10,'alive':True}]}
                need(not fair.fair_admission_decision(state,10,1,'second',p.config)['fits_now'],'later cannot overtake')
                need(fair.fair_admission_decision(state,10,1,'first',p.config)['fits_now'],'front can admit');return True
            check('original_FIFO_queue_order_retained',fifo)
            def rate_failure():
                p.release(tickets[0],usage={'total_tokens':17},rate_failure=True,retry_after=2)
                with p.locked(api.MODELS[0]) as state:
                    need(state['client_tpm']==30000 and state['cooldown_until']>api.time.time()+60,'failure cooldown/budget')
                    need(state['entries'][0]['tokens']==17,'actual reported usage adjustment')
                return True
            check('shared_usage_429_cooldown_no_automatic_retry',rate_failure)
            def conflict():
                key=hashlib.sha256(api.MODELS[0].encode()).hexdigest()[:20];path=owned/'logs/siliconflow_pacing'/(key+'.json');state=json.loads(path.read_text());state['config']['client_rpm']=99;path.write_text(json.dumps(state));before=path.read_bytes()
                try:
                    with p.locked(api.MODELS[0]):pass
                except api.ProbeError:pass
                else:raise AssertionError('incompatible profile must reject')
                need(path.read_bytes()==before,'mismatched profile not overwritten');return True
            check('existing_shared_config_conflict_failclosed_unchanged',conflict)
            for name,sha in pins.items():need(api.digest((root/name).read_bytes())==sha,'original pacing source changed')
        finally:
            for name,previous in old.items():
                if previous is None:sys.modules.pop(name,None)
                else:sys.modules[name]=previous
    need((root/'src/sccomics_api_transport_v1.py').read_bytes()==client,'client bytes changed')
    result={'scope':'OWNED_SYNTHETIC_SOURCE_ONLY_NO_HTTP_KEY_NN_SC','checks':rows,'total':len(rows),'passed':sum(x['passed'] for x in rows),'failed':sum(not x['passed'] for x in rows),'client_sha256':api.digest(client)}
    (out/'actual_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('total','passed','failed','client_sha256')}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
