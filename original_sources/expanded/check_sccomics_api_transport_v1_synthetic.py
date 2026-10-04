"""Owned fictional transport checks. No external key, HTTP or Torch import."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import traceback
import types


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    root=Path(__file__).absolute().parents[1];output=Path(args.output_dir).absolute()
    if '..' in Path(args.output_dir).parts or not output.is_relative_to(root/'research') or output.exists():raise ValueError('fresh research artificial output')
    output.mkdir(parents=True);source=(root/'src/sccomics_api_transport_v1.py').read_bytes()
    (output/'client.before.py.txt').write_bytes(source)
    spec=importlib.util.spec_from_file_location('_INVENTED_sccomics_transport',root/'src/sccomics_api_transport_v1.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    results=[]
    def check(name,fn):
        try:detail=fn();results.append({'id':name,'passed':True,'detail':detail})
        except BaseException as e:results.append({'id':name,'passed':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
    def rejected(fn):
        try:fn()
        except (api.ProbeError,UnicodeError) as e:return {'rejected':True,'type':type(e).__name__,'reason':str(e)}
        raise AssertionError('expected explicit failure')
    source_text='INVENTED: La La and B doping at 5 K.'
    graph={'entities':[{'id':'T1','type':'Main','spans':[{'quote':'La','occurrence':0}]},
                       {'id':'T2','type':'Element','spans':[{'quote':'La','occurrence':1}]},
                       {'id':'T3','type':'Doping','spans':[{'quote':'doping','occurrence':0}]},
                       {'id':'T4','type':'UnknownFutureLabel','spans':[{'quote':'B','occurrence':0},{'quote':'5 K','occurrence':0}]}],
           'relations':[{'id':'R1','type':'UnknownRelation','head':'E1','tail':'T4'},{'id':'R2','type':'Target','head':'T1','tail':'E2'}],
           'events':[{'id':'E1','type':'Doping','trigger':'T3','roles':[{'type':'Dopant','target':'T4'},{'type':'Dopant','target':'T4'},{'type':'UnknownRole','target':'E2'}]},
                     {'id':'E2','type':'UnknownEvent','trigger':'T3','roles':[]}]}
    check('complete_unknown_discontinuous_nested_shared_zero_and_repeated_roles_retained',lambda:api.validate_graph(graph,source_text))
    def many():
        g=copy.deepcopy(graph);g['events']=[{'id':'E'+str(i),'type':'Doping','trigger':'T3','roles':[]} for i in range(1,1502)]
        d=api.validate_graph(g,source_text);assert d['record_counts']['events']==1501;return d
    check('no_implicit_record_list_cap_1501_events',many)
    def same_signature():
        g=copy.deepcopy(graph);g['entities'].append(dict(copy.deepcopy(g['entities'][0]),id='T5'))
        g['events'].append(dict(copy.deepcopy(g['events'][0]),id='E3'));g['relations'].append(dict(copy.deepcopy(g['relations'][0]),id='R3'))
        d=api.validate_graph(g,source_text);assert d['record_counts']=={'entities':5,'relations':3,'events':3};return d
    check('identical_signatures_distinct_ids_not_deduplicated',same_signature)
    def order():
        g=copy.deepcopy(graph);g['entities'][0]['spans']=[{'quote':'La','occurrence':1},{'quote':'La','occurrence':0}]
        d=api.validate_graph(g,source_text);assert d['quotation_anchors_for_format_diagnostic_only']['T1']==[[13,15],[10,12]];return {'output_order_retained':True}
    check('segment_output_order_no_hull_merge_or_sort',order)
    for value in (True,1.0,-1,2):
        def bad_occ(value=value):
            g=copy.deepcopy(graph);g['entities'][0]['spans'][0]['occurrence']=value;return rejected(lambda:api.validate_graph(g,source_text))
        check('invalid_occurrence_'+repr(value),bad_occ)
    for mutation in ('missing_event_type','duplicate_id','bad_reference','null_roles','nonlist_entities','nonexistent_quote'):
        def invalid_graph(mutation=mutation):
            g=copy.deepcopy(graph)
            if mutation=='missing_event_type':g['events'][0].pop('type')
            elif mutation=='duplicate_id':g['entities'][1]['id']='T1'
            elif mutation=='bad_reference':g['relations'][0]['head']='E999'
            elif mutation=='null_roles':g['events'][0]['roles']=None
            elif mutation=='nonlist_entities':g['entities']={}
            else:g['entities'][0]['spans'][0]['quote']='INVENTED_NOT_IN_SOURCE'
            return rejected(lambda:api.validate_graph(g,source_text))
        check(mutation,invalid_graph)
    for raw in (b'{"x":1,"x":2}',b'{"nested":{"x":0,"x":1}}',b'{"x":NaN}',b'{"x":1e400}'):
        check('JSON_reject_'+raw.decode(),lambda raw=raw:rejected(lambda:api.unique_json(raw)))
    for value in (True,2.0,-1):
        check('request_integer_reject_'+repr(value),lambda value=value:rejected(lambda:api.validate_payload({'model':api.MODELS[0],'messages':[{'role':'user','content':'JSON '+source_text}],'stream':False,'max_tokens':value})))
    with tempfile.TemporaryDirectory(prefix='INVENTED_sccomics_api_') as temporary:
        owned=Path(temporary).resolve();fake_key='INVENTED_NOT_AN_ACTUAL_CREDENTIAL_123';key_reads=[];calls=[]
        class Pacer:
            def __init__(self):self.releases=[]
            def acquire(self,model,tokens):return {'model':model,'ticket':'INVENTED_TICKET','reserved_tokens':tokens}
            def release(self,ticket,**kwargs):self.releases.append(kwargs)
        def key_reader():key_reads.append(True);return fake_key
        def frozen(stream=False,mode='native_graph'):
            payload={'model':api.MODELS[0],'messages':[{'role':'user','content':'Return complete JSON for '+source_text}],'stream':stream,'max_tokens':2048,'enable_thinking':False,'response_format':{'type':'json_object'}}
            p=api.serialized(payload);s=source_text.encode();f={'version':1,'purpose':'synthetic_format_probe','source_is_invented':True,'endpoint':api.ENDPOINT,'payload_sha256':api.digest(p),'source_sha256':api.digest(s),'client_sha256':api.digest(source),'pacing_source_sha256':{n:api.digest((root/n).read_bytes()) for n in api.PACING_PATHS},'output_mode':mode,'schema_sha256':api.digest(api.serialized(api.GRAPH_SCHEMA if mode=='native_graph' else {'type':'object'})),
             'limits':{'idle_timeout_s':1,'wall_timeout_s':5,'max_response_bytes':1000000,'reserved_tokens':len(p)+4096,'pacing_policy':'shared_fair_conservative_v1'},'root_authorized_actual_synthetic_probe':False}
            return p,s,api.serialized(f)
        def response(content=None,finish='stop'):
            return {'id':'INVENTED_RESPONSE','model':api.MODELS[0],'choices':[{'index':0,'message':{'content':json.dumps(graph) if content is None else content},'finish_reason':finish}], 'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300}}
        def run_case(name,body=None,status=200,stream=False,exception=None,header=None,mode='native_graph',limits=None):
            p,s,f=frozen(stream,mode);pace=Pacer()
            if limits:
                value=api.unique_json(f);value['limits'].update(limits);f=api.serialized(value)
            raw=api.serialized(response()) if body is None else body
            def exchange(payload,key,options,sink):
                calls.append(name);assert payload==p and key==fake_key
                for at in range(0,len(raw),19):sink(raw[at:at+19])
                if exception:raise exception
                return {'http_status':status,'safe_response_headers':header or {'content-type':'application/json'}}
            result=api.run_probe(owned,p,s,f,owned/name,exchange=exchange,key_reader=key_reader,pacer=pace)
            return result,pace,(p,s,f),owned/name
        success=None
        def plain():
            nonlocal success
            success=run_case('plain');r,pace,_,directory=success
            assert r['passed'] and r['graph_or_output']==graph and len(pace.releases)==1 and r['usage']['total_tokens']==300
            assert (directory/'response_raw.safe').read_bytes()==api.serialized(response())
            return {'complete_public_input_and_raw_response_preserved':True,'actual_API':False}
        check('nonstream_complete_graph_preserved',plain)
        def cache_hit():
            r,pace,inputs,directory=success;n=len(key_reads)
            found=api.run_probe(owned,*inputs,directory,exchange=lambda *x:(_ for _ in ()).throw(AssertionError('cache must not invoke HTTP')),key_reader=lambda:(_ for _ in ()).throw(AssertionError('cache must not read key')),pacer=Pacer())
            assert found['passed'] and len(key_reads)==n;return {'no_key_or_HTTP_on_verified_success_cache':True}
        check('cache_identity_hash_and_no_key_read',cache_hit)
        def cache_tamper():
            _,_,inputs,directory=success;(directory/'result.json').write_bytes(b'{}');return rejected(lambda:api.run_probe(owned,*inputs,directory,exchange=lambda*x:None,key_reader=key_reader,pacer=Pacer()))
        check('cache_tamper_rejected',cache_tamper)
        def length():
            p,s,f=frozen();v=api.unique_json(p);v['max_tokens']=2049;p=api.serialized(v);z=api.unique_json(f);z['payload_sha256']=api.digest(p);f=api.serialized(z)
            pace=Pacer();body=api.serialized(response(finish='length'))
            def exchange(*args):args[3](body);return {'http_status':200}
            r=api.run_probe(owned,p,s,f,owned/'length',exchange=exchange,key_reader=key_reader,pacer=pace)
            assert not r['passed'] and r['graph_or_output'] is None and r['finish_reason']=='length' and r['usage']['total_tokens']==300
            n=len(key_reads);rejected(lambda:api.run_probe(owned,p,s,f,owned/'length',exchange=exchange,key_reader=key_reader,pacer=Pacer()))
            rejected(lambda:api.run_probe(owned,p,s,f,owned/'length_again',exchange=exchange,key_reader=key_reader,pacer=Pacer()))
            assert len(key_reads)==n;return {'failed_length_full_raw_usage_preserved_no_identical_retry':True}
        check('length_terminal_no_empty_success_no_retry',length)
        def unique_variant(name,body,status=200,stream=False,exception=None,header=None,mode='native_graph',cap=None):
            # Distinct prompt variant makes each fictional probe a distinct
            # request; same failed-request attempts are separately tested above.
            p,s,f=frozen(stream,mode);v=api.unique_json(p);v['messages'][0]['content']+=' '+name;p=api.serialized(v);z=api.unique_json(f);z['payload_sha256']=api.digest(p)
            if cap:z['limits']['max_response_bytes']=cap
            f=api.serialized(z);pace=Pacer()
            def exchange(payload,key,options,sink):
                sink(body)
                if exception:raise exception
                return {'http_status':status,'safe_response_headers':header or {}}
            r=api.run_probe(owned,p,s,f,owned/name,exchange=exchange,key_reader=key_reader,pacer=pace)
            return r,pace,owned/name
        def sse_body(finish='stop',done=True):
            content=json.dumps(graph,ensure_ascii=False);parts=[content[:23],content[23:]];chunks=[]
            for p in parts:chunks.append({'id':'INVENTED_SSE','model':api.MODELS[0],'choices':[{'index':0,'delta':{'content':p},'finish_reason':None}]})
            chunks.append({'choices':[{'index':0,'delta':{},'finish_reason':finish}]});chunks.append({'choices':[],'usage':{'total_tokens':321}})
            return (''.join('data: '+json.dumps(x)+'\n\n' for x in chunks)+('data: [DONE]\n\n' if done else '')).encode()
        def sse():
            raw=sse_body();r,_,d=unique_variant('sse_complete',raw,stream=True);assert r['passed'] and r['usage']['total_tokens']==321 and (d/'response_raw.safe').read_bytes()==raw
            return {'SSE_events_DONE_usage_complete':True}
        check('complete_SSE_raw_and_assembled_graph',sse)
        for name,raw in (('sse_no_DONE',sse_body(done=False)),('sse_length',sse_body(finish='length')),('sse_content_after_DONE',sse_body()+b'data: {}\n\n')):
            def bad_stream(name=name,raw=raw):
                r,_,d=unique_variant(name,raw,stream=True);assert not r['passed'] and r['graph_or_output'] is None and (d/'response_raw.safe').read_bytes()==raw;return {'failed_explicitly':True}
            check(name,bad_stream)
        def redaction():
            raw=('HTTP '+fake_key+'\n'+json.dumps({'error':''.join('\\u%04x'%ord(c) for c in fake_key)})+'\n').encode()
            # Escaped raw JSON token with actual Unicode escapes, not a literal
            # escape string, exercises the decoded-string redaction path.
            raw+=('"'+''.join('\\u%04x'%ord(c) for c in fake_key)+'"').encode()
            r,pace,d=unique_variant('provider429_secret_echo',raw,status=429,header={'retry-after':'2','authorization':fake_key})
            assert not r['passed'] and pace.releases[0]['rate_failure'] is True
            for p in d.iterdir():assert fake_key.encode() not in p.read_bytes()
            assert 'authorization' not in r['safe_response_headers'];return {'full_safe_failure_bytes_saved_with_redaction':True,'headers_not_saved':True}
        check('key_echo_plain_escaped_and_headers_redacted',redaction)
        def timeout():
            raw=b'data: {"partial":true}\n\n';r,_,d=unique_variant('timeout_partial',raw,stream=True,exception=TimeoutError('INVENTED timeout '+fake_key))
            assert not r['passed'] and (d/'response_raw.safe').read_bytes()==raw and fake_key not in r['error'];return {'partial_raw_and_safe_timeout_retained':True}
        check('timeout_preserves_partial_response_and_safe_error',timeout)
        for name,content in (('invalid_top_list','[]'),('invalid_graph_shape','{"entities":null,"relations":[],"events":[]}'),('content_duplicate_key','{"entities":[],"entities":[],"relations":[],"events":[]}')):
            def bad_output(name=name,content=content):
                raw=api.serialized(response(content=content));r,_,d=unique_variant(name,raw);assert not r['passed'] and r['graph_or_output'] is None and (d/'response_raw.safe').read_bytes()==raw;return {'original_invalid_output_retained':True}
            check(name,bad_output)
        def byte_limit():
            r,_,d=unique_variant('explicit_response_byte_budget',b'x'*200,cap=100);assert not r['passed'] and len((d/'response_raw.safe').read_bytes())==200;return {'explicit_budget_failure_retains_received_chunk':True}
        check('explicit_raw_byte_budget_not_silent_truncation',byte_limit)
        def critic_mode():
            raw=api.serialized(response(content='{"critic_tickets":[]}'));r,_,_=unique_variant('generic_critic_JSON',raw,mode='json_object');assert r['passed'] and r['parsed_output']=={'critic_tickets':[]};return {'generic_strict_JSON_separate_from_native_graph':True}
        check('critic_generic_JSON_object_mode',critic_mode)
        def default_denied():
            p,s,f=frozen();return rejected(lambda:api.run_probe(owned,p,s,f,owned/'actual_without_authority'))
        check('default_actual_request_denied_before_key_and_HTTP',default_denied)
        def symlink():
            alias=owned/'alias';alias.symlink_to(owned,target_is_directory=True);p,s,f=frozen();return rejected(lambda:api.run_probe(owned,p,s,f,alias/'outside',exchange=lambda*x:None,key_reader=key_reader,pacer=Pacer()))
        check('symlink_output_boundary_rejected',symlink)
        def traversal():
            p,s,f=frozen();return rejected(lambda:api.run_probe(owned,p,s,f,owned/'a/../outside',exchange=lambda*x:None,key_reader=key_reader,pacer=Pacer()))
        check('parent_traversal_output_rejected',traversal)
        def binding():
            p,s,f=frozen();return rejected(lambda:api.run_probe(owned,p+b' ',s,f,owned/'wrong_binding',exchange=lambda*x:None,key_reader=key_reader,pacer=Pacer()))
        check('exact_payload_byte_binding_rejected_on_change',binding)
        # All execution artifacts are fake and are copied only after checks.
        import shutil
        shutil.copytree(owned,output/'INVENTED_outputs',symlinks=True)
    if (root/'src/sccomics_api_transport_v1.py').read_bytes()!=source:raise RuntimeError('Source changed while executing checks')
    result={'identity':'ACTUAL_EXECUTED_OWNED_SYNTHETIC_TRANSPORT_SOURCE_CHECKS_NOT_API_INFERENCE','source_sha256':hashlib.sha256(source).hexdigest(),'checks':results,'total':len(results),'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),'actual_HTTP_or_external_key_or_SC_or_Torch_or_scientific_score':False}
    (output/'actual_results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    files=[]
    for p in sorted(output.rglob('*')):
        if not p.is_symlink() and p.is_file() and p.name!='.DS_Store':
            raw=p.read_bytes();files.append({'path':str(p.relative_to(output)),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)})
    (output/'output_manifest.json').write_text(json.dumps({'files':files,'symlinks_metadata_only':[{'path':str(p.relative_to(output)),'target':str(p.readlink())} for p in sorted(output.rglob('*')) if p.is_symlink()],'scope':'SOURCE_AND_OWNED_ARTIFICIAL_ONLY'},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'total':result['total'],'passed':result['passed'],'failed':result['failed'],'source_sha256':result['source_sha256']}))
    return 0 if result['failed']==0 else 1

if __name__=='__main__':raise SystemExit(main())
