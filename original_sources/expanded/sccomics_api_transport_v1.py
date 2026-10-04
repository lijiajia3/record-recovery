"""SiliconFlow synthetic format-probe transport; no requests on import.

Stdlib HTTPS only. Actual requests require an explicit frozen root-authorized
synthetic probe. This module has no corpus/Gold, NN, tokenizer or scoring loader.
Original FIFO modules/profiles are immutable; their shared fair admission class
is loaded lazily only for an actual request, with new-model conservative limits.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import time
import urllib.parse

ENDPOINT='https://api.siliconflow.cn/v1/chat/completions'
KEY_PATH=Path('/Users/jiajia/.siliconflow_feedback_key')
MODELS=('deepseek-ai/DeepSeek-V4-Pro','zai-org/GLM-5.3')
CLIENT_PATH='src/sccomics_api_transport_v1.py'
PACING_PATHS=('src/siliconflow_pacing.py','src/siliconflow_fair_pacing.py')
PACING_CONFIG={'initial_client_tpm':60000,'minimum_client_tpm':30000,
 'maximum_client_tpm':60000,'client_rpm':6,'max_inflight':2,
 'cooldown_seconds':65,'healthy_growth_interval_seconds':300,
 'healthy_growth_multiplier':1.0,'rate_failure_multiplier':.5}
MODEL_PACING={model:dict(PACING_CONFIG) for model in MODELS}

def _obj(properties,required):
    return {'type':'object','properties':properties,'required':required,'additionalProperties':False}
_string={'type':'string','minLength':1}
_span=_obj({'quote':_string,'occurrence':{'type':'integer','minimum':0}},['quote','occurrence'])
GRAPH_SCHEMA=_obj({
 'entities':{'type':'array','items':_obj({'id':_string,'type':_string,'spans':{'type':'array','minItems':1,'items':_span}},['id','type','spans'])},
 'relations':{'type':'array','items':_obj({'id':_string,'type':_string,'head':_string,'tail':_string},['id','type','head','tail'])},
 'events':{'type':'array','items':_obj({'id':_string,'type':_string,'trigger':_string,'roles':{'type':'array','items':_obj({'type':_string,'target':_string},['type','target'])}},['id','type','trigger','roles'])},
 },['entities','relations','events'])

class ProbeError(ValueError):pass
class CachedFailure(ProbeError):pass
class ResponseProblem(ProbeError):
    def __init__(self,message,metadata):
        super().__init__(message);self.metadata=metadata

def require(ok,message):
    if not ok:raise ProbeError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()
def serialized(obj):
    finite_json(obj)
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def finite_json(obj):
    stack=[obj]
    while stack:
        value=stack.pop()
        if type(value) is dict:
            require(all(type(k) is str for k in value),'JSON string keys required');stack.extend(value.values())
        elif type(value) is list:stack.extend(value)
        elif type(value) is float:require(math.isfinite(value),'nonfinite JSON number')
        else:require(type(value) in (str,int,bool,type(None)),'JSON primitive type required')
    return obj
def unique_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate JSON key');d[k]=v
        return d
    def invalid(_):raise ProbeError('nonfinite JSON constant')
    try:return finite_json(json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid))
    except (UnicodeError,json.JSONDecodeError,RecursionError,ValueError) as e:
        if isinstance(e,ProbeError):raise
        raise ProbeError('invalid JSON encoding or syntax') from None
def sha_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'explicit SHA256 required')
def integer(value,minimum=0):require(type(value) is int and value>=minimum,'integer range/type required')
def number(value,minimum,maximum):
    require(type(value) in (int,float) and math.isfinite(value) and minimum<=value<=maximum,'finite numeric range/type required')
def closed(obj,keys):require(type(obj) is dict and set(obj)==set(keys),'closed JSON object schema')
def text(value):require(type(value) is str and len(value)>0,'nonempty string required')

def quote_occurrence(source,quote,occurrence):
    text(quote);integer(occurrence);at=-1
    for _ in range(occurrence+1):
        at=source.find(quote,at+1);require(at>=0,'source quote occurrence is out of range')
    return [at,at+len(quote)]

def validate_graph(obj,source):
    """Format/quotation validation only; no projection, Gold or model score.

    Unknown labels, distinct IDs with identical signatures, repeated roles,
    shared triggers, zero-role events and nested/cyclic E references are retained.
    Span list order is retained; no hull, sorting, deduplication or list cap.
    """
    closed(obj,('entities','relations','events'));require(type(source) is str,'source text string')
    for k in obj:require(type(obj[k]) is list,'complete native record lists required')
    ids={};anchors={}
    for family,prefix in (('entities','T'),('relations','R'),('events','E')):
        for record in obj[family]:
            require(type(record) is dict,'record object required');identifier=record.get('id')
            require(type(identifier) is str and re.fullmatch(prefix+'[1-9][0-9]*',identifier) is not None,'native record ID prefix/range')
            require(identifier not in ids,'duplicate native record ID');ids[identifier]=prefix
            text(record.get('type'))
    for record in obj['entities']:
        closed(record,('id','type','spans'));require(type(record['spans']) is list and record['spans'],'entity nonempty span list')
        values=[]
        for span in record['spans']:
            closed(span,('quote','occurrence'));values.append(quote_occurrence(source,span['quote'],span['occurrence']))
        anchors[record['id']]=values
    for record in obj['relations']:
        closed(record,('id','type','head','tail'))
        for key in ('head','tail'):require(type(record[key]) is str and ids.get(record[key]) in ('T','E'),'relation native T/E reference required')
    for record in obj['events']:
        closed(record,('id','type','trigger','roles'));require(type(record['trigger']) is str and ids.get(record['trigger'])=='T','event trigger T reference required')
        require(type(record['roles']) is list,'event roles list required')
        for role in record['roles']:
            closed(role,('type','target'));text(role['type']);require(type(role['target']) is str and ids.get(role['target']) in ('T','E'),'event role native T/E reference required')
    return {'quotation_anchors_for_format_diagnostic_only':anchors,'record_counts':{k:len(v) for k,v in obj.items()},'unknown_label_strings_retained':True,'projection_or_Gold_scoring_completed':False}

def validate_payload(payload):
    require(type(payload) is dict,'request object required')
    known={'model','messages','stream','max_tokens','enable_thinking','thinking_budget','reasoning_effort','temperature','top_p','top_k','frequency_penalty','n','response_format','stop'}
    require(set(payload)<=known and {'model','messages','max_tokens','stream'}<=set(payload),'explicit request fields; no headers/credential/unknown control')
    require(payload['model'] in MODELS,'declared account-model ID required');integer(payload['max_tokens'],1)
    require(type(payload['stream']) is bool,'explicit stream boolean');require(type(payload['messages']) is list and payload['messages'],'messages required')
    for m in payload['messages']:
        closed(m,('role','content'));require(m['role'] in ('system','user','assistant'),'message role');text(m['content'])
    if 'n' in payload:require(type(payload['n']) is int and payload['n']==1,'single complete graph output required')
    if 'enable_thinking' in payload:require(type(payload['enable_thinking']) is bool,'thinking boolean')
    if 'thinking_budget' in payload:integer(payload['thinking_budget'],128);require(payload['thinking_budget']<=32768,'documented thinking budget range')
    if 'reasoning_effort' in payload:require(payload['reasoning_effort'] in ('high','max'),'documented effort values; exact-model support unproven')
    for k,lo,hi in (('temperature',0,2),('top_p',0,1),('top_k',0,100),('frequency_penalty',-2,2)):
        if k in payload:number(payload[k],lo,hi)
    if 'stop' in payload:
        v=payload['stop'];require(type(v) is str or (type(v) is list and len(v)<=4 and all(type(x) is str for x in v)),'documented stop shape')
    if 'response_format' in payload:
        r=payload['response_format'];require(type(r) is dict and r.get('type') in ('text','json_object','json_schema'),'documented response format')
        if r['type']=='json_schema':closed(r,('type','json_schema'));require(type(r['json_schema']) is dict,'JSON schema object')
        else:closed(r,('type',))
    finite_json(payload);return payload

def safe_bytes(raw,key):
    """Keep complete body except credential redaction, including escaped JSON.

    No request/response Authorization headers are stored. JSON/SSE quoted secret
    strings are recognized after unescaping; unaffected body bytes stay exact.
    """
    require(type(raw) is bytes and type(key) is str,'redaction inputs')
    if not key:return raw,False
    changed=False
    variants={key.encode(),json.dumps(key,ensure_ascii=True)[1:-1].encode(),urllib.parse.quote(key,safe='').encode()}
    out=raw
    for needle in variants:
        if needle and needle in out:out=out.replace(needle,b'[REDACTED]');changed=True
    def fix_string(match):
        nonlocal changed
        token=match.group(0)
        try:value=json.loads(token)
        except (ValueError,UnicodeError):return token
        if key in value:
            changed=True;return json.dumps(value.replace(key,'[REDACTED]'),ensure_ascii=False).encode()
        return token
    out=re.sub(rb'"(?:[^"\\]|\\.)*"',fix_string,out)
    return out,changed

def checked_bytes(path):
    p=Path(path).absolute();require(not any(x.is_symlink() for x in (p,*p.parents)),'input/output path alias rejected')
    before=p.stat(follow_symlinks=False);require(stat.S_ISREG(before.st_mode),'regular file required')
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        start=os.fstat(fd);require((start.st_dev,start.st_ino)==(before.st_dev,before.st_ino),'opened file identity changed')
        chunks=[]
        while b:=os.read(fd,1048576):chunks.append(b)
        end=os.fstat(fd);after=p.stat(follow_symlinks=False)
        require(all(getattr(start,k)==getattr(end,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'file changed during capture')
        return b''.join(chunks)
    finally:os.close(fd)
def write_new(path,raw):
    require(not any(x.is_symlink() for x in (path,*path.parents)),'output alias rejected')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        at=0
        while at<len(raw):at+=os.write(fd,raw[at:])
        os.fsync(fd)
    finally:os.close(fd)

def validate_freeze(freeze,payload_raw,source_raw):
    closed(freeze,('version','purpose','source_is_invented','endpoint','payload_sha256','source_sha256','client_sha256','pacing_source_sha256','output_mode','schema_sha256','limits','root_authorized_actual_synthetic_probe'))
    require(type(freeze['version']) is int and freeze['version']==1 and freeze['purpose']=='synthetic_format_probe' and freeze['source_is_invented'] is True,'synthetic-only frozen scope')
    require(freeze['endpoint']==ENDPOINT and type(freeze['root_authorized_actual_synthetic_probe']) is bool,'official fixed endpoint and explicit authority')
    for k in ('payload_sha256','source_sha256','client_sha256','schema_sha256'):sha_string(freeze[k])
    require(freeze['payload_sha256']==digest(payload_raw) and freeze['source_sha256']==digest(source_raw),'exact public input byte binding differs')
    require(freeze['output_mode'] in ('native_graph','json_object'),'explicit output parser mode')
    expected=GRAPH_SCHEMA if freeze['output_mode']=='native_graph' else {'type':'object'}
    require(freeze['schema_sha256']==digest(serialized(expected)),'frozen format schema differs')
    require(type(freeze['pacing_source_sha256']) is dict and set(freeze['pacing_source_sha256'])==set(PACING_PATHS),'exact two immutable fair pacing source pins')
    for v in freeze['pacing_source_sha256'].values():sha_string(v)
    limits=freeze['limits'];closed(limits,('idle_timeout_s','wall_timeout_s','max_response_bytes','reserved_tokens','pacing_policy'))
    for k in ('idle_timeout_s','wall_timeout_s'):number(limits[k],.01,86400)
    require(limits['wall_timeout_s']>=limits['idle_timeout_s'],'wall timeout must cover one idle interval')
    integer(limits['max_response_bytes'],1);integer(limits['reserved_tokens'],1)
    require(limits['pacing_policy']=='shared_fair_conservative_v1','explicit fair pacing policy')
    payload=validate_payload(unique_json(payload_raw))
    require(limits['reserved_tokens']>=len(payload_raw)+payload['max_tokens']+payload.get('thinking_budget',0),'declared conservative reservation excludes no payload/output bytes')
    source=source_raw.decode('utf-8','strict')
    require(any(source in m['content'] for m in payload['messages']),'supplied source must be present verbatim in request')
    return payload,source

def _parse_response(raw,stream,source,output_mode,state):
    """Return all JSON/SSE metadata and explicit terminal/format diagnostics."""
    events=[];usage=None;content='';reasoning='';finish=None;reported_models=[];response_ids=[]
    if stream:
        done=False
        decoded=raw.decode('utf-8','strict').replace('\r\n','\n').replace('\r','\n')
        for block in decoded.split('\n\n'):
            data='\n'.join(line[5:].lstrip(' ') for line in block.split('\n') if line.startswith('data:'))
            if not data:continue
            if data=='[DONE]':require(not done,'duplicate SSE DONE');done=True;continue
            require(not done,'SSE content after DONE');obj=unique_json(data);events.append(obj)
            require(type(obj) is dict and 'error' not in obj,'provider stream error')
            if obj.get('usage') is not None:usage=obj['usage']
            if obj.get('model') is not None:reported_models.append(obj['model'])
            if obj.get('id') is not None:response_ids.append(obj['id'])
            choices=obj.get('choices');require(type(choices) is list,'SSE choices list required')
            if not choices:continue
            require(len(choices)==1 and type(choices[0]) is dict and choices[0].get('index')==0 and type(choices[0].get('index')) is int,'single SSE choice0')
            choice=choices[0];delta=choice.get('delta');require(type(delta) is dict,'SSE delta object')
            for name in ('content','reasoning_content'):
                v=delta.get(name)
                require(v is None or type(v) is str,'stream textual delta required')
                if v is not None:
                    if name=='content':content+=v
                    else:reasoning+=v
            if choice.get('finish_reason') is not None:
                require(finish is None,'multiple SSE finish records');finish=choice['finish_reason']
            state.update(usage=usage,finish_reason=finish,raw_content=content,raw_reasoning_content=reasoning,reported_models=reported_models,response_ids=response_ids,SSE_JSON_events=events)
        state.update(usage=usage,finish_reason=finish,raw_content=content,raw_reasoning_content=reasoning,reported_models=reported_models,response_ids=response_ids,SSE_JSON_events=events)
        require(done,'incomplete SSE: DONE missing')
    else:
        obj=unique_json(raw);events=[obj];require(type(obj) is dict and 'error' not in obj,'provider JSON error')
        choices=obj.get('choices');require(type(choices) is list and len(choices)==1,'single completion choice required')
        choice=choices[0];require(type(choice) is dict,'choice object');message=choice.get('message')
        require(type(message) is dict and type(message.get('content')) is str,'assistant textual content required')
        content=message['content'];reasoning=message.get('reasoning_content') or '';require(type(reasoning) is str,'reasoning text required')
        finish=choice.get('finish_reason');usage=obj.get('usage')
        if obj.get('model') is not None:reported_models=[obj['model']]
        if obj.get('id') is not None:response_ids=[obj['id']]
    state.update(usage=usage,finish_reason=finish,raw_content=content,raw_reasoning_content=reasoning,reported_models=reported_models,response_ids=response_ids,SSE_JSON_events=events if stream else None)
    if usage is not None:
        require(type(usage) is dict,'usage object');finite_json(usage)
        for k in ('prompt_tokens','completion_tokens','total_tokens'):
            if k in usage:integer(usage[k])
    require(all(type(v) is str for v in reported_models+response_ids),'provider metadata strings')
    require(finish=='stop','provider did not complete: finish_reason='+str(finish))
    output=unique_json(content);state['parsed_output']=output
    require(type(output) is dict,'top output JSON object required')
    diagnostic=validate_graph(output,source) if output_mode=='native_graph' else {'strict_JSON_object':True,'native_projection_or_scoring':False}
    return {'parsed_output':output,'format_diagnostic':diagnostic,'usage':usage,'usage_reported':usage is not None,'finish_reason':finish,
            'raw_content':content,'raw_reasoning_content':reasoning,'reported_models':reported_models,'response_ids':response_ids,'SSE_JSON_events':events if stream else None}

def parse_response(raw,stream,source,output_mode):
    state={'usage':None,'finish_reason':None,'parsed_output':None}
    try:return _parse_response(raw,stream,source,output_mode,state)
    except (ProbeError,UnicodeError) as error:
        raise ResponseProblem(str(error),state) from None

def https_exchange(payload_raw,key,limits,on_chunk):
    """One HTTPS attempt. Never redirects, retries or prints request headers."""
    conn=None;deadline=time.monotonic()+limits['wall_timeout_s']
    try:
        conn=http.client.HTTPSConnection('api.siliconflow.cn',timeout=limits['idle_timeout_s'],context=ssl.create_default_context())
        conn.request('POST','/v1/chat/completions',body=payload_raw,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept-Encoding':'identity'})
        require(time.monotonic()<deadline,'wall timeout before response')
        if conn.sock:conn.sock.settimeout(min(limits['idle_timeout_s'],max(.001,deadline-time.monotonic())))
        response=conn.getresponse()
        headers={k.lower():v for k,v in response.getheaders() if k.lower() in ('content-type','retry-after','x-siliconcloud-trace-id')}
        while True:
            remaining=deadline-time.monotonic();require(remaining>0,'wall response timeout')
            if conn.sock:conn.sock.settimeout(min(limits['idle_timeout_s'],remaining))
            b=response.read1(65536)
            if not b:break
            on_chunk(b)
        return {'http_status':response.status,'safe_response_headers':headers}
    finally:
        if conn is not None:conn.close()

def read_actual_key():
    # Called only after actual root authority, exact inputs, source pins and
    # pacing admission. Never called by import or the source-only self-check.
    key=checked_bytes(KEY_PATH).decode('utf-8','strict').strip()
    require(key and all(33<=ord(c)<=126 for c in key),'external credential must be single ASCII token')
    return key

def actual_pacer(root,model,pins):
    for name,wanted in pins.items():require(digest(checked_bytes(root/name))==wanted,'immutable shared fair pacing source differs')
    from siliconflow_fair_pacing import FairSharedPacer
    class CompatibleFair(FairSharedPacer):
        @contextmanager
        def locked(self,model):
            with super().locked(model) as state:
                require(state.get('model')==model and state.get('config')==self.config,'shared model pacing configuration conflict')
                yield state
    return CompatibleFair(root=root/'logs/siliconflow_pacing',config=MODEL_PACING[model])

@contextmanager
def request_history(root,identity,output):
    history=root/'logs/sccomics_api_transport_v1_history'
    require(not any(p.is_symlink() for p in (history,*history.parents)),'history path alias')
    history.mkdir(parents=True,exist_ok=True)
    lockpath=history/(identity+'.lock');fd=os.open(lockpath,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX);record=history/(identity+'.json')
        if record.exists():raise CachedFailure('identical canonical request previously claimed; no implicit failed or in-flight retry')
        write_new(record,serialized({'identity':identity,'output_dir':str(output),'status':'claimed_before_single_attempt','implicit_retry_permitted':False}))
        yield
    finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

def cached_result(output,expected):
    require(not any(p.is_symlink() for p in (output,*output.parents)),'cache alias')
    manifest=unique_json(checked_bytes(output/'output_manifest.json'));closed(manifest,('identity','input_identity','files'))
    require(manifest['input_identity']==expected,'cache exact input binding differs')
    require(type(manifest['files']) is dict and set(manifest['files'])=={'request.json','source.txt','freeze.json','response_raw.safe','result.json'},'complete exact cache file population')
    for name,b in manifest['files'].items():
        closed(b,('sha256','size_bytes'));raw=checked_bytes(output/name);require(len(raw)==b['size_bytes'] and digest(raw)==b['sha256'],'cache hash/size mismatch')
    result=unique_json(checked_bytes(output/'result.json'))
    require(result.get('input_identity')==expected and type(result.get('passed')) is bool,'cache result identity/status')
    if not result['passed']:raise CachedFailure('same failed probe preserved; no identical retry')
    return result

def run_probe(root,payload_raw,source_raw,freeze_raw,output,*,execute_actual=False,exchange=None,key_reader=None,pacer=None):
    """Root-facing actual API or explicitly injected owned invented transport.

    The injected transport path never uses the external key or actual HTTP.
    Caller persists full safe raw responses; a failed graph is never replaced [].
    """
    require('..' not in Path(root).parts and '..' not in Path(output).parts,'root/output parent traversal rejected')
    root=Path(root).absolute();output=Path(output).absolute();freeze=unique_json(freeze_raw)
    payload,source=validate_freeze(freeze,payload_raw,source_raw)
    invented=exchange is not None
    if invented:
        require(not execute_actual and callable(key_reader) and pacer is not None,'explicit invented adapter; no actual defaults')
        require(root.name.startswith('INVENTED_') and output.is_relative_to(root),'owned invented fixture boundary')
    else:
        require(execute_actual and freeze['root_authorized_actual_synthetic_probe'] is True,'default no actual request authority')
        require(digest(checked_bytes(root/CLIENT_PATH))==freeze['client_sha256'],'actual client Source hash differs')
        require(output.is_relative_to(root/'research') and not output.is_relative_to(root/'data'),'owned research transport output')
        exchange=https_exchange;key_reader=read_actual_key
    require(not any(p.is_symlink() for p in (output,*output.parents)),'output path/ancestor alias')
    identity=digest(serialized({'canonical_request':payload,'source_sha256':digest(source_raw),'output_mode':freeze['output_mode'],'schema_sha256':freeze['schema_sha256']}))
    bindings={'payload_sha256':digest(payload_raw),'source_sha256':digest(source_raw),'freeze_sha256':digest(freeze_raw),'client_sha256':freeze['client_sha256'],'canonical_request_identity':identity}
    if output.exists():return cached_result(output,bindings)
    output.parent.mkdir(parents=True,exist_ok=True)
    with request_history(root,identity,output):
        output.mkdir();raw_parts=[];received=0;ticket=None;key='';usage=None;status=None;headers={};start=time.monotonic()
        result={'identity':'SILICONFLOW_SYNTHETIC_FORMAT_PROBE_TRANSPORT_NOT_SCIENTIFIC_SCORE','input_identity':bindings,'passed':False,'transport_completed':False,
                'graph_or_output':None,'model_requested':payload['model'],'endpoint':ENDPOINT,'transport':'SSE' if payload['stream'] else 'nonstream','attempt_count':1,
                'actual_API_attempt_started':False,'actual_paid_synthetic_request':False,'Gold_projection_scoring_or_local_NN':False,'requested_payload_controls':sorted(set(payload)-{'model','messages'})}
        def receive(b):
            nonlocal received
            require(type(b) is bytes,'HTTP byte chunks required');raw_parts.append(b);received+=len(b)
            require(received<=freeze['limits']['max_response_bytes'],'explicit response byte budget exceeded; retained partial response')
        try:
            if pacer is None:pacer=actual_pacer(root,payload['model'],freeze['pacing_source_sha256'])
            ticket=pacer.acquire(payload['model'],freeze['limits']['reserved_tokens'])
            key=key_reader();require(type(key) is str and key,'credential reader must supply a nonempty token')
            require(not safe_bytes(payload_raw,key)[1] and not safe_bytes(source_raw,key)[1] and not safe_bytes(freeze_raw,key)[1],'credential found in supplied artifact input; request blocked')
            result['actual_API_attempt_started']=not invented;result['actual_paid_synthetic_request']=not invented
            answer=exchange(payload_raw,key,freeze['limits'],receive);status=answer['http_status'];integer(status,100)
            headers={k:safe_bytes(v.encode(),key)[0].decode() for k,v in answer.get('safe_response_headers',{}).items() if k in ('content-type','retry-after','x-siliconcloud-trace-id')}
            result['http_status']=status;result['safe_response_headers']=headers
            raw=b''.join(raw_parts);safe,redacted=safe_bytes(raw,key);result['response_secret_redaction_applied']=redacted
            require(status==200,'HTTP provider failure status='+str(status))
            parsed=parse_response(safe,payload['stream'],source,freeze['output_mode']);usage=parsed['usage']
            result.update(parsed);result.update(passed=True,transport_completed=True,graph_or_output=parsed['parsed_output'])
        except BaseException as error:
            if isinstance(error,ResponseProblem):
                result.update(error.metadata)
                candidate=error.metadata.get('usage')
                if type(candidate) is dict and type(candidate.get('total_tokens')) is int and candidate['total_tokens']>=0:usage=candidate
            safe_message=safe_bytes(str(error).encode('utf-8','replace'),key)[0].decode('utf-8','replace')
            result.update(error_type=type(error).__name__,error=safe_message,passed=False)
        finally:
            safe,redacted=safe_bytes(b''.join(raw_parts),key);result['response_secret_redaction_applied']=redacted
            result['response_received_bytes_before_redaction']=received;result['safe_response_bytes_saved']=len(safe)
            result['latency_seconds_including_admission']=time.monotonic()-start;result['pacing_admission']=ticket
            if not invented:
                try:
                    require(digest(checked_bytes(root/CLIENT_PATH))==freeze['client_sha256'],'client Source changed during request')
                    for name,wanted in freeze['pacing_source_sha256'].items():require(digest(checked_bytes(root/name))==wanted,'fair pacing Source changed during request')
                except ProbeError as error:result.update(passed=False,source_integrity_error=str(error))
            if pacer is not None and ticket is not None:
                retry_after=0
                try:retry_after=max(0,float(headers.get('retry-after','0')))
                except (ValueError,TypeError):pass
                if not math.isfinite(retry_after):retry_after=0
                try:pacer.release(ticket,usage=usage,rate_failure=status==429,retry_after=retry_after)
                except BaseException as error:
                    result.update(passed=False,pacing_release_error=safe_bytes(str(error).encode('utf-8','replace'),key)[0].decode('utf-8','replace'))
            inputs={'request.json':payload_raw,'source.txt':source_raw,'freeze.json':freeze_raw,'response_raw.safe':safe}
            for name,raw in inputs.items():write_new(output/name,safe_bytes(raw,key)[0])
            write_new(output/'result.json',serialized(result))
            files={name:{'sha256':digest(checked_bytes(output/name)),'size_bytes':len(checked_bytes(output/name))} for name in (*inputs,'result.json')}
            write_new(output/'output_manifest.json',serialized({'identity':'COMPLETE_SAFE_SYNTHETIC_TRANSPORT_ARTIFACTS_NOT_SCORE','input_identity':bindings,'files':files}))
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('payload','source','freeze','freeze-sha256','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute-root-synthetic-probe',action='store_true');args=p.parse_args()
    require(args.execute_root_synthetic_probe,'no actual request without explicit root synthetic execution')
    root=Path(__file__).absolute().parents[1];sha_string(args.freeze_sha256)
    for value,suffix in ((args.payload,'.json'),(args.freeze,'.json'),(args.source,'.txt')):
        path=Path(value).absolute()
        require('..' not in Path(value).parts and path.is_relative_to(root/'research') and path.suffix==suffix,'closed research synthetic input paths/types')
    freeze=checked_bytes(args.freeze);require(digest(freeze)==args.freeze_sha256,'frozen input protocol SHA differs')
    result=run_probe(root,checked_bytes(args.payload),checked_bytes(args.source),freeze,Path(args.output_dir),execute_actual=True)
    print(json.dumps({'passed':result['passed'],'output_dir':str(Path(args.output_dir).absolute()),'scope':'synthetic_transport_only'}))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
