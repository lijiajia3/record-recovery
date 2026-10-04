"""One explicitly identified scientific SiliconFlow attempt, no import I/O.

Uses the unchanged reviewed HTTPS/SSE/redaction/fair-pacing transport helpers.
An explicit request-repeat intent is part of scientific identity: cached output
never becomes an additional repeat. Native malformed objects are NOT filtered
by the probe's validate_graph. Caller must compile exact raw_content separately.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time
import sccomics_api_transport_v1 as t

SELF='src/sccomics_api_scientific_transport_v2.py'
STAGES=('B','G_critic','G_repair','T_critic','T_repair','S_repair')
PINNED_DEPENDENCIES=(t.CLIENT_PATH,*t.PACING_PATHS)
FIELDS=('study_id','protocol_sha256','split','document_id','model','repeat','stage')
def intent_identity(intent):
    t.closed(intent,FIELDS)
    t.require(intent['study_id']=='SC_NATIVE_API_FEEDBACK_V1','study identity')
    t.sha_string(intent['protocol_sha256'])
    t.require(intent['split'] in ('development','test','invented'),'split identity')
    t.integer(intent['document_id'],1)
    t.require(intent['model'] in t.MODELS,'exact model identity')
    t.require(type(intent['repeat']) is int and intent['repeat'] in (0,1,2),'three request repeats, no best selection')
    t.require(intent['stage'] in STAGES,'one of six actual request slots')
    if intent['split']=='development':t.require(101<=intent['document_id']<=112,'exact 12 development source IDs')
    if intent['split']=='test':t.require(1<=intent['document_id']<=100,'exact 100 test source IDs')
    return t.digest(t.serialized(intent))

def validate_request(freeze, payload_raw, source_raw):
    t.closed(freeze,('version','purpose','endpoint','protocol_sha256','intent','payload_sha256','source_sha256','source_sha256_pins','limits','root_authorized_scientific_request'))
    t.require(type(freeze['version']) is int and freeze['version']==1 and freeze['purpose']=='identified_scientific_API_request','scientific scope')
    t.require(freeze['endpoint']==t.ENDPOINT,'fixed official SiliconFlow endpoint')
    identity=intent_identity(freeze['intent'])
    t.require(freeze['protocol_sha256']==freeze['intent']['protocol_sha256'],'single protocol identity')
    t.require(type(freeze['root_authorized_scientific_request']) is bool,'explicit authority boolean')
    t.require(freeze['payload_sha256']==t.digest(payload_raw) and freeze['source_sha256']==t.digest(source_raw),'complete input hashes')
    pins=freeze['source_sha256_pins'];t.require(type(pins) is dict and set(pins)=={SELF,*PINNED_DEPENDENCIES},'exact four transport source pins')
    for sha in pins.values():t.sha_string(sha)
    payload=t.validate_payload(t.unique_json(payload_raw))
    t.require(payload['model']==freeze['intent']['model'],'model request-intent identity')
    t.require(payload['stream'] is True and payload.get('n',1)==1,'fixed scientific streaming/single output')
    t.require(payload.get('temperature')==.2 and type(payload.get('temperature')) is float,'fixed scientific temperature')
    t.require(payload.get('enable_thinking') is False,'same requested thinking flag, not evidence it is honored')
    t.require(payload.get('response_format')=={'type':'json_object'},'exact common JSON object request')
    t.require(set(payload)=={'model','messages','stream','max_tokens','temperature','enable_thinking','response_format'},'fixed common provider controls')
    tokens=2048 if freeze['intent']['stage'].endswith('critic') else 8192
    t.require(type(payload['max_tokens']) is int and payload['max_tokens']==tokens,'frozen slot token budget')
    source=source_raw.decode('utf8','strict')
    t.require(any(source in m['content'] for m in payload['messages']),'original source present verbatim')
    limits=freeze['limits'];t.closed(limits,('idle_timeout_s','wall_timeout_s','max_response_bytes','reserved_tokens','pacing_policy'))
    t.require(limits['idle_timeout_s']==120 and limits['wall_timeout_s']==1800,'explicit finite per-attempt limits')
    t.require(type(limits['max_response_bytes']) is int and limits['max_response_bytes']==67108864,'64MiB whole-response safeguard, no native population cap')
    t.integer(limits['reserved_tokens'],1)
    t.require(limits['reserved_tokens']>=len(payload_raw)+tokens,'conservative payload byte plus output reservation')
    t.require(limits['pacing_policy']=='shared_fair_conservative_v1','fixed fair model pacing')
    return identity,payload,source

@contextmanager
def identity_lock(root,identity):
    history=root/'logs/sccomics_api_scientific_transport_v2_history'
    t.require(not any(p.is_symlink() for p in (history,*history.parents)),'history path alias')
    history.mkdir(parents=True,exist_ok=True)
    fd=os.open(history/(identity+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield history/(identity+'.json')
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

def reopen_attempt(output,binding):
    manifest=t.unique_json(t.checked_bytes(output/'output_manifest.json'))
    t.closed(manifest,('binding','files'))
    t.require(manifest['binding']==binding,'cached attempt identity')
    t.require(set(manifest['files'])=={'request.json','source.txt','freeze.json','response_raw.safe','result.json'},'exact five complete attempt members')
    for name,row in manifest['files'].items():
        t.closed(row,('sha256','size_bytes'));b=t.checked_bytes(output/name)
        t.require(t.digest(b)==row['sha256'] and len(b)==row['size_bytes'],'cached complete member hash/size')
    result=t.unique_json(t.checked_bytes(output/'result.json'))
    t.require(result['binding']==binding and type(result['passed']) is bool and type(result['retryable_transport_failure']) is bool,'cached result identity/status')
    return result

def run_attempt(root,payload_raw,source_raw,freeze_raw,output,*,attempt=0,execute_actual=False,exchange=None,key_reader=None,pacer=None):
    """Exactly one attempt per intent/attempt index; explicit retry of transport only.

    Actual attempt i>0 is admitted only after the immediately prior completed
    artifact says retryable HTTP429/5xx or network failure. Source-integrity,
    credential, parser, length and task-format errors never auto-resample. An
    incomplete prior attempt is a hard recovery boundary, not a second charge.
    No annotation, Native scorer, task performance or baseline fallback is here.
    """
    t.require('..' not in Path(root).parts and '..' not in Path(output).parts,'parent traversal')
    root=Path(root).absolute();output=Path(output).absolute();t.integer(attempt)
    freeze=t.unique_json(freeze_raw);identity,payload,source=validate_request(freeze,payload_raw,source_raw)
    invented=exchange is not None
    if invented:
        t.require(not execute_actual and root.name.startswith('INVENTED_') and output.is_relative_to(root) and callable(key_reader) and pacer is not None,'owned fake transport; no actual defaults')
        t.require(freeze['intent']['split']=='invented','fake fixtures use invented identity')
    else:
        t.require(execute_actual and freeze['root_authorized_scientific_request'] is True and freeze['intent']['split']!='invented','explicit root scientific authority')
        t.require(output.is_relative_to(root/'results/sccomics_api_feedback_v1'),'closed scientific output tree')
        protocol=root/'research/sccomics_api_feedback_v1/protocol_before_corpus.json'
        t.require(t.digest(t.checked_bytes(protocol))==freeze['protocol_sha256'],'actual fixed protocol byte identity')
        p=t.unique_json(t.checked_bytes(protocol))
        t.require(p.get('state')=='FROZEN_BEFORE_CORPUS_API' and p.get('scientific_request_authority') is True,'actual protocol is execution authority')
        t.require(p.get('transport_source_sha256_pins')==freeze['source_sha256_pins'],'protocol and request share source pins')
        exchange=t.https_exchange;key_reader=t.read_actual_key
    for name,wanted in freeze['source_sha256_pins'].items():
        if not invented:t.require(t.digest(t.checked_bytes(root/name))==wanted,'current transport source bytes')
    t.require(not any(p.is_symlink() for p in (output,*output.parents)),'output path aliases')
    binding={'intent_identity':identity,'intent':freeze['intent'],'attempt':attempt,'payload_sha256':t.digest(payload_raw),'source_sha256':t.digest(source_raw),'freeze_sha256':t.digest(freeze_raw)}
    if output.exists():return reopen_attempt(output,binding)
    output.parent.mkdir(parents=True,exist_ok=True)
    with identity_lock(root,identity) as ledger:
        canonical={'intent_identity':identity,'payload_sha256':binding['payload_sha256'],'source_sha256':binding['source_sha256'],'freeze_sha256':binding['freeze_sha256']}
        if ledger.exists():
            old=t.unique_json(t.checked_bytes(ledger))
            t.require(old.get('canonical')==canonical,'same intent with changed source/payload/protocol')
            t.require(attempt==old['attempt']+1,'sequential explicit attempt index, no duplicate concurrent billing')
            prior=reopen_attempt(Path(old['output']),dict(binding,attempt=attempt-1))
            t.require(prior['retryable_transport_failure'] and not prior['passed'],'only completed retryable transport may be attempted again')
        else:t.require(attempt==0,'first request attempt must be zero')
        # Atomic ledger replacement happens BEFORE pacing/key/network; no stale
        # incomplete request can be repeated by choosing another output path.
        raw=t.serialized({'canonical':canonical,'attempt':attempt,'output':str(output)})
        tmp=ledger.with_suffix('.next')
        t.write_new(tmp,raw);os.replace(tmp,ledger)
        output.mkdir();parts=[];received=0;ticket=None;key='';status=None;headers={};usage=None;start=time.monotonic()
        result={'binding':binding,'passed':False,'retryable_transport_failure':False,'actual_API_attempt_started':False,'paid_inference_provider':'SiliconFlow','http_status':None,'raw_content':None,'parsed_output':None,'native_objects_filtered_or_Gold_scored':False,'cache_is_additional_repeat':False}
        def receive(b):
            nonlocal received
            t.require(type(b) is bytes,'response bytes');parts.append(b);received+=len(b)
            t.require(received<=freeze['limits']['max_response_bytes'],'response size resource safeguard exceeded; all received bytes retained')
        try:
            if pacer is None:pacer=t.actual_pacer(root,payload['model'],{n:freeze['source_sha256_pins'][n] for n in t.PACING_PATHS})
            ticket=pacer.acquire(payload['model'],freeze['limits']['reserved_tokens'])
            key=key_reader();t.require(type(key) is str and key,'nonempty credential token')
            t.require(not any(t.safe_bytes(b,key)[1] for b in (payload_raw,source_raw,freeze_raw)),'credential must not appear in inputs')
            result['actual_API_attempt_started']=not invented
            answer=exchange(payload_raw,key,freeze['limits'],receive);status=answer['http_status'];t.integer(status,100)
            headers={k:t.safe_bytes(str(v).encode(),key)[0].decode() for k,v in answer.get('safe_response_headers',{}).items() if k in ('content-type','retry-after','x-siliconcloud-trace-id')}
            result.update(http_status=status,safe_response_headers=headers)
            if status!=200:
                result['retryable_transport_failure']=status==429 or 500<=status<=599
                raise t.ProbeError('HTTP status '+str(status))
            safe,redacted=t.safe_bytes(b''.join(parts),key)
            # Strict whole JSON object parsing only; per-object malformed native
            # population reaches the separate reviewed compiler unchanged.
            parsed=t.parse_response(safe,True,source,'json_object');usage=parsed['usage']
            t.require(bool(parsed['reported_models']) and all(m==payload['model'] for m in parsed['reported_models']),'reported provider model differs from requested exact model')
            result.update(parsed,passed=True)
        except BaseException as error:
            if isinstance(error,t.ResponseProblem):
                result.update(error.metadata);usage=error.metadata.get('usage')
            if isinstance(error,(OSError,TimeoutError)) and ticket is not None and key and status is None:
                result['retryable_transport_failure']=True
            result.update(error_type=type(error).__name__,error=t.safe_bytes(str(error).encode('utf8','replace'),key)[0].decode('utf8','replace'),passed=False)
        finally:
            safe,redacted=t.safe_bytes(b''.join(parts),key)
            result.update(response_secret_redaction_applied=redacted,response_received_bytes_before_redaction=received,latency_seconds_including_admission=time.monotonic()-start,pacing_admission=ticket)
            if not invented:
                try:
                    for name,wanted in freeze['source_sha256_pins'].items():t.require(t.digest(t.checked_bytes(root/name))==wanted,'transport source changed during attempt')
                    t.require(t.digest(t.checked_bytes(protocol))==freeze['protocol_sha256'],'protocol changed during attempt')
                except BaseException as error:result.update(passed=False,retryable_transport_failure=False,source_integrity_error=t.safe_bytes(str(error).encode(),key)[0].decode('utf8','replace'))
            if pacer is not None and ticket is not None:
                try:
                    retry_after=max(0,float(headers.get('retry-after','0')))
                    if not math.isfinite(retry_after):retry_after=0
                except (ValueError,TypeError):retry_after=0
                try:pacer.release(ticket,usage=usage,rate_failure=status==429,retry_after=retry_after)
                except BaseException as error:result.update(passed=False,retryable_transport_failure=False,pacing_release_error=t.safe_bytes(str(error).encode(),key)[0].decode('utf8','replace'))
            files={'request.json':payload_raw,'source.txt':source_raw,'freeze.json':freeze_raw,'response_raw.safe':safe}
            # Provider can reflect credentials into content or reasoning. The
            # entire saved result is redacted as well as the raw wire artifact.
            for name,b in files.items():t.write_new(output/name,t.safe_bytes(b,key)[0])
            t.write_new(output/'result.json',t.safe_bytes(t.serialized(result),key)[0])
            manifest={name:{'sha256':t.digest(t.checked_bytes(output/name)),'size_bytes':len(t.checked_bytes(output/name))} for name in (*files,'result.json')}
            t.write_new(output/'output_manifest.json',t.serialized({'binding':binding,'files':manifest}))
        return t.unique_json(t.checked_bytes(output/'result.json'))
