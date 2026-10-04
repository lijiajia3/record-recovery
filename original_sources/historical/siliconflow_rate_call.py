"""Frozen completion function with separately auditable admission scheduling.

Derived from the exact archived pilot.call. Transport changes are documented in
research/rate_limit_transport_amendment.md. No model/task method changes occur.
"""
from pilot import *
from siliconflow_pacing import PACER

def call(model,prompt,path):
    global IN_FLIGHT, SERVICE_BLOCKED
    if path.exists():return read_json(path)
    key_path=Path(os.environ.get('SILICONFLOW_KEY_FILE',str(Path.home()/'.siliconflow_key')))
    key=key_path.read_text(encoding='utf-8').strip()
    max_output=int(os.environ.get('FEEDBACK_MAX_OUTPUT_TOKENS','4096'))
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'temperature':float(os.environ.get('FEEDBACK_API_TEMPERATURE','0')),
             'max_tokens':max_output,'enable_thinking':False,'response_format':{'type':'json_object'}}
    if 'FEEDBACK_API_TOP_P' in os.environ:payload['top_p']=float(os.environ['FEEDBACK_API_TOP_P'])
    if 'FEEDBACK_API_TOP_K' in os.environ:payload['top_k']=int(os.environ['FEEDBACK_API_TOP_K'])
    if 'FEEDBACK_API_MIN_P' in os.environ:payload['min_p']=float(os.environ['FEEDBACK_API_MIN_P'])
    if 'FEEDBACK_API_FREQUENCY_PENALTY' in os.environ:payload['frequency_penalty']=float(os.environ['FEEDBACK_API_FREQUENCY_PENALTY'])
    streaming=os.environ.get('FEEDBACK_API_STREAM','0')=='1'
    if streaming:payload['stream']=True
    path.parent.mkdir(parents=True,exist_ok=True)
    # Public dataset inputs only; save the prompt even when the API fails.
    path.with_suffix('.prompt.txt').write_text(prompt,encoding='utf-8')
    preflight = context_preflight(prompt, max_output)
    if preflight and not preflight['fits']:
        out = {'relations': [], 'error': 'local context preflight overflow; no request sent',
               'local_preflight': preflight}
        write_json(path, out)
        return out
    max_retries=max(1,int(os.environ.get('FEEDBACK_API_RETRIES','3')))
    for retry in range(max_retries):
        raw_received=None
        errors=ROOT/'logs'/os.environ.get('FEEDBACK_API_ERROR_LOG','api_errors.jsonl')
        with REQUEST_LOCK:
            error_count=len(errors.read_text(encoding='utf-8').splitlines()) if errors.exists() else 0
            if SERVICE_BLOCKED:
                raise RuntimeError('permanent API service failure; no new requests')
            cap=execution_limit('FEEDBACK_MAX_ERRORS','api_error_event_limit',30)
            if cap is not None and error_count+IN_FLIGHT>=cap:
                raise RuntimeError(f'{cap} global API/parse error budget reserved or reached; stop')
            IN_FLIGHT+=1
        reservation=(preflight['input_tokens_estimate']+preflight['chat_wrapper_margin']
                     if preflight else max(1,len(prompt)//3)+256)+max_output
        ticket=PACER.acquire(model,reservation)
        rate_failure=False;retry_after=0;observed_usage=None
        start=time.monotonic()
        try:
            timeout=float(os.environ.get('FEEDBACK_API_TIMEOUT','120'))
            response=requests.post(API,headers={'Authorization':'Bearer '+key},json=payload,timeout=timeout,stream=streaming)
            if not response.ok:
                safe_body=response.text.replace(key,'[REDACTED]')
                write_json(path.with_suffix(f'.http_failure{retry}.json'),{
                    'status_code':response.status_code,'body':safe_body[:16000],
                    'safe_headers':{k:v for k,v in response.headers.items() if k.lower() in
                                    ['content-type','retry-after']},
                    'model':model,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'pacing':ticket,'transport_attempt':retry})
                rate_failure=response.status_code==429
                try:retry_after=float(response.headers.get('retry-after',0))
                except (ValueError,TypeError):retry_after=0
                code=response.json().get('code') if response.headers.get('content-type','').startswith('application/json') else None
                if response.status_code in (400,401,402,403,404,422):
                    with REQUEST_LOCK:
                        SERVICE_BLOCKED=True
                    append(errors,{'stage_path':str(path.relative_to(ROOT)),'retry':retry,'error':f'HTTP {response.status_code}, code {code}','permanent':True})
                    out={'relations':[],'error':f'HTTP {response.status_code}, code {code}','permanent':True}
                    write_json(path,out)
                    return out
                raise RuntimeError(f'HTTP {response.status_code}, code {code}')
            if streaming:
                wall=float(os.environ.get('FEEDBACK_API_WALL_TIMEOUT','900'))
                raw=assemble_stream(response.iter_lines(),path.with_suffix(f'.stream_retry{retry}.jsonl'),start+wall)
            else:raw=response.json()
            response.close()
            raw_received=raw;observed_usage=raw.get('usage'); choice=raw['choices'][0]
            content=choice['message'].get('content','').strip()
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
            # Repeating an identical request cannot repair a hard output ceiling.
            # Preserve the terminal response and return an explicit failed item.
            if choice.get('finish_reason')=='length':raise ValueError('truncated output: provider finish_reason=length')
            parsed=json.loads(content)
            if not isinstance(parsed,dict) or not isinstance(parsed.get('relations'),list):
                raise ValueError('invalid relations JSON schema')
            out={'relations':parsed['relations'],'usage':raw.get('usage',{}),'latency_s':time.monotonic()-start,'model':model,'endpoint':API,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'response_id':raw.get('id'),'finish_reason':choice.get('finish_reason'),'request_timeout_s':timeout,'transport':'SSE' if streaming else 'nonstream','temperature':payload['temperature'],'max_output_tokens':payload['max_tokens'],'raw_content':content}
            out['pacing']=ticket
            if 'top_p' in payload:out['top_p']=payload['top_p']
            out['request_sampling']={k:payload[k] for k in ['temperature','top_p','top_k','min_p','frequency_penalty'] if k in payload}
            if preflight:out['local_preflight']=preflight
            write_json(path,out)
            # Save complete public-input prompts, never keys or private manuscripts.
            path.with_suffix('.prompt.txt').write_text(prompt,encoding='utf-8')
            append(ROOT/'logs/requests.jsonl',{'path':str(path.relative_to(ROOT)),'model':model,'usage':out['usage'],'latency_s':out['latency_s'],'response_id':out['response_id']})
            return out
        except (requests.RequestException,ValueError,KeyError,RuntimeError) as exc:
            if raw_received is not None:
                write_json(path.with_suffix(f'.failed_response{retry}.json'),{
                    'raw_response':raw_received,'error':str(exc)[:180],
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'model':model,'endpoint':API,'retry':retry})
            append(errors,{'stage_path':str(path.relative_to(ROOT)),'retry':retry,'error':str(exc)[:180]})
            terminal_length=raw_received is not None and raw_received.get('choices',[{}])[0].get('finish_reason')=='length'
            if retry==max_retries-1 or terminal_length:
                out={'relations':[],'error':str(exc)[:180],
                     'output_truncated':terminal_length,'retry_count':retry+1}
                if raw_received is not None:out['failed_response_usage']=raw_received.get('usage',{})
                write_json(path,out);return out
            time.sleep(2*(retry+1))
        finally:
            PACER.release(ticket,usage=observed_usage,rate_failure=rate_failure,retry_after=retry_after)
            with REQUEST_LOCK:
                IN_FLIGHT-=1
    raise RuntimeError('unreachable')

