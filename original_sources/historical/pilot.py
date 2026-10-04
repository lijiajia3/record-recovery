#!/usr/bin/env python3
"""Locked SciERC pilot. All paid inference goes only to SiliconFlow.

Generation inputs and evaluation labels are exported separately. Public gold
entities are task inputs; gold relations never enter dev/test prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import threading
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/scierc'
API = 'https://api.siliconflow.cn/v1/chat/completions'
LABELS = ['COMPARE','CONJUNCTION','EVALUATE-FOR','FEATURE-OF','HYPONYM-OF','PART-OF','USED-FOR']
SYMMETRIC = {'COMPARE','CONJUNCTION'}
LOCK = threading.Lock()
REQUEST_LOCK = threading.Lock()
IN_FLIGHT = 0
SERVICE_BLOCKED = False
PROTOCOL_VERSION = os.environ.get('FEEDBACK_PROTOCOL_VERSION','v1')
DEFINITIONS = '''COMPARE: two entities explicitly compared (symmetric).
CONJUNCTION: coordinated scientific entities sharing a role (symmetric).
EVALUATE-FOR: metric used to evaluate a method/task; metric is head.
FEATURE-OF: feature/property of an entity; feature is head.
HYPONYM-OF: specific instance/subtype of general entity; specific is head.
PART-OF: component part of a whole; component is head.
USED-FOR: method/material/entity used for a task/application; used entity is head.
Use only relations expressed by the sentence, not background knowledge. Include
all expressed relations between provided entities, including nested mentions.
No relation is also a valid outcome. Do not invent transitive or cross-sentence edges.'''
if PROTOCOL_VERSION=='v2':
    DEFINITIONS=DEFINITIONS.replace('EVALUATE-FOR: metric used to evaluate a method/task; metric is head.',
        'EVALUATE-FOR: an evaluation criterion, dataset, task, or other evaluator/context used to assess an entity. The evaluator/context is head; the assessed entity is tail. Do not restrict endpoint types to Metric.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def append(path, value):
    with LOCK:
        with Path(path).open('a', encoding='utf-8') as f:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')


def begin_attempt(stage, model, split):
    path = ROOT/'logs/attempts.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    # Distinct worker processes must reserve distinct audit IDs as well.
    with path.open('a+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.seek(0)
        starts = [x for line in f if (x:=json.loads(line)).get('event')=='start']
        cap=execution_limit('FEEDBACK_MAX_CONFIGS','configuration_start_limit',30)
        if cap is not None and len(starts)>=cap:
            raise RuntimeError(f'{cap} research attempts reached; stop')
        idx=len(starts)+1
        f.write(json.dumps({'event':'start','attempt':idx,'stage':stage,'model':model,
            'split':split,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})+'\n')
        f.flush()
        fcntl.flock(f,fcntl.LOCK_UN)
    return idx


def execution_limit(env_name,setting_name,legacy_default):
    if env_name in os.environ:
        value=os.environ[env_name].strip().lower()
        return None if value in ['none','unlimited',''] else int(value)
    settings=ROOT/'research/campaign_plan.json'
    if settings.exists():return read_json(settings).get(setting_name,legacy_default)
    return legacy_default


def load_source(split):
    return [json.loads(s) for s in (SOURCE/f'{split}.json').read_text(encoding='utf-8').splitlines() if s.strip()]


def convert(doc):
    sentences, gold = [], []
    tokens = [t for sent in doc['sentences'] for t in sent]
    for s, text in enumerate(doc['sentences']):
        entities = [{'id':f'e{a}_{b}','name':' '.join(tokens[a:b+1]),'type':typ} for a,b,typ in doc['ner'][s]]
        sentences.append({'s':s,'text':' '.join(text),'entities':entities})
        for a,b,c,d,r in doc['relations'][s]:
            gold.append({'s':s,'h':f'e{a}_{b}','t':f'e{c}_{d}','r':r})
    return {'doc_key':doc['doc_key'],'sentences':sentences},gold


def prepare():
    selected = {}
    for split,n in [('dev',8),('test',24)]:
        docs = sorted(load_source(split),key=lambda d:hashlib.sha256(('feedback-pilot-v1|'+d['doc_key']).encode()).hexdigest())[:n]
        ins,golds=[],{}
        for d in docs:
            inp,gold=convert(d);ins.append(inp);golds[d['doc_key']]=gold
        write_json(ROOT/f'data/{split}_inputs.json',ins)
        write_json(ROOT/f'data/{split}_gold.json',golds)
        selected[split]=[d['doc_key'] for d in docs]
    examples=[]; sig=defaultdict(Counter)
    for doc in load_source('train'):
        inp,gold=convert(doc)
        for sent in inp['sentences']:
            if len(sent['entities']) < 2:continue
            rs=[r for r in gold if r['s']==sent['s']]
            entities={e['id']:e for e in sent['entities']}
            for r in rs:
                if r['h'] in entities and r['t'] in entities:
                    sig[r['r']][(entities[r['h']]['type'],entities[r['t']]['type'])]+=1
                    if r['r'] in SYMMETRIC:sig[r['r']][(entities[r['t']]['type'],entities[r['h']]['type'])]+=1
            if len(sent['text']) <= 1000:
                examples.append({'doc_key':doc['doc_key'],'sentence':sent,'relations':rs})
    write_json(ROOT/'data/train_examples.json',examples)
    write_json(ROOT/'data/train_signatures.json',{r:{'|'.join(k):v for k,v in c.items()} for r,c in sig.items()})
    selected['source_files_sha256']={s:hashlib.sha256((SOURCE/f'{s}.json').read_bytes()).hexdigest() for s in ['train','dev','test']}
    selected['protocol_sha256']=hashlib.sha256((ROOT/'research/protocol.md').read_bytes()).hexdigest()
    write_json(ROOT/'data/manifest.json',selected)
    print('PREPARED', {s:len(selected[s]) for s in ['dev','test']}, 'train examples',len(examples),flush=True)


def words(s):return re.findall(r'[a-z][a-z0-9-]{2,}',s.lower())


def retrieve(inp):
    examples=read_json(ROOT/'data/train_examples.json')
    query=Counter(words(' '.join(s['text'] for s in inp['sentences'])))
    df=Counter(w for x in examples for w in set(words(x['sentence']['text'])))
    idf={w:math.log((len(examples)+1)/(n+1))+1 for w,n in df.items()}
    qnorm=math.sqrt(sum((c*idf.get(w,1))**2 for w,c in query.items())) or 1
    scored=[]
    for ex in examples:
        ec=Counter(words(ex['sentence']['text']))
        norm=math.sqrt(sum((c*idf.get(w,1))**2 for w,c in ec.items())) or 1
        dot=sum(query[w]*c*idf.get(w,1)**2 for w,c in ec.items())
        scored.append((dot/(qnorm*norm),ex))
    scored.sort(key=lambda x:-x[0])
    picked=[];seen=set()
    for score,ex in scored:
        if ex['doc_key'] in seen:continue
        seen.add(ex['doc_key']);picked.append(ex)
        if len(picked)==3:break
    return picked


def input_prompt(inp):
    assert 'relations' not in inp and 'gold' not in inp
    return ('Extract within-sentence scientific semantic relations between the GIVEN entity spans.\n'+DEFINITIONS+
            '\nReturn JSON {"relations":[{"s":sentence_id,"h":head_entity_id,"t":tail_entity_id,"r":label,"evidence":exact_short_quote_from_sentence}]}.'
            '\nEach quote must cover both entity surface mentions. Do not return reasoning. All endpoints must be from the SAME sentence.\nINPUT:\n'+json.dumps(inp['sentences'],ensure_ascii=False))


def base_prompt(inp, use_retrieval=False):
    prompt=input_prompt(inp)
    if use_retrieval:
        prompt+='\nTRAINING EXAMPLES (different abstracts; illustrate annotation conventions):\n'
        prompt+=json.dumps(retrieve(inp),ensure_ascii=False)
    return prompt


def cache_path(model,split,stage,inp):
    cache_root='results/cache_v2' if PROTOCOL_VERSION=='v2' else 'results/cache'
    return ROOT/cache_root/model.replace('/','__')/split/stage/(hashlib.sha256(inp['doc_key'].encode()).hexdigest()[:16]+'.json')


def assemble_stream(lines, transcript=None, deadline=None):
    """Reassemble documented SSE without treating a partial stream as success."""
    content=[];reasoning=[];usage={};response_id=None;model=None;finish=None;done=False
    f=Path(transcript).open('w',encoding='utf-8') if transcript else None
    try:
        for line in lines:
            if deadline is not None and time.monotonic()>deadline:
                raise RuntimeError('stream total execution deadline reached')
            if isinstance(line,bytes):line=line.decode('utf-8')
            line=line.strip()
            if not line or not line.startswith('data:'):continue
            body=line[5:].strip()
            if f:f.write(body+'\n');f.flush()
            if body=='[DONE]':done=True;break
            event=json.loads(body)
            if event.get('usage'):usage=event['usage']
            response_id=event.get('id',response_id);model=event.get('model',model)
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content'):content.append(delta['content'])
                if delta.get('reasoning_content'):reasoning.append(delta['reasoning_content'])
                if choice.get('finish_reason'):finish=choice['finish_reason']
        if not done or not finish:raise RuntimeError('incomplete SSE response; no terminal event')
        return {'id':response_id,'model':model,'usage':usage,'transport':'SSE',
                'choices':[{'message':{'content':''.join(content),'reasoning_content':''.join(reasoning)},
                            'finish_reason':finish}]}
    finally:
        if f:f.close()


_PROMPT_TOKENIZERS = {}


def context_preflight(prompt, max_output):
    tokenizer_path = os.environ.get('FEEDBACK_PROMPT_TOKENIZER')
    limit = os.environ.get('FEEDBACK_API_CONTEXT_LIMIT')
    if not tokenizer_path or not limit:
        return None
    from tokenizers import Tokenizer
    with REQUEST_LOCK:
        if tokenizer_path not in _PROMPT_TOKENIZERS:
            _PROMPT_TOKENIZERS[tokenizer_path] = Tokenizer.from_file(tokenizer_path)
        tokenizer = _PROMPT_TOKENIZERS[tokenizer_path]
    count = len(tokenizer.encode(prompt).ids)
    margin = int(os.environ.get('FEEDBACK_CHAT_WRAPPER_MARGIN', '256'))
    return {'input_tokens_estimate': count, 'tokenizer_file': tokenizer_path,
            'context_limit': int(limit), 'chat_wrapper_margin': margin,
            'reserved_output_tokens': max_output,
            'fits': count + max_output + margin <= int(limit)}


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
        start=time.monotonic()
        try:
            timeout=float(os.environ.get('FEEDBACK_API_TIMEOUT','120'))
            response=requests.post(API,headers={'Authorization':'Bearer '+key},json=payload,timeout=timeout,stream=streaming)
            if not response.ok:
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
            raw_received=raw; choice=raw['choices'][0]
            content=choice['message'].get('content','').strip()
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
            # Repeating an identical request cannot repair a hard output ceiling.
            # Preserve the terminal response and return an explicit failed item.
            if choice.get('finish_reason')=='length':raise ValueError('truncated output: provider finish_reason=length')
            parsed=json.loads(content)
            if not isinstance(parsed,dict) or not isinstance(parsed.get('relations'),list):
                raise ValueError('invalid relations JSON schema')
            out={'relations':parsed['relations'],'usage':raw.get('usage',{}),'latency_s':time.monotonic()-start,'model':model,'endpoint':API,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'response_id':raw.get('id'),'finish_reason':choice.get('finish_reason'),'request_timeout_s':timeout,'transport':'SSE' if streaming else 'nonstream','temperature':payload['temperature'],'max_output_tokens':payload['max_tokens'],'raw_content':content}
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
            with REQUEST_LOCK:
                IN_FLIGHT-=1
    raise RuntimeError('unreachable')


def entities(inp):return {e['id']:(sent['s'],e) for sent in inp['sentences'] for e in sent['entities']}


def warnings(inp,rels):
    ents=entities(inp);sig=read_json(ROOT/'data/train_signatures.json');out=[]
    for r in rels:
        if not isinstance(r,dict):out.append({'kind':'invalid_record'});continue
        h,t=r.get('h'),r.get('t');label=r.get('r')
        if h not in ents or t not in ents or label not in LABELS or h==t or ents[h][0]!=ents[t][0]:
            out.append({'relation':r,'kind':'invalid_structure'});continue
        pair=ents[h][1]['type']+'|'+ents[t][1]['type']
        if pair not in sig.get(label,{}):
            out.append({'relation':r,'kind':'unseen_training_type_pair','note':'warning only, not a logical impossibility'})
    return out


def tuple_key(r,symmetric=True):
    if not isinstance(r,dict):return ('INVALID',str(r),'INVALID','INVALID')
    h,t=str(r.get('h','INVALID')),str(r.get('t','INVALID'))
    if symmetric and r.get('r') in SYMMETRIC:h,t=sorted([h,t])
    return (str(r.get('s','INVALID')),h,t,str(r.get('r','INVALID')))


def evidence_valid(inp,r):
    ents=entities(inp)
    if not isinstance(r,dict) or r.get('h') not in ents or r.get('t') not in ents:return False
    h,t=ents[r['h']],ents[r['t']]
    if h[0]!=t[0] or str(r.get('s'))!=str(h[0]) or r.get('r') not in LABELS or r['h']==r['t']:return False
    clean=lambda x:' '.join(str(x).lower().split())
    quote=clean(r.get('evidence',''))
    sentence=clean(inp['sentences'][h[0]]['text'])
    return bool(quote) and quote in sentence and clean(h[1]['name']) in quote and clean(t[1]['name']) in quote


def guard(inp,original,proposal):
    old={tuple_key(r):r for r in original['relations']}
    new={tuple_key(r):r for r in proposal['relations']}
    removable={tuple_key(x['relation']) for x in warnings(inp,original['relations']) if 'relation' in x}
    accepted=dict(old)
    for k in old.keys()-new.keys():
        if k in removable:accepted.pop(k)
    for k in new.keys()-old.keys():
        if evidence_valid(inp,new[k]):accepted[k]=new[k]
    return {'relations':list(accepted.values()),'offline_guard':True,'proposed_additions':len(new.keys()-old.keys()),'accepted_additions':sum(k in accepted for k in new.keys()-old.keys()),'proposed_deletions':len(old.keys()-new.keys()),'accepted_deletions':sum(k not in accepted for k in old.keys()-new.keys())}


def run_stage(model,split,stage,workers):
    idx=begin_attempt(stage,model,split)
    inputs=read_json(ROOT/f'data/{split}_inputs.json')
    def work(inp):
        path=cache_path(model,split,stage,inp)
        if path.exists():return read_json(path)
        if stage in ['baseline','retrieval']:
            prompt=base_prompt(inp,stage=='retrieval')
        else:
            parent={'self_refine':'baseline','retrieval_self_refine':'retrieval','feedback':'retrieval','guarded':'retrieval','self_refine3':'self_refine'}[stage]
            prev_path=cache_path(model,split,parent,inp)
            if not prev_path.exists():raise RuntimeError('missing parent '+str(prev_path))
            prev=read_json(prev_path)
            if stage=='guarded':
                proposal=read_json(cache_path(model,split,'feedback',inp))
                out=guard(inp,prev,proposal);write_json(path,out);return out
            prompt=base_prompt(inp,stage in ['retrieval_self_refine','feedback'])+'\nCURRENT OUTPUT:\n'+json.dumps(prev['relations'],ensure_ascii=False)
            if stage=='feedback':
                prompt+='\nEXECUTABLE AUDIT WARNINGS:\n'+json.dumps(warnings(inp,prev['relations']),ensure_ascii=False)
                prompt+='\nPerform a targeted evidence audit: inspect endpoint direction and relation meaning, nested-entity omissions, coordinated entities, and unsupported inferred edges. For each edit check the exact sentence. Warnings are not proof: rare type pairs can be valid. Preserve all already correct relations. Return a COMPLETE improved relations list with exact evidence, not patches.'
            else:
                prompt+='\nReview your extraction carefully for any mistakes and missing relations. Correct it based on the original text and return a COMPLETE improved relations list with exact evidence.'
        out=call(model,prompt,path)
        if out.get('error') and stage not in ['baseline','retrieval']:
            out['relations']=prev['relations'];out['fallback_to_parent']=True;write_json(path,out)
        return out
    completed=0;failed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(work,inp) for inp in inputs]
        for f in as_completed(futures):
            res=f.result();completed+=1;failed+=int(bool(res.get('error')))
            if completed%4==0 or completed==len(inputs):print(f'{split} {stage}: {completed}/{len(inputs)} errors={failed}',flush=True)
    append(ROOT/'logs/attempts.jsonl',{'event':'complete','attempt':idx,'failed_documents':failed})


def prf(counts):
    tp,fp,fn=np.asarray(counts,dtype=float)
    return {'precision':float(tp/(tp+fp)) if tp+fp else 0.,'recall':float(tp/(tp+fn)) if tp+fn else 0.,'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,'tp':int(tp),'fp':int(fp),'fn':int(fn)}


def evaluate(model,split,stages):
    inputs=read_json(ROOT/f'data/{split}_inputs.json');gold=read_json(ROOT/f'data/{split}_gold.json')
    result={'model':model,'split':split,'protocol_version':PROTOCOL_VERSION,'n_documents':len(inputs),'stages':{},'comparisons':{}}
    counts={};predsets={};per_doc=[]
    for stage in stages:
        cells=[];directed=[];evok=0;nrels=0;errs=0;sets=[];types=defaultdict(lambda:np.zeros(3,dtype=int))
        for inp in inputs:
            res=read_json(cache_path(model,split,stage,inp));errs+=int(bool(res.get('error')))
            p={tuple_key(r) for r in res['relations']};g={tuple_key(r) for r in gold[inp['doc_key']]}
            pd={tuple_key(r,False) for r in res['relations']};gd={tuple_key(r,False) for r in gold[inp['doc_key']]}
            c=[len(p&g),len(p-g),len(g-p)];cells.append(c);sets.append(p)
            directed.append([len(pd&gd),len(pd-gd),len(gd-pd)])
            nrels+=len(res['relations']);evok+=sum(evidence_valid(inp,r) for r in res['relations'])
            for r in LABELS:
                pl={k for k in p if k[-1]==r};gl={k for k in g if k[-1]==r}
                types[r]+=np.array([len(pl&gl),len(pl-gl),len(gl-pl)])
            per_doc.append({'stage':stage,'doc_key':inp['doc_key'],'tp':c[0],'fp':c[1],'fn':c[2]})
        counts[stage]=np.array(cells);predsets[stage]=sets
        summary=prf(counts[stage].sum(axis=0));summary.update({'raw_directed':prf(np.array(directed).sum(axis=0)),'evidence_valid':evok,'emitted_relation_records':nrels,'failed_documents':errs,'relation_wise':{r:prf(c) for r,c in types.items()}})
        parent='retrieval' if stage in ['feedback','guarded','retrieval_self_refine'] else 'baseline'
        if parent in predsets and stage!=parent:
            edit=Counter()
            for i,inp in enumerate(inputs):
                old,new=predsets[parent][i],sets[i];g={tuple_key(r) for r in gold[inp['doc_key']]}
                edit.update({'correct_deleted':len((old-new)&g),'wrong_deleted':len((old-new)-g),'correct_added':len((new-old)&g),'wrong_added':len((new-old)-g)})
            summary['edits_vs_'+parent]=dict(edit)
        result['stages'][stage]=summary
    comps=[('retrieval','baseline'),('feedback','retrieval_self_refine'),('guarded','feedback'),('guarded','baseline')] if PROTOCOL_VERSION=='v2' else [('retrieval','baseline'),('feedback','retrieval_self_refine'),('guarded','feedback'),('guarded','self_refine3')]
    rng=np.random.default_rng(20261003)
    for a,b in comps:
        if a not in counts or b not in counts:continue
        delta=100*(prf(counts[a].sum(axis=0))['f1']-prf(counts[b].sum(axis=0))['f1'])
        bs=[];perm=[];n=len(inputs)
        for _ in range(5000):
            ix=rng.integers(0,n,n);bs.append(100*(prf(counts[a][ix].sum(axis=0))['f1']-prf(counts[b][ix].sum(axis=0))['f1']))
            swap=rng.integers(0,2,n).astype(bool)
            ca=np.where(swap[:,None],counts[b],counts[a]);cb=np.where(swap[:,None],counts[a],counts[b])
            perm.append(100*(prf(ca.sum(axis=0))['f1']-prf(cb.sum(axis=0))['f1']))
        p=(sum(abs(x)>=abs(delta)-1e-12 for x in perm)+1)/5001
        result['comparisons'][a+' vs '+b]={'delta_f1_points':delta,'paired_document_bootstrap_ci95':np.quantile(bs,[.025,.975]).tolist(),'permutation_p':p}
    ordered=sorted(result['comparisons'],key=lambda k:result['comparisons'][k]['permutation_p']);largest=0
    for rank,k in enumerate(ordered):
        largest=max(largest,min(1,result['comparisons'][k]['permutation_p']*(len(ordered)-rank)))
        result['comparisons'][k]['holm_p']=largest
    suffix='__v2' if PROTOCOL_VERSION=='v2' else ''
    write_json(ROOT/f'results/{model.replace("/","__")}{suffix}_{split}_summary.json',result)
    write_json(ROOT/f'results/{model.replace("/","__")}{suffix}_{split}_per_document.json',per_doc)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','evaluate'])
    parser.add_argument('--model',default='Qwen/Qwen3-8B');parser.add_argument('--split',default='dev',choices=['dev','test']);parser.add_argument('--stages',default='baseline,retrieval,self_refine,retrieval_self_refine,feedback,guarded,self_refine3');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();stages=args.stages.split(',')
    if args.action=='prepare':prepare()
    elif args.action=='run':
        for stage in stages:run_stage(args.model,args.split,stage,args.workers)
    else:evaluate(args.model,args.split,stages)


if __name__=='__main__':main()
