"""Source-only shared-context POLYIE adapter primitives; no target-file reads."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROLES=('CN','PN','PV','Condition')
MODEL_DIR=ROOT/'data/local_baselines/matscibert'
CACHE=ROOT/'results/local_baseline/polyie_shared_context_v1/source_cache'
CONTENT=510
STRIDE=384


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def key(record): return tuple(tuple(sorted(set(map(str,record[r])))) for r in ROLES)


def ordered_roles(inp):
    return [[e for e in sorted(inp['entities'],key=lambda e:(e['start'],e['end'],e['id']))
             if e['type']==role] for role in ROLES]


def is_owner(inp, endpoints):
    # Earlier window ends at current start+200; all present endpoints are already
    # at or after current start and therefore also within that earlier window.
    return inp['start']==0 or any(e['end']>inp['start']+200 for e in endpoints)


def population_count(inp):
    roles=ordered_roles(inp)
    total=np.prod([len(x)for x in roles[:3]],dtype=np.int64)*(len(roles[3])+1)
    earlier=0
    if inp['start']:
        n=[sum(e['end']<=inp['start']+200 for e in x)for x in roles]
        earlier=n[0]*n[1]*n[2]*(n[3]+1)
    return int(total-earlier)


def candidate_batches(inp, batch_size=8192):
    roles=ordered_roles(inp);conditions=[None]+roles[3];batch=[]
    for cn,pn,pv,condition in itertools.product(*roles[:3],conditions):
        endpoints=[cn,pn,pv]+([condition]if condition else [])
        if not is_owner(inp,endpoints):continue
        batch.append(tuple(e['id']for e in endpoints))
        if len(batch)==batch_size:yield batch;batch=[]
    if batch:yield batch


def record_for(ids):
    assert len(ids)in(3,4)
    return {r:([str(ids[i])]if i<len(ids)else [])for i,r in enumerate(ROLES)}


def negative_sample(inp, positive_ids, count, rng):
    """Uniform rejection from full source population, without replacement."""
    roles=ordered_roles(inp);dims=[len(x)for x in roles[:3]]+[len(roles[3])+1]
    if any(n==0 for n in dims):return []
    full=int(np.prod(dims,dtype=np.int64));wanted=min(count,population_count(inp)-len(positive_ids))
    selected=set()
    while len(selected)<wanted:
        index=int(rng.integers(full));p=[]
        for size in reversed(dims):p.append(index%size);index//=size
        a,b,c,d=reversed(p)
        endpoints=[roles[0][a],roles[1][b],roles[2][c]]+([roles[3][d-1]]if d else [])
        ids=tuple(e['id']for e in endpoints)
        if is_owner(inp,endpoints)and ids not in positive_ids:selected.add(ids)
    return sorted(selected)


def chunk_encoding(inp, tokenizer):
    text=' '.join(inp['tokens']);assert text==inp['text']
    encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
    offsets=encoded['offset_mapping'];ids=encoded['input_ids']
    starts=[];pos=0
    for token in inp['tokens']:starts.append(pos);pos+=len(token)+1
    entities=[e for e in inp['entities']if e['type']in ROLES]
    # All wordpieces overlapping the supplied complete entity span. The pinned
    # normalizer can remove PDF private-use ligatures; preserve source IDs/bounds.
    global_positions={}
    for e in entities:
        local=e['start']-inp['start'];last=e['end']-inp['start']-1
        char_start=starts[local];char_end=starts[last]+len(inp['tokens'][last])
        matches=[j for j,(a,b)in enumerate(offsets)if a<char_end and b>char_start]
        if not matches:raise ValueError(f"Entity span has no wordpiece: {inp['window_id']} {e['id']} at{char_start}")
        global_positions[e['id']]=matches
    chunks=[];positions={e['id']:[]for e in entities}
    for start in range(0,len(ids),STRIDE):
        stop=min(start+CONTENT,len(ids));number=len(chunks)
        chunks.append({'input_ids':[tokenizer.cls_token_id]+ids[start:stop]+[tokenizer.sep_token_id],
                       'wordpiece_start':start,'wordpiece_end':stop})
        if stop==len(ids):break
    for identity,pieces in global_positions.items():
        for j in pieces:
            covering=[(number,chunk)for number,chunk in enumerate(chunks)
                      if chunk['wordpiece_start']<=j<chunk['wordpiece_end']]
            assert covering,'Chunking lost a supplied entity wordpiece'
            for number,chunk in covering:
                positions[identity].append((number,j-chunk['wordpiece_start']+1,
                                           1/(len(pieces)*len(covering))))
        assert abs(sum(x[2]for x in positions[identity])-1)<1e-10
    assert all(positions.values()),'Chunking lost an entity span'
    return chunks,positions


def validate_model():
    manifest=json.loads((MODEL_DIR/'source_manifest.json').read_text())
    assert manifest['revision']=='ced9d8f5f208712c4a90f98a246fe32155b29995'
    for item in manifest['files']:assert sha(MODEL_DIR/item['name'])==item['sha256'],item['name']
    return manifest


def cache_path(split, inp):
    digest=hashlib.sha256(inp['window_id'].encode()).hexdigest()[:20]
    return CACHE/split/(digest+'.pt')


def prepare(split, freeze_path):
    """Only train/dev source files. No fitting, target reads or test input."""
    import torch
    from transformers import BertModel, AutoTokenizer
    assert split in('train','dev')
    freeze=json.loads(freeze_path.read_text())
    for name,expected in freeze['file_sha256'].items():assert sha(ROOT/name)==expected,name
    validate_model();torch.set_num_threads(2)
    device='mps'if torch.backends.mps.is_available()else'cpu'
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False)
    inputs=json.loads((ROOT/f'data/polyie/{split}_inputs.json').read_text())
    # Resolve every original start and full candidate population before encoder work.
    descriptors=[]
    for inp in inputs:
        chunks,positions=chunk_encoding(inp,tokenizer)
        descriptors.append((inp,chunks,positions,population_count(inp)))
    base=CACHE/split;base.mkdir(parents=True,exist_ok=True)
    plan_sha=sha(freeze_path)
    state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'pid':__import__('os').getpid(),
        'split':split,'device':device,'target_files_read':False,'test_input_read':False,
        'freeze_sha256':plan_sha,'windows':len(inputs),'complete_windows':0,
        'owner_candidate_population':sum(x[3]for x in descriptors),'status':'running'}
    (base/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    model=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,
        trust_remote_code=False,use_safetensors=True).to(device).eval()
    for inp,chunks,positions,count in descriptors:
        path=cache_path(split,inp)
        input_sha=hashlib.sha256(json.dumps(inp,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['source_sha256']==input_sha and saved['freeze_sha256']==plan_sha
        else:
            tensors=[]
            with torch.no_grad():
                for chunk in chunks:
                    word_ids=torch.tensor([chunk['input_ids']],device=device)
                    mask=torch.ones_like(word_ids);extended=model.get_extended_attention_mask(mask,mask.shape)
                    hidden=model.embeddings(input_ids=word_ids,token_type_ids=torch.zeros_like(word_ids))
                    for layer in model.encoder.layer[:11]:hidden=layer(hidden,attention_mask=extended)[0]
                    tensors.append({'hidden':hidden.detach().cpu().float(),
                        'mask':extended.detach().cpu().float()})
            saved={'source_sha256':input_sha,'freeze_sha256':plan_sha,'chunks':tensors,
                   'positions':positions,'window_id':inp['window_id'],'candidate_population':count}
            temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path)
        state['complete_windows']+=1
        (base/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'split':split,'window':inp['window_id'],'chunks':len(chunks),
            'owner_candidates':count,'complete':state['complete_windows'],'total':len(inputs)}),flush=True)
    state.update(status='completed_source_cache',ended_at_utc=datetime.now(timezone.utc).isoformat(),
        cache_sha256={str(p.relative_to(ROOT)):sha(p)for p in base.glob('*.pt')})
    (base/'state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev'],required=True)
    ap.add_argument('--freeze',type=Path,default=ROOT/'research/polyie_local_source_cache_freeze.json')
    args=ap.parse_args();prepare(args.split,args.freeze)
