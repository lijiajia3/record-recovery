"""Text-only nested spans and source representations; no annotation file loader."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from polyie_local_adapter import ROOT,MODEL_DIR,sha,validate_model
from mulms_experiment import TYPES,LABELS

TYPE_IDS=tuple(sorted(TYPES));RELATION_IDS=tuple(sorted(LABELS))
WIDTH=32;CONTENT=510;STRIDE=384
OUTPUT=ROOT/'results/local_baseline/mulms_supervised_v1'
CACHE=OUTPUT/'source_cache'


def source_encoding(inp,tokenizer):
    assert set(inp)=={'id','doc_key','text'},'Only original sentence input may enter encoding'
    encoded=tokenizer(inp['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping'];chunks=[]
    if not ids:
        chunks.append({'start':0,'end':0,'ids':[tokenizer.cls_token_id,tokenizer.sep_token_id]})
    else:
        for start in range(0,len(ids),STRIDE):
            end=min(start+CONTENT,len(ids))
            chunks.append({'start':start,'end':end,'ids':[tokenizer.cls_token_id]+ids[start:end]+[tokenizer.sep_token_id]})
            if end==len(ids):break
    return {'chunks':chunks,'offsets':offsets,'wordpieces':len(ids)}


def span_candidates(offsets):
    """First source representation of each exact character span, without Gold."""
    candidates=[];seen=set()
    for a,(start,_)in enumerate(offsets):
        for b in range(a,min(a+WIDTH,len(offsets))):
            end=offsets[b][1]
            if start>=end:continue
            identity=(int(start),int(end))
            if identity not in seen:
                seen.add(identity);candidates.append((a,b,int(start),int(end)))
    return candidates


def span_key(entity):return int(entity['start']),int(entity['end']),entity['type']


def source_cache_path(split,inp):
    return CACHE/split/(hashlib.sha256(inp['id'].encode()).hexdigest()[:20]+'.pt')


def token_features(last_layer,saved,device):
    pieces=[];global_ids=[]
    for chunk in saved['chunks']:
        hidden=last_layer(chunk['hidden'].to(device),attention_mask=chunk['mask'].to(device))[0][0]
        pieces.append(hidden[1:1+chunk['end']-chunk['start']])
        global_ids.extend(range(chunk['start'],chunk['end']))
    if saved['wordpieces']==0:return pieces[0].new_zeros((0,768))
    features=torch.cat(pieces,dim=0)
    indices=torch.tensor(global_ids,dtype=torch.long,device=device)
    sums=features.new_zeros((saved['wordpieces'],768)).index_add(0,indices,features)
    counts=features.new_zeros(saved['wordpieces']).index_add(0,indices,features.new_ones(len(indices)))
    assert torch.all(counts>0),'Missing source WordPiece, never silently truncate'
    return sums/counts[:,None]


class SpanDetector(nn.Module):
    def __init__(self,last_layer):
        super().__init__();self.last_layer=last_layer;self.norm=nn.LayerNorm(768)
        self.width=nn.Embedding(WIDTH+1,32)
        self.head=nn.Sequential(nn.Linear(768*3+32,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,len(TYPE_IDS)))

    def tokens(self,saved,device):return token_features(self.last_layer,saved,device)

    def spans(self,tokens,candidates):
        normalized=self.norm(tokens);cumulative=torch.cat([normalized.new_zeros((1,768)),normalized.cumsum(dim=0)],dim=0)
        a=torch.tensor([c[0]for c in candidates],dtype=torch.long,device=tokens.device)
        b=torch.tensor([c[1]for c in candidates],dtype=torch.long,device=tokens.device)
        size=b-a+1;mean=(cumulative[b+1]-cumulative[a])/size[:,None]
        return self.head(torch.cat([normalized[a],normalized[b],mean,self.width(size)],dim=1))


def span_supervision(candidates,entities,rng):
    labels={}
    for entity in entities:labels.setdefault((entity['start'],entity['end']),set()).add(entity['type'])
    positive=[i for i,c in enumerate(candidates)if(c[2],c[3])in labels]
    negatives=[i for i,c in enumerate(candidates)if(c[2],c[3])not in labels]
    take=min(len(negatives),10*max(1,len(positive)))
    negative=sorted(map(int,rng.choice(negatives,size=take,replace=False)))if take else[]
    selected=positive+negative;target=np.zeros((len(selected),len(TYPE_IDS)),dtype=np.float32)
    for row,i in enumerate(positive):
        for kind in labels[candidates[i][2],candidates[i][3]]:target[row,TYPE_IDS.index(kind)]=1
    return selected,target


def entities_from_probs(inp,candidates,probabilities,threshold):
    assert probabilities.shape==(len(candidates),len(TYPE_IDS))and np.isfinite(probabilities).all()
    result=[]
    for row,kind in np.argwhere(probabilities>=threshold):
        _,_,start,end=candidates[int(row)];surface=inp['text'][start:end]
        assert surface
        result.append({'start':start,'end':end,'type':TYPE_IDS[int(kind)],'text':surface,
            'probability':float(probabilities[row,kind])})
    assert len(result)==len({span_key(e)for e in result})
    return result


def entity_matrix(tokens,offsets,entities):
    """Full mention WordPieces, including long train-only supervised Gold spans."""
    pool=np.zeros((len(entities),len(offsets)),dtype=np.float32)
    for row,entity in enumerate(entities):
        a,b=entity['start'],entity['end']
        covered=[i for i,(start,end)in enumerate(offsets)if start<b and end>a]
        if not covered:raise ValueError('Entity has no source WordPiece overlap; no silent omission')
        pool[row,covered]=1/len(covered)
    return torch.from_numpy(pool).to(tokens.device)@tokens
