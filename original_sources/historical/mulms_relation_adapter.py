"""Prospectively declared full directed-pair, multi-label relation heads.

This source representation module has no annotation-file loader or selection.
"""
import math
import numpy as np
import torch
from torch import nn
from mulms_local_adapter import TYPE_IDS,RELATION_IDS,token_features,entity_matrix,span_key
from mulms_experiment import canonical,valid_key,occurrences


def pair_candidates(entities):
    assert len(entities)==len({span_key(e)for e in entities})
    return [(h,t)for h in range(len(entities))for t in range(len(entities))]


def pair_supervision(entities,records,text,rng):
    pairs=pair_candidates(entities);index={span_key(e):i for i,e in enumerate(entities)};labels={}
    for record in records:
        key=canonical(record,text)
        assert valid_key(key)and key[0]in index and key[1]in index,'Missing supervised endpoint; never omit Gold'
        pair=index[key[0]],index[key[1]]
        labels.setdefault(pair,set()).add(key[2])
    positive=[i for i,pair in enumerate(pairs)if pair in labels]
    negative=[i for i,pair in enumerate(pairs)if pair not in labels]
    count=min(len(negative),10*max(1,len(positive)))
    chosen=sorted(map(int,rng.choice(negative,size=count,replace=False)))if count else[]
    selected=positive+chosen;targets=np.zeros((len(selected),len(RELATION_IDS)),dtype=np.float32)
    for row,i in enumerate(positive):
        for label in labels[pairs[i]]:targets[row,RELATION_IDS.index(label)]=1
    return [pairs[i]for i in selected],targets


def explicit_features(entities,pairs,text_length,device):
    result=np.zeros((len(pairs),31),dtype=np.float32);denominator=math.log1p(max(1,text_length))
    for row,(h,t)in enumerate(pairs):
        a,b=entities[h],entities[t]
        result[row,TYPE_IDS.index(a['type'])]=1
        result[row,13+TYPE_IDS.index(b['type'])]=1
        distance=b['start']-a['start'];signed=math.copysign(math.log1p(abs(distance)),distance)/denominator
        result[row,26:]=[signed,abs(signed),int(a['start']<b['end']and b['start']<a['end']),
            math.log1p(a['end']-a['start'])/denominator,math.log1p(b['end']-b['start'])/denominator]
    return torch.from_numpy(result).to(device)


def source_entity_matrix(tokens,offsets,entities,allow_empty_source=False):
    """Retain empty-token source spans with an explicit, fixed zero representation.

    Three TRAIN annotations have no WordPiece overlap under the frozen tokenizer.
    No character/type/pair/target is removed. Predicted dev/test entities are
    source WordPiece candidates and must never reach this fallback.
    """
    supported=[i for i,e in enumerate(entities)if any(a<e['end']and b>e['start']for a,b in offsets)]
    if len(supported)!=len(entities)and not allow_empty_source:
        raise ValueError('Empty source representation prohibited outside explicit train supervision')
    result=tokens.new_zeros((len(entities),768))
    if supported:
        values=entity_matrix(tokens,offsets,[entities[i]for i in supported])
        result=result.index_copy(0,torch.tensor(supported,device=tokens.device),values)
    return result


class RelationHead(nn.Module):
    def __init__(self,last_layer,architecture):
        super().__init__();assert architecture in ['mean','typed','capacity_mean']
        self.last_layer=last_layer;self.architecture=architecture;self.norm=nn.LayerNorm(768)
        if architecture=='mean':
            self.output=nn.Sequential(nn.Dropout(.1),nn.Linear(768,len(RELATION_IDS)))
        else:
            self.content=nn.Sequential(nn.Linear(1536,128),nn.GELU())
            self.geometry=nn.Sequential(nn.Linear(31,32),nn.GELU())
            self.output=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,len(RELATION_IDS)))

    def entities(self,saved,entities,device,allow_empty_source=False):
        return source_entity_matrix(token_features(self.last_layer,saved,device),saved['offsets'],entities,allow_empty_source=allow_empty_source)

    def pairs(self,matrix,entities,pairs,text_length):
        indices=torch.tensor(pairs,dtype=torch.long,device=matrix.device);h=matrix[indices[:,0]];t=matrix[indices[:,1]]
        if self.architecture=='mean':return self.output(self.norm((h+t)/2))
        h,t=self.norm(h),self.norm(t)
        if self.architecture=='typed':
            content=torch.cat([h,t],dim=1);features=explicit_features(entities,pairs,text_length,matrix.device)
        else:
            mean=(h+t)/2;content=torch.cat([mean,mean],dim=1);features=matrix.new_zeros((len(pairs),31))
        return self.output(torch.cat([self.content(content),self.geometry(features)],dim=1))


def source_endpoint(entity,text):
    start,end,kind=span_key(entity);surface=text[start:end]
    assert surface==entity['text']and surface
    return {'text':surface,'occurrence':occurrences(text,surface).index(start),'type':kind}


def records_from_probabilities(entities,pairs,probabilities,text,threshold):
    assert probabilities.shape==(len(pairs),len(RELATION_IDS))and np.isfinite(probabilities).all()
    records=[]
    for row,label in np.argwhere(probabilities>=threshold):
        h,t=pairs[int(row)];a,b=entities[h],entities[t]
        records.append({'h':source_endpoint(a,text),'t':source_endpoint(b,text),'r':RELATION_IDS[int(label)],
            'evidence':text[min(a['start'],b['start']):max(a['end'],b['end'])]})
    return records
