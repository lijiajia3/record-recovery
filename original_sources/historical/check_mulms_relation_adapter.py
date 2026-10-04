"""Invented source/pair/selection checks; no scientific annotation or API reads."""
import copy
import numpy as np
import torch
from torch import nn
import mulms_local_relation_training as training
from mulms_relation_adapter import RelationHead,pair_candidates,pair_supervision,explicit_features,records_from_probabilities,source_entity_matrix
from mulms_local_adapter import TYPE_IDS,RELATION_IDS,span_key
from mulms_experiment import canonical,valid_key


def main():
    text='A B';entities=[{'start':0,'end':1,'text':'A','type':'MAT'},{'start':2,'end':3,'text':'B','type':'FORM'}]
    pairs=pair_candidates(entities);assert pairs==[(0,0),(0,1),(1,0),(1,1)]
    records=[{'h':{'text':'A','occurrence':0,'type':'MAT'},'t':{'text':'B','occurrence':0,'type':'FORM'},'r':r}for r in RELATION_IDS[:2]]
    selected,targets=pair_supervision(entities,records,text,np.random.default_rng(1))
    assert len(selected)==4 and selected[0]==(0,1)and targets[0].sum()==2 and targets[1:].sum()==0
    noedge_selected,noedge_target=pair_supervision(entities,[],text,np.random.default_rng(1))
    assert len(noedge_selected)==4 and noedge_target.sum()==0
    try:pair_supervision(entities,records+[dict(records[0],t={'text':'A','occurrence':0,'type':'VALUE'})],text,np.random.default_rng(1))
    except AssertionError:pass
    else:raise AssertionError('Missing supervised endpoint was silently omitted')
    feature=explicit_features(entities,pairs,len(text),'cpu').numpy()
    assert feature.shape==(4,31)and feature[:,0:26].sum(axis=1).tolist()==[2]*4
    assert feature[1,26]>0 and feature[2,26]<0 and feature[0,28]==1 and feature[1,28]==0
    token=torch.randn(2,768,requires_grad=True)
    blank={'start':1,'end':2,'text':' ','type':'MAT'}
    try:source_entity_matrix(token,[(0,1),(2,3)],[entities[0],blank,entities[1]])
    except ValueError:pass
    else:raise AssertionError('Default inference silently accepted empty-source entity')
    retained=source_entity_matrix(token,[(0,1),(2,3)],[entities[0],blank,entities[1]],allow_empty_source=True)
    assert retained.shape==(3,768)and torch.count_nonzero(retained[1])==0
    assert torch.equal(retained[0],token[0])and torch.equal(retained[2],token[1])
    retained.sum().backward();assert torch.equal(token.grad,torch.ones_like(token))
    torch.manual_seed(3);typed=RelationHead(nn.Linear(1,1),'typed')
    torch.manual_seed(3);matched=RelationHead(nn.Linear(1,1),'capacity_mean')
    assert sum(p.numel()for p in typed.parameters())==sum(p.numel()for p in matched.parameters())
    assert list(typed.state_dict())==list(matched.state_dict())
    assert all(torch.equal(typed.state_dict()[k],v)for k,v in matched.state_dict().items())
    matrix=torch.randn((2,768));matched.eval();out=matched.pairs(matrix,entities,[(0,1),(1,0)],len(text))
    assert torch.allclose(out[0],out[1],atol=1e-6,rtol=1e-6)
    out.sum().backward();assert torch.count_nonzero(matched.geometry[0].weight.grad)==0
    probabilities=np.zeros((4,len(RELATION_IDS)),dtype=np.float32);probabilities[1,:2]=1;probabilities[0,0]=1
    prediction=records_from_probabilities(entities,pairs,probabilities,text,.5)
    assert len(prediction)==3 and sum(valid_key(canonical(r,text))for r in prediction)==2
    assert {canonical(r,text)for r in records}<={canonical(r,text)for r in prediction}

    # All Gold endpoints are retained in evaluation even when not detected.
    class Fake:
        def eval(self):pass
        def entities(self,*_):return matrix
        def pairs(self,_matrix,_entities,batch,_length):
            logits=torch.full((len(batch),len(RELATION_IDS)),-20.)
            for row,pair in enumerate(batch):
                if pair==(0,1):logits[row,:2]=4
                if pair==(0,0):logits[row,0]=4
            return logits
    missing={'h':{'text':'A','occurrence':0,'type':'VALUE'},'t':{'text':'B','occurrence':0,'type':'FORM'},'r':RELATION_IDS[0]}
    inputs=[{'id':'a','doc_key':'paper','text':text},{'id':'b','doc_key':'paper','text':text}]
    oldload,oldevent=training.load_source,training.event;training.load_source=lambda *_:None;training.event=lambda *_:None
    try:
        counts=training.development_counts(Fake(),inputs,{'a':records+[missing],'b':records},{'a':entities,'b':[]},'cpu','typed',1,1)
    finally:training.load_source,training.event=oldload,oldevent
    assert counts['0.5']['paper']=={'tp':2,'fp':1,'fn':3},counts
    assert counts['0.995']['paper']=={'tp':0,'fp':0,'fn':5}
    results={str(seed):{str(e):{'development_counts':{str(t):{'paper':{'tp':1 if e==2 and t==.9 else 0,'fp':0,'fn':1 if e==2 and t==.9 else 2}}
        for t in training.THRESHOLDS}}for e in [1,2,3]}for seed in training.SEEDS}
    chosen=training.select(results);assert chosen['chosen']['epoch']==2 and chosen['chosen']['threshold']==.9 and chosen['chosen']['pooled']['tp']==3
    del results[str(training.SEEDS[-1])]['3']
    try:training.select(results)
    except AssertionError:pass
    else:raise AssertionError('Pooled choice selected before all three seeds/epochs')
    print('PASS: full/self pairs, multilabel BCE, no-edge negatives, missing Gold retained, nominal equal initialization, restricted mean invariance, all-seed pooled selection')


if __name__=='__main__':main()
