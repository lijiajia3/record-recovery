"""Invented nested-span dev/selection checks; no real files, models or API."""
import numpy as np
import torch
import mulms_local_ner_training as module
from mulms_local_adapter import TYPE_IDS


class InventedDetector:
    def eval(self):return self
    def tokens(self,saved,device):return torch.zeros(2,768)
    def spans(self,tokens,candidates):
        logits=torch.full((len(candidates),len(TYPE_IDS)),-100.)
        for i,c in enumerate(candidates):
            if c[2:]==(0,2):logits[i,TYPE_IDS.index('MAT')]=8
            if c[2:]==(0,3):logits[i,TYPE_IDS.index('NUM')]=.5
        return logits


module.event=lambda value:None
module.load_source=lambda split,inp:{'offsets':[(0,2),(2,3)]}
inputs=[{'id':'fixture','doc_key':'invented','text':'aab'}]
labels={'fixture':[{'start':0,'end':2,'type':'MAT'}, {'start':1,'end':3,'type':'UNIT'}]}
counts=module.development_counts(InventedDetector(),inputs,labels,'cpu',1,1)
assert counts['0.5']['invented']=={'tp':1,'fp':1,'fn':1}
assert counts['0.99']['invented']=={'tp':1,'fp':0,'fn':1}
results={}
for index,seed in enumerate(module.SEEDS):
    results[str(seed)]={}
    for epoch in range(1,4):
        if epoch==1:cell={'tp':10 if index==0 else 0,'fp':0,'fn':0 if index==0 else 10}
        elif epoch==2:cell={'tp':7,'fp':1,'fn':3}
        else:cell={'tp':6,'fp':1,'fn':4}
        results[str(seed)][str(epoch)]={'development_counts':{str(t):{'invented':cell}for t in module.THRESHOLDS}}
selection=module.select(results);assert selection['chosen']['epoch']==2 and selection['chosen']['threshold']==.99
assert selection['chosen']['pooled']['tp']==21 and selection['chosen']['pooled']['fp']==3 and selection['chosen']['pooled']['fn']==9
try:module.select({k:v for k,v in results.items()if k!=str(module.SEEDS[-1])})
except AssertionError:pass
else:raise AssertionError('Partial seeds reached common selection')
print('PASS full unsupported-boundary FN, invalid prediction FP, all-seed pooling and no lucky-seed selection')
