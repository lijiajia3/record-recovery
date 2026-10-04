"""Independent fitting-control checks, including owner labels and pooled ties."""
import numpy as np
import torch
from torch import nn
from polyie_local_baseline_training import Classifier, owner_positives, select, full_document_sensitivity, SEEDS, THRESHOLDS, EPOCHS


class IdentityLayer(nn.Module):
    def forward(self,hidden,attention_mask=None):return (hidden,)


model=Classifier(IdentityLayer())
saved={'chunks':[{'hidden':torch.arange(4*768,dtype=torch.float32).reshape(1,4,768),'mask':torch.zeros(1,1,1,4)},
                 {'hidden':torch.arange(3*768,dtype=torch.float32).reshape(1,3,768)+10000,'mask':torch.zeros(1,1,1,3)}],
       'positions':{'a':[(0,1,.25),(1,1,.25),(0,2,.5)],'b':[(0,3,1.)],'c':[(1,2,1.)]}}
ids,matrix=model.entities(saved,'cpu')
expected=.25*saved['chunks'][0]['hidden'][0,1]+.25*saved['chunks'][1]['hidden'][0,1]+.5*saved['chunks'][0]['hidden'][0,2]
assert torch.equal(matrix[0],expected)
model.eval();scores=model.candidates(ids,matrix,[('a','b','c')]);assert scores.shape==(1,)and torch.isfinite(scores).all()
entities=[{'id':'1','type':'CN','start':800,'end':801},{'id':'2','type':'PN','start':830,'end':831},
          {'id':'3','type':'PV','start':900,'end':901},{'id':'4','type':'PV','start':1000,'end':1001}]
inp={'start':800,'doc_key':'p','entities':entities}
labels={'p':{'eligible':[{'CN':['1'],'PN':['2'],'PV':['3'],'Condition':[]},
                         {'CN':['1'],'PN':['2'],'PV':['4'],'Condition':[]}]}}
assert owner_positives(inp,labels)=={('1','2','4')}
rows={str(seed):{str(epoch):{'development_counts':{str(t):{'p':{'tp':2,'fp':1,'fn':3}}for t in THRESHOLDS}}
                    for epoch in range(1,EPOCHS+1)}for seed in SEEDS}
choice=select(rows)['chosen'];assert choice['epoch']==1 and choice['threshold']==max(THRESHOLDS)
assert(choice['tp'],choice['fp'],choice['fn'])==(6,3,9)
# Stronger one-seed result cannot select an epoch over the genuinely pooled total.
rows[str(SEEDS[0])]['2']['development_counts']['0.5']['p']={'tp':5,'fp':0,'fn':0}
for seed in SEEDS[1:]:rows[str(seed)]['2']['development_counts']['0.5']['p']={'tp':0,'fp':100,'fn':5}
assert select(rows)['chosen']['epoch']==1
group={'CN':['1'],'PN':['2'],'PV':['3'],'Condition':[]}
uncovered={'CN':['1'],'PN':['2'],'PV':['other'],'Condition':[]}
full=full_document_sensitivity({'0.5':{'p':{'tp':1,'fp':2,'fn':0}}},
    {'p':{'eligible':[group],'all_valid':[group,uncovered]}})
assert full['0.5']['p']=={'tp':1,'fp':2,'fn':1}
print('PASS independent weighted-span pooling, no-condition logit, owner positives and pooled three-seed tie/selection')
