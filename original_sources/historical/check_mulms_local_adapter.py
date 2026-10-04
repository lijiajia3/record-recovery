"""Independent invented spans: nesting, normalization, no truncation and negatives."""
import numpy as np
import torch
from torch import nn
from mulms_local_adapter import source_encoding,span_candidates,span_supervision,entities_from_probs,TYPE_IDS,token_features,entity_matrix


class FakeTokenizer:
    cls_token_id=101;sep_token_id=102
    def __call__(self,text,**kwargs):return {'input_ids':list(range(len(text))),'offset_mapping':[(i,i+1)for i in range(len(text))]}


class IdentityLayer(nn.Module):
    def forward(self,hidden,attention_mask=None):return(hidden,)


source={'id':'fixture','doc_key':'invented','text':'a'*600}
encoded=source_encoding(source,FakeTokenizer());assert [(c['start'],c['end'])for c in encoded['chunks']]==[(0,510),(384,600)]
saved={'wordpieces':600,'chunks':[]}
for c in encoded['chunks']:
    values=torch.arange(c['start'],c['end']).float()[:,None].repeat(1,768)
    hidden=torch.cat([torch.zeros(1,768),values,torch.zeros(1,768)],dim=0)[None]
    saved['chunks'].append(dict(c,hidden=hidden,mask=torch.zeros(1)))
tokens=token_features(IdentityLayer(),saved,'cpu');assert torch.equal(tokens[:,0],torch.arange(600).float())
offsets=[(0,2),(2,3),(4,6)];candidates=span_candidates(offsets)
assert len(candidates)==6 and (0,1,0,3)in candidates and (1,2,2,6)in candidates
entities=[{'start':0,'end':3,'type':'MAT'},{'start':2,'end':3,'type':'NUM'},
    {'start':0,'end':3,'type':'FORM'},{'start':1,'end':2,'type':'UNIT'}]
selected,target=span_supervision(candidates,entities,np.random.default_rng(1))
assert target.sum()==3 and len(selected)==6 # unsupported boundary never inserted
zero=np.zeros((6,len(TYPE_IDS)));zero[1,TYPE_IDS.index('MAT')]=.9;zero[3,TYPE_IDS.index('NUM')]=.9
out=entities_from_probs(dict(source,text='aab cd'),candidates,zero,.5)
assert {(e['start'],e['end'],e['type'])for e in out}=={(0,3,'MAT'),(2,3,'NUM')}
matrix=entity_matrix(torch.arange(3).float()[:,None].repeat(1,768),offsets,[{'start':1,'end':6}])
assert torch.allclose(matrix,torch.ones(1,768)) # all overlapping WordPieces retained
empty_indices,empty_targets=span_supervision(candidates,[],np.random.default_rng(2))
assert len(empty_indices)==6 and empty_targets.sum()==0
print('PASS source-complete chunk averaging, nested/multi-type spans, unsupported boundary and full negatives')
