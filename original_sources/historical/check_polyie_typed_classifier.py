"""Invented-source ablation check: role order, position-only features and gradients."""
import torch
from torch import nn
from polyie_typed_classifier import TypedClassifier


class IdentityLayer(nn.Module):
    def forward(self,hidden,attention_mask=None):return(hidden,)


torch.manual_seed(7);model=TypedClassifier(IdentityLayer())
inp={'entities':[{'id':str(i),'start':s}for i,s in [(1,10),(2,20),(3,30),(4,40)]]}
model.bind_source(inp);matrix=torch.randn(4,768,requires_grad=True)
seen=[]
handle=model.geometry[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
model.eval();logits=model.candidates(['1','2','3','4'],matrix,[('1','2','3'),('1','2','3','4')]);handle.remove()
assert logits.shape==(2,)and torch.isfinite(logits).all()
geometry=seen[0]
assert geometry.shape==(2,13)and geometry[0,12]==0 and geometry[1,12]==1
assert geometry[0,2]==0 and geometry[0,4]==0 and geometry[0,5]==0
assert torch.equal(geometry[:,:6].abs(),geometry[:,6:12])
assert geometry[1,2]>geometry[1,1]>geometry[1,0]>0
logits.sum().backward();assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
assert set(model.state_dict())and not hasattr(model,'norm'),'Do not retain unused mean-head weights'
print('PASS invented fixed role ordering, signed/absolute source geometry, missing Condition and gradient flow')
