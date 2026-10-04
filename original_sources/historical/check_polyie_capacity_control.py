"""Independent same-capacity, order-invariance and geometry-absence checks."""
import torch
from torch import nn
from polyie_typed_classifier import TypedClassifier
from polyie_capacity_mean_classifier import CapacityMeanClassifier


class IdentityLayer(nn.Module):
    def forward(self,hidden,attention_mask=None):return(hidden,)


torch.manual_seed(9);typed=TypedClassifier(IdentityLayer())
torch.manual_seed(9);control=CapacityMeanClassifier(IdentityLayer())
assert sum(p.numel()for p in typed.parameters())==sum(p.numel()for p in control.parameters())
assert all(torch.equal(typed.state_dict()[k],v)for k,v in control.state_dict().items())
matrix=torch.randn(4,768);seen=[]
hook=control.geometry[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
control.eval();out=control.candidates(['a','b','c','d'],matrix,[('a','b','c'),('c','a','b'),('a','b','c','d')]);hook.remove()
assert torch.allclose(out[0],out[1],rtol=1e-6,atol=1e-6)and seen[0].shape==(3,13)and torch.count_nonzero(seen[0])==0
print('PASS matched architecture/initial weights, permutation-invariant control and absent geometry')
