"""Invented CPU/MPS shape/direction/backward checks, no annotation reads."""
import torch
from torch import nn
from mulms_biaffine_reference import BiaffineReference
from mulms_local_adapter import RELATION_IDS


def main():
    for device in ['cpu']+(['mps']if torch.backends.mps.is_available()else[]):
        torch.manual_seed(20261003);model=BiaffineReference(nn.Linear(1,1)).to(device);model.eval()
        matrix=torch.randn(3,768,device=device,requires_grad=True)
        entities=[{'type':'MAT'},{'type':'FORM'},{'type':'MEASUREMENT'}]
        logits=model.pairs(matrix,entities,[(0,1),(1,0),(0,0),(2,1)],12)
        assert logits.shape==(4,len(RELATION_IDS))and torch.isfinite(logits).all()
        assert not torch.allclose(logits[0],logits[1]),'Directed reference unexpectedly symmetric'
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)).backward()
        assert model.biaffine.grad is not None and torch.isfinite(model.biaffine.grad).all()
        assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
        print('PASS invented biaffine directed/multilabel/diagonal finite forward-backward:',device)


if __name__=='__main__':main()
