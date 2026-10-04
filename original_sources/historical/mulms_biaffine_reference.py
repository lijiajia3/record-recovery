"""Independent classical directed head; no annotation or selection loaders."""
import torch
from torch import nn
from mulms_local_adapter import TYPE_IDS,RELATION_IDS,token_features
from mulms_relation_adapter import source_entity_matrix


class BiaffineReference(nn.Module):
    def __init__(self,last_layer,architecture='biaffine'):
        super().__init__();assert architecture=='biaffine';self.architecture=architecture
        self.last_layer=last_layer;self.norm=nn.LayerNorm(768);self.types=nn.Embedding(len(TYPE_IDS),32)
        self.head=nn.Sequential(nn.Linear(800,128),nn.GELU(),nn.Dropout(.1))
        self.tail=nn.Sequential(nn.Linear(800,128),nn.GELU(),nn.Dropout(.1))
        self.biaffine=nn.Parameter(torch.empty(len(RELATION_IDS),129,129))
        for label in self.biaffine:nn.init.xavier_uniform_(label)

    def entities(self,saved,entities,device,allow_empty_source=False):
        return source_entity_matrix(token_features(self.last_layer,saved,device),saved['offsets'],entities,
            allow_empty_source=allow_empty_source)

    def pairs(self,matrix,entities,pairs,text_length):
        kinds=torch.tensor([TYPE_IDS.index(e['type'])for e in entities],device=matrix.device)
        content=torch.cat([self.norm(matrix),self.types(kinds)],dim=1)
        h=self.head(content);t=self.tail(content);h=torch.cat([h,h.new_ones((len(h),1))],dim=1)
        t=torch.cat([t,t.new_ones((len(t),1))],dim=1);index=torch.tensor(pairs,device=matrix.device)
        return torch.einsum('ni,kij,nj->nk',h[index[:,0]],self.biaffine,t[index[:,1]])
