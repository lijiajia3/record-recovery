"""Explicit role/relative-position interaction ablation; source inputs only."""
import numpy as np
import torch
from torch import nn
from polyie_local_baseline_training import Classifier as MeanClassifier


class TypedClassifier(MeanClassifier):
    def __init__(self,last_layer):
        nn.Module.__init__(self);self.last_layer=last_layer
        self.entity_norm=nn.LayerNorm(768)
        self.content=nn.Sequential(nn.Linear(4*768,128),nn.GELU())
        self.geometry=nn.Sequential(nn.Linear(13,32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        self.source_entities=None

    def bind_source(self,inp):
        self.source_entities={e['id']:e for e in inp['entities']}

    def candidates(self,identities,matrix,rows):
        assert self.source_entities is not None,'Bind supplied source inventory before scoring'
        index={identity:i+1 for i,identity in enumerate(identities)}
        indices=np.zeros((len(rows),4),dtype=np.int64);geometry=np.zeros((len(rows),13),dtype=np.float32)
        starts=np.zeros((len(rows),4),dtype=np.float32);condition=np.zeros(len(rows),dtype=bool)
        pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
        for number,ids in enumerate(rows):
            for role,identity in enumerate(ids):
                indices[number,role]=index[identity];starts[number,role]=self.source_entities[identity]['start']
            condition[number]=len(ids)==4
        for pair,(a,b)in enumerate(pairs):
            delta=starts[:,b]-starts[:,a]
            value=np.sign(delta)*np.log1p(np.abs(delta))/np.log1p(1000)
            if b==3:value=np.where(condition,value,0)
            geometry[:,pair]=value;geometry[:,pair+6]=np.abs(value)
        geometry[:,12]=condition
        normalized=self.entity_norm(matrix)
        full=torch.cat([normalized.new_zeros((1,768)),normalized],dim=0)
        content=full[torch.from_numpy(indices).to(matrix.device)].flatten(start_dim=1)
        content=self.content(content);position=self.geometry(torch.from_numpy(geometry).to(matrix.device))
        return self.head(torch.cat([content,position],dim=1)).flatten()
