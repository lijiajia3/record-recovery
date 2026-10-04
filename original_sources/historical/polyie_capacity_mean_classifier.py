"""Same learned architecture as typed head, with unordered mean and zero geometry."""
import numpy as np
import torch
from polyie_typed_classifier import TypedClassifier


class CapacityMeanClassifier(TypedClassifier):
    def candidates(self,identities,matrix,rows):
        index={identity:i+1 for i,identity in enumerate(identities)}
        indices=np.zeros((len(rows),4),dtype=np.int64);sizes=np.zeros((len(rows),1),dtype=np.float32)
        for number,ids in enumerate(rows):
            sizes[number]=len(ids)
            for role,identity in enumerate(ids):indices[number,role]=index[identity]
        normalized=self.entity_norm(matrix)
        full=torch.cat([normalized.new_zeros((1,768)),normalized],dim=0)
        mean=full[torch.from_numpy(indices).to(matrix.device)].sum(dim=1)/torch.from_numpy(sizes).to(matrix.device)
        content=self.content(mean.repeat(1,4))
        position=self.geometry(matrix.new_zeros((len(rows),13)))
        return self.head(torch.cat([content,position],dim=1)).flatten()
