"""Same nominal learned head with ordered context and no explicit geometry.

Both endpoint slots remain distinct, so this control avoids the reverse-edge
invariance of the original capacity-matched unordered mean. No Gold loader.
"""
import torch
from mulms_relation_adapter import RelationHead


class OrderedContextReference(RelationHead):
    def __init__(self, last_layer, architecture='ordered_context'):
        assert architecture == 'ordered_context'
        super().__init__(last_layer, 'typed')
        self.architecture = 'ordered_context'

    def pairs(self, matrix, entities, pairs, text_length):
        indices = torch.tensor(pairs, dtype=torch.long, device=matrix.device)
        h, t = self.norm(matrix[indices[:, 0]]), self.norm(matrix[indices[:, 1]])
        content = torch.cat([h, t], dim=1)
        # Biases of this branch remain trainable; all explicit type/distance/
        # overlap/length inputs are absent. Context embeddings can retain types.
        features = matrix.new_zeros((len(pairs), 31))
        return self.output(torch.cat([self.content(content), self.geometry(features)], dim=1))
