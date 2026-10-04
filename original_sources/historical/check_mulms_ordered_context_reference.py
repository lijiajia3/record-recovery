"""Invented ordered-control initialization/direction/gradient checks."""
import torch
from mulms_relation_adapter import RelationHead
from mulms_ordered_context_reference import OrderedContextReference


def main():
    torch.manual_seed(10203)
    typed = RelationHead(torch.nn.Identity(), 'typed')
    torch.manual_seed(10203)
    ordered = OrderedContextReference(torch.nn.Identity())
    assert sum(p.numel() for p in typed.parameters()) == sum(p.numel() for p in ordered.parameters())
    assert typed.state_dict().keys() == ordered.state_dict().keys()
    assert all(torch.equal(a, b) for a, b in zip(typed.state_dict().values(), ordered.state_dict().values()))
    for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
        ordered = ordered.to(device).eval()
        matrix = torch.randn(3, 768, device=device, requires_grad=True)
        entities = [{'start': i * 3, 'end': i * 3 + 1, 'type': 'MAT', 'text': 'A'} for i in range(3)]
        pairs = [(0, 1), (1, 0), (0, 0)]
        logits = ordered.pairs(matrix, entities, pairs, 20)
        assert logits.shape == (3, 15) and torch.isfinite(logits).all()
        assert not torch.allclose(logits[0], logits[1]), 'Ordered context must not enforce reverse invariance'
        changed = [dict(e, type='NUM', start=10 * i, end=10 * i + 9) for i, e in enumerate(entities)]
        assert torch.equal(logits, ordered.pairs(matrix, changed, pairs, 999)), 'Explicit type/geometry entered control'
        logits.square().mean().backward()
        assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ordered.parameters())
        print(device, 'PASS same initial weights/nominal params; distinct direction; explicit feature invariance; finite backward')
    print('No dataset, tokenizer, fitted checkpoint or API accessed.')


if __name__ == '__main__':
    main()
