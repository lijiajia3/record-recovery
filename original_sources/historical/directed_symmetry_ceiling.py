"""Gold-aware descriptive orbit bound; no dataset/file/model interface."""
import itertools


def reverse(edge):
    assert len(edge) == 3 and edge[0] != edge[1]
    return edge[1], edge[0], edge[2]


def counts_f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def ceiling(gold, available_entities=None):
    gold = set(gold)
    assert all(len(e) == 3 and e[0] != e[1] for e in gold)
    reachable = gold if available_entities is None else {
        e for e in gold if e[0] in available_entities and e[1] in available_entities}
    closure = reachable | {reverse(e) for e in reachable}
    tp, fp, fn = len(closure & gold), len(closure - gold), len(gold - closure)
    # Reversing an edge keeps both endpoints, so reachable is closed with respect
    # to endpoint availability, even when its same-label reverse is not Gold.
    assert tp == len(reachable)
    return {'gold_edges': len(gold), 'reachable_gold_edges': len(reachable),
            'reachable_single_direction_gold_orbits': fp,
            'oracle_counts': {'tp': tp, 'fp': fp, 'fn': fn},
            'gold_aware_symmetric_oracle_f1': counts_f1(tp, fp, fn)}


def graph_diagnostic(edges):
    edges = set(edges)
    assert all(len(e) == 3 and e[0] != e[1] for e in edges)
    return {'valid_non_diagonal_predicted_edges': len(edges),
            'edges_missing_same_label_reverse': sum(reverse(e) not in edges for e in edges)}


def exhaustive_check():
    # Invalid diagonal emissions are outside valid reverse orbits. They add a
    # false positive and cannot improve the valid-edge optimum.
    toy = ceiling({('a', 'b', 'L1')})
    c = toy['oracle_counts']
    assert counts_f1(c['tp'], c['fp'] + 1, c['fn']) < toy['gold_aware_symmetric_oracle_f1']
    try:
        reverse(('a', 'a', 'L1'))
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid diagonal treated as valid reverse orbit')
    # Three directed endpoint-pair orbits, two labels, plus unreachable endpoints.
    universe = {(h, t, label) for h in ('a', 'b', 'c') for t in ('a', 'b', 'c')
                if h != t for label in ('L1', 'L2')}
    orbits = sorted({frozenset((e, reverse(e))) for e in universe}, key=repr)
    checked = 0
    for gold_bits in range(1 << len(universe)):
        ordered = sorted(universe)
        gold = {e for i, e in enumerate(ordered) if (gold_bits >> i) & 1}
        for available in ({'a', 'b', 'c'}, {'a', 'b'}, set()):
            result = ceiling(gold, available)
            eligible_orbits = [o for o in orbits if all(e[0] in available and e[1] in available for e in o)]
            optimum = 0.0
            for choices in itertools.product((False, True), repeat=len(eligible_orbits)):
                predicted = set().union(*(o for o, take in zip(eligible_orbits, choices) if take))
                f1 = counts_f1(len(predicted & gold), len(predicted - gold), len(gold - predicted))
                optimum = max(optimum, f1)
            assert abs(optimum - result['gold_aware_symmetric_oracle_f1']) <= 1e-12
            checked += 1
    return checked


if __name__ == '__main__':
    print({'invented_exhaustive_gold_and_entity_availability_cases': exhaustive_check(),
           'dataset_gold_loaded': False, 'model_or_API_calls': 0})
