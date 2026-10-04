"""Prospective primary exact search v2, preserving full native populations.

The original bound parser/VF2 closure implementation is unchanged. This version
replaces recursive global combination with an iterative exact stack and a valid
Gold-root upper bound. A root with no candidate is mathematically unmatchable;
it remains in the caller's complete FP/FN denominator, not a filtered document.
No annotation, cache, model or filesystem loader is present.
"""
from __future__ import annotations

from sccomics_native_graph import (DEFAULT_MAX_SEARCH_STEPS, MatchingBudgetExceeded,
    _Budget, _CountedMultiDiGraphMatcher, _edge_match, _node_match, rooted_graph)


def exact_graph_match(pred, gold, family, *, max_search_steps=DEFAULT_MAX_SEARCH_STEPS):
    if family not in {'E', 'R'}:
        raise ValueError('exact native family E/R required')
    if type(max_search_steps) is not int or not 1 <= max_search_steps <= DEFAULT_MAX_SEARCH_STEPS:
        raise ValueError('fixed positive search budget at most one million required')
    p, g = pred.family(family, valid_only=True), gold.family(family, valid_only=True)
    budget = _Budget(max_search_steps)
    if not p or not g:
        budget.tick()
        return {'tp': 0, 'matched_roots': [], 'identifier_mapping': {},
            'search_steps': budget.used, 'search_step_limit': max_search_steps,
            'sharing_scope': 'consistent_across_all_matched_root_closures',
            'algorithm': 'iterative_exact_global_v2', 'unmatchable_search_roots': len(p),
            'complete_valid_prediction_roots': len(p), 'complete_valid_Gold_roots': len(g)}
    pg = {r.identifier: rooted_graph(pred, r) for r in p}
    gg = {r.identifier: rooted_graph(gold, r) for r in g}
    choices = {}
    for pr in p:
        options = []
        for gr in g:
            budget.tick()
            if pr.label != gr.label:
                continue
            a, b = pg[pr.identifier], gg[gr.identifier]
            if len(a) != len(b) or a.number_of_edges() != b.number_of_edges():
                continue
            matcher = _CountedMultiDiGraphMatcher(a, b, budget=budget,
                node_match=_node_match, edge_match=_edge_match)
            for mapping in matcher.isomorphisms_iter():
                budget.tick()
                options.append((gr.identifier, mapping))
        choices[pr.identifier] = options
    order = sorted((r for r in p if choices[r.identifier]),
        key=lambda r: (len(choices[r.identifier]), r.occurrence))
    # Each state carries the complete globally consistent forward/inverse map.
    # No Python recursion depth is consumed by many unmatchable/root objects.
    stack = [(0, frozenset(), {}, {}, [])]
    best_pairs, best_map = [], {}
    while stack:
        k, used, forward, inverse, pairs = stack.pop()
        budget.tick()
        if len(pairs) > len(best_pairs):
            best_pairs, best_map = list(pairs), dict(forward)
        possible = min(len(order) - k, len(g) - len(used))
        if len(pairs) + possible <= len(best_pairs) or k == len(order):
            continue
        ident = order[k].identifier
        # LIFO enters candidate branches before the unmatched branch. Reversal
        # retains original candidate order for deterministic tie witnesses.
        stack.append((k + 1, used, forward, inverse, pairs))
        for gold_root, extension in reversed(choices[ident]):
            if gold_root in used:
                continue
            if any((a in forward and forward[a] != b) or
                   (b in inverse and inverse[b] != a) for a, b in extension.items()):
                continue
            merged, inv = dict(forward), dict(inverse)
            merged.update(extension)
            inv.update({b: a for a, b in extension.items()})
            stack.append((k + 1, used | {gold_root}, merged, inv,
                pairs + [(ident, gold_root)]))
    return {'tp': len(best_pairs), 'matched_roots': [list(pair) for pair in best_pairs],
        'identifier_mapping': best_map, 'search_steps': budget.used,
        'search_step_limit': max_search_steps,
        'sharing_scope': 'consistent_across_all_matched_root_closures',
        'algorithm': 'iterative_exact_global_v2',
        'unmatchable_search_roots': len(p) - len(order),
        'complete_valid_prediction_roots': len(p), 'complete_valid_Gold_roots': len(g)}
