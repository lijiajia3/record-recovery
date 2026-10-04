"""Complete-grid native development choices, without any filesystem access.

Only the declared primary R/E metrics are searched during development. Full-R
and root-local event metrics are descriptive test/replay work, not extra
selection criteria. Every failed primary setting must stop the complete family.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json

from sccomics_primary_matching_v2 import exact_graph_match

SEEDS = (20261013, 20261014, 20261015)
DEV_IDS = tuple(range(101, 201))
SPAN_THRESHOLDS = (.05, .1, .25, .5, .75, .9, .95, .99)
PAIR_THRESHOLDS = (.5, .75, .9, .95, .99)
ENDPOINTS = ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(tp, predicted, gold):
    require(all(type(x) is int and x >= 0 for x in (tp, predicted, gold)) and
            tp <= min(predicted, gold), 'invalid native count')
    return {'tp': tp, 'fp': predicted - tp, 'fn': gold - tp}


def fraction_f1(row):
    require(set(row) == {'tp', 'fp', 'fn'} and
            all(type(v) is int and v >= 0 for v in row.values()), 'strict integer count row required')
    denominator = 2 * row['tp'] + row['fp'] + row['fn']
    return Fraction(2 * row['tp'], denominator) if denominator else Fraction(0)


def pooled(rows):
    rows = tuple(rows)
    for row in rows:
        fraction_f1(row)
    return {key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')}


def text_counts(pred, gold):
    require(gold.gold and not pred.gold and pred.source_bytes == gold.source_bytes and
            pred.registry == gold.registry and all(r.valid for r in gold.records),
            'complete matching native documents required')
    p = Counter(r.mention_key() for r in pred.family('T') if r.valid)
    g = Counter(r.mention_key() for r in gold.family('T'))
    return counts(sum((p & g).values()), len(pred.family('T')), len(gold.family('T')))


def _endpoint(doc, identifier):
    record = doc.by_id[identifier]
    if record.family == 'E':
        record = doc.by_id[record.trigger]
    return record.mention_key()


def relation_counts(pred, gold):
    require(gold.gold and not pred.gold and pred.source_bytes == gold.source_bytes and
            pred.registry == gold.registry and all(r.valid for r in gold.records),
            'same-registry intact matching native documents required')
    def population(doc):
        return Counter((r.label, _endpoint(doc, r.arg1), _endpoint(doc, r.arg2))
            for r in doc.family('R') if r.valid)
    p, g = population(pred), population(gold)
    return counts(sum((p & g).values()), len(pred.family('R')), len(gold.family('R')))


def event_fingerprint(doc):
    """Cache only identical resolved T/E bytes/validity, not a guessed shortcut.

    Event rooted closures never traverse R records. Nevertheless parser-derived
    validity/issues of every T/E are included, so R-dependent identifier damage
    cannot accidentally turn into a cache hit. Complete source identity remains.
    """
    payload = {'source_sha256': hashlib.sha256(doc.source_bytes).hexdigest(),
        'T_E_records': [{'occurrence': r.occurrence, 'identifier': r.identifier,
            'family': r.family, 'label': r.label, 'raw_hex': r.raw.hex(),
            'valid': r.valid, 'issues': list(r.issues)}
            for r in doc.records if r.family in {'T', 'E'}]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def primary_counts(pred, gold, *, event_cache=None):
    require(gold.gold and not pred.gold and pred.source_bytes == gold.source_bytes and
            pred.registry == gold.registry and all(r.valid for r in gold.records), 'intact full native Gold required')
    key = event_fingerprint(pred), event_fingerprint(gold)
    if event_cache is not None and key in event_cache:
        match = event_cache[key]
        reused = True
    else:
        match = exact_graph_match(pred, gold, 'E', max_search_steps=1_000_000)
        reused = False
        if event_cache is not None:
            event_cache[key] = match
    return {ENDPOINTS[0]: relation_counts(pred, gold),
        ENDPOINTS[1]: counts(match['tp'], len(pred.family('E')), len(gold.family('E'))),
        'E_matching': match, 'E_exact_fingerprint_result_reused': reused,
        'invalid_predictions': [{'family': r.family, 'occurrence': r.occurrence,
            'issues': list(r.issues)} for r in pred.records if not r.valid]}


def _grid_vectors(records, expected_keys, endpoints):
    """Require each complete seed/doc setting and fixed native denominators."""
    require(isinstance(records, list) and len(records) == len(expected_keys),
            'incomplete development setting population')
    lookup, supports = {}, {}
    for setting in records:
        require(set(setting) == {'epoch', 'thresholds', 'by_seed_document'}, 'setting fields differ')
        require(type(setting['epoch']) is int and 1 <= setting['epoch'] <= 10,
                'undeclared development epoch')
        key = setting['epoch'], tuple(setting['thresholds'])
        require(key in expected_keys and key not in lookup, 'duplicate/undeclared development setting')
        rows = setting['by_seed_document']
        require(set(rows) == {str(s) for s in SEEDS}, 'all three declared fits required')
        for seed in SEEDS:
            require(set(rows[str(seed)]) == {str(i) for i in DEV_IDS}, 'all 100 development sources required')
            for source_id in DEV_IDS:
                row = rows[str(seed)][str(source_id)]
                require(set(row) == set(endpoints), 'complete declared endpoint rows required')
                for endpoint in endpoints:
                    fraction_f1(row[endpoint])
                    support = row[endpoint]['tp'] + row[endpoint]['fn']
                    duty = source_id, endpoint
                    require(duty not in supports or supports[duty] == support,
                            'native development denominator changed across fits/settings')
                    supports[duty] = support
        lookup[key] = setting
    require(set(lookup) == expected_keys, 'complete fixed development grid required')
    return lookup


def select_detector(records):
    keys = {(epoch, (threshold,)) for epoch in range(1, 11) for threshold in SPAN_THRESHOLDS}
    lookup = _grid_vectors(records, keys, ('T_complete_native',))
    candidates = []
    for key in sorted(lookup):
        setting = lookup[key]
        total = pooled([setting['by_seed_document'][str(seed)][str(i)]['T_complete_native']
                        for seed in SEEDS for i in DEV_IDS])
        score = fraction_f1(total)
        candidates.append({'epoch': key[0], 'threshold': key[1][0], 'pooled': total,
            'pooled_f1_fraction': [score.numerator, score.denominator]})
    chosen = max(candidates, key=lambda c: (Fraction(*c['pooled_f1_fraction']),
        -c['pooled']['fp'], c['threshold'], -c['epoch']))
    return {'chosen': chosen, 'all_pooled_candidates': candidates, 'seed_ids': list(SEEDS),
        'development_source_ids': list(DEV_IDS), 'rule': 'pooled native T F1; fewer FP; higher threshold; earlier epoch',
        'seed_selection_or_ensemble': False, 'test_Gold_consulted': False}


def select_pair_head(records):
    keys = {(epoch, (rt, et)) for epoch in range(1, 11)
            for rt in PAIR_THRESHOLDS for et in PAIR_THRESHOLDS}
    lookup = _grid_vectors(records, keys, ENDPOINTS)
    candidates = []
    for key in sorted(lookup):
        setting = lookup[key]
        totals = {endpoint: pooled([setting['by_seed_document'][str(seed)][str(i)][endpoint]
            for seed in SEEDS for i in DEV_IDS]) for endpoint in ENDPOINTS}
        objective = sum((fraction_f1(totals[e]) for e in ENDPOINTS), Fraction(0)) / 2
        candidates.append({'epoch': key[0], 'relation_threshold': key[1][0],
            'role_threshold': key[1][1], 'pooled': totals,
            'objective_fraction': [objective.numerator, objective.denominator]})
    chosen = max(candidates, key=lambda c: (Fraction(*c['objective_fraction']),
        -sum(c['pooled'][e]['fp'] for e in ENDPOINTS), c['relation_threshold'],
        c['role_threshold'], -c['epoch']))
    return {'chosen': chosen, 'all_pooled_candidates': candidates, 'seed_ids': list(SEEDS),
        'development_source_ids': list(DEV_IDS),
        'rule': 'mean of pooled R and global-E rational F1; fewer R+E FP; higher R threshold; higher role threshold; earlier epoch',
        'seed_selection_or_ensemble': False, 'test_Gold_consulted': False}
