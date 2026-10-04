"""Prospective API native-count cluster statistics, without file/Gold/API I/O.

Adapted reviewed exact-count/PCG64 mathematics in a DISTINCT 12-family module.
No old six-family module is imported or patched. Not an independent replay,
execution/source barrier, scientific freeze, or physical-truth certificate.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import platform

import numpy as np

POLICY = 'sccomics_API_native_cluster_statistics_v1'
MODELS = ('deepseek-ai/DeepSeek-V4-Pro', 'zai-org/GLM-5.3')
ARMS = ('B', 'G', 'T', 'S')
REPEATS = (0, 1, 2)
ENDPOINTS = ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete')
CONTRASTS = tuple((f'SC_API:{model}:{endpoint}:T_minus_{other}', model, endpoint, other)
                  for model in MODELS for endpoint in ENDPOINTS for other in ('B', 'G', 'S'))
PRIMARY_IDS = tuple(row[0] for row in CONTRASTS)
HISTORICAL_IDS = ('polyie:typed_minus_mean', 'polyie:typed_minus_capacity_mean',
                  'mulms:typed_minus_mean', 'mulms:typed_minus_capacity_mean')
HISTORICAL_VALUES = (Fraction(1, 8192), Fraction(1, 8192), Fraction(1, 64), Fraction(1, 64))
HISTORICAL_SOURCE_SHA256 = 'c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0'
PYTHON_VERSION, NUMPY_VERSION = '3.13.7', '2.2.6'
EXACT_MAX_CLUSTERS = 20
MC_DRAWS, MC_SEED = 100000, 20261031
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10000, 20261032
ALPHA = Fraction(1, 20)


class APIStatisticsIntegrityError(ValueError):
    """Incomplete/invalid supplied families cannot produce a partial analysis."""


def require(ok, message):
    if not ok:
        raise APIStatisticsIntegrityError(message)


def _runtime():
    require(platform.python_version() == PYTHON_VERSION and np.__version__ == NUMPY_VERSION,
            'fixed Python3.13.7/NumPy2.2.6 stream runtime required')


def _integer(value, *, positive=False):
    require(type(value) is int and value >= (1 if positive else 0),
            'original nonnegative Python int required; bool/float/fixed-width rejected')
    return value


def _counts(value):
    require(type(value) in (tuple, list) and len(value) == 3, 'TP/FP/FN triple required')
    return tuple(_integer(v) for v in value)


def _ids(value):
    require(type(value) in (tuple, list) and len(value) > 0, 'explicit nonempty ordered source-cluster IDs required')
    require(all(type(i) is str and i and not any(c in i for c in '\r\n\t') for i in value), 'lexical source-cluster IDs')
    require(len(set(value)) == len(value), 'duplicate source-cluster ID')
    return tuple(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def rational_json(value):
    require(type(value) is Fraction, 'exact Fraction required')
    return {'numerator': value.numerator, 'denominator': value.denominator, 'decimal': float(value)}


def f1(value):
    t, p, n = _counts(value)
    denominator = 2*t+p+n
    return Fraction(2*t, denominator) if denominator else Fraction(0)


def metrics(value):
    t, p, n = _counts(value)
    return {'counts': [t, p, n], 'precision': rational_json(Fraction(t, t+p) if t+p else Fraction(0)),
            'recall': rational_json(Fraction(t, t+n) if t+n else Fraction(0)), 'f1': rational_json(f1(value))}


def _total(rows):
    rows = tuple(rows)
    return tuple(sum(v[j] for v in rows) for j in range(3))


def _difference_parts(a, b):
    ta, pa, na = a
    tb, pb, nb = b
    da, db = 2*ta+pa+na, 2*tb+pb+nb
    numerator_a, numerator_b = 2*ta, 2*tb
    if not da:
        numerator_a, da = 0, 1
    if not db:
        numerator_b, db = 0, 1
    return numerator_a*db-numerator_b*da, da*db


def difference_parts(a, b):
    return _difference_parts(_counts(a), _counts(b))


def _stream(payload):
    _runtime()
    raw = _canonical(payload)
    digest = hashlib.sha256(raw).digest()
    seed = int.from_bytes(digest, 'big')
    return np.random.Generator(np.random.PCG64(seed)), {
        'payload': payload, 'sorted_compact_json_utf8': raw.decode('utf8'),
        'payload_sha256': digest.hex(), 'digest_to_seed': '256-bit unsigned big-endian integer',
        'bit_generator': 'PCG64', 'numpy_version': np.__version__, 'seed_integer': seed}


def mask_stream(ordered_cluster_ids, contrast_id):
    ids = _ids(ordered_cluster_ids)
    require(contrast_id in PRIMARY_IDS, 'one of the exact twelve contrast IDs required')
    generator, meta = _stream({'policy': 'sccomics_API_cluster_swap_v1', 'seed': MC_SEED,
                              'ordered_cluster_ids': list(ids), 'contrast_id': contrast_id})
    masks = generator.integers(0, 2, size=(MC_DRAWS, len(ids)), dtype=np.uint8)
    meta.update({'shape': [MC_DRAWS, len(ids)], 'dtype': 'uint8',
                 'serialization': 'C-order one byte per Bernoulli bit',
                 'mask_stream_sha256': hashlib.sha256(masks.tobytes(order='C')).hexdigest(),
                 'assignments_with_replacement': True, 'independent_fair_Bernoulli_bits': True})
    return masks, meta


def bootstrap_index_stream(ordered_cluster_ids):
    ids = _ids(ordered_cluster_ids)
    generator, meta = _stream({'policy': 'sccomics_API_cluster_bootstrap_v1', 'seed': BOOTSTRAP_SEED,
                              'ordered_cluster_ids': list(ids)})
    indices = generator.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)), dtype=np.int64)
    meta.update({'shape': [BOOTSTRAP_DRAWS, len(ids)], 'sampling_with_replacement': True,
                 'serialization': 'C-order little-endian unsigned64 indices',
                 'index_stream_sha256': hashlib.sha256(indices.astype('<u8', copy=False).tobytes(order='C')).hexdigest()})
    return indices, meta


def paired_randomization(a, b, *, ordered_cluster_ids, contrast_id):
    """Swap whole retained-three-repeat clusters; inclusive exact rational tails."""
    _runtime()
    ids = _ids(ordered_cluster_ids)
    require(contrast_id in PRIMARY_IDS and type(a) in (list, tuple) and type(b) in (list, tuple) and
            len(a) == len(b) == len(ids), 'complete paired contrast population')
    a, b = tuple(map(_counts, a)), tuple(map(_counts, b))
    require(all(x[0]+x[2] == y[0]+y[2] for x, y in zip(a, b)), 'paired Gold support mismatch')
    ca, cb = _total(a), _total(b)
    observed_num, observed_den = _difference_parts(ca, cb)
    changes = [(i, tuple(y[j]-x[j] for j in range(3))) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    hits = 0
    if len(ids) <= EXACT_MAX_CLUSTERS:
        assignments = 1 << len(ids)
        digest = hashlib.sha256()
        width = (len(ids)+7)//8
        for mask in range(assignments):
            digest.update(mask.to_bytes(width, 'little'))
            change = [0, 0, 0]
            for i, delta in changes:
                if mask & (1 << i):
                    for j in range(3):
                        change[j] += delta[j]
            aa = tuple(ca[j]+change[j] for j in range(3))
            bb = tuple(cb[j]-change[j] for j in range(3))
            numerator, denominator = _difference_parts(aa, bb)
            hits += abs(numerator)*observed_den >= abs(observed_num)*denominator
        raw_numerator, raw_denominator = hits, assignments
        stream = {'branch': 'exhaustive_all_masks', 'assignments': assignments,
                  'mask_stream_sha256': digest.hexdigest(),
                  'serialization': 'ascending integer masks; packed little-endian; cluster0 is least-significant bit',
                  'includes_identity_and_complement': True, 'plus_one_added': False}
    else:
        masks, stream = mask_stream(ids, contrast_id)
        # Equal rows still occupy all their bits in the complete fixed stream.
        for mask in masks:
            change = [0, 0, 0]
            for i, delta in changes:
                if mask[i]:
                    for j in range(3):
                        change[j] += delta[j]
            aa = tuple(ca[j]+change[j] for j in range(3))
            bb = tuple(cb[j]-change[j] for j in range(3))
            numerator, denominator = _difference_parts(aa, bb)
            hits += abs(numerator)*observed_den >= abs(observed_num)*denominator
        raw_numerator, raw_denominator = hits+1, MC_DRAWS+1
        stream.update({'branch': 'fixed_conservative_Monte_Carlo', 'assignments': MC_DRAWS,
                       'plus_one_added': True, 'exact_or_unbiased_p_claim': False})
    return {'contrast_id': contrast_id, 'source_cluster_count': len(ids),
            'pooled_T_counts': list(ca), 'pooled_reference_counts': list(cb),
            'T_f1': rational_json(f1(ca)), 'reference_f1': rational_json(f1(cb)),
            'signed_T_minus_reference': rational_json(Fraction(observed_num, observed_den)),
            'absolute_statistic': rational_json(abs(Fraction(observed_num, observed_den))),
            'observed_unreduced_difference': {'numerator': observed_num, 'denominator': observed_den},
            'inclusive_exact_crossproduct_ties': True, 'hits': int(hits),
            'raw_unreduced_p_fraction': {'numerator': int(raw_numerator), 'denominator': raw_denominator},
            'raw_p': rational_json(Fraction(raw_numerator, raw_denominator)), 'stream': stream,
            'null': 'joint system-label exchangeability under whole-source-cluster swaps conditional on retained API outputs/inventory; equal F1 alone is insufficient'}


def holm_family(pvalues, *, expected_ids):
    require(type(pvalues) is dict and type(expected_ids) in (list, tuple) and len(expected_ids) > 0 and
            all(type(i) is str and i for i in expected_ids) and len(set(expected_ids)) == len(expected_ids) and
            set(pvalues) == set(expected_ids), 'complete exact closed Holm family')
    require(all(type(pvalues[name]) is Fraction and 0 <= pvalues[name] <= 1 for name in expected_ids),
            'Holm requires bounded Fraction values, not float')
    ranked = sorted(enumerate(expected_ids), key=lambda row: (pvalues[row[1]], row[0]))
    adjusted, ranks, previous = {}, {}, Fraction(0)
    for rank, (_, name) in enumerate(ranked):
        previous = max(previous, (len(expected_ids)-rank)*pvalues[name])
        adjusted[name] = min(Fraction(1), previous)
        ranks[name] = rank+1
    return {'method': 'Holm_step_down_Bonferroni', 'family_size': len(expected_ids), 'alpha': rational_json(ALPHA),
            'ordered_family_ids': list(expected_ids), 'tie_order': 'original declared family order',
            'entries': [{'id': name, 'rank': ranks[name], 'raw': rational_json(pvalues[name]),
                         'adjusted': rational_json(adjusted[name]), 'reject_at_fixed_alpha': adjusted[name] <= ALPHA}
                        for name in expected_ids]}


def validate_historical(records):
    require(type(records) is dict and set(records) == set(HISTORICAL_IDS), 'complete original historical four')
    values = {}
    for name, expected in zip(HISTORICAL_IDS, HISTORICAL_VALUES):
        row = records[name]
        require(type(row) is dict and set(row) == {'numerator', 'denominator', 'source_sha256'}, 'historical record schema')
        numerator, denominator = _integer(row['numerator']), _integer(row['denominator'], positive=True)
        require(type(row['source_sha256']) is str and row['source_sha256'] == HISTORICAL_SOURCE_SHA256 and
                (numerator, denominator) == (expected.numerator, expected.denominator), 'frozen historical raw identity/value differs')
        values[name] = Fraction(numerator, denominator)
    return values


def validate_population(clusters, expected_cluster_ids):
    """Capture only explicit primitive counts; full model/arm/repeat/endpoint coverage."""
    ids = _ids(expected_cluster_ids)
    require(type(clusters) in (list, tuple) and len(clusters) == len(ids), 'complete ordered source-cluster population')
    pooled = {model: {arm: {endpoint: [] for endpoint in ENDPOINTS} for arm in ARMS} for model in MODELS}
    repeat_totals = {model: {arm: {endpoint: [[0, 0, 0] for _ in REPEATS] for endpoint in ENDPOINTS}
                           for arm in ARMS} for model in MODELS}
    copied = []
    for cluster, expected in zip(clusters, ids):
        require(type(cluster) is dict and set(cluster) == {'source_cluster_id', 'gold_support_by_endpoint', 'models'} and
                type(cluster['source_cluster_id']) is str and cluster['source_cluster_id'] == expected,
                'closed cluster schema/exact inventory order')
        gold = cluster['gold_support_by_endpoint']
        require(type(gold) is dict and set(gold) == set(ENDPOINTS), 'both complete Gold supports')
        support = {endpoint: _integer(gold[endpoint]) for endpoint in ENDPOINTS}
        require(type(cluster['models']) is dict and set(cluster['models']) == set(MODELS), 'both exact model families required')
        record = {'source_cluster_id': expected, 'gold_support_by_endpoint': support, 'models': {}}
        for model in MODELS:
            arms = cluster['models'][model]
            require(type(arms) is dict and set(arms) == set(ARMS), 'all four arms required for each model')
            record['models'][model] = {}
            for arm in ARMS:
                endpoints = arms[arm]
                require(type(endpoints) is dict and set(endpoints) == set(ENDPOINTS), 'both endpoints in every arm')
                record['models'][model][arm] = {}
                for endpoint in ENDPOINTS:
                    repetitions = endpoints[endpoint]
                    require(type(repetitions) in (list, tuple) and len(repetitions) == 3, 'all original request repeats0/1/2')
                    vectors, kept = [], []
                    for i, (row, repeat) in enumerate(zip(repetitions, REPEATS)):
                        require(type(row) is dict and set(row) == {'repeat', 'tp', 'fp', 'fn'} and
                                type(row['repeat']) is int and row['repeat'] == repeat, 'repeat schema/order/type')
                        vector = _counts((row['tp'], row['fp'], row['fn']))
                        require(vector[0]+vector[2] == support[endpoint], 'Gold support differs across model/arm/repeat')
                        vectors.append(vector)
                        kept.append({'repeat': repeat, 'tp': vector[0], 'fp': vector[1], 'fn': vector[2]})
                        for j in range(3):
                            repeat_totals[model][arm][endpoint][i][j] += vector[j]
                    pooled[model][arm][endpoint].append(_total(vectors))
                    record['models'][model][arm][endpoint] = kept
        copied.append(record)
    return ids, pooled, repeat_totals, copied


def _paired_bootstrap(pooled, ids):
    indices, metadata = bootstrap_index_stream(ids)  # ONE common draw invocation for all sixteen outputs.
    samples = {(model, arm, endpoint): [] for model in MODELS for arm in ARMS for endpoint in ENDPOINTS}
    differences = {name: [] for name in PRIMARY_IDS}
    raw_counts = []
    for draw in indices:
        multiplicities = Counter(int(i) for i in draw)
        row, exact = {}, {}
        for model in MODELS:
            row[model] = {}
            for arm in ARMS:
                row[model][arm] = {}
                for endpoint in ENDPOINTS:
                    vector = tuple(sum(pooled[model][arm][endpoint][i][j]*frequency
                                       for i, frequency in multiplicities.items()) for j in range(3))
                    row[model][arm][endpoint] = list(vector)
                    exact[model, arm, endpoint] = f1(vector)
                    samples[model, arm, endpoint].append(float(exact[model, arm, endpoint]))
        for name, model, endpoint, reference in CONTRASTS:
            differences[name].append(float(exact[model, 'T', endpoint]-exact[model, reference, endpoint]))
        raw_counts.append(row)
    def interval(values):
        array = np.asarray(values, dtype=np.float64)
        require(np.isfinite(array).all(), 'finite float64 bootstrap statistics')
        lower, upper = np.quantile(array, [.025, .975], method='linear')
        return {'lower': float(lower), 'upper': float(upper), 'confidence_level': .95,
                'method': 'percentile_linear', 'pointwise_conditional_only': True}
    return {'draws': BOOTSTRAP_DRAWS, 'source_cluster_count': len(ids), 'all_three_repeats_retained_with_source': True,
            'common_paired_indices_all_models_arms_endpoints': True, 'stream': metadata,
            'model_arm_F1_intervals': {model: {arm: {endpoint: interval(samples[model, arm, endpoint])
                                       for endpoint in ENDPOINTS} for arm in ARMS} for model in MODELS},
            'signed_difference_intervals': {name: interval(differences[name]) for name in PRIMARY_IDS},
            'replicate_pooled_count_vectors': raw_counts,
            'replicate_count_vectors_sha256': hashlib.sha256(_canonical(raw_counts)).hexdigest(),
            'replicate_count_serialization': 'sorted compact ASCII JSON; original Python integer counts',
            'raw_exact_reconstruction': 'Each replicate F1 and signed difference reconstructs from retained integer counts as Fraction; only final linear percentile quantiles use float64',
            'zero_denominator_f1': 0, 'zero_replicates_dropped': False,
            'interpretation': 'Conditional source-cluster resampling of retained API outputs; not algorithm/random-seed/provider-drift/simultaneous/physical uncertainty'}


def analyze_population(clusters, *, expected_cluster_ids, historical_raw):
    """Complete supplied counts -> pooled-repeat micro-F1/12 tests/Holm12 + sensitivity16.

    Each closed cluster has source_cluster_id, gold_support_by_endpoint, models.
    models[exact_model][B/G/T/S][exact_endpoint] holds [{repeat,tp,fp,fn}] for 0/1/2.
    Gold support is the per-original-source/endpoint target count in ONE request.
    Caller independently freezes100 original source clusters/test (or12 pilot),
    full graphs and first-Gold gates; K2/K21 artificial inputs do not certify that
    actual cohort. Repeats/models/objects never multiply the inferential units.
    """
    _runtime()
    ids, pooled, repeat_totals, copied = validate_population(clusters, expected_cluster_ids)
    historical = validate_historical(historical_raw)
    comparisons, raw_values = [], {}
    for name, model, endpoint, reference in CONTRASTS:
        row = paired_randomization(pooled[model]['T'][endpoint], pooled[model][reference][endpoint],
                                   ordered_cluster_ids=ids, contrast_id=name)
        row.update({'model': model, 'endpoint': endpoint, 'reference_arm': reference})
        comparisons.append(row)
        raw = row['raw_unreduced_p_fraction']
        raw_values[name] = Fraction(raw['numerator'], raw['denominator'])
    primary = holm_family(raw_values, expected_ids=PRIMARY_IDS)
    sensitivity = holm_family({**raw_values, **historical}, expected_ids=PRIMARY_IDS+HISTORICAL_IDS)
    sensitivity.update({'historically_informed_sensitivity_only': True, 'retroactive_confirmatory_control_claim': False,
                        'historical_source_sha256': HISTORICAL_SOURCE_SHA256,
                        'historical_rerandomization_or_old_family_replacement': False,
                        'historical_exact_raw_records': {name: {**rational_json(value), 'source_sha256': HISTORICAL_SOURCE_SHA256}
                                                         for name, value in historical.items()}})
    bootstrap = _paired_bootstrap(pooled, ids)
    return {'status': 'complete_statistics_for_explicit_supplied_API_counts_only', 'policy': POLICY,
            'ordered_cluster_ids': list(ids), 'ordered_cluster_inventory_sha256': hashlib.sha256(_canonical(list(ids))).hexdigest(),
            'source_cluster_count': len(ids), 'models': list(MODELS), 'arms': list(ARMS), 'request_repeats': list(REPEATS),
            'endpoints': list(ENDPOINTS), 'inferential_N_not_multiplied_by_repeats_or_models': True,
            'input_count_objects': copied, 'input_count_objects_sha256': hashlib.sha256(_canonical(copied)).hexdigest(),
            'pooled_count_vectors_by_source_cluster': pooled,
            'pooled_summaries': {model: {arm: {endpoint: metrics(_total(pooled[model][arm][endpoint]))
                                for endpoint in ENDPOINTS} for arm in ARMS} for model in MODELS},
            'individual_request_repeat_summaries': {model: {arm: {endpoint: [dict(repeat=repeat, **metrics(vector))
                for repeat, vector in zip(REPEATS, repeat_totals[model][arm][endpoint])]
                for endpoint in ENDPOINTS} for arm in ARMS} for model in MODELS},
            'primary_comparisons': comparisons, 'holm12_primary': primary,
            'holm16_historically_informed_sensitivity': sensitivity, 'paired_conditional_bootstrap': bootstrap,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
            'source_Gold_graph_population_execution_or_semantic_barriers_certified': False,
            'independent_replay_claim': False,
            'limitations': ['Caller must establish frozen actual source clusters, complete graphs and original native scoring before first Gold',
                'Three-repeat pooled counts are not an ensemble or mean-repeat F1; request repetitions are not independent original sources',
                'The two API families are not independent source samples or a proof of independent pretraining',
                'Whole-cluster swap null requires joint label exchangeability, not merely equal F1 or a causal intervention',
                'Pointwise bootstrap conditions on retained outputs/inventory; does not cover algorithm randomness, provider drift or annotation truth',
                'Historical16 is informed sensitivity; no retroactive confirmatory claim or old Holm4/family change',
                'No outcome-based reseeding, draw-budget, family, branch, alpha or source exclusion is supported']}
