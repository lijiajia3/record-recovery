"""Pure fixed-protocol SC cluster statistics, without corpus/model/file I/O.

Only explicit integer count objects enter this module. Source/fit/Gold-read
barriers and native semantic scoring belong to the separately reviewed caller.
These functions do not certify those barriers or physical/source independence.
All constants implement configuration SHA742fb701..., not outcome-driven rules.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import platform

import numpy as np

CONFIGURATION_SHA256 = "742fb7011f2162cb935bd6de3068e38a8c0f3a67c58df8ff266dc73a8814103c"
PYTHON_VERSION = "3.13.7"
NUMPY_VERSION = "2.2.6"
SEEDS = (20261013, 20261014, 20261015)
SYSTEMS = ("aligned", "permuted", "ordered_context", "biaffine")
ENDPOINTS = ("R_trigger_projected_complete", "E_global_sharing_consistent_complete")
CONTRASTS = tuple((f"SC_{short}_aligned_minus_{other}", endpoint, other)
                  for short, endpoint in zip(("R", "E"), ENDPOINTS)
                  for other in SYSTEMS[1:])
PRIMARY_IDS = tuple(c[0] for c in CONTRASTS)
HISTORICAL_IDS = ("polyie:typed_minus_mean", "polyie:typed_minus_capacity_mean",
                  "mulms:typed_minus_mean", "mulms:typed_minus_capacity_mean")
HISTORICAL_VALUES = (Fraction(1, 8192), Fraction(1, 8192), Fraction(1, 64), Fraction(1, 64))
HISTORICAL_SOURCE_SHA256 = "c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0"
ALPHA = Fraction(1, 20)
EXACT_MAX_CLUSTERS = 20
MC_DRAWS = 100000
MC_SEED = 20261031
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261032


class StatisticsIntegrityError(ValueError):
    """No complete inferential family is valid for an incomplete/invalid input."""


def require(ok, message):
    if not ok:
        raise StatisticsIntegrityError(message)


def _require_runtime():
    require(platform.python_version() == PYTHON_VERSION, f"fixed Python runtime required: {PYTHON_VERSION}")
    require(np.__version__ == NUMPY_VERSION, f"fixed NumPy runtime required: {NUMPY_VERSION}")


def integer(value, *, positive=False, label="integer"):
    # A fixed-width NumPy scalar may already have overflowed. Require the
    # caller's original arbitrary-precision Python integer, not a silent cast.
    require(type(value) is int and value >= (1 if positive else 0),
            f"{label}: nonnegative Python int required; bool/float/fixed-width rejected")
    return value


def count_vector(value):
    require(type(value) in (tuple, list) and len(value) == 3, "TP/FP/FN triple required")
    return tuple(integer(x, label="TP/FP/FN") for x in value)


def cluster_ids(ids):
    require(type(ids) in (tuple, list) and len(ids) > 0, "nonempty explicit ordered cluster inventory required")
    require(all(type(x) is str and x and not any(c in x for c in "\r\n\t") for x in ids),
            "cluster IDs must be nonempty lexical strings")
    require(len(set(ids)) == len(ids), "duplicate cluster IDs")
    return tuple(ids)


def rational_json(value):
    require(type(value) is Fraction, "exact Fraction required")
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}


def f1(value):
    t, p, n = count_vector(value)
    d = 2 * t + p + n
    return Fraction(2 * t, d) if d else Fraction(0, 1)


def _difference_parts(a, b):
    # Unreduced exact rational representation; its cross-product comparison
    # avoids floats and fixed-width arithmetic even at extreme count sizes.
    ta, pa, na = a
    tb, pb, nb = b
    da, db = 2 * ta + pa + na, 2 * tb + pb + nb
    na_, nb_ = 2 * ta, 2 * tb
    if not da:
        na_, da = 0, 1
    if not db:
        nb_, db = 0, 1
    return na_ * db - nb_ * da, da * db


def difference_parts(a, b):
    """Validated standalone exact difference; inference uses validated internals."""
    return _difference_parts(count_vector(a), count_vector(b))


def stream(payload):
    _require_runtime()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    seed_int = int.from_bytes(digest, "big")
    return np.random.Generator(np.random.PCG64(seed_int)), {
        "payload": payload, "sorted_compact_json_utf8": raw.decode("utf-8"),
        "payload_sha256": digest.hex(), "digest_to_seed": "256-bit unsigned big-endian integer",
        "bit_generator": "PCG64", "numpy_version": np.__version__, "seed_integer": seed_int}


def mask_stream(ids, contrast_id):
    ids = cluster_ids(ids)
    require(contrast_id in PRIMARY_IDS, "undeclared contrast ID")
    rng, metadata = stream({"policy": "sccomics_cluster_swap_v1", "seed": MC_SEED,
                            "ordered_cluster_ids": list(ids), "contrast_id": contrast_id})
    # One fixed invocation, no chunk-dependent RNG buffering policy.
    masks = rng.integers(0, 2, size=(MC_DRAWS, len(ids)), dtype=np.uint8)
    metadata.update({"shape": [MC_DRAWS, len(ids)], "dtype": "uint8", "serialization": "C-order one byte per Bernoulli bit",
                     "mask_stream_sha256": hashlib.sha256(masks.tobytes(order="C")).hexdigest(),
                     "assignments_with_replacement": True, "independent_fair_Bernoulli_bits": True})
    return masks, metadata


def bootstrap_index_stream(ids):
    ids = cluster_ids(ids)
    rng, metadata = stream({"policy": "sccomics_cluster_bootstrap_v1", "seed": BOOTSTRAP_SEED,
                            "ordered_cluster_ids": list(ids)})
    indices = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)), dtype=np.int64)
    metadata.update({"shape": [BOOTSTRAP_DRAWS, len(ids)], "sampling_with_replacement": True,
                     "serialization": "C-order little-endian unsigned64 indices",
                     "index_stream_sha256": hashlib.sha256(indices.astype("<u8", copy=False).tobytes(order="C")).hexdigest()})
    return indices, metadata


def paired_randomization(a, b, *, ordered_cluster_ids, contrast_id):
    _require_runtime()
    ids = cluster_ids(ordered_cluster_ids)
    require(contrast_id in PRIMARY_IDS, "undeclared contrast ID")
    require(type(a) in (list, tuple) and type(b) in (list, tuple) and len(a) == len(b) == len(ids),
            "paired count population/inventory mismatch")
    a, b = tuple(map(count_vector, a)), tuple(map(count_vector, b))
    require(all(x[0] + x[2] == y[0] + y[2] for x, y in zip(a, b)), "paired Gold support mismatch")
    ca = tuple(sum(x[j] for x in a) for j in range(3))
    cb = tuple(sum(x[j] for x in b) for j in range(3))
    observed_num, observed_den = _difference_parts(ca, cb)
    # Sparse delta calculation changes no sampling, population, masks or hits.
    # Every K cluster remains a Bernoulli bit, including all-zero/tied clusters.
    changes = [(i, tuple(y[j] - x[j] for j in range(3))) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    hits = 0
    if len(ids) <= EXACT_MAX_CLUSTERS:
        assignments = 1 << len(ids)
        digest = hashlib.sha256()
        width = (len(ids) + 7) // 8
        for mask in range(assignments):
            digest.update(mask.to_bytes(width, "little"))
            change = [0, 0, 0]
            for i, delta in changes:
                if mask & (1 << i):
                    for j in range(3):
                        change[j] += delta[j]
            aa, bb = tuple(ca[j] + change[j] for j in range(3)), tuple(cb[j] - change[j] for j in range(3))
            num, den = _difference_parts(aa, bb)
            hits += abs(num) * observed_den >= abs(observed_num) * den
        raw_numerator, raw_denominator = hits, assignments
        randomization = {"branch": "exhaustive_all_masks", "assignments": assignments,
                         "mask_stream_sha256": digest.hexdigest(), "serialization": "ascending integer masks; packed little-endian; cluster0 is least-significant bit",
                         "includes_identity_and_complement": True, "plus_one_added": False}
    else:
        masks, randomization = mask_stream(ids, contrast_id)
        for mask in masks:
            change = [0, 0, 0]
            for i, delta in changes:
                if mask[i]:
                    for j in range(3):
                        change[j] += delta[j]
            aa, bb = tuple(ca[j] + change[j] for j in range(3)), tuple(cb[j] - change[j] for j in range(3))
            num, den = _difference_parts(aa, bb)
            hits += abs(num) * observed_den >= abs(observed_num) * den
        raw_numerator, raw_denominator = hits + 1, MC_DRAWS + 1
        randomization.update({"branch": "fixed_conservative_Monte_Carlo", "assignments": MC_DRAWS,
                              "plus_one_added": True, "exact_or_unbiased_p_claim": False})
    p = Fraction(raw_numerator, raw_denominator)
    return {"contrast_id": contrast_id, "source_cluster_count": len(ids),
            "pooled_aligned_counts": list(ca), "pooled_reference_counts": list(cb),
            "aligned_f1": rational_json(f1(ca)), "reference_f1": rational_json(f1(cb)),
            "signed_aligned_minus_reference": rational_json(Fraction(observed_num, observed_den)),
            "absolute_statistic": rational_json(abs(Fraction(observed_num, observed_den))),
            "observed_unreduced_difference": {"numerator": observed_num, "denominator": observed_den},
            "inclusive_exact_crossproduct_ties": True, "hits": int(hits), "raw_p": rational_json(p),
            "raw_unreduced_p_fraction": {"numerator": int(raw_numerator), "denominator": raw_denominator},
            "stream": randomization, "null": "joint fitted-system-label invariance under every whole-cluster swap, conditional on retained fits/source inventory; equal F1 alone is insufficient"}


def holm_family(pvalues, *, expected_ids):
    require(type(pvalues) is dict and type(expected_ids) in (list, tuple) and len(expected_ids) > 0,
            "explicit complete Holm family required")
    require(all(type(name) is str and name for name in expected_ids), "Holm family IDs must be explicit strings")
    require(len(set(expected_ids)) == len(expected_ids) and set(pvalues) == set(expected_ids),
            "incomplete/extra/duplicated Holm family")
    require(all(type(pvalues[i]) is Fraction and Fraction(0) <= pvalues[i] <= Fraction(1) for i in expected_ids),
            "Holm requires exact bounded Fraction values, not floats")
    rank = sorted(enumerate(expected_ids), key=lambda t: (pvalues[t[1]], t[0]))
    previous, adjusted, ranks = Fraction(0), {}, {}
    m = len(expected_ids)
    for offset, (_, name) in enumerate(rank):
        previous = max(previous, (m - offset) * pvalues[name])
        adjusted[name] = min(Fraction(1), previous)
        ranks[name] = offset + 1
    return {"method": "Holm_step_down_Bonferroni", "family_size": m, "alpha": rational_json(ALPHA),
            "ordered_family_ids": list(expected_ids), "tie_order": "original declared family order",
            "entries": [{"id": name, "rank": ranks[name], "raw": rational_json(pvalues[name]),
                         "adjusted": rational_json(adjusted[name]), "reject_at_fixed_alpha": adjusted[name] <= ALPHA}
                        for name in expected_ids]}


def validate_historical(records):
    require(type(records) is dict and set(records) == set(HISTORICAL_IDS), "complete exact historical four required")
    values = {}
    for name, expected in zip(HISTORICAL_IDS, HISTORICAL_VALUES):
        r = records[name]
        require(type(r) is dict and set(r) == {"numerator", "denominator", "source_sha256"}, "historical raw record schema")
        n, d = integer(r["numerator"], label="historical numerator"), integer(r["denominator"], positive=True, label="historical denominator")
        require(type(r["source_sha256"]) is str and r["source_sha256"] == HISTORICAL_SOURCE_SHA256
                and (n, d) == (expected.numerator, expected.denominator),
                "historical raw identity/value differs from frozen original")
        values[name] = Fraction(n, d)
    return values


def validate_population(clusters, expected_cluster_ids):
    ids = cluster_ids(expected_cluster_ids)
    require(type(clusters) in (list, tuple) and len(clusters) == len(ids), "complete cluster population required")
    pooled = {a: {e: [] for e in ENDPOINTS} for a in SYSTEMS}
    source_records = []
    seed_totals = {a: {e: [[0, 0, 0] for _ in SEEDS] for e in ENDPOINTS} for a in SYSTEMS}
    for c, expected_id in zip(clusters, ids):
        require(type(c) is dict and set(c) == {"cluster_id", "gold_support_by_endpoint", "systems"}, "cluster schema")
        require(c["cluster_id"] == expected_id, "cluster order/ID mismatch")
        gold = c["gold_support_by_endpoint"]
        require(type(gold) is dict and set(gold) == set(ENDPOINTS), "complete endpoint Gold support required")
        gold = {e: integer(gold[e], label="Gold support") for e in ENDPOINTS}
        require(type(c["systems"]) is dict and set(c["systems"]) == set(SYSTEMS), "complete four-system population required")
        copied = {"cluster_id": expected_id, "gold_support_by_endpoint": gold, "systems": {}}
        for a in SYSTEMS:
            require(type(c["systems"][a]) is dict and set(c["systems"][a]) == set(ENDPOINTS), "complete two-endpoint population required")
            copied["systems"][a] = {}
            for e in ENDPOINTS:
                seeds = c["systems"][a][e]
                require(type(seeds) in (list, tuple) and len(seeds) == len(SEEDS), "all three fixed seeds required")
                vectors, copied_seeds = [], []
                for i, (r, expected_seed) in enumerate(zip(seeds, SEEDS)):
                    require(type(r) is dict and set(r) == {"seed", "tp", "fp", "fn"}, "seed-count record schema")
                    require(type(r["seed"]) is int and r["seed"] == expected_seed, "fixed seed identity/order mismatch")
                    v = count_vector((r["tp"], r["fp"], r["fn"]))
                    require(v[0] + v[2] == gold[e], "paired/seed Gold support differs")
                    vectors.append(v)
                    copied_seeds.append({"seed": expected_seed, "tp": v[0], "fp": v[1], "fn": v[2]})
                    for j in range(3):
                        seed_totals[a][e][i][j] += v[j]
                pooled[a][e].append(tuple(sum(v[j] for v in vectors) for j in range(3)))
                copied["systems"][a][e] = copied_seeds
        source_records.append(copied)
    return ids, pooled, seed_totals, source_records


def _paired_bootstrap(pooled, ids):
    # Called once for the complete four-system/two-endpoint population.
    indices, metadata = bootstrap_index_stream(ids)
    values = {(a, e): [] for a in SYSTEMS for e in ENDPOINTS}
    differences = {name: [] for name in PRIMARY_IDS}
    raw_counts = []
    for draw in indices:
        multiplicities = Counter(int(i) for i in draw)
        row = {a: {} for a in SYSTEMS}
        exact_values = {}
        for a in SYSTEMS:
            for e in ENDPOINTS:
                v = tuple(sum(pooled[a][e][i][j] * multiplicity for i, multiplicity in multiplicities.items()) for j in range(3))
                row[a][e] = list(v)
                exact_values[a, e] = f1(v)
                values[a, e].append(float(exact_values[a, e]))
        for name, e, other in CONTRASTS:
            differences[name].append(float(exact_values["aligned", e] - exact_values[other, e]))
        raw_counts.append(row)
    def interval(data):
        array = np.asarray(data, dtype=np.float64)
        require(np.isfinite(array).all(), "nonfinite bootstrap statistic")
        lo, hi = np.quantile(array, [0.025, 0.975], method="linear")
        return {"lower": float(lo), "upper": float(hi), "confidence_level": 0.95, "method": "percentile_linear",
                "marginal_conditional_only": True}
    return {"draws": BOOTSTRAP_DRAWS, "source_cluster_count": len(ids), "all_seeds_retained_with_cluster": True,
            "common_paired_indices_all_systems_and_endpoints": True, "stream": metadata,
            "system_intervals": {a: {e: interval(values[a, e]) for e in ENDPOINTS} for a in SYSTEMS},
            "signed_difference_intervals": {name: interval(differences[name]) for name in PRIMARY_IDS},
            "replicate_pooled_count_vectors": raw_counts,
            "raw_exact_reconstruction": "Every replicate F1 and signed difference reconstructs as exact Fraction from the retained integer TP/FP/FN vectors; only final linear quantiles use float64",
            "zero_denominator_f1": 0, "zero_replicates_dropped": False,
            "interpretation": "Conditional empirical source-cluster resampling with retained fits; population95% coverage needs unverified independent/representative sampling; no algorithm-level, simultaneous or physical guarantee"}


def analyze_population(clusters, *, expected_cluster_ids, historical_raw):
    """Complete fixed six-family/paired-bootstrap and separate Holm10.

    clusters: ordered list of {cluster_id,gold_support_by_endpoint,systems}.
    systems has all four names; each has both fixed endpoint names, each mapped
    to ordered [{seed,tp,fp,fn}] for all three fixed seeds. Gold support is the
    number of native Gold roots for that cluster/endpoint in one fit. This
    function requires caller-frozen source IDs and historical exact records.
    It does not read files or decide annotation/model-source completeness.
    """
    _require_runtime()
    ids, pooled, seed_totals, source_records = validate_population(clusters, expected_cluster_ids)
    historical = validate_historical(historical_raw)
    comparisons, pvalues = [], {}
    for name, endpoint, other in CONTRASTS:
        result = paired_randomization(pooled["aligned"][endpoint], pooled[other][endpoint],
                                      ordered_cluster_ids=ids, contrast_id=name)
        comparisons.append(result)
        raw = result["raw_unreduced_p_fraction"]
        pvalues[name] = Fraction(raw["numerator"], raw["denominator"])
    holm6 = holm_family(pvalues, expected_ids=PRIMARY_IDS)
    sensitivity = holm_family({**pvalues, **historical}, expected_ids=PRIMARY_IDS + HISTORICAL_IDS)
    sensitivity.update({"historically_informed_sensitivity_only": True, "retroactive_confirmatory_control_claim": False,
                        "historical_source_sha256": HISTORICAL_SOURCE_SHA256,
                        "historical_rerandomization_or_old_family_replacement": False,
                        "historical_exact_raw_records": {name: {**rational_json(value), "source_sha256": HISTORICAL_SOURCE_SHA256}
                                                         for name, value in historical.items()}})
    bootstrap = _paired_bootstrap(pooled, ids)
    return {"status": "complete_statistics_for_explicit_supplied_counts_only", "configuration_sha256": CONFIGURATION_SHA256,
            "ordered_cluster_ids": list(ids), "source_cluster_count": len(ids), "seeds": list(SEEDS),
            "independent_document_N_not_multiplied_by_seeds": True,
            "input_count_objects": source_records,
            "individual_seed_summaries": {a: {e: [{"seed": s, "counts": v, "f1": rational_json(f1(v))}
                for s, v in zip(SEEDS, seed_totals[a][e])] for e in ENDPOINTS} for a in SYSTEMS},
            "primary_comparisons": comparisons, "holm6_primary": holm6, "holm10_historically_informed_sensitivity": sensitivity,
            "paired_conditional_bootstrap": bootstrap, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "limitations": ["Caller must establish complete native scoring and prediction/Gold-read barriers independently",
                "Pooled three-fit counts are neither an ensemble nor mean-seed F1",
                "Source-text clusters are not certified independent physical experiments or publications",
                "Cluster swap null requires joint system-label invariance conditional on retained fits, not merely equal F1",
                "Bootstrap conditions on retained fits and does not integrate training/search/pretraining/annotation uncertainty",
                "No favorable-outcome reseeding, branch, draw-budget, alpha or family selection is supported"]}
