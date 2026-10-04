"""Independent arithmetic and procedure-identity replay; supplied counts only.

No production statistics, model, parser or analysis module is imported. The
fixed NumPy PCG64 stream is deliberately shared by specification. Aggregation,
Fraction tails, Holm correction and linear percentiles are implemented here.
The caller must independently establish real input and annotation chronology.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
import platform

import numpy as np

SEEDS = (20261013, 20261014, 20261015)
SYSTEMS = ("aligned", "permuted", "ordered_context", "biaffine")
ENDPOINTS = ("R_trigger_projected_complete", "E_global_sharing_consistent_complete")
CONTRASTS = tuple((f"SC_{short}_aligned_minus_{other}", endpoint, other)
                  for short, endpoint in zip(("R", "E"), ENDPOINTS)
                  for other in SYSTEMS[1:])
IDS = tuple(c[0] for c in CONTRASTS)
OLD_IDS = ("polyie:typed_minus_mean", "polyie:typed_minus_capacity_mean",
           "mulms:typed_minus_mean", "mulms:typed_minus_capacity_mean")
OLD_P = (Fraction(1, 8192), Fraction(1, 8192), Fraction(1, 64), Fraction(1, 64))
OLD_SHA = "c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0"
MC = 100000
BOOT = 10000
INTERVAL_ABSOLUTE_TOLERANCE = 1e-14


class ReplayIntegrityError(ValueError):
    """No complete replay certificate may be returned."""


def need(condition, message):
    if not condition:
        raise ReplayIntegrityError(message)


def integer(value):
    need(type(value) is int and value >= 0, "Original nonnegative Python int required")
    return value


def f1(v):
    need(type(v) in (list, tuple) and len(v) == 3, "Count triple required")
    t, p, n = map(integer, v)
    d = 2*t+p+n
    return Fraction(2*t, d) if d else Fraction(0)


def total(rows):
    rows = tuple(rows)
    return tuple(sum(row[j] for row in rows) for j in range(3))


def fraction_record(value):
    need(type(value) is dict and type(value.get("numerator")) is int and
         type(value.get("denominator")) is int and value["denominator"] > 0,
         "Explicit exact rational required")
    return Fraction(value["numerator"], value["denominator"])


def identical(a, b):
    """Exact JSON-semantic types, including bool versus integer."""
    if type(a) is not type(b): return False
    if type(a) is dict:
        return set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def payload_generator(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(raw).digest(), "big")))


def payload_metadata(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return {"payload": payload, "sorted_compact_json_utf8": raw.decode("utf-8"),
            "payload_sha256": digest.hex(), "digest_to_seed": "256-bit unsigned big-endian integer",
            "bit_generator": "PCG64", "numpy_version": "2.2.6", "seed_integer": int.from_bytes(digest, "big")}


def contrast_stream_metadata(ordered_ids, name, digest):
    k = len(ordered_ids)
    if k <= 20:
        return {"branch": "exhaustive_all_masks", "assignments": 1 << k,
                "mask_stream_sha256": digest,
                "serialization": "ascending integer masks; packed little-endian; cluster0 is least-significant bit",
                "includes_identity_and_complement": True, "plus_one_added": False}
    result = payload_metadata({"policy": "sccomics_cluster_swap_v1", "seed": 20261031,
                "ordered_cluster_ids": list(ordered_ids), "contrast_id": name})
    result.update({"shape": [MC, k], "dtype": "uint8", "serialization": "C-order one byte per Bernoulli bit",
                   "mask_stream_sha256": digest, "assignments_with_replacement": True,
                   "independent_fair_Bernoulli_bits": True, "branch": "fixed_conservative_Monte_Carlo",
                   "assignments": MC, "plus_one_added": True, "exact_or_unbiased_p_claim": False})
    return result


def validate(clusters, ordered_ids, historical):
    need(type(ordered_ids) in (tuple, list) and len(ordered_ids) > 0 and
         all(type(i) is str and i and not any(c in i for c in "\r\n\t") for i in ordered_ids) and
         len(set(ordered_ids)) == len(ordered_ids), "Complete ordered cluster identities required")
    need(type(clusters) in (tuple, list) and len(clusters) == len(ordered_ids), "Complete cluster population required")
    captured = copy.deepcopy(list(clusters))
    by_system = {(a, e): [] for a in SYSTEMS for e in ENDPOINTS}
    seed_totals = {(a, e, s): [0, 0, 0] for a in SYSTEMS for e in ENDPOINTS for s in SEEDS}
    for name, c in zip(ordered_ids, captured):
        need(type(c) is dict and set(c) == {"cluster_id", "gold_support_by_endpoint", "systems"} and
             c["cluster_id"] == name, "Cluster schema or order mismatch")
        gold = c["gold_support_by_endpoint"]
        need(type(gold) is dict and set(gold) == set(ENDPOINTS), "Complete two-endpoint support required")
        for e in ENDPOINTS: integer(gold[e])
        need(type(c["systems"]) is dict and set(c["systems"]) == set(SYSTEMS), "All four systems required")
        for a in SYSTEMS:
            v = c["systems"][a]
            need(type(v) is dict and set(v) == set(ENDPOINTS), "Both native endpoints required")
            for e in ENDPOINTS:
                fits = v[e]
                need(type(fits) in (tuple, list) and len(fits) == 3, "All three retained fits required")
                triples = []
                for r, s in zip(fits, SEEDS):
                    need(type(r) is dict and set(r) == {"seed", "tp", "fp", "fn"} and
                         type(r["seed"]) is int and r["seed"] == s, "Fixed seed schema/order mismatch")
                    triple = tuple(integer(r[k]) for k in ("tp", "fp", "fn"))
                    need(triple[0]+triple[2] == gold[e], "Gold support differs across fitted systems")
                    triples.append(triple)
                    for j in range(3): seed_totals[a, e, s][j] += triple[j]
                by_system[a, e].append(total(triples))
                v[e] = [{"seed": s, "tp": t[0], "fp": t[1], "fn": t[2]}
                        for s, t in zip(SEEDS, triples)]
    need(type(historical) is dict and set(historical) == set(OLD_IDS), "Exact four historical records required")
    old = {}
    for name, expected in zip(OLD_IDS, OLD_P):
        r = historical[name]
        need(type(r) is dict and set(r) == {"numerator", "denominator", "source_sha256"} and
             type(r["numerator"]) is int and type(r["denominator"]) is int and
             (r["numerator"], r["denominator"]) == (expected.numerator, expected.denominator) and
             type(r["source_sha256"]) is str and r["source_sha256"] == OLD_SHA,
             "Historical value or source identity differs")
        old[name] = expected
    return captured, by_system, seed_totals, old


def randomization(a, b, ordered_ids, name):
    """Rebuild swapped systems directly, rather than production delta sums."""
    need(type(ordered_ids) in (list, tuple) and len(ordered_ids) > 0 and
         all(type(i) is str and i and not any(c in i for c in "\r\n\t") for i in ordered_ids) and
         len(set(ordered_ids)) == len(ordered_ids), "Distinct ordered cluster identities required")
    need(name in IDS and len(a) == len(b) == len(ordered_ids) and len(a) > 0, "Paired inventory mismatch")
    for x, y in zip(a, b):
        f1(x); f1(y)
        need(x[0]+x[2] == y[0]+y[2], "Paired Gold support mismatch")
    left, right = total(a), total(b)
    observed = abs(f1(left)-f1(right))
    # Equal rows contribute equally in every assignment; retain their bits in
    # the whole fixed stream while adding their original counts once per side.
    fixed = total(x for x, y in zip(a, b) if x == y)
    changing = tuple(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    digest = hashlib.sha256()
    if len(ordered_ids) <= 20:
        assignments = 1 << len(ordered_ids)
        width = (len(ordered_ids)+7)//8
        masks = range(assignments)
        def bits(mask):
            digest.update(mask.to_bytes(width, "little"))
            return tuple(bool(mask & (1 << i)) for i in changing)
        raw_denominator, add = assignments, 0
    else:
        masks = payload_generator({"policy": "sccomics_cluster_swap_v1", "seed": 20261031,
                    "ordered_cluster_ids": list(ordered_ids), "contrast_id": name}).integers(
                    0, 2, size=(MC, len(ordered_ids)), dtype=np.uint8)
        digest.update(masks.tobytes(order="C"))
        def bits(mask): return tuple(bool(mask[i]) for i in changing)
        assignments, raw_denominator, add = MC, MC+1, 1
    hits = 0
    for mask in masks:
        swap = bits(mask)
        x = total(itertools.chain((fixed,), (b[i] if s else a[i] for i, s in zip(changing, swap))))
        y = total(itertools.chain((fixed,), (a[i] if s else b[i] for i, s in zip(changing, swap))))
        hits += abs(f1(x)-f1(y)) >= observed
    return {"id": name, "pooled_aligned": list(left), "pooled_reference": list(right),
            "signed": f1(left)-f1(right), "hits": hits, "assignments": assignments,
            "raw_numerator": hits+add, "raw_denominator": raw_denominator,
            "stream_sha256": digest.hexdigest()}


def holm(values, names):
    need(type(values) is dict and set(values) == set(names) and len(set(names)) == len(names) and
         all(type(values[n]) is Fraction and 0 <= values[n] <= 1 for n in names), "Complete exact Holm family required")
    order = sorted(range(len(names)), key=lambda i: (values[names[i]], i))
    result = {}
    for rank, i in enumerate(order):
        # Independent closed maximum over the preceding ordered Bonferroni
        # terms; no production running-value implementation is called.
        adjusted = min(Fraction(1), max((len(names)-j)*values[names[order[j]]] for j in range(rank+1)))
        result[names[i]] = {"rank": rank+1, "adjusted": adjusted, "reject": adjusted <= Fraction(1, 20)}
    return result


def linear_percentile(values, q):
    need(type(values) is list and len(values) > 0 and all(type(x) is float and math.isfinite(x) for x in values),
         "Complete finite percentile population required")
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    i = math.floor(position); j = min(i+1, len(ordered)-1)
    return ordered[i]+(position-i)*(ordered[j]-ordered[i])


def replay_statistics(clusters, *, expected_cluster_ids, historical_raw, primary_statistics):
    """Verify complete supplied statistics; no actual-file or execution claim."""
    need(platform.python_version() == "3.13.7" and np.__version__ == "2.2.6", "Fixed Python/NumPy stream runtime required")
    ids = tuple(expected_cluster_ids)
    captured, base, seed_totals, old = validate(clusters, expected_cluster_ids, historical_raw)
    primary = copy.deepcopy(primary_statistics)
    need(type(primary) is dict and primary.get("status") == "complete_statistics_for_explicit_supplied_counts_only" and
         identical(primary.get("ordered_cluster_ids"), list(ids)) and identical(primary.get("source_cluster_count"), len(ids)) and
         identical(primary.get("seeds"), list(SEEDS)) and identical(primary.get("input_count_objects"), captured) and
         primary.get("configuration_sha256") == "742fb7011f2162cb935bd6de3068e38a8c0f3a67c58df8ff266dc73a8814103c" and
         primary.get("independent_document_N_not_multiplied_by_seeds") is True and
         primary.get("runtime") == {"python": "3.13.7", "numpy": "2.2.6"},
         "Primary supplied-count population identity differs")
    need(type(primary.get("primary_comparisons")) is list and len(primary["primary_comparisons"]) == 6 and
         [r.get("contrast_id") for r in primary["primary_comparisons"]] == list(IDS), "Complete declared six contrasts required")
    contrasts, pvalues = [], {}
    for original, (name, e, other) in zip(primary["primary_comparisons"], CONTRASTS):
        row = randomization(base["aligned", e], base[other, e], ids, name)
        need(identical(original["pooled_aligned_counts"], row["pooled_aligned"]) and
             identical(original["pooled_reference_counts"], row["pooled_reference"]) and
             fraction_record(original["signed_aligned_minus_reference"]) == row["signed"] and
             fraction_record(original["aligned_f1"]) == f1(row["pooled_aligned"]) and
             fraction_record(original["reference_f1"]) == f1(row["pooled_reference"]) and
             fraction_record(original["absolute_statistic"]) == abs(row["signed"]) and
             type(original["hits"]) is int and original["hits"] == row["hits"] and
             identical(original["raw_unreduced_p_fraction"], {"numerator": row["raw_numerator"], "denominator": row["raw_denominator"]}) and
             identical(original["source_cluster_count"], len(ids)) and
             original["inclusive_exact_crossproduct_ties"] is True and
             fraction_record(original["observed_unreduced_difference"]) ==
                 Fraction(row["signed"]) and
             identical(original["stream"], contrast_stream_metadata(ids, name, row["stream_sha256"])),
             "Independent rational randomization or fixed procedure identity mismatch: "+name)
        p = Fraction(row["raw_numerator"], row["raw_denominator"])
        need(fraction_record(original["raw_p"]) == p, "Reported raw p differs")
        pvalues[name] = p
        row["signed"] = {"numerator": row["signed"].numerator, "denominator": row["signed"].denominator}
        contrasts.append(row)
    for field, values, names in (("holm6_primary", pvalues, IDS),
                                ("holm10_historically_informed_sensitivity", {**pvalues, **old}, IDS+OLD_IDS)):
        family = primary[field]; recomputed = holm(values, names)
        need(identical(family["family_size"], len(names)) and identical(family["ordered_family_ids"], list(names)) and
             family["method"] == "Holm_step_down_Bonferroni" and family["tie_order"] == "original declared family order" and
             fraction_record(family["alpha"]) == Fraction(1, 20) and len(family["entries"]) == len(names) and
             [r["id"] for r in family["entries"]] == list(names), "Holm family/alpha identity differs")
        for row in family["entries"]:
            n = row["id"]; want = recomputed[n]
            need(type(row["rank"]) is int and row["rank"] == want["rank"] and
                 fraction_record(row["raw"]) == values[n] and fraction_record(row["adjusted"]) == want["adjusted"] and
                 type(row["reject_at_fixed_alpha"]) is bool and row["reject_at_fixed_alpha"] == want["reject"], "Independent Holm mismatch")
    sensitivity = primary["holm10_historically_informed_sensitivity"]
    need(sensitivity["historically_informed_sensitivity_only"] is True and
         sensitivity["retroactive_confirmatory_control_claim"] is False and
         sensitivity["historical_rerandomization_or_old_family_replacement"] is False and
         sensitivity["historical_source_sha256"] == OLD_SHA and
         set(sensitivity["historical_exact_raw_records"]) == set(OLD_IDS), "Historical exposure/sensitivity scope differs")
    for name, value in old.items():
        record = sensitivity["historical_exact_raw_records"][name]
        need(fraction_record(record) == value and record["source_sha256"] == OLD_SHA, "Recorded historical source/raw p differs")
    draws = payload_generator({"policy": "sccomics_cluster_bootstrap_v1", "seed": 20261032,
                    "ordered_cluster_ids": list(ids)}).integers(0, len(ids), size=(BOOT, len(ids)), dtype=np.int64)
    stream_sha = hashlib.sha256(draws.astype("<u8").tobytes(order="C")).hexdigest()
    expected_bootstrap_stream = payload_metadata({"policy": "sccomics_cluster_bootstrap_v1", "seed": 20261032,
                "ordered_cluster_ids": list(ids)})
    expected_bootstrap_stream.update({"shape": [BOOT, len(ids)], "sampling_with_replacement": True,
                "serialization": "C-order little-endian unsigned64 indices", "index_stream_sha256": stream_sha})
    boot = primary["paired_conditional_bootstrap"]
    need(identical(boot["draws"], BOOT) and identical(boot["source_cluster_count"], len(ids)) and
         identical(boot["stream"], expected_bootstrap_stream) and
         type(boot["replicate_pooled_count_vectors"]) is list and len(boot["replicate_pooled_count_vectors"]) == BOOT and
         boot["zero_replicates_dropped"] is False and boot["all_seeds_retained_with_cluster"] is True and
         boot["common_paired_indices_all_systems_and_endpoints"] is True and
         type(boot["zero_denominator_f1"]) is int and boot["zero_denominator_f1"] == 0,
         "Complete common bootstrap stream required")
    samples = {key: [] for key in base}; differences = {name: [] for name in IDS}
    for index, draw in enumerate(draws):
        row = {a: {} for a in SYSTEMS}; ratios = {}
        for a, e in base:
            v = total(base[a, e][int(i)] for i in draw)
            row[a][e] = list(v); ratios[a, e] = f1(v); samples[a, e].append(float(ratios[a, e]))
        need(identical(row, boot["replicate_pooled_count_vectors"][index]), "Raw integer bootstrap replicate differs")
        for name, e, other in CONTRASTS: differences[name].append(float(ratios["aligned", e]-ratios[other, e]))
    errors = []
    for name, values in itertools.chain(samples.items(), differences.items()):
        interval = boot["system_intervals"][name[0]][name[1]] if type(name) is tuple else boot["signed_difference_intervals"][name]
        need(type(interval["confidence_level"]) is float and interval["confidence_level"] == .95 and
             interval["method"] == "percentile_linear" and interval["marginal_conditional_only"] is True,
             "Marginal conditional linear-percentile interval scope differs")
        for k, q in (("lower", .025), ("upper", .975)):
            need(type(interval[k]) is float and math.isfinite(interval[k]), "Finite float interval bound required")
            error = abs(interval[k]-linear_percentile(values, q)); errors.append(error)
            need(error <= INTERVAL_ABSOLUTE_TOLERANCE, "Independent linear percentile differs")
    for a, e, s in seed_totals:
        rows = primary["individual_seed_summaries"][a][e]
        need(type(rows) is list and len(rows) == 3, "Complete individual fitted-seed summaries required")
        row = rows[SEEDS.index(s)]
        need(type(row["seed"]) is int and row["seed"] == s and identical(row["counts"], seed_totals[a, e, s]) and
             fraction_record(row["f1"]) == f1(row["counts"]), "Individual seed counts or rational score differs")
    return {"status": "independent_arithmetic_replay_passed_for_supplied_counts_only", "source_cluster_count": len(ids),
            "ordered_cluster_ids": list(ids), "independent_statistical_arithmetic": True,
            "production_statistics_module_imported": False, "fixed_RNG_library_and_stream_shared_by_design": True,
            "actual_source_or_execution_barriers_certified": False, "independent_physical_truth_claim": False,
            "contrasts": contrasts, "raw_integer_bootstrap_replicates_verified": BOOT,
            "bootstrap_index_stream_sha256": stream_sha, "holm6_and_historical10_recomputed": True,
            "bootstrap_intervals_verified": 14, "maximum_absolute_quantile_error": max(errors),
            "fixed_procedure_population_branch_seed_payload_shapes_type_strictly_verified": True,
            "prospective_quantile_absolute_tolerance": INTERVAL_ABSOLUTE_TOLERANCE,
            "rational_tails_raw_counts_and_Holm_tolerance": 0,
            "limitations": ["Same fixed PCG64 streams are replayed, not independent random samples",
                            "Conditional fitted-system/source-cluster inference retains original assumptions",
                            "Pure supplied-object checks do not authorize actual annotations, fitting or source independence"]}
