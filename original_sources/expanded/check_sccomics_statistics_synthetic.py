"""Implementer self-checks using invented counts only; no corpus input option.

This helper intentionally has no annotation, prediction, model, source-data or
old-statistics loader. It does not constitute independent source review. Run
with Python3.13.7 / NumPy2.2.6 and a fresh empty artificial output directory.
"""
from __future__ import annotations

import argparse
import builtins
import copy
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

import sccomics_statistics as st


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def exact(obj):
    return Fraction(obj["numerator"], obj["denominator"])


def oracle_f1(v):
    numerator = 2 * v[0]
    denominator = numerator + v[1] + v[2]
    return Fraction(numerator, denominator) if denominator != 0 else Fraction(0, 1)


def oracle_sum(vectors):
    return tuple(sum(row[j] for row in vectors) for j in range(3))


def oracle_permutation(a, b):
    observed = oracle_f1(oracle_sum(a)) - oracle_f1(oracle_sum(b))
    values = []
    for bits in itertools.product((0, 1), repeat=len(a)):
        left = oracle_sum([b[i] if bit else a[i] for i, bit in enumerate(bits)])
        right = oracle_sum([a[i] if bit else b[i] for i, bit in enumerate(bits)])
        values.append(oracle_f1(left) - oracle_f1(right))
    hits = sum(abs(x) >= abs(observed) for x in values)
    return observed, hits, Fraction(hits, len(values)), values


def oracle_holm(values, names):
    ranked = sorted(names, key=lambda name: (values[name], names.index(name)))
    result = {}
    for i, name in enumerate(ranked):
        terms = [Fraction(len(names) - j) * values[ranked[j]] for j in range(i + 1)]
        result[name] = min(Fraction(1), max(terms))
    return result


def oracle_linear_quantile(values, q):
    ordered = sorted(values)
    location = Fraction(len(ordered) - 1) * q
    floor = location.numerator // location.denominator
    fraction = float(location - floor)
    if floor == len(ordered) - 1:
        return ordered[floor]
    return ordered[floor] * (1.0 - fraction) + ordered[floor + 1] * fraction


def oracle_rng(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(raw).digest(), "big"))), sha(raw)


def historical():
    return {name: {"numerator": 1, "denominator": denominator,
                   "source_sha256": "c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0"}
            for name, denominator in zip(st.HISTORICAL_IDS, (8192, 8192, 64, 64))}


def rich_population():
    # Third source is truly empty for every system/seed, and is retained.
    # First two sources include over/under-prediction and unequal seed F1s.
    templates = {
        "aligned": [[(3, 2, 0), (0, 0, 3), (3, 0, 0)], [(1, 5, 1), (2, 0, 0), (2, 2, 0)]],
        "permuted": [[(0, 4, 3), (1, 0, 2), (2, 0, 1)], [(2, 0, 0), (1, 2, 1), (0, 1, 2)]],
        "ordered_context": [[(3, 0, 0), (3, 0, 0), (3, 0, 0)], [(2, 0, 0), (2, 0, 0), (2, 0, 0)]],
        "biaffine": [[(1, 0, 2), (1, 2, 2), (1, 1, 2)], [(1, 0, 1), (1, 0, 1), (1, 0, 1)]],
    }
    clusters = []
    for i, name in enumerate(("INVENTED_rich_A", "INVENTED_rich_B", "INVENTED_empty_C")):
        support_r = (3, 2, 0)[i]
        support_e = (1, 1, 0)[i]
        c = {"cluster_id": name, "gold_support_by_endpoint": dict(zip(st.ENDPOINTS, (support_r, support_e))), "systems": {}}
        for system_index, system in enumerate(st.SYSTEMS):
            c["systems"][system] = {}
            r_vectors = templates[system][i] if i < 2 else [(0, 0, 0)] * 3
            # Event values are independent invented counts; no relation→event
            # derivation, physical endpoint or schema evidence is claimed.
            e_vectors = [(int((i + system_index + j) % 3 != 0), (system_index + j) % 2,
                          int((i + system_index + j) % 3 == 0)) for j in range(3)] if i < 2 else [(0, 0, 0)] * 3
            for endpoint, vectors in zip(st.ENDPOINTS, (r_vectors, e_vectors)):
                c["systems"][system][endpoint] = [dict(zip(("seed", "tp", "fp", "fn"), (seed, *v)))
                                                 for seed, v in zip(st.SEEDS, vectors)]
        clusters.append(c)
    return clusters


def empty_population(n):
    return [{"cluster_id": f"INVENTED_empty_{i:03d}", "gold_support_by_endpoint": {e: 0 for e in st.ENDPOINTS},
             "systems": {a: {e: [{"seed": s, "tp": 0, "fp": 0, "fn": 0} for s in st.SEEDS]
                             for e in st.ENDPOINTS} for a in st.SYSTEMS}} for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("fresh empty artificial output directory required; historical outputs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (output, *output.parents)):
        parser.error("artificial artifact output path must not use symlinks")
    checks = []
    module_path = Path(st.__file__).resolve()
    helper_path = Path(__file__).resolve()
    module_before, helper_before = module_path.read_bytes(), helper_path.read_bytes()
    (output / "statistics_before.py").write_bytes(module_before)
    (output / "helper_before.py").write_bytes(helper_before)
    rich, empty100 = rich_population(), empty_population(100)
    rich_ids, empty_ids = [c["cluster_id"] for c in rich], [c["cluster_id"] for c in empty100]
    inputs = {"identity": "INVENTED_IMPLEMENTER_FIXTURES_ONLY", "rich_three_cluster_population": rich,
              "empty100_population": empty100, "historical_exact_values_copied_from_authorized_configuration": historical(),
              "historical_actual_result_file_read": False}
    write_json(output / "invented_input_counts.json", inputs)
    results = {}

    def check(name, function):
        try:
            detail = function()
            checks.append({"id": name, "passed": True, "detail": detail})
        except Exception as error:
            checks.append({"id": name, "passed": False, "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()})

    def rejected(function):
        try:
            function()
        except st.StatisticsIntegrityError as error:
            return {"rejected": True, "message": str(error)}
        raise AssertionError("invalid object accepted; expected StatisticsIntegrityError")

    def rich_run():
        original_open = builtins.open
        attempts = []
        def forbidden_open(*args, **kwargs):
            attempts.append(str(args[0]) if args else "missing path")
            raise AssertionError("pure statistics attempted filesystem access")
        builtins.open = forbidden_open
        try:
            result = st.analyze_population(rich, expected_cluster_ids=rich_ids, historical_raw=historical())
        finally:
            builtins.open = original_open
        assert not attempts
        results["rich"] = result
        write_json(output / "rich_three_cluster_statistics.json", result)
        return {"filesystem_open_attempts": attempts, "K": result["source_cluster_count"],
                "bootstrap_draws": result["paired_conditional_bootstrap"]["draws"]}
    check("rich_complete_pure_execution", rich_run)

    def rich_oracles():
        result = results["rich"]
        vectors = {a: {e: [tuple(sum(r[k] for r in c["systems"][a][e]) for k in ("tp", "fp", "fn"))
                              for c in rich] for e in st.ENDPOINTS} for a in st.SYSTEMS}
        pvalues = {}
        for name, endpoint, reference in st.CONTRASTS:
            observed, hits, raw, values = oracle_permutation(vectors["aligned"][endpoint], vectors[reference][endpoint])
            actual = next(r for r in result["primary_comparisons"] if r["contrast_id"] == name)
            assert exact(actual["signed_aligned_minus_reference"]) == observed
            assert actual["hits"] == hits and exact(actual["raw_p"]) == raw
            assert actual["stream"]["assignments"] == 8
            assert all(values.count(x) == values.count(-x) for x in values)
            pvalues[name] = raw
        assert exact(result["primary_comparisons"][1]["signed_aligned_minus_reference"]) < 0
        for field, names, values in (("holm6_primary", list(st.PRIMARY_IDS), pvalues),
                                     ("holm10_historically_informed_sensitivity", list(st.PRIMARY_IDS + st.HISTORICAL_IDS),
                                      {**pvalues, **dict(zip(st.HISTORICAL_IDS, (Fraction(1,8192), Fraction(1,8192), Fraction(1,64), Fraction(1,64))))})):
            expected = oracle_holm(values, names)
            assert result[field]["family_size"] == len(names)
            for item in result[field]["entries"]:
                assert exact(item["adjusted"]) == expected[item["id"]]
                assert item["reject_at_fixed_alpha"] == (expected[item["id"]] <= Fraction(1,20))
        pooled_f1 = oracle_f1(oracle_sum(vectors["aligned"][st.ENDPOINTS[0]]))
        mean_seed = sum(exact(r["f1"]) for r in result["individual_seed_summaries"]["aligned"][st.ENDPOINTS[0]]) / 3
        assert pooled_f1 != mean_seed
        for a in st.SYSTEMS:
            for e in st.ENDPOINTS:
                for i, seed in enumerate(st.SEEDS):
                    wanted = tuple(sum(c["systems"][a][e][i][key] for c in rich) for key in ("tp", "fp", "fn"))
                    actual = result["individual_seed_summaries"][a][e][i]
                    assert actual["counts"] == list(wanted) and actual["seed"] == seed
                    assert exact(actual["f1"]) == oracle_f1(wanted)
        return {"enumerations": "six independent Fraction/brute-mask enumerations", "negative_signed_effect_retained": True,
                "pooled_f1_differs_from_mean_seed_f1": True, "holm6_and_separate_holm10_exactly_checked": True}
    check("rich_exact_and_holm_oracles", rich_oracles)

    def bootstrap_oracle():
        result = results["rich"]["paired_conditional_bootstrap"]
        rng, payload_sha = oracle_rng({"policy":"sccomics_cluster_bootstrap_v1", "seed":20261032, "ordered_cluster_ids":rich_ids})
        indices = rng.integers(0, 3, size=(10000, 3), dtype=np.int64)
        assert result["stream"]["payload_sha256"] == payload_sha
        assert result["stream"]["index_stream_sha256"] == sha(indices.astype("<u8", copy=False).tobytes(order="C"))
        assert len(result["replicate_pooled_count_vectors"]) == 10000
        variables = {(a,e): [] for a in st.SYSTEMS for e in st.ENDPOINTS}
        differences = {name: [] for name in st.PRIMARY_IDS}
        for draw, actual in zip(indices, result["replicate_pooled_count_vectors"]):
            fvalues = {}
            for a in st.SYSTEMS:
                for e in st.ENDPOINTS:
                    wanted = tuple(sum(rich[int(i)]["systems"][a][e][j][key] for i in draw for j in range(3))
                                   for key in ("tp", "fp", "fn"))
                    assert actual[a][e] == list(wanted)
                    fvalues[a,e] = oracle_f1(wanted)
                    variables[a,e].append(float(fvalues[a,e]))
            for name, e, other in st.CONTRASTS:
                differences[name].append(float(fvalues["aligned",e] - fvalues[other,e]))
        for (a,e), values in variables.items():
            interval = result["system_intervals"][a][e]
            assert abs(interval["lower"] - oracle_linear_quantile(values, Fraction(1,40))) <= 1e-14
            assert abs(interval["upper"] - oracle_linear_quantile(values, Fraction(39,40))) <= 1e-14
        for name, values in differences.items():
            interval = result["signed_difference_intervals"][name]
            assert abs(interval["lower"] - oracle_linear_quantile(values, Fraction(1,40))) <= 1e-14
            assert abs(interval["upper"] - oracle_linear_quantile(values, Fraction(39,40))) <= 1e-14
        write_json(output / "rich_common_bootstrap_indices.json", {"identity":"INVENTED_ONLY", "indices":indices.tolist(),
                                                                  "payload_sha256":payload_sha})
        return {"all_10000_paired_replicates_checked": True, "eight_system_and_six_difference_linear_intervals_checked":True,
                "float_tolerance_for_display_quantile_only":1e-14, "tail_probability_tolerance":0}
    check("common_bootstrap_all_replicate_and_quantile_oracle", bootstrap_oracle)

    def exact_boundary():
        ids = [f"INVENTED_exact20_{i}" for i in range(20)]
        result = st.paired_randomization([(0,0,0)]*20, [(0,0,0)]*20, ordered_cluster_ids=ids, contrast_id=st.PRIMARY_IDS[0])
        assert result["hits"] == 1048576 and result["stream"]["assignments"] == 1048576
        assert exact(result["raw_p"]) == 1 and result["stream"]["plus_one_added"] is False
        digest = hashlib.sha256()
        for mask in range(1048576):
            digest.update(mask.to_bytes(3,"little"))
        assert result["stream"]["mask_stream_sha256"] == digest.hexdigest()
        write_json(output / "exact20_boundary.json", result)
        return {"all_1048576_masks_enumerated":True,"inclusive_all_zero_ties":True,"hash":digest.hexdigest()}
    check("K20_all_masks_inclusive_zero_ties", exact_boundary)

    def mc_boundary():
        ids = [f"INVENTED_mc21_{i}" for i in range(21)]
        a, b = [(4,0,0),(1,7,2)] + [(0,0,0)]*19, [(1,2,3),(3,0,0)] + [(0,0,0)]*19
        result = st.paired_randomization(a,b,ordered_cluster_ids=ids,contrast_id=st.PRIMARY_IDS[0])
        rng, payload_sha = oracle_rng({"policy":"sccomics_cluster_swap_v1", "seed":20261031,
                                     "ordered_cluster_ids":ids,"contrast_id":st.PRIMARY_IDS[0]})
        masks = rng.integers(0,2,size=(100000,21),dtype=np.uint8)
        observed = oracle_f1(oracle_sum(a)) - oracle_f1(oracle_sum(b))
        categories = {}
        for bits in itertools.product((0,1),repeat=2):
            left = oracle_sum([b[i] if bits[i] else a[i] for i in range(2)])
            right = oracle_sum([a[i] if bits[i] else b[i] for i in range(2)])
            categories[bits] = abs(oracle_f1(left)-oracle_f1(right)) >= abs(observed)
        wanted_hits = sum(categories[int(row[0]),int(row[1])] for row in masks)
        assert result["hits"] == wanted_hits
        assert result["raw_unreduced_p_fraction"] == {"numerator":wanted_hits+1,"denominator":100001}
        assert result["stream"]["payload_sha256"] == payload_sha
        assert result["stream"]["mask_stream_sha256"] == sha(masks.tobytes(order="C"))
        assert result["stream"]["shape"] == [100000,21]
        write_json(output / "mc21_boundary.json", result)
        return {"full_100000x21_stream_checked":True,"independent_four_category_tail_oracle":True,
                "with_replacement":True,"all_zero_clusters_remain_bits":19}
    check("K21_Monte_Carlo_boundary_full_stream_and_tail", mc_boundary)

    def mc_floor():
        ids = [f"INVENTED_extreme100_{i}" for i in range(100)]
        result = st.paired_randomization([(1,0,0)]*100,[(0,0,1)]*100,ordered_cluster_ids=ids,contrast_id=st.PRIMARY_IDS[0])
        rng, payload_sha = oracle_rng({"policy":"sccomics_cluster_swap_v1", "seed":20261031,
                                     "ordered_cluster_ids":ids,"contrast_id":st.PRIMARY_IDS[0]})
        masks = rng.integers(0,2,size=(100000,100),dtype=np.uint8)
        # Here |D|=1 occurs only at all-zero or all-one masks; all other
        # assignments have |D|<1. This is an invented mathematical endpoint.
        hits = int(np.count_nonzero(np.all(masks==0,axis=1) | np.all(masks==1,axis=1)))
        assert result["hits"] == hits
        assert result["raw_unreduced_p_fraction"] == {"numerator":hits+1,"denominator":100001}
        assert exact(result["raw_p"]) >= Fraction(1,100001)
        assert result["stream"]["payload_sha256"] == payload_sha
        assert result["stream"]["mask_stream_sha256"] == sha(masks.tobytes(order="C"))
        write_json(output / "mc100_extreme_floor.json", result)
        return {"fixed_100000x100_stream":True,"nonzero_plus_one_floor":True,"hits":hits,
                "identity":"INVENTED_not_actual_system_performance"}
    check("fixed_Monte_Carlo_nonzero_resolution_floor", mc_floor)

    def hundred_run():
        result = st.analyze_population(empty100, expected_cluster_ids=empty_ids, historical_raw=historical())
        assert result["source_cluster_count"] == 100 and len(result["input_count_objects"]) == 100
        assert result["seeds"] == [20261013,20261014,20261015]
        assert len(result["primary_comparisons"]) == 6
        for comparison in result["primary_comparisons"]:
            assert comparison["hits"] == 100000 and exact(comparison["raw_p"]) == 1
            assert comparison["stream"]["shape"] == [100000,100]
            assert exact(comparison["signed_aligned_minus_reference"]) == 0
        bootstrap = result["paired_conditional_bootstrap"]
        assert bootstrap["stream"]["shape"] == [10000,100]
        assert len(bootstrap["replicate_pooled_count_vectors"]) == 10000
        assert all(all(v==[0,0,0] for eps in r.values() for v in eps.values()) for r in bootstrap["replicate_pooled_count_vectors"])
        for intervals in bootstrap["system_intervals"].values():
            assert all(item["lower"] == item["upper"] == 0 for item in intervals.values())
        assert all(item["lower"] == item["upper"] == 0 for item in bootstrap["signed_difference_intervals"].values())
        write_json(output / "empty100_complete_statistics.json", result)
        return {"all100_clusters_kept":True,"all_three_seeds_kept":True,"N_not300":True,
                "six_fixed_100000_masks_and_one_common10000_bootstrap":True,"zero_denominator_F1_zero":True}
    check("INVENTED100_shape_complete_six_and_common_bootstrap", hundred_run)

    def big_integer():
        big = 2**100 + 123456789
        a, b = [(big,big+1,1),(1,0,big)], [(1,big,big),(big,1,1)]
        result = st.paired_randomization(a,b,ordered_cluster_ids=["INVENTED_big_A","INVENTED_big_B"],contrast_id=st.PRIMARY_IDS[0])
        effect,hits,raw,_ = oracle_permutation(a,b)
        assert exact(result["signed_aligned_minus_reference"]) == effect
        assert result["hits"] == hits and exact(result["raw_p"]) == raw
        assert result["pooled_aligned_counts"] == list(oracle_sum(a))
        assert st.f1((big,big,0)) == Fraction(2,3)
        assert st.difference_parts((big,big,0),(big,0,big)) == (0,9*big*big)
        write_json(output / "arbitrary_precision_bigint.json", result)
        return {"counts_exceed_uint64":True,"exact_Fraction_and_crossproduct_checked":True}
    check("Python_bigints_no_fixed_width_overflow", big_integer)

    def tiny_exact_tail():
        big = 2**100
        a, b = [(big,0,0)]*2, [(big-1,0,1)]*2
        result = st.paired_randomization(a,b,ordered_cluster_ids=["INVENTED_tiny_A","INVENTED_tiny_B"],contrast_id=st.PRIMARY_IDS[0])
        effect,hits,raw,_ = oracle_permutation(a,b)
        assert effect > 0 and float(oracle_f1(oracle_sum(a))) - float(oracle_f1(oracle_sum(b))) == 0.0
        assert hits == 2 and raw == Fraction(1,2)
        assert exact(result["signed_aligned_minus_reference"]) == effect and result["hits"] == 2
        assert exact(result["raw_p"]) == Fraction(1,2)
        write_json(output / "tiny_exact_tail_float_cancellation_counterexample.json",result)
        return {"float_subtracts_to_zero_but_exact_effect_positive":True,
                "exact_hits2of4_not_float_tie4of4":True,"no_tail_tolerance_used":True}
    check("extreme_bigint_exact_tail_survives_float_cancellation",tiny_exact_tail)

    for label, bad in (("bool",True),("float",1.0),("negative",-1),("nan",float("nan")),
                       ("infinite",float("inf")),("numpy_int64",np.int64(1)),("numpy_uint64",np.uint64(2**64-1)),
                       ("string","1")):
        check(f"reject_count_{label}", lambda bad=bad: rejected(lambda: st.f1((bad,0,0))))
        check(f"reject_difference_count_{label}", lambda bad=bad: rejected(lambda: st.difference_parts((0,0,0),(bad,0,0))))
    check("reject_count_vector_missing",lambda:rejected(lambda:st.f1((0,0))))
    check("reject_count_vector_extra",lambda:rejected(lambda:st.f1((0,0,0,0))))
    check("reject_paired_Gold_support_mismatch",lambda:rejected(lambda:st.paired_randomization([(1,0,0)],[(0,0,0)],ordered_cluster_ids=["INVENTED"],contrast_id=st.PRIMARY_IDS[0])))
    check("reject_zero_clusters",lambda:rejected(lambda:st.paired_randomization([],[],ordered_cluster_ids=[],contrast_id=st.PRIMARY_IDS[0])))
    check("reject_duplicate_clusters",lambda:rejected(lambda:st.paired_randomization([(0,0,0)]*2,[(0,0,0)]*2,ordered_cluster_ids=["INVENTED","INVENTED"],contrast_id=st.PRIMARY_IDS[0])))
    check("reject_undeclared_contrast",lambda:rejected(lambda:st.paired_randomization([(0,0,0)],[(0,0,0)],ordered_cluster_ids=["INVENTED"],contrast_id="OTHER")))
    check("reject_paired_population_length",lambda:rejected(lambda:st.paired_randomization([(0,0,0)],[],ordered_cluster_ids=["INVENTED"],contrast_id=st.PRIMARY_IDS[0])))

    def population_rejection(mutator):
        bad = copy.deepcopy(rich)
        mutator(bad)
        return rejected(lambda:st.validate_population(bad,rich_ids))
    mutations = {
        "missing_cluster":lambda c:c.pop(),
        "reordered_cluster":lambda c:c.reverse(),
        "extra_system":lambda c:c[0]["systems"].update({"OTHER":{}}),
        "missing_system":lambda c:c[0]["systems"].pop("biaffine"),
        "missing_endpoint":lambda c:c[0]["systems"]["aligned"].pop(st.ENDPOINTS[1]),
        "missing_seed":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]].pop(),
        "duplicated_seed":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][1].update({"seed":20261013}),
        "seed_bool":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"seed":True}),
        "seed_float":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"seed":20261013.0}),
        "count_bool":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"tp":True}),
        "negative_count":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"fp":-1}),
        "float_count":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"fp":0.0}),
        "Gold_support_change":lambda c:c[0]["gold_support_by_endpoint"].update({st.ENDPOINTS[0]:4}),
        "extra_Gold_endpoint":lambda c:c[0]["gold_support_by_endpoint"].update({"OTHER":0}),
        "cluster_extra_key":lambda c:c[0].update({"OTHER":0}),
        "seed_extra_key":lambda c:c[0]["systems"]["aligned"][st.ENDPOINTS[0]][0].update({"OTHER":0}),
    }
    for name, mutation in mutations.items():
        check(f"reject_population_{name}",lambda mutation=mutation:population_rejection(mutation))
    good_p = {name:Fraction(1,100) for name in st.PRIMARY_IDS}
    check("reject_Holm_missing",lambda:rejected(lambda:st.holm_family({k:v for k,v in good_p.items() if k!=st.PRIMARY_IDS[0]},expected_ids=st.PRIMARY_IDS)))
    check("reject_Holm_extra",lambda:rejected(lambda:st.holm_family({**good_p,"OTHER":Fraction(1)},expected_ids=st.PRIMARY_IDS)))
    check("reject_Holm_duplicate_declaration",lambda:rejected(lambda:st.holm_family(good_p,expected_ids=st.PRIMARY_IDS+(st.PRIMARY_IDS[0],))))
    for label, bad in (("float",.01),("bool",True),("negative",Fraction(-1)),("above1",Fraction(2)),("nan",float("nan"))):
        check(f"reject_Holm_{label}",lambda bad=bad:rejected(lambda:st.holm_family({**good_p,st.PRIMARY_IDS[0]:bad},expected_ids=st.PRIMARY_IDS)))
    check("reject_Holm_nonstring_ID",lambda:rejected(lambda:st.holm_family({1:Fraction(1)},expected_ids=(1,))))

    def historical_rejection(mutator):
        bad = historical()
        mutator(bad)
        return rejected(lambda:st.validate_historical(bad))
    historical_mutations = {
        "missing":lambda r:r.pop(st.HISTORICAL_IDS[0]),
        "extra":lambda r:r.update({"OTHER":r[st.HISTORICAL_IDS[0]]}),
        "wrong_source":lambda r:r[st.HISTORICAL_IDS[0]].update({"source_sha256":"0"*64}),
        "changed_value":lambda r:r[st.HISTORICAL_IDS[0]].update({"numerator":2}),
        "zero_denominator":lambda r:r[st.HISTORICAL_IDS[0]].update({"denominator":0}),
        "bool_numerator":lambda r:r[st.HISTORICAL_IDS[0]].update({"numerator":True}),
        "float_denominator":lambda r:r[st.HISTORICAL_IDS[0]].update({"denominator":8192.0}),
        "rescaled_fraction":lambda r:r[st.HISTORICAL_IDS[0]].update({"numerator":2,"denominator":16384}),
        "extra_record_key":lambda r:r[st.HISTORICAL_IDS[0]].update({"OTHER":0}),
    }
    for name, mutation in historical_mutations.items():
        check(f"reject_historical_{name}",lambda mutation=mutation:historical_rejection(mutation))

    def invalid_main_failclosed():
        bad = copy.deepcopy(rich)
        bad[-1]["systems"].pop("biaffine")
        original_mask, original_boot = st.mask_stream, st.bootstrap_index_stream
        calls = []
        def forbidden(*args,**kwargs):
            calls.append("random_stream_called")
            raise AssertionError("invalid input reached random sampling")
        st.mask_stream = st.bootstrap_index_stream = forbidden
        try:
            rejected(lambda:st.analyze_population(bad,expected_cluster_ids=rich_ids,historical_raw=historical()))
            wrong_historical = historical()
            wrong_historical.pop(st.HISTORICAL_IDS[0])
            rejected(lambda:st.analyze_population(rich,expected_cluster_ids=rich_ids,historical_raw=wrong_historical))
        finally:
            st.mask_stream, st.bootstrap_index_stream = original_mask, original_boot
        assert not calls
        return {"sampling_calls":calls,"invalid_population_or_history_no_partial_inferential_family":True}
    check("main_failclosed_before_sampling",invalid_main_failclosed)

    def runtime_guard():
        original = st.NUMPY_VERSION
        st.NUMPY_VERSION = "INVENTED_incompatible_runtime"
        try:
            return rejected(lambda:st.analyze_population(rich,expected_cluster_ids=rich_ids,historical_raw=historical()))
        finally:
            st.NUMPY_VERSION = original
    check("reject_incompatible_NumPy_runtime",runtime_guard)

    def python_runtime_guard():
        original = st.PYTHON_VERSION
        st.PYTHON_VERSION = "INVENTED_incompatible_Python"
        try:
            return rejected(lambda:st.analyze_population(rich,expected_cluster_ids=rich_ids,historical_raw=historical()))
        finally:
            st.PYTHON_VERSION = original
    check("reject_incompatible_Python_runtime",python_runtime_guard)

    def binding_recheck():
        assert module_path.read_bytes() == module_before
        assert helper_path.read_bytes() == helper_before
        return {"module_bytes_unchanged":True,"helper_bytes_unchanged":True}
    check("source_before_after_bytes_identical",binding_recheck)
    summary = {"identity":"INVENTED_IMPLEMENTER_SELF_CHECK_NOT_INDEPENDENT_REVIEW", "real_SC_annotations_or_predictions_read":False,
               "actual_old_statistics_file_read":False,"actual_SC_statistics_produced":False,
               "source_sha256":sha(module_before),"helper_sha256":sha(helper_before),
               "configuration_sha256":st.CONFIGURATION_SHA256,"runtime":{"python":sys.version,"numpy":np.__version__},
               "checks":checks,"total":len(checks),"passed":sum(c["passed"] for c in checks),
               "failed":sum(not c["passed"] for c in checks)}
    write_json(output / "actual_synthetic_results.json",summary)
    bindings = []
    for path in sorted(output.iterdir(), key=lambda p:p.name):
        if path.is_file():
            raw = path.read_bytes()
            bindings.append({"path":str(path),"sha256":sha(raw),"size_bytes":len(raw)})
    write_json(output / "output_manifest.json",{"identity":"INVENTED_SOURCE_IMPLEMENTER_CHECK_OUTPUTS_ONLY",
                                               "all_listed_files_read_and_hashed":True,"files":bindings})
    print(json.dumps({"output_dir":str(output),"total":summary["total"],"passed":summary["passed"],"failed":summary["failed"],
                      "source_sha256":summary["source_sha256"],"helper_sha256":summary["helper_sha256"]},sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
