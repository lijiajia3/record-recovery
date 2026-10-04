"""Complete supplied API populations -> primary and autonomous native counts.

No filesystem, network, credential, training, or actual Gold authorization.
The caller must rebuild the complete raw source barrier before opening any
annotation. Missing predictions are errors here, never manufactured FN-only
graphs. All emissions, including invalid objects, are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import copy

import sccomics_native_graph as native
import sccomics_independent_scoring as independent
import sccomics_test_analysis as profiles
from sccomics_api_native_projection_v1 import TRAIN_REGISTRY, compile_graph_json

MODELS = ('deepseek-ai/DeepSeek-V4-Pro', 'zai-org/GLM-5.3')
ARMS = ('B', 'G', 'T', 'S')
REPEATS = (0, 1, 2)
IDS = {'development': tuple(str(i) for i in range(101, 113)),
       'test': tuple(str(i) for i in range(1, 101))}
ENDPOINTS = ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete')


class PopulationIntegrityError(ValueError): pass
class PopulationResourceError(RuntimeError): pass


@dataclass(frozen=True)
class SourceGraph:
    source_utf8: bytes
    raw_assistant_json_utf8: bytes
    emissions: tuple[native.Emission, ...]


def need(condition, message):
    if not condition: raise PopulationIntegrityError(message)


def exact(obj, keys, message):
    keys = tuple(keys)
    need(type(obj) is dict and set(obj) == set(keys) and
         all(type(k) is type(wanted) for k in obj for wanted in keys if k == wanted), message)


def intake(phase, gold_sources, predictions, clusters):
    need(type(phase) is str and phase in IDS, 'explicit complete development/test phase')
    ids = IDS[phase]
    exact(gold_sources, ids, 'complete original Gold inventory')
    for value in gold_sources.values():
        need(type(value) is native.GoldSource and type(value.text_utf8) is bytes and
             type(value.annotation_utf8) is bytes, 'original supplied text/annotation bytes')
    exact(predictions, MODELS, 'both model populations')
    captured = {}
    for model in MODELS:
        exact(predictions[model], ARMS, 'all four arms')
        captured[model] = {}
        for arm in ARMS:
            exact(predictions[model][arm], REPEATS, 'all three original request repetitions')
            captured[model][arm] = {}
            for repeat in REPEATS:
                docs = predictions[model][arm][repeat]
                exact(docs, ids, 'complete explicit source graph inventory; no missing fallback')
                captured[model][arm][repeat] = {}
                for identifier in ids:
                    graph = docs[identifier]
                    need(type(graph) is SourceGraph and type(graph.source_utf8) is bytes and
                         type(graph.raw_assistant_json_utf8) is bytes and type(graph.emissions) is tuple,
                         'complete original graph/source/emission object')
                    need(graph.source_utf8 == gold_sources[identifier].text_utf8, 'original source identity')
                    for emission in graph.emissions:
                        need(type(emission) is native.Emission and emission.family in ('T', 'R', 'E') and
                             type(emission.payload) is bytes and
                             (emission.declared_label is None or type(emission.declared_label) is str),
                             'every original family-declared occurrence retained')
                    compiled = compile_graph_json(graph.source_utf8, graph.raw_assistant_json_utf8)
                    need(compiled.emissions is not None and compiled.document is not None and
                         compiled.emissions == graph.emissions,
                         'raw JSON recompiled to the exact original emission population')
                    captured[model][arm][repeat][identifier] = graph
    need(type(clusters) is list and len(clusters) == len(ids), 'complete singleton source cluster inventory')
    kept, names, covered = [], set(), set()
    for row in clusters:
        exact(row, ('cluster_id', 'source_ids'), 'explicit cluster map schema')
        name, members = row['cluster_id'], row['source_ids']
        need(type(name) is str and name and not any(c in name for c in '\r\n\t') and name not in names,
             'unique lexical source cluster name')
        need(type(members) is list and len(members) == 1 and type(members[0]) is int and
             str(members[0]) in ids and members[0] not in covered, 'complete fixed singleton sources')
        names.add(name); covered.add(members[0]); kept.append(copy.deepcopy(row))
    need(covered == set(map(int, ids)), 'every original source cluster retained')
    return ids, dict(gold_sources), captured, kept


def native_counts(phase, gold_sources, predictions, clusters, *, autonomous=False):
    """Pure supplied objects only; no generation/first-Gold barrier certificate."""
    need(type(autonomous) is bool, 'explicit parser/scorer choice')
    ids, sources, graphs, clusters = intake(phase, gold_sources, predictions, clusters)
    parser = independent if autonomous else native
    registry = parser.Registry(entity_labels=TRAIN_REGISTRY.entity_labels,
        event_labels=TRAIN_REGISTRY.event_labels, relation_labels=TRAIN_REGISTRY.relation_labels,
        event_roles=TRAIN_REGISTRY.event_roles,
        allow_numeric_role_suffix=TRAIN_REGISTRY.allow_numeric_role_suffix)
    try:
        gold = {i: parser.parse_document(sources[i].text_utf8, sources[i].annotation_utf8,
                         gold=True, registry=registry) for i in ids}
        support = {i: {f: len(gold[i].family(f)) for f in ('T', 'R', 'E', 'AUX')} for i in ids}
        results = {}
        for model in MODELS:
            results[model] = {}
            for arm in ARMS:
                results[model][arm] = {}
                for repeat in REPEATS:
                    results[model][arm][repeat] = {}
                    for i in ids:
                        graph = graphs[model][arm][repeat][i]
                        emissions = tuple(parser.Emission(e.family, e.payload, e.declared_label)
                                          for e in graph.emissions)
                        pred = parser.parse_emissions(graph.source_utf8, emissions, registry=registry)
                        need(len(pred.records) == len(emissions), 'emission count conservation')
                        profile = profiles._independent_profile if autonomous else profiles._primary_profile
                        results[model][arm][repeat][i] = profile(pred, gold[i], detector=False)
        cluster_rows = []
        for cluster in clusters:
            i = str(cluster['source_ids'][0])
            cluster_rows.append({'source_cluster_id': cluster['cluster_id'],
                'gold_support_by_endpoint': {ENDPOINTS[0]: support[i]['R'], ENDPOINTS[1]: support[i]['E']},
                'models': {m: {a: {ep: [dict(repeat=r, **results[m][a][r][i]['metrics'][ep])
                    for r in REPEATS] for ep in ENDPOINTS} for a in ARMS} for m in MODELS}})
    except (native.MatchingBudgetExceeded, independent.MatchingBudgetExceeded,
            independent.MatchingResourceError, RecursionError, MemoryError) as error:
        raise PopulationResourceError('complete native family cannot finish; no partial family score') from error
    return {'status': 'complete_native_counts_for_supplied_API_population_only', 'phase': phase,
        'source_ids': list(ids), 'models': list(MODELS), 'arms': list(ARMS), 'request_repeats': list(REPEATS),
        'source_clusters': clusters, 'source_cluster_count': len(clusters),
        'gold_native_support_by_source': support, 'by_model_arm_repeat_source': results,
        'cluster_statistics_input': cluster_rows,
        'gold_audit_by_source': {i: parser.document_json(gold[i]) for i in ids},
        'input_graph_byte_identity': {m: {a: {r: {i: {
            'source_sha256': hashlib.sha256(graphs[m][a][r][i].source_utf8).hexdigest(),
            'raw_assistant_json_sha256': hashlib.sha256(graphs[m][a][r][i].raw_assistant_json_utf8).hexdigest()}
            for i in ids} for r in REPEATS} for a in ARMS} for m in MODELS},
        'source_gold_byte_identity': {i: {'text': hashlib.sha256(sources[i].text_utf8).hexdigest(),
            'annotation': hashlib.sha256(sources[i].annotation_utf8).hexdigest()} for i in ids},
        'autonomous_native_parser_scorer': autonomous,
        'independent_statistical_algorithm_or_actual_barriers_certified': False,
        'repeats_are_original_requests_not_ensemble_or_independent_sources': True,
        'matching_budget_per_metric_document': 1_000_000,
        'physical_or_annotation_truth_claimed': False, 'paper_peer_review_contribution': 0}


def compare_native_replay(primary, replay):
    """Compare semantic counts and literal labels; optimum witness IDs may differ."""
    need(type(primary) is dict and type(replay) is dict and
         primary.get('autonomous_native_parser_scorer') is False and
         replay.get('autonomous_native_parser_scorer') is True, 'distinct native algorithm reports')
    for name in ('status', 'phase', 'source_ids', 'models', 'arms', 'request_repeats',
                 'source_clusters', 'source_cluster_count', 'gold_native_support_by_source',
                 'cluster_statistics_input', 'input_graph_byte_identity', 'source_gold_byte_identity'):
        need(primary[name] == replay[name], 'native replay identity/counts mismatch: ' + name)
    count = 0
    for model in MODELS:
        for arm in ARMS:
            for repeat in REPEATS:
                for i in primary['source_ids']:
                    a = primary['by_model_arm_repeat_source'][model][arm][repeat][i]
                    b = replay['by_model_arm_repeat_source'][model][arm][repeat][i]
                    for name in ('metrics', 'scores', 'by_label', 'by_literal_native_label', 'artifact_status'):
                        need(a[name] == b[name], 'native semantic metric mismatch: ' + name)
                    count += 1
    need(count == len(primary['source_ids']) * 24, 'complete count-comparison population')
    return {'status': 'exact_native_counts_and_literal_labels_match', 'graph_comparisons': count,
        'same_optimum_witness_identity_required': False, 'statistical_replay_certified': False,
        'actual_population_or_first_Gold_gate_certified': False}
