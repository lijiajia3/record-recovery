"""Pure supplied-object test analysis; no actual-source authorization or I/O.

Primary parsing uses immutable b593 native code; E/full-R explicitly use the
prospective iterative matching v2, not native.score_document. The native replay
uses the autonomous standard-library parser/scorer. Both statistical calls use
the same reviewed fixed statistics module: replay is independent native counts
plus deterministic statistical recomputation, not an independent statistics
algorithm. Callers must complete the external fit/prediction/Gold/freeze gates.
"""
from __future__ import annotations

from collections import Counter
import copy
from fractions import Fraction
import hashlib

import sccomics_native_graph as native
import sccomics_development_selection as selector
import sccomics_primary_matching_v2 as matching_v2
import sccomics_independent_scoring as independent
import sccomics_statistics as statistics


TEST_IDS = tuple(str(i) for i in range(1, 101))
SEEDS = (20261013, 20261014, 20261015)
SYSTEMS = ('aligned', 'permuted', 'ordered_context', 'biaffine')
ENDPOINTS = ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete')
SEARCH_LIMIT = 1_000_000
T_COMPLETE = 'T_complete_native'
T_EIGHT = 'T_eight_entities'
T_TRIGGER = 'T_Doping_triggers'
T_NINE = 'T_known_nine_descriptive'
R_FULL = 'R_full_endpoint_graph_descriptive'
E_LOCAL = 'E_root_local_descriptive'
METRIC_ALIASES = {T_COMPLETE: 'text_bound_all_native', T_EIGHT: 'entities_eight',
                  T_TRIGGER: 'triggers', T_NINE: 'combined_text_bound',
                  ENDPOINTS[0]: 'relations_trigger_projected', ENDPOINTS[1]: 'events_full_graph',
                  R_FULL: 'relations_full_graph', E_LOCAL: 'events_root_local_diagnostic'}
REQUIRED_IMPORTS = ('sccomics_native_graph', 'sccomics_development_selection',
                    'sccomics_primary_matching_v2', 'sccomics_independent_scoring', 'sccomics_statistics')
PENDING_EXTERNAL_GATES = (
    'complete scientific freeze and exact captured source/import/runtime identity',
    'all 150 checkpoint/fit records and complete development choices',
    'all 15 by 100 source prediction graphs and their same-seed detector/probability provenance',
    'complete finished predictions-before-test-Gold authorization and byte barrier',
    'exact real 100 source-cluster IDs, source identity and pre/post input/output integrity',
    'different-agent SOURCE review and independent full-statistical replay helper',
)


class TestAnalysisIntegrityError(ValueError):
    """Incomplete or incompatible supplied objects; no analysis is valid."""


class TestAnalysisResourceError(RuntimeError):
    """A complete graph metric failed; no partial family may be returned."""


def require(ok, message):
    if not ok:
        raise TestAnalysisIntegrityError(message)


def _exact_keys(value, names, label):
    require(type(value) is dict and all(type(k) is str for k in value) and set(value) == set(names),
            label + ': exact complete string-key population required')


def _graphs(graphs, *, detector):
    _exact_keys(graphs, tuple(str(s) for s in SEEDS), 'all three fit populations')
    captured = {}
    for seed in SEEDS:
        values = graphs[str(seed)]
        _exact_keys(values, TEST_IDS, 'all 100 test source graphs')
        captured[str(seed)] = {}
        for source_id in TEST_IDS:
            records = values[source_id]
            require(type(records) is list, 'present explicit original JSON record list required; None is not empty prediction')
            for record in records:
                require(type(record) is dict and type(record.get('family')) is str and
                        record['family'] in ({'T'} if detector else {'T', 'R', 'E', 'AUX'}),
                        'every object requires a declared allowed native family')
            captured[str(seed)][source_id] = copy.deepcopy(records)
    return captured


def _intake(gold_sources, ner_predictions, head_predictions, source_clusters, historical_raw):
    _exact_keys(gold_sources, TEST_IDS, 'complete original GoldSource inventory')
    for source in gold_sources.values():
        require(type(source) is native.GoldSource and type(source.text_utf8) is bytes and
                type(source.annotation_utf8) is bytes, 'original unchanged source/Gold bytes required')
    ner = _graphs(ner_predictions, detector=True)
    _exact_keys(head_predictions, SYSTEMS, 'all four pair-head systems')
    heads = {name: _graphs(head_predictions[name], detector=False) for name in SYSTEMS}
    require(type(source_clusters) is list and len(source_clusters) == 100, 'exact 100 prospective singleton source clusters required')
    clusters, used_names, covered = [], set(), set()
    for row in source_clusters:
        require(type(row) is dict and set(row) == {'cluster_id', 'source_ids'}, 'explicit cluster map schema required')
        name, ids = row['cluster_id'], row['source_ids']
        require(type(name) is str and name and not any(c in name for c in '\r\n\t') and name not in used_names,
                'unique lexical cluster identity required')
        require(type(ids) is list and len(ids) == 1 and type(ids[0]) is int and 1 <= ids[0] <= 100 and ids[0] not in covered,
                'one distinct test ID per prospective singleton component')
        used_names.add(name); covered.add(ids[0]); clusters.append({'cluster_id': name, 'source_ids': list(ids)})
    require(covered == set(range(1, 101)), 'prospective singleton components must cover all test IDs 1..100')
    history = copy.deepcopy(historical_raw)
    statistics.validate_historical(history)
    # GoldSource is a frozen dataclass containing immutable bytes. Capture its
    # mapping too, so later caller mutations cannot change the parsed population
    # or its returned identity during the same supplied-object call.
    return dict(gold_sources), ner, heads, clusters, history


def _row(tp, pred_count, gold_count):
    require(type(tp) is int and 0 <= tp <= min(pred_count, gold_count), 'native population conservation failure')
    return {'tp': tp, 'fp': pred_count-tp, 'fn': gold_count-tp}


def _bag(predicted, gold, pkey, gkey):
    return _row(sum((Counter(pkey(r) for r in predicted if r.valid) &
                     Counter(gkey(r) for r in gold)).values()), len(predicted), len(gold))


def _category(record):
    known = native.ENTITY_LABELS | native.EVENT_LABELS if record.family == 'T' else native.RELATION_LABELS if record.family == 'R' else native.EVENT_LABELS
    return record.label if record.label in known else 'OTHER_NATIVE'


def _relation_key(doc, record):
    def endpoint(identifier):
        target = doc.by_id[identifier]
        if target.family == 'E': target = doc.by_id[target.trigger]
        return target.mention_key()
    return record.label, endpoint(record.arg1), endpoint(record.arg2)


def _inventory(doc):
    return {family: {'count': len(doc.family(family)), 'valid_count': len(doc.family(family, valid_only=True)),
                     'invalid_count': sum(not r.valid for r in doc.family(family)),
                     'raw_labels': dict(Counter(r.label or '<missing_label>' for r in doc.family(family)))}
            for family in ('T', 'R', 'E', 'AUX')}


def _primary_profile(pred, gold, *, detector):
    p, g = pred.family('T'), gold.family('T')
    metrics = {T_COMPLETE: selector.text_counts(pred, gold)}
    for name, labels in ((T_EIGHT, native.ENTITY_LABELS), (T_TRIGGER, native.EVENT_LABELS),
                         (T_NINE, native.ENTITY_LABELS | native.EVENT_LABELS)):
        metrics[name] = _bag([r for r in p if r.label in labels], [r for r in g if r.label in labels],
                            lambda r: r.mention_key(), lambda r: r.mention_key())
    matches = {}
    if not detector:
        metrics[ENDPOINTS[0]] = selector.relation_counts(pred, gold)
        for family, name in (('E', ENDPOINTS[1]), ('R', R_FULL)):
            result = matching_v2.exact_graph_match(pred, gold, family, max_search_steps=SEARCH_LIMIT)
            matches[name] = result
            metrics[name] = _row(result['tp'], len(pred.family(family)), len(gold.family(family)))
        result = native.root_local_event_match(pred, gold, max_search_steps=SEARCH_LIMIT)
        matches[E_LOCAL] = result
        metrics[E_LOCAL] = _row(result['tp'], len(pred.family('E')), len(gold.family('E')))
    tables = {}
    for family, name in (('T', T_COMPLETE), ('R', ENDPOINTS[0]), ('E', ENDPOINTS[1])):
        if name not in metrics: continue
        pr, gr = pred.family(family), gold.family(family)
        rows = {}
        for category in sorted({_category(r) for r in pr+gr}):
            a, b = [r for r in pr if _category(r) == category], [r for r in gr if _category(r) == category]
            if family == 'T': rows[category] = _bag(a, b, lambda r:r.mention_key(), lambda r:r.mention_key())
            elif family == 'R': rows[category] = _bag(a, b, lambda r:_relation_key(pred,r), lambda r:_relation_key(gold,r))
            else: rows[category] = _row(sum(_category(pred.by_id[x]) == category for x,y in matches[name]['matched_roots']),len(a),len(b))
        for field in ('tp', 'fp', 'fn'):
            require(sum(r[field] for r in rows.values()) == metrics[name][field], 'label/native family counts disagree')
        tables[family] = rows
    return _profile_result(pred, gold, metrics, tables, matches, native.document_json)


def _independent_profile(pred, gold, *, detector):
    # All native semantic count expectations come from the autonomous scorer,
    # including its own parser/closure/matcher and operation counter.
    if detector:
        # The detector has no R/E scoring objective. Its independently parsed
        # complete T population needs no unused graph searches or Gold filtering.
        p, g = pred.family('T'), gold.family('T')
        metrics = {T_COMPLETE: _bag(p,g,lambda r:r.mention_key(),lambda r:r.mention_key())}
        for name,labels in ((T_EIGHT,independent.ENTITY_LABELS),(T_TRIGGER,independent.EVENT_LABELS),
                            (T_NINE,independent.ENTITY_LABELS|independent.EVENT_LABELS)):
            metrics[name] = _bag([r for r in p if r.label in labels],[r for r in g if r.label in labels],
                                 lambda r:r.mention_key(),lambda r:r.mention_key())
        table = {}
        for label in sorted({_category(r) for r in p+g}):
            table[label] = _bag([r for r in p if _category(r)==label],[r for r in g if _category(r)==label],
                                lambda r:r.mention_key(),lambda r:r.mention_key())
        return _profile_result(pred,gold,metrics,{'T':table},{},independent.document_json)
    result = independent.score_document(pred, gold, max_search_steps=SEARCH_LIMIT)
    names = (T_COMPLETE, T_EIGHT, T_TRIGGER, T_NINE) if detector else tuple(METRIC_ALIASES)
    metrics = {name: {field: result['metrics'][METRIC_ALIASES[name]][field] for field in ('tp','fp','fn')} for name in names}
    tables = {family: {label: {field: row[field] for field in ('tp','fp','fn')} for label,row in rows.items()}
              for family,rows in result['by_label'].items() if not detector or family == 'T'}
    matches = {name: result['graph_matches'][METRIC_ALIASES[name]] for name in (ENDPOINTS[1],R_FULL,E_LOCAL)} if not detector else {}
    return _profile_result(pred, gold, metrics, tables, matches, independent.document_json)


def _profile_result(pred, gold, metrics, tables, matches, audit):
    native_labels = {}
    for family, name in (('T', T_COMPLETE), ('R', ENDPOINTS[0]), ('E', ENDPOINTS[1])):
        if name not in metrics: continue
        pr, gr = pred.family(family), gold.family(family)
        rows = []
        for label in sorted({r.label for r in pr+gr}, key=lambda v:(v is not None, v or '')):
            p, g = [r for r in pr if r.label == label], [r for r in gr if r.label == label]
            if family == 'T': counts = _bag(p,g,lambda r:r.mention_key(),lambda r:r.mention_key())
            elif family == 'R': counts = _bag(p,g,lambda r:_relation_key(pred,r),lambda r:_relation_key(gold,r))
            else: counts = _row(sum(pred.by_id[x].label == label for x,y in matches[name]['matched_roots']),len(p),len(g))
            rows.append({'label':label,'counts':counts})
        require(all(sum(r['counts'][k] for r in rows)==metrics[name][k] for k in ('tp','fp','fn')),
                'literal native-label/family counts disagree')
        native_labels[family] = rows
    return {'metrics': metrics, 'scores': _scores(metrics), 'by_label': tables,
            'by_literal_native_label':native_labels, 'graph_matches': matches,
            'prediction_inventory': _inventory(pred), 'gold_inventory': _inventory(gold),
            'invalid_prediction_records': [{'occurrence':r.occurrence,'family':r.family,'identifier':r.identifier,
                'label':r.label,'issues':list(r.issues)} for r in pred.records if not r.valid],
            'prediction_audit': audit(pred), 'artifact_status': pred.artifact_status,
            'gold_diagnostics': list(gold.diagnostics), 'prediction_diagnostics': list(pred.diagnostics),
            'auxiliary_scope': 'losslessly_preserved_outside_T_R_E_metrics_not_physical_truth'}


def _sum_profiles(profiles):
    profiles = tuple(profiles)
    require(bool(profiles), 'nonempty complete summary population required')
    names = tuple(profiles[0]['metrics'])
    require(all(set(row['metrics']) == set(names) for row in profiles), 'incomplete metric summary family')
    return {name: {field: sum(p['metrics'][name][field] for p in profiles) for field in ('tp','fp','fn')} for name in names}


def _scores(metrics):
    result = {}
    for name, row in metrics.items():
        f = selector.fraction_f1(row)
        precision = Fraction(row['tp'],row['tp']+row['fp']) if row['tp']+row['fp'] else Fraction(0)
        recall = Fraction(row['tp'],row['tp']+row['fn']) if row['tp']+row['fn'] else Fraction(0)
        result[name] = {'counts':dict(row),'f1_fraction':[f.numerator,f.denominator],
                        'precision_fraction':[precision.numerator,precision.denominator],
                        'recall_fraction':[recall.numerator,recall.denominator],
                        'prediction_count':row['tp']+row['fp'],'gold_count':row['tp']+row['fn']}
    return result


def _cluster_input(clusters, heads, gold_support):
    rows = []
    for cluster in clusters:
        ids = tuple(str(i) for i in cluster['source_ids'])
        support = {endpoint: sum(gold_support[i]['R' if endpoint==ENDPOINTS[0] else 'E'] for i in ids) for endpoint in ENDPOINTS}
        systems = {}
        for system in SYSTEMS:
            systems[system] = {}
            for endpoint in ENDPOINTS:
                fits = []
                for seed in SEEDS:
                    counts = {field: sum(heads[system][str(seed)][i]['metrics'][endpoint][field] for i in ids) for field in ('tp','fp','fn')}
                    require(counts['tp']+counts['fn'] == support[endpoint], 'per-fit native Gold support changed')
                    fits.append({'seed':seed,**counts})
                systems[system][endpoint] = fits
        rows.append({'cluster_id':cluster['cluster_id'],'gold_support_by_endpoint':support,'systems':systems})
    return rows


def _run(gold_sources, ner_predictions, head_predictions, source_clusters, historical_raw, *, replay):
    statistics._require_runtime()
    gold_sources, ner, raw_heads, clusters, history = _intake(gold_sources, ner_predictions, head_predictions, source_clusters, historical_raw)
    parser = independent if replay else native
    registry = parser.Registry()
    try:
        # Validate every complete Gold before producing any metric/result.
        gold = {i: parser.parse_document(gold_sources[i].text_utf8, gold_sources[i].annotation_utf8,gold=True,registry=registry) for i in TEST_IDS}
        support = {i:{family:len(gold[i].family(family)) for family in ('T','R','E','AUX')} for i in TEST_IDS}
        profile = _independent_profile if replay else _primary_profile
        ner_results = {str(seed):{} for seed in SEEDS}
        heads = {system:{str(seed):{} for seed in SEEDS} for system in SYSTEMS}
        for seed in SEEDS:
            for i in TEST_IDS:
                pred = parser.parse_prediction_records(gold_sources[i].text_utf8,ner[str(seed)][i],registry=registry)
                ner_results[str(seed)][i] = profile(pred,gold[i],detector=True)
            for system in SYSTEMS:
                for i in TEST_IDS:
                    pred = parser.parse_prediction_records(gold_sources[i].text_utf8,raw_heads[system][str(seed)][i],registry=registry)
                    heads[system][str(seed)][i] = profile(pred,gold[i],detector=False)
        cluster_rows = _cluster_input(clusters,heads,support)
        stats = statistics.analyze_population(cluster_rows,expected_cluster_ids=[c['cluster_id'] for c in clusters],historical_raw=history)
    except (native.MatchingBudgetExceeded, independent.MatchingBudgetExceeded, independent.MatchingResourceError, RecursionError, MemoryError) as exc:
        raise TestAnalysisResourceError('complete supplied test family failed exact native resource completion; no partial analysis') from exc
    summaries = {'NER': {str(seed):_scores(_sum_profiles(ner_results[str(seed)].values())) for seed in SEEDS},
                 'heads':{system:{str(seed):_scores(_sum_profiles(heads[system][str(seed)].values())) for seed in SEEDS} for system in SYSTEMS}}
    pooled = {'NER':_scores(_sum_profiles(p for seed in SEEDS for p in ner_results[str(seed)].values())),
              'heads':{system:_scores(_sum_profiles(p for seed in SEEDS for p in heads[system][str(seed)].values())) for system in SYSTEMS}}
    audit = parser.document_json
    return {'status':'complete_analysis_of_supplied_test_objects_only','actual_source_or_execution_barriers_certified':False,
            'test_source_ids':list(TEST_IDS),'seed_ids':list(SEEDS),'systems':list(SYSTEMS),
            'source_clusters':clusters,'source_cluster_count':100,'independent_source_units_not_300':True,
            'gold_native_support_by_source':support,'gold_audit_by_source':{i:audit(gold[i]) for i in TEST_IDS},
            'original_source_and_gold_byte_sha256':{i:{'text':hashlib.sha256(gold_sources[i].text_utf8).hexdigest(),
                'annotation':hashlib.sha256(gold_sources[i].annotation_utf8).hexdigest()} for i in TEST_IDS},
            'input_NER_original_source_records':ner,'input_head_original_source_records':raw_heads,
            'NER_by_seed_source':ner_results,'heads_by_system_seed_source':heads,
            'per_seed_summaries':summaries,'three_fit_pooled_count_scores':pooled,
            'three_fit_pooling_is_not_ensemble_or_mean_seed_F1':True,
            'cluster_statistics_input':cluster_rows,'statistics':stats,
            'native_algorithm':'independent_standard_library_native_parser_and_scorer' if replay else 'b593_JSON_and_Gold_parser_selector_T_R_iterative_exact_global_v2_E_R_and_b593_root_local_E',
            'independent_statistical_algorithm':False,'statistics_scope':'same_reviewed_9b92_fixed_module_deterministic_recomputation',
            'matching_budget_per_metric_document':SEARCH_LIMIT,'pending_external_gates':list(PENDING_EXTERNAL_GATES),
            'physical_or_algorithmic_truth_claimed':False,'paper_peer_review_contribution':0}


def analyze_test_population(gold_sources, ner_predictions, head_predictions, source_clusters, historical_raw):
    """Primary supplied-object analysis. Original parser's score_document is not used."""
    return _run(gold_sources,ner_predictions,head_predictions,source_clusters,historical_raw,replay=False)


def _closure_edges(doc, roots):
    colors, edges, pending = {}, Counter(), list(roots)
    while pending:
        ident = pending.pop()
        if ident in colors: continue
        require(ident in doc.by_id,'witness contains missing/invalid identifier')
        r = doc.by_id[ident]
        colors[ident] = ('T',r.label,r.segments) if r.family=='T' else (r.family,r.label)
        links = [(r.trigger,('trigger',))]+[(target,('role',role)) for role,target in r.roles] if r.family=='E' else [(r.arg1,('arg','Arg1')),(r.arg2,('arg','Arg2'))] if r.family=='R' else []
        for target,role in links: edges[ident,target,role]+=1;pending.append(target)
    return colors,edges


def _validate_witness(pred, gold, family, witness, expected_tp):
    pairs,mapping = witness['matched_roots'],witness['identifier_mapping']
    require(type(pairs) is list and type(mapping) is dict and type(witness['tp']) is int and
            witness['tp']==expected_tp and len(pairs)==expected_tp and
            all(type(p) is str and type(g) is str for p,g in mapping.items()),
            'primary graph witness/root count differs')
    require(all(type(pair) in (tuple,list) and len(pair)==2 and type(pair[0]) is str and type(pair[1]) is str for pair in pairs), 'root-pair witness schema')
    require(len({p for p,g in pairs})==len(pairs) and len({g for p,g in pairs})==len(pairs),'noninjective root witness')
    require(all(p in pred.by_id and g in gold.by_id and pred.by_id[p].family==gold.by_id[g].family==family and mapping.get(p)==g for p,g in pairs), 'invalid rooted witness')
    pc,pe = _closure_edges(pred,[p for p,g in pairs]);gc,ge = _closure_edges(gold,[g for p,g in pairs])
    require(set(mapping)==set(pc) and set(mapping.values())==set(gc) and len(set(mapping.values()))==len(mapping), 'incomplete/noninjective global witness mapping')
    require(all(pc[p]==gc[g] for p,g in mapping.items()),'witness node color differs')
    require(Counter({(mapping[a],mapping[b],role):count for (a,b,role),count in pe.items()})==ge,'witness directed multiplicity/role edges differ')


def _validate_local_witness(pred, gold, witness, expected_tp):
    pairs = witness['matched_roots']
    require(type(pairs) is list and type(witness['tp']) is int and witness['tp']==expected_tp and len(pairs)==expected_tp,
            'local witness/root count differs')
    require(all(type(pair) in (tuple,list) and len(pair)==2 and all(type(i) is str for i in pair) for pair in pairs),
            'local root-pair schema')
    require(len({p for p,g in pairs})==len(pairs) and len({g for p,g in pairs})==len(pairs), 'noninjective local root witness')
    # The independent engine checks each complete rooted closure separately;
    # no cross-root map is imposed on this descriptive metric. One inclusive
    # independent 1m budget covers all credited root-pair validations in a doc.
    budget = independent._Budget(SEARCH_LIMIT)
    for p,g in pairs:
        budget.charge('primary_local_witness_root_pairs')
        require(p in pred.by_id and g in gold.by_id and pred.by_id[p].family==gold.by_id[g].family=='E',
                'invalid local root witness')
        pc = independent._closure(pred,pred.by_id[p]);gc = independent._closure(gold,gold.by_id[g])
        require(next(independent._maps(pc,gc,{},{},budget),None) is not None,
                'primary local witness is not a complete rooted isomorphism')
    return budget.report()


def replay_test_population(gold_sources, ner_predictions, head_predictions, source_clusters, historical_raw, primary_analysis):
    """Independent native semantics; statistics reused and labeled deterministic.

    Witnesses are validated against complete independently parsed closures;
    equally optimal mapping identities or implementation operation counts need
    not agree. A mismatch/resource failure returns no replay success object.
    """
    gold_sources,ner_predictions,head_predictions,source_clusters,historical_raw = _intake(
        gold_sources,ner_predictions,head_predictions,source_clusters,historical_raw)
    replay = _run(gold_sources,ner_predictions,head_predictions,source_clusters,historical_raw,replay=True)
    require(type(primary_analysis) is dict and primary_analysis.get('status')=='complete_analysis_of_supplied_test_objects_only', 'complete primary supplied-object result required')
    for field in ('test_source_ids','seed_ids','systems','source_clusters','gold_native_support_by_source',
                  'original_source_and_gold_byte_sha256','input_NER_original_source_records','input_head_original_source_records',
                  'per_seed_summaries','three_fit_pooled_count_scores','cluster_statistics_input','statistics'):
        require(primary_analysis[field]==replay[field],'independent native/deterministic statistical replay disagrees: '+field)
    local_validation = {system:{str(seed):{} for seed in SEEDS} for system in SYSTEMS}
    for seed in SEEDS:
        for i in TEST_IDS:
            a,b = primary_analysis['NER_by_seed_source'][str(seed)][i],replay['NER_by_seed_source'][str(seed)][i]
            require(all(a[k]==b[k] for k in ('metrics','scores','by_label','by_literal_native_label','prediction_inventory','gold_inventory')), 'independent NER source population disagrees')
        for system in SYSTEMS:
            for i in TEST_IDS:
                a,b = primary_analysis['heads_by_system_seed_source'][system][str(seed)][i],replay['heads_by_system_seed_source'][system][str(seed)][i]
                require(all(a[k]==b[k] for k in ('metrics','scores','by_label','by_literal_native_label','prediction_inventory','gold_inventory')), 'independent head source population disagrees')
                source = gold_sources[i]
                p = independent.parse_prediction_records(source.text_utf8,head_predictions[system][str(seed)][i])
                g = independent.parse_document(source.text_utf8,source.annotation_utf8,gold=True)
                try:
                    for family,name in (('E',ENDPOINTS[1]),('R',R_FULL)):
                        _validate_witness(p,g,family,a['graph_matches'][name],b['metrics'][name]['tp'])
                    local_validation[system][str(seed)][i] = _validate_local_witness(p,g,a['graph_matches'][E_LOCAL],b['metrics'][E_LOCAL]['tp'])
                except (independent.MatchingBudgetExceeded,independent.MatchingResourceError,RecursionError,MemoryError) as exc:
                    raise TestAnalysisResourceError('complete independent primary-witness validation failed; no replay certificate') from exc
    replay['status']='independent_native_counts_and_deterministic_statistics_replay_for_supplied_objects_only'
    replay['primary_witnesses_validated_against_independent_finite_closures']=True
    replay['primary_local_witness_validation_by_system_seed_source']=local_validation
    replay['optimal_witness_identity_or_algorithm_operation_count_equality_required']=False
    return replay
