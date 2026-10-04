"""Independent read-only scoring replay; no training/inference/API imports.

Requires every primary and supplementary source graph before interpreting Gold.
Uses tokenizers to reconstruct source spans, pyarrow for released annotation
tables and NumPy for resampling. It never loads a Torch checkpoint or cache.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

SEEDS = (20261003, 20261004, 20261005)
ARCHITECTURES = ('mean', 'typed', 'capacity_mean')
REFERENCES = ('biaffine', 'ordered_context')
TYPES = frozenset(('MAT', 'FORM', 'INSTRUMENT', 'DEV', 'NUM', 'UNIT', 'RANGE',
                   'VALUE', 'CITE', 'PROPERTY', 'TECHNIQUE', 'SAMPLE', 'MEASUREMENT'))
LABELS = tuple(sorted(('hasForm', 'measuresProperty', 'propertyValue', 'conditionProperty',
    'usedAs', 'conditionSampleFeatures', 'conditionPropertyValue', 'usesTechnique',
    'measuresPropertyValue', 'usedTogether', 'conditionEnvironment', 'conditionInstrument',
    'usedIn', 'takenFrom', 'dopedBy')))
LOCK = 'research/mulms_neural_family_test_freeze.json'
DEST = 'results/local_baseline/mulms_neural_family_v1'
TRAIN = 'results/local_baseline/mulms_supervised_v1'
COUNT_KEYS = ('tp', 'fp', 'fn')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(root, name):
    return json.loads((root / name).read_text())


def check_files(root, hashes):
    assert hashes
    for name, expected in hashes.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, name
        assert digest(root / relative) == expected, name


def metric(c):
    a = np.asarray(c, dtype=float)
    denominator = 2 * a[..., 0] + a[..., 1] + a[..., 2]
    return np.divide(2 * a[..., 0], denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def totals(rows):
    c = {k: sum(row[k] for row in rows.values()) for k in COUNT_KEYS}
    a, b, f = (c[k] for k in COUNT_KEYS)
    return dict(c, precision=a / (a + b) if a + b else 0,
                recall=a / (a + f) if a + f else 0,
                f1=2 * a / (2 * a + b + f) if 2 * a + b + f else 0)


def counts(predicted, gold):
    return {'tp': len(predicted & gold), 'fp': len(predicted - gold),
            'fn': len(gold - predicted)}


def equal(actual, expected, label):
    """Actual dictionaries deliberately verify subsets of saved richer reports."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            equal(value, expected[key], label + '/' + str(key))
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), label
        for i, value in enumerate(actual):
            equal(value, expected[i], label + '/' + str(i))
    elif isinstance(actual, (float, np.floating)):
        assert np.isfinite(actual) and np.isfinite(expected), label
        assert abs(actual - expected) <= 1e-10, label
    else:
        assert actual == expected, label


def common_choice(root, name, chosen, thresholds, expected_papers):
    epochs = read(root, name)
    assert set(epochs) == {str(s) for s in SEEDS}
    assert expected_papers
    candidates = []
    for seed in SEEDS:
        assert set(epochs[str(seed)]) == {'1', '2', '3'}
        for entry in epochs[str(seed)].values():
            assert digest(root / entry['checkpoint']) == entry['checkpoint_sha256']
    for epoch in (1, 2, 3):
        for t in thresholds:
            rows = []
            for seed in SEEDS:
                result = epochs[str(seed)][str(epoch)]['development_counts']
                assert set(result) == {str(x) for x in thresholds}
                assert set(result[str(t)]) == set(expected_papers), name + '/development_papers'
                for row in result[str(t)].values():
                    assert set(row) == set(COUNT_KEYS), name + '/count_fields'
                    assert all(type(row[k]) is int and row[k] >= 0 for k in COUNT_KEYS), name + '/nonnegative_integer_counts'
                rows.extend(result[str(t)].values())
            c = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
            candidates.append({'epoch': epoch, 'threshold': t,
                               'pooled': dict(c, f1=float(metric([c[k] for k in COUNT_KEYS])))})
    best = max(candidates, key=lambda c: (c['pooled']['f1'], -c['pooled']['fp'],
                                         c['threshold'], -c['epoch']))
    equal(best, chosen, name + '/chosen')
    return epochs, candidates


def entity_key(entity):
    return int(entity['start']), int(entity['end']), entity['type']


def endpoint(endpoint, text):
    if not isinstance(endpoint, dict) or endpoint.get('type') not in TYPES:
        return None
    surface, occurrence = endpoint.get('text'), endpoint.get('occurrence')
    if not isinstance(surface, str) or not surface or type(occurrence) is not int or occurrence < 0:
        return None
    starts = []
    cursor = text.find(surface)
    while cursor >= 0:
        starts.append(cursor)
        cursor = text.find(surface, cursor + 1)
    if occurrence >= len(starts):
        return None
    start = starts[occurrence]
    return start, start + len(surface), endpoint['type']


def canonical(record, text):
    if isinstance(record, dict):
        h, t, label = endpoint(record.get('h'), text), endpoint(record.get('t'), text), record.get('r')
        if h is not None and t is not None and h != t and label in LABELS:
            return h, t, label
    return 'INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False)


def valid(k):
    return len(k) == 3 and isinstance(k[0], tuple) and isinstance(k[1], tuple)


def boundaries(offsets):
    # Reconstruct all unique width<=32 WordPiece character spans independently.
    return {(offsets[a][0], offsets[b][1])
            for a in range(len(offsets)) for b in range(a, min(a + 32, len(offsets)))
            if offsets[a][0] < offsets[b][1]}


def validate_relation_graph(root, entry, architecture, seed, lock, chosen, checkpoint,
                            inputs, detector, detector_sha, freeze_sha):
    assert digest(root / entry['path']) == entry['sha256']
    graph = read(root, entry['path'])
    assert graph['seed'] == seed and graph['architecture'] == architecture
    assert graph['annotation_content_used'] is False
    assert graph['freeze_sha256'] == freeze_sha
    assert graph['source_input_sha256'] == digest(root / 'data/mulms/test_inputs.json')
    equal(graph['chosen'], chosen, architecture + '/chosen')
    assert graph['checkpoint_sha256'] == checkpoint['sha256'] == digest(root / checkpoint['path'])
    assert graph['detector_graph_sha256'] == detector_sha
    ids = {i['id'] for i in inputs}
    assert set(graph['graphs']) == set(graph['candidate_pair_counts']) == set(graph['candidate_label_counts']) == ids
    for inp in inputs:
        entities = {entity_key(e) for e in detector['graphs'][inp['id']]}
        n = len(entities) ** 2
        assert graph['candidate_pair_counts'][inp['id']] == n
        assert graph['candidate_label_counts'][inp['id']] == n * 15
        records = graph['graphs'][inp['id']]
        assert len(records) == len({json.dumps(r, sort_keys=True) for r in records})
        for r in records:
            assert endpoint(r.get('h'), inp['text']) in entities
            assert endpoint(r.get('t'), inp['text']) in entities
            assert r.get('r') in LABELS
    return graph


def source_barrier(root):
    """Only source and development content may be parsed before return."""
    lock = read(root, LOCK)
    assert lock['architecture_ids'] == list(ARCHITECTURES) and lock['seed_ids'] == list(SEEDS)
    assert lock['required_complete_detector_graphs'] == 3 and lock['required_complete_relation_graphs'] == 9
    assert lock['holm_family_size'] == 2 and lock['global_two_domain_sensitivity_family_size'] == 4
    check_files(root, lock['file_sha256'])
    freeze_sha = digest(root / LOCK)
    manifest = read(root, DEST + '/generation_manifest.json')
    assert manifest['status'] == 'all_three_detector_and_nine_relation_graphs_complete'
    assert manifest['freeze_sha256'] == freeze_sha and manifest['annotation_content_used'] is False
    assert set(manifest['detector_graphs']) == {str(s) for s in SEEDS}
    assert set(manifest['relation_graphs']) == set(ARCHITECTURES)
    assert all(set(v) == {str(s) for s in SEEDS} for v in manifest['relation_graphs'].values())
    inputs = read(root, 'data/mulms/test_inputs.json')
    ids = {i['id'] for i in inputs}
    papers = manifest['paper_ids']
    assert len(inputs) == len(ids) == manifest['source_sentences'] == 1114
    assert len(papers) == len(set(papers)) == 7 and set(papers) == {i['doc_key'] for i in inputs}
    assert all(set(i) == {'id', 'doc_key', 'text'} for i in inputs)
    cache = read(root, TRAIN + '/source_cache/test/state.json')
    assert cache['status'] == 'completed_source_only_cache' and cache['targets_read'] is False
    assert cache['sentences'] == cache['complete_sentences'] == 1114 and cache['freeze_sha256'] == freeze_sha
    paths = {TRAIN + '/source_cache/test/' + hashlib.sha256(i['id'].encode()).hexdigest()[:20] + '.pt' for i in inputs}
    assert set(cache['cache_sha256']) == paths
    check_files(root, cache['cache_sha256'])
    tokenizer = Tokenizer.from_file(str(root / 'data/local_baselines/matscibert/tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    population = {i['id']: boundaries(tokenizer.encode(i['text'], add_special_tokens=False).offsets) for i in inputs}
    dev_inputs = read(root, 'data/mulms/dev_inputs.json')
    dev_papers = {i['doc_key'] for i in dev_inputs}
    assert len(dev_inputs) == len({i['id'] for i in dev_inputs}) == 1532 and len(dev_papers) == 7
    ner_epochs, ner_candidates = common_choice(root, TRAIN + '/ner/development_all_epochs.json',
        lock['detector_selection']['chosen'], (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99), dev_papers)
    equal(ner_candidates, lock['detector_selection']['all_pooled_candidates'], 'ner/search')
    detectors, graphs = {}, {}
    for architecture in ARCHITECTURES:
        selected = lock['relation_selections'][architecture]
        rel_epochs, choices = common_choice(root, TRAIN + '/relations/' + architecture + '/development_all_epochs.json',
            selected['chosen'], (.5, .75, .9, .95, .975, .99, .995, .999, .9995, .9999), dev_papers)
        equal(choices, selected['all_pooled_candidates'], architecture + '/search')
        for seed in SEEDS:
            cp = rel_epochs[str(seed)][str(selected['chosen']['epoch'])]
            assert cp['checkpoint_sha256'] == lock['relation_checkpoints'][architecture][str(seed)]['sha256']
            assert cp['checkpoint'] == lock['relation_checkpoints'][architecture][str(seed)]['path']
    for seed in SEEDS:
        entry = manifest['detector_graphs'][str(seed)]
        assert digest(root / entry['path']) == entry['sha256']
        graph = read(root, entry['path'])
        assert graph['seed'] == seed and graph['annotation_content_used'] is False and graph['freeze_sha256'] == freeze_sha
        assert graph['source_input_sha256'] == digest(root / 'data/mulms/test_inputs.json')
        equal(graph['chosen'], lock['detector_selection']['chosen'], 'detector/chosen')
        cp = lock['detector_checkpoints'][str(seed)]
        assert graph['checkpoint_sha256'] == cp['sha256'] == digest(root / cp['path'])
        assert cp['sha256'] == ner_epochs[str(seed)][str(graph['chosen']['epoch'])]['checkpoint_sha256']
        assert cp['path'] == ner_epochs[str(seed)][str(graph['chosen']['epoch'])]['checkpoint']
        assert set(graph['graphs']) == set(graph['candidate_span_counts']) == ids
        for inp in inputs:
            entities = graph['graphs'][inp['id']]
            assert len(entities) == len({entity_key(e) for e in entities})
            assert graph['candidate_span_counts'][inp['id']] == len(population[inp['id']])
            for e in entities:
                assert (e['start'], e['end']) in population[inp['id']] and e['type'] in TYPES
                assert e['text'] == inp['text'][e['start']:e['end']] and e['text']
                assert np.isfinite(e['probability']) and graph['chosen']['threshold'] <= e['probability'] <= 1
        detectors[seed] = graph
        for architecture in ARCHITECTURES:
            graphs[architecture, seed] = validate_relation_graph(root, manifest['relation_graphs'][architecture][str(seed)],
                architecture, seed, lock, lock['relation_selections'][architecture]['chosen'],
                lock['relation_checkpoints'][architecture][str(seed)], inputs, graph, entry['sha256'], freeze_sha)
    refs = {}
    for architecture in REFERENCES:
        ref_lock_name = 'research/mulms_' + architecture + '_reference_test_freeze.json'
        ref_dest = 'results/local_baseline/mulms_' + architecture + '_reference_test_v1'
        ref_lock = read(root, ref_lock_name)
        check_files(root, ref_lock['file_sha256'])
        assert ref_lock['test_annotation_content_used'] is False
        assert ref_lock['primary_freeze_sha256'] == freeze_sha
        assert ref_lock['primary_generation_manifest_sha256'] == digest(root / (DEST + '/generation_manifest.json'))
        ref_manifest = read(root, ref_dest + '/generation_manifest.json')
        assert ref_manifest['status'] == 'all_three_reference_graphs_complete'
        assert ref_manifest['freeze_sha256'] == digest(root / ref_lock_name)
        assert ref_manifest['sentences_per_seed'] == 1114 and set(ref_manifest['graphs']) == {str(s) for s in SEEDS}
        assert ref_manifest['source_input_sha256'] == digest(root / 'data/mulms/test_inputs.json')
        assert ref_manifest['primary_generation_manifest_sha256'] == ref_lock['primary_generation_manifest_sha256']
        ref_train = TRAIN + '/' + architecture + '_reference/' + architecture
        ref_epochs, candidates = common_choice(root, ref_train + '/development_all_epochs.json', ref_lock['chosen'],
            (.5, .75, .9, .95, .975, .99, .995, .999, .9995, .9999), dev_papers)
        choice = read(root, ref_train + '/immutable_development_selection.json')
        equal(candidates, choice['all_pooled_candidates'], architecture + '/search')
        refs[architecture] = (ref_lock, ref_dest, ref_manifest)
        for seed in SEEDS:
            cp = ref_epochs[str(seed)][str(ref_lock['chosen']['epoch'])]
            assert cp['checkpoint_sha256'] == ref_lock['checkpoints'][str(seed)]['sha256']
            assert cp['checkpoint'] == ref_lock['checkpoints'][str(seed)]['path']
            graphs[architecture, seed] = validate_relation_graph(root, ref_manifest['graphs'][str(seed)],
                architecture, seed, ref_lock, ref_lock['chosen'], ref_lock['checkpoints'][str(seed)], inputs,
                detectors[seed], manifest['detector_graphs'][str(seed)]['sha256'], digest(root / ref_lock_name))
    return lock, manifest, inputs, detectors, graphs, refs


def targets(root, lock, inputs):
    assert digest(root / 'data/mulms/test_gold.json') == lock['test_gold_sha256']
    edges = read(root, 'data/mulms/test_gold.json')
    source = root / 'data/mulms/source/test.parquet'
    assert digest(source) == lock['test_ner_source_sha256']
    lookup = {i['id']: i for i in inputs}
    assert set(edges) == set(lookup)
    ner = {}
    for row in pq.read_table(source, columns=['doc_id', 'sentence', 'beginOffset', 'NER_labels']).to_pylist():
        identity = row['doc_id'] + '|' + str(row['beginOffset'])
        assert identity in lookup and identity not in ner
        assert row['sentence'] == lookup[identity]['text'] and row['doc_id'] == lookup[identity]['doc_key']
        labels = row['NER_labels']
        assert len({len(v) for v in labels.values()}) == 1
        records = []
        for j in range(len(labels['id'])):
            a, b, kind = int(labels['begin'][j]), int(labels['end'][j]), labels['value'][j]
            assert kind in TYPES and 0 <= a < b <= len(row['sentence'])
            assert row['sentence'][a:b] == labels['text'][j]
            records.append((a, b, kind))
        assert len(records) == len(set(records))
        ner[identity] = set(records)
    assert set(ner) == set(lookup)
    return edges, ner


def root_graphs(edges):
    grouped = defaultdict(set)
    for identity, key in edges:
        if valid(key) and key[0][2] == 'MEASUREMENT':
            grouped[identity, key[0]].add((key[1], key[2]))
    return {(identity, head, frozenset(tails)) for (identity, head), tails in grouped.items()}


def relation_counts(inputs, graph, gold, papers):
    by_paper = {p: dict.fromkeys(COUNT_KEYS, 0) for p in papers}
    by_label = {r: dict.fromkeys(COUNT_KEYS, 0) for r in LABELS}
    pred_roots, gold_roots = set(), set()
    for inp in inputs:
        predicted = {canonical(r, inp['text']) for r in graph[inp['id']]}
        target = {canonical(r, inp['text']) for r in gold[inp['id']]}
        assert all(valid(k) for k in target)
        c = counts(predicted, target)
        for k in COUNT_KEYS:
            by_paper[inp['doc_key']][k] += c[k]
        for label in LABELS:
            p = {canonical(r, inp['text']) for r in graph[inp['id']] if r['r'] == label}
            c = counts(p, {r for r in target if r[2] == label})
            for k in COUNT_KEYS:
                by_label[label][k] += c[k]
        pred_roots.update((inp['id'], k) for k in predicted)
        gold_roots.update((inp['id'], k) for k in target)
    return {'by_paper': by_paper, 'total': totals(by_paper),
            'per_label': {r: totals({'label': c}) for r, c in by_label.items()},
            'observed_measurement_root_exact_graph_descriptive': totals({'root': counts(root_graphs(pred_roots), root_graphs(gold_roots))})}


def pooled_result(seeds, papers):
    pooled = {p: {k: sum(s['by_paper'][p][k] for s in seeds.values()) for k in COUNT_KEYS} for p in papers}
    vector = np.asarray([[pooled[p][k] for k in COUNT_KEYS] for p in papers], dtype=np.int64)
    draws = np.random.default_rng(20261003).integers(len(papers), size=(10000, len(papers)))
    return {'all_seeds': seeds, 'pooled_by_original_paper': pooled, 'pooled_primary': totals(pooled),
            'paper_bootstrap_95_f1_ci': np.quantile(metric(vector[draws].sum(1)), [.025, .975]).tolist()}, vector


def paired_statistics(a, b):
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3 and 0 < len(a) <= 20
    assert np.all(a >= 0) and np.all(b >= 0)
    assert np.array_equal(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    n = len(a)
    observed = float(metric(a.sum(0)) - metric(b.sum(0)))
    extreme = 0
    for assignment in range(1 << n):
        selected = np.asarray([(assignment >> i) & 1 for i in range(n)], dtype=bool)[:, None]
        left, right = np.where(selected, a, b), np.where(selected, b, a)
        difference = float(metric(left.sum(0)) - metric(right.sum(0)))
        extreme += abs(difference) >= abs(observed) - 1e-12
    draws = np.random.default_rng(20261003).integers(0, n, size=(10000, n))
    boot = 100 * (metric(a[draws].sum(1)) - metric(b[draws].sum(1)))
    return {'delta_f1_points': 100 * observed, 'raw_exact_p': extreme / (1 << n),
            'permutation_units': n, 'enumerated_assignments': 1 << n,
            'extreme_assignments': int(extreme),
            'paired_paper_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist()}


def holm(values):
    assert all(np.isfinite(p) and 0 <= p <= 1 for p in values)
    output, maximum = [0.] * len(values), 0.
    for rank, i in enumerate(sorted(range(len(values)), key=lambda j: values[j])):
        maximum = max(maximum, min(1., (len(values) - rank) * values[i]))
        output[i] = maximum
    return output


def direction_diagnostic(inputs, detectors, graphs, gold, papers):
    output = {}
    for seed in SEEDS:
        per_paper = {p: {name: dict.fromkeys(COUNT_KEYS, 0) for name in
                        ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts')} for p in papers}
        for p in papers:
            per_paper[p]['architectures'] = {a: dict.fromkeys(('valid_non_diagonal_predicted_edges',
                'edges_missing_same_label_reverse', 'invalid_including_diagonal_predictions'), 0) for a in ARCHITECTURES}
        for inp in inputs:
            target = {canonical(r, inp['text']) for r in gold[inp['id']]}
            assert all(valid(k) for k in target)
            available = {entity_key(e) for e in detectors[seed]['graphs'][inp['id']]}
            for name, reachable in (('full_gold_entity_oracle_counts', target),
                ('predicted_entity_oracle_counts', {e for e in target if e[0] in available and e[1] in available})):
                reverse_closed = reachable | {(e[1], e[0], e[2]) for e in reachable}
                c = counts(reverse_closed, target)
                for k in COUNT_KEYS:
                    per_paper[inp['doc_key']][name][k] += c[k]
            for architecture in ARCHITECTURES:
                predicted = {canonical(r, inp['text']) for r in graphs[architecture, seed]['graphs'][inp['id']]}
                good = {e for e in predicted if valid(e)}
                c = {'valid_non_diagonal_predicted_edges': len(good),
                     'edges_missing_same_label_reverse': sum((e[1], e[0], e[2]) not in good for e in good),
                     'invalid_including_diagonal_predictions': len(predicted - good)}
                for k, value in c.items():
                    per_paper[inp['doc_key']]['architectures'][architecture][k] += value
        output[str(seed)] = {'by_original_paper': per_paper}
        for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts'):
            c = {k: sum(p[name][k] for p in per_paper.values()) for k in COUNT_KEYS}
            output[str(seed)][name] = c
            output[str(seed)][name + '_f1'] = float(metric([c[k] for k in COUNT_KEYS]))
        output[str(seed)]['architectures'] = {a: {k: sum(p['architectures'][a][k] for p in per_paper.values())
            for k in per_paper[papers[0]]['architectures'][a]} for a in ARCHITECTURES}
    return output


def global_four(root, mu_vectors):
    global_report = read(root, 'results/local_baseline/joint_supervised_global_holm4.json')
    assert global_report['all_four_contrasts_present'] and global_report['local_holm2_reports_modified'] is False
    rows = []
    for domain, n in (('polyie', 14), ('mulms', 7)):
        name = 'results/local_baseline/' + domain + '_neural_family_v1/test_summary.json'
        assert digest(root / name) == global_report['source_summary_sha256'][domain]
        report = read(root, name)
        planned = report['planned_contrasts']
        assert len(planned) == 2 and {c['name'] for c in planned} == {'typed_minus_mean', 'typed_minus_capacity_mean'}
        equal(holm([c['raw_exact_p'] for c in report['planned_contrasts']]),
              [c['holm_p_two_planned_neural_contrasts'] for c in report['planned_contrasts']],
              domain + '/local_holm2')
        papers = report['paper_ids']
        assert len(papers) == len(set(papers)) == n
        assert papers == read(root, 'results/local_baseline/' + domain + '_neural_family_v1/generation_manifest.json')['paper_ids']
        for control in ('mean', 'capacity_mean'):
            if domain == 'mulms':
                a, b = mu_vectors['typed'], mu_vectors[control]
            else:
                vectors = []
                for architecture in ('typed', control):
                    by_paper = report['architecture_results'][architecture]['pooled_by_original_paper']
                    assert set(by_paper) == set(papers)
                    vectors.append(np.asarray([[by_paper[p][k] for k in COUNT_KEYS] for p in papers]))
                a, b = vectors
            statistics = paired_statistics(a, b)
            stored = next(c for c in report['planned_contrasts'] if c['name'] == 'typed_minus_' + control)
            equal(statistics, stored, domain + '/contrast')
            rows.append({'domain': domain, 'contrast': 'typed_minus_' + control,
                'raw_exact_p': statistics['raw_exact_p'], 'delta_f1_points': statistics['delta_f1_points'],
                'original_local_holm2_p': stored['holm_p_two_planned_neural_contrasts']})
    for row, corrected in zip(rows, holm([r['raw_exact_p'] for r in rows])):
        row['global_holm4_p'] = corrected
    equal(rows, global_report['contrasts'], 'global_holm4')
    return rows


def replay(root):
    lock, manifest, inputs, detectors, graphs, references = source_barrier(root)
    # First semantic access to held-out Gold: all 3+9+3+3 source graphs passed.
    gold, ner_gold = targets(root, lock, inputs)
    papers = manifest['paper_ids']
    reported = read(root, DEST + '/test_summary.json')
    assert reported['freeze_sha256'] == digest(root / LOCK)
    assert reported['generation_manifest_sha256'] == digest(root / (DEST + '/generation_manifest.json'))
    assert reported['seeds_are_independent_papers'] is False and reported['test_gold_loaded_only_after_all_source_graphs'] is True
    assert reported['test_papers'] == 7 and reported['test_sentences'] == 1114
    ner_seeds = {}
    for seed in SEEDS:
        by_paper = {p: dict.fromkeys(COUNT_KEYS, 0) for p in papers}
        for inp in inputs:
            c = counts({entity_key(e) for e in detectors[seed]['graphs'][inp['id']]}, ner_gold[inp['id']])
            for k in COUNT_KEYS:
                by_paper[inp['doc_key']][k] += c[k]
        ner_seeds[str(seed)] = {'by_paper': by_paper, 'total': totals(by_paper)}
    ner, _ = pooled_result(ner_seeds, papers)
    equal({'all_seeds': ner['all_seeds'], 'pooled_by_original_paper': ner['pooled_by_original_paper'],
           'pooled': ner['pooled_primary']}, reported['detector_descriptive'], 'detector')
    results, vectors = {}, {}
    for architecture in (*ARCHITECTURES, *REFERENCES):
        seeds = {str(s): relation_counts(inputs, graphs[architecture, s]['graphs'], gold, papers) for s in SEEDS}
        result, vector = pooled_result(seeds, papers)
        results[architecture], vectors[architecture] = result, vector
        if architecture in ARCHITECTURES:
            result['pooled_per_label_descriptive'] = {r: totals({'label': {k: sum(s['per_label'][r][k] for s in seeds.values())
                for k in COUNT_KEYS}}) for r in LABELS}
            equal(result, reported['architecture_results'][architecture], architecture)
        else:
            ref_lock, ref_dest, ref_manifest = references[architecture]
            report = read(root, ref_dest + '/test_summary.json')
            equal(result, report, architecture)
            assert report['freeze_sha256'] == digest(root / ('research/mulms_' + architecture + '_reference_test_freeze.json'))
            assert report['generation_manifest_sha256'] == digest(root / (ref_dest + '/generation_manifest.json'))
            assert report['fresh_independent_holdout_after_primary_score_claimed'] is False
            assert report['planned_primary_holm2_and_global4_modified'] is False
            assert report['primary_summary_existed_before_reference_test_lock'] == ref_lock['primary_summary_already_exists']
    contrasts = [paired_statistics(vectors['typed'], vectors[control]) for control in ('mean', 'capacity_mean')]
    for row, corrected in zip(contrasts, holm([r['raw_exact_p'] for r in contrasts])):
        row['holm_p_two_planned_neural_contrasts'] = corrected
    equal(contrasts, reported['planned_contrasts'], 'planned_contrasts')
    symmetry = direction_diagnostic(inputs, detectors, graphs, gold, papers)
    symmetry_report = read(root, DEST + '/descriptive_direction_symmetry.json')
    equal(symmetry, symmetry_report['all_seeds'], 'symmetry')
    pooled_symmetry = {}
    for p in papers:
        pooled_symmetry[p] = {name: {k: sum(s['by_original_paper'][p][name][k] for s in symmetry.values())
            for k in COUNT_KEYS} for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts')}
        pooled_symmetry[p]['architectures'] = {a: {k: sum(s['by_original_paper'][p]['architectures'][a][k]
            for s in symmetry.values()) for k in symmetry[str(SEEDS[0])]['architectures'][a]} for a in ARCHITECTURES}
    equal(pooled_symmetry, symmetry_report['pooled_by_original_paper'], 'symmetry/pooled_papers')
    oracle_totals = {}
    for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts'):
        c = {k: sum(p[name][k] for p in pooled_symmetry.values()) for k in COUNT_KEYS}
        oracle_totals[name] = dict(c, f1=float(metric([c[k] for k in COUNT_KEYS])))
    oracle_totals['architectures'] = {a: {k: sum(p['architectures'][a][k] for p in pooled_symmetry.values())
        for k in symmetry[str(SEEDS[0])]['architectures'][a]} for a in ARCHITECTURES}
    equal(oracle_totals, symmetry_report['pooled_counts'], 'symmetry/pooled_totals')
    assert symmetry_report['freeze_sha256'] == digest(root / LOCK)
    assert symmetry_report['generation_manifest_sha256'] == digest(root / (DEST + '/generation_manifest.json'))
    assert symmetry_report['oracle_is_actual_implementable_method'] is False
    return {'scope': 'MuLMS primary and two fixed supplementary directed-reference scoring replay',
        'replay_source_sha256': digest(Path(__file__).resolve()), 'independent_scoring_and_statistics_implementation': True,
        'paid_api_calls': 0, 'Torch_or_model_inference_used': False, 'test_papers': 7, 'test_sentences': 1114,
        'complete_detector_graphs': 3, 'complete_relation_graphs': 15,
        'all_eighteen_test_graphs_before_replay_Gold': True,
        'all_seeds_and_full_gold_FN_and_diagonal_FP_verified': True,
        'native_source_tokenizer_full_span_population_reconstructed': True,
        'source_cache_hashes_verified_but_cache_tensor_contents_not_loaded': True,
        'all_nine_epoch_dev_results_and_checkpoint_bytes_per_family_verified': True,
        'detector_pooled': ner['pooled_primary'],
        'pooled_primary': {a: r['pooled_primary'] for a, r in results.items()},
        'planned_contrasts': contrasts, 'global_four_contrasts': global_four(root, vectors),
        'direction_symmetry_all_seed_counts_verified': True,
        'source_summary_sha256': {name: digest(root / name) for name in (
            DEST + '/test_summary.json', DEST + '/descriptive_direction_symmetry.json',
            'results/local_baseline/joint_supervised_global_holm4.json',
            'results/local_baseline/mulms_biaffine_reference_test_v1/test_summary.json',
            'results/local_baseline/mulms_ordered_context_reference_test_v1/test_summary.json')},
        'freeze_sha256': digest(root / LOCK), 'matches_saved_summaries': True,
        'training_rerun_or_inference_reproduced': False,
        'Poly_summary_count_vectors_used_only_for_global_Holm4': True,
        'Poly_Gold_graph_replay_requires_separate_Poly_script': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = replay(args.root.resolve())
    if args.out:
        assert not args.out.exists(), 'Read-only replay must not replace prior evidence'
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
