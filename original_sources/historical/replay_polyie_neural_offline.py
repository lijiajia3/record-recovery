"""Independent NumPy-only replay of the completed POLYIE neural experiment.

Does not import the training, inference, task-scoring, statistics or API clients.
Checks all nine source graphs before interpreting test annotations. Read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (20261003, 20261004, 20261005)
ARCH = {'mean': 'polyie_shared_context_v1', 'typed': 'polyie_typed_interaction_v1',
        'capacity_mean': 'polyie_capacity_mean_v1'}
ROLES = ('CN', 'PN', 'PV', 'Condition')
LOCK = 'research/polyie_neural_family_test_freeze.json'
DEST = 'results/local_baseline/polyie_neural_family_v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(root, name):
    return json.loads((root / name).read_text())


def metric(counts):
    a = np.asarray(counts, dtype=float)
    denom = 2 * a[..., 0] + a[..., 1] + a[..., 2]
    return np.divide(2 * a[..., 0], denom, out=np.zeros_like(denom), where=denom > 0)


def summarize(rows):
    totals = {k: sum(row[k] for row in rows.values()) for k in ('tp', 'fp', 'fn')}
    a, b, c = (totals[k] for k in ('tp', 'fp', 'fn'))
    return {**totals, 'precision': a / (a + b) if a + b else 0,
            'recall': a / (a + c) if a + c else 0,
            'f1': 2 * a / (2 * a + b + c) if 2 * a + b + c else 0}


def candidate_counts(inputs, papers):
    counts = {p: 0 for p in papers}
    for inp in inputs:
        n = [sum(e['type'] == role for e in inp['entities']) for role in ROLES]
        count = n[0] * n[1] * n[2] * (n[3] + 1)
        if inp['start']:
            previous = [sum(e['type'] == role and e['end'] <= inp['start'] + 200
                            for e in inp['entities']) for role in ROLES]
            count -= previous[0] * previous[1] * previous[2] * (previous[3] + 1)
        assert count >= 0
        counts[inp['doc_key']] += count
    return counts


def selection_from_epochs(epochs):
    assert set(epochs) == {str(s) for s in SEEDS}
    choices = []
    for epoch in (1, 2, 3):
        thresholds = list(epochs[str(SEEDS[0])][str(epoch)]['development_counts'])
        for threshold in thresholds:
            rows = [r for seed in SEEDS for r in
                    epochs[str(seed)][str(epoch)]['development_counts'][threshold].values()]
            c = {k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')}
            choices.append({'epoch': epoch, 'threshold': float(threshold), **c,
                            'f1': float(metric([c[k] for k in ('tp', 'fp', 'fn')]))})
    return max(choices, key=lambda c: (c['f1'], -c['fp'], c['threshold'], -c['epoch']))


def source_barrier(root):
    """No test Gold is semantically parsed here; frozen bytes may be hashed."""
    lock = read(root, LOCK)
    assert lock['architecture_ids'] == list(ARCH) and lock['seed_ids'] == list(SEEDS)
    assert lock['required_complete_graphs'] == 9 and lock['holm_family_size'] == 2
    for name, expected in lock['file_sha256'].items():
        assert digest(root / name) == expected, name
    manifest = read(root, DEST + '/generation_manifest.json')
    assert manifest['freeze_sha256'] == digest(root / LOCK)
    assert manifest['test_targets_read'] is False
    inputs = read(root, 'data/polyie/test_inputs.json')
    assert digest(root / 'data/polyie/test_inputs.json') == lock['test_source_sha256']
    assert manifest['test_source_sha256'] == lock['test_source_sha256']
    assert len(inputs) == len({i['window_id'] for i in inputs}) == 61
    papers = sorted({i['doc_key'] for i in inputs})
    assert len(papers) == 14 and papers == manifest['paper_ids']
    assert manifest['scope'] == {'papers': 14, 'windows': 61}
    assert manifest['architecture_ids'] == list(ARCH) and manifest['seed_ids'] == list(SEEDS)
    expected_population = candidate_counts(inputs, papers)
    assert set(manifest['complete_graphs']) == {a + '|' + str(s) for a in ARCH for s in SEEDS}
    assert set(manifest['source_cache_sha256']) == {
        DEST + '/source_cache/' + str(i) + '.pt' for i in range(len(inputs))}
    for name, expected in manifest['source_cache_sha256'].items():
        assert digest(root / name) == expected, name
    graphs = {}
    for architecture, directory in ARCH.items():
        base = 'results/local_baseline/' + directory
        selected = read(root, base + '/immutable_development_selection.json')
        assert selected['chosen'] == selection_from_epochs(read(root, base + '/development_all_epochs.json'))
        assert selected['chosen'] == lock['selections'][architecture]['chosen']
        assert digest(root / (base + '/immutable_development_selection.json')) == lock['selections'][architecture]['sha256']
        for seed in SEEDS:
            entry = manifest['complete_graphs'][architecture + '|' + str(seed)]
            assert digest(root / entry['path']) == entry['sha256']
            graph = read(root, entry['path'])
            assert graph['architecture'] == architecture and graph['seed'] == seed
            assert graph['test_targets_read'] is False
            assert graph['epoch'] == selected['chosen']['epoch']
            assert graph['threshold'] == selected['chosen']['threshold']
            assert graph['selection_sha256'] == lock['selections'][architecture]['sha256']
            checkpoint = lock['checkpoints'][architecture][str(seed)]
            assert graph['checkpoint_sha256'] == checkpoint['sha256'] == digest(root / checkpoint['path'])
            assert set(graph['graphs']) == set(papers)
            assert graph['enumerated_candidates_by_paper'] == expected_population
            assert all(len(graph['graphs'][p]) <= expected_population[p] for p in papers)
            graphs[architecture, seed] = graph
    return lock, manifest, graphs


def canonical(record, entities):
    invalid = ('INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False))
    if not isinstance(record, dict):
        return invalid
    result = []
    for role in ROLES:
        value = record.get(role)
        if not isinstance(value, list) or any(not isinstance(x, (str, int)) for x in value):
            return invalid
        ids = tuple(sorted(set(map(str, value))))
        if role != 'Condition' and not ids:
            return invalid
        if any(i not in entities or entities[i]['type'] != role for i in ids):
            return invalid
        result.append(ids)
    return tuple(result)


def count_sets(predicted, gold):
    return {'tp': len(predicted & gold), 'fp': len(predicted - gold), 'fn': len(gold - predicted)}


def paired_statistics(a, b):
    assert np.array_equal(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    n = len(a)
    delta = float(metric(a.sum(0)) - metric(b.sum(0)))
    extreme = 0
    # Exhaustive independent method-label swapping; no Monte Carlo P value.
    for assignment in range(2 ** n):
        choose = np.array([(assignment >> i) & 1 for i in range(n)], dtype=bool)
        x, y = np.where(choose[:, None], a, b), np.where(choose[:, None], b, a)
        value = float(metric(x.sum(0)) - metric(y.sum(0)))
        extreme += abs(value) >= abs(delta) - 1e-12
    rng = np.random.default_rng(20261003)
    draws = rng.integers(0, n, size=(10000, n))
    bootstrap = 100 * (metric(a[draws].sum(1)) - metric(b[draws].sum(1)))
    return {'delta_f1_points': 100 * delta, 'raw_exact_p': extreme / 2 ** n,
            'permutation_units': n, 'enumerated_assignments': 2 ** n,
            'extreme_assignments': int(extreme),
            'paired_paper_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist()}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1, values[i] * (len(values) - rank)))
        adjusted[i] = previous
    return adjusted


def assert_equal(actual, expected, label):
    if isinstance(actual, dict):
        for k, value in actual.items():
            assert_equal(value, expected[k], label + '/' + k)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), label
        for i, value in enumerate(actual):
            assert_equal(value, expected[i], label + '/' + str(i))
    elif isinstance(actual, (float, np.floating)):
        assert abs(actual - expected) <= 1e-10, label
    else:
        assert actual == expected, label


def replay(root):
    lock, manifest, graphs = source_barrier(root)
    # First semantic access to held-out Gold, after all nine graphs pass.
    assert digest(root / 'data/polyie/test_gold.json') == lock['test_gold_sha256']
    labels = read(root, 'data/polyie/test_gold.json')
    docs = read(root, 'data/polyie/test_documents.json')
    entities = {d['doc_key']: {e['id']: e for e in d['entities']} for d in docs}
    papers = manifest['paper_ids']
    assert set(labels) == set(entities) == set(papers)
    reported = read(root, DEST + '/test_summary.json')
    assert reported['freeze_sha256'] == digest(root / LOCK)
    assert reported['generation_manifest_sha256'] == digest(root / (DEST + '/generation_manifest.json'))
    architectures, vectors = {}, {}
    for architecture in ARCH:
        pooled = {p: {'tp': 0, 'fp': 0, 'fn': 0} for p in papers}
        seeds = {}
        for seed in SEEDS:
            rows = {name: {} for name in ('covered_primary', 'all_valid_full_document_sensitivity',
                                         'released_malformed_FN_sensitivity')}
            for p in papers:
                pred = {canonical(r, entities[p]) for r in graphs[architecture, seed]['graphs'][p]}
                eligible = {canonical(r, entities[p]) for r in labels[p]['eligible']}
                full = {canonical(r, entities[p]) for r in labels[p]['all_valid']}
                malformed = {('MALFORMED_RELEASED_GOLD', json.dumps(r, sort_keys=True))
                             for r in labels[p]['schema_excluded']}
                for name, gold in zip(rows, (eligible, full, full | malformed)):
                    rows[name][p] = count_sets(pred, gold)
                for k in pooled[p]:
                    pooled[p][k] += rows['covered_primary'][p][k]
            seeds[str(seed)] = {name: {'by_paper': counts, 'total': summarize(counts)}
                                for name, counts in rows.items()}
        vector = np.array([[pooled[p][k] for k in ('tp', 'fp', 'fn')] for p in papers])
        vectors[architecture] = vector
        draws = np.random.default_rng(20261003).integers(0, len(papers), size=(10000, len(papers)))
        architectures[architecture] = {'all_seeds': seeds, 'pooled_by_original_paper': pooled,
            'pooled_primary': summarize(pooled),
            'paper_bootstrap_95_f1_ci': np.quantile(metric(vector[draws].sum(1)), [.025, .975]).tolist()}
    assert_equal(architectures, reported['architecture_results'], 'architecture_results')
    contrasts = [paired_statistics(vectors['typed'], vectors[control]) for control in ('mean', 'capacity_mean')]
    for result, corrected in zip(contrasts, holm([r['raw_exact_p'] for r in contrasts])):
        result['holm_p_two_planned_neural_contrasts'] = corrected
    assert_equal(contrasts, reported['planned_contrasts'], 'planned_contrasts')
    return {'scope': 'completed POLYIE neural nine-graph scoring replay only',
            'replay_source_sha256': digest(Path(__file__).resolve()),
            'independent_scoring_and_statistics_implementation': True,
            'paid_api_calls': 0, 'graphs': 9, 'papers': 14,
            'all_seeds_and_three_target_sensitivities_verified': True,
            'full_source_candidate_population_verified_before_gold': True,
            'all_nine_epoch_dev_results_per_architecture_verify_common_choice': True,
            'pooled_primary': {a: r['pooled_primary'] for a, r in architectures.items()},
            'planned_contrasts': contrasts, 'freeze_sha256': digest(root / LOCK),
            'matches_saved_summary': True, 'training_rerun_or_inference_reproduced': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = replay(args.root.resolve())
    if args.out:
        assert not args.out.exists(), 'Read-only evidence replay must not replace an earlier audit'
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
