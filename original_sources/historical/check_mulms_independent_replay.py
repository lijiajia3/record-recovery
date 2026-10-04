"""Invented fixtures for replay barriers and strict endpoints; no corpus Gold."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

MODULE = Path(__file__).with_name('replay_mulms_neural_offline.py')
spec = importlib.util.spec_from_file_location('isolated_replay', MODULE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': name, 'sha256': replay.digest(path)}


def selection(root, directory, thresholds, target):
    epochs, candidates = {}, []
    for seed in replay.SEEDS:
        epochs[str(seed)] = {}
        for epoch in (1, 2, 3):
            cp = save(root, directory + '/seed' + str(seed) + '/epoch' + str(epoch) + '.pt',
                      {'invented_checkpoint_bytes': [seed, epoch, directory]})
            rows = {str(t): {'invented_dev-' + str(p): {'tp': int((epoch, t) == target), 'fp': 0,
                        'fn': int((epoch, t) != target)} for p in range(7)} for t in thresholds}
            epochs[str(seed)][str(epoch)] = {'checkpoint': cp['path'], 'checkpoint_sha256': cp['sha256'],
                                            'development_counts': rows}
    for epoch in (1, 2, 3):
        for t in thresholds:
            yes = (epoch, t) == target
            candidates.append({'epoch': epoch, 'threshold': t,
                               'pooled': {'tp': 21 if yes else 0, 'fp': 0, 'fn': 0 if yes else 21, 'f1': 1. if yes else 0.}})
    chosen = next(c for c in candidates if (c['epoch'], c['threshold']) == target)
    save(root, directory + '/development_all_epochs.json', epochs)
    save(root, directory + '/immutable_development_selection.json', {'chosen': chosen, 'all_pooled_candidates': candidates})
    cp = {str(seed): {'path': epochs[str(seed)][str(target[0])]['checkpoint'],
                     'sha256': epochs[str(seed)][str(target[0])]['checkpoint_sha256']} for seed in replay.SEEDS}
    return {'chosen': chosen, 'all_pooled_candidates': candidates}, cp


def fixture(root):
    # Copy tokenizer bytes only; never use a released sentence or annotation.
    tokenizer_source = MODULE.parents[1] / 'data/local_baselines/matscibert/tokenizer.json'
    target = root / 'data/local_baselines/matscibert/tokenizer.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(tokenizer_source.read_bytes())
    text = 'A A modulus'
    inputs = [{'id': 'synthetic-' + str(i), 'doc_key': 'invented-' + str(i % 7), 'text': text} for i in range(1114)]
    papers = sorted({i['doc_key'] for i in inputs})
    save(root, 'data/mulms/test_inputs.json', inputs)
    save(root, 'data/mulms/dev_inputs.json', [{'id': 'invented_dev-' + str(i),
        'doc_key': 'invented_dev-' + str(i % 7), 'text': text} for i in range(1532)])
    anchor = save(root, 'research/invented_anchor.json', {'not_actual_data': True})
    ner_t = (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99)
    rel_t = (.5, .75, .9, .95, .975, .99, .995, .999, .9995, .9999)
    detector_choice, detector_cp = selection(root, replay.TRAIN + '/ner', ner_t, (3, .8))
    selections, checkpoints = {}, {}
    for architecture in replay.ARCHITECTURES:
        selections[architecture], checkpoints[architecture] = selection(root, replay.TRAIN + '/relations/' + architecture,
                                                                         rel_t, (2, .99))
    lock = {'architecture_ids': list(replay.ARCHITECTURES), 'seed_ids': list(replay.SEEDS),
        'required_complete_detector_graphs': 3, 'required_complete_relation_graphs': 9, 'holm_family_size': 2,
        'global_two_domain_sensitivity_family_size': 4, 'file_sha256': {anchor['path']: anchor['sha256']},
        'detector_selection': detector_choice, 'detector_checkpoints': detector_cp,
        'relation_selections': selections, 'relation_checkpoints': checkpoints}
    save(root, replay.LOCK, lock)
    freeze = replay.digest(root / replay.LOCK)
    tokenizer = replay.Tokenizer.from_file(str(target))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    population = replay.boundaries(tokenizer.encode(text, add_special_tokens=False).offsets)
    entities = [{'start': 0, 'end': 1, 'type': 'MAT', 'text': 'A', 'probability': .99},
                {'start': 2, 'end': 3, 'type': 'MAT', 'text': 'A', 'probability': .99}]
    record = {'h': {'text': 'A', 'occurrence': 0, 'type': 'MAT'},
              't': {'text': 'A', 'occurrence': 1, 'type': 'MAT'}, 'r': 'usedTogether'}
    source_sha = replay.digest(root / 'data/mulms/test_inputs.json')
    manifest = {'status': 'all_three_detector_and_nine_relation_graphs_complete', 'freeze_sha256': freeze,
        'annotation_content_used': False, 'source_sentences': 1114, 'paper_ids': papers,
        'detector_graphs': {}, 'relation_graphs': {a: {} for a in replay.ARCHITECTURES}}
    for seed in replay.SEEDS:
        detector = {'seed': seed, 'annotation_content_used': False, 'freeze_sha256': freeze,
            'source_input_sha256': source_sha, 'chosen': detector_choice['chosen'],
            'checkpoint_sha256': detector_cp[str(seed)]['sha256'],
            'graphs': {i['id']: entities for i in inputs},
            'candidate_span_counts': {i['id']: len(population) for i in inputs}}
        manifest['detector_graphs'][str(seed)] = save(root, replay.DEST + '/detector' + str(seed) + '.json', detector)
        for architecture in replay.ARCHITECTURES:
            graph = {'seed': seed, 'architecture': architecture, 'annotation_content_used': False, 'freeze_sha256': freeze,
                'source_input_sha256': source_sha, 'chosen': selections[architecture]['chosen'],
                'checkpoint_sha256': checkpoints[architecture][str(seed)]['sha256'],
                'detector_graph_sha256': manifest['detector_graphs'][str(seed)]['sha256'],
                'graphs': {i['id']: [record] for i in inputs},
                'candidate_pair_counts': {i['id']: 4 for i in inputs},
                'candidate_label_counts': {i['id']: 60 for i in inputs}}
            manifest['relation_graphs'][architecture][str(seed)] = save(root, replay.DEST + '/' + architecture + str(seed) + '.json', graph)
    save(root, replay.DEST + '/generation_manifest.json', manifest)
    caches = {}
    for inp in inputs:
        name = replay.TRAIN + '/source_cache/test/' + replay.hashlib.sha256(inp['id'].encode()).hexdigest()[:20] + '.pt'
        entry = save(root, name, {'invented_cache_bytes_not_tensors': inp['id']})
        caches[name] = entry['sha256']
    save(root, replay.TRAIN + '/source_cache/test/state.json', {'status': 'completed_source_only_cache',
        'targets_read': False, 'sentences': 1114, 'complete_sentences': 1114, 'freeze_sha256': freeze, 'cache_sha256': caches})
    for architecture in replay.REFERENCES:
        choice, cps = selection(root, replay.TRAIN + '/' + architecture + '_reference/' + architecture, rel_t, (1, .975))
        ref_lock = {'file_sha256': {anchor['path']: anchor['sha256']}, 'test_annotation_content_used': False,
            'primary_freeze_sha256': freeze, 'primary_generation_manifest_sha256': replay.digest(root / (replay.DEST + '/generation_manifest.json')),
            'chosen': choice['chosen'], 'checkpoints': cps}
        lock_name = 'research/mulms_' + architecture + '_reference_test_freeze.json'
        save(root, lock_name, ref_lock)
        ref_dest = 'results/local_baseline/mulms_' + architecture + '_reference_test_v1'
        ref_manifest = {'status': 'all_three_reference_graphs_complete', 'freeze_sha256': replay.digest(root / lock_name),
            'sentences_per_seed': 1114, 'source_input_sha256': source_sha,
            'primary_generation_manifest_sha256': ref_lock['primary_generation_manifest_sha256'], 'graphs': {}}
        for seed in replay.SEEDS:
            graph = {'seed': seed, 'architecture': architecture, 'annotation_content_used': False,
                'freeze_sha256': ref_manifest['freeze_sha256'], 'source_input_sha256': source_sha, 'chosen': choice['chosen'],
                'checkpoint_sha256': cps[str(seed)]['sha256'], 'detector_graph_sha256': manifest['detector_graphs'][str(seed)]['sha256'],
                'graphs': {i['id']: [record] for i in inputs}, 'candidate_pair_counts': {i['id']: 4 for i in inputs},
                'candidate_label_counts': {i['id']: 60 for i in inputs}}
            ref_manifest['graphs'][str(seed)] = save(root, ref_dest + '/seed' + str(seed) + '.json', graph)
        save(root, ref_dest + '/generation_manifest.json', ref_manifest)


def reject_before_gold(root, mutate, restore):
    original = replay.targets
    def forbidden(*args):
        raise RuntimeError('Invented Gold reader reached before failing barrier')
    replay.targets = forbidden
    mutate()
    try:
        try:
            replay.replay(root)
        except (AssertionError, FileNotFoundError, KeyError):
            pass
        else:
            raise AssertionError('Incomplete or tampered fixture accepted')
    finally:
        restore()
        replay.targets = original


def main():
    text = 'A A modulus'
    h = {'text': 'A', 'occurrence': 0, 'type': 'MAT'}
    t = dict(h, occurrence=1)
    record = {'h': h, 't': t, 'r': 'usedTogether'}
    assert replay.canonical(record, text) == ((0, 1, 'MAT'), (2, 3, 'MAT'), 'usedTogether')
    assert not replay.valid(replay.canonical(dict(record, t=h), text))
    assert replay.endpoint(dict(h, occurrence=True), text) is None
    assert replay.endpoint(dict(h, occurrence=2), text) is None
    multilabel = {replay.canonical(record, text), replay.canonical(dict(record, r='hasForm'), text)}
    diagonal = replay.canonical(dict(record, t=h), text)
    assert replay.counts(multilabel | {diagonal}, multilabel) == {'tp': 2, 'fp': 1, 'fn': 0}
    assert replay.counts(set(), multilabel) == {'tp': 0, 'fp': 0, 'fn': 2}
    a = np.tile([1, 0, 0], (7, 1))
    b = np.tile([0, 0, 1], (7, 1))
    statistics = replay.paired_statistics(a, b)
    assert statistics['raw_exact_p'] == 2 / 128 and statistics['extreme_assignments'] == 2
    assert statistics['delta_f1_points'] == 100
    assert replay.holm([2 / 16384, 2 / 16384, 2 / 128, 1]) == [8 / 16384, 8 / 16384, 4 / 128, 1]
    with tempfile.TemporaryDirectory(prefix='invented-mulms-replay-') as temporary:
        root = Path(temporary)
        fixture(root)
        _, _, inputs, detectors, graphs, _ = replay.source_barrier(root)
        assert len(inputs) == 1114 and len(detectors) == 3 and len(graphs) == 15
        path = root / 'results/local_baseline/mulms_ordered_context_reference_test_v1/generation_manifest.json'
        saved = path.read_bytes()
        manifest = json.loads(saved)
        incomplete = copy.deepcopy(manifest)
        del incomplete['graphs'][str(replay.SEEDS[-1])]
        reject_before_gold(root, lambda: path.write_text(json.dumps(incomplete)), lambda: path.write_bytes(saved))
        graph_path = root / manifest['graphs'][str(replay.SEEDS[0])]['path']
        original = graph_path.read_bytes()
        bad = json.loads(original)
        bad['candidate_label_counts'][inputs[0]['id']] = 59
        def mutate_population():
            graph_path.write_text(json.dumps(bad))
            manifest['graphs'][str(replay.SEEDS[0])]['sha256'] = replay.digest(graph_path)
            path.write_text(json.dumps(manifest))
        reject_before_gold(root, mutate_population, lambda: (graph_path.write_bytes(original), path.write_bytes(saved)))
        cache = next((root / replay.TRAIN / 'source_cache/test').glob('*.pt'))
        cached = cache.read_bytes()
        reject_before_gold(root, lambda: cache.write_bytes(b'tampered invented cache'), lambda: cache.write_bytes(cached))
        replay.source_barrier(root)
    print(json.dumps({'invented_fixture_sentences': 1114, 'paper_units': 7, 'complete_source_graphs': 18,
        'missing_reference_graph_and_full_population_and_cache_tamper_rejected_before_Gold': True,
        'strict_repeated_endpoint_and_diagonal_FP_and_empty_FN_checks': True,
        'seven_paper_exact_exchange_and_global_Holm4_known_answers': True,
        'actual_Mu_data_or_scores_loaded': False, 'API_or_model_calls': 0}))


if __name__ == '__main__':
    main()
