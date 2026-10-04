"""Replay saved SC-CoMIcs graphs using the autonomous scorer and mathematics.

Run only from a completely verified extracted evidence root. This wrapper
does not authorize new predictions, change any choice, import the primary
scorer, train a model, or access a credential. It is not neural regeneration.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

ACTION = 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004_score_test/completed_local_action_receipt.json'
HEADS = ('aligned', 'permuted', 'ordered_context', 'biaffine')
SEEDS = ('20261013', '20261014', '20261015')
IDS = tuple(str(i) for i in range(1, 101))
ALIASES = {
    'T_complete_native': 'text_bound_all_native',
    'T_eight_entities': 'entities_eight',
    'T_Doping_triggers': 'triggers',
    'T_known_nine_descriptive': 'combined_text_bound',
    'R_trigger_projected_complete': 'relations_trigger_projected',
    'E_global_sharing_consistent_complete': 'events_full_graph',
    'R_full_endpoint_graph_descriptive': 'relations_full_graph',
    'E_root_local_descriptive': 'events_root_local_diagnostic',
}
PINS = {
    'sccomics_independent_scoring': '27846cb4792ac7485d2ca41378ff5c0530330d7ae940ce5ca2913dabe92cf367',
    'sccomics_independent_statistics_v2': 'b83c9959b567a016d2177d862c4347465c4f346ac4f4081680b9c5a1532b14c7',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(root, name):
    p = PurePosixPath(name)
    need(type(name) is str and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
         'Only relative nontraversing evidence paths are allowed')
    result = root / p
    need(result.is_file() and not result.is_symlink() and result.resolve().is_relative_to(root),
         'Evidence must be an included regular file')
    return result


def binding(path):
    h, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
            n += len(chunk)
    return {'sha256': h.hexdigest(), 'size_bytes': n}


def captured(root, descriptor):
    p = safe(root, descriptor['path'])
    need(binding(p) == {k: descriptor[k] for k in ('sha256', 'size_bytes')}, 'Captured output changed')
    return json.loads(p.read_bytes())


def load_module(root, name):
    import sys
    path = safe(root, 'src/' + name + '.py')
    need(binding(path)['sha256'] == PINS[name], 'Autonomous implementation changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def counts(report, expected):
    need(set(expected['metrics']) in (set(ALIASES), set(list(ALIASES)[:4])), 'Complete detector or head metrics required')
    for name in expected['metrics']:
        alias = ALIASES[name]
        actual = report['metrics'][alias]
        want = expected['metrics'][name]
        need(all(type(actual[k]) is int and actual[k] == want[k] for k in ('tp', 'fp', 'fn')),
             'Autonomous native count mismatch: ' + name)
    return {name: {k: report['metrics'][ALIASES[name]][k] for k in ('tp', 'fp', 'fn')}
            for name in expected['metrics']}


def replay(root):
    root = root.resolve()
    done = json.loads(safe(root, ACTION).read_bytes())
    need(done['status'] == 'completed_local_score_test', 'A complete real analysis is required')
    for name, want in done['output_bindings_before_completion'].items():
        need(binding(safe(root, name)) == want, 'Actual completed-action output changed')
    result = done['result']
    primary = captured(root, result['primary_analysis'])
    original_native = captured(root, result['different_native_count_replay'])
    original_math = captured(root, result['different_statistical_arithmetic_replay'])
    need(primary['status'] == 'complete_analysis_of_supplied_test_objects_only' and
         original_native['status'] == 'independent_native_counts_and_deterministic_statistics_replay_for_supplied_objects_only' and
         original_math['status'] == 'independent_arithmetic_replay_passed_for_supplied_counts_only',
         'All three completed analyses are required')
    need(primary['test_source_ids'] == list(IDS) and primary['seed_ids'] == [int(s) for s in SEEDS] and
         primary['systems'] == list(HEADS) and primary['source_cluster_count'] == 100,
         'The fixed complete population is required')
    scorer = load_module(root, 'sccomics_independent_scoring')
    math = load_module(root, 'sccomics_independent_statistics_v2')
    gold, texts, source_bindings = {}, {}, {}
    for ident in IDS:
        stem = f'{int(ident):04}'
        text_path = safe(root, f'data/sccomics_round4/source_v3/raw_text/{stem}.txt')
        ann_path = safe(root, f'data/sccomics_round4/source_v3/raw_annotations/{stem}.ann')
        wanted = primary['original_source_and_gold_byte_sha256'][ident]
        need(binding(text_path)['sha256'] == wanted['text'] and binding(ann_path)['sha256'] == wanted['annotation'],
             'Original text/Gold bytes changed')
        texts[ident] = text_path.read_bytes()
        gold[ident] = scorer.parse_document(texts[ident], ann_path.read_bytes(), gold=True)
        source_bindings[ident] = wanted
    rows = {'NER': {s: {} for s in SEEDS}, 'heads': {h: {s: {} for s in SEEDS} for h in HEADS}}
    documents = 0
    for seed in SEEDS:
        for ident in IDS:
            prediction = scorer.parse_prediction_records(texts[ident], primary['input_NER_original_source_records'][seed][ident])
            report = scorer.score_document(prediction, gold[ident], max_search_steps=1_000_000)
            rows['NER'][seed][ident] = counts(report, primary['NER_by_seed_source'][seed][ident])
            documents += 1
        for head in HEADS:
            for ident in IDS:
                prediction = scorer.parse_prediction_records(texts[ident], primary['input_head_original_source_records'][head][seed][ident])
                report = scorer.score_document(prediction, gold[ident], max_search_steps=1_000_000)
                rows['heads'][head][seed][ident] = counts(report, primary['heads_by_system_seed_source'][head][seed][ident])
                documents += 1
    need(documents == 1500, 'Every graph must be semantically replayed')
    for cluster in primary['cluster_statistics_input']:
        item = next(c for c in primary['source_clusters'] if c['cluster_id'] == cluster['cluster_id'])
        ident = str(item['source_ids'][0])
        need(len(item['source_ids']) == 1, 'Fixed singleton source clusters required')
        for head in HEADS:
            for endpoint in ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete'):
                want = cluster['systems'][head][endpoint]
                need(want == [{'seed': int(seed), **rows['heads'][head][seed][ident][endpoint]} for seed in SEEDS],
                     'Statistical counts differ from autonomous native replay')
    history_path = safe(root, 'results/local_baseline/joint_supervised_global_holm4.json')
    history_sha = binding(history_path)['sha256']
    need(history_sha == 'c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0',
         'The actual unchanged historical source is required')
    history = json.loads(history_path.read_bytes())
    historical = {}
    for row in history['contrasts']:
        value = Fraction(row['raw_exact_p'])
        historical[row['domain'] + ':' + row['contrast']] = {
            'numerator': value.numerator, 'denominator': value.denominator, 'source_sha256': history_sha}
    arithmetic = math.replay_statistics(primary['cluster_statistics_input'],
        expected_cluster_ids=[c['cluster_id'] for c in primary['source_clusters']],
        historical_raw=historical, primary_statistics=primary['statistics'])
    need(arithmetic == original_math, 'Fresh extracted arithmetic differs from the completed original replay')
    return {'status': 'fresh_extracted_SC_native_and_arithmetic_replay_complete',
        'source_cluster_count': 100, 'original_source_graphs_semantically_replayed': documents,
        'all_document_T_R_E_and_descriptive_counts_match': True,
        'source_bytes': source_bindings, 'independent_math_output': arithmetic,
        'autonomous_scoring_source_SHA256': PINS, 'primary_scorer_imported': False,
        'model_training_or_inference': False, 'credential_read': False,
        'shared_fixed_random_streams_not_independent_random_samples': True,
        'historical_record_loaded': bool(history), 'physical_truth_certified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'Never overwrite a prior replay')
    start = datetime.now(timezone.utc).isoformat()
    result = replay(args.root)
    result.update(started_utc=start, completed_utc=datetime.now(timezone.utc).isoformat(),
                  wrapper_source=binding(Path(__file__)))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'source_cluster_count', 'original_source_graphs_semantically_replayed')}))


if __name__ == '__main__':
    main()
