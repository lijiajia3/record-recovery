"""Bind a prospective methods draft to control metadata, not actual outcomes."""
import datetime
import hashlib
import json
from pathlib import Path


def binding(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def main():
    root = Path(__file__).resolve().parents[1]
    draft = 'manuscript/work_in_progress/sccomics_prospective_methods_pending_actual_execution.md'
    sources = [draft,
        'research/round4_sccomics_experimental_configuration_before_fit.json',
        'research/round4_sccomics_execution_details_before_fit.json',
        'research/round4_sccomics_execution_details_before_fit.md',
        'research/round4_sccomics_pre_fit_ner_denominator_amendment.json',
        'research/round4_sccomics_primary_matching_v2_prospective_contract.json',
        'research/round4_sccomics_train_schema_actual_root_crosscheck.json',
        'research/round4_sccomics_actual_cache_root_crosscheck.json',
        'research/round4_sccomics_native_task_contract_source_audit.json',
        'research/round4_sccomics_native_task_contract_source_audit.md',
        'research/round4_sccomics_independent_statistics_v2_before_fit_gate_repair.json',
        'research/round4_sccomics_independent_statistics_source_review.json',
        'research/round4_sccomics_test_analysis_independent_source_review.json',
        'research/round4_sccomics_supervised_v2_stable_handoff_root_binding_check.json',
        'data/sccomics_round4/source_v3/source_acquisition_manifest.json',
        'results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json']
    before = {name: binding(root / name) for name in sources}
    def read(name):
        return json.loads((root / name).read_bytes())
    config = read(sources[1])
    schema = read('research/round4_sccomics_train_schema_actual_root_crosscheck.json')
    cache = read('research/round4_sccomics_actual_cache_root_crosscheck.json')
    acquired = read('data/sccomics_round4/source_v3/source_acquisition_manifest.json')
    checks = {
        'TRAIN800': len(schema['native_ids']) == 800,
        'TRAIN_T34032': schema['native_record_counts']['T'] == 34032,
        'TRAIN_R1981': schema['native_record_counts']['R'] == 1981,
        'TRAIN_E2337': schema['native_record_counts']['E'] == 2337,
        'Doping2087': schema['native_labels']['T']['Doping'] == 2087,
        'shared161': schema['shared_trigger_groups'] == 161,
        'SOURCE1000': acquired['native_source_abstract_records'] == 1000,
        'TRAIN_ids': acquired['split_ids']['train'] == list(range(201, 1001)),
        'DEV_ids': acquired['split_ids']['dev'] == list(range(101, 201)),
        'TEST_ids': acquired['split_ids']['test'] == list(range(1, 101)),
        'WP247935': cache['total_wordpieces'] == 247935,
        'chunks1019': cache['total_chunks'] == 1019,
        'spans7437920': cache['total_distinct_source_span_candidates'] == 7437920,
        'three_fixed_seeds': config['seeds'] == [20261013, 20261014, 20261015],
        'ten_epochs': config['epochs_per_fit'] == 10,
        '150_CP': config['expected_new_epoch_checkpoints'] == 150,
        'four_heads': config['architectures'] == ['aligned', 'permuted', 'ordered_context', 'biaffine'],
        'span80_grid': len(config['span_thresholds']) * 10 == 80,
        'pair250_grid': len(config['R_thresholds']) * len(config['role_thresholds']) * 10 == 250,
        'NER3000_source_records': 3 * 10 * 100 == 3000,
        'PAIR12000_source_records': 4 * 3 * 10 * 100 == 12000,
        'TEST1500_graphs': (3 + 4 * 3) * 100 == 1500,
        'six_contrasts': len(config['primary_contrast_ids']) == 6,
        'four_historical': len(config['historic_additional_sensitivity_ids']) == 4,
        'MC100000': config['MC_draws_with_replacement'] == 100000,
        'bootstrap10000': config['bootstrap_draws'] == 10000,
        'budget1000000': config['search_budget_per_native_metric_document'] == 1000000,
        'no_hidden_vector_regeneration': cache['hidden_vector_values_independently_regenerated'] is False,
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})
    after = {name: binding(root / name) for name in sources}
    if after != before:
        raise ValueError('Control/draft bytes changed during working-text check.')
    skill = Path('/Users/jiajia/.codex/skills/nature-writing/SKILL.md')
    report = {
        'identity': 'PROSPECTIVE_METHODS_WORKING_DRAFT_NUMERIC_SOURCE_BINDINGS_NOT_FINAL_PAPER_REVIEW',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'draft': draft,
        'word_count_entire_working_file': len((root / draft).read_text().split()),
        'numeric_design_checks': checks,
        'checks_passed': len(checks),
        'input_bindings': before,
        'helper': {'path': str(Path(__file__).relative_to(root)), **binding(Path(__file__))},
        'skill': {'path': str(skill), **binding(skill)},
        'axes': 'manuscript / research / method / zh-to-en / generic; target Science Advances',
        'preparatory_inline_failures': [
            'Incorrect guessed gate-repair metadata filename; no report created.',
            'len() called on actual shared-trigger-group integer; no report created.'
        ],
        'actual_SC_fits_or_dev_test_semantics_read_by_this_check': False,
        'raw_corpus_annotation_model_or_tensor_cache_bytes_read_by_this_check': False,
        'new_results_claimed_or_final_peer_humanizer_pass_incremented': False,
        'original_helper_exit_or_actual_runtime_compatibility_certified': False,
    }
    output = root / 'research/round4_sccomics_prospective_methods_working_draft_record.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'passed': True, 'numeric_checks': len(checks),
                      'words_entire_working_file': report['word_count_entire_working_file'],
                      'record_sha256': binding(output)['sha256']}))


if __name__ == '__main__':
    main()
