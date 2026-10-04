"""Amended root assembler for reviewed v3 routes and true v4 import proof.

The original prepared assembler remains immutable and was never executed.

This is a document assembler. It does not import scientific libraries, parse
native annotations, load weights/caches, fit a model, or claim performance.
Individual independent reviews, actual tool-observed import compatibility and
fixed original material receipts must all be present before authority files
are created. It never replaces an existing authority or invents a pass.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
REVIEW_OUTPUT = 'research/round4_sccomics_supervised_complete_source_review.json'
FREEZE_OUTPUT = 'research/round4_sccomics_complete_scientific_freeze.json'
GATE_OUTPUT = 'research/round4_sccomics_supervised_final_gate.json'
FIT_OUTPUT = 'results/local_baseline/sccomics_native_v1/supervised_v1'
RUNTIME_RECEIPT = 'research/round4_sccomics_supervised_runtime_receipt.json'
RUNTIME_SHA = 'dd5e65a4134ab97d71454907b71bf4aea3f7654dbbc66291dbd5ae38344d9cb4'
CORE_REPORT = 'research/round4_sccomics_core_review_closure_root_check.json'
MATERIAL_REPORT = 'research/round4_sccomics_actual_material_metadata_closure_root_check.json'
COMPAT_ROOT = 'research/round4_sccomics_actual_v4_import_compatibility_root_receipt.json'
COMPAT_REVIEW = 'research/round4_sccomics_runtime_import_compatibility_v4_source_review.json'
IMPORT_SOURCE = 'src/check_sccomics_runtime_imports_source_only_v4.py'
RUNTIME = {'python': '3.13.7', 'numpy': '2.2.6', 'torch': '2.9.1',
           'transformers': '4.57.3', 'tokenizers': '0.22.1', 'networkx': '3.6.1'}
SPLITS = {'train': list(range(201, 1001)), 'dev': list(range(101, 201)),
          'test': list(range(1, 101))}
ENTRIES = ['torch', 'numpy', 'tokenizers', 'transformers', 'networkx', 'safetensors']
LEGACY_INDEPENDENCE_IDENTITIES = {
    'research/round4_sccomics_native_graph_independent_source_review.json':
        ('reviewer_identity', 'independent model-based source reviewer /root/sccomics_native_review; separate from implementation agent; not human/expert/paper-peer review'),
    'research/round4_sccomics_source_model_independent_review.json':
        ('review_identity', 'model-based independent source reviewer /root/literature of root-authored source-model primitives; no human expert certificate'),
    'research/round4_sccomics_training_projection_independent_source_review.json':
        ('review_identity', 'model-based independent source reviewer of root-authored projection/emitter primitives; not human expert or paper peer review'),
}
CONTROL_NAMES = [
    'research/round4_sccomics_experimental_configuration_before_fit.json',
    'research/round4_sccomics_execution_details_before_fit.json',
    'research/round4_sccomics_pre_fit_ner_denominator_amendment.json',
    'research/round4_sccomics_supervised_stages_v2_before_fit_source_contract.json',
    'research/round4_sccomics_supervised_stages_v2_independent_findings_before_fit_addendum.json',
    'research/round4_sccomics_supervised_stages_v2_late_findings_statistics_addendum.json',
    'research/round4_sccomics_supervised_stages_v2_runtime_and_fresh_dev_before_fit_addendum.json',
    'research/round4_sccomics_text_provenance_actual/test_source_text_clusters.json',
    'results/local_baseline/joint_supervised_global_holm4.json',
    'data/sccomics_round4/source_v3/source_acquisition_manifest.json',
    'results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json',
    'data/local_baselines/matscibert/source_manifest.json',
    REVIEW_OUTPUT, RUNTIME_RECEIPT,
]
CONTROL_NAMES += [name.removesuffix('.json') + '.md' for name in CONTROL_NAMES[:7]]
CONTROL_NAMES += [
    'research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum.json',
    'research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum.md',
]
SUPPLEMENTS = [
    CORE_REPORT, MATERIAL_REPORT,
    'research/round4_sccomics_actual_raw_acquisition_root_crosscheck.json',
    'research/round4_sccomics_train_schema_actual_root_crosscheck.json',
    'research/round4_sccomics_actual_cache_root_crosscheck.json',
    'research/round4_sccomics_actual_text_provenance_root_crosscheck.json',
    'research/round4_sccomics_native_task_contract_source_audit.json',
    'research/round4_sccomics_native_task_contract_chronology_addendum.json',
    'research/round4_sccomics_primary_matching_v2_prospective_contract.json',
    'research/round4_sccomics_primary_matching_v2_prospective_contract.md',
    'research/round4_sccomics_independent_statistics_v2_before_fit_gate_repair.json',
    'research/round4_sccomics_independent_statistics_v2_before_fit_gate_repair.md',
    'research/round4_sccomics_independent_statistics_final94_receipt_root_check.json',
    'research/round4_sccomics_runtime_sourceonly_helpers149_root_binding_check.json',
    'research/round4_sccomics_actual_runtime_source_preparation_root_receipt.json',
    'research/round4_sccomics_actual_source_only_import_failure_before_entries.json',
    'research/round4_sccomics_runtime_dirfd_v3_198_handoff_root_binding_check.json',
    'research/round4_sccomics_runtime_dirfd_v3_302_review_root_binding_check.json',
    'research/round4_sccomics_runtime_dirfd_v2_independent_review_inputs/initial_af93_independent_source_findings.md',
    'research/round4_sccomics_actual_v3_runtime_optional_import_failure_root_receipt.json',
    'research/round4_sccomics_runtime_missing_import_source_design_preparation.json',
    'research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum.json',
    'research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum.md',
    'src/assemble_sccomics_complete_authority_root.py',
    COMPAT_REVIEW, COMPAT_ROOT,
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw(name):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Regular canonical SOURCE/control proof required: ' + name)
    return path.read_bytes()


def digest(name):
    value = hashlib.sha256()
    count = 0
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Regular canonical evidence required: ' + name)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
            count += len(chunk)
    return {'sha256': value.hexdigest(), 'size_bytes': count}


def read_json(name):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate proof key.')
            result[key] = value
        return result
    return json.loads(raw(name), object_pairs_hook=unique)


def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def name_check(name):
    path = Path(name)
    require(name and '..' not in path.parts, 'Noncanonical proof binding.')
    if not path.is_absolute():
        require(path.parts[0] in {'src', 'research', 'data', 'results'},
                'Undeclared proof-tree namespace.')
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compatibility-root-receipt-sha256', required=True)
    parser.add_argument('--new-helper-independent-report', required=True)
    parser.add_argument('--new-helper-independent-report-sha256', required=True)
    args = parser.parse_args()
    require(digest(COMPAT_ROOT)['sha256'] == args.compatibility_root_receipt_sha256,
            'Explicit actual compatibility receipt differs.')
    observed = read_json(COMPAT_ROOT)
    require(observed['identity'] == 'ROOT_TOOL_OBSERVED_ACTUAL_V4_SIX_IMPORT_COMPATIBILITY_NOT_FIT_AUTHORITY' and
            observed['captured_exec_exit_code'] == 0 and
            observed['actual_compatibility_passed'] is True and
            observed['actual_fitting_or_SC_semantic_permission'] is False and
            observed['runtime_receipt_sha256'] == RUNTIME_SHA,
            'Actual six-import success with observed exit0 is required.')
    actual_name = name_check(observed['result_binding']['path'])
    require(digest(actual_name) == {key: observed['result_binding'][key]
                                 for key in ('sha256', 'size_bytes')}, 'Actual result identity differs.')
    compatibility = read_json(actual_name)
    require(compatibility['identity'] == 'ACTUAL_FINITE_RUNTIME_IMPORT_COMPATIBILITY_ONLY_NOT_FIT_AUTHORITY' and
            compatibility['passed'] is True and compatibility['source_only'] is True and
            compatibility['tokenizer_model_or_backend_constructed'] is False and
            compatibility['actual_SC_annotation_text_model_weight_cache_checkpoint_prediction_read'] is False and
            compatibility['actual_fitting_inference_or_scientific_statistic'] is False and
            compatibility['stage_fitting_registry_unchanged'] is True and
            [value['module'] for value in compatibility['entry_libraries_imported']] == ENTRIES and
            compatibility['runtime_validation']['actual_file_hashes_checked'] == 11969,
            'Real source-only six-entry/runtime result not a genuine complete pass.')
    # Bind the successful captured import run to the current exact helper,
    # route, receipt and review/protocol bytes. Merely reading a newly edited
    # review carrying the same status text cannot authorize different code.
    for name, expected in compatibility['original_input_bindings'].items():
        name_check(name)
        require(digest(name) == {key: expected[key] for key in ('sha256', 'size_bytes')},
                'Current input differs from successful captured import run.')
    import_review = read_json(COMPAT_REVIEW)
    import_source_sha = import_review['bound_source_files'][IMPORT_SOURCE]['sha256']
    require(digest(RUNTIME_RECEIPT)['sha256'] == RUNTIME_SHA and
            digest(IMPORT_SOURCE)['sha256'] == import_source_sha,
            'Canonical runtime receipt/reviewed helper changed.')
    import_review = read_json(COMPAT_REVIEW)
    require(import_review['status'] == 'source_only_review_passed' and
            import_review['independent_from_implementer'] is True and
            import_review['scope'] == 'finite_runtime_import_compatibility_only' and
            import_review['actual_fit_or_SC_semantic_permission'] is False and
            import_review['entry_libraries'] == ENTRIES and
            import_review['bound_source_files'][IMPORT_SOURCE]['sha256'] == import_source_sha,
            'Distinct v4 independent SOURCE review must have passed its finite scope.')
    new_report_name = name_check(args.new_helper_independent_report)
    require(digest(new_report_name)['sha256'] == args.new_helper_independent_report_sha256,
            'Explicit different-helper review differs.')
    new_report = read_json(new_report_name)
    require(new_report['status'] == 'source_only_review_passed' and
            new_report['independent_from_implementer'] is True,
            'Different new helper artificial/SOURCE review required.')
    core = read_json(CORE_REPORT)
    require(core['passed'] is True and core['review_count'] == 10 and
            core['distinct_bound_files_checked'] == 522 and
            core['repeated_bindings_checked_for_agreement'] == 122 and
            not core['failures'] and len(core['execution_sources']) == 12,
            'Root scientific SOURCE closure is not complete.')
    material = read_json(MATERIAL_REPORT)
    require(material['passed'] is True and material['expanded_material_count'] == 3007 and
            material['complete_native_split_ids'] == SPLITS and
            material['corpus_tensor_cold_or_annotation_content_opened'] is False,
            'Root actual material receipt/metadata closure is required.')
    all_reviews = dict(core['review_bindings'])
    all_reviews[new_report_name] = digest(new_report_name)
    # Corrected ca832 preparer has its own different review. Its original 879
    # import helper remains blocked by the retained actual pre-entry failure.
    previous_helper_review = 'research/round4_sccomics_runtime_sourceonly_helpers_independent_source_review.json'
    all_reviews[previous_helper_review] = digest(previous_helper_review)
    proof_bindings = {}
    source_review_sources = set()
    external_allowlist = set(read_json(core['external_SOURCE_only_allowlist']['path'])['allowed_external_source_files'])
    for name, expected in all_reviews.items():
        require(digest(name) == expected, 'Current independent review changed.')
        review = read_json(name)
        require(review['status'] in {'source_only_review_passed',
                'source_only_configuration_coherence_review_complete_pending_full_code_freeze'},
                'Individual review does not pass its declared SOURCE scope.')
        if name in LEGACY_INDEPENDENCE_IDENTITIES:
            # These immutable early reports declare the separate reviewer in
            # their identity text rather than in the later boolean schema.
            # Check that exact bound declaration; do not mutate the originals
            # or default an absent independence field to a pass.
            field, identity = LEGACY_INDEPENDENCE_IDENTITIES[name]
            require(review[field] == identity,
                    'Original separate-reviewer declaration differs.')
        else:
            require(review['independent_from_implementer'] is True,
                    'Individual review not distinct from its implementer.')
        for member, wanted in review['file_sha256'].items():
            name_check(member)
            if Path(member).is_absolute():
                require(member in external_allowlist or member == '/Users/jiajia/.agents/skills/research/SKILL.md',
                        'Unlisted external SOURCE/skill dependency.')
            actual = digest(member)
            require(actual['sha256'] == wanted, 'Source/artificial review member changed: ' + member)
            require(member not in proof_bindings or proof_bindings[member] == actual,
                    'Conflicting independent member identities.')
            proof_bindings[member] = actual
        proof_bindings[name] = expected
        mapping = review.get('reviewed_source_sha256_by_path', review.get('reviewed_source_sha256'))
        if isinstance(mapping, dict):
            source_review_sources.update((member, value) for member, value in mapping.items())
        elif isinstance(mapping, str):
            source_review_sources.update((member, wanted) for member, wanted in review['file_sha256'].items()
                                        if member.startswith('src/') and wanted == mapping)
    source_bindings = dict(core['execution_sources'])
    # The ten unchanged scientific modules retain their prior exact-byte
    # independent reviews; the two new routing/runtime modules require the
    # NEW different full-source review. Older reviews do not authorize them.
    legacy_routes = {
        'src/sccomics_supervised_stages_v2.py': '6bba0ec13e93490d680afb8b306cf9f997cf5d7ed86dabc241aaafc428702308',
        'src/sccomics_supervised_runtime_v2.py': 'a46bd4fef4f8c8d9e2afe93cfdb96e286845f86a49c8113e43e847cec3a8f78a',
    }
    for name, wanted in legacy_routes.items():
        require(source_bindings.pop(name)['sha256'] == wanted,
                'Original v2 routing provenance differs.')
    for name in ('src/sccomics_supervised_stages_v3.py',
                 'src/sccomics_supervised_runtime_v3.py'):
        expected = import_review['bound_source_files'][name]
        require(expected['duty'] == 'code', 'New route source duty differs.')
        source_bindings[name] = {key: expected[key] for key in ('sha256', 'size_bytes')}
    require(len(source_bindings) == 12, 'Complete execution source population differs.')
    for name, expected in source_bindings.items():
        require(digest(name) == expected and (name, expected['sha256']) in source_review_sources,
                'Current execution source lacks its different-implementer exact-byte review.')
    for name in SUPPLEMENTS + [new_report_name, str(Path(__file__).relative_to(ROOT))]:
        proof_bindings[name] = digest(name)
    for name, expected in observed['actual_output_bindings'].items():
        name_check(name)
        require(digest(name) == expected, 'Actual compatibility output changed.')
        proof_bindings[name] = expected
    proof_bindings[RUNTIME_RECEIPT] = digest(RUNTIME_RECEIPT)
    require(len(CONTROL_NAMES) == 23 and len(set(CONTROL_NAMES)) == 23,
            'Exact control population schema differs.')
    require(not any((ROOT / name).exists() for name in (REVIEW_OUTPUT, FREEZE_OUTPUT, GATE_OUTPUT, FIT_OUTPUT)),
            'Existing actual authority/output must not be overwritten.')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    aggregate = {
        'status': 'source_only_review_passed',
        'identity': 'ROOT_AGGREGATION_OF_GENUINELY_DIFFERENT_MODULE_SOURCE_REVIEWS_NOT_A_NEW_HUMAN_EXPERT_REVIEW',
        'created_at_utc': now,
        'independent_from_implementer': True,
        'independence_meaning': 'Each captured execution module is covered by a review from a different implementer; root aggregates and verifies those existing reports and does not claim personal independence from every module.',
        'all_individual_source_reviews_and_artificial_checks_verified': True,
        'reviewed_source_sha256_by_path': {name: value['sha256'] for name, value in source_bindings.items()},
        'individual_source_review_bindings': all_reviews,
        'supplemental_evidence_bindings': proof_bindings,
        'actual_compatibility_root_receipt_sha256': args.compatibility_root_receipt_sha256,
        'actual_models_or_scientific_performance_certified': False,
        'paper_peer_review_humanizer_or_ranking_count': 0,
        'old_helper879_and_af93_import_boundary_pass_claim': False,
        'finite_runtime_OS_GPU_or_hostile_ABA_atomicity_claim': False,
        'loaded_definitions_all_reexecuted_claim': False,
    }
    aggregate_bytes = serialized(aggregate)
    aggregate_binding = {'sha256': hashlib.sha256(aggregate_bytes).hexdigest(),
                         'size_bytes': len(aggregate_bytes), 'duty': 'control'}
    inputs = {name: dict(expected, duty='code') for name, expected in source_bindings.items()}
    for name in CONTROL_NAMES:
        inputs[name] = aggregate_binding if name == REVIEW_OUTPUT else dict(digest(name), duty='control')
    require(len(inputs) == 35, 'Final direct source/control closure differs.')
    space = shutil.disk_usage(ROOT)
    # This is a current capacity observation, not a fit/JSON/pair feasibility
    # claim. A whole-family resource failure remains a failed experiment.
    require(space.free >= 4_399_364_760 + 2 * 1024**3,
            'Known float32 checkpoint payload plus basic disk reserve unavailable.')
    freeze = {
        'status': 'actual_complete_scientific_freeze', 'complete': True,
        'synthetic_fixture': False, 'created_at_utc': now,
        'runtime': RUNTIME, 'native_split_ids': SPLITS,
        'implemented_final_scoring_and_independent_replay': True,
        'input_bindings': inputs,
        'material_binding_expansion_receipt_metadata': material['input_material_bindings_from_fixed_original_receipts'],
        'annotations_before_semantic_stage': 'Original acquisition receipt SHA, strict canonical path and current size only; no fresh held-annotation content hash or semantic interpretation is claimed at this boundary.',
        'fixed_scientific_family': {'seeds': [20261013, 20261014, 20261015],
            'epochs_per_fit': 10, 'fits': 15, 'all_epoch_checkpoints': 150,
            'complete_dev_source_probability_records': 15000,
            'complete_test_source_records_before_Gold': 1500,
            'post_test_adaptation_or_seed_selection': False},
        'resource_failure_policy': 'Stop complete family with retained partial trajectories; no denominator, cap, seed, checkpoint dtype, threshold or architecture change after outputs.',
        'observed_free_disk_bytes': space.free,
        'known_checkpoint_float32_tensor_payload_bytes': 4_399_364_760,
        'complete_pair_output_or_final_export_feasibility_certified': False,
        'scientific_fit_or_performance_already_completed': False,
    }
    freeze_bytes = serialized(freeze)
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    gate = {'status': 'root_authorized_actual_supervised_stage_run',
            'freeze_sha256': freeze_sha, 'stage': 'full_supervised_pipeline',
            'output_relative_path': FIT_OUTPUT, 'actual_fitting_allowed': True,
            'paid_API_or_credentials_allowed': False, 'synthetic_fixture': False,
            'test_Gold_semantics_allowed_after_complete_source_barrier': True}
    gate_bytes = serialized(gate)
    # All current SOURCE/review/actual compatibility bytes must remain fixed
    # before the separate gate is saved. Scientific material is validated by
    # the captured runner according to its unopened-annotation policy.
    for name, expected in proof_bindings.items():
        require(digest(name) == expected, 'Proof changed during root assembly.')
    for name, expected in inputs.items():
        if name != REVIEW_OUTPUT:
            require(digest(name) == {key: expected[key] for key in ('sha256', 'size_bytes')},
                    'Direct closure changed during root assembly.')
    for name, payload in ((REVIEW_OUTPUT, aggregate_bytes), (FREEZE_OUTPUT, freeze_bytes), (GATE_OUTPUT, gate_bytes)):
        with (ROOT / name).open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(raw(name) == payload, 'Saved root authority bytes differ.')
    print(json.dumps({'review_sha256': aggregate_binding['sha256'],
                      'freeze_sha256': freeze_sha,
                      'final_gate_sha256': hashlib.sha256(gate_bytes).hexdigest(),
                      'direct_binding_count': 35,
                      'material_receipt_expansion_count': 3007,
                      'actual_fitting_completed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
