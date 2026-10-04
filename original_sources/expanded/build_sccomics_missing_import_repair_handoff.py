"""Bind explicit SOURCE/metadata and completed owned invented output populations."""
from __future__ import annotations
import hashlib
import json
import os
import stat
import time
from pathlib import Path

BASE = 'research/round4_sccomics_runtime_missing_helpers_v4_inputs'
FULL = 'research/round4_sccomics_supervised_stages_v3_sourceonly_inputs/INVENTED_v1'
REPORT = 'research/round4_sccomics_runtime_missing_import_repair_implementation'
FAILURE = 'research/round4_sccomics_supervised_stages_v2_runtime_sourceonly_inputs/ACTUAL_IMPORT_CHECK_202610031852'
OLD = 'research/round4_sccomics_runtime_dirfd_helpers_v3_inputs/STABLE_HELPER_HANDOFF/implementation_delivery_manifest.json'


def relative(name):
    if type(name) is not str or not name or '\\' in name or '\x00' in name:
        raise ValueError('explicit relative SOURCE/artificial path required')
    if Path(name).is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('artifact traversal rejected')
    return Path(name)


def capture(path, *, return_raw=False):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('SOURCE/artificial alias rejected')
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('regular SOURCE/artificial file required')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        start = os.fstat(fd)
        if not stat.S_ISREG(start.st_mode) or (start.st_dev, start.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('opened identity changed')
        h = hashlib.sha256()
        parts = []
        size = 0
        while b := os.read(fd, 1048576):
            h.update(b)
            size += len(b)
            if return_raw:
                parts.append(b)
        end = os.fstat(fd)
        after = path.stat(follow_symlinks=False)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(start, k) != getattr(end, k) or getattr(start, k) != getattr(after, k) for k in keys) or size != start.st_size:
            raise ValueError('file changed during capture')
        return {'sha256': h.hexdigest(), 'size_bytes': size}, b''.join(parts) if return_raw else None
    finally:
        os.close(fd)


def main():
    root = Path(__file__).absolute().parents[1]
    destination = root / BASE / 'STABLE_SOURCE_HANDOFF'
    if destination.exists() or any(p.is_symlink() for p in (destination.parent, *destination.parent.parents)):
        raise ValueError('fresh nonsymlink SOURCE output required')
    # Read completion receipt before making any handoff directory. A running
    # PID, intermediate population or prior result cannot certify completion.
    full_execution_path = root / FULL / 'actual_execution_receipt.json'
    _, completion_raw = capture(full_execution_path, return_raw=True)
    completion = json.loads(completion_raw)
    if completion.get('captured_process_exit_code') != 0 or completion.get('session_id') != 13862 or completion.get('actual_fitting_or_SC_payloads_read') is not False:
        raise ValueError('actual full artificial process completion required')
    destination.mkdir()
    files = {}

    def add(name, expected=None):
        binding, _ = capture(root / relative(name))
        if expected is not None and binding != expected:
            raise ValueError('saved artifact hash/size differs: ' + name)
        if name in files and files[name] != binding:
            raise ValueError('artifact changed: ' + name)
        files[name] = binding
        return binding

    def document(name):
        add(name)
        _, raw = capture(root / name, return_raw=True)
        return json.loads(raw)

    old = document(OLD)
    if files[OLD]['sha256'] != 'cbacb95426f4f2fe8f9b3784f369ab6529fc1221ed44e3a595fdbf4fbc670a7c' or len(old['bindings']) != 198:
        raise ValueError('retained prior handoff differs')
    for b in old['bindings']:
        add(b['path'], {k: b[k] for k in ('sha256', 'size_bytes')})
    add('research/round4_sccomics_runtime_dirfd_helper_v3_implementation.json')
    sources = {}
    modes = []
    for version in ('INVENTED_normal_v1', 'INVENTED_O_v1', 'INVENTED_PYTHONOPTIMIZE_v1'):
        folder = BASE + '/' + version
        manifest = document(folder + '/output_manifest.json')
        for b in manifest['bindings']:
            add(folder + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
        result = document(folder + '/actual_synthetic_results.json')
        execution = document(folder + '/actual_execution_receipt.json')
        if (result.get('total'), result.get('passed'), result.get('failed')) != (72, 72, 0) or execution.get('captured_exit_code') != 0:
            raise ValueError('three-mode artificial result incomplete')
        if execution['result_sha256'] != files[folder + '/actual_synthetic_results.json']['sha256']:
            raise ValueError('mode completion result differs')
        if result['actual_SC_text_annotation_archive_model_weight_cache_prediction_fit_or_scientific_score'] is not False or result['actual_installed_runtime_population_byte_hash_or_real_six_library_import_smoke'] is not False:
            raise ValueError('artificial source-only scope differs')
        for name, b in result['source_bindings'].items():
            add(name, b)
            if name in sources and sources[name] != b:
                raise ValueError('source changed across modes')
            sources[name] = b
        modes.append({'directory': version, 'execution': execution, 'result': files[folder + '/actual_synthetic_results.json']})
    finder_history = BASE + '/INITIAL_BINDER_FINDER_FAILURE'
    for leaf in ('builder.before.py.txt', 'actual_failure_receipt.json', 'actual_full_population_reopen.json'):
        add(finder_history + '/' + leaf)
    finder = document(finder_history + '/actual_full_population_reopen.json')
    fm = document(FULL + '/output_manifest.json')
    if finder['original_manifest_sha256'] != files[FULL + '/output_manifest.json']['sha256'] or finder['substantive_mismatch_count'] != 0 or finder['Finder_metadata_excluded_exact_basename'] != '.DS_Store':
        raise ValueError('substantive vs Finder reopen proof differs')
    included = 0
    for b in fm['files']:
        if Path(b['path']).name == '.DS_Store':
            continue
        add(FULL + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
        included += 1
    if included != finder['substantive_file_count']:
        raise ValueError('substantive fake population count differs')
    full = document(FULL + '/actual_synthetic_results.json')
    add(FULL + '/actual_execution_receipt.json')
    if (full.get('total'), full.get('passed'), full.get('failed')) != (92, 92, 0) or full.get('real_HF_or_task_fit_executed') is not False or full.get('real_SC_annotations_weights_cache_or_predictions_read') is not False:
        raise ValueError('complete normal+zero artificial result differs')
    for name, b in full['source_bindings'].items():
        add(name, {k: b[k] for k in ('sha256', 'size_bytes')})
        sources[name] = files[name]
    full_helper = 'src/check_sccomics_supervised_stages_v3_synthetic.py'
    add(full_helper)
    if files[full_helper]['sha256'] != full['helper_sha256']:
        raise ValueError('full helper source differs')
    invariant = document(BASE + '/standalone_source_invariant_check.json')
    for name, b in invariant['source_bindings'].items():
        add(name, b)
    add(BASE + '/static_scientific_source_invariants.json')
    add(BASE + '/scientific_repair_and_import_gate_source_binding.json')
    add(BASE + '/prospective_test_execution_status.json')
    add('research/round4_sccomics_runtime_missing_import_fixture_scope_clarification.md')
    for leaf in ('failed_builder.before.py.txt', 'actual_failed_closed_receipt.json', 'implementation_delivery_manifest.json'):
        add(BASE + '/SECOND_BINDER_METADATA_CHANGE_FAILURE/' + leaf)
    for leaf in ('failed_builder.before.py.txt', 'actual_failed_closed_receipt.json', 'implementation_delivery_manifest.json', 'actual_published_manifest_metadata_diagnostic.json'):
        add(BASE + '/THIRD_BINDER_METADATA_CHANGE_FAILURE/' + leaf)
    for base in ('research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum',
                 'research/round4_sccomics_runtime_import_compatibility_v4_source_only_contract'):
        add(base + '.md')
        add(base + '.json')
    failed = document(FAILURE + '/output_manifest.json')
    for b in failed['bindings']:
        add(FAILURE + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
    actual_failure = document(FAILURE + '/actual_import_compatibility_result.json')
    if actual_failure['passed'] is not False or actual_failure['error_type'] != 'StageIntegrityError' or 'msvcrt' not in actual_failure['error']:
        raise ValueError('retained real absent-module failure differs')
    add(REPORT + '.md')
    builder = 'src/build_sccomics_missing_import_repair_handoff.py'
    add(builder)
    _, builder_raw = capture(root / builder, return_raw=True)
    (destination / 'builder.before.py.txt').write_bytes(builder_raw)
    add(str((destination / 'builder.before.py.txt').relative_to(root)), files[builder])
    for name in sources:
        _, raw = capture(root / name, return_raw=True)
        p = destination / 'before' / relative(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        add(str(p.relative_to(root)), files[name])
    for name, b in files.items():
        if capture(root / name)[0] != b:
            raise ValueError('final explicit file binding differs: ' + name)
    flat = {'identity': 'SOURCE_PROTOCOL_RECEIPT_JSON_AND_COMPLETED_OWNED_ARTIFICIAL_POPULATIONS_ONLY',
            'bindings': [{'path': n, **b} for n, b in sorted(files.items())],
            'all_listed_files_actual_reopened_and_final_hash_size_checked': True,
            'actual_runtime_payload_reprepared_or_SC_model_gold_payloads_read': False,
            'actual_fit_or_import_authority': False}
    mp = destination / 'implementation_delivery_manifest.json'
    encoded = (json.dumps(flat, sort_keys=True, indent=2) + '\n').encode()
    expected_mp = {'sha256': hashlib.sha256(encoded).hexdigest(), 'size_bytes': len(encoded)}
    mp.write_bytes(encoded)
    # Settling only this newly published output's metadata is not an input or
    # annotation retry. The full Source/science capture guards remain strict.
    # The bytes must equal the exact pre-write serialized population, even
    # after the delay; an altered stable file cannot obtain a certificate.
    time.sleep(0.5)
    if capture(mp)[0] != expected_mp:
        raise ValueError('published manifest differs from exact serialized bytes')
    report = {'identity': 'MISSING_MODULE_RUNTIME_CONTROLLER_SOURCE_REPAIR_IMPLEMENTATION_NOT_INDEPENDENT_REVIEW',
              'status': 'stable_source_ready_for_different_agent_review', 'source_bindings': sources,
              'implementation_md': files[REPORT + '.md'],
              'flat_manifest': {'path': str(mp.relative_to(root)), **capture(mp)[0], 'binding_count': len(files)},
              'actual_three_mode_helper_results': modes,
              'actual_full_normal_zero_population': {'result': files[FULL + '/actual_synthetic_results.json'], 'process_completion': completion},
              'invariant_check': files[BASE + '/standalone_source_invariant_check.json'],
              'canonical_runtime_receipt_JSON_only': files['research/round4_sccomics_supervised_runtime_receipt.json'],
              'real_installed_six_import_or_SC_NN_training_API_statistics_or_actual_score': False,
              'old_source_contracts_receipts_reviews_and_failures_retained_unchanged': True,
              'original_full_manifest_retained_despite_Finder_only_staleness': True,
              'substantive_full_fake_files_verified': finder['substantive_file_count'],
              'Finder_metadata_exact_basename_excluded': '.DS_Store',
              'Finder_metadata_excluded_count': finder['Finder_metadata_count'],
              'Finder_metadata_changed_count_at_first_reopen': finder['Finder_metadata_changed_count'],
              'different_agent_SOURCE_passed_by_this_report': False, 'actual_fit_or_final_scientific_freeze_permission': False}
    rp = root / (REPORT + '.json')
    if rp.exists():
        raise ValueError('retain old implementation report')
    rp.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    result = {'identity': 'ACTUAL_SOURCE_ARTIFICIAL_HANDOFF_BUILD_NOT_AUTHORITY', 'passed': True,
              'flat_binding_count': len(files), 'manifest': capture(mp)[0], 'report_json': capture(rp)[0], 'actual_fit_authority': False}
    (destination / 'actual_builder_result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
