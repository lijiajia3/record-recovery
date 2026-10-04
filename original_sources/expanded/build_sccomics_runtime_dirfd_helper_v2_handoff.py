"""Bind explicit SOURCE, receipt JSON and owned artificial dir-fd evidence only."""
from __future__ import annotations
import hashlib
import json
import os
import stat
from pathlib import Path

PREFIX = 'research/round4_sccomics_runtime_dirfd_helpers_v2_inputs'
REPORT = 'research/round4_sccomics_runtime_dirfd_helper_v2_implementation'
OLD_FAILURE = 'research/round4_sccomics_supervised_stages_v2_runtime_sourceonly_inputs/ACTUAL_IMPORT_CHECK_202610031006'


def relative(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('explicit relative artifact path required')
    p = Path(name)
    if p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('artifact traversal rejected')
    return p


def checked(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('artifact symlink rejected')
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('regular artifact required')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        start = os.fstat(fd)
        if not stat.S_ISREG(start.st_mode) or (start.st_dev, start.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('opened artifact identity changed')
        h = hashlib.sha256()
        chunks = []
        size = 0
        while chunk := os.read(fd, 1048576):
            h.update(chunk)
            chunks.append(chunk)
            size += len(chunk)
        end = os.fstat(fd)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        after = path.stat(follow_symlinks=False)
        if any(getattr(start, k) != getattr(end, k) or getattr(start, k) != getattr(after, k) for k in keys) or size != start.st_size:
            raise ValueError('artifact changed during capture')
        return {'sha256': h.hexdigest(), 'size_bytes': size}, b''.join(chunks)
    finally:
        os.close(fd)


def main():
    root = Path(__file__).absolute().parents[1]
    out = root / PREFIX / 'STABLE_HELPER_HANDOFF'
    if out.exists():
        raise ValueError('retain old handoff; fresh output required')
    if any(p.is_symlink() for p in (out.parent, *out.parent.parents)):
        raise ValueError('output ancestor symlink rejected')
    out.mkdir()
    files = {}
    captured = {}

    def add(name, expected=None):
        path = root / relative(name)
        b, raw = checked(path)
        if expected is not None and b != expected:
            raise ValueError('saved binding differs: ' + name)
        if name in files and files[name] != b:
            raise ValueError('artifact changed: ' + name)
        files[name], captured[name] = b, raw
        return b

    final_name = PREFIX + '/INVENTED_v3/actual_synthetic_results.json'
    add(final_name)
    final = json.loads(captured[final_name])
    if (final.get('total'), final.get('passed'), final.get('failed')) != (48, 48, 0):
        raise ValueError('final artificial counts differ')
    if final.get('actual_installed_runtime_population_byte_hash_or_real_six_library_import_smoke') is not False or final.get('actual_SC_text_annotation_archive_model_weight_cache_prediction_fit_or_scientific_score') is not False:
        raise ValueError('source-only artificial scope differs')
    for name, b in final['source_bindings'].items():
        add(name, b)
    histories = []
    for version, expected in (('INVENTED_v1', (46, 45, 1)), ('INVENTED_v2', (46, 46, 0)), ('INVENTED_v3', (48, 48, 0))):
        folder = PREFIX + '/' + version
        mn = folder + '/output_manifest.json'
        add(mn)
        manifest = json.loads(captured[mn])
        for b in manifest['bindings']:
            add(folder + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
        result = json.loads(captured[folder + '/actual_synthetic_results.json'])
        if (result.get('total'), result.get('passed'), result.get('failed')) != expected:
            raise ValueError('artificial history counts changed')
        histories.append({'directory': version, 'total': expected[0], 'passed': expected[1], 'failed': expected[2], 'captured_process_exit_code': 1 if expected[2] else 0})

    fixed = (
        'src/check_sccomics_runtime_imports_source_only.py',
        'src/prepare_sccomics_supervised_runtime_v2_source_only.py',
        'research/round4_sccomics_supervised_runtime_receipt.json',
        'research/round4_sccomics_runtime_import_compatibility_source_review.json',
        'research/round4_sccomics_runtime_sourceonly_helpers_independent_source_review.md',
        'research/round4_sccomics_runtime_sourceonly_helpers_independent_source_review.json',
        'research/round4_sccomics_runtime_sourceonly_helpers_implementation.md',
        'research/round4_sccomics_runtime_sourceonly_helpers_implementation.json',
        'research/round4_sccomics_runtime_sourceonly_helpers_inputs/STABLE_HELPER_HANDOFF/implementation_delivery_manifest.json',
        'research/round4_sccomics_runtime_dirfd_independent_review_inputs/reviewer_dirfd_failure_and_contract.py',
        'research/round4_sccomics_runtime_dirfd_independent_review_inputs/plan_before_execution.json',
        'research/round4_sccomics_runtime_dirfd_independent_review_inputs/actual_results.json',
        'research/round4_sccomics_runtime_dirfd_independent_review_inputs/actual_execution.log',
        'research/round4_sccomics_runtime_dirfd_independent_review_inputs/actual_execution_receipt.json',
        PREFIX + '/INITIAL_BUILDER_MANIFEST_KEY_FAILURE/builder.before.py.txt',
        PREFIX + '/INITIAL_BUILDER_MANIFEST_KEY_FAILURE/actual_failure_receipt.json',
        PREFIX + '/INITIAL_BUILDER_MANIFEST_KEY_FAILURE/actual_failure.log',
    )
    for name in fixed:
        add(name)
    for base in ('research/round4_sccomics_runtime_import_compatibility_v2_source_only_contract',
                 'research/round4_sccomics_runtime_import_compatibility_source_only_contract',
                 'research/round4_sccomics_supervised_stages_v2_runtime_import_compatibility_prospective_amendment'):
        add(base + '.md')
        add(base + '.json')
    failure_manifest_name = OLD_FAILURE + '/output_manifest.json'
    add(failure_manifest_name)
    for b in json.loads(captured[failure_manifest_name])['bindings']:
        add(OLD_FAILURE + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
    failure_name = OLD_FAILURE + '/actual_import_compatibility_result.json'
    add(failure_name)
    if json.loads(captured[failure_name]).get('passed') is not False:
        raise ValueError('retained real failure status changed')
    canonical = 'research/round4_sccomics_supervised_runtime_receipt.json'
    if files[canonical]['sha256'] != 'dd5e65a4134ab97d71454907b71bf4aea3f7654dbbc66291dbd5ae38344d9cb4':
        raise ValueError('canonical JSON receipt changed')
    retained = 'src/prepare_sccomics_supervised_runtime_v2_source_only.py'
    if files[retained]['sha256'] != '940abd90d0de917d7b6db8ebd69d021b9dadca7f21016f4fe56154f098f75810':
        raise ValueError('retained original940 changed')
    add(REPORT + '.md')
    builder = 'src/build_sccomics_runtime_dirfd_helper_v2_handoff.py'
    add(builder)
    (out / 'builder.before.py.txt').write_bytes(captured[builder])
    add(str((out / 'builder.before.py.txt').relative_to(root)))
    for name in final['source_bindings']:
        p = out / 'before' / relative(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(captured[name])
        add(str(p.relative_to(root)), files[name])
    for name, b in files.items():
        if checked(root / name)[0] != b:
            raise ValueError('final SOURCE/artificial binding changed: ' + name)
    flat = {'identity': 'SOURCE_RECEIPT_JSON_AND_OWNED_ARTIFICIAL_DIRFD_HANDOFF_NOT_ACTUAL_RUNTIME_IMPORT_OR_FIT_CERTIFICATE',
            'bindings': [{'path': n, **b} for n, b in sorted(files.items())],
            'all_listed_hash_sizes_actually_reopened_and_final_checked': True,
            'transitive_declared_actual_runtime_or_SC_payloads_read': False,
            'fit_or_final_freeze_permission': False}
    manifest_path = out / 'implementation_delivery_manifest.json'
    manifest_path.write_text(json.dumps(flat, sort_keys=True, indent=2) + '\n')
    report = {'identity': 'SOURCEONLY_DIRFD_HELPER_IMPLEMENTATION_NOT_DIFFERENT_AGENT_REVIEW',
              'status': 'stable_source_ready_for_different_agent_review',
              'current_source_bindings': final['source_bindings'],
              'new_prospective_contract_json': files['research/round4_sccomics_runtime_import_compatibility_v2_source_only_contract.json'],
              'new_prospective_contract_md': files['research/round4_sccomics_runtime_import_compatibility_v2_source_only_contract.md'],
              'implementation_report_md': files[REPORT + '.md'],
              'flat_delivery_manifest': {'path': str(manifest_path.relative_to(root)), **checked(manifest_path)[0], 'binding_count': len(files)},
              'actual_artificial_history': histories,
              'initial_failed_check': 'descriptor test setup wrote an undeclared artificial file after audit activation; production policy rejected it; intended descriptor counterexample not reached',
              'production_helper_same_af93_source_in_all_three_runs': True,
              'initial_handoff_builder_manifest_key_failure_retained': True,
              'canonical_runtime_receipt_JSON_only_binding': files[canonical],
              'root_reported_actual_runtime_preparation': {'files': 11969, 'bytes': 590219888, 'captured_exit_code': 0},
              'retained_old_actual_import_failure': {'path': failure_name, **files[failure_name], 'failed_before_six_entry_imports': True},
              'new_helper_real_installed_imports_or_repreparation_run': False,
              'actual_SC_text_annotation_archive_model_weight_cache_prediction_fit_API_or_new_scientific_statistic': False,
              'different_source_review_passed_by_this_report': False,
              'actual_complete_freeze_gate_permission': False,
              'finite_Python_audit_not_global_native_OS_IO_or_credential_sandbox': True}
    rp = root / (REPORT + '.json')
    if rp.exists():
        raise ValueError('retain existing report')
    rp.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    result = {'identity': 'ACTUAL_SOURCE_METADATA_ARTIFICIAL_HANDOFF_BINDING_CHECK', 'passed': True,
              'binding_count': len(files), 'manifest': checked(manifest_path)[0], 'report_json': checked(rp)[0],
              'no_transitive_actual_runtime_SC_or_model_payload_read': True, 'fit_authority': False}
    (out / 'actual_builder_result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
