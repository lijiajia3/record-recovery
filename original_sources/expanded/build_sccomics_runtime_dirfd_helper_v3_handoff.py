"""Bind SOURCE, receipt JSON and owned artificial alias-repair evidence only."""
from __future__ import annotations
import hashlib
import json
import os
import stat
from pathlib import Path

PREFIX = 'research/round4_sccomics_runtime_dirfd_helpers_v3_inputs'
REPORT = 'research/round4_sccomics_runtime_dirfd_helper_v3_implementation'
OLD_FLAT = 'research/round4_sccomics_runtime_dirfd_helpers_v2_inputs/STABLE_HELPER_HANDOFF/implementation_delivery_manifest.json'
FINDINGS = 'research/round4_sccomics_runtime_dirfd_v2_independent_review_inputs'


def relative(name):
    if type(name) is not str or not name or '\\' in name or '\x00' in name:
        raise ValueError('explicit relative artifact required')
    p = Path(name)
    if p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('artifact traversal rejected')
    return p


def capture(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('artifact aliases rejected')
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('regular SOURCE/metadata artifact required')
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
        after = path.stat(follow_symlinks=False)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(start, k) != getattr(end, k) or getattr(start, k) != getattr(after, k) for k in keys) or size != start.st_size:
            raise ValueError('artifact changed during capture')
        return {'sha256': h.hexdigest(), 'size_bytes': size}, b''.join(chunks)
    finally:
        os.close(fd)


def main():
    root = Path(__file__).absolute().parents[1]
    out = root / PREFIX / 'STABLE_HELPER_HANDOFF'
    if out.exists() or any(p.is_symlink() for p in (out.parent, *out.parent.parents)):
        raise ValueError('fresh nonsymlink output required')
    out.mkdir()
    files, raw = {}, {}

    def add(name, expected=None):
        binding, data = capture(root / relative(name))
        if expected is not None and binding != expected:
            raise ValueError('saved artifact hash/size differs: ' + name)
        if name in files and binding != files[name]:
            raise ValueError('artifact changed: ' + name)
        files[name], raw[name] = binding, data
        return binding

    add(OLD_FLAT, {'sha256': 'f0944bd91e17e1634ef76df0bf560dc79a7578e0af1870fa5ed75f69081cf356', 'size_bytes': 28393})
    old = json.loads(raw[OLD_FLAT])
    if len(old['bindings']) != 107 or old['transitive_declared_actual_runtime_or_SC_payloads_read'] is not False:
        raise ValueError('retained old finite SOURCE handoff differs')
    for b in old['bindings']:
        add(b['path'], {k: b[k] for k in ('sha256', 'size_bytes')})
    add('research/round4_sccomics_runtime_dirfd_helper_v2_implementation.json')
    add('research/round4_sccomics_runtime_dirfd_helpers_v2_inputs/STABLE_HELPER_HANDOFF/actual_flat_binding_recheck.json')
    histories = []
    sources = None
    for version in ('INVENTED_normal_v1', 'INVENTED_O_v1', 'INVENTED_PYTHONOPTIMIZE_v1'):
        folder = PREFIX + '/' + version
        name = folder + '/output_manifest.json'
        add(name)
        for b in json.loads(raw[name])['bindings']:
            add(folder + '/' + str(relative(b['path'])), {k: b[k] for k in ('sha256', 'size_bytes')})
        execution = folder + '/actual_execution_receipt.json'
        add(execution)
        record = json.loads(raw[execution])
        result_name = folder + '/actual_synthetic_results.json'
        result = json.loads(raw[result_name])
        if (result.get('total'), result.get('passed'), result.get('failed')) != (64, 64, 0) or record.get('captured_exit_code') != 0:
            raise ValueError('artificial execution history differs')
        if result.get('actual_SC_text_annotation_archive_model_weight_cache_prediction_fit_or_scientific_score') is not False or result.get('actual_installed_runtime_population_byte_hash_or_real_six_library_import_smoke') is not False:
            raise ValueError('artificial source-only scope differs')
        if record['actual_result_sha256'] != files[result_name]['sha256']:
            raise ValueError('mode execution receipt result differs')
        if sources is not None and sources != result['source_bindings']:
            raise ValueError('source changed between modes')
        sources = result['source_bindings']
        histories.append({'directory': version, 'captured_execution': record, 'result': files[result_name]})
    for name, b in sources.items():
        add(name, b)
    for name in ('initial_af93_independent_source_findings.md', 'initial_af93_independent_execution_history.json',
                 'initial_af93_handoff_actual_byte_check.json', 'reviewer_external_alias_failure.py',
                 'reviewer_external_parent_alias_failure.py', 'reviewer_external_suffix_alias_failure.py',
                 'external_alias_before_source.py.txt', 'external_alias_actual_failure.json',
                 'external_parent_alias_actual_failure.json', 'external_suffix_alias_actual_failure.json',
                 'external_alias_actual_failure.log', 'external_parent_alias_actual_failure.log',
                 'external_suffix_alias_actual_failure.log'):
        add(FINDINGS + '/' + name)
    for ext in ('.md', '.json'):
        add('research/round4_sccomics_runtime_import_compatibility_v3_source_only_contract' + ext)
    contract = json.loads(raw['research/round4_sccomics_runtime_import_compatibility_v3_source_only_contract.json'])
    if contract['contract_md_sha256'] != files['research/round4_sccomics_runtime_import_compatibility_v3_source_only_contract.md']['sha256']:
        raise ValueError('prospective contract MD/JSON differ')
    if files['research/round4_sccomics_supervised_runtime_receipt.json']['sha256'] != 'dd5e65a4134ab97d71454907b71bf4aea3f7654dbbc66291dbd5ae38344d9cb4':
        raise ValueError('canonical receipt JSON changed')
    add(REPORT + '.md')
    builder = 'src/build_sccomics_runtime_dirfd_helper_v3_handoff.py'
    add(builder)
    (out / 'builder.before.py.txt').write_bytes(raw[builder])
    add(str((out / 'builder.before.py.txt').relative_to(root)), files[builder])
    for name in sources:
        p = out / 'before' / relative(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw[name])
        add(str(p.relative_to(root)), files[name])
    for name, b in files.items():
        if capture(root / name)[0] != b:
            raise ValueError('final SOURCE/artifact consistency failed: ' + name)
    manifest = {'identity': 'EXPLICIT_SOURCE_METADATA_AND_OWNED_ARTIFICIAL_ALIAS_REPAIR_HANDOFF_NOT_FIT_OR_IMPORT_CERTIFICATE',
                'bindings': [{'path': n, **b} for n, b in sorted(files.items())],
                'all_listed_hash_sizes_actually_reopened_and_final_checked': True,
                'transitive_actual_runtime_SC_model_payloads_read': False, 'fit_authority': False}
    mp = out / 'implementation_delivery_manifest.json'
    mp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    report = {'identity': 'SOURCEONLY_V3_ALIAS_REPAIR_IMPLEMENTATION_NOT_DIFFERENT_AGENT_REVIEW',
              'status': 'stable_source_ready_for_different_agent_review', 'current_source_bindings': sources,
              'implementation_report_md': files[REPORT + '.md'],
              'prospective_v3_contract_json': files['research/round4_sccomics_runtime_import_compatibility_v3_source_only_contract.json'],
              'prospective_v3_contract_md': files['research/round4_sccomics_runtime_import_compatibility_v3_source_only_contract.md'],
              'flat_manifest': {'path': str(mp.relative_to(root)), **capture(mp)[0], 'binding_count': len(files)},
              'actual_artificial_three_modes': histories, 'all_three_actual_exit_codes_zero': True,
              'prior_af93_independent_pass_blocked': True, 'prior_three_actual_INVENTED_bypasses_preserved': True,
              'old_af93_contract_107_handoff_controller_runtime_preparer_receipt_unchanged': True,
              'actual_six_installed_library_import_or_runtime_repreparation_or_SC_NN_ANN_API_scores': False,
              'different_agent_review_passed_by_this_report': False, 'actual_fitting_or_final_freeze_permission': False,
              'finite_Python_boundary_not_atomic_hostile_OS_ABA_or_global_native_sandbox': True}
    rp = root / (REPORT + '.json')
    if rp.exists():
        raise ValueError('retain old report')
    rp.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    result = {'identity': 'ACTUAL_SOURCE_METADATA_AND_ARTIFICIAL_BINDING_BUILD', 'passed': True,
              'binding_count': len(files), 'manifest': capture(mp)[0], 'report_json': capture(rp)[0], 'actual_fit_authority': False}
    (out / 'actual_builder_result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
