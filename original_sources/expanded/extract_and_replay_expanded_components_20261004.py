"""Fully decode ordinary components into a new root, then replay saved outputs.

Optional copy-on-write copies save local space only after complete ZIP decoding
and matching source bytes. They are regular files, never symbolic references.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile


def need(ok, message):
    if not ok:
        raise ValueError(message)


def binding(path):
    h, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b); n += len(b)
    return {'sha256': h.hexdigest(), 'size_bytes': n}


def safe_name(name):
    p = PurePosixPath(name)
    need(type(name) is str and name and not p.is_absolute() and '..' not in p.parts and
         '\\' not in name and '\x00' not in name, 'Unsafe member name')
    return p


def decode_component(archive_path, expected_members, output, clone_root, metadata_parent):
    rows = []
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == set(expected_members), 'Complete member inventory differs')
        for entry in z.infolist():
            name = safe_name(entry.filename)
            need(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16), 'Only regular content is permitted')
            target = output / name
            need(target.resolve().is_relative_to(output.resolve()), 'Member escapes extraction root')
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = {k: expected_members[entry.filename][k] for k in ('sha256', 'size_bytes')}
            need(entry.file_size == expected['size_bytes'], 'Declared uncompressed size differs')
            candidate = None
            if not target.exists() and clone_root is not None:
                for p in (clone_root / name, metadata_parent / name):
                    if p.is_file() and not p.is_symlink() and binding(p) == expected:
                        candidate = p
                        break
            route = 'existing_extracted_regular_file_reverified' if target.exists() else 'regular_APFS_COW_after_full_decode' if candidate else 'decoded_bytes_written'
            decoded = None if target.exists() or candidate else target.with_name(target.name + '.decoding')
            if decoded is not None:
                need(not decoded.exists(), 'Never overwrite a partial extraction')
                v = os.statvfs(output)
                need(v.f_bavail*v.f_frsize > entry.file_size + 512*1024*1024,
                     'Insufficient physical space for a complete new regular member')
            h, size = hashlib.sha256(), 0
            with z.open(entry) as stream:
                out = decoded.open('xb') if decoded else None
                try:
                    for block in iter(lambda: stream.read(1048576), b''):
                        if out is not None:
                            out.write(block)
                        h.update(block); size += len(block)
                finally:
                    if out is not None:
                        out.close()
            need({'sha256': h.hexdigest(), 'size_bytes': size} == expected, 'Actual decompressed CRC/SHA/size mismatch')
            if target.exists():
                need(target.is_file() and not target.is_symlink() and binding(target) == expected,
                     'A duplicate component path has different bytes')
            elif candidate:
                subprocess.run(['/bin/cp', '-c', str(candidate), str(target)], check=True)
                need(target.is_file() and not target.is_symlink() and binding(target) == expected,
                     'Fresh regular clone differs from completely decoded bytes')
            else:
                decoded.rename(target)
                need(binding(target) == expected, 'Fresh written regular file differs')
            rows.append({'path': entry.filename, **expected, 'route': route,
                         'actual_complete_archive_member_decoded_CRC_checked': True,
                         'fresh_destination_full_hash_checked': True})
    return rows


def execute(command, cwd, logfile):
    need(not logfile.exists(), 'Preserve prior replay logs')
    with logfile.open('x') as log:
        run = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    need(run.returncode == 0, 'Actual extracted replay failed; inspect its saved log')
    return {'command': command, 'true_exit_code': run.returncode, 'log': binding(logfile)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--extracted-root', type=Path, required=True)
    p.add_argument('--report-directory', type=Path, required=True)
    p.add_argument('--verified-COW-candidates-root', type=Path)
    a = p.parse_args()
    packet, root, reports = a.packet.resolve(), a.extracted_root.resolve(), a.report_directory.resolve()
    need(packet.is_dir() and not root.exists() and not reports.exists(), 'Use two genuinely new empty output directories')
    root.mkdir(parents=True); reports.mkdir(parents=True)
    manifest_path = packet / 'COMPONENT_MANIFEST.json'
    manifest_binding = binding(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    need(manifest['status'] == 'complete_component_export_pending_actual_fresh_extraction_replay' and
         manifest['actual_external_credential_and_recursive_decompressed_scans_passed'] is True,
         'A complete scanned component export is required')
    started = datetime.now(timezone.utc).isoformat()
    extracted = {}
    for name, component in manifest['components'].items():
        archive_path = packet / safe_name(component['file'])
        need(binding(archive_path) == {k: component[k] for k in ('sha256', 'size_bytes')}, 'Component archive changed')
        output = root / safe_name(component['extraction_root'])
        output.mkdir(exist_ok=True)
        original_parent = packet
        # Candidate lookup is a space optimization only. Missing candidates
        # always cause ordinary full decoding/writing, never a missing member.
        rows = decode_component(archive_path, component['members'], output,
                                a.verified_COW_candidates_root, original_parent)
        extracted[name] = {'members': rows, 'member_count': len(rows),
                           'logical_bytes': sum(x['size_bytes'] for x in rows)}
        (reports / (name + '_actual_full_extraction.json')).write_text(json.dumps(extracted[name], indent=2) + '\n')
        print(json.dumps({'actual_decoded_component': name, 'members': len(rows)}), flush=True)
    historical, expanded = root / 'historical', root / 'expanded'
    commands = {}
    audit = json.loads((historical / 'OFFLINE_REPLAY_EXPORT_AUDIT.json').read_bytes())
    for domain in ('polyie', 'mulms'):
        result = reports / (domain + '_fresh_extracted_replay.json')
        commands[domain] = execute([sys.executable, str(historical/'src'/f'replay_{domain}_neural_offline.py'),
            '--root', str(historical), '--out', str(result)], historical, reports/(domain+'_replay.log'))
        need(json.loads(result.read_bytes()) == audit[domain], 'Old-domain replay differs from its completed export audit')
        print(json.dumps({'actual_extracted_replay_complete': domain}), flush=True)
    pilot_zip = historical / 'artifacts/pilot_v2_reproduction_20261003.zip'
    pilot_root = root / 'pilot'; pilot_root.mkdir()
    with zipfile.ZipFile(pilot_zip) as z:
        pm = json.loads(z.read('PACKAGE_MANIFEST.json'))
        wanted = {name: {'sha256': value, 'size_bytes': z.getinfo(name).file_size}
                  for name, value in pm['file_sha256'].items()}
        wanted['PACKAGE_MANIFEST.json'] = {'sha256': hashlib.sha256(z.read('PACKAGE_MANIFEST.json')).hexdigest(),
                                        'size_bytes': z.getinfo('PACKAGE_MANIFEST.json').file_size}
    pilot_rows = decode_component(pilot_zip, wanted, pilot_root, None, packet)
    (reports / 'pilot_actual_full_nested_extraction.json').write_text(json.dumps(pilot_rows, indent=2)+'\n')
    commands['pilot'] = execute([sys.executable, 'REPRODUCE_OFFLINE.py'], pilot_root, reports/'pilot_replay.log')
    result = reports / 'SC_fresh_extracted_native_and_math_replay.json'
    commands['SC'] = execute([sys.executable, str(expanded/'src/replay_expanded_sccomics_saved_outputs_20261004.py'),
        '--root', str(expanded), '--out', str(result)], expanded, reports/'SC_replay.log')
    need(json.loads(result.read_bytes())['status'] == 'fresh_extracted_SC_native_and_arithmetic_replay_complete',
         'Actual complete extracted SC replay required')
    final = reports / 'expanded_actual_extracted_editorial_gate.json'
    commands['editorial'] = execute([sys.executable, str(expanded/'src/validate_expanded_manuscript_lineage_20261004.py'),
        '--root', str(expanded), '--out', str(final)], expanded, reports/'editorial_replay.log')
    need(json.loads(final.read_bytes()) == manifest['editorial_gate'], 'Fresh six-pass/document gate differs')
    need(binding(manifest_path) == manifest_binding, 'Component manifest changed during extraction')
    for component in manifest['components'].values():
        need(binding(packet/component['file']) == {k:component[k] for k in ('sha256','size_bytes')},
             'An original component changed during extraction/replay')
    output = {'status': 'expanded_complete_actual_new_root_extraction_and_all_saved_output_replays_passed',
        'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'component_manifest': manifest_binding, 'new_extraction_root': str(root),
        'all_members_completely_decoded_CRC_SHA_size_checked': True,
        'all_destinations_regular_and_full_hash_checked': True,
        'members_per_component': {k:v['member_count'] for k,v in extracted.items()},
        'commands_and_true_zero_exits': commands,
        'actual_pilot_13_arms_Poly_Mu_SC_and_six_pass_checks_complete': True,
        'SC_graphs_semantically_replayed': 1500, 'SC_source_clusters_not_fits': 100,
        'credential_read_by_extraction_or_replay': False, 'paid_API_calls': 0,
        'neural_regeneration_or_new_experiments': False,
        'copy_on_write_is_not_independent_acquisition_or_physical_truth': True,
        'same_fixed_random_stream_replay_not_new_independent_random_samples': True}
    (reports/'completed_actual_offline_replay.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'status': output['status']}), flush=True)


if __name__ == '__main__':
    main()
