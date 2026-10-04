"""Actually extract a completed packet and run its included scoring programs.

This wrapper is an execution record, not a replacement for either independent
domain implementation. It never imports training code or reads API credentials.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_verified(packet, root):
    assert root.is_dir() and not any(root.iterdir()), 'Use a new empty root'
    total = 0
    with zipfile.ZipFile(packet) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive member'
        for name in names:
            p = PurePosixPath(name)
            assert name and not p.is_absolute() and '..' not in p.parts
            assert '\\' not in name and '\x00' not in name
        header = archive.getinfo('PACKET_MANIFEST.json')
        assert header.file_size < 64 * 1024 * 1024, 'Unexpectedly large manifest'
        manifest = json.loads(archive.read(header))
        assert set(names) == set(manifest['files']) | {'PACKET_MANIFEST.json'}
        for entry in archive.infolist():
            assert not entry.is_dir(), 'Only regular content files are expected'
            assert not stat.S_ISLNK(entry.external_attr >> 16), 'Archive symlink'
            target = root / entry.filename
            assert target.resolve().is_relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            h, size = hashlib.sha256(), 0
            with archive.open(entry) as stream, target.open('xb') as output:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    output.write(block)
                    h.update(block)
                    size += len(block)
            assert size == entry.file_size
            if entry.filename != 'PACKET_MANIFEST.json':
                expected = manifest['files'][entry.filename]
                assert size == expected['bytes'] and h.hexdigest() == expected['sha256'], entry.filename
                total += size
    return manifest, {'content_files': len(manifest['files']),
        'members_including_manifest': len(names), 'verified_uncompressed_bytes': total,
        'actual_every_member_extracted_and_hash_size_checked': True}


def portable_images(root, manifest):
    root = root.resolve()
    original = (root / 'manuscript/manuscript.md').read_bytes().decode('utf-8')
    portable = (root / 'manuscript/portable_manuscript.md').read_bytes().decode('utf-8')
    old_lines, new_lines = original.splitlines(keepends=True), portable.splitlines(keepends=True)
    assert len(old_lines) == len(new_lines)
    images = []
    for old, new in zip(old_lines, new_lines):
        if '![' not in old:
            assert new == old, 'Non-image manuscript prose changed'
            continue
        match = re.fullmatch(r'!\[([^\]]*)\]\(<(\.\./figures/[^<>\n]+)>\)(\r?\n)?', new)
        assert match is not None, 'Portable image must have an unambiguous relative destination'
        assert old.split('](', 1)[0] == new.split('](', 1)[0], 'Image label changed'
        old_match = re.fullmatch(r'!\[[^\]]*\]\((<[^<>\n]+>|[^\s()]+)\)(\r?\n)?', old)
        assert old_match is not None, 'Original image syntax is ambiguous'
        old_destination = old_match.group(1)
        if old_destination.startswith('<'):
            old_destination = old_destination[1:-1]
        old_path = Path(old_destination)
        if old_path.is_absolute():
            old_target = old_path.resolve()
        elif old_destination.startswith('figures/'):
            old_target = (root / old_path).resolve()
        elif old_destination.startswith('../figures/'):
            old_target = (root / 'manuscript' / old_path).resolve()
        else:
            raise AssertionError('Unrecognized original image destination')
        target = (root / 'manuscript' / match.group(2)).resolve()
        assert old_target == target, 'Portable destination refers to a different figure'
        assert target.is_file() and target.is_relative_to(root / 'figures')
        name = target.relative_to(root).as_posix()
        assert digest(target) == manifest['files'][name]['sha256']
        images.append(name)
    assert len(images) == len(set(images)) == 7
    return {'all_seven_relative_images_resolve_from_extracted_manuscript_directory': True,
        'non_image_manuscript_prose_byte_identical': True, 'image_files': images}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('packet', type=Path)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    packet, report = args.packet.resolve(), args.report.resolve()
    assert packet.is_file() and not report.exists()
    assert report.parent.is_dir()
    start = datetime.now(timezone.utc).isoformat()
    packet_sha = digest(packet)
    prefix = report.with_suffix('')
    outputs = {kind: Path(str(prefix) + '_' + kind + '.json') for kind in ('polyie', 'mulms', 'editorial')}
    logs = {kind: Path(str(prefix) + '_' + kind + '.log') for kind in outputs}
    assert all(not q.exists() for q in [*outputs.values(), *logs.values()])
    with tempfile.TemporaryDirectory(prefix='completed_scope_fresh_replay_') as temporary:
        root = Path(temporary) / 'extracted'
        root.mkdir()
        manifest, extraction = extract_verified(packet, root)
        image_check = portable_images(root, manifest)
        stored = json.loads((root / 'OFFLINE_REPLAY_EXPORT_AUDIT.json').read_text())
        commands = {}
        for kind in ('polyie', 'mulms'):
            command = [sys.executable, str(root / 'src' / ('replay_' + kind + '_neural_offline.py')),
                '--root', str(root), '--out', str(outputs[kind])]
            commands[kind] = command
            with logs[kind].open('x') as log:
                completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            assert completed.returncode == 0, 'Extracted domain replay failed: ' + kind
            actual = json.loads(outputs[kind].read_text())
            assert actual == stored[kind], 'Extracted replay differs from original exported record: ' + kind
            print(json.dumps({'completed_extracted_domain': kind, 'matches_exported_audit': True}), flush=True)
        expression = ("import json,sys;from pathlib import Path;"
            "root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'));"
            "from final_scope_evidence_gate import validate;"
            "Path(sys.argv[2]).write_text(json.dumps(validate(root),indent=2)+'\\n')")
        command = [sys.executable, '-c', expression, str(root), str(outputs['editorial'])]
        commands['editorial'] = command
        with logs['editorial'].open('x') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, 'Extracted editorial lineage gate failed'
        assert json.loads(outputs['editorial'].read_text()) == stored['editorial_lineage']
        assert digest(packet) == packet_sha, 'Original packet changed during extraction/replay'
        result = {'scope': 'Actual fresh-root extraction and independent scoring replay of completed packet',
            'started_at_utc': start, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'wrapper_source_sha256': digest(Path(__file__).resolve()),
            'packet': str(packet), 'packet_sha256': packet_sha, 'packet_bytes': packet.stat().st_size,
            'fresh_extracted_root': str(root), 'temporary_root_removed_after_success': True,
            **extraction, 'portable_manuscript_validation': image_check,
            'both_included_domain_scoring_programs_actually_executed': True,
            'each_matches_own_exported_audit': True, 'actual_extracted_six_pass_document_gate_passed': True,
            'commands_with_explicit_extracted_root': commands,
            'saved_outputs_sha256': {str(q): digest(q) for q in outputs.values()},
            'saved_logs_sha256': {str(q): digest(q) for q in logs.values()},
            'paid_api_calls': 0, 'new_model_inference_or_training': False,
            'credential_read_by_wrapper': False, 'independent_physical_validation_claimed': False,
            'ranking_or_journal_acceptance_claimed': False}
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
