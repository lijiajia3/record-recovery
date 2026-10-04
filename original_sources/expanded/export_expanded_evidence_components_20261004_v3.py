"""Export complete ordinary-file components of the expanded saved-output study.

Original phase archives remain unchanged. The compact supplement contains
local labels, choices, graphs, analysis, final manuscript and editing records.
No credential is included; its external bytes are used only for stream scans.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile

from validate_expanded_manuscript_lineage_20261004 import validate as editorial

ROOT = Path(__file__).resolve().parents[1]
PHASES = {
    'fit_ner': ('fit_ner', 'SC_V5_FIT_NER_20261004_original_outputs.zip'),
    'ner_dev': ('ner_dev', 'SC_V5_NER_DEV_20261004_original_outputs.zip'),
    'fit_heads': ('fit_heads', 'SC_V5_FIT_HEADS_20261004_original_outputs.zip'),
    'pair_dev': ('pair_dev', 'SC_V5_PAIR_DEV_20261004_original_outputs.zip'),
    'test_sources': ('test_sources_migration', 'SC_TEST_GPU_MIGRATION_20261004_V1_original_outputs.zip'),
}
ACTION = 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004_score_test/completed_local_action_receipt.json'
TEXT = {'.py', '.md', '.json', '.jsonl', '.txt', '.cjs', '.js', '.tex', '.bib', '.csv', '.log', '.diff', '.patch', '.stdout', '.stderr', '.yaml', '.toml', '.xml'}
ARCHIVES = {'.zip', '.docx', '.npz', '.pt', '.xlsx', '.pptx'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def binding(path):
    h, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk); n += len(chunk)
    return {'sha256': h.hexdigest(), 'size_bytes': n}


def safe_names(entries, allow_empty_directories=False):
    names = [e.filename for e in entries]
    need(len(names) == len(set(names)), 'Duplicate component member')
    for entry in entries:
        p = PurePosixPath(entry.filename)
        need(not p.is_absolute() and '..' not in p.parts and
             '\\' not in entry.filename and '\x00' not in entry.filename and
             stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR),
             'Only safe archive names and regular or directory metadata are permitted')
        if entry.is_dir():
            need(allow_empty_directories and entry.file_size == 0,
                 'Only nested empty-directory metadata is permitted')
        else:
            need(not stat.S_ISDIR(entry.external_attr >> 16), 'Directory mode on regular payload refused')


def scan(stream, name, secret):
    h, size, previous = hashlib.sha256(), 0, b''
    textual = Path(name).suffix.lower() in TEXT or Path(name).name.upper().startswith(('LICENSE', 'README', 'NOTICE'))
    carry = max(256, len(secret) - 1)
    for block in iter(lambda: stream.read(1048576), b''):
        data = previous + block
        need(secret not in data, 'External credential bytes found; stopped without displaying them')
        if textual:
            need(re.search(rb'sk-[A-Za-z0-9]{32,}', data) is None, 'Credential-like literal found; export stopped')
        h.update(block); size += len(block); previous = data[-carry:]
    return {'sha256': h.hexdigest(), 'size_bytes': size}


def inspect_archive(path, secret, depth=0, allow_empty_directories=False):
    need(depth < 8, 'Unexpected archive nesting requires inspection')
    rows, nested_members = {}, 0
    with zipfile.ZipFile(path) as z:
        safe_names(z.infolist(), allow_empty_directories=allow_empty_directories)
        for entry in z.infolist():
            with z.open(entry) as stream, tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as spool:
                for block in iter(lambda: stream.read(1048576), b''):
                    spool.write(block)
                spool.seek(0)
                row = scan(spool, entry.filename, secret)
                row['entry_kind'] = 'empty_directory_metadata' if entry.is_dir() else 'regular_payload'
                need(row['size_bytes'] == entry.file_size, 'Archive decompressed size differs')
                spool.seek(0); prefix = spool.read(4); spool.seek(0)
                nested = zipfile.is_zipfile(spool)
                # Torch checkpoints may be a legacy raw format, so the ZIP
                # signature rather than suffix makes their nested scan mandatory.
                declared = Path(entry.filename).suffix.lower() in ARCHIVES - {'.pt'} or prefix in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08')
                need(nested or not declared, 'Truncated declared nested archive')
                if nested:
                    child = inspect_archive(spool, secret, depth + 1, allow_empty_directories=True)
                    row['nested_decompressed_member_count'] = len(child['members']) + child['nested_member_count']
                    nested_members += row['nested_decompressed_member_count']
                rows[entry.filename] = row
    return {'members': rows, 'nested_member_count': nested_members,
            'all_members_actual_CRC_SHA_size_and_recursive_credential_scan': True}


def components():
    result = {'historical_two_domain': {
        'path': ROOT / 'artifacts/completed_two_domain_evidence_20261003_finalscope_v2.zip',
        'extraction_root': 'historical', 'original_expected': {
            'sha256': 'e86a1e43582c8cf6cc6582bada2c94891f3b5afbada99140bbe54e446832da5c',
            'size_bytes': 5375914218}}}
    result['SC_source_inputs'] = {'path': ROOT / 'cloud_deployments/SC_CUDA_SCIENCE_V5_20261004_inputs_only.zip',
                                'extraction_root': 'expanded', 'original_expected': {
                                    'sha256': '684d2a37177b33d58fcc79988ef1ff3a0a2d2e8ea56b34339a234b156c7be44c',
                                    'size_bytes': 1137838620}}
    for phase, (directory, filename) in PHASES.items():
        folder = ROOT / f'research/round4_sccomics_cloud_cuda_v5_actual_{directory}_complete'
        receipt = json.loads((folder / f'root_actual_complete_original_{phase}_intake_receipt.json').read_bytes())
        result['SC_' + phase] = {'path': folder / filename, 'extraction_root': 'expanded',
                                'original_expected': receipt['original_archive']}
    return result


def supplement_files():
    files = set()
    def tree(name):
        p = ROOT / name
        need(p.is_dir(), 'Required completed evidence directory missing: ' + name)
        files.update(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name != '.DS_Store')
    for name in ('data/sccomics_round4/source_v3', 'cloud_bridges/SC_NATIVE_FOLD1_V5_CUDA_20261004',
                 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_CUDA_20261004_select_ner',
                 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_CUDA_20261004_select_heads',
                 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004_verify_test',
                 'local_cloud_analysis/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004_score_test',
                 'reviews/expanded_scope_20261004', 'manuscript/final_expanded_20261004',
                 'manuscript/templates/science_template_v1.1'):
        tree(name)
    for path in (ROOT / 'src').glob('*.py'):
        if any(term in path.name for term in ('sccomics', 'expanded', 'science_template')):
            files.add(path)
    for path in (ROOT / 'src').glob('*science_template*.cjs'):
        files.add(path)
    for path in (ROOT / 'figures').glob('*.py'):
        files.add(path)
    figure_record = json.loads((ROOT / 'reviews/expanded_scope_20261004/actual_final_figures_validation.json').read_bytes())
    for row in figure_record['all_artifacts']:
        files.add(ROOT / row['path'])
    # Preserve complete scientific protocols, raw failures and review records.
    # Redundant generated artificial-fixture trees are not scientific outputs;
    # their generators and review/exit records are supplied instead.
    for path in (ROOT / 'research').iterdir():
        if path.is_file() and any(term in path.name for term in ('sccomics', 'SCComics', 'expanded', 'additional_eight')):
            files.add(path)
        elif path.is_dir() and ('sccomics' in path.name) and any(t in path.name for t in ('actual', 'history', 'failure', 'complete')):
            for q in path.rglob('*'):
                if q.is_file() and '__pycache__' not in q.parts and not any('INVENTED' in v for v in q.parts):
                    if q.suffix.lower() in TEXT or (q.suffix == '.zip' and q.stat().st_size < 10*1024*1024):
                        files.add(q)
    files.add(ROOT / 'results/local_baseline/joint_supervised_global_holm4.json')
    files.add(ROOT / 'research/round4_sccomics_text_provenance_actual/test_source_text_clusters.json')
    files.add(ROOT / 'research/mulms_completed_scope_science_advances_comparison.md')
    files.add(ROOT / 'research/mulms_completed_scope_science_advances_comparison.json')
    files.add(ROOT / 'research/expanded_scope_additional_eight_hours_completed_20261004.json')
    for p in files:
        need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT), 'Supplement requires real included files')
    return sorted(files)


def clone_file(source, target):
    need(not target.exists(), 'Do not overwrite evidence')
    subprocess.run(['/bin/cp', '-c', str(source), str(target)], check=True)
    need(target.is_file() and not target.is_symlink(), 'Clone must be an independent regular file')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new evidence-directory name')
    gate = editorial(ROOT)
    time_record = json.loads((ROOT / 'research/expanded_scope_additional_eight_hours_completed_20261004.json').read_bytes())
    need(time_record['closed_actual_useful_seconds_since_new_request'] >= 28800 and time_record['open_sessions_credited'] is False,
         'The new eight useful hours must actually be completed')
    key_path = Path(os.environ.get('SILICONFLOW_KEY_FILE', '/Users/jiajia/.siliconflow_feedback_key'))
    need(not key_path.resolve().is_relative_to(ROOT), 'Credential must remain outside the project')
    secret = key_path.read_bytes().strip()
    need(len(secret) >= 20, 'Actual external credential required only for scanning')
    partial = args.output.with_name(args.output.name + '.partial')
    need(not partial.exists(), 'Preserve previous export attempts')
    partial.mkdir(parents=True)
    c = components()
    original = {name: binding(row['path']) for name, row in c.items()}
    for name, row in c.items():
        need(row['original_expected'] is None or original[name] == row['original_expected'], 'Original phase archive changed')
    files = supplement_files()
    entries = {}
    for p in files:
        with p.open('rb') as stream:
            value = scan(stream, p.name, secret)
        if zipfile.is_zipfile(p):
            value['recursive_archive_scan'] = inspect_archive(p, secret, allow_empty_directories=True)
        entries[p.relative_to(ROOT).as_posix()] = value
    supplement = partial / 'expanded_local_evidence.zip'
    with zipfile.ZipFile(supplement, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    c['SC_completed_local_supplement'] = {'path': supplement, 'extraction_root': 'expanded', 'original_expected': None}
    copied = {}
    for name, row in c.items():
        source = row['path']
        audited = inspect_archive(source, secret)
        expected = binding(source)
        target = partial / (name + '.zip')
        if source != supplement:
            clone_file(source, target)
            need(binding(target) == expected, 'Component clone differs')
        else:
            target = source
        copied[name] = {'file': target.name, **expected, 'extraction_root': row['extraction_root'], **audited}
        print(json.dumps({'component_complete': name, 'members': len(audited['members'])}), flush=True)
    for name in original:
        need(binding(c[name]['path']) == original[name], 'Original component changed during export')
    for p in files:
        need(binding(p) == {k: entries[p.relative_to(ROOT).as_posix()][k] for k in ('sha256', 'size_bytes')},
             'Supplement source changed during export')
    readme = '''# Expanded research evidence

All component files are ordinary regular files and must travel together. The
historical root contains the completed pilot/POLYIE/MuLMS package and original
ten-paper/ranking records. The expanded root contains all SC input texts,
caches, cold weights, 150 epoch checkpoints, 15,000 development outputs,
1,500 test outputs, local labels/choices/graphs/analyses, final manuscript,
official template, figure artifacts and six actual editing records.

Use the supplied outer bootstrap with Python 3.13:

    python3 RUN_OFFLINE.py --packet . --extracted-root ../fresh_expanded_root --report-directory ../fresh_expanded_reports

Both output directories must be new and empty/nonexistent. The observed replay
runtime uses NumPy 2.2.6, tokenizers 0.22.1, PyArrow 22.0.0 and requests 2.32.5;
see the included dependency record. Dependencies are not bundled as installed
binaries. No GPU, training, network call or API credential is required. Portable
extraction needs capacity for all uncompressed members. On the original APFS
host an optional --verified-COW-candidates-root path can save physical space,
but each archive member is still fully decoded and compared before an actual
new regular-file clone is made; no source path becomes a symbolic dependency.

Every member must be decoded fully and checked, even when an optional regular
APFS copy-on-write copy saves space. No symbolic link or active-workspace
reference substitutes for an included byte stream. Replay scores the saved
outputs; it does not regenerate neural outputs or establish physical truth.
The shared fixed random streams provide deterministic arithmetic replay.

Original SC texts and annotations retain CC BY-NC 3.0 and CC BY 4.0 separately.
Original MuLMS code and corpus licenses remain in the historical component.
SC repository code without confirmed redistribution permission is excluded.
Artificial preparation fixture replicas are not empirical results: their
generators, SOURCE review records and actual failure/exit logs are retained.
The final draft still requires human author and declaration information.
'''
    (partial / 'README.md').write_text(readme)
    bootstrap = ROOT / 'src/extract_and_replay_expanded_components_20261004.py'
    clone_file(bootstrap, partial / 'RUN_OFFLINE.py')
    need(binding(bootstrap) == binding(partial / 'RUN_OFFLINE.py'), 'Offline bootstrap copy differs')
    manifest = {'status': 'complete_component_export_pending_actual_fresh_extraction_replay',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'components': copied,
        'exporter': binding(Path(__file__)), 'editorial_gate': gate,
        'outer_bootstrap': {'file': 'RUN_OFFLINE.py', **binding(partial/'RUN_OFFLINE.py')},
        'outer_README': {'file': 'README.md', **binding(partial/'README.md')},
        'actual_external_credential_and_recursive_decompressed_scans_passed': True,
        'regular_APFS_COW_copies_not_symbolic_links': True,
        'final_journal_acceptance_or_new_ranking_certified': False,
        'paid_API_calls_during_export': 0, 'neural_training_or_generation_during_export': False}
    (partial / 'COMPONENT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    partial.rename(args.output)
    print(json.dumps({'exported_directory': str(args.output), 'components': len(copied),
                      'fresh_extraction_replay_still_required': True}), flush=True)


if __name__ == '__main__':
    main()
