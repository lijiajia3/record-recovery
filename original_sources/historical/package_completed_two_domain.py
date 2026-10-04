"""Stream a new completed-scope review-ready packet; never publish or rerank.

Missing scientific/editorial prerequisites stop export. Completed raw datasets,
models and prior packets are preserved. Credentials are read only for scanning.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import zipfile

from final_scope_evidence_gate import validate as editorial_gate, safe, digest
from replay_polyie_neural_offline import replay as poly_replay, source_barrier as poly_barrier
from replay_mulms_neural_offline import replay as mu_replay, source_barrier as mu_barrier
from verify_streamed_evidence_zip import verify as verify_actual_archive

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = frozenset(('.py', '.cjs', '.js', '.ts', '.sh', '.md', '.json', '.jsonl', '.txt',
    '.csv', '.log', '.bib', '.yaml', '.yml', '.toml', '.ini', '.tex', '.html', '.xml', '.typ', '.css', '.patch', '.diff', '.rst'))
POLY_DIRS = ('polyie_shared_context_v1', 'polyie_typed_interaction_v1', 'polyie_capacity_mean_v1',
             'polyie_neural_family_v1')
MU_DIRS = ('mulms_supervised_v1', 'mulms_neural_family_v1',
           'mulms_biaffine_reference_test_v1', 'mulms_ordered_context_reference_test_v1')
PILOT = 'artifacts/pilot_v2_reproduction_20261003.zip'
PILOT_SHA = 'c58d597f72a37df94d58a139fc250ff77a7161fccd4b4c3cd869df2294b71e82'
CORPUS = 'literature/related_journal_papers_10.zip'
# Anchors already recorded by the original/second fixed-corpus reviewer, not
# recomputed from a subsequently revised corpus or rubric.
FIXED_ANCHORS = {
    'literature/corpus10/manifest.json': '0c570c66b2c04f3b8badac5f270f551508a1d8d4eb3ccd16f0908195acf2f992',
    'literature/corpus10/corpus_selection.md': 'fd2518538754089aaf2050bbe17772cf34a88f2907576bad94f34a4ae402e99f',
    'reviews/ranking_rubric.json': 'd8f052f8bf81b5d83480393b5f358291f41ed53a31ee870418d815460f4d7d82',
    'reviews/ranking_protocol.md': '631e5a35582526de641f590123240abab4ffe45d6f0ecb91e2be4f75d6e6fa1f'}


def text_name(name):
    p = Path(name)
    return p.suffix.lower() in TEXT_SUFFIXES or p.name.upper().startswith(('LICENSE', 'README', 'COPYRIGHT', 'NOTICE'))


def scan_stream(stream, name, secrets):
    assert secrets and all(len(s) >= 20 for s in secrets)
    carry = max(256, *(len(s) - 1 for s in secrets))
    previous, size, h = b'', 0, hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        joined = previous + block
        assert all(secret not in joined for secret in secrets), 'Credential found; stopped without displaying it'
        if text_name(name):
            assert not re.search(rb'sk-[A-Za-z0-9]{32,}', joined), 'Credential-like text literal found; stopped'
        h.update(block)
        size += len(block)
        previous = joined[-carry:]
    return {'sha256': h.hexdigest(), 'bytes': size}


def archive_names(names):
    assert len(names) == len(set(names)), 'Duplicate archive member'
    for name in names:
        assert name and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        assert '\\' not in name and '\x00' not in name, 'Unsafe archive member'


def scan_archive(source, secrets, depth=0):
    """Stream decompressed contents, including nested ZIP/Word/Torch archives."""
    assert depth < 8, 'Excessively nested archive must be inspected separately'
    members, decompressed = 0, 0
    with zipfile.ZipFile(source) as archive:
        archive_names(archive.namelist())
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            with archive.open(entry) as stream:
                result = scan_stream(stream, entry.filename, secrets)
            assert result['bytes'] == entry.file_size, 'Archive member size changed'
            members += 1
            decompressed += result['bytes']
            # Open a separate stream: is_zipfile can seek to the central index.
            with archive.open(entry) as stream:
                nested = zipfile.is_zipfile(stream)
            if nested:
                with archive.open(entry) as stream:
                    child = scan_archive(stream, secrets, depth + 1)
                members += child['scanned_decompressed_members']
                decompressed += child['scanned_decompressed_bytes']
    return {'scanned_decompressed_members': members, 'scanned_decompressed_bytes': decompressed}


def scan_file(path, secrets):
    with path.open('rb') as stream:
        value = scan_stream(stream, str(path), secrets)
    if zipfile.is_zipfile(path):
        value['nested_archive_scan'] = scan_archive(path, secrets)
    return value


def add_tree(files, directory):
    assert directory.is_dir(), directory
    files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)


def add_epoch_checkpoints(files, directory):
    epochs = json.loads((directory / 'development_all_epochs.json').read_text())
    assert set(epochs) == {'20261003', '20261004', '20261005'}
    for seed, records in epochs.items():
        assert set(records) == {'1', '2', '3'}
        for epoch, record in records.items():
            path = safe(ROOT, record['checkpoint'])
            assert path == directory / ('seed' + seed) / ('epoch' + epoch + '.pt')
            assert digest(path) == record['checkpoint_sha256']
            files.add(path)


def corpus_component():
    for name, expected in FIXED_ANCHORS.items():
        assert digest(safe(ROOT, name)) == expected, 'Fixed reviewer corpus/policy anchor changed: ' + name
    manifest = json.loads((ROOT / 'literature/corpus10/manifest.json').read_text())
    papers = manifest['papers']
    assert manifest['item_count'] == len(papers) == 10
    assert len({p['doi'] for p in papers}) == len({p['id'] for p in papers}) == 10
    with zipfile.ZipFile(ROOT / CORPUS) as archive:
        archive_names(archive.namelist())
        embedded = json.loads(archive.read('manifest.json'))
        assert embedded == manifest
        pdfs = {Path(p['local_path']).name for p in papers}
        assert {n for n in archive.namelist() if n.lower().endswith('.pdf')} == pdfs
        for paper in papers:
            h, size = hashlib.sha256(), 0
            with archive.open(Path(paper['local_path']).name) as stream:
                head = stream.read(5)
                assert head == b'%PDF-'
                h.update(head)
                size += len(head)
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    size += len(block)
            assert h.hexdigest() == paper['sha256'] and size == paper['file_size_bytes']
    return {'component': CORPUS, 'sha256': digest(ROOT / CORPUS), 'fixed_original_journal_PDFs': 10,
            'DOIs': [p['doi'] for p in papers]}


def verify_training_caches(root, dataset):
    """Validate original completed state hashes and exact source-path sets."""
    config = {'polyie': ('polyie_shared_context_v1', 'window_id', (285, 58)),
              'mulms': ('mulms_supervised_v1', 'id', (7538, 1532))}
    directory, identity_key, numbers = config[dataset]
    freeze = root / ('research/' + dataset + '_local_source_cache_freeze.json')
    assert freeze.is_file()
    files = {freeze}
    for split, expected_n in zip(('train', 'dev'), numbers):
        inputs = json.loads((root / 'data' / dataset / (split + '_inputs.json')).read_text())
        assert len(inputs) == len({i[identity_key] for i in inputs}) == expected_n
        prefix = 'results/local_baseline/' + directory + '/source_cache/' + split
        expected = {prefix + '/' + hashlib.sha256(i[identity_key].encode()).hexdigest()[:20] + '.pt' for i in inputs}
        state_path = root / prefix / 'state.json'
        state = json.loads(state_path.read_text())
        assert state['freeze_sha256'] == digest(freeze)
        if dataset == 'polyie':
            assert state['status'] == 'completed_source_cache' and state['target_files_read'] is False
            assert state['windows'] == state['complete_windows'] == expected_n
        else:
            assert state['status'] == 'completed_source_only_cache' and state['targets_read'] is False
            assert state['sentences'] == state['complete_sentences'] == expected_n
        assert set(state['cache_sha256']) == expected
        assert {str(p.relative_to(root)) for p in (root / prefix).glob('*.pt')} == expected
        files.add(state_path)
        for name, wanted in state['cache_sha256'].items():
            path = safe(root, name)
            assert digest(path) == wanted, 'Original training cache bytes changed: ' + name
            files.add(path)
    return files


def portable_markdown(original, root):
    prefix = str(root / 'figures') + '/'
    return original.replace('](<' + prefix, '](<../figures/').replace('](' + prefix, '](../figures/')


def postwrite_source_check(files, entries, root):
    # Recheck ALL files after copying ALL members, rather than merely checking
    # each immediately after its own write while later sources may still change.
    for path in files:
        name = str(path.relative_to(root))
        assert path.stat().st_size == entries[name]['bytes'] and digest(path) == entries[name]['sha256'], 'Source changed during complete export: ' + name


def reviewed_export_sources():
    path = ROOT / 'research/completed_two_domain_export_source_review.json'
    review = json.loads(path.read_text())
    expected_sources = {'src/package_completed_two_domain.py', 'src/final_scope_evidence_gate.py',
                        'src/check_completed_packet_gates.py'}
    assert expected_sources.issubset(review['file_sha256']), 'Export source audit must bind all three current programs'
    for name, expected in review['file_sha256'].items():
        assert digest(safe(ROOT, name)) == expected, 'Export source changed after its independent review: ' + name
    return {path, *(safe(ROOT, name) for name in review['file_sha256'])}


def verify_partial_archive(partial, manifest, files, entries, root, secrets):
    # Compare copied member bytes, not just the source before/after. This also
    # catches a source changing and restoring its original bytes during a copy.
    with zipfile.ZipFile(partial) as archive:
        assert json.loads(archive.read('PACKET_MANIFEST.json')) == manifest
    verified = verify_actual_archive(partial)
    scanned = scan_archive(partial, secrets)
    postwrite_source_check(files, entries, root)
    return {'actual_zip_content_verification': verified, 'actual_zip_decompressed_credential_scan': scanned}


def content_files(editorial):
    poly_lock, _, _ = poly_barrier(ROOT)
    mu_lock, _, _, _, _, _ = mu_barrier(ROOT)
    files = {ROOT / p for p in editorial['files']}
    files.update(reviewed_export_sources())
    files.update(ROOT / p for p in poly_lock['file_sha256'])
    files.update(ROOT / p for p in mu_lock['file_sha256'])
    files.update(verify_training_caches(ROOT, 'polyie'))
    files.update(verify_training_caches(ROOT, 'mulms'))
    for lock_name in ('research/polyie_neural_family_test_freeze.json',
                      'research/mulms_neural_family_test_freeze.json',
                      'research/mulms_biaffine_reference_test_freeze.json',
                      'research/mulms_ordered_context_reference_test_freeze.json',
                      'research/mulms_independent_replay_source_review.json'):
        files.add(ROOT / lock_name)
        value = json.loads((ROOT / lock_name).read_text())
        for name, expected in value['file_sha256'].items():
            p = safe(ROOT, name)
            assert digest(p) == expected
            files.add(p)
    for directory in (*POLY_DIRS, *MU_DIRS):
        add_tree(files, ROOT / 'results/local_baseline' / directory)
    for directory in POLY_DIRS[:3]:
        add_epoch_checkpoints(files, ROOT / 'results/local_baseline' / directory)
    base = ROOT / 'results/local_baseline/mulms_supervised_v1'
    add_epoch_checkpoints(files, base / 'ner')
    for architecture in ('mean', 'typed', 'capacity_mean'):
        add_epoch_checkpoints(files, base / 'relations' / architecture)
    for architecture in ('biaffine', 'ordered_context'):
        add_epoch_checkpoints(files, base / (architecture + '_reference') / architecture)
    for dataset in ('polyie', 'mulms'):
        add_tree(files, ROOT / 'data' / dataset)
    add_tree(files, ROOT / 'data/local_baselines/matscibert')
    add_tree(files, ROOT / 'data/local_baselines/mulms')
    for name in ('data/mulms/source/LICENSE', 'data/mulms/source/github/LICENSE_CODE',
                 'research/author_baseline_reproduction_audit.md', 'research/author_baseline_reproduction_audit.json',
                 'research/science_advances_reference_comparison.md', 'research/combined_completed_packet_plan.md',
                 'research/completed_two_domain_export_source_review.md',
                 'research/completed_two_domain_export_source_review.json',
                 'research/completed_two_domain_source_component_check.json',
                 'research/polyie_neural_independent_replay_v2.json', 'research/mulms_neural_independent_replay.json',
                 'reviews/ranking_rubric.json', 'reviews/ranking_protocol.md',
                 'reviews/ranking_metadata_correction.md',
                 'reviews/history/round1_original_protocol_metadata/ranking_protocol.md',
                 'reviews/history/round1_original_protocol_metadata/ranking_rubric.json',
                 'reviews/final_scope_artifact_schema.md',
                 'literature/corpus10/manifest.json', 'literature/corpus10/corpus_selection.md',
                 'artifacts/pilot_v2_reproduction_20261003.manifest.json',
                 PILOT, CORPUS):
        files.add(safe(ROOT, name))
    for pattern in ('polyie_*.md', 'polyie_*.json', 'mulms_*.md', 'mulms_*.json'):
        files.update((ROOT / 'research').glob(pattern))
    for directory in ('research/polyie_neural_external_replay_review_inputs',
                      'research/mulms_independent_replay_source_review_inputs',
                      'research/completed_two_domain_export_source_review_inputs',
                      'research/applied_skill_sources'):
        d = ROOT / directory
        assert d.is_dir()
        # Small independent reviewer snapshots, helpers, outputs and licensed
        # skill sources; unrelated author repository mirrors remain separate.
        add_tree(files, d)
    for prefix in ('ranking_round1', 'ranking_round2'):
        files.update((ROOT / 'reviews').glob(prefix + '.*'))
    files.update(ROOT / 'src' / p for p in ('package_completed_two_domain.py', 'final_scope_evidence_gate.py',
        'check_completed_packet_gates.py', 'replay_polyie_neural_offline.py', 'replay_mulms_neural_offline.py',
        'verify_streamed_evidence_zip.py', 'plot_polyie_neural_family.py', 'plot_mulms_completed_family.py',
        'make_docx.cjs'))
    # Stable local imports only; no entire live src directory snapshot.
    pending = [p for p in files if p.parent == ROOT / 'src' and p.suffix == '.py']
    inspected = set()
    while pending:
        source = pending.pop()
        if source in inspected:
            continue
        inspected.add(source)
        for node in ast.walk(ast.parse(source.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                local = ROOT / 'src' / (module.split('.')[0] + '.py')
                if local.is_file() and local not in files:
                    files.add(local)
                    pending.append(local)
    for p in files:
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()), p
    return sorted(files)


def readme():
    return '''# Completed two-domain scientific extraction evidence

This review-ready packet contains the final readable Markdown/Word/PDF and its
three new-scope peer-review revisions and three humanizer passes, completed
POLYIE and MuLMS supervised source graphs, all fitted epochs and development
choices, source caches and pinned base MatSciBERT bytes, fixed data/protocols,
independent scoring-replay programs and source audits. The two additional Mu
directed references are descriptive reused-holdout comparisons; their actual
chronology is retained. No fresh training/inference replication is certified.

The prior SciERC/bulk-modulus pilot remains a separate original ZIP component
under artifacts/. Extract it to its own separate directory to retain historical
code paths. Its embedded paper is an earlier snapshot; the current paper is
manuscript/manuscript.pdf or manuscript.docx. The fixed ten original journal PDFs
are in literature/related_journal_papers_10.zip with their DOI/hash manifest.
No unfinished language-model development campaign is presented as a completed
external validation. Transitive helper code is included for dependency closure;
its presence does not certify completion of every experiment it can run.

1. Verify this actual ZIP's content hashes and sizes with bounded memory:
   python3 src/verify_streamed_evidence_zip.py /path/to/this.zip
2. Extract to a new empty directory and run its included standalone programs:
   python3 src/replay_polyie_neural_offline.py --root /path/to/extracted
   python3 src/replay_mulms_neural_offline.py --root /path/to/extracted
   Dependencies for scoring replay: NumPy, tokenizers and pyarrow. No Torch,
   transformers, API credentials or model inference is used by these programs.
   Mu checks all 18 source graphs before reading its held-out Gold; Poly checks
   all nine. Gold-missed entities and invalid-diagonal predictions remain scored.
   Exact exchanges use original papers; fitted seeds move together. Paper
   bootstrap intervals condition on those fits. Both local Holm2 families and
   the separate four-contrast sensitivity must match the stored completed data.
3. Inspect the six-pass lineage and bound final document/figure/reference records
   under reviews/final_scope/ and reviews/final_scope_validation.json. These are
   skill-assisted author/model reviews, not journal referee decisions.
4. Training replication requires a separate derived working copy and a new
   recorded protocol-preserving run. Do not overwrite bundled checkpoints,
   choices or test graphs. Hash/scoring replay alone does not refit any model.

The original manuscript source keeps its six-pass byte lineage. A separate
manuscript/portable_manuscript.md copy changes only absolute figure destinations
to relative paths, for viewing outside the original workspace. Word/PDF embed
their figures. Both source versions and the portability transformation are bound
in the packet manifest; the original source is not silently rewritten.

Mu corpus: original CC BY-SA license data/mulms/source/LICENSE. Four unmodified
author code snapshots required by the relation freeze retain their AGPL license
data/mulms/source/github/LICENSE_CODE and provenance in the author-source audit.
POLYIE: original source provenance and Apache2.0 license included. MatSciBERT:
MIT model card and pinned source manifest included. Do not relabel these source
licenses or interpret an adapted head as exact author-model reproduction.

This export does not establish physical measurement truth, journal acceptance,
the requested top-three simulated rank or ten useful hours. Those task criteria
have separate recorded outcomes. All included bytes and decompressed embedded
archive members were scanned for project-external credential bytes, without
including or displaying the credentials.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[a-z0-9_-]{1,64}', args.tag), 'Use a distinct simple export version'
    dest = ROOT / ('artifacts/completed_two_domain_evidence_20261003_' + args.tag + '.zip')
    assert not dest.exists() and not dest.with_suffix('.zip.partial').exists()
    editorial = editorial_gate(ROOT)
    reviewed_export_sources()
    assert digest(ROOT / PILOT) == PILOT_SHA, 'Preserve original pilot component'
    corpus = corpus_component()
    poly, mu = poly_replay(ROOT), mu_replay(ROOT)
    saved_mu = json.loads((ROOT / 'research/mulms_neural_independent_replay.json').read_text())
    assert saved_mu == mu, 'Source/replay changed since stored completed audit'
    files = content_files(editorial)
    key_path = Path(os.environ.get('SILICONFLOW_KEY_FILE', '/Users/jiajia/.siliconflow_feedback_key'))
    assert not key_path.resolve().is_relative_to(ROOT.resolve()), 'Credential scan source must stay outside the project'
    secret = key_path.read_bytes().strip()
    assert len(secret) >= 20
    secrets = (secret,)
    entries = {str(p.relative_to(ROOT)): scan_file(p, secrets) for p in files}
    original = (ROOT / 'manuscript/manuscript.md').read_text()
    portable = portable_markdown(original, ROOT)
    docs = {'README_REPRODUCTION.md': readme().encode(),
        'manuscript/portable_manuscript.md': portable.encode(),
        'OFFLINE_REPLAY_EXPORT_AUDIT.json': (json.dumps({'polyie': poly, 'mulms': mu,
            'editorial_lineage': editorial}, indent=2) + '\n').encode()}
    for name, value in docs.items():
        import io
        entries[name] = scan_stream(io.BytesIO(value), name, secrets)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'completed supervised two-domain study, prior pilot component and final six-pass readable manuscript',
        'review_ready_not_task_completion_certificate': True,
        'unfinished_language_model_holdout_results_claimed_complete': False,
        'source_freeze_sha256': {'polyie': poly['freeze_sha256'], 'mulms': mu['freeze_sha256']},
        'all_27_Poly_and_54_Mu_epoch_checkpoints_included': True,
        'paper_source_sha256': editorial['document_sha256']['manuscript/manuscript.md'],
        'portable_markdown_changes': 'Only original absolute figure destinations replaced by ../figures/',
        'prior_pilot_component_sha256': PILOT_SHA, 'fixed_journal_corpus_component': corpus,
        'actual_external_credential_bytes_and_decompressed_archive_scans_passed': True,
        'textual_credential_like_literal_scan_passed': True, 'paid_API_calls_during_export': 0,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {n: importlib.metadata.version(n) for n in
                ('numpy', 'torch', 'transformers', 'safetensors', 'tokenizers', 'pyarrow', 'matplotlib', 'pypdf')}},
        'files': entries}
    dest.parent.mkdir(exist_ok=True)
    partial = dest.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for p in files:
            name = str(p.relative_to(ROOT))
            archive.write(p, name)
            assert scan_file(p, secrets) == entries[name], 'File changed during export: ' + name
        for name, value in docs.items():
            archive.writestr(name, value)
        postwrite_source_check(files, entries, ROOT)
        archive.writestr('PACKET_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    actual_archive_audit = verify_partial_archive(partial, manifest, files, entries, ROOT, secrets)
    partial.rename(dest)
    final = {'packet': str(dest.relative_to(ROOT)), 'packet_bytes': dest.stat().st_size,
        'packet_sha256': digest(dest), 'content_files': len(entries),
        'scope': manifest['scope'], 'export_scans_passed': True,
        **actual_archive_audit,
        'new_training_or_inference_replication_claimed': False,
        'actual_zip_member_verification_completed_before_final_rename': True,
        'actual_extracted_root_replays_still_required': True}
    audit = dest.with_suffix('.manifest.json')
    assert not audit.exists()
    audit.write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
