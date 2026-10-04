"""Verify a real six-pass editorial lineage before review-ready export.

This checks saved artifacts and byte links, not reviewer competence, manuscript
truth, journal acceptance, the requested rank or useful work time. It never runs
experiments, edits prose or fabricates a missing review.
"""
import difflib
import hashlib
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree

from pypdf import PdfReader

SCOPE = 'completed_two_domain_supervised_and_prior_pilot'
VALIDATION = 'reviews/final_scope_validation.json'
DOCUMENTS = ('manuscript/manuscript.md', 'manuscript/manuscript.docx', 'manuscript/manuscript.pdf')
PASS_IDS = tuple('peer_review_round' + str(i) for i in range(1, 4)) + tuple('humanizer_pass' + str(i) for i in range(1, 4))
SUMMARY_PATHS = (
    'results/local_baseline/polyie_neural_family_v1/test_summary.json',
    'results/local_baseline/mulms_neural_family_v1/test_summary.json',
    'results/local_baseline/mulms_biaffine_reference_test_v1/test_summary.json',
    'results/local_baseline/mulms_ordered_context_reference_test_v1/test_summary.json',
    'results/local_baseline/mulms_neural_family_v1/descriptive_direction_symmetry.json',
    'results/local_baseline/joint_supervised_global_holm4.json')


def safe(root, name):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts, name
    path = root / relative
    assert path.is_file() and not path.is_symlink(), name
    assert path.resolve().is_relative_to(root.resolve()), name
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(root, name):
    return json.loads(safe(root, name).read_text())


def pass_record(root, name, identity, expected_input):
    manifest = read(root, name)
    assert manifest['scope'] == SCOPE and manifest['pass_id'] == identity
    assert manifest['status'] == 'completed_with_applied_revision'
    assert manifest['input_sha256'] == expected_input
    before = safe(root, manifest['input_path'])
    after = safe(root, manifest['output_path'])
    assert digest(before) == manifest['input_sha256']
    assert digest(after) == manifest['output_sha256']
    assert before.read_bytes() != after.read_bytes(), 'A copied or unchanged file is not an applied pass'
    old, new = before.read_text(), after.read_text()
    expected_diff = ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=manifest['input_path'], tofile=manifest['output_path']))
    patch = safe(root, manifest['diff_path'])
    assert patch.read_text() == expected_diff and expected_diff, identity + '/actual_revision_diff'
    assert digest(patch) == manifest['diff_sha256']
    assert manifest['skill_name'] == ('peer-review' if identity.startswith('peer_') else 'humanizer')
    skill = safe(root, manifest['archived_skill_path'])
    assert digest(skill) == manifest['skill_source_sha256']
    required = ('review_report',) if identity.startswith('peer_') else ('draft', 'audit', 'final')
    paths = {name, manifest['input_path'], manifest['output_path'], manifest['diff_path'], manifest['archived_skill_path']}
    for key in required:
        entry = manifest[key]
        file = safe(root, entry['path'])
        assert file.stat().st_size > 0 and digest(file) == entry['sha256'], identity + '/' + key
        paths.add(entry['path'])
    if identity.startswith('humanizer_'):
        assert manifest['final']['sha256'] == manifest['output_sha256']
        assert len({manifest[key]['path'] for key in ('draft', 'audit', 'final')}) == 3, 'Humanizer draft/audit/final need distinct records'
        binding_entry = manifest['audit_binding']
        binding_path = safe(root, binding_entry['path'])
        assert digest(binding_path) == binding_entry['sha256']
        binding = json.loads(binding_path.read_text())
        assert binding['input_sha256'] == manifest['input_sha256']
        assert binding['draft_sha256'] == manifest['draft']['sha256']
        assert binding['audit_notes_sha256'] == manifest['audit']['sha256']
        assert binding['final_sha256'] == manifest['final']['sha256']
        assert isinstance(binding['remaining_tells'], list) and isinstance(binding['post_audit_changes'], list)
        if manifest['draft']['sha256'] != manifest['final']['sha256']:
            assert binding['post_audit_changes'], 'Describe the changes made after the actual draft audit'
        paths.add(binding_entry['path'])
    # Simulation identity is disclosed; a model pass cannot be labeled an
    # actual journal review, outside expert consultation or acceptance.
    assert manifest['review_or_edit_identity'] in ('author_skill_assisted_revision', 'model_simulated_review')
    return manifest['output_sha256'], paths


def validate(root):
    root = root.resolve()
    record = read(root, VALIDATION)
    assert record['scope'] == SCOPE and record['status'] == 'completed_document_validation'
    assert record['all_scientific_claims_checked_against_completed_evidence'] is True
    assert set(record['document_sha256']) == set(DOCUMENTS)
    assert set(record['scientific_summary_sha256']) == set(SUMMARY_PATHS)
    paths = {VALIDATION, *DOCUMENTS, *SUMMARY_PATHS}
    for name, expected in {**record['document_sha256'], **record['scientific_summary_sha256']}.items():
        assert digest(safe(root, name)) == expected, name
    peer = record['peer_review_manifests']
    human = record['humanizer_manifests']
    assert len(peer) == len(human) == 3 and len(set(peer + human)) == 6
    assert all(Path(n).parts[:2] == ('reviews', 'final_scope') for n in peer + human), 'Old pilot passes do not satisfy final scope'
    first = read(root, peer[0])
    previous = first['input_sha256']
    for name, identity in zip(peer + human, PASS_IDS):
        previous, pass_paths = pass_record(root, name, identity, previous)
        paths.update(pass_paths)
    assert previous == record['document_sha256']['manuscript/manuscript.md'], 'Final paper differs from completed lineage'
    render = read(root, record['render_validation_path'])
    assert render['document_sha256'] == record['document_sha256']
    assert type(render['page_count']) is int and render['page_count'] > 0
    assert len(render['inspected_page_numbers']) == render['page_count']
    assert set(render['inspected_page_numbers']) == set(range(1, render['page_count'] + 1))
    assert render['readability_issues_resolved'] is True
    reference = read(root, record['reference_validation_path'])
    assert reference['manuscript_sha256'] == previous
    assert reference['all_cited_sources_verified'] is True
    assert reference['fabricated_or_unresolved_references'] == []
    paths.update((record['render_validation_path'], record['reference_validation_path']))
    for name, expected in record['figure_sha256'].items():
        assert Path(name).parts[0] == 'figures' and digest(safe(root, name)) == expected
        paths.add(name)
    assert record['figure_sha256'], 'Rendered final manuscript needs bound figure evidence'
    pdf = safe(root, 'manuscript/manuscript.pdf')
    with pdf.open('rb') as stream:
        assert stream.read(5) == b'%PDF-'
    reader = PdfReader(pdf, strict=True)
    assert not reader.is_encrypted and len(reader.pages) == render['page_count'], 'PDF page count differs from actual render'
    with zipfile.ZipFile(safe(root, 'manuscript/manuscript.docx')) as document:
        assert 'word/document.xml' in document.namelist()
        xml = ElementTree.fromstring(document.read('word/document.xml'))
        assert xml.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document'
    return {'scope': SCOPE, 'verified_actual_peer_review_revisions': 3,
        'verified_actual_humanizer_passes': 3, 'all_six_passes_form_one_final_source_lineage': True,
        'document_sha256': record['document_sha256'], 'files': sorted(paths),
        'journal_peer_review_or_acceptance_certified': False,
        'top_three_rank_or_ten_useful_hours_certified': False}
