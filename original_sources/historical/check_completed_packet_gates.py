"""Invented six-pass artifacts and secret-bearing archives; never real Gold/key."""
import difflib
import io
import json
from pathlib import Path
import tempfile
import zipfile

from pypdf import PdfWriter

from final_scope_evidence_gate import validate, digest, SCOPE, PASS_IDS, SUMMARY_PATHS, DOCUMENTS
from package_completed_two_domain import scan_archive, scan_stream, archive_names, portable_markdown, postwrite_source_check, verify_partial_archive


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return {'path': name, 'sha256': digest(path)}


def reject(action):
    try:
        action()
    except (AssertionError, FileNotFoundError, KeyError):
        return
    raise AssertionError('Invalid invented fixture accepted')


def fixture(root):
    previous = '# INVENTED FIXTURE ONLY\nNo real scientific claim.\n'
    peer, human = [], []
    for number, identity in enumerate(PASS_IDS, 1):
        base = 'reviews/final_scope/' + identity
        before = save(root, base + '/before.md', previous)
        following = previous + 'Invented applied edit ' + str(number) + '.\n'
        after = save(root, base + '/final.md', following)
        difference = ''.join(difflib.unified_diff(previous.splitlines(keepends=True), following.splitlines(keepends=True),
            fromfile=before['path'], tofile=after['path']))
        patch = save(root, base + '/changes.diff', difference)
        skill = 'peer-review' if identity.startswith('peer_') else 'humanizer'
        archived = save(root, 'research/applied_skill_sources/' + skill + '/SKILL.md', 'INVENTED SKILL FIXTURE ONLY\n')
        manifest = {'scope': SCOPE, 'pass_id': identity, 'status': 'completed_with_applied_revision',
            'input_path': before['path'], 'output_path': after['path'], 'input_sha256': before['sha256'],
            'output_sha256': after['sha256'], 'diff_path': patch['path'], 'diff_sha256': patch['sha256'],
            'skill_name': skill, 'archived_skill_path': archived['path'], 'skill_source_sha256': archived['sha256'],
            'review_or_edit_identity': 'author_skill_assisted_revision'}
        if identity.startswith('peer_'):
            manifest['review_report'] = save(root, base + '/report.md', 'INVENTED review: apply an explicit edit.\n')
            peer.append(base + '/manifest.json')
        else:
            manifest['draft'] = save(root, base + '/draft.md', following + 'Invented draft only.\n')
            manifest['audit'] = save(root, base + '/audit.md', 'INVENTED audit: remove the draft-only line.\n')
            manifest['final'] = after
            manifest['audit_binding'] = save(root, base + '/audit_binding.json', json.dumps({
                'input_sha256': before['sha256'], 'draft_sha256': manifest['draft']['sha256'],
                'audit_notes_sha256': manifest['audit']['sha256'], 'final_sha256': after['sha256'],
                'remaining_tells': ['invented draft-only line'], 'post_audit_changes': ['removed invented draft-only line']}))
            human.append(base + '/manifest.json')
        save(root, base + '/manifest.json', json.dumps(manifest))
        previous = following
    save(root, DOCUMENTS[0], previous)
    docx = root / DOCUMENTS[1]
    with zipfile.ZipFile(docx, 'w') as archive:
        archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>')
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.write(root / DOCUMENTS[2])
    figures = save(root, 'figures/invented.svg', '<svg xmlns="http://www.w3.org/2000/svg"/>')
    for name in SUMMARY_PATHS:
        save(root, name, json.dumps({'invented_fixture_only': True}))
    document_sha = {name: digest(root / name) for name in DOCUMENTS}
    render = {'document_sha256': document_sha, 'page_count': 2, 'inspected_page_numbers': [1, 2], 'readability_issues_resolved': True}
    save(root, 'reviews/final_scope/render.json', json.dumps(render))
    refs = {'manuscript_sha256': document_sha[DOCUMENTS[0]], 'all_cited_sources_verified': True,
            'fabricated_or_unresolved_references': []}
    save(root, 'reviews/final_scope/references.json', json.dumps(refs))
    record = {'scope': SCOPE, 'status': 'completed_document_validation',
        'all_scientific_claims_checked_against_completed_evidence': True,
        'document_sha256': document_sha, 'scientific_summary_sha256': {name: digest(root / name) for name in SUMMARY_PATHS},
        'peer_review_manifests': peer, 'humanizer_manifests': human,
        'render_validation_path': 'reviews/final_scope/render.json',
        'reference_validation_path': 'reviews/final_scope/references.json',
        'figure_sha256': {figures['path']: figures['sha256']}}
    save(root, 'reviews/final_scope_validation.json', json.dumps(record))
    return record


def zip_bytes(name, content):
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return result.getvalue()


def main():
    with tempfile.TemporaryDirectory(prefix='invented-completed-packet-gates-') as temporary:
        root = Path(temporary)
        record = fixture(root)
        assert validate(root)['verified_actual_humanizer_passes'] == 3
        path = root / record['peer_review_manifests'][1]
        saved = path.read_bytes()
        bad = json.loads(saved)
        bad['input_sha256'] = '0' * 64
        path.write_text(json.dumps(bad))
        reject(lambda: validate(root))
        path.write_bytes(saved)
        patch = root / 'reviews/final_scope/humanizer_pass3/changes.diff'
        saved = patch.read_bytes()
        patch.write_text('invented metadata claim without actual diff\n')
        reject(lambda: validate(root))
        patch.write_bytes(saved)
        original = (root / DOCUMENTS[0]).read_bytes()
        (root / DOCUMENTS[0]).write_bytes(original + b'unreviewed final edit\n')
        reject(lambda: validate(root))
        (root / DOCUMENTS[0]).write_bytes(original)
        render = root / record['render_validation_path']
        saved = render.read_bytes()
        bad = json.loads(saved)
        bad['page_count'] = 3
        bad['inspected_page_numbers'] = [1, 2, 3]
        render.write_text(json.dumps(bad))
        reject(lambda: validate(root))
        render.write_bytes(saved)
        assert validate(root)['all_six_passes_form_one_final_source_lineage']
        human_manifest = root / record['humanizer_manifests'][0]
        saved = human_manifest.read_bytes()
        bad = json.loads(saved)
        bad['draft'] = bad['final']
        human_manifest.write_text(json.dumps(bad))
        reject(lambda: validate(root))
        human_manifest.write_bytes(saved)
        source = save(root, 'source/a.txt', 'invented original source\n')
        files = [root / source['path']]
        entries = {source['path']: {'sha256': source['sha256'], 'bytes': files[0].stat().st_size}}
        postwrite_source_check(files, entries, root)
        files[0].write_text('invented later concurrent change\n')
        reject(lambda: postwrite_source_check(files, entries, root))
        links = '![a](' + str(root / 'figures/a.png') + ')\n![b](<' + str(root / 'figures/b c.png') + '>)\n'
        assert portable_markdown(links, root) == '![a](../figures/a.png)\n![b](<../figures/b c.png>)\n'
        # A long invented literal crosses the 1MiB scanner boundary. It remains
        # detectable when compressed inside Word inside a nested component ZIP.
        invented = b'INVENTED_SCAN_CREDENTIAL_' + b'Q' * 900
        body = b'.' * (1024 * 1024 - 400) + invented + b'...'
        word = zip_bytes('word/document.xml', body)
        outer = zip_bytes('historical/invented.docx', word)
        reject(lambda: scan_archive(io.BytesIO(outer), (invented,)))
        shaped = b'sk-' + b'A' * 40
        reject(lambda: scan_archive(io.BytesIO(zip_bytes('review/changes.diff', shaped)), (invented,)))
        valid = zip_bytes('word/document.xml', b'INVENTED harmless XML fixture')
        assert scan_archive(io.BytesIO(valid), (invented,))['scanned_decompressed_members'] == 1
        reject(lambda: archive_names(['safe.txt', 'safe.txt']))
        reject(lambda: archive_names(['../unsafe.txt']))
        assert scan_stream(io.BytesIO(b'harmless fixture'), 'metadata.json', (invented,))['bytes'] == 16
        # The source is restored, but the archive captured different bytes.
        files[0].write_text('invented original source\n')
        manifest = {'scope': 'INVENTED EXPORT FIXTURE ONLY', 'files': entries}
        partial = root / 'invented.zip.partial'
        with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(source['path'], 'different bytes copied during transient change\n')
            archive.writestr('PACKET_MANIFEST.json', json.dumps(manifest))
        postwrite_source_check(files, entries, root)
        reject(lambda: verify_partial_archive(partial, manifest, files, entries, root, (invented,)))
        with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(files[0], source['path'])
            archive.writestr('PACKET_MANIFEST.json', json.dumps(manifest))
        audit = verify_partial_archive(partial, manifest, files, entries, root, (invented,))
        assert audit['actual_zip_content_verification']['all_content_hashes_and_sizes_match']
        assert audit['actual_zip_decompressed_credential_scan']['scanned_decompressed_members'] == 2
    print(json.dumps({'invented_six_pass_lineage_and_real_two_page_fixture_PDF_verified': True,
        'broken_chain_fake_diff_unreviewed_final_edit_and_wrong_actual_PDF_pagecount_rejected': True,
        'long_invented_secret_inside_nested_compressed_Word_detected': True,
        'shaped_diff_secret_duplicate_and_unsafe_archive_members_rejected': True,
        'humanizer_alias_postwrite_concurrent_change_and_both_image_link_syntaxes_checked': True,
        'transiently_changed_archive_member_rejected_even_when_source_restored': True,
        'real_manuscript_review_outputs_or_dataset_Gold_or_credentials_read': False,
        'paid_API_calls': 0}))


if __name__ == '__main__':
    main()
