"""Validate the six actual expanded-scope edits and final document evidence.

The check binds existing critique, draft, audit, revision and visual records.
It cannot perform a peer review, see a page, certify a journal or invent a pass.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(root, name):
    p = PurePosixPath(name)
    need(type(name) is str and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
         'Only contained relative paths are permitted')
    target = root / p
    need(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root),
         'An actual included regular file is required')
    return target


def binding(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return {'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}


def captured(root, d):
    p = safe(root, d['path'])
    need(binding(p) == {k: d[k] for k in ('sha256', 'size_bytes')}, 'Bound artifact changed')
    return p


def validate(root, name='reviews/expanded_scope_20261004/completed_six_pass_lineage.json'):
    root = root.resolve()
    manifest_path = safe(root, name)
    d = json.loads(manifest_path.read_bytes())
    need(d['scope'] == 'Completed pilot, POLYIE, MuLMS and SC-CoMIcs scientific manuscript' and
         d['model_assisted_not_external_human_peer_review'] is True,
         'Correct scientific scope and review provenance required')
    before = captured(root, d['initial_document'])
    records = d['peer_rounds'] + d['humanizer_passes']
    need(len(d['peer_rounds']) == len(d['humanizer_passes']) == 3, 'Three real passes of each skill required')
    for index, row in enumerate(records):
        need(row['sequence'] == index + 1, 'Actual sequential editing lineage required')
        input_path = captured(root, row['input'])
        need(binding(input_path) == binding(before), 'Editing lineage is not continuous')
        output = captured(root, row['output'])
        a, b = input_path.read_text(), output.read_text()
        need(a != b and '{{SC_' not in b, 'A completed manuscript and an actual revision are required')
        skill = captured(root, row['applied_skill'])
        need(skill.name == 'SKILL.md', 'Capture the actual skill instructions')
        wanted = ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
            fromfile=row['input']['path'], tofile=row['output']['path']))
        need(captured(root, row['diff']).read_text() == wanted, 'Actual full-document diff differs')
        if index < 3:
            review = captured(root, row['review']).read_text()
            response = captured(root, row['response']).read_text()
            need(len(review.split()) >= 250 and len(response.split()) >= 100,
                 'Actual scientific critique and concrete response records required')
        else:
            draft = captured(root, row['draft'])
            audit = captured(root, row['audit']).read_text()
            need(len(draft.read_text().split()) > 5000 and len(audit.split()) >= 100,
                 'Actual whole-manuscript draft and style audit required')
        before = output
    final = captured(root, d['final_document'])
    need(binding(final) == binding(before), 'Final document is not the last completed edit')
    text = final.read_text()
    need('{{SC_' not in text and 'independent test scoring remains pending' not in text,
         'Unfinished scientific result slots cannot be final')
    for key in ('scientific_completed_action', 'native_replay', 'statistical_replay',
                'references_validation', 'figure_validation', 'official_template_validation'):
        captured(root, d[key])
    action = json.loads(captured(root, d['scientific_completed_action']).read_bytes())
    need(action['status'] == 'completed_local_score_test', 'Actual completed SC science required')
    native = json.loads(captured(root, d['native_replay']).read_bytes())
    math = json.loads(captured(root, d['statistical_replay']).read_bytes())
    need(native['status'] == 'independent_native_counts_and_deterministic_statistics_replay_for_supplied_objects_only' and
         math['status'] == 'independent_arithmetic_replay_passed_for_supplied_counts_only',
         'Actual completed semantic and mathematical replay required')
    template = json.loads(captured(root, d['official_template_validation']).read_bytes())
    need(template['final_source_sha256'] == binding(final)['sha256'] and
         template['actual_TeX_compile_exit_code'] == 0 and template['actual_all_PDF_pages_visually_inspected'] is True and
         template['overfull_body_or_table_blocks'] == [] and template['missing_citations'] == [] and
         template['figure_or_table_clipping_observed'] is False,
         'Actual final official-template checks required')
    need(template['visually_inspected_PDF_pages'] == list(range(1, template['PDF_page_count'] + 1)),
         'Every actual final PDF page requires inspection')
    for document in template['document_files']:
        captured(root, document)
    return {'status': 'expanded_six_actual_skill_edits_and_bound_document_checks_validated',
            'actual_peer_revisions': 3, 'actual_humanizer_draft_audit_final_passes': 3,
            'final_manuscript': d['final_document'], 'lineage_manifest': binding(manifest_path),
            'independent_human_peer_review_or_journal_approval': False,
            'SC_source_semantic_and_math_replays_completed': True,
            'visual_inspection_performed_by_this_program': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    need(not a.out.exists(), 'Preserve previous validation reports')
    result = validate(a.root)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
