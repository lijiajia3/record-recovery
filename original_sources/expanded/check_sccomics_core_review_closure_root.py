"""Verify existing independent SOURCE evidence; creates no fitting authority."""
import datetime
import hashlib
import json
from pathlib import Path


REVIEWS = (
    'round4_sccomics_source_model_independent_review',
    'round4_sccomics_native_graph_independent_source_review',
    'round4_sccomics_training_projection_independent_source_review',
    'round4_sccomics_fitting_core_independent_source_review',
    'round4_sccomics_statistics_independent_source_review',
    'round4_sccomics_independent_scoring_source_review',
    'round4_sccomics_test_analysis_independent_source_review',
    'round4_sccomics_independent_statistics_source_review',
    'round4_sccomics_supervised_stages_v2_independent_source_review',
    'round4_sccomics_full_configuration_source_review',
)
SOURCES = {
    'sccomics_source_models': '8ff50b5bb88c3fa6296b76ff9f5dab63a43177e1c0c54178aea01dafc59900f0',
    'sccomics_native_graph': 'b59306e02e7b8132e16366f70c729ee63df83ed10ee39e25ab0c38f74e321b53',
    'sccomics_training_projection': '08d9b2bd079c9be4a0ee1ece208105521dce15a741800750c19f4b5dcb34b7fa',
    'sccomics_fitting_core': '4bb8be6ea316de1cf18498a9436bf4b089413be8b2a6579013763a29f5cb0359',
    'sccomics_development_selection': '03335c910d1a7939801b7e1493deb62d18c2a7c39be40f199d6667771500c317',
    'sccomics_primary_matching_v2': '53167cf381710b4ee8b67960caf6cf297fd977c78e2c462917e2fd0f606c779d',
    'sccomics_statistics': '9b92e1e684dd0690589c5b497c18bdaec535f299fcb949c6b4f17e57dc361322',
    'sccomics_independent_scoring': '27846cb4792ac7485d2ca41378ff5c0530330d7ae940ce5ca2913dabe92cf367',
    'sccomics_test_analysis': '565f8ee4cd93e3ee045f5d74964b34692ece64f4b2e8c047bb17c98fcf0e7eb8',
    'sccomics_independent_statistics_v2': 'b83c9959b567a016d2177d862c4347465c4f346ac4f4081680b9c5a1532b14c7',
    'sccomics_supervised_stages_v2': '6bba0ec13e93490d680afb8b306cf9f997cf5d7ed86dabc241aaafc428702308',
    'sccomics_supervised_runtime_v2': 'a46bd4fef4f8c8d9e2afe93cfdb96e286845f86a49c8113e43e847cec3a8f78a',
}


def digest(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Regular non-symlink source/evidence file required.')
    value = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
            size += len(block)
    return {'sha256': value.hexdigest(), 'size_bytes': size}


def main():
    root = Path(__file__).resolve().parents[1]
    allowlist_path = root / 'research/round4_sccomics_core_review_closure_root_inputs/external_source_binding_allowlist.json'
    external = set(json.loads(allowlist_path.read_bytes())['allowed_external_source_files'])
    proof = {}
    files = {}
    mappings = {}
    failures = []
    repeated = 0
    statuses = {'source_only_review_passed',
                'source_only_configuration_coherence_review_complete_pending_full_code_freeze'}
    for stem in REVIEWS:
        name = 'research/' + stem + '.json'
        before = digest(root / name)
        review = json.loads((root / name).read_bytes())
        if review['status'] not in statuses:
            raise ValueError('A selected independent review does not pass its declared SOURCE scope.')
        proof[name] = before
        flat = review['file_sha256']
        for path, wanted in flat.items():
            if '..' in Path(path).parts:
                raise ValueError('Noncanonical review binding.')
            if Path(path).is_absolute():
                if path not in external:
                    raise ValueError('External file is not in the explicit existing review code/skill list.')
            elif path.split('/')[0] not in {'src', 'research'}:
                raise ValueError('This audit reads SOURCE/research evidence, not actual data/weights.')
            if path in files:
                repeated += 1
                if files[path]['sha256'] != wanted:
                    failures.append({'path': path, 'review': name, 'conflicting_review_identity': wanted})
            else:
                actual = digest(root / path)
                if actual['sha256'] != wanted:
                    failures.append({'path': path, 'review': name, 'expected': wanted, 'actual': actual})
                files[path] = actual
        source_map = review.get('reviewed_source_sha256_by_path', review.get('reviewed_source_sha256'))
        mappings[name] = source_map
        if digest(root / name) != before:
            raise ValueError('Review changed during verification.')
    source = {}
    for stem, wanted in SOURCES.items():
        name = 'src/' + stem + '.py'
        actual = digest(root / name)
        if actual['sha256'] != wanted or name not in files or files[name] != actual:
            failures.append({'path': name, 'current_execution_identity_disagrees': actual})
        covered = any((isinstance(m, dict) and m.get(name) == wanted) or m == wanted for m in mappings.values())
        if not covered:
            failures.append({'path': name, 'independent_source_mapping_missing': True})
        source[name] = actual
    output = root / 'research/round4_sccomics_core_review_closure_root_check.json'
    report = {
        'identity': 'CURRENT12_EXECUTION_SOURCE_AND_INDEPENDENT_REVIEW_CLOSURE_CHECK_NOT_FIT_GATE',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'review_count': len(proof),
        'distinct_bound_files_checked': len(files),
        'repeated_bindings_checked_for_agreement': repeated,
        'review_bindings': proof,
        'execution_sources': source,
        'failures': failures,
        'passed': not failures,
        'actual_SC_data_weight_cache_or_annotations_read': False,
        'actual_fit_or_test_Gold_or_final_paper_review_authority': False,
        'corrected_real_import_compatibility_still_required': True,
        'full_configuration_review_remains_pending_final_code_freeze': True,
        'helper': {'path': str(Path(__file__).relative_to(root)), **digest(Path(__file__))},
        'external_SOURCE_only_allowlist': {'path': str(allowlist_path.relative_to(root)), **digest(allowlist_path)},
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('passed', 'review_count', 'distinct_bound_files_checked', 'repeated_bindings_checked_for_agreement', 'failures')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
