"""Read-only root binding check; SOURCE and owned invented files only.

This does not execute the controller, touch SC data, or authorize fitting.
Finder exclusions are reported from the original manifest, never hidden.
"""
import collections
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
            size += len(block)
    return {'sha256': h.hexdigest(), 'size_bytes': size}


def main():
    root = Path(__file__).resolve().parents[1]
    name = 'research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/STABLE_SOURCE_HANDOFF/implementation_delivery_manifest.json'
    expected = '02395862d1f9a3489cb786e424253187dbcd6e088aabce2dc46be121917edbed'
    report_path = root / 'research/round4_sccomics_supervised_v2_stable_handoff_root_binding_check.json'
    if report_path.exists():
        raise ValueError('Preserve existing root verification; choose a new reviewed identity.')
    manifest_path = root / name
    before = digest(manifest_path)
    if before['sha256'] != expected:
        raise ValueError('Stable handoff identity changed.')
    manifest = json.loads(manifest_path.read_bytes())
    bindings = manifest['bindings']
    seen = set()
    failures = []
    checked = 0
    size = 0
    duties = collections.Counter()
    for item in bindings:
        text = item['path']
        relative = PurePosixPath(text)
        if relative.is_absolute() or any(p in {'.', '..'} for p in relative.parts):
            raise ValueError('Non-relative binding.')
        if str(relative) != text or relative.parts[0] not in {'src', 'research'}:
            raise ValueError('Only source/research handoff population may be read.')
        if text in seen or relative.name == '.DS_Store':
            raise ValueError('Duplicate or Finder file inside substantive population.')
        seen.add(text)
        path = root / text
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError('Source binding is not a regular non-symlink file.')
        actual = digest(path)
        wanted = {k: item[k] for k in ('sha256', 'size_bytes')}
        if actual != wanted:
            failures.append({'path': text, 'expected': wanted, 'actual': actual})
        checked += 1
        size += actual['size_bytes']
        duties[item['duty']] += 1
    after = digest(manifest_path)
    if after != before:
        raise ValueError('Handoff manifest changed during root verification.')
    result = {
        'identity': 'ROOT_READ_ONLY_SOURCE_AND_INVENTED_BINDING_VERIFICATION_NOT_INDEPENDENT_PIPELINE_REVIEW',
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest': {'path': name, **before},
        'substantive_bindings_checked': checked,
        'substantive_bytes_read': size,
        'duties': dict(sorted(duties.items())),
        'mismatches': failures,
        'passed': not failures and checked == 43829,
        'original_Finder_exclusions': manifest['original_manifest_Finder_bindings_excluded_from_flat_population'],
        'excluded_Finder_current_bytes_rechecked_by_this_helper': False,
        'original_helper_process_exit_observed': False,
        'actual_SC_data_cache_weights_predictions_or_held_annotations_read': False,
        'actual_pipeline_source_review_fit_authority_or_paper_peer_review': False,
        'helper': {'path': str(Path(__file__).relative_to(root)), **digest(Path(__file__))},
    }
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'substantive_bindings_checked', 'substantive_bytes_read', 'mismatches')}, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
