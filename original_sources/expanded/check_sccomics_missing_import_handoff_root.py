"""Stream the final owned SOURCE/artificial handoff; no actual SC payloads."""
import datetime
import hashlib
import json
from pathlib import Path


BASE = 'research/round4_sccomics_runtime_missing_helpers_v4_inputs/STABLE_SOURCE_HANDOFF/'
MANIFEST_SHA = 'ccf4bc1d4a3f8622efddbeaad8c7da063e2f45afe35daefe53bbe14f9b4338e5'
SUPPLEMENT_SHA = 'f1e8aa9bf361d6292b8c8787844c5e3a2ddbd93232bf671c8961a13593578dff'


def digest(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Regular no-alias SOURCE/artificial file required.')
    total = 0
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
            total += len(block)
    return {'sha256': value.hexdigest(), 'size_bytes': total}


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_name = BASE + 'implementation_delivery_manifest.json'
    supplement_name = BASE + 'stable_handoff_supplement.json'
    manifest_before = digest(root / manifest_name)
    supplement_before = digest(root / supplement_name)
    if manifest_before['sha256'] != MANIFEST_SHA or supplement_before['sha256'] != SUPPLEMENT_SHA:
        raise ValueError('Explicit final handoff identities differ.')
    manifest = json.loads((root / manifest_name).read_bytes())
    supplement = json.loads((root / supplement_name).read_bytes())
    if len(manifest['bindings']) != 44076 or len(supplement['bindings']) != 21:
        raise ValueError('Exact final handoff populations differ.')
    if supplement['actual_import_or_fit_final_freeze_authority'] is not False:
        raise ValueError('Handoff cannot grant scientific/import execution authority.')
    seen = set()
    total = 0
    failed = []
    for entry in manifest['bindings']:
        name = entry['path']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in {'src', 'research'}:
            raise ValueError('Undeclared real payload/credential/external-file duty.')
        if name in seen:
            raise ValueError('Duplicate explicit handoff member.')
        seen.add(name)
        actual = digest(root / name)
        total += actual['size_bytes']
        if actual != {key: entry[key] for key in ('sha256', 'size_bytes')}:
            failed.append({'path': name, 'actual': actual})
    for entry in supplement['bindings']:
        name = entry['path']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in {'src', 'research'}:
            raise ValueError('Supplement outside SOURCE/owned evidence scope.')
        if digest(root / name) != {key: entry[key] for key in ('sha256', 'size_bytes')}:
            failed.append({'supplement_path': name})
    if digest(root / manifest_name) != manifest_before or digest(root / supplement_name) != supplement_before:
        raise ValueError('Stable handoff changed during root verification.')
    report = {
        'identity': 'ROOT44076_SOURCE_ARTIFICIAL_STREAM_CHECK_NOT_ACTUAL_MODEL_OR_COMPATIBILITY_CERTIFICATE',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': not failed, 'actual_files_checked': len(seen),
        'actual_bytes_checked': total, 'key_supplement_bindings_checked': 21,
        'manifest_binding': manifest_before, 'supplement_binding': supplement_before,
        'failed_bindings': failed, 'helper_binding': digest(Path(__file__)),
        'SC_corpus_cold_cache_ANN_or_model_or_real_six_library_smoke': False,
        'final_different_SOURCE_review_and_actual_import_still_required': True,
        'scientific_fit_or_paper_peer_or_humanizer_or_rank_contribution': 0,
    }
    destination = root / 'research/round4_sccomics_missing_import_handoff_actual_root_check.json'
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('passed', 'actual_files_checked', 'actual_bytes_checked', 'failed_bindings')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
