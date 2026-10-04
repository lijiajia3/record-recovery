"""Check the author's 198 SOURCE/artificial bindings, not model performance."""
from pathlib import Path
import datetime
import hashlib
import json


def digest(path):
    if any(value.is_symlink() for value in (path, *path.parents)) or not path.is_file():
        raise ValueError('Regular non-symlink source/artificial evidence required.')
    value = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
            size += len(block)
    return {'sha256': value.hexdigest(), 'size_bytes': size}


def main():
    root = Path(__file__).resolve().parents[1]
    name = 'research/round4_sccomics_runtime_dirfd_helpers_v3_inputs/STABLE_HELPER_HANDOFF/implementation_delivery_manifest.json'
    wanted = 'cbacb95426f4f2fe8f9b3784f369ab6529fc1221ed44e3a595fdbf4fbc670a7c'
    before = digest(root / name)
    if before['sha256'] != wanted:
        raise ValueError('Stable manifest SHA differs.')
    manifest = json.loads((root / name).read_bytes())
    if manifest['fit_authority'] is not False or manifest['transitive_actual_runtime_SC_model_payloads_read'] is not False:
        raise ValueError('This is an artificial/SOURCE handoff only.')
    if len(manifest['bindings']) != 198:
        raise ValueError('Exact author handoff count differs.')
    seen = set()
    failures = []
    actual_bindings = {}
    for entry in manifest['bindings']:
        name = entry['path']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in {'src', 'research'}:
            raise ValueError('Handoff path outside explicit source/evidence scope.')
        if name in seen:
            raise ValueError('Duplicate flat handoff binding.')
        seen.add(name)
        actual = digest(root / name)
        actual_bindings[name] = actual
        if actual != {key: entry[key] for key in ('sha256', 'size_bytes')}:
            failures.append({'path': name, 'expected': entry, 'actual': actual})
    if digest(root / 'research/round4_sccomics_runtime_dirfd_helpers_v3_inputs/STABLE_HELPER_HANDOFF/implementation_delivery_manifest.json') != before:
        raise ValueError('Handoff manifest changed during verification.')
    report = {
        'identity': 'ROOT198_SOURCE_ARTIFICIAL_DELIVERY_BYTE_CHECK_NOT_INDEPENDENT_REVIEW_OR_IMPORT_PERMISSION',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': not failures, 'checked_bindings': len(actual_bindings),
        'failures': failures, 'actual_bindings': actual_bindings,
        'manifest': before, 'helper': digest(Path(__file__)),
        'real_entry_imports_SC_model_Gold_or_fitting': False,
        'different_SOURCE_review_and_actual_compatibility_still_required': True,
    }
    output = root / 'research/round4_sccomics_runtime_dirfd_v3_198_handoff_root_binding_check.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({'passed': report['passed'], 'checked_bindings': len(actual_bindings), 'failures': failures}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
