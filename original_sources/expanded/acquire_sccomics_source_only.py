"""Acquire pinned public source bytes without parsing annotation semantics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(path, *, file_required=False):
    """Check lexical ancestors before resolution can hide a symlink."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink path or ancestor: ' + str(part))
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), 'Path outside workspace: ' + str(path))
    if file_required:
        require(resolved.is_file(), 'Missing regular source file: ' + str(path))
    return resolved


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def captured_json(path):
    path = checked_path(path, file_required=True)
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest(), json.loads(raw)


def verify_inputs(expected):
    for name, wanted in expected.items():
        path = checked_path(ROOT / name, file_required=True)
        require(sha(path) == wanted, 'Changed source-stage input: ' + name)


def verify_saved(path, wanted_sha, wanted_size):
    path = checked_path(path, file_required=True)
    require(path.stat().st_size == wanted_size and sha(path) == wanted_sha,
            'Saved source byte integrity mismatch: ' + str(path))


def raw_archive(archive_path, destination, extension, expected_ids):
    """Only filenames, regular-file attributes, byte counts and hashes are read."""
    archive_path = checked_path(archive_path, file_required=True)
    destination = checked_path(destination)
    require(not destination.exists(), 'Previous extraction namespace')
    destination.mkdir(parents=True)
    entries, seen, targets = {}, set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate member name')
        for entry in archive.infolist():
            name = entry.filename
            p = PurePosixPath(name)
            require(entry.orig_filename == name and '\x00' not in entry.orig_filename,
                    'Truncated or NUL archive member name')
            require(name and not p.is_absolute() and '..' not in name.split('/'),
                    'Unsafe archive member path')
            require('\\' not in name and '.' not in name.split('/'),
                    'Ambiguous archive member path')
            file_type = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir():
                require(file_type in (0, stat.S_IFDIR), 'Non-directory archive member')
                continue
            require(file_type in (0, stat.S_IFREG), 'Nonregular archive member')
            if p.suffix.lower() == extension:
                require(re.fullmatch(r'\d+', p.stem), 'Unrecognized source ID')
                source_id = int(p.stem)
                require(source_id in expected_ids and source_id not in seen,
                        'Duplicate or unexpected source ID')
                seen.add(source_id)
                target = destination / (f'{source_id:04d}' + extension)
                category = 'raw_source_document' if extension == '.txt' else 'unparsed_annotation_bytes'
            else:
                target = destination / 'ancillary' / Path(*p.parts)
                category = 'unparsed_ancillary_bytes'
            target = checked_path(target)
            require(target not in targets, 'Duplicate canonical output path')
            targets.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            h, size = hashlib.sha256(), 0
            with archive.open(entry) as source, checked_path(target).open('xb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    h.update(block)
                    size += len(block)
            require(size == entry.file_size, 'Archive member size mismatch')
            verify_saved(target, h.hexdigest(), size)
            entries[name] = {'canonical_path': str(target), 'bytes': size,
                'sha256': h.hexdigest(), 'category': category}
    require(seen == set(expected_ids), 'Incomplete source population')
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    protocol_path, protocol_sha, protocol = captured_json(args.protocol)
    review_path, review_sha, review = captured_json(args.review)
    require(protocol['status'] == 'frozen_source_acquisition_and_native_split_only',
            'Unfrozen source-stage protocol')
    require(review['status'] == 'source_only_review_passed', 'Source review not passed')
    expected = dict(protocol['file_sha256'])
    for name, wanted in review['file_sha256'].items():
        require(name not in expected or expected[name] == wanted,
                'Conflicting input SHA bindings: ' + name)
        expected[name] = wanted
    protocol_name = str(protocol_path.relative_to(ROOT))
    review_name = str(review_path.relative_to(ROOT))
    require(review['file_sha256'][protocol_name] == protocol_sha,
            'Captured protocol differs from reviewed bytes')
    require(review['file_sha256']['src/acquire_sccomics_source_only.py'] ==
            sha(checked_path(Path(__file__), file_required=True)), 'Unreviewed acquisition source')
    require(review_name not in expected or expected[review_name] == review_sha,
            'Conflicting review SHA binding')
    expected[review_name] = review_sha
    expected[protocol_name] = protocol_sha
    metadata_path, metadata_sha, metadata = captured_json(ROOT / protocol['official_file_metadata_path'])
    require(protocol['file_sha256'][str(metadata_path.relative_to(ROOT))] == metadata_sha,
            'Captured official metadata differs from frozen bytes')
    verify_inputs(expected)
    ids = set(range(1, 1001))
    require(protocol['split_ids'] == {'train': list(range(201, 1001)),
        'dev': list(range(101, 201)), 'test': list(range(1, 101))}, 'Changed native source split')
    mapping = protocol['normalized_doi_by_id']
    require(set(mapping) == {str(i) for i in ids} and len(set(mapping.values())) == 1000,
            'Nonunique or incomplete DOI source population')
    require(protocol['archives'] == {'SC-CoMIcs-Abstract1000_CC_BY-NC-3.0.zip': '.txt',
        'SC-CoMIcs-Annotation1000_CC_BY_4.0.zip': '.ann'}, 'Changed official source archive set')
    destination = checked_path(ROOT / 'data/sccomics_round4/source_v3')
    state_path = checked_path(ROOT / 'logs/sccomics_round4_source_acquisition_state.json')
    require(not destination.exists(), 'Preserve previous acquisition namespace')
    require(not state_path.exists(), 'Preserve previous acquisition state')
    destination.mkdir(parents=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {'scope': 'Raw source acquisition only; no training or semantic annotation parsing',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'acquiring_official_source_bytes', 'protocol_sha256': protocol_sha,
        'source_review_sha256': review_sha, 'semantic_annotation_read': False,
        'paid_api_calls': 0, 'credentials_read': False, 'completed_archives': []}
    def save():
        checked_path(state_path).write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        records = {}
        by_name = {record['filename']: record for record in metadata}
        require(len(by_name) == len(metadata), 'Duplicate official metadata record')
        for name, extension in protocol['archives'].items():
            record = by_name[name]
            wanted = record['content_details']['sha256_hash']
            expected_size = record['size']
            url = record['content_details']['download_url']
            require(url.startswith('https://data.mendeley.com/public-files/datasets/xc9fjz2p3h/'),
                    'Unexpected public source endpoint')
            target = checked_path(destination / name)
            partial = target.with_suffix(target.suffix + '.partial')
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://data.mendeley.com/datasets/xc9fjz2p3h/3'})
            h, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=120) as response, checked_path(partial).open('xb') as output:
                require(response.status == 200, 'Public source HTTP response not 200')
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
                    h.update(block)
                    size += len(block)
            require(size == expected_size and h.hexdigest() == wanted, 'Official archive integrity mismatch')
            verify_saved(partial, wanted, expected_size)
            checked_path(partial, file_required=True).rename(checked_path(target))
            verify_saved(target, wanted, expected_size)
            folder = destination / ('raw_text' if extension == '.txt' else 'raw_annotations')
            entries = raw_archive(target, folder, extension, ids)
            records[name] = {'url': url, 'official_sha256': wanted, 'bytes': size,
                'archive': str(target.relative_to(ROOT)), 'members': entries,
                'annotation_semantics_parsed': False}
            state['completed_archives'].append(name)
            save()
        for record in records.values():
            verify_saved(ROOT / record['archive'], record['official_sha256'], record['bytes'])
            for entry in record['members'].values():
                verify_saved(Path(entry['canonical_path']), entry['sha256'], entry['bytes'])
        verify_inputs(expected)
        manifest = {'scope': state['scope'], 'protocol_sha256': protocol_sha,
            'source_review_sha256': review_sha, 'native_source_abstract_records': 1000,
            'unique_normalized_dois': 1000, 'split_ids': protocol['split_ids'],
            'normalized_doi_by_id': mapping, 'official_archives': records,
            'input_file_sha256': expected, 'all_saved_bytes_rehashed_after_all_copies': True,
            'semantic_annotation_read': False, 'label_statistics_computed': False,
            'model_training_or_inference_performed': False, 'paid_api_calls': 0,
            'independence_or_pretraining_disjointness_certified': False}
        manifest_path = checked_path(destination / 'source_acquisition_manifest.json')
        manifest_raw = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
        with manifest_path.open('xb') as output:
            output.write(manifest_raw)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        verify_saved(manifest_path, manifest_sha, len(manifest_raw))
        verify_inputs(expected)
        state.update(status='source_metadata_and_raw_byte_acquisition_completed_only',
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            manifest_path=str(manifest_path.relative_to(ROOT)), manifest_sha256=manifest_sha)
        save()
        print(json.dumps(state, indent=2))
    except Exception as error:
        state.update(status='source_acquisition_failed_preserved', exception_class=type(error).__name__,
            failed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
