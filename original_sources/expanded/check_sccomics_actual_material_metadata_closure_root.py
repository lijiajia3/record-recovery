"""Independently expand fixed actual receipts without reading corpus/tensor bytes.

This preparatory check creates no complete scientific freeze or fit gate. It
checks receipt identities, complete split/path populations, source/cache links,
and current no-follow regular-file metadata. Later captured stage validation
must still check every original file's content hash.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = {
    'data/sccomics_round4/source_v3/source_acquisition_manifest.json':
        'a38f1637af74b15246b6b028d3d58b5836094e905d25845e5922668031fee1c7',
    'results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json':
        'fe7b66198773d49e7652c7be64d3f5923fdba27c528cb73303f0b95a36194139',
    'data/local_baselines/matscibert/source_manifest.json':
        'ffdcf65ccf91c8cce33d380701928fa039ffe20edeb3375b68b5bbf6152c6b0c',
}
SPLITS = {'train': list(range(201, 1001)),
          'dev': list(range(101, 201)), 'test': list(range(1, 101))}
COLD_NAMES = {'README.md', 'config.json', 'model.safetensors',
              'special_tokens_map.json', 'tokenizer.json',
              'tokenizer_config.json', 'vocab.txt'}


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate receipt JSON key.')
        result[key] = value
    return result


def relative_path(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    name = path.as_posix()
    require(name and name == str(path) and
            not any(part in {'.', '..'} for part in path.parts),
            'Noncanonical receipt material path.')
    return name


def nofollow_stat(name):
    require(not any(part.is_symlink() for part in (ROOT, *ROOT.parents)),
            'Project-root alias.')
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = Path(name).parts
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY |
                              os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        value = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        require(stat.S_ISREG(value.st_mode), 'Regular original material required.')
        return value
    finally:
        os.close(fd)


def binding(entry, duty):
    digest = entry['sha256']
    size = entry['bytes']
    require(type(digest) is str and len(digest) == 64 and
            all(char in '0123456789abcdef' for char in digest), 'Invalid source SHA.')
    require(type(size) is int and size >= 0, 'Invalid source size.')
    return {'sha256': digest, 'size_bytes': size, 'duty': duty}


def main():
    receipts = {}
    receipt_bindings = {}
    for name, expected in RECEIPTS.items():
        nofollow_stat(name)
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'Fixed receipt changed.')
        receipts[name] = json.loads(raw, object_pairs_hook=unique_object)
        receipt_bindings[name] = {'sha256': expected, 'size_bytes': len(raw)}
    acquisition, cache, cold = (receipts[name] for name in RECEIPTS)
    require(acquisition['split_ids'] == SPLITS and
            acquisition['native_source_abstract_records'] == 1000,
            'Fixed actual source population/split changed.')
    materials = {}
    for archive in acquisition['official_archives'].values():
        for entry in archive['members'].values():
            duty = {'raw_source_document': 'text',
                    'unparsed_annotation_bytes': 'annotation'}[entry['category']]
            name = relative_path(entry['canonical_path'])
            require(name not in materials, 'Duplicate receipt material.')
            materials[name] = binding(entry, duty)
    for duty, directory, suffix in [('text', 'raw_text', 'txt'),
                                    ('annotation', 'raw_annotations', 'ann')]:
        expected = {f'data/sccomics_round4/source_v3/{directory}/{i:04}.{suffix}'
                    for i in range(1, 1001)}
        require({name for name, value in materials.items() if value['duty'] == duty}
                == expected, 'Complete native original path population differs.')
    require(cache['status'] == 'completed_cold_source_text_cache_only' and
            cache['complete_native_records'] == 1000 and
            cache['all_captured_source_model_and_cache_hashes_size_final_checked'] is True,
            'Actual source-cache receipt incomplete.')
    require(len(cache['cache_bindings']) == 1000, 'Complete cache receipt count.')
    ids = set()
    for entry in cache['cache_bindings']:
        identifier = entry['native_id']
        split = entry['split']
        require(type(identifier) is int and identifier in SPLITS[split] and
                identifier not in ids, 'Cache ID/fold population differs.')
        ids.add(identifier)
        name = relative_path(entry['path'])
        require(name == f'results/local_baseline/sccomics_native_v1/source_cache/{split}/{identifier:04}.pt',
                'Exact source-cache native path differs.')
        require(entry['source_sha256'] == materials[
            f'data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt']['sha256'],
            'Source/cache identity link differs.')
        require(name not in materials, 'Duplicate cache material.')
        materials[name] = binding(entry, 'cache')
    require(ids == set(range(1, 1001)), 'Incomplete unique cache native IDs.')
    require(cold['revision'] == 'ced9d8f5f208712c4a90f98a246fe32155b29995' and
            cold['base_encoder_not_task_finetuned_checkpoint'] is True and
            len(cold['files']) == 7 and
            {entry['name'] for entry in cold['files']} == COLD_NAMES,
            'Original cold model receipt differs.')
    for entry in cold['files']:
        name = 'data/local_baselines/matscibert/' + entry['name']
        require(name not in materials, 'Duplicate cold material.')
        materials[name] = binding(entry, 'cold')
    require(len(materials) == 3007, 'Exact expanded material population differs.')
    metadata = {}
    for name, expected in materials.items():
        value = nofollow_stat(name)
        require(value.st_size == expected['size_bytes'], 'Current material size differs.')
        metadata[name] = {'device': value.st_dev, 'inode': value.st_ino,
                          'size_bytes': value.st_size,
                          'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}
    for name, expected in receipt_bindings.items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected['sha256'] and
                len(raw) == expected['size_bytes'], 'Receipt changed during expansion.')
    report = {
        'identity': 'ACTUAL3007_RECEIPT_EXPANSION_AND_CURRENT_METADATA_ONLY_NOT_AUTHORITY',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': True, 'expanded_material_count': len(materials),
        'complete_native_split_ids': SPLITS, 'receipt_bindings': receipt_bindings,
        'input_material_bindings_from_fixed_original_receipts': materials,
        'current_regular_no_follow_metadata': metadata,
        'actual_material_content_hashes_freshly_recomputed': False,
        'corpus_tensor_cold_or_annotation_content_opened': False,
        'annotation_semantics_or_model_execution_or_final_authority': False,
        'later_captured_stage_content_hash_validation_still_required': True,
        'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = ROOT / 'research/round4_sccomics_actual_material_metadata_closure_root_check.json'
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'passed': True, 'material_count': len(materials),
                      'actual_material_content_read': False, 'fitting_authority': False}))


if __name__ == '__main__':
    main()
