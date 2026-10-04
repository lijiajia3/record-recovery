"""Read-only, bounded-memory content verification of an evidence ZIP."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP member'
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                   and '\\' not in n for n in names), 'Unsafe ZIP member'
        manifest = json.loads(z.read('PACKET_MANIFEST.json'))
        assert set(names) == set(manifest['files']) | {'PACKET_MANIFEST.json'}
        total = 0
        for name, entry in manifest['files'].items():
            h, size = hashlib.sha256(), 0
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    size += len(block)
            assert h.hexdigest() == entry['sha256'] and size == entry['bytes'], name
            total += size
        return {'packet': str(path.resolve()), 'scope': manifest['scope'],
                'members_including_manifest': len(names), 'verified_content_files': len(manifest['files']),
                'verified_uncompressed_bytes': total, 'all_content_hashes_and_sizes_match': True,
                'duplicate_or_unsafe_members': False, 'model_inference_or_training_repeated': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('packet', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    report = verify(args.packet)
    if args.out:
        assert not args.out.exists(), 'Do not replace previous verification'
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
