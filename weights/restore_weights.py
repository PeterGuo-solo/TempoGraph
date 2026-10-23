"""Restore chunked weights after git lfs pull, with SHA256 checks."""
import hashlib
import json
import os
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'weights/shards/manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        target = root / item['restore_path']
        if target.exists():
            if file_sha(target) != item['sha256']:
                raise SystemExit('Existing weight has a different hash: ' + str(target))
            print('Already verified:', item['restore_path'])
            continue
        for part in item['parts']:
            source = root / part['file']
            if not source.is_file() or source.stat().st_size != part['bytes'] or file_sha(source) != part['sha256']:
                raise SystemExit('Missing/invalid weight chunk; run git lfs pull: ' + part['file'])
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.assembling')
        created = False
        try:
            with temp.open('xb') as output:
                created = True
                for part in item['parts']:
                    with (root / part['file']).open('rb') as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            output.write(chunk)
            if temp.stat().st_size != item['bytes'] or file_sha(temp) != item['sha256']:
                raise SystemExit('Reconstructed weight failed verification: ' + item['restore_path'])
            if target.exists():
                raise SystemExit('Target appeared during assembly; refusing to overwrite it.')
            os.replace(temp, target)
        finally:
            if created and temp.exists():
                temp.unlink()
        print('Restored:', item['restore_path'])


if __name__ == '__main__':
    main()
