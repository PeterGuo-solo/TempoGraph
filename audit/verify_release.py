"""Verify released bytes using only the Python standard library."""
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / 'audit/files.sha256.json').read_text(encoding='utf-8'))
    failures = []
    for row in rows:
        path = root / row['file']
        if not path.is_file():
            failures.append(row['file'] + ': missing')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
            failures.append(row['file'] + ': hash/size mismatch')
    print(f'Checked {len(rows)} files; {len(failures)} mismatches.')
    for failure in failures:
        print(failure)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
