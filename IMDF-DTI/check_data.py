"""Check local benchmark files against the archived input identities (standard library only)."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('DataSets'))
    parser.add_argument('--dataset', choices=['DrugBank', 'BIOSNAP', 'Davis'])
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / 'DataSets/manifest.json').read_text())
    ok = True
    for name in ([args.dataset] if args.dataset else manifest):
        path = args.data_dir / f'{name}.txt'
        if not path.is_file():
            print(f'{name}: MISSING ({path}); see DataSets/README.md')
            ok = False
            continue
        raw = path.read_bytes()
        rows = raw.decode('utf-8').strip().split('\n')
        digest = hashlib.sha256(raw).hexdigest()
        match = digest == manifest[name]['sha256']
        print(f'{name}: rows={len(rows)} SHA256={digest} matches_archive={match}')
        ok = ok and match
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
