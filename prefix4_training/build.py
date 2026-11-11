"""Rebuild only four selected tail15 prefixes. Never restore removed rows."""
import argparse
import json
from pathlib import Path

from prefix_training.build import acquire, materialize
from . import spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    if args.offline:
        verified = json.loads((args.source_cache / 'hf_sources_verified.json').read_text())
        if verified['revision'] != spec.selection()['source_revision']:
            raise ValueError('Wrong immutable source revision')
        hashes = verified['files']
    else:
        hashes = acquire(args.source_cache, spec_module=spec)
    print(materialize(args.source_cache, args.output, hashes, spec_module=spec))


if __name__ == '__main__':
    main()
