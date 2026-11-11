import argparse
import json
from pathlib import Path

from prefix_training.validate import validate as validate_prefixes
from .dataset import FourPrefixBank


def validate(root, source_cache=None, norm_stats=None, write_report=True):
    return validate_prefixes(root, source_cache, norm_stats, write_report,
                             bank_class=FourPrefixBank)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Validate all four unchanged tail15 prefixes')
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--source-cache', type=Path)
    ap.add_argument('--norm-stats', type=Path)
    args = ap.parse_args()
    print(json.dumps(validate(args.dataset, args.source_cache, args.norm_stats), indent=2))
