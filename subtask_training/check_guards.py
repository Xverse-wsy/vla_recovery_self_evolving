"""Bounded regression checks for no-padding/index guards and quantile parity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .dataset import SubtaskDataset,normalized_actions
from .fetch import dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--recovery-pair',type=Path)
    args=ap.parse_args()
    root=args.output/'segmented'
    dataset=SubtaskDataset(root)
    # Corrupt an in-memory test copy only: the source files and dataset stay unchanged.
    w=dataset.windows[0]
    original=dict(w)
    table=dataset.numeric(w['parquet'])
    for invalid_start in (-1,table.num_rows-w['horizon']+1,table.num_rows-1):
        w['start']=invalid_start
        try:
            dataset[0]
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-bounds start was silently padded or wrapped')
    w.update(original)
    short=json.loads((root/'skipped_short_segments.json').read_text())
    assert all(s['segment_id'] not in {w['segment_id'] for w in dataset.windows} for s in short)
    norm=args.output/'assets/norm_stats.json'
    actions=dataset[0]['actions']
    normalized,mask,_=normalized_actions(actions,norm)
    assert normalized.dtype==np.float32 and normalized.shape==(10,32)
    assert mask[:,:7].all() and not mask[:,7:].any()
    assert not normalized[:,7:].any()
    # Check against the existing trainer's actual quantile implementation, if supplied.
    report=dict(state='passed',out_of_bounds_cases=3,short_segments_excluded=len(short),
                model_padding_dimensions_zero=True,model_padding_mask_false=True,
                existing_trainer_normalizer_compared=False)
    if args.recovery_pair:
        sys.path.insert(0,str(args.recovery_pair.resolve()))
        from recovery_pair.data import QuantileAssets
        assets=QuantileAssets(norm)
        largest=0.
        for i in range(len(dataset)):
            win=dataset.windows[i]
            raw=np.asarray(dataset.numeric(win['parquet'])['actions'].slice(win['start'],win['horizon']).to_pylist(),np.float32)
            actual=normalized_actions(raw,norm)[0][:,:7]
            largest=max(largest,float(np.max(np.abs(actual-assets.normalize('actions',raw)))))
        assert largest==0
        report.update(existing_trainer_normalizer_compared=True,normalizer_windows=len(dataset),
                      normalization_max_error=largest)
    dump(root/'guard_validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
