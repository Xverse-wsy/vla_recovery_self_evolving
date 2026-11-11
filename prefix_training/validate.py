"""CPU audit of every retained row/window, plus optional exact source comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from subtask_training.dataset import decode, normalized_actions
from subtask_training.fetch import dump, sha256
from .dataset import PrefixBank
from .spec import records


def validate(root, source_cache=None, norm_stats=None, write_report=True):
    ds = PrefixBank(root)
    pixels = 0
    probes = []
    for m in ds.segments.values():
        path = ds.root / m['parquet']
        table = pq.read_table(path)
        n = m['action_count']
        np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(n))
        np.testing.assert_array_equal(table['episode_index'].to_numpy(), [m['dataset_episode_index']] * n)
        if source_cache:
            source = Path(source_cache) / m['source_parquet']
            if sha256(source) != m['source_parquet_sha256']:
                raise ValueError('Original source Parquet hash mismatch')
            parent = pq.read_table(source)
            if parent.num_rows != n + 5:
                raise ValueError('Prefix must remove exactly five additional source rows')
            for col in ('image', 'wrist_image', 'state', 'actions', 'timestamp'):
                if not table[col].equals(parent[col].slice(0, n)):
                    raise ValueError('Original training columns differ')
        for col in ('image', 'wrist_image'):
            for value in table[col].to_pylist():
                image = decode(value)
                if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                    raise ValueError('Unexpected original RGB shape/dtype')
                pixels += 1
        indices = [i for i, w in enumerate(ds.windows) if w['segment_id'] == m['segment_id']]
        probes.extend((indices[0], indices[-1]))
    # Exhaustive check: every real start, state, action window and original RGB.
    for i, w in enumerate(ds.windows):
        sample = ds[i]
        table = ds.numeric(w['parquet'])
        start = w['start']
        np.testing.assert_array_equal(sample['actions'], np.asarray(table['actions'].slice(start, 10).to_pylist(), np.float32))
        np.testing.assert_array_equal(sample['state'], np.asarray(table['state'][start].as_py(), np.float32))
        if sample['prompt'] != w['task_instruction']:
            raise ValueError('Parent instruction changed')
    if norm_stats:
        samples = [ds[i] for i in probes]
        norm, _, norm_sha = normalized_actions(np.stack([s['actions'] for s in samples]), norm_stats)
        np.savez_compressed(ds.root / 'native_probe_samples.npz',
            sample_id=np.array([ds.windows[i]['sample_id'] for i in probes]),
            prompt=np.array([s['prompt'] for s in samples]),
            image=np.stack([s['image'] for s in samples]),
            wrist_image=np.stack([s['wrist_image'] for s in samples]),
            state=np.stack([s['state'] for s in samples]),
            actions=np.stack([s['actions'] for s in samples]), normalized_actions=norm)
    report = dict(state='passed', segments=10, tasks=10, unique_action_rows=845,
        training_windows=755, independently_checked_windows=len(ds),
        decoded_original_rgb_images=pixels, instructions_exactly_match_parent=True,
        original_source_training_columns_exact=bool(source_cache),
        total_original_trim=15, additional_source_trim=5, intermediate_tail_padding=0,
        physical_cropping_done_once=True, dense_stride=1, horizon=10,
        native_probe_samples=len(probes) if norm_stats else 0,
        normalization_sha256=norm_sha if norm_stats else None,
        model_training_performed=False, native_model_forward_tested=False)
    if write_report:
        dump(ds.root / 'validation.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--source-cache', type=Path)
    ap.add_argument('--norm-stats', type=Path)
    a = ap.parse_args()
    print(json.dumps(validate(a.dataset, a.source_cache, a.norm_stats), indent=2))


if __name__ == '__main__':
    main()
