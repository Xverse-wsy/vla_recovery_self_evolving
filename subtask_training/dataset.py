"""Dense no-padding dataset with the raw keys expected by native SFT repack transforms."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def decode(value):
    if not isinstance(value, dict) or not value.get('bytes'):
        raise ValueError('Self-contained embedded RGB bytes required')
    with Image.open(io.BytesIO(value['bytes'])) as im:
        if im.mode != 'RGB':
            raise ValueError('Expected original RGB, no silent mode/orientation conversion')
        return np.asarray(im).copy()


class SubtaskDataset:
    """PyTorch map-style protocol; __getitem__ returns raw, unnormalized data.

    Only starts explicitly present in windows.jsonl are accessible. Intermediate
    subtask clips never repeat; a recorded FINAL task-success endpoint may repeat.
    Pass each result through the SAME model data transforms as the normal trainer.
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        report = json.loads((self.root / 'build_state.json').read_text())
        if report['state'] != 'complete':
            raise ValueError('Dataset build incomplete')
        self.windows = [json.loads(s) for s in (self.root / 'windows.jsonl').read_text().splitlines() if s.strip()]
        if len(self.windows) != report['training_windows']:
            raise ValueError('Window index length mismatch')
        self.segments={r['segment_id']:r for r in (json.loads(s) for s in (self.root/'meta/provenance.jsonl').read_text().splitlines() if s.strip())}
        seen=set()
        for w in self.windows:
            if w['sample_id'] in seen:
                raise ValueError('Duplicate window')
            seen.add(w['sample_id'])
            meta=self.segments[w['segment_id']]
            if (w['task_instruction']!=meta['task_instruction'] or w['parquet']!=meta['parquet']
                    or w['dataset_episode_index']!=meta['dataset_episode_index']):
                raise ValueError('Window instruction/episode differs from original segment')

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=16)
    def parquet_file(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Parquet must be inside dataset root')
        return pq.ParquetFile(path, memory_map=True)

    @lru_cache(maxsize=68)
    def numeric(self, relative):
        return self.parquet_file(relative).read(columns=['state', 'actions'])

    @lru_cache(maxsize=8)
    def rgb_group(self, relative, group):
        # Bounded ~64-row RGB cache, rather than rereading whole episodes on shuffle.
        return self.parquet_file(relative).read_row_group(group, columns=['image','wrist_image'])

    def __getitem__(self, i):
        w = self.windows[i]
        table = self.numeric(w['parquet'])
        start, h = w['start'], w['horizon']
        meta=self.segments[w['segment_id']]
        n=table.num_rows
        real=min(h,n-start)
        repeat=w.get('tail_policy')=='repeat_last'
        if (h<=0 or start<0 or start>=n or (real<h and not repeat)
                or repeat and (meta.get('padding_policy')!='repeat_last' or not meta.get('endpoint_has_recorded_task_success'))
                or w.get('real_steps',real)!=real or w.get('repeated_steps',h-real)!=h-real):
            raise ValueError('Invalid window: intermediate tails must not repeat or cross boundaries')
        actions=np.asarray(table['actions'].slice(start,real).to_pylist(),np.float32)
        if real<h:
            actions=np.concatenate([actions,np.repeat(actions[-1:],h-real,axis=0)])
        x = dict(prompt=w['task_instruction'],
                 actions=actions)
        x['state'] = np.asarray(table['state'][start].as_py(), np.float32)
        pf = self.parquet_file(w['parquet'])
        offset = 0
        for group in range(pf.metadata.num_row_groups):
            size = pf.metadata.row_group(group).num_rows
            if start < offset + size:
                images = self.rgb_group(w['parquet'],group)
                x['image'] = decode(images['image'][start-offset].as_py())
                x['wrist_image'] = decode(images['wrist_image'][start-offset].as_py())
                break
            offset += size
        return self.transform(x) if self.transform else x

    def metadata(self, i):
        # Metadata is separate so labels and future evidence never become model inputs.
        return dict(self.windows[i])


def env_batch(raw_batch):
    """Use policy.env_obs_to_observation(env_batch(batch)) in the RLinf backend."""
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    return dict(states=array(raw_batch['state']),
                main_images=array(raw_batch['image']),
                wrist_images=array(raw_batch['wrist_image']),
                task_descriptions=list(raw_batch['prompt']))


def normalized_actions(raw_actions, norm_stats_path, model_dim=32):
    """RLinf π0.5 quantile action normalization; NOT to be applied twice.

    Observations are normalized/tokenized by the policy's native EnvIO. This
    function is for trainers using that EnvIO plus compute_loss directly.
    """
    raw = np.asarray(raw_actions, np.float32)
    if raw.ndim < 2 or raw.shape[-1] != 7 or model_dim < 7 or not np.isfinite(raw).all():
        raise ValueError('Expected finite OSC actions [..., H, 7]')
    p = Path(norm_stats_path)
    stats = json.loads(p.read_text())
    actions = stats.get('norm_stats', stats)['actions']
    low, high = (np.asarray(actions[k], np.float32)[:7] for k in ('q01', 'q99'))
    if low.shape != (7,) or high.shape != (7,) or not np.isfinite([low,high]).all() or (high < low).any():
        raise ValueError('Invalid base-model normalization asset')
    out = np.zeros(raw.shape[:-1] + (model_dim,), np.float32)
    out[..., :7] = (raw - low) / (high - low + 1e-6) * 2 - 1
    mask = np.zeros_like(out, bool)
    mask[..., :7] = True
    return out, mask, hashlib.sha256(p.read_bytes()).hexdigest()
