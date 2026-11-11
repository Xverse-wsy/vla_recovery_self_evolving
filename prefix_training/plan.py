"""CPU-only complete data, A60 recipe and fifty full epoch acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from direct_data import build_raw
from epoch_plan import EpochPlan, ExactEpochSampler
from settings import load_config
from subtask_training.fetch import dump
from subtask_training.training_data import audit_recipe, smoke_indices
from .validate import validate


def main(*, experiment='a60_prefix10', expected_windows=10619,
         expected_episodes=70, expected_actions=10709, bank_validator=validate):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--world-size', type=int, default=1)
    ap.add_argument('--global-batch-size', type=int)
    ap.add_argument('--micro-batch-size', type=int)
    a = ap.parse_args()
    cfg, base = load_config(experiment), load_config('a60')
    for c in (cfg, base):
        if a.global_batch_size is not None:
            c.actor.global_batch_size = a.global_batch_size
        if a.micro_batch_size is not None:
            c.actor.micro_batch_size = a.micro_batch_size
    parity = audit_recipe(cfg, base)
    batch, micro = int(cfg.actor.global_batch_size), int(cfg.actor.micro_batch_size)
    if min(batch, micro, a.world_size) < 1 or batch % (a.world_size * micro):
        raise ValueError('Global batch must divide world_size * microbatch')
    raw = build_raw(OmegaConf.to_container(cfg.data.sources, resolve=True), horizon=10)
    if (len(raw), len(raw.episodes), len(raw.tasks)) != (expected_windows, expected_episodes, 40):
        raise ValueError('Wrong data: expected unchanged A60 plus selected fixed prefixes')
    bank_audit = bank_validator(raw.root, write_report=False)
    plan = EpochPlan(len(raw), a.world_size, micro, batch // (a.world_size * micro))
    # Original complete-success episode tails must retain normal A60 repetition.
    for i, rec in enumerate(raw.base.episodes):
        start = raw.base.ends[i - 1] if i else 0
        for local in (0, rec['length'] - 1):
            sample = raw[start + local]
            positions = np.minimum(local + np.arange(10), rec['length'] - 1)
            np.testing.assert_array_equal(sample['actions'], rec['actions'][positions])
            if sample['prompt'] != rec['task']:
                raise ValueError('A60 instruction changed')
    batch_data = next(iter(DataLoader(Subset(raw, [0, 6632, 9864, len(raw)-1]), batch_size=4, num_workers=0)))
    if tuple(batch_data['actions'].shape) != (4, 10, 7):
        raise ValueError('Mixed-source batch shape mismatch')
    epochs = int(cfg.runner.max_epochs)
    for epoch in range(epochs):
        indices = []
        for rank in range(a.world_size):
            sampler = ExactEpochSampler(len(raw), a.world_size, rank, seed=cfg.actor.seed, micro_batch_size=micro)
            sampler.set_epoch(epoch)
            indices.extend(i for i in sampler if i >= 0)
        if sorted(indices) != list(range(len(raw))):
            raise ValueError('Epoch lost or duplicated training windows')
        if sum(plan.global_samples(o, plan.update_batches(o))
               for o in range(0, plan.batches_per_epoch, plan.accumulation)) != len(raw):
            raise ValueError('Logical batch/tail weights mismatch')
    probes = smoke_indices(len(raw), len(raw.base), raw.base.source_frames['author40'])
    if not any(i >= 9864 for i in probes):
        raise ValueError('New prefixes absent from smoke')
    raw.assert_unchanged()
    a.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, a.output / 'resolved_config.yaml')
    dump(a.output / 'dataset_manifest.json', raw.summary())
    report = dict(state='passed', experiment=experiment, episodes=len(raw.episodes), tasks=40,
        training_windows=len(raw), unique_action_rows=expected_actions,
        source_windows={'author40': 6632, 'autonomous20': 3232, raw.name: len(raw.clips)},
        prefix_action_rows=bank_audit['unique_action_rows'], prefix_segments=len(raw.clips.segments), total_original_trim=15,
        prefix_padding=0, parent_instructions_only=True, a60_recipe_parity=parity,
        bank_audit=bank_audit, epoch_plan=plan.describe(), epochs=epochs,
        updates=epochs * plan.updates_per_epoch, all_50_epoch_permutations_checked=True,
        smoke_source_indices=probes, gpu_used=False, model_training_performed=False)
    dump(a.output / 'preflight.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
