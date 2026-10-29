"""Compare the direct reader with real LeRobot samples and audit full epochs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from direct_data import DirectEpisodes, build_transformed
from direct_loader import build_loader
from epoch_plan import EpochPlan, ExactEpochSampler
from settings import load_config


class SelectedLeRobotReference(LeRobotDataset):
    """Pinned LeRobot indexes selected bounds by ordinal, while rows keep sparse IDs.

    Adapt only the reference's boundary lookup; never modify LeRobot or data.
    """
    def _get_query_indices(self, idx, ep_idx):
        ordinal = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        return super()._get_query_indices(idx, ordinal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for experiment in ('a60', 'b80'):
        cfg = load_config(experiment)
        raw, transformed, config = build_transformed(cfg)
        norm = Path(cfg.actor.model.openpi_data.norm_stats_path)
        stats = json.loads(norm.read_text())['norm_stats']
        for key in ['state', 'actions']:
            for field in ['mean', 'std', 'q01', 'q99']:
                np.testing.assert_array_equal(getattr(config.norm_stats[key], field), stats[key][field])
        checks = 0
        # Same images, states, instructions and padded actions as native LeRobot,
        # including ALL tail positions of ALL episodes. No action-space conversion.
        for source in OmegaConf.to_container(cfg.data.sources, resolve=True):
            root = Path(source['path'])
            ids = [r['original_episode_id'] for r in raw.episodes if r['source'] == source['name']]
            reference = SelectedLeRobotReference(str(root), root=root, episodes=ids,
                       delta_timestamps={'actions': [i / 10 for i in range(10)]})
            positions = [r for r in raw.episodes if r['source'] == source['name']]
            offset = 0
            for rec in positions:
                original_position = next(i for i,r in enumerate(raw.episodes) if r is rec)
                global_start = raw.ends[original_position - 1] if original_position else 0
                for local in sorted(set([0, rec['length']//2, *range(max(0,rec['length']-10),rec['length'])])):
                    expected = reference[offset + local]
                    actual = raw[global_start + local]
                    np.testing.assert_array_equal(actual['state'], expected['state'].numpy())
                    np.testing.assert_array_equal(actual['actions'], expected['actions'].numpy())
                    assert actual['prompt'] == expected['task']
                    if local == 0:
                        for image in ['image','wrist_image']:
                            decoded = (expected[image].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                            np.testing.assert_array_equal(actual[image], decoded)
                    checks += 1
                offset += rec['length']
            assert offset == len(reference)
            print('LEROBOT_MATCH',experiment,source['name'],len(reference),flush=True)
            del reference
        plan = EpochPlan(len(raw), 2, cfg.actor.micro_batch_size,
                         cfg.actor.global_batch_size // (2*cfg.actor.micro_batch_size))
        for epoch in range(50):
            ranks = [ExactEpochSampler(len(raw),2,rank,seed=cfg.actor.seed) for rank in range(2)]
            for s in ranks:
                s.set_epoch(epoch)
            seen = sum([list(s) for s in ranks], [])
            assert len(seen) == len(raw) and sorted(seen) == list(range(len(raw)))
            # Every update's loss denominator counts each real sample, including the tail.
            consumed = sum(plan.global_samples(o,plan.update_batches(o))
                           for o in range(0,plan.batches_per_epoch,plan.accumulation))
            assert consumed == len(raw)
        loader, _, _, _ = build_loader(cfg, 2, 0)
        obs, actions, valid_samples = next(iter(loader))
        assert valid_samples == 2
        assert actions.shape == (2,10,32) and torch.isfinite(actions).all()
        assert obs.state.shape == (2,32)
        assert obs.tokenized_prompt.shape == (2,200)
        assert all(v.shape == (2,3,224,224) and torch.isfinite(v).all() for v in obs.images.values())
        assert obs.image_masks['base_0_rgb'].all() and obs.image_masks['left_wrist_0_rgb'].all()
        assert not obs.image_masks['right_wrist_0_rgb'].any()
        raw.assert_unchanged()
        reports[experiment] = dict(state='complete', dataset=raw.summary(),
            reference_windows_checked=checks, matches_native_lerobot=True,
            original_norm_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
            epoch_plan=plan.describe(), epochs=50, updates_50_epochs=50*plan.updates_per_epoch,
            exact_sampler_verified_all_50_epochs=True, source_data_unchanged=True)
        (args.output/f'{experiment}_data.json').write_text(json.dumps(reports[experiment],indent=2)+'\n')
        print('VALIDATED',experiment,checks,plan.describe(),flush=True)
        del raw, transformed, loader
    (args.output/'data_validation.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
