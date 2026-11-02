"""CPU parity of real Stage1 inputs/targets against the original A60 data path."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


def validate(prepared, checkpoint, norm_stats):
    import jax
    from openpi.models.model import Observation
    from openpi.training.data_loader import _collate_fn, transform_dataset
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.env_io import EnvIO
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
    from rlinf.models.embodiment.openpi.modules import model
    from .openpi_backend import ObservationLoader
    from .schema import load_windows
    package = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package))
    from direct_data import DirectEpisodes
    data_kwargs = dict(norm_stats_path=str(norm_stats), extra_delta_transform=False)
    config = get_openpi_config('pi05_libero', model_path=str(checkpoint),
        batch_size=1, data_kwargs=data_kwargs)
    data_config = config.data.create(config.assets_dirs, config.model)

    class Policy(EnvIO):
        device = torch.device('cpu')
        config_name = 'pi05_libero'
        state_indices = None
    policy = Policy()
    inputs, outputs = build_openpi_transforms(str(checkpoint), 'pi05_libero', data_kwargs=data_kwargs)
    policy.setup_transforms(inputs, outputs)
    loader = ObservationLoader(SimpleNamespace(policy=policy))
    records = load_windows(Path(prepared) / 'training_pool.jsonl')
    probes = []
    for category in sorted({r.bank_label['source_category'] for r in records}):
        group = [r for r in records if r.bank_label['source_category'] == category]
        probes.append(next(r for r in group if not r.source['fm_repeated_steps']))
        tails = [r for r in group if r.source['fm_repeated_steps']]
        if tails:
            probes.append(tails[-1])
    readers, reports = {}, []
    for record in probes:
        root = Path(record.source['dataset'])
        if str(root) not in readers:
            info = json.loads((root / 'meta/info.json').read_text())
            source = dict(name='original_A60_reader', path=str(root), episodes=info['total_episodes'])
            if record.controller == 'expert':
                source.update(episodes_file='episodes.json', episodes=40)
            raw = DirectEpisodes([source], horizon=record.spec.horizon)
            readers[str(root)] = raw, transform_dataset(raw, data_config)
        raw, transformed = readers[str(root)]
        ref = record.image_refs['front']
        episode = next(i for i, ep in enumerate(raw.episodes) if ep['path'] == ref['parquet'])
        index = (raw.ends[episode-1] if episode else 0) + ref['row']
        batch = jax.tree.map(torch.as_tensor, _collate_fn([transformed[index]]))
        # Official OpenPI containers use BCHW; EnvIO uses BHWC. The pinned
        # compute_loss canonicalizes both before applying the same augmentation.
        a = model.Observation.from_observation_like(Observation.from_dict(batch))
        a = model.preprocess_observation(a, train=False)
        b = model.preprocess_observation(loader([record]), train=False)
        # OpenPI's NumPy quantile transform returns float64; EnvIO explicitly
        # casts state to float32. Compare in the model's state precision, while
        # retaining the original storage difference in the audit report.
        state_storage_error = float((a.state.double()-b.state.double()).abs().max())
        state_dtypes = dict(a60=str(a.state.dtype), stage1=str(b.state.dtype))

        def equal(x, y):
            if x is None or y is None:
                return x is y
            if isinstance(x, dict):
                return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
            return torch.equal(torch.as_tensor(x), torch.as_tensor(y))
        fields = ['images', 'image_masks', 'state', 'tokenized_prompt', 'tokenized_prompt_mask',
                  'token_ar_mask', 'token_loss_mask']
        for field in fields:
            left, right = getattr(a, field), getattr(b, field)
            if field == 'state':
                left, right = left.float(), right.float()
            if not equal(left, right):
                left, right = getattr(a, field), getattr(b, field)
                detail = {'sample_id': record.sample_id, 'category': record.bank_label['source_category']}
                if field == 'state':
                    left, right = torch.as_tensor(left), torch.as_tensor(right)
                    detail.update(a60=left.tolist(), stage1=right.tolist(),
                                  a60_dtype=str(left.dtype), stage1_dtype=str(right.dtype))
                raise ValueError('Stage1 observation differs from A60: '+field+' '+json.dumps(detail))
        target = np.asarray(batch['actions'])[0]
        error = float(np.abs(target-record.payload()['actions']).max())
        if error > 2e-6:
            raise ValueError('Stage1 normalized action target differs from A60')
        reports.append(dict(sample_id=record.sample_id, category=record.bank_label['source_category'],
            repeated_steps=record.source['fm_repeated_steps'], native_preprocessed_observation_equal=True,
            state_fp32_equal=True, state_storage_dtypes=state_dtypes,
            state_storage_max_abs_error=state_storage_error,
            normalized_action_max_abs_error=error))
    return dict(state='complete', real_data=True, model_weights_loaded=False,
        reference='original DirectEpisodes + official OpenPI transform_dataset',
        state_comparison='float32 exact; original float64/float32 storage error recorded separately',
        image_layout_canonicalization='native preprocess_observation: BCHW/BHWC -> BHWC', probes=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'checkpoint', 'norm-stats', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.prepared, args.checkpoint, args.norm_stats)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
