"""Finite, resumable loader using the original OpenPI transformations."""
import jax
import torch
from openpi.models.model import Observation
from openpi.training.data_loader import _collate_fn, _worker_init_fn
from torchdata.stateful_dataloader import StatefulDataLoader

from direct_data import build_transformed
from epoch_plan import EpochPlan, ExactEpochSampler


class MaskedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # -1 is a communication-only placeholder on an otherwise empty rank.
        # It contributes exactly zero loss and is not a training example.
        return dict(self.dataset[max(0, index)], _valid_sample=int(index >= 0))


class DirectLoader:
    def __init__(self, dataset, config, cfg, world_size, rank):
        self.config = config
        self.epoch = 0
        self.sampler = ExactEpochSampler(len(dataset), world_size, rank, seed=cfg.actor.seed,
                                         micro_batch_size=cfg.actor.micro_batch_size)
        workers = int(cfg.data.num_workers)
        self.raw = StatefulDataLoader(
            MaskedDataset(dataset), batch_size=cfg.actor.micro_batch_size, sampler=self.sampler,
            num_workers=workers, collate_fn=_collate_fn, worker_init_fn=_worker_init_fn,
            drop_last=False, persistent_workers=workers > 0,
            multiprocessing_context='spawn' if workers else None,
            generator=torch.Generator().manual_seed(cfg.actor.seed),
        )

    def __len__(self):
        return len(self.raw)

    def data_config(self):
        return self.config

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        for batch in self.raw:
            batch = jax.tree.map(torch.as_tensor, batch)
            yield Observation.from_dict(batch), batch['actions'], int(batch['_valid_sample'].sum())

    def state_dict(self):
        return dict(epoch=self.epoch, loader=self.raw.state_dict())

    def load_state_dict(self, state):
        self.set_epoch(state['epoch'])
        self.raw.load_state_dict(state['loader'])


def build_loader(cfg, world_size, rank):
    raw, dataset, config = build_transformed(cfg)
    if cfg.data.get('validation_indices') is not None:
        dataset = torch.utils.data.Subset(dataset, list(cfg.data.validation_indices))
    accumulation = cfg.actor.global_batch_size // (world_size * cfg.actor.micro_batch_size)
    plan = EpochPlan(len(dataset), world_size, cfg.actor.micro_batch_size, accumulation)
    return DirectLoader(dataset, config, cfg, world_size, rank), config, raw, plan
