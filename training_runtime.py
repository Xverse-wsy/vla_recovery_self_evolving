"""External training adapters: resumable data, audited updates, atomic retention."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import random
import numpy as np

import jax
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from openpi.models.model import Observation

from rlinf.runners.sft_runner import SFTRunner
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from epoch_plan import EpochPlan, ExactEpochSampler


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class ResumableObservations:
    """Keep official transforms/collation, add finite epochs and loader state."""
    def __init__(self, official, sampler=None, drop_last=None):
        raw = official._data_loader._data_loader
        self.official = official
        self.epoch = 0
        self.raw = StatefulDataLoader(
            raw.dataset, batch_size=raw.batch_size, sampler=raw.sampler if sampler is None else sampler,
            num_workers=raw.num_workers, collate_fn=raw.collate_fn,
            worker_init_fn=raw.worker_init_fn, drop_last=raw.drop_last if drop_last is None else drop_last,
            generator=raw.generator, multiprocessing_context=raw.multiprocessing_context,
            persistent_workers=raw.persistent_workers,
        )
        self.sampler = self.raw.sampler

    def __len__(self):
        return len(self.raw)

    def data_config(self):
        return self.official.data_config()

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for batch in self.raw:
            batch = jax.tree.map(torch.as_tensor, batch)
            yield Observation.from_dict(batch), batch['actions']

    def state_dict(self):
        return dict(epoch=getattr(self.sampler, 'epoch', self.epoch), loader=self.raw.state_dict())

    def load_state_dict(self, state):
        self.set_epoch(state['epoch'])
        self.raw.load_state_dict(state['loader'])


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        if torch.is_tensor(tensor):
            data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(data))
        else:
            digest.update(repr(tensor).encode())
    return digest.hexdigest()


class TrainingWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        from direct_loader import build_loader
        loader, config, self.direct_dataset, self.epoch_plan = build_loader(
            self.cfg, self._world_size, self._rank)
        return loader, config

    def get_max_steps_per_epoch(self):
        if getattr(self, 'epoch_plan', None) is not None:
            return self.epoch_plan.updates_per_epoch
        return super().get_max_steps_per_epoch()

    def model_provider_func(self):
        original = torch.nn.Module.load_state_dict
        checkpoint = {}

        def audited(module, state, *args, **kwargs):
            result = original(module, state, *args, **kwargs)
            if type(module).__name__ == 'Pi0':
                checkpoint.update(keys=len(state), missing=list(result.missing_keys),
                                  unexpected=list(result.unexpected_keys))
            return result

        torch.nn.Module.load_state_dict = audited
        try:
            model = super().model_provider_func()
        finally:
            torch.nn.Module.load_state_dict = original
        assert checkpoint and not checkpoint['missing'] and not checkpoint['unexpected'], checkpoint
        self.checkpoint_audit = checkpoint
        return model

    def init_worker(self):
        from gpu_budget import set_limit, check_headroom
        set_limit(float(self.cfg.actor.gpu_memory_limit_gib))
        free, _ = torch.cuda.mem_get_info()
        minimum = float(self.cfg.actor.gpu_memory_limit_gib + self.cfg.actor.gpu_headroom_gib + 1)
        assert free / 2**30 >= minimum, f'Worker needs {minimum} GiB initially, found {free / 2**30:.2f}'
        super().init_worker()
        # Upstream seeds the sampler but does not seed SFT flow noise in Ray
        # workers. Make both experiments start from the same per-rank RNGs.
        seed = int(self.cfg.actor.seed) + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.cpu_gradient_group = (torch.distributed.new_group(backend='gloo')
                                   if self.cfg.actor.fsdp_config.cpu_offload else None)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.metadata = dict(rank=self._rank, pid=os.getpid(),
                             device_uuid=str(props.uuid), checkpoint=self.checkpoint_audit,
                             trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                             batches_per_epoch=len(self.data_loader),
                             gradient_accumulation=self.gradient_accumulation)
        self.metadata.update(gpu_allocator_limit_gib=self.cfg.actor.gpu_memory_limit_gib,
                             gpu_free_headroom_gib=self.cfg.actor.gpu_headroom_gib,
                             cpu_offload=bool(self.cfg.actor.fsdp_config.cpu_offload), flow_noise_seed=seed)
        if self.epoch_plan is not None:
            self.metadata['epoch_plan'] = self.epoch_plan.describe()
            assert len(self.data_loader) == self.epoch_plan.batches_per_epoch
        write_json(Path(self.cfg.runner.logger.log_path) / f'worker_{self._rank}.json', self.metadata)

    def get_train_model_output(self, batch):
        result = super().get_train_model_output(batch[:2])
        if not math.isfinite(float(result[0].detach())):
            raise RuntimeError('Non-finite training loss')
        if getattr(self, '_update_global_samples', None) is not None:
            # Upstream divides by accumulation, and replica gradients are averaged.
            # Weight local mean losses by sample count, including the uneven 2+1 tail.
            local_samples = int(batch[2]) if len(batch) == 3 else batch[1].shape[0]
            weight = self._world_size * local_samples * self.gradient_accumulation / self._update_global_samples
            metrics = dict(result[1])
            metrics['loss'] *= weight
            result = (result[0] * weight, metrics)
        return result

    def optimizer_step(self):
        # The current NO_SHARD upstream path produced different replica hashes
        # after one update. Average all full-replica gradients before clipping;
        # averaging an already synchronized gradient is idempotent. This also
        # covers parameters whose gradients bypass upstream FSDP hooks.
        assert self.cfg.actor.fsdp_config.sharding_strategy == 'no_shard'
        pending = []
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if self.optimizer_steps == 0:
            signatures = [None] * self._world_size
            signature = [(tuple(g.shape), str(g.dtype)) for g in gradients]
            torch.distributed.all_gather_object(signatures, signature)
            assert all(s == signatures[0] for s in signatures), 'Gradient layouts differ'
        for grad in gradients:
            pending.append(torch.distributed.all_reduce(grad, group=self.cpu_gradient_group if grad.device.type == 'cpu' else None, async_op=True))
        for handle in pending:
            handle.wait()
        for grad in gradients:
            grad.mul_(1.0 / self._world_size)
        result = super().optimizer_step()
        if not math.isfinite(float(result[0])):
            raise RuntimeError('Non-finite gradient norm')
        return result

    def audit_state(self, include_optimizer=False, require_synced=True):
        model_digest = tensor_digest(self.model.named_parameters())
        state = dict(rank=self._rank, optimizer_steps=self.optimizer_steps,
                     model_digest=model_digest,
                     scheduler=self.lr_scheduler.state_dict(),
                     data_epoch=self._data_epoch, data_offset=self._data_iter_offset)
        if include_optimizer:
            values = ((f'{index}/{key}', value)
                      for index, entry in enumerate(self.optimizer.state.values())
                      for key, value in sorted(entry.items()))
            state['optimizer_digest'] = tensor_digest(values)
        states = [None] * self._world_size
        torch.distributed.all_gather_object(states, state)
        if require_synced:
            assert len({s['model_digest'] for s in states}) == 1, f'Rank weights differ: {states}'
        return state

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        torch.save(dict(loader=self.data_loader.state_dict(),
                        epoch_plan=None if self.epoch_plan is None else self.epoch_plan.describe(),
                        dataset_identity=self.direct_dataset.summary(),
                        sharding_strategy=self.cfg.actor.fsdp_config.sharding_strategy,
                        cpu_offload=bool(self.cfg.actor.fsdp_config.cpu_offload),
                        data_epoch=self._data_epoch, data_offset=self._data_iter_offset,
                        optimizer_steps=self.optimizer_steps,
                        rng=get_rng_state(), scaler=self.grad_scaler.state_dict()),
                   Path(save_path) / f'input_rank_{self._rank}.pt')
        torch.distributed.barrier()

    def load_checkpoint(self, load_path):
        state = torch.load(Path(load_path) / f'input_rank_{self._rank}.pt',
                           weights_only=False, map_location='cpu')
        assert state['dataset_identity'] == self.direct_dataset.summary(), 'Checkpoint dataset mismatch'
        assert state['sharding_strategy'] == self.cfg.actor.fsdp_config.sharding_strategy, 'Checkpoint sharding mismatch'
        assert state['cpu_offload'] == bool(self.cfg.actor.fsdp_config.cpu_offload), 'Checkpoint offload mismatch'
        plan = None if self.epoch_plan is None else self.epoch_plan.describe()
        assert state.get('epoch_plan') == plan, 'Checkpoint uses a different data/epoch plan; start a fresh run'
        if self.epoch_plan is not None:
            expected_steps = (state['data_epoch'] * self.epoch_plan.updates_per_epoch
                              + math.ceil(state['data_offset'] / self.epoch_plan.accumulation))
            assert state['optimizer_steps'] == expected_steps, 'Checkpoint input/update counters disagree'
        super().load_checkpoint(load_path)
        self.data_loader.load_state_dict(state['loader'])
        self._data_epoch = state['data_epoch']
        self._data_iter_offset = state['data_offset']
        self.optimizer_steps = state['optimizer_steps']
        self.grad_scaler.load_state_dict(state['scaler'])
        self.data_iter = iter(self.data_loader)
        set_rng_state(state['rng'])
        torch.distributed.barrier()

    def _run_complete_epoch_update(self):
        plan = self.epoch_plan
        if self._data_iter_offset == len(self.data_loader):
            # Exhaust the suspended loader before changing its sampler epoch.
            try:
                next(self.data_iter)
            except StopIteration:
                pass
            else:
                raise AssertionError('DataLoader contains more batches than the epoch plan')
            self._data_epoch += 1
            self._data_iter_offset = 0
            self.data_loader.set_epoch(self._data_epoch)
            self.data_iter = iter(self.data_loader)
        assert self._data_epoch < self.cfg.runner.max_epochs, 'All configured epochs are already complete'
        epoch = self._data_epoch
        offset = self._data_iter_offset
        original_accumulation = self.gradient_accumulation
        self.gradient_accumulation = plan.update_batches(offset)
        self._update_global_samples = plan.global_samples(offset, self.gradient_accumulation)
        try:
            result = super().run_training()
        finally:
            self.gradient_accumulation = original_accumulation
            self._update_global_samples = None
        assert self._data_epoch == epoch, 'An optimizer update crossed an epoch boundary'
        consumed = sum(plan.local_samples(rank, 0, self._data_iter_offset)
                       for rank in range(self._world_size))
        result.update(samples_seen=epoch * plan.samples + consumed,
                      completed_epochs=epoch + int(self._data_iter_offset == len(self.data_loader)))
        return result

    def run_training(self):
        from gpu_budget import check_headroom
        check_headroom(float(self.cfg.actor.gpu_headroom_gib), distributed=True)
        self.data_loader.set_epoch(self._data_epoch)
        result = (self._run_complete_epoch_update() if self.epoch_plan is not None
                  else super().run_training())
        if self.optimizer_steps == 1 or self.optimizer_steps % 500 == 0:
            self.audit_state()
        write_json(Path(self.cfg.runner.logger.log_path) / f'progress_rank_{self._rank}.json',
                   dict(step=self.optimizer_steps, metrics={k: float(v) for k, v in result.items()},
                        data_epoch=self._data_epoch, data_offset=self._data_iter_offset,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        unix_time=time.time()))
        return result


def prune_checkpoints(root, keep=3):
    """Only remove completed checkpoints in this run, after a new save succeeds."""
    root = Path(root).resolve()
    completed = sorted((p for p in root.iterdir()
                        if re.fullmatch(r'global_step_\d+', p.name)
                        and p.is_dir() and not p.is_symlink()
                        and (p / 'COMPLETE.json').is_file()),
                       key=lambda p: int(p.name.rsplit('_', 1)[1]))
    removed = []
    for path in completed[:-keep]:
        assert path.resolve().parent == root
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


class RetainingRunner(SFTRunner):
    def _save_checkpoint(self, is_best=False):
        assert not is_best, 'Evaluation and best-model selection are disabled.'
        root = Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name / 'checkpoints'
        root.mkdir(parents=True, exist_ok=True)
        final = root / f'global_step_{self.global_step}'
        staging = root / f'.incomplete_global_step_{self.global_step}'
        if staging.exists() or final.exists():
            raise FileExistsError(f'Refusing to overwrite a checkpoint: {final}')
        actor_path = staging / 'actor'
        actor_path.mkdir(parents=True)
        self.actor.save_checkpoint(str(actor_path), self.global_step).wait()
        # The future resume path depends on this exact topology and input state.
        assert (actor_path / 'dcp_checkpoint/.metadata').is_file()
        assert (actor_path / 'model_state_dict/full_weights.pt').is_file()
        assert all((actor_path / f'input_rank_{rank}.pt').is_file() for rank in range(int(self.cfg.validation_world_size)))
        write_json(staging / 'COMPLETE.json', dict(step=self.global_step, unix_time=time.time(), world_size=int(self.cfg.validation_world_size)))
        staging.rename(final)
        write_json(root / 'latest.json', dict(step=self.global_step, path=str(final)))
        removed = prune_checkpoints(root, self.cfg.runner.keep_last_checkpoints)
        print('CHECKPOINT_COMMITTED', str(final), 'removed=', removed, flush=True)
