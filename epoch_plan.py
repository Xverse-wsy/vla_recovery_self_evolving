"""Exact, unpadded data partitions and epoch-aligned optimizer updates."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochPlan:
    samples: int
    world_size: int
    micro_batch_size: int
    accumulation: int

    def __post_init__(self):
        if min(self.samples, self.world_size, self.micro_batch_size, self.accumulation) < 1:
            raise ValueError('Epoch plan values must be positive')

    def rank_samples(self, rank):
        return max(0, (self.samples + self.world_size - 1 - rank) // self.world_size)

    def rank_batches(self, rank):
        # Ranks with no real sample in the final call use one zero-weight
        # synchronization placeholder. Real examples are never duplicated.
        largest = (self.samples + self.world_size - 1) // self.world_size
        return (largest + self.micro_batch_size - 1) // self.micro_batch_size

    def dummy_samples(self, rank):
        minimum = (self.batches_per_epoch - 1) * self.micro_batch_size + 1
        return max(0, minimum - self.rank_samples(rank))

    @property
    def batches_per_epoch(self):
        return self.rank_batches(0)

    @property
    def updates_per_epoch(self):
        return (self.batches_per_epoch + self.accumulation - 1) // self.accumulation

    def update_batches(self, offset):
        if not 0 <= offset < self.batches_per_epoch or offset % self.accumulation:
            raise ValueError(f'Invalid optimizer-update batch offset: {offset}')
        return min(self.accumulation, self.batches_per_epoch - offset)

    def local_samples(self, rank, offset, batches):
        start = offset * self.micro_batch_size
        end = min(self.rank_samples(rank), (offset + batches) * self.micro_batch_size)
        return max(0, end - start)

    def global_samples(self, offset, batches):
        return sum(self.local_samples(rank, offset, batches) for rank in range(self.world_size))

    def describe(self):
        tail_offset = (self.updates_per_epoch - 1) * self.accumulation
        tail_batches = self.update_batches(tail_offset)
        return dict(samples_per_epoch=self.samples, world_size=self.world_size,
                    micro_batch_size=self.micro_batch_size, accumulation=self.accumulation,
                    batches_per_rank=self.batches_per_epoch,
                    updates_per_epoch=self.updates_per_epoch,
                    last_update_samples=self.global_samples(tail_offset, tail_batches),
                    drop_last=False, real_sample_duplicates=False,
                    zero_weight_sync_padding=any(self.dummy_samples(r) for r in range(self.world_size)),
                    updates_cross_epochs=False)


class ExactEpochSampler:
    """Each index appears on exactly one rank, exactly once in an epoch."""
    def __init__(self, samples, world_size, rank, seed=0, shuffle=True, micro_batch_size=None):
        self.samples = samples
        self.num_replicas = world_size
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.micro_batch_size = micro_batch_size

    def _dummy_count(self):
        if self.micro_batch_size is None:
            return 0
        return EpochPlan(self.samples, self.num_replicas, self.micro_batch_size, 1).dummy_samples(self.rank)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return max(0, (self.samples + self.num_replicas - 1 - self.rank) // self.num_replicas) + self._dummy_count()

    def __iter__(self):
        if self.shuffle:
            import torch
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.samples, generator=generator).tolist()
        else:
            indices = range(self.samples)
        return iter(list(indices[self.rank::self.num_replicas]) + [-1] * self._dummy_count())
