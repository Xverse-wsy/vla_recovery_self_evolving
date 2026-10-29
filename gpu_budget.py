"""Bound only our PyTorch allocator and stop our work if shared headroom shrinks."""
import torch


def set_limit(limit_gib):
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if not 0 < limit_gib * 2**30 < total:
        raise ValueError('Invalid per-process GPU memory limit')
    torch.cuda.set_per_process_memory_fraction(limit_gib * 2**30 / total)


def check_headroom(minimum_gib=12, distributed=False):
    free, _ = torch.cuda.mem_get_info()
    value = torch.tensor(free / 2**30, device='cuda', dtype=torch.float64)
    if distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
    if value.item() < minimum_gib:
        raise RuntimeError(f'Our job stops to preserve shared GPU space: free={value.item():.2f} GiB, required={minimum_gib} GiB')
