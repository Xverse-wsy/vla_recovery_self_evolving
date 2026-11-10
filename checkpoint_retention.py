"""Checkpoint retention without importing the GPU training runtime."""
from pathlib import Path
import re
import shutil


def smoke_checkpoint_keep(max_updates, save_interval, configured_keep):
    """Keep the early checkpoint used by smoke replay until that replay finishes."""
    if min(max_updates, save_interval, configured_keep) < 1:
        raise ValueError('Checkpoint counts and interval must be positive')
    # The runner also saves the final update when it is off the periodic grid.
    saves = (max_updates + save_interval - 1) // save_interval
    return max(configured_keep, saves)


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
