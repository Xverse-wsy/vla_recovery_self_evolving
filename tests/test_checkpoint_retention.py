import pytest

from checkpoint_retention import prune_checkpoints, smoke_checkpoint_keep
from epoch_plan import EpochPlan


@pytest.mark.parametrize('world,micro', [(1, 1), (2, 1), (1, 2)])
def test_smoke_replay_checkpoint_survives_every_scheduled_save(tmp_path, world, micro):
    plan = EpochPlan(15, world, micro, 3)
    final_step = plan.updates_per_epoch * 2
    keep = smoke_checkpoint_keep(final_step, 2, 3)
    for step in range(1, final_step + 1):
        if step % 2 == 0 or step == final_step:
            saved = tmp_path / f'global_step_{step}'
            saved.mkdir()
            (saved / 'COMPLETE.json').write_text('{}')
            prune_checkpoints(tmp_path, keep)
            # train.py must still load this checkpoint after the FINAL save.
            assert (tmp_path / 'global_step_2/COMPLETE.json').is_file()
    assert (tmp_path / f'global_step_{final_step}/COMPLETE.json').is_file()


def test_formal_default_still_keeps_latest_three_completed_checkpoints(tmp_path):
    for step in (2, 4, 6, 8, 10):
        saved = tmp_path / f'global_step_{step}'
        saved.mkdir()
        (saved / 'COMPLETE.json').write_text('{}')
    incomplete = tmp_path / '.incomplete_global_step_12'
    incomplete.mkdir()
    assert prune_checkpoints(tmp_path) == ['global_step_2', 'global_step_4']
    assert {p.name for p in tmp_path.iterdir()} == {
        'global_step_6', 'global_step_8', 'global_step_10', incomplete.name}
