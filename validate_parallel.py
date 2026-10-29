"""CPU checks for coverage, incomplete merges, mismatched weights and locks."""
import argparse
import json
from pathlib import Path
import tempfile

from eval_jobs import SUITES, jobs, lock, merge, partition, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    all_jobs = jobs(SUITES, list(range(10)), 50)
    for world in [1, 2, 3, 16, 64, 2048]:
        shards = [partition(all_jobs, rank, world) for rank in range(world)]
        flat = [item for shard in shards for item in shard]
        assert len(flat) == len(set(flat)) == 2000
        assert set(flat) == set(all_jobs)
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = dict(label='fewshot', checkpoint='/test/model', suites=SUITES,
                        tasks=[0], trials=1, flow_steps=3, action_chunk=5,
                        seed=0, max_steps=0, save_video=True, num_shards=2)
        base = {k: v for k, v in manifest.items() if k != 'num_shards'}
        base.update(checkpoint_sha256='identical-weights', norm_sha256='identical-norm')
        save(root / 'parallel.json', manifest)
        subset = jobs(SUITES, [0], 1)
        for rank in range(2):
            worker = root / 'workers' / f'worker_{rank:03d}'
            save(worker / 'run.json', base | {'shard': dict(index=rank, count=2)})
            save(worker / 'status.json', dict(state='complete'))
            for suite, task, trial in partition(subset, rank, 2):
                path = worker / 'episodes' / suite / f'task_{task:02d}' / f'trial_{trial:02d}.json'
                record = dict(state='complete', suite=suite, task_id=task, trial_id=trial,
                              success=rank == 0, steps=5, step_limit=240,
                              initial_state_sha256=suite, rollout_training_init_overlap=False)
                save(path, record)
                path.with_suffix('.mp4').write_bytes(b'nonempty-test-video')
            if rank == 0:
                assert merge(root)['state'] == 'waiting_for_workers'
                assert not (root / 'run.json').exists()
        result = merge(root, require_complete=True)
        assert result['summary']['episodes'] == 4 and result['summary']['successes'] == 2
        assert len(list((root / 'episodes').glob('*/*/*.mp4'))) == 4
        assert merge(root, require_complete=True) == result
        worker_run = root / 'workers/worker_001/run.json'
        original = json.loads(worker_run.read_text())
        save(worker_run, original | {'checkpoint_sha256': 'wrong-weights'})
        try:
            merge(root, require_complete=True)
            raise AssertionError('Mismatched weights were accepted')
        except AssertionError as exc:
            assert 'different weights' in str(exc)
        save(worker_run, original)
        with lock(root / '.test.lock'):
            try:
                with lock(root / '.test.lock'):
                    raise AssertionError('Concurrent lock was accepted')
            except RuntimeError as exc:
                assert 'Another launcher owns' in str(exc)
        save(root / 'workers/worker_001/status.json', dict(state='failed'))
        try:
            merge(root, require_complete=True)
            raise AssertionError('Failed worker was accepted')
        except RuntimeError as exc:
            assert 'incomplete' in str(exc)
    report = dict(state='complete', global_episodes=2000, tested_world_sizes=[1, 2, 3, 16, 64, 2048],
                  no_duplicates_or_omissions=True, incomplete_and_mismatched_runs_rejected=True,
                  merge_idempotent=True, overlapping_launchers_rejected=True, gpu_used=False)
    if args.output:
        save(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
