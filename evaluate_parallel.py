"""One evaluation worker per allocated GPU, with disjoint global episode shards."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from eval_jobs import SUITES, jobs, lock, merge, partition, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', default=os.environ.get('PI05_EVAL_GPUS'))
    ap.add_argument('--label', choices=['fewshot', 'a60', 'b80'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--suites', nargs='+', default=SUITES, choices=SUITES)
    ap.add_argument('--tasks', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--flow-steps', type=int, default=3)
    ap.add_argument('--action-chunk', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--num-shards', type=int, help='Across all nodes; default: number of local GPUs')
    ap.add_argument('--shard-offset', type=int, default=0, help='First global shard on this node')
    ap.add_argument('--merge-only', action='store_true')
    args = ap.parse_args()
    output = args.output.expanduser().resolve()
    if args.merge_only:
        print(json.dumps(merge(output, require_complete=True), indent=2))
        return
    if not args.gpus:
        ap.error('Set PI05_EVAL_GPUS or pass --gpus')
    gpus = [int(s) for s in args.gpus.split(',')]
    assert gpus and len(gpus) == len(set(gpus)) and min(gpus) >= 0
    world = len(gpus) if args.num_shards is None else args.num_shards
    assert world > 0 and 0 <= args.shard_offset and args.shard_offset + len(gpus) <= world
    assert args.flow_steps > 0 and 1 <= args.action_chunk <= 10 and args.max_steps >= 0
    all_jobs = jobs(args.suites, args.tasks, args.trials)
    ranks = list(range(args.shard_offset, args.shard_offset + len(gpus)))
    manifest = dict(label=args.label, checkpoint=str(args.checkpoint.expanduser().resolve(strict=True)),
                    suites=args.suites, tasks=args.tasks, trials=args.trials, flow_steps=args.flow_steps,
                    action_chunk=args.action_chunk, seed=args.seed, max_steps=args.max_steps,
                    save_video=not args.no_video, num_shards=world)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Per-worker locks also reject overlapping allocations on two nodes.
        for rank in ranks:
            stack.enter_context(lock(output / 'workers' / f'worker_{rank:03d}' / '.worker.lock'))
        with lock(output / '.manifest.lock', blocking=True):
            path = output / 'parallel.json'
            if path.exists():
                assert json.loads(path.read_text()) == manifest, 'Parallel evaluation settings changed'
                for rank in ranks:
                    worker = output / 'workers' / f'worker_{rank:03d}'
                    assert args.resume or not (worker / 'run.json').exists(), 'Use --resume to continue existing workers'
            else:
                assert not args.resume and not (output / 'run.json').exists(), 'Use a fresh parallel output directory'
                save(path, manifest)
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        free = {int(v[0]): int(v[1]) for row in rows.splitlines() if (v := row.split(','))}
        for rank, gpu in zip(ranks, gpus):
            assert gpu in free, f'Unknown GPU {gpu}'
            worker_status = output / 'workers' / f'worker_{rank:03d}' / 'status.json'
            completed = worker_status.exists() and json.loads(worker_status.read_text())['state'] == 'complete'
            if partition(all_jobs, rank, world) and not completed:
                assert free[gpu] >= 29 * 1024, f'GPU {gpu}: need 29 GiB free, found {free[gpu]/1024:.2f}'
        active = []
        package = Path(__file__).resolve().parent
        local_status = output / 'launchers' / f'offset_{args.shard_offset:03d}.json'

        def stop():
            for proc in active:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 10
            for proc in active:
                remaining = max(0, deadline - time.monotonic())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')

        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            save(local_status, dict(state='running', gpus=gpus, shards=ranks, pid=os.getpid()))
            for rank, gpu in zip(ranks, gpus):
                worker = output / 'workers' / f'worker_{rank:03d}'
                log = stack.enter_context((worker / 'worker.log').open('a'))
                command = ['bash', str(package / 'run_eval.sh'), '--label', args.label,
                           '--checkpoint', manifest['checkpoint'], '--output', str(worker),
                           '--suites', *args.suites, '--tasks', *map(str, args.tasks),
                           '--trials', str(args.trials), '--flow-steps', str(args.flow_steps),
                           '--action-chunk', str(args.action_chunk), '--seed', str(args.seed),
                           '--max-steps', str(args.max_steps), '--num-shards', str(world),
                           '--shard-index', str(rank)]
                if args.no_video:
                    command.append('--no-video')
                if args.resume and (worker / 'run.json').exists():
                    command.append('--resume')
                env = os.environ | {'PI05_EVAL_GPU': str(gpu), 'PI05_PARALLEL_WORKER': '1',
                                    'MUJOCO_EGL_DEVICE_ID': str(gpu),
                                    'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append(proc)
                print(f'WORKER shard={rank}/{world} gpu={gpu} episodes={len(partition(all_jobs,rank,world))} log={worker/"worker.log"}', flush=True)
            while any(proc.poll() is None for proc in active):
                failed = [proc.returncode for proc in active if proc.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Evaluation worker failed: {failed}; see worker.log')
                time.sleep(0.5)
            assert all(proc.returncode == 0 for proc in active), 'Evaluation worker failed; see worker.log'
            save(local_status, dict(state='complete', gpus=gpus, shards=ranks))
            # Another node may still be working; publish only after every shard.
            result = merge(output)
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as exc:
            stop()
            save(local_status, dict(state='failed', error=repr(exc), gpus=gpus, shards=ranks))
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


if __name__ == '__main__':
    main()
