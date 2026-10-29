"""Deterministic evaluation partitions and strict, resumable result merging."""
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def jobs(suites, tasks, trials):
    assert suites and len(suites) == len(set(suites)) and set(suites) <= set(SUITES)
    assert tasks and len(tasks) == len(set(tasks)) and all(0 <= t < 10 for t in tasks)
    assert 1 <= trials <= 50
    return [(suite, task, trial) for suite in suites for task in tasks for trial in range(trials)]


def partition(all_jobs, rank, world):
    assert world > 0 and 0 <= rank < world
    return all_jobs[rank::world]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f'Another launcher owns {path}') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['suite']].append(record)
    def count(items):
        wins = sum(r['success'] for r in items)
        return dict(episodes=len(items), successes=wins,
                    success_rate=wins / len(items) if items else None)
    result = count(records)
    result['suites'] = {suite: count(items) | {
        'rollout_training_init_overlap': sum(r['rollout_training_init_overlap'] for r in items)
    } for suite, items in grouped.items()}
    return result


def publish(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        assert destination.stat().st_size == source.stat().st_size, f'Conflicting existing result: {destination}'
        with source.open('rb') as left, destination.open('rb') as right:
            while block := left.read(1024 * 1024):
                assert block == right.read(len(block)), f'Conflicting existing result: {destination}'
        return
    try:
        os.link(source, destination)
    except OSError:
        temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def merge(output, require_complete=False):
    """Publish only when every global shard is complete, including other nodes."""
    output = Path(output)
    with lock(output / '.merge.lock', blocking=True):
        manifest = json.loads((output / 'parallel.json').read_text())
        all_jobs = jobs(manifest['suites'], manifest['tasks'], manifest['trials'])
        base, records, publications, pending = None, [], [], []
        for rank in range(manifest['num_shards']):
            worker = output / 'workers' / f'worker_{rank:03d}'
            status_path = worker / 'status.json'
            if not status_path.is_file() or json.loads(status_path.read_text())['state'] != 'complete':
                pending.append(rank)
                continue
            run = json.loads((worker / 'run.json').read_text())
            shard = run.pop('shard')
            assert shard == dict(index=rank, count=manifest['num_shards']), 'Shard identity changed'
            for key in ['label', 'checkpoint', 'suites', 'tasks', 'trials', 'flow_steps',
                        'action_chunk', 'seed', 'max_steps', 'save_video']:
                assert run[key] == manifest[key], f'Worker configuration changed: {key}'
            if base is None:
                base = run
            assert run == base, 'Workers used different weights or inference settings'
            assigned = set(partition(all_jobs, rank, manifest['num_shards']))
            seen = set()
            for path in sorted((worker / 'episodes').glob('*/*/*.json')):
                record = json.loads(path.read_text())
                identity = (record['suite'], record['task_id'], record['trial_id'])
                assert identity in assigned and identity not in seen, f'Duplicate/wrong shard episode: {identity}'
                assert record['state'] == 'complete' and 0 < record['steps'] <= record['step_limit']
                assert path.relative_to(worker / 'episodes') == Path(identity[0]) / f'task_{identity[1]:02d}' / f'trial_{identity[2]:02d}.json'
                seen.add(identity)
                records.append(record)
                publications.append((path, output / 'episodes' / path.relative_to(worker / 'episodes')))
                if run['save_video']:
                    video = path.with_suffix('.mp4')
                    assert video.is_file() and video.stat().st_size > 0, video
                    publications.append((video, output / 'episodes' / video.relative_to(worker / 'episodes')))
            assert seen == assigned, f'Missing shard episodes: {assigned - seen}'
        if pending:
            report = dict(state='waiting_for_workers', pending_workers=pending,
                          completed_episodes=len(records), expected_episodes=len(all_jobs))
            save(output / 'status.json', report)
            if require_complete:
                raise RuntimeError(f'Evaluation incomplete: pending workers {pending}')
            return report
        assert len(records) == len(all_jobs)
        assert len({(r['suite'], r['task_id'], r['trial_id']) for r in records}) == len(all_jobs)
        for source, destination in publications:
            publish(source, destination)
        summary = summarize(records)
        save(output / 'run.json', base)
        save(output / 'summary.json', summary)
        report = dict(state='complete', run=base, summary=summary,
                      parallel=dict(num_shards=manifest['num_shards'], exactly_once=True))
        save(output / 'status.json', report)
        return report
