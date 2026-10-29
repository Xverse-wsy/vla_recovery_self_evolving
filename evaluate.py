"""Pure pi0.5 full LIBERO evaluation; same states and settings for all models."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from eval_jobs import jobs, partition
from settings import load_paths

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(output):
    groups = defaultdict(list)
    for p in (output / 'episodes').glob('*/*/*.json'):
        r = json.loads(p.read_text())
        if r['state'] == 'complete':
            groups[r['suite']].append(r)
    report = dict(suites={})
    for suite, records in groups.items():
        report['suites'][suite] = dict(episodes=len(records), successes=sum(r['success'] for r in records),
                                       success_rate=sum(r['success'] for r in records) / len(records),
                                       rollout_training_init_overlap=sum(r['rollout_training_init_overlap'] for r in records))
    all_records = sum(groups.values(), [])
    report.update(episodes=len(all_records), successes=sum(r['success'] for r in all_records),
                  success_rate=sum(r['success'] for r in all_records) / len(all_records) if all_records else None)
    save(output / 'summary.json', report)
    return report


def training_overlap(root, label):
    paths = load_paths()
    sources = [Path(paths.autonomous20), Path(paths.recovery20)]
    sources = [] if label == 'fewshot' else sources[:1] if label == 'a60' else sources
    result = set()
    for folder in sources:
        for line in (folder / 'meta/provenance.jsonl').read_text().splitlines():
            p = json.loads(line)
            rec = p['source_record']
            tid = rec.get('evaluation_task_id', rec.get('task_id'))
            trial = rec.get('trial_id')
            assert tid is not None and trial is not None, rec
            result.add((p['suite'], int(tid), int(trial)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--label', required=True, choices=['fewshot', 'a60', 'b80'])
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--suites', nargs='+', default=SUITES, choices=SUITES)
    ap.add_argument('--tasks', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--flow-steps', type=int, default=3)
    ap.add_argument('--action-chunk', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=0, help='0: native suite budget; nonzero: bounded smoke only')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    args = ap.parse_args()
    assert 1 <= args.trials <= 50 and 1 <= args.action_chunk <= 10 and args.flow_steps > 0
    assert args.max_steps >= 0 and args.tasks and len(args.tasks) == len(set(args.tasks))
    assert all(0 <= i < 10 for i in args.tasks)
    assert len(args.suites) == len(set(args.suites))
    assigned = set(partition(jobs(args.suites, args.tasks, args.trials), args.shard_index, args.num_shards))
    root = Path(os.environ['PI05_ROOT']).resolve()
    norm = Path(load_paths().norm_stats)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights, resolve_model_safetensors
    weights = resolve_full_weights(args.checkpoint) or resolve_model_safetensors(args.checkpoint)
    if weights is None:
        raise FileNotFoundError(args.checkpoint)
    run = dict(label=args.label, checkpoint=str(args.checkpoint.resolve()), suites=args.suites,
               tasks=args.tasks, trials=args.trials, flow_steps=args.flow_steps, action_chunk=args.action_chunk,
               seed=args.seed, max_steps=args.max_steps, benchmark='standard_LIBERO',
               checkpoint_sha256=fingerprint(weights),
               wcm_enabled=False, capx_enabled=False, norm_sha256=fingerprint(norm),
               save_video=not args.no_video,
               scope='full2000' if args.suites == SUITES and args.tasks == list(range(10))
                     and args.trials == 50 and args.max_steps == 0 else 'validation_subset')
    # Explicit identity is also present for one-card parallel launcher runs.
    if args.num_shards != 1 or os.environ.get('PI05_PARALLEL_WORKER') == '1':
        run['shard'] = dict(index=args.shard_index, count=args.num_shards)
    if (output / 'run.json').exists():
        assert args.resume and json.loads((output / 'run.json').read_text()) == run, 'Evaluation configuration changed'
    else:
        assert not args.resume, 'Cannot resume an absent run'
        save(output / 'run.json', run)
    save(output / 'status.json', dict(state='initializing', run=run))
    try:
        # Completed/empty workers validate checkpoint identity but need no GPU.
        pending = []
        for suite, task, trial in assigned:
            path = output / 'episodes' / suite / f'task_{task:02d}' / f'trial_{trial:02d}.json'
            if not path.is_file() or json.loads(path.read_text())['state'] != 'complete':
                pending.append((suite, task, trial))
            elif not args.no_video:
                assert path.with_suffix('.mp4').is_file() and path.with_suffix('.mp4').stat().st_size > 0, path
        if not pending:
            summary = summarize(output)
            assert summary['episodes'] == len(assigned)
            save(output / 'status.json', dict(state='complete', run=run, summary=summary))
            return
        from rlinf.envs.sim.libero.libero_env import LiberoEnv
        from rlinf.models.embodiment.openpi import get_model
        from gpu_budget import set_limit, check_headroom
        set_limit(16)
        free, _ = torch.cuda.mem_get_info()
        assert free / 2**30 >= 29, 'Evaluation needs 16 GiB allocator budget plus 12 GiB shared headroom and context margin'
        torch.set_num_threads(4)
        model_cfg = OmegaConf.create(dict(
            model_path=run['checkpoint'], precision=None, pi05=True, action_dim=7,
            num_action_chunks=args.action_chunk, num_steps=args.flow_steps, add_value_head=False,
            openpi_data=dict(norm_stats_path=str(norm), extra_delta_transform=False),
            openpi=dict(task='eval', config_name='pi05_libero', model_action_dim=32,
                        action_env_dim=7, action_horizon=10, action_chunk=args.action_chunk,
                        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                        max_token_len=200, discrete_state_input=False, torch_compile=False)))
        policy = get_model(model_cfg).to('cuda:0').eval().requires_grad_(False)
        assert policy.action_horizon == 10
        overlap = training_overlap(root, args.label)
        for suite in args.suites:
            if not any(item[0] == suite for item in pending):
                continue
            cfg = OmegaConf.load(root / f'RLinf/examples/embodiment/config/env/{suite}.yaml')
            cfg.total_num_envs = 1
            cfg.auto_reset = False
            cfg.ignore_terminations = False
            cfg.is_eval = True
            cfg.specific_reset_id = 0
            cfg.seed = args.seed
            cfg.video_cfg.save_video = False
            if args.max_steps:
                cfg.max_episode_steps = args.max_steps
                cfg.max_steps_per_rollout_epoch = args.max_steps
            env = LiberoEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
            try:
                assert env.task_suite.get_num_tasks() == 10
                assert all(n >= args.trials for n in env.trial_id_bins)
                offsets = [0, *np.cumsum(env.trial_id_bins).tolist()]
                env.reset()  # Consume the wrapper's initial fixed-reset setup before explicit trial IDs.
                for task in args.tasks:
                    for trial in range(args.trials):
                        if (suite, task, trial) not in assigned:
                            continue
                        result_path = output / 'episodes' / suite / f'task_{task:02d}' / f'trial_{trial:02d}.json'
                        if result_path.exists():
                            assert args.resume
                            previous = json.loads(result_path.read_text())
                            if previous['state'] == 'complete':
                                continue
                            archive = output / 'failed_attempts' / suite / f'task_{task:02d}' / f'trial_{trial:02d}_{time.time_ns()}'
                            archive.mkdir(parents=True)
                            result_path.rename(archive / result_path.name)
                            if result_path.with_suffix('.mp4').exists():
                                result_path.with_suffix('.mp4').rename(archive / result_path.with_suffix('.mp4').name)
                        torch.manual_seed(args.seed)
                        np.random.seed(args.seed)
                        random.seed(args.seed)
                        rid = offsets[task] + trial
                        observation, _ = env.reset(reset_state_ids=np.array([rid]))
                        assert int(env.task_ids[0]) == task and int(env.trial_ids[0]) == trial
                        state = np.asarray(env.task_suite.get_task_init_states(task)[trial])
                        report = dict(state='running', suite=suite, task_id=task, trial_id=trial,
                                      reset_id=rid, instruction=env.task_descriptions[0],
                                      initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                                      step_limit=int(cfg.max_episode_steps), rollout_training_init_overlap=(suite,task,trial) in overlap)
                        video = None
                        started = time.monotonic()
                        digest = hashlib.sha256()
                        success, steps, queries = False, 0, 0
                        try:
                            if not args.no_video:
                                import imageio.v2 as imageio
                                result_path.parent.mkdir(parents=True, exist_ok=True)
                                video = imageio.get_writer(str(result_path.with_suffix('.mp4')), fps=20)
                                video.append_data(np.ascontiguousarray(env.current_raw_obs[0]['agentview_image'][::-1,::-1]))
                            done = False
                            while not done and steps < cfg.max_episode_steps:
                                check_headroom(12)
                                with torch.inference_mode():
                                    actions, aux = policy.predict_action_batch(observation, mode='eval')
                                assert actions.shape == (1, args.action_chunk, 7)
                                assert torch.isfinite(actions).all()
                                queries += 1
                                for action in actions[0]:
                                    digest.update(action.detach().cpu().numpy().copy().tobytes())
                                    observation, reward, terminated, truncated, info = env.step(action[None].clone(), auto_reset=False)
                                    steps += 1
                                    success = bool(terminated.any().item())
                                    done = success or bool(truncated.any().item()) or steps >= cfg.max_episode_steps
                                    if video:
                                        video.append_data(np.ascontiguousarray(env.current_raw_obs[0]['agentview_image'][::-1,::-1]))
                                    if done:
                                        break
                            report.update(state='complete', success=success, steps=steps, queries=queries,
                                          elapsed_seconds=time.monotonic()-started, action_sha256=digest.hexdigest())
                        except BaseException as exc:
                            report.update(state='failed', error=repr(exc), steps=steps)
                            raise
                        finally:
                            if video:
                                video.close()
                            save(result_path, report)
                        summary = summarize(output)
                        save(output / 'status.json', dict(state='running', latest=report, summary=summary))
                        print('EPISODE', args.label, suite, task, trial, success, steps, flush=True)
            finally:
                env.close()
        summary = summarize(output)
        expected = len(assigned)
        assert summary['episodes'] == expected
        save(output / 'status.json', dict(state='complete', run=run, summary=summary))
    except BaseException as exc:
        save(output / 'status.json', dict(state='failed', error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
