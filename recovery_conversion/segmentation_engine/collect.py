"""Select original CAP-X failures that remain failures after full OSC conversion.

Reuse the validated goal_encoding control law. Original successes, technical
exceptions, incomplete recordings and conversions stopped at blocked targets
are not eligible. Actual saved 7D commands must independently replay to failure.
"""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
sys.path.insert(0, str(ROOT/'training_configs/pi05_matched17_compensation_fill_20261028'))
import fill_recoveries as helpers

WORK = ROOT/'trajectory_conversion_tests/pi05_recovery_failure20_20261029'
SOURCE = WORK/'indexed_sources'
ATTEMPTS = WORK/'attempts'
INDEPENDENT = WORK/'independent_replays'
LABELS = WORK/'fixed_goals'
TARGET = 20
SETTINGS = helpers.SETTINGS
BASE_RUN = ROOT/'wcm_pi05_monitor/outputs/hybrid_independent_gpu1_20261019_172047'
save = helpers.save
read = helpers.read


def failure_source_cache(ep, cache):
    """Same lossless source read as the success helper, explicitly check failure."""
    path = SOURCE/'sources/capx_recovery17'/ep['trajectory_path']
    meta = read(path/'manifest.json')
    assert meta['complete'] and not meta['success'] and not ep['success'] and ep['used_capx']
    assert meta['steps'] == ep['steps']
    files = ['manifest.json', 'initial.npz', 'model.xml'] + [s['file'] for s in meta['shards']]
    digests = {name:helpers.engine.sha(path/name) for name in files}
    for shard in meta['shards']:
        assert digests[shard['file']] == shard['sha256']
    cache.mkdir(parents=True, exist_ok=True)
    name = f'{ep["suite"]}_task{ep["task_id"]:02d}_trial{ep["trial_id"]:02d}'
    out = cache/(name+'.npz')
    identity = cache/(name+'.source.json')
    if out.exists():
        assert read(identity) == digests
        with np.load(out, allow_pickle=False) as f:
            return {k:f[k].copy() for k in f.files}
    keys = ['sim/state', 'raw/robot0_joint_pos', 'raw/robot0_joint_vel',
            'raw/robot0_eef_pos', 'raw/robot0_eef_quat', 'raw/robot0_gripper_qpos']
    parts = {k:[] for k in keys}
    with np.load(path/'initial.npz', allow_pickle=False) as f:
        for key in keys:
            parts[key].append(f[key][None].copy())
    actions, owners = [], []
    for shard in meta['shards']:
        with np.load(path/shard['file'], allow_pickle=False) as f:
            for key in keys:
                parts[key].append(f[key].copy())
            actions.append(f['transition/action'].copy())
            owners.append(f['transition/owner'].copy())
            assert not f['transition/success'].any(), 'Original contains a task success'
    data = {k:np.concatenate(v) for k,v in parts.items()}
    data.update(actions=np.concatenate(actions), owners=np.concatenate(owners))
    assert len(data['actions']) == ep['steps']
    assert len(data['sim/state']) == ep['steps']+1
    np.savez_compressed(out, **data)
    save(identity, digests)
    return data


def configure():
    helpers.SOURCE = SOURCE
    helpers.LABELS = LABELS
    helpers.ATTEMPTS = ATTEMPTS
    helpers.engine.SOURCE = SOURCE
    helpers.engine.OFFLINE = LABELS
    helpers.replay.SOURCE = SOURCE
    sys.path.insert(0, str(helpers.engine.HELPERS))
    # The old cache reader intentionally accepts only successful originals.
    # Replace only this input reader; do not change success labels/control law.
    importlib.import_module('probe_conversion').source_cache = failure_source_cache


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK/'plan.json').exists():
        plan = read(WORK/'plan.json')
        assert plan['target'] == TARGET and plan['settings'] == SETTINGS
        return plan
    records = helpers.engine.rows(helpers.ORIGINAL_SOURCE/'index.jsonl')
    taskmap = {(r['suite'],r['evaluation_task_id']):r for r in records[:40]}
    template = next(r for r in records if r['source']=='capx_recovery_success')
    native = SOURCE/'sources/capx_recovery17'
    native.mkdir(parents=True, exist_ok=True)
    reader = SOURCE/'read_trajectory.py'
    if not reader.exists():
        reader.symlink_to(helpers.ORIGINAL_SOURCE/'read_trajectory.py')
    candidates, excluded, seen = [], [], set()
    for p in sorted((BASE_RUN/'eval/results').glob('*/*/*/report.json')):
        report = read(p)
        if not report.get('used_capx') or report.get('success'):
            continue
        manifest = read(p.parent/'trajectory/manifest.json')
        if not manifest['complete'] or report.get('synthetic_alarm_test'):
            excluded.append(dict(path=str(p.parent), reason='incomplete_or_test'))
            continue
        assert not manifest['success'] and manifest['steps'] == report['steps']
        assert report['outcome'] in {'time_limit','capx_stopped'}
        assert report['steps_by_owner']['capx'] > 0
        digest = helpers.engine.sha(p.parent/'actions.jsonl')
        assert digest not in seen
        seen.add(digest)
        suite, tid, trial = report['suite'], report['task_id'], report['trial_id']
        task = taskmap[(suite,tid)]
        assert task['instruction'] == report['instruction']
        eid = 1000+len(candidates)
        rel = f'episodes/{suite}/task_{tid:02d}/trial_{trial:02d}'
        link = native/rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(p.parent, target_is_directory=True)
        handoff = report['handoff']['step']
        ep = dict(suite=suite, task_id=tid, trial_id=trial, instruction=report['instruction'],
            seed=report['seed'], sample_type='capx_recovery_failure', path=rel,
            trajectory_path=rel+'/trajectory', source_episode=str(p.parent),
            success=False, used_capx=True, steps=report['steps'], observations=report['steps']+1,
            steps_by_owner=report['steps_by_owner'], segments=report['segments'],
            handoff_step=handoff, validated=True, original_outcome=report['outcome'])
        rec = copy.deepcopy(template)
        rec.update(source='capx_recovery_failure', episode_index=eid, suite=suite,
            evaluation_task_id=tid, canonical_task_index=task['canonical_task_index'],
            instruction=report['instruction'], trial_id=trial, seed=report['seed'],
            path='sources/capx_recovery17/'+rel,
            trajectory_path='sources/capx_recovery17/'+rel+'/trajectory',
            action_rows=report['steps'], observations=report['steps']+1,
            steps_by_owner=report['steps_by_owner'], segments=report['segments'],
            handoff_step=handoff, recovery_action_range_inclusive=[handoff+1,report['steps']],
            source_dataset=str(native), source_episode=str(p.parent), source_record=ep,
            original_action_log_sha256=digest)
        candidates.append(rec)
    groups = defaultdict(list)
    for rec in candidates:
        groups[(rec['suite'],rec['evaluation_task_id'])].append(rec)
    for group in groups.values():
        group.sort(key=lambda r:(r['action_rows'],r['trial_id']))
    ordered = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop(0))
    (SOURCE/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    plan = dict(target=TARGET, settings=SETTINGS, candidates=ordered, excluded=excluded,
        candidate_order='round-robin by task; shorter first within each task',
        eligible_source_count=len(candidates), original_success_required=False,
        converted_success_required=False, independent_replay_success_required=False,
        require_full_source_sequence=True, exclude_blocked_or_technical_conversions=True,
        source_unchanged=True, training_started=False)
    save(WORK/'plan.json', plan)
    return plan


def convert_one(rec):
    configure()
    eid = rec['episode_index']
    helpers.fixed_labels(rec)
    report = helpers.engine.run_episode(rec, eid, ATTEMPTS, SETTINGS)
    item = dict(record=rec, report=str(ATTEMPTS/f'episode_{eid:06d}/report.json'),
        replay=str(ATTEMPTS), original_success=False, converted_success=report['success'],
        converted_steps=report['executed_steps'], stop_reason=report['stop_reason'])
    # A blocked converter is not a fully converted failed demonstration.
    full = report['last_source_action_step'] == report['source_steps']
    if report['success'] or not full or report['stop_reason'] != 'source_sequence_exhausted':
        return item, None
    checked_path = INDEPENDENT/f'episode_{eid:06d}/report.json'
    if not checked_path.exists():
        helpers.replay.replay_episode(report, ATTEMPTS, INDEPENDENT)
    checked = read(checked_path)
    assert not checked['success'] and checked['state']=='complete'
    assert checked['max_sim_state_error'] < 1e-8
    assert checked['source_report_sha256'] == helpers.engine.sha(Path(item['report']))
    assert checked['action_record_sha256'] == helpers.engine.sha(Path(item['report']).parent/'states_actions.npz')
    item.update(independent_report=str(checked_path), independent_success=False,
        added_compensation_steps=report['added_compensation_steps'])
    return item, item


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    plan = prepare()
    print('FAILURE20_PREPARED', json.dumps(dict(candidates=len(plan['candidates']),
        excluded=len(plan['excluded']), target=TARGET)), flush=True)
    if args.prepare_only:
        return
    if (WORK/'summary.json').exists() and read(WORK/'summary.json')['state']=='complete':
        print('ALREADY_COMPLETE', flush=True)
        return
    ATTEMPTS.mkdir(exist_ok=True)
    INDEPENDENT.mkdir(exist_ok=True)
    accepted, results, errors = [], [], []
    queue = deque(plan['candidates'])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {}
        while pending or (queue and len(accepted)<TARGET):
            while queue and len(pending)<args.workers and len(accepted)+len(pending)<TARGET:
                rec = queue.popleft()
                pending[pool.submit(convert_one,rec)] = rec
            if not pending:
                break
            future = next(as_completed(pending))
            rec = pending.pop(future)
            try:
                item, chosen = future.result()
                results.append(item)
                if chosen:
                    accepted.append(chosen)
            except Exception as exc:
                errors.append(dict(episode_index=rec['episode_index'], error=repr(exc), traceback=traceback.format_exc()))
                print('FAILURE20_TECHNICAL_ERROR', json.dumps(errors[-1]), flush=True)
            status = dict(state='collecting', accepted=len(accepted), target=TARGET,
                attempted=len(results), technical_errors=len(errors), pending=len(pending),
                remaining_candidates=len(queue), training_started=False)
            save(WORK/'progress.json', dict(**status, accepted_items=accepted, results=results, errors=errors))
            save(WORK/'status.json', status)
            print('FAILURE20_STATUS', json.dumps(status), flush=True)
    accepted.sort(key=lambda x:(x['record']['suite'],x['record']['evaluation_task_id'],x['record']['trial_id']))
    summary = dict(state='complete' if len(accepted)==TARGET else 'source_pool_exhausted',
        target=TARGET, accepted_count=len(accepted), accepted=accepted, results=results,
        technical_errors=errors, eligible_sources=len(plan['candidates']),
        per_suite=dict(Counter(x['record']['suite'] for x in accepted)),
        unique_tasks=len({(x['record']['suite'],x['record']['evaluation_task_id']) for x in accepted}),
        original_source_unchanged=True, training_started=False)
    save(WORK/'summary.json', summary)
    save(WORK/'status.json', {k:v for k,v in summary.items() if k not in {'accepted','results','technical_errors'}})
    if len(accepted)!=TARGET:
        raise RuntimeError(f'Only {len(accepted)} valid failure conversions; do not publish as 20')


if __name__ == '__main__':
    main()
