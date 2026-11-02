"""Convert the fixed scheme1 success20 only; never search replacement episodes.

The historical control law is unmodified. This runner adds an explicit selection,
verified-result reuse, program-boundary audit and independent saved-action replay.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
ENGINE = HERE / 'historical_engine'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configure(root, work):
    # Preload local copies before historical modules add their old import paths.
    sys.path.insert(0, str(ENGINE))
    import probe_conversion as probe
    import convert_recovery as engine
    import replay_compensated17 as replay
    import fill_recoveries as helper
    engine.ROOT = replay.ROOT = probe.ROOT = root
    engine.HELPERS = replay.HELPERS = ENGINE
    helper.SOURCE = engine.SOURCE = replay.SOURCE = work / 'indexed_sources'
    helper.LABELS = engine.OFFLINE = work / 'fixed_goals'
    helper.ATTEMPTS = work / 'attempts'
    helper.WORK = work
    probe.SOURCE = helper.SOURCE / 'sources/capx_recovery17'
    return helper


def identity(ep):
    return ep['suite'], ep['task_id'], ep['trial_id'], ep.get('seed', 0)


def historical_inventory(root, selected, settings):
    """Only reuse byte-matched actions with hash-bound, exact independent replay."""
    selected_by_key = {identity(read(Path(x['snapshot']) / 'report.json')): x for x in selected}
    found, invalid = {}, []
    jobs = [
        (root / 'trajectory_conversion_tests/pi05_matched17_osc_split_20261028_goal_encoding',
         'episode_*/report.json', 'independent_full17_replay_20261028'),
        (root / 'trajectory_conversion_tests/pi05_matched17_compensation_fill_20261028',
         'alternate_attempts/episode_*/report.json', 'independent_alternates'),
        (root / 'trajectory_conversion_tests/pi05_long_recovery20_20261028',
         'attempts/episode_*/report.json', 'independent_replays'),
    ]
    for base, pattern, checked_dir in jobs:
        for report_file in sorted(base.glob(pattern)):
            try:
                r = read(report_file)
                ep = r['source_record']['source_record']
                selected_ep = selected_by_key.get(identity(ep))
                if selected_ep is None or r['settings'] != settings:
                    continue
                # Compare original input manifest/model/initial and all action bytes.
                snapshot = Path(selected_ep['snapshot'])
                old_source = Path(ep['source_episode'])
                for rel in ['actions.jsonl', 'trajectory/manifest.json',
                            'trajectory/model.xml', 'trajectory/initial.npz']:
                    assert sha(snapshot / rel) == sha(old_source / rel), rel
                checked_file = base / checked_dir / report_file.parent.name / 'report.json'
                c = read(checked_file)
                assert c['success'] == r['success'] and c['max_sim_state_error'] < 1e-8
                assert c['source_report_sha256'] == sha(report_file)
                assert c['action_record_sha256'] == sha(report_file.parent / 'states_actions.npz')
                assert r['continuous_execution'] and r['intermediate_resets'] == 0
                assert read(report_file.parent / 'trajectory/manifest.json')['complete']
                found.setdefault(selected_ep['dataset_episode_index'], dict(
                    report=str(report_file), independent_report=str(checked_file),
                    report_sha256=sha(report_file), success=r['success']))
            except Exception as exc:
                invalid.append(dict(report=str(report_file), reason=repr(exc)))
    return found, invalid


def prepare(root, raw, work, reuse=True):
    helper = configure(root, work)
    index = raw / 'recovery_success20_raw/episodes.jsonl'
    failure = raw / 'recovery_failure20_raw/selection.json'
    selected = rows(index)
    assert len(selected) == 20
    assert len({x['source_episode'] for x in selected}) == 20
    assert dict(Counter(x['suite'] for x in selected)) == dict(
        libero_10=3, libero_goal=5, libero_spatial=10, libero_object=2)
    source = work / 'indexed_sources'
    source.mkdir(parents=True, exist_ok=True)
    reader = source / 'read_trajectory.py'
    if not reader.exists():
        reader.symlink_to(ENGINE / 'read_trajectory.py')
    tasks = {(x['suite'], x['evaluation_task_id']): x for x in read(HERE / 'tasks.json')}
    assert len(tasks) == 40
    records = []
    for item in selected:
        assert item['original_success'] and item['original_trajectory_complete']
        snapshot = (raw / item['snapshot_relative_path']).resolve()
        assert raw.resolve() in snapshot.parents
        report = read(snapshot / 'report.json')
        assert report['success'] and report['used_capx'] and not report.get('synthetic_alarm_test')
        assert sha(snapshot / 'report.json') == item['source_report_sha256']
        assert report['steps'] == item['original_steps']
        task = tasks[(item['suite'], item['evaluation_task_id'])]
        assert task['instruction'] == report['instruction'] == item['instruction']
        manifest = read(snapshot / 'trajectory/manifest.json')
        assert manifest['complete'] and manifest['success'] and manifest['steps'] == report['steps']
        eid = item['dataset_episode_index']
        rel = 'episodes/episode_%06d' % eid
        link = source / 'sources/capx_recovery17' / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(snapshot, target_is_directory=True)
        assert link.resolve() == snapshot
        handoff = report['handoff']['step']
        ep = dict(suite=report['suite'], task_id=report['task_id'], trial_id=report['trial_id'],
            instruction=report['instruction'], seed=report['seed'], sample_type='capx_recovery_success',
            path=rel, trajectory_path=rel + '/trajectory', source_episode=str(snapshot),
            original_source_episode=item['source_episode'], success=True, used_capx=True,
            steps=report['steps'], observations=report['steps'] + 1, steps_by_owner=report['steps_by_owner'],
            segments=report['segments'], handoff_step=handoff, validated=True)
        rec = dict(episode_index=eid, source='capx_recovery_success', suite=ep['suite'],
            evaluation_task_id=ep['task_id'], canonical_task_index=task['canonical_task_index'],
            instruction=ep['instruction'], trial_id=ep['trial_id'], seed=ep['seed'],
            path='sources/capx_recovery17/' + rel, trajectory_path='sources/capx_recovery17/' + rel + '/trajectory',
            action_rows=ep['steps'], observations=ep['observations'], steps_by_owner=ep['steps_by_owner'],
            segments=ep['segments'], handoff_step=handoff, recovery_action_range_inclusive=[handoff + 1, ep['steps']],
            source_dataset=str(source / 'sources/capx_recovery17'), source_episode=str(snapshot), source_record=ep,
            original_action_log_sha256=sha(snapshot / 'actions.jsonl'))
        ranges = helper.program_ranges(ep)
        # Contiguous original actions, one program per round; never merge retries.
        cursor = handoff + 1
        for p in ranges:
            assert p['start'] == cursor and p['end'] >= p['start'] - 1
            cursor = p['end'] + 1
        assert cursor == ep['steps'] + 1
        records.append(rec)
        item['snapshot'] = str(snapshot)
    assert sorted(r['episode_index'] for r in records) == list(range(20))
    found, invalid = historical_inventory(root, selected, helper.SETTINGS) if reuse else ({}, [])
    plan = dict(version=1, raw=str(raw), root=str(root), settings=helper.SETTINGS,
        selection_sha256=sha(index), failure_selection_sha256=sha(failure),
        engine_sha256={p.name: sha(p) for p in sorted(ENGINE.glob('*.py'))},
        records=records, reuse={str(k): v for k, v in found.items()},
        unverified_historical_results=invalid, fixed_selection=True, replacements_allowed=False,
        original_minimum_step_filter=False, training_started=False)
    plan_file = work / 'plan.json'
    if plan_file.exists():
        assert read(plan_file) == plan, 'Plan changed: use a new output directory'
    else:
        (source / 'index.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
        save(plan_file, plan)
    return plan


def boundary_audit(helper, rec, report_file):
    report_file = Path(report_file)
    r = read(report_file)
    trace = rows(report_file.parent / 'action_provenance.jsonl')
    assert len(trace) == r['executed_steps']
    ranges = helper.program_ranges(rec['source_record'])
    original_success_turn = next(p['turn'] for p in ranges if p['start'] <= rec['action_rows'] <= p['end'])
    groups = []
    for n, p in enumerate(trace):
        matches = [x for x in ranges if x['start'] <= p['source_action_step'] <= x['end']]
        turn = None if p['source_action_step'] <= rec['handoff_step'] else matches[0]['turn']
        assert p['source_turn'] == turn
        if turn is not None:
            assert Path(p['source_program']).name == 'turn_%02d.py' % turn
            expected = helper.SOURCE / rec['path'] / 'capx' / Path(p['source_program']).name
            assert sha(expected) == sha(report_file.parent / p['copied_program'])
        if not groups or groups[-1]['source_turn'] != turn:
            groups.append(dict(source_turn=turn, controller='VLA' if turn is None else 'code',
                converted_start_action=n + 1, converted_end_action=n + 1,
                contains_converted_first_success=False))
        groups[-1]['converted_end_action'] = n + 1
        groups[-1]['contains_converted_first_success'] |= p['success_after_action']
    successes = [p for p in trace if p['success_after_action']]
    assert len(successes) == int(r['success'])
    successful_turn = successes[0]['source_turn'] if successes else None
    return dict(original_success=True, converted_success=r['success'],
        original_success_turn=original_success_turn, converted_first_success_turn=successful_turn,
        success_program_agrees=(successful_turn == original_success_turn) if successes else None,
        program_ranges=[dict(turn=p['turn'], start=p['start'], end=p['end'], program=p['path'].name) for p in ranges],
        converted_program_segments=groups, whole_episode_positive_label_assigned=False)


def retain_episode(report_file, checked_file, work, eid):
    """Give every selected episode its own stable folder, including reused failures.

    Large immutable recordings share storage; small manifests/reports are copied.
    Never edit the shared files in-place.
    """
    original = report_file.parent
    dest = work / 'converted_episodes' / ('episode_%06d' % eid)
    for src in sorted(original.rglob('*')):
        if not src.is_file():
            continue
        dst = dest / src.relative_to(original)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert dst.stat().st_size == src.stat().st_size and sha(dst) == sha(src)
            continue
        if src.stat().st_size >= 1024 * 1024:
            try:
                os.link(src, dst)
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
    shutil.copy2(checked_file, dest / 'independent_replay_report.json')
    assert sha(dest / 'states_actions.npz') == sha(original / 'states_actions.npz')
    return dest


def convert_one(root, work, rec, reused):
    helper = configure(root, work)
    eid = rec['episode_index']
    result_file = work / 'results' / ('episode_%06d.json' % eid)
    if reused:
        report_file = Path(reused['report'])
        checked_file = Path(reused['independent_report'])
        assert sha(report_file) == reused['report_sha256']
    else:
        helper.fixed_labels(rec)
        report_file = helper.ATTEMPTS / ('episode_%06d/report.json' % eid)
        r = helper.engine.run_episode(rec, eid, helper.ATTEMPTS, helper.SETTINGS)
        independent = work / 'independent_replays'
        independent.mkdir(parents=True, exist_ok=True)
        checked_file = independent / ('episode_%06d/report.json' % eid)
        if not checked_file.exists():
            helper.replay.replay_episode(r, helper.ATTEMPTS, independent)
    r, c = read(report_file), read(checked_file)
    assert r['settings'] == helper.SETTINGS and c['success'] == r['success']
    assert c['max_sim_state_error'] < 1e-8
    assert c['source_report_sha256'] == sha(report_file)
    assert c['action_record_sha256'] == sha(report_file.parent / 'states_actions.npz')
    audit = boundary_audit(helper, rec, report_file)
    retained = retain_episode(report_file, checked_file, work, eid)
    result = dict(episode_index=eid, suite=rec['suite'], task_id=rec['evaluation_task_id'], trial_id=rec['trial_id'],
        original_source_episode=rec['source_record']['original_source_episode'], snapshot=rec['source_episode'],
        reused=bool(reused), original_success=True, success=r['success'],
        converted_success=r['success'], conversion_outcome='success' if r['success'] else 'failure',
        retained=True, independently_verified=True,
        original_steps=rec['action_rows'], converted_steps=r['executed_steps'],
        added_compensation_steps=r['added_compensation_steps'], stop_reason=r['stop_reason'],
        report=str(report_file), independent_report=str(checked_file),
        retained_episode=str(retained), trajectory_path=str(retained / 'trajectory'),
        video_path=str(retained / 'execution.mp4'), action_path=str(retained / 'states_actions.npz'),
        report_sha256=sha(report_file), action_record_sha256=c['action_record_sha256'],
        max_replay_state_error=c['max_sim_state_error'], boundary_audit=audit,
        eligible_for_success_export=bool(r['success']), train_ready=False,
        label_scope='conversion outcome only; segment quality not reassigned')
    save(result_file, result)
    save(retained / 'conversion_annotation.json', result)
    print('SELECTED_CONVERSION_DONE ' + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--episodes', nargs='+', type=int)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-reuse', action='store_true')
    args = ap.parse_args()
    root, raw, work = args.root.resolve(), args.raw.resolve(), args.output.resolve()
    assert raw != work and raw not in work.parents and work not in raw.parents
    assert args.workers > 0
    work.mkdir(parents=True, exist_ok=True)
    plan = prepare(root, raw, work, reuse=not args.no_reuse)
    if args.prepare_only:
        print('PREPARED', json.dumps(dict(total=20, reusable=len(plan['reuse']), output=str(work))), flush=True)
        return
    records = plan['records']
    if args.episodes:
        assert set(args.episodes) <= set(range(20))
        records = [r for r in records if r['episode_index'] in args.episodes]
    errors = []
    def update():
        results = [read(p) for p in sorted((work / 'results').glob('episode_*.json'))]
        state = 'complete' if len(results) == 20 and not errors else 'technical_error' if errors else 'running'
        summary = dict(state=state, total=20, completed=len(results),
            success=sum(x['success'] for x in results), failed=sum(not x['success'] for x in results),
            reused=sum(x['reused'] for x in results), technical_errors=errors,
            success_program_mismatches=sum(x['boundary_audit']['success_program_agrees'] is False for x in results),
            original_failure20_unchanged=sha(raw / 'recovery_failure20_raw/selection.json') == plan['failure_selection_sha256'],
            results=results, training_started=False, train_ready=False)
        assert summary['original_failure20_unchanged']
        save(work / 'summary.json', summary)
        save(work / 'accepted.json', dict(accepted=[x for x in results if x['success']], train_ready=False))
        save(work / 'status.json', {k: v for k, v in summary.items() if k not in ['results', 'technical_errors']})
        print('SELECTED_STATUS', json.dumps(read(work / 'status.json')), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(convert_one, root, work, r, plan['reuse'].get(str(r['episode_index']))): r for r in records}
        update()
        for future in as_completed(futures):
            rec = futures[future]
            try:
                future.result()
            except Exception as exc:
                error = dict(episode_index=rec['episode_index'], error=repr(exc), traceback=traceback.format_exc())
                errors.append(error)
                save(work / 'errors' / ('episode_%06d.json' % rec['episode_index']), error)
                print('SELECTED_TECHNICAL_ERROR', json.dumps(error), flush=True)
            update()
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
