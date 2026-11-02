"""Convert fixed failure20 without outcome filtering; retain startup-only records."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import multiprocessing
from pathlib import Path
import traceback

import convert_selected_success as core


def configure(root, work, source=None):
    helper = core.configure(root, work)
    spec = importlib.util.spec_from_file_location('failure_cache_reader', core.HERE / 'segmentation_engine/collect.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reader.SOURCE = source or work / 'indexed_sources'
    reader.helpers = helper
    import probe_conversion as probe
    probe.source_cache = reader.failure_source_cache
    helper.replay.SOURCE = reader.SOURCE
    return helper


def prepare(root, raw, work):
    helper = configure(root, work)
    selected = core.rows(raw / 'recovery_failure20_raw/episodes.jsonl')
    assert len(selected) == 20
    tasks = {(x['suite'], x['evaluation_task_id']): x for x in core.read(core.HERE / 'tasks.json')}
    source = work / 'indexed_sources'
    source.mkdir(parents=True, exist_ok=True)
    reader = source / 'read_trajectory.py'
    if not reader.exists():
        reader.symlink_to(core.ENGINE / 'read_trajectory.py')
    previous = root / 'trajectory_conversion_tests/pi05_recovery_failure20_20261029'
    known = {}
    for p in sorted((previous / 'attempts').glob('episode_*/report.json')):
        r = core.read(p)
        if r['settings'] == helper.SETTINGS:
            known[core.identity(r['source_record']['source_record'])] = p
    records = []
    for item in selected:
        snapshot = raw / item['snapshot_relative_path']
        original = core.read(snapshot / 'report.json')
        assert not original['success'] and core.sha(snapshot / 'report.json') == item['source_report_sha256']
        eid = item['dataset_episode_index']
        if not item['cap_steps']:
            assert not item['recording_audit']['complete']
            core.save(work / 'results' / ('episode_%06d.json' % eid), dict(
                episode_index=eid, suite=item['suite'], task_id=item['evaluation_task_id'],
                trial_id=item['trial_id'], original_success=False, converted_success=None,
                conversion_status='not_applicable_CAP_startup_failure', snapshot=str(snapshot),
                original_source_episode=item['source_episode'], available_vla_steps=item['vla_steps'],
                cap_actions=0, original_recording_complete=False, retained=True))
            continue
        assert item['recording_audit']['complete']
        rel = 'episodes/episode_%06d' % eid
        link = source / 'sources/capx_recovery17' / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(snapshot, target_is_directory=True)
        task = tasks[(item['suite'], item['evaluation_task_id'])]
        assert task['instruction'] == original['instruction']
        ep = dict(suite=item['suite'], task_id=item['evaluation_task_id'], trial_id=item['trial_id'],
            seed=original['seed'], instruction=original['instruction'], path=rel, trajectory_path=rel + '/trajectory',
            source_episode=str(snapshot), original_source_episode=item['source_episode'],
            success=False, used_capx=True, steps=original['steps'], handoff_step=item['handoff_step'],
            observations=original['steps'] + 1, steps_by_owner=original['steps_by_owner'], segments=original['segments'])
        rec = dict(episode_index=eid, source='capx_recovery_failure', suite=ep['suite'],
            evaluation_task_id=ep['task_id'], canonical_task_index=task['canonical_task_index'],
            trial_id=ep['trial_id'], seed=ep['seed'], instruction=ep['instruction'],
            path='sources/capx_recovery17/' + rel, trajectory_path='sources/capx_recovery17/' + rel + '/trajectory',
            source_episode=str(snapshot), source_record=ep, handoff_step=ep['handoff_step'],
            action_rows=ep['steps'], observations=ep['observations'], steps_by_owner=ep['steps_by_owner'],
            segments=ep['segments'], original_action_log_sha256=core.sha(snapshot / 'actions.jsonl'))
        old = known.get(core.identity(ep))
        if old:
            old_ep = core.read(old)['source_record']['source_record']
            for relfile in ['actions.jsonl', 'trajectory/manifest.json', 'trajectory/model.xml', 'trajectory/initial.npz']:
                assert core.sha(snapshot / relfile) == core.sha(Path(old_ep['source_episode']) / relfile)
        records.append(dict(record=rec, reused_report=str(old) if old else None))
    plan = dict(records=records, total=20, cap_recordings=18, startup_only=2,
        original_selection_sha256=core.sha(raw / 'recovery_failure20_raw/selection.json'),
        settings=helper.SETTINGS, outcome_filter=False, replacements_allowed=False)
    if (work / 'plan.json').exists():
        assert core.read(work / 'plan.json') == plan
    else:
        (source / 'index.jsonl').write_text(''.join(__import__('json').dumps(x['record']) + '\n' for x in records))
        core.save(work / 'plan.json', plan)
    return plan


def convert_one(root, work, item):
    rec = item['record']; eid = rec['episode_index']
    helper = configure(root, work)
    if item['reused_report']:
        report_file = Path(item['reused_report'])
        r = core.read(report_file)
        previous = report_file.parent.parent.parent
        configure(root, work, previous / 'indexed_sources')
        checked_file = previous / 'independent_replays' / report_file.parent.name / 'report.json'
        if not checked_file.exists():
            independent = work / 'independent_reused'
            independent.mkdir(exist_ok=True)
            checked_file = independent / report_file.parent.name / 'report.json'
            if not checked_file.exists():
                helper.replay.replay_episode(r, report_file.parent.parent, independent)
    else:
        helper.fixed_labels(rec)
        r = helper.engine.run_episode(rec, eid, work / 'attempts', helper.SETTINGS)
        report_file = work / 'attempts' / ('episode_%06d/report.json' % eid)
        independent = work / 'independent_replays'
        independent.mkdir(exist_ok=True)
        checked_file = independent / ('episode_%06d/report.json' % eid)
        if not checked_file.exists():
            helper.replay.replay_episode(r, work / 'attempts', independent)
    c = core.read(checked_file)
    assert c['success'] == r['success'] and c['max_sim_state_error'] < 1e-8
    assert c['source_report_sha256'] == core.sha(report_file)
    assert c['action_record_sha256'] == core.sha(report_file.parent / 'states_actions.npz')
    retained = core.retain_episode(report_file, checked_file, work, eid)
    result = dict(episode_index=eid, suite=rec['suite'], task_id=rec['evaluation_task_id'], trial_id=rec['trial_id'],
        canonical_task_index=rec['canonical_task_index'], original_success=False, converted_success=r['success'],
        conversion_status='complete', reused=bool(item['reused_report']), retained=True,
        snapshot=rec['source_episode'], original_source_episode=rec['source_record']['original_source_episode'],
        retained_episode=str(retained), report=str(report_file), independent_report=str(checked_file),
        converted_steps=r['executed_steps'], added_compensation_steps=r['added_compensation_steps'],
        max_replay_state_error=c['max_sim_state_error'], stop_reason=r['stop_reason'])
    core.save(retained / 'conversion_annotation.json', result)
    core.save(work / 'results' / ('episode_%06d.json' % eid), result)
    print('FAILURE_SOURCE_CONVERTED', eid, r['success'], r['executed_steps'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args(); root = args.root.resolve()
    raw = root / 'dataset_subsets/pi05_recovery_raw_scheme1_20261102_v1'
    work = root / 'trajectory_conversion_tests/pi05_scheme1_failure20_20261102'
    work.mkdir(parents=True, exist_ok=True)
    plan = prepare(root, raw, work)
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(convert_one, root, work, x) for x in plan['records']]
        for f in as_completed(jobs):
            try:
                f.result()
            except Exception as exc:
                errors.append(dict(error=repr(exc), traceback=traceback.format_exc()))
            results = [core.read(x) for x in sorted((work / 'results').glob('episode_*.json'))]
            core.save(work / 'status.json', dict(state='running', completed=len(results), total=20,
                technical_errors=errors, startup_only=2, retained_all_outcomes=True))
    assert not errors, errors
    assert len(results) == 20
    assert core.sha(raw / 'recovery_failure20_raw/selection.json') == plan['original_selection_sha256']
    core.save(work / 'summary.json', dict(state='complete', results=results, technical_errors=[]))
    core.save(work / 'status.json', dict(state='complete', completed=20, total=20, startup_only=2, technical_errors=[]))


if __name__ == '__main__':
    main()
