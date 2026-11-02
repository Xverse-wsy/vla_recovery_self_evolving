import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('balanced_raw', Path(__file__).resolve().parents[1]/'prepare_balanced_recovery_raw.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pool():
    rows = []
    # Match the number of distinct successful tasks and enough candidates.
    for suite, capacities in [('libero_10', [7,6,9,2,4,2,10,3,3]),
                              ('libero_goal', [6,2,2,4,5]),
                              ('libero_spatial', [1,2,2,5,2,9]), ('libero_object', [1,1])]:
        for task, capacity in enumerate(capacities):
            for trial in range(capacity):
                rows.append(dict(suite=suite, instruction=str(task), original_success=True,
                    source_episode=f'{suite}/{task}/{trial}', trajectory_exists=True,
                    original_trajectory_complete=True))
    return rows


def test_scheme1_exact_quotas_maximizes_success_task_coverage_and_is_deterministic():
    from collections import Counter
    rows = pool()
    failures = [dict(suite='libero_10', instruction='0', source_episode='failure')]
    before = json.dumps(failures, sort_keys=True)
    picked = m.select_success(rows, failures)
    assert len(picked) == len({r['source_episode'] for r in picked}) == 20
    assert Counter(r['suite'] for r in picked) == m.SUCCESS_QUOTAS
    assert len({m.task_key(r) for r in picked}) == 16
    assert max(Counter(m.task_key(r) for r in picked).values()) == 2
    assert picked == m.select_success(list(reversed(rows)), failures)
    assert json.dumps(failures, sort_keys=True) == before


def test_success_selection_rejects_shortage_or_conflicting_outcomes():
    rows = pool()
    with pytest.raises(ValueError, match='Not enough'):
        m.select_success([r for r in rows if r['suite'] != 'libero_object'], [])
    bad = [dict(r) for r in rows]
    bad[0]['original_success'] = False
    with pytest.raises(ValueError, match='non-success'):
        m.select_success(bad, [])
    with pytest.raises(ValueError, match='overlap'):
        m.select_success(rows, [dict(rows[0])])


def test_snapshot_copies_small_metadata_and_links_large_files(tmp_path):
    import io
    source, out = tmp_path/'original', tmp_path/'package'
    source.mkdir(); out.mkdir()
    (source/'report.json').write_text('{"success": true}')
    (source/'large.npz').write_bytes(b'x'*(1024*1024+1))
    listing = io.StringIO()
    result = m.snapshot(source, out/'episode', listing, out, 'hardlink')
    assert result['files'] == 2 and result['hardlinked_files'] == 1
    assert (source/'large.npz').stat().st_ino == (out/'episode/large.npz').stat().st_ino
    assert (source/'report.json').stat().st_ino != (out/'episode/report.json').stat().st_ino
    records = [json.loads(line) for line in listing.getvalue().splitlines()]
    assert {r['path'] for r in records} == {'episode/report.json','episode/large.npz'}
    assert all((out/r['path']).stat().st_size == r['bytes'] for r in records)
    assert next(r for r in records if r['method'] == 'copy_sha256')['sha256'] == m.digest(source/'report.json')


def test_outcome_audit_is_fail_closed_and_does_not_require_failed_replay(tmp_path):
    source = tmp_path/'suite/task_00/trial_00'
    source.mkdir(parents=True)
    p = source/'report.json'
    p.write_text(json.dumps(dict(success=False, used_capx=True, suite='suite', instruction='task')))
    row = dict(source_episode=str(source), source_report_sha256=m.digest(p),
               suite='suite', instruction='task', converted_success=True)
    assert m.check_original(row, tmp_path, False) == source
    with pytest.raises(ValueError, match='outcome'):
        m.check_original(row, tmp_path, True)
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError, match='changed'):
        m.check_original(row, tmp_path, False)


def test_shard_audit_checks_original_checksum_and_preserves_incomplete_recording(tmp_path):
    root = tmp_path/'trajectory'; root.mkdir()
    (root/'initial.npz').write_bytes(b'initial')
    (root/'model.xml').write_text('<model/>')
    shard = root/'steps.npz'; shard.write_bytes(b'original')
    manifest = dict(complete=False, steps=2, observations=3,
        shards=[dict(file='steps.npz', first_step=1, last_step=2, sha256=m.digest(shard))])
    (root/'manifest.json').write_text(json.dumps(manifest))
    row = dict(original_steps=3)
    report = m.audit_trajectory(tmp_path, row)
    assert report['saved_steps'] == 2 and report['complete'] is False
    manifest['complete'] = True
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Complete recording'):
        m.audit_trajectory(tmp_path, row)
    manifest['complete'] = False
    (root/'manifest.json').write_text(json.dumps(manifest))
    shard.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        m.audit_trajectory(tmp_path, row)


def test_complete_raw_package_keeps_confirmed_failures_and_separates_40_episodes(tmp_path):
    results = tmp_path/'results'

    def original(suite, task, trial, success, complete=True):
        source = results/suite/f'task_{task:02d}'/f'trial_{trial:03d}'
        trajectory = source/'trajectory'; trajectory.mkdir(parents=True)
        cap = 1 if complete else 0
        report = dict(success=success, used_capx=True, suite=suite, task_id=task,
            trial_id=trial, instruction=str(task), steps=2,
            steps_by_owner=dict(pi05=2-cap, capx=cap), trajectory=dict(complete=complete))
        (source/'report.json').write_text(json.dumps(report))
        (trajectory/'initial.npz').write_bytes(b'initial')
        (trajectory/'model.xml').write_text('<model/>')
        shard = trajectory/'steps.npz'; shard.write_bytes(b'raw-payload')
        saved = 2 if complete else 1
        (trajectory/'manifest.json').write_text(json.dumps(dict(complete=complete,
            steps=saved, observations=saved+1, shards=[dict(file='steps.npz', first_step=1,
                last_step=saved, sha256=m.digest(shard))])))
        return dict(source_episode=str(source), source_report_sha256=m.digest(source/'report.json'),
            suite=suite, evaluation_task_id=task, trial_id=trial, instruction=str(task),
            original_success=success, original_steps=2, vla_steps=2-cap, cap_steps=cap,
            original_trajectory_complete=complete, trajectory_exists=True)

    for row in pool():
        suite,task,trial = row['source_episode'].split('/')
        original(suite,int(task),int(trial),True)
    failures=[]
    for suite,quota in m.FAILURE_QUOTAS.items():
        for i in range(quota):
            failures.append(original(suite,i%4,100+i,False,
                complete=not (suite=='libero_goal' and i<2)))
    n = len(list(results.glob('*/task_*/trial_*/report.json')))
    for i in range(400-n):
        p = results/'libero_10/task_99'/f'trial_{i:03d}'/'report.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(success=True, used_capx=False)))
    confirmed = tmp_path/'confirmed'; confirmed.mkdir()
    selection = confirmed/'selection.json'
    selection.write_text(json.dumps(dict(selected=failures)))
    before = m.digest(selection)
    (confirmed/'CURRENT_SELECTION.json').write_text(json.dumps(dict(status='user_selected',
        selection_manifest_sha256=before)))
    output = tmp_path/'package'
    report = m.prepare(results, selection, output)
    assert report['state']=='complete' and report['combined40']['episodes']==40
    assert report['combined40']['by_suite']==dict(libero_10=13,libero_goal=13,libero_spatial=12,libero_object=2)
    assert report['success20']['tasks']==16
    assert report['failure20']['zero_cap_action_episodes']==2
    assert report['failure20']['incomplete_recordings']==2
    assert report['saved_raw_steps']==78
    assert report['train_ready'] is False and report['conversion_executed'] is False
    assert m.digest(selection)==m.digest(output/'recovery_failure20_raw/selection.json')==before
    for group in ['recovery_success20_raw','recovery_failure20_raw']:
        entries=[json.loads(x) for x in (output/group/'episodes.jsonl').read_text().splitlines()]
        assert len(entries)==20
        assert all((output/r['snapshot_relative_path']/'trajectory/manifest.json').is_file() for r in entries)
    with pytest.raises(FileExistsError):
        m.prepare(results, selection, output)
