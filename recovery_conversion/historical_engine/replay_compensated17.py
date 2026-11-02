"""Independently replay all 17 saved excess-compensated OSC action sequences.

Measure LIBERO task success from actual stepping, including failed candidates.
No conversion feedback code, strict pose-arrival gate, model inference or training.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from convert_recovery import ROOT, SOURCE, OUTPUT, HELPERS, rows, save_json, sha

DEFAULT_REPLAY = OUTPUT.with_name(OUTPUT.name+'_goal_encoding')


def replay_episode(report, replay, output):
    start = time.monotonic()
    eid = report['episode_index']
    destination = output/f'episode_{eid:06d}'
    destination.mkdir(exist_ok=False)
    assert report['settings']['completion_rule']=='goal_encoding'
    sys.path.insert(0,str(HELPERS))
    import probe_conversion as probe
    probe.SOURCE = SOURCE/'sources/capx_recovery17'
    ep = report['source_record']['source_record']
    source = probe.source_cache(ep,replay/'source_cache')
    original = replay/f'episode_{eid:06d}'
    file = original/'states_actions.npz'
    with np.load(file,allow_pickle=False) as f:
        actions = f['actions'].copy()
        reference = f['sim_states'].copy()
        source_steps = f['source_action_steps'].copy()
        inserted = f['inserted_compensation_actions'].copy()
    provenance = rows(original/'action_provenance.jsonl')
    assert actions.shape==(report['executed_steps'],7)
    assert len(reference)==len(actions)+1 and len(provenance)==len(actions)
    assert np.isfinite(actions).all() and np.isfinite(reference).all()
    assert np.max(np.abs(actions[ep['handoff_step']:]))<=1
    assert np.all(np.diff(source_steps)>=0)
    np.testing.assert_array_equal(inserted,np.r_[False,np.diff(source_steps)==0])
    assert int(inserted.sum())==report['added_compensation_steps']
    for i,p in enumerate(provenance):
        np.testing.assert_array_equal(actions[i],np.asarray(p['action'],np.float32))
        assert int(source_steps[i])==p['source_action_step']
        if inserted[i]:
            previous = provenance[i-1]
            assert p['absolute_goal_position']==previous['absolute_goal_position']
            assert p['absolute_goal_quat_xyzw']==previous['absolute_goal_quat_xyzw']
            assert p['source_program']==previous['source_program']
            assert p['action'][-1]==previous['action'][-1]
    assert int(inserted.sum())==sum(p['inserted_compensation_action'] for p in provenance)
    env = probe.environment(ep,images=False)
    maximum = prefix_error = 0.
    success = False
    trace = (destination/'steps.jsonl').open('x')
    try:
        probe.restore_initial(env,source)
        np.testing.assert_array_equal(env.sim.get_state().flatten(),reference[0])
        for i,action in enumerate(actions):
            _,reward,done,_ = env.step(action.astype(np.float64))
            # Do not forward/alter authoritative MuJoCo geometry between step
            # and its original LIBERO success predicate.
            success = bool(env.check_success())
            state = env.sim.get_state().flatten().copy()
            error = float(np.max(np.abs(state-reference[i+1])))
            maximum = max(maximum,error)
            assert error < 1e-8,(eid,i+1,error)
            assert abs(state[0]-reference[i,0]-.05)<1e-8
            if i < ep['handoff_step']:
                delta = float(np.max(np.abs(state-source['sim/state'][i+1])))
                prefix_error = max(prefix_error,delta)
                assert delta<1e-8
            assert success==provenance[i]['success_after_action'],(eid,i+1,success)
            assert not success or i==len(actions)-1
            assert not done or success or i==len(actions)-1
            trace.write(json.dumps(dict(step=i+1,source_action_step=int(source_steps[i]),
                inserted_compensation_action=bool(inserted[i]),success=success,
                reward=float(reward),terminated=bool(done),max_state_error=error))+'\n')
            if (i+1)%400==0:
                trace.flush()
                print('COMPENSATION_REPLAY_PROGRESS '+json.dumps(dict(episode=eid,step=i+1,total=len(actions))),flush=True)
        assert success==report['success']
        result = dict(state='complete',episode_index=eid,suite=ep['suite'],task_id=ep['task_id'],
            trial_id=ep['trial_id'],instruction=ep['instruction'],steps=len(actions),success=success,
            original_conversion_success=report['success'],added_compensation_steps=int(inserted.sum()),
            source_stop_reason=report['stop_reason'],max_sim_state_error=maximum,
            prefix_sim_state_error=prefix_error,goal_pose_recomputed_during_replay=False,
            exact_saved_osc_actions=True,strict_2mm_arrival_gate=False,
            action_record_sha256=sha(file),source_report_sha256=sha(original/'report.json'),
            elapsed_seconds=time.monotonic()-start,training_started=False)
        save_json(destination/'report.json',result)
        print('COMPENSATION_REPLAY_DONE '+json.dumps(result),flush=True)
        return result
    finally:
        trace.close()
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',type=Path,default=DEFAULT_REPLAY)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--workers',type=int,default=3)
    args = ap.parse_args()
    source_summary = args.replay/'summary.json'
    source = json.loads(source_summary.read_text())
    assert source['state']=='complete' and source['completed']==17 and not source['technical_errors']
    records = sorted(source['results'],key=lambda r:r['episode_index'])
    assert [r['episode_index'] for r in records]==list(range(57,74))
    assert all(r['settings']['completion_rule']=='goal_encoding' for r in records)
    output = args.output or args.replay/'independent_full17_replay_20261028'
    output.mkdir(parents=True,exist_ok=False)
    before = sha(source_summary)
    save_json(output/'plan.json',dict(method='fixed_absolute_goal_bounded_excess_compensation',
        replay=str(args.replay),total=17,workers=args.workers,source_summary_sha256=before,
        source_successes=source['successful'],strict_2mm_arrival_gate=False,training_started=False))
    results,errors = [],[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(replay_episode,r,args.replay,output):r for r in records}
        for future in as_completed(pending):
            r = pending[future]
            try:
                results.append(future.result())
            except Exception as exc:
                item=dict(episode_index=r['episode_index'],error=repr(exc),traceback=traceback.format_exc())
                errors.append(item)
                print('COMPENSATION_REPLAY_TECHNICAL_ERROR '+json.dumps(item),flush=True)
            save_json(output/'status.json',dict(state='running',completed=len(results),total=17,
                successful=sum(r['success'] for r in results),technical_errors=errors))
    results.sort(key=lambda r:r['episode_index'])
    per_suite={}
    for suite in sorted({r['suite'] for r in results}):
        subset=[r for r in results if r['suite']==suite]
        n=sum(r['success'] for r in subset)
        per_suite[suite]=dict(episodes=len(subset),successful=n,success_rate=n/len(subset))
    assert sha(source_summary)==before
    final=dict(state='complete' if not errors else 'technical_error',total=17,completed=len(results),
        successful=sum(r['success'] for r in results),failed=sum(not r['success'] for r in results),
        success_rate=sum(r['success'] for r in results)/17,technical_errors=errors,
        per_suite=per_suite,all_outcomes_match_conversion=not errors and len(results)==17,
        max_sim_state_error=max((r['max_sim_state_error'] for r in results),default=None),
        added_compensation_steps=sum(r['added_compensation_steps'] for r in results),
        failure_reasons=dict(Counter(r['source_stop_reason'] for r in results if not r['success'])),
        population='17 original CAP-X-success trajectories converted to 7D OSC; not evaluation of a newly trained policy',
        source_unchanged=True,training_started=False,results=results)
    save_json(output/'summary.json',final)
    save_json(output/'status.json',{k:v for k,v in final.items() if k!='results'})
    print('COMPENSATION_FULL17_RESULT '+json.dumps({k:v for k,v in final.items() if k!='results'}),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
