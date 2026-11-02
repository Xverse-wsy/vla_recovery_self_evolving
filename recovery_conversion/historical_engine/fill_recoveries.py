"""Keep validated OSC successes; try other original recoveries for missing tasks.

The existing goal_encoding converter is reused without changing its control law.
Each task stops searching after one independently replayed successful candidate.
Original datasets and simulation records are read-only. No training is launched.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
BASE = ROOT/'training_configs/pi05_matched17_osc_split_20261028'
sys.path.insert(0, str(BASE))
import convert_recovery as engine
import replay_compensated17 as replay

ORIGINAL_SOURCE = engine.SOURCE
ORIGINAL_REPLAY = engine.OUTPUT.with_name(engine.OUTPUT.name+'_goal_encoding')
WORK = ROOT/'trajectory_conversion_tests/pi05_matched17_compensation_fill_20261028'
SOURCE = WORK/'indexed_sources'
LABELS = WORK/'fixed_goals'
ATTEMPTS = WORK/'alternate_attempts'
SETTINGS = dict(engine.SETTINGS, version=2, completion_rule='goal_encoding',
    split_completion='the final bounded action encodes the full original absolute target without saturation; physical convergence is recorded separately',
    blocked_target='stop and reject if the target still cannot be encoded within 200 controls')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    engine.save_json(path, value)


def taskkey(rec):
    return rec['suite'], rec['evaluation_task_id']


def accepted_originals():
    source = read(ORIGINAL_REPLAY/'summary.json')
    independent = read(ORIGINAL_REPLAY/'independent_full17_replay_20261028/summary.json')
    assert source['completed']==17 and source['successful']==10 and not source['technical_errors']
    assert independent['completed']==17 and independent['successful']==10
    assert independent['all_outcomes_match_conversion'] and independent['max_sim_state_error']==0
    verified = {r['episode_index']:r for r in independent['results']}
    good, missing = [], []
    for report in source['results']:
        assert report['settings']==SETTINGS
        check = verified[report['episode_index']]
        assert check['success']==report['success']
        assert check['source_report_sha256']==engine.sha(ORIGINAL_REPLAY/f'episode_{report["episode_index"]:06d}/report.json')
        assert check['action_record_sha256']==engine.sha(ORIGINAL_REPLAY/f'episode_{report["episode_index"]:06d}/states_actions.npz')
        item = dict(record=report['source_record'], replay=str(ORIGINAL_REPLAY),
                    report=str(ORIGINAL_REPLAY/f'episode_{report["episode_index"]:06d}/report.json'),
                    independent_report=str(ORIGINAL_REPLAY/'independent_full17_replay_20261028'/f'episode_{report["episode_index"]:06d}/report.json'),
                    replacement=False)
        (good if report['success'] else missing).append(item)
    assert len(good)==10 and len(missing)==7
    return good, missing


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK/'plan.json').exists():
        return read(WORK/'plan.json')
    good, missing = accepted_originals()
    native = SOURCE/'sources/capx_recovery17'
    native.mkdir(parents=True, exist_ok=True)
    (SOURCE/'read_trajectory.py').symlink_to(ORIGINAL_SOURCE/'read_trajectory.py')
    originals = {taskkey(x['record']):x['record'] for x in missing}
    output_root = ROOT/'wcm_pi05_monitor/outputs'
    main = output_root/'hybrid_independent_gpu1_20261019_172047/eval/results'
    roots = [main]+[p/'eval/results' for p in sorted(output_root.glob('hybrid_*'))
                   if p.is_dir() and p/'eval/results'!=main and 'smoke' not in p.name]
    candidates, groups = [], []
    next_eid = 100
    for key, template in originals.items():
        suite, tid = key
        used_ep = Path(template['source_record']['source_episode'])
        used_actions = engine.sha(used_ep/'actions.jsonl')
        seen = {used_actions}
        options = []
        for root in roots:
            for path in sorted((root/suite/f'task_{tid:02d}').glob('trial_*/report.json')):
                report = read(path)
                if not (report.get('success') and report.get('used_capx')) or report.get('synthetic_alarm_test'):
                    continue
                manifest = path.parent/'trajectory/manifest.json'
                if not manifest.exists() or not read(manifest).get('complete'):
                    continue
                digest = engine.sha(path.parent/'actions.jsonl')
                if digest in seen:
                    continue
                seen.add(digest)
                options.append((report['steps'], report['trial_id'], str(path.parent), report, digest))
        options.sort(key=lambda x:(x[0], x[1], x[2]))
        recs = []
        for steps, trial, original, report, digest in options:
            rec = copy.deepcopy(template)
            rel = f'episodes/{suite}/task_{tid:02d}/candidate_{next_eid:06d}_trial_{trial:02d}'
            link = native/rel
            link.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to(original, target_is_directory=True)
            handoff = report['handoff']['step']
            ep = dict(suite=suite,task_id=tid,trial_id=trial,instruction=report['instruction'],
                seed=report['seed'],sample_type='capx_recovery_success',path=rel,
                trajectory_path=rel+'/trajectory',source_episode=original,success=True,used_capx=True,
                steps=steps,observations=steps+1,steps_by_owner=report['steps_by_owner'],
                segments=report['segments'],handoff_step=handoff,validated=True)
            rec.update(episode_index=next_eid,trial_id=trial,seed=report['seed'],
                path='sources/capx_recovery17/'+rel,trajectory_path='sources/capx_recovery17/'+rel+'/trajectory',
                action_rows=steps,observations=steps+1,steps_by_owner=report['steps_by_owner'],
                segments=report['segments'],handoff_step=handoff,recovery_action_range_inclusive=[handoff+1,steps],
                source_dataset=str(native),source_episode=original,source_record=ep,
                replaces_matched_episode_index=template['episode_index'],original_action_log_sha256=digest)
            candidates.append(rec)
            recs.append(rec)
            next_eid += 1
        groups.append(dict(suite=suite,task_id=tid,original_episode_index=template['episode_index'],
                           candidates=recs,available_alternatives=len(recs)))
    (SOURCE/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidates))
    plan = dict(state='prepared',method='goal_encoding',settings=SETTINGS,
        original_successes=good,missing_originals=missing,task_groups=groups,
        alternate_candidates=len(candidates),candidate_order='shorter source sequence, then trial_id',
        search_roots=[str(p) for p in roots],source_unchanged=True,training_started=False)
    save(WORK/'plan.json',plan)
    save(WORK/'initial_successes.json',dict(state='complete',accepted=good,training_started=False))
    print('PREPARED',json.dumps(dict(original_successes=10,missing_tasks=7,
          alternatives=[dict(suite=g['suite'],task=g['task_id'],count=len(g['candidates'])) for g in groups])),flush=True)
    return plan


def program_ranges(ep):
    directory = SOURCE/'sources/capx_recovery17'/ep['path']/'capx'
    requests = sorted(directory.glob('turn_*_request.json'))
    assert requests
    starts = []
    for message in read(requests[-1])['input']:
        if message.get('role')!='user':
            continue
        content = message.get('content',[])
        if isinstance(content,str):
            content=[dict(text=content)]
        for block in content:
            text=block.get('text','')
            if 'Current robot state:' in text:
                starts.append(int(json.loads(text.split('Current robot state:',1)[1])['authoritative_environment']['steps']))
    ranges=[dict(turn=i,start=s+1,end=e,path=directory/f'turn_{i:02d}.py')
            for i,(s,e) in enumerate(zip(starts,starts[1:]+[ep['steps']]))
            if (directory/f'turn_{i:02d}.py').exists()]
    assert ranges and ranges[0]['start']==ep['handoff_step']+1
    return ranges


def fixed_labels(rec):
    sys.path.insert(0,str(engine.HELPERS))
    import probe_conversion as probe
    probe.SOURCE=SOURCE/'sources/capx_recovery17'
    ep=rec['source_record']
    data=probe.source_cache(ep,ATTEMPTS/'source_cache')
    out=LABELS/f'offline_labels/episode_{rec["episode_index"]:06d}'
    if (out/'labels_and_tcp.npz').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(SOURCE/rec['trajectory_path']/'model.xml'))
    kin=mujoco.MjData(model)
    qids=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'robot0_joint{i}')] for i in range(1,8)])
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
    assert site>=0
    n=len(data['actions']); handoff=ep['handoff_step']
    arrays=dict(original_action=data['actions'],owner=data['owners'],capx_goal_valid=np.arange(n)>=handoff,
        q_goal=np.full((n,7),np.nan),target_position=np.full((n,3),np.nan),
        target_quat_xyzw=np.full((n,4),np.nan),clipped=np.zeros((n,6),bool))
    ranges=program_ranges(ep)
    details=[]
    for i in range(n):
        matches=[p for p in ranges if p['start']<=i+1<=p['end']]
        if i<handoff:
            assert data['owners'][i]=='pi05'
            details.append(dict(source_program=None,source_turn=None))
            continue
        assert data['owners'][i]=='capx' and len(matches)==1
        details.append(dict(source_program=str(matches[0]['path']),source_turn=matches[0]['turn']))
        state=data['sim/state'][i]
        kin.qpos[:]=state[1:1+model.nq]
        mujoco.mj_kinematics(model,kin)
        before_p=kin.site_xpos[site].copy(); before_r=kin.site_xmat[site].reshape(3,3).copy()
        qgoal=kin.qpos[qids]+np.clip(data['actions'][i,:7],-1,1)*.05
        kin.qpos[qids]=qgoal
        mujoco.mj_kinematics(model,kin)
        p=kin.site_xpos[site].copy(); r=kin.site_xmat[site].reshape(3,3).copy()
        arrays['q_goal'][i]=qgoal; arrays['target_position'][i]=p
        arrays['target_quat_xyzw'][i]=Rotation.from_matrix(r).as_quat()
        delta=np.r_[(p-before_p)/.05,Rotation.from_matrix(r@before_r.T).as_rotvec()/.5]
        arrays['clipped'][i]=np.abs(delta)>1
    np.savez_compressed(out/'labels_and_tcp.npz',**arrays)
    (out/'action_provenance.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in details))
    print('FIXED_GOALS_READY',rec['episode_index'],n,flush=True)


def run_task(group):
    engine.SOURCE=SOURCE
    engine.OFFLINE=LABELS
    replay.SOURCE=SOURCE
    attempts=[]
    chosen=None
    for rec in group['candidates']:
        eid=rec['episode_index']
        try:
            fixed_labels(rec)
            report=engine.run_episode(rec,eid,ATTEMPTS,SETTINGS)
            result=dict(episode_index=eid,trial_id=rec['trial_id'],success=report['success'],
                steps=report['executed_steps'],inserted=report['added_compensation_steps'],
                stop_reason=report['stop_reason'],report=str(ATTEMPTS/f'episode_{eid:06d}/report.json'))
            if report['success']:
                verified_path=WORK/'independent_alternates'/f'episode_{eid:06d}/report.json'
                if verified_path.exists():
                    checked=read(verified_path)
                    assert checked['success'] and checked['source_report_sha256']==engine.sha(Path(result['report']))
                    assert checked['action_record_sha256']==engine.sha(ATTEMPTS/f'episode_{eid:06d}/states_actions.npz')
                else:
                    checked=replay.replay_episode(report,ATTEMPTS,WORK/'independent_alternates')
                assert checked['success'] and checked['max_sim_state_error']<1e-8
                result['independent_validation']=str(verified_path)
                chosen=dict(record=rec,replay=str(ATTEMPTS),report=result['report'],
                    independent_report=str(verified_path),replacement=True,
                    replaces_matched_episode_index=group['original_episode_index'])
            attempts.append(result)
        except Exception as exc:
            attempts.append(dict(episode_index=eid,trial_id=rec['trial_id'],technical_error=repr(exc),
                                 traceback=traceback.format_exc(),success=False))
            print('ALTERNATIVE_TECHNICAL_ERROR',json.dumps(attempts[-1]),flush=True)
        save(WORK/f'tasks/{group["suite"]}_task{group["task_id"]:02d}.json',
             dict(suite=group['suite'],task_id=group['task_id'],attempts=attempts,accepted=chosen))
        if chosen:
            break
    return dict(suite=group['suite'],task_id=group['task_id'],original_episode_index=group['original_episode_index'],
        available_alternatives=len(group['candidates']),attempts=attempts,accepted=chosen,
        state='complete',exhausted=chosen is None)


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    plan=prepare()
    if args.prepare_only:
        return
    ATTEMPTS.mkdir(parents=True,exist_ok=True)
    (WORK/'independent_alternates').mkdir(exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run_task,g) for g in plan['task_groups']]
        for future in as_completed(jobs):
            results.append(future.result())
            save(WORK/'status.json',dict(state='converting_alternates',completed_tasks=len(results),total_tasks=7,
                accepted_recoveries=10+sum(r['accepted'] is not None for r in results),training_started=False))
    accepted=plan['original_successes']+[r['accepted'] for r in results if r['accepted']]
    accepted.sort(key=lambda x:taskkey(x['record']))
    assert len({taskkey(x['record']) for x in accepted})==len(accepted)
    errors=[a for r in results for a in r['attempts'] if 'technical_error' in a]
    summary=dict(state='complete' if not errors else 'technical_error',method='goal_encoding',
        original_tested=17,original_successes=10,replacement_tasks=sum(r['accepted'] is not None for r in results),
        accepted_recoveries=len(accepted),missing_tasks=[dict(suite=r['suite'],task_id=r['task_id']) for r in results if not r['accepted']],
        alternate_attempts=sum(len(r['attempts']) for r in results),technical_errors=errors,
        accepted=accepted,task_results=sorted(results,key=lambda r:(r['suite'],r['task_id'])),
        source_unchanged=True,training_started=False)
    save(WORK/'summary.json',summary)
    save(WORK/'accepted.json',dict(state=summary['state'],accepted=accepted,training_started=False))
    save(WORK/'status.json',{k:v for k,v in summary.items() if k not in ['accepted','task_results','technical_errors']})
    print('FILL_DONE',json.dumps({k:v for k,v in summary.items() if k not in ['accepted','task_results','technical_errors']}),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
