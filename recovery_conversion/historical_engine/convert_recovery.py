"""Replay matched CAP-X recoveries using fixed FK goals and bounded OSC steps.

This does not train, replan, reset midway, or change the original data. Extra
steps get fresh observations from the one continuously running environment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
SOURCE = ROOT/'dataset_subsets/pi05_matched17_comparison_20261028/author40_autonomous17_capx17_74'
OFFLINE = ROOT/'dataset_subsets/pi05_author40_autonomous34_capx17_offline_osc91_20261027'
HELPERS = ROOT/'training_configs/pi05_sft_merged91_20261027'
OUTPUT = ROOT/'trajectory_conversion_tests/pi05_matched17_osc_split_20261028'
SETTINGS = dict(version=1, position_tolerance_m=.002, rotation_tolerance_deg=2.,
                consecutive_confirmations=3, max_controls_per_split_target=200,
                translation_range_per_axis_m=.05, rotation_range_per_axis_rad=.5,
                split_trigger='initial command to the fixed original goal exceeds any OSC axis bound',
                unsplit_target='execute once, like the original control sequence',
                blocked_target='stop and reject; never silently skip compensation',
                source_goal='FK of the ORIGINAL source state plus officially clipped joint delta',
                current_pose='independent kinematics data; never forward authoritative physics for measurement')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def state8(raw):
    q = np.asarray(raw['robot0_eef_quat']).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1-q[3]**2)
    aa = np.zeros(3) if np.isclose(den, 0.) else q[:3]*2*np.arccos(q[3])/den
    return np.r_[raw['robot0_eef_pos'], aa, raw['robot0_gripper_qpos']].astype(np.float32)


class Kinematics:
    """Share the immutable model, never the authoritative simulation data."""
    def __init__(self, env):
        self.model = env.sim.model._model
        self.data = mujoco.MjData(self.model)
        self.site = env.robots[0].eef_site_id
        self.qids = np.asarray(env.robots[0]._ref_joint_pos_indexes)

    def pose(self, state, qgoal=None):
        state = np.asarray(state)
        assert len(state) == 1+self.model.nq+self.model.nv
        self.data.qpos[:] = state[1:1+self.model.nq]
        if qgoal is not None:
            self.data.qpos[self.qids] = qgoal
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()


def verify_goals(kin, data, label, handoff):
    np.testing.assert_array_equal(label['original_action'], data['actions'])
    np.testing.assert_array_equal(label['owner'], data['owners'])
    assert not label['capx_goal_valid'][:handoff].any()
    assert label['capx_goal_valid'][handoff:].all()
    max_p = max_r = 0.
    target_matrices = np.full((len(data['actions']), 3, 3), np.nan)
    for i in range(handoff, len(data['actions'])):
        source = data['sim/state'][i]
        q = source[1:1+kin.model.nq][kin.qids]
        qgoal = q + np.clip(data['actions'][i, :7], -1, 1)*.05
        np.testing.assert_allclose(qgoal, label['q_goal'][i], rtol=0, atol=1e-12)
        p, r = kin.pose(source, qgoal)
        target_matrices[i] = r
        err_p = float(np.linalg.norm(p-label['target_position'][i]))
        err_r = float(Rotation.from_matrix(r @ Rotation.from_quat(label['target_quat_xyzw'][i]).as_matrix().T).magnitude())
        max_p, max_r = max(max_p, err_p), max(max_r, err_r)
    # The older labels passed through robosuite's float32 matrix->quaternion
    # conversion. Check its roundoff, then use the newly computed full FK matrix.
    assert max_p < 1e-10 and max_r < 2e-6, (max_p, max_r)
    label['target_rotation_recomputed'] = target_matrices
    return dict(checked_goals=len(data['actions'])-handoff, max_position_error_m=max_p,
                max_orientation_error_rad=max_r, recomputed_from_original_sim_state=True)


def command(p, r, goal_p, goal_r, grip):
    delta = np.r_[(goal_p-p)/.05, Rotation.from_matrix(goal_r @ r.T).as_rotvec()/.5]
    if np.linalg.norm(delta[3:]) < 1e-12:
        delta[3:] = [1e-10, 0, 0]
    action = np.r_[np.clip(delta, -1, 1), np.clip(grip, -1, 1)].astype(np.float32)
    return action, bool(np.any(np.abs(delta) > 1+1e-10))


def frame(raw, rec, steps, source_step, extra, error, success):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (512, 320), 'black')
    for x, key in [(0, 'agentview_image'), (256, 'robot0_eye_in_hand_image')]:
        canvas.paste(Image.fromarray(np.ascontiguousarray(raw[key][::-1, ::-1])), (x, 0))
    draw = ImageDraw.Draw(canvas)
    for n, text in enumerate([
        f'{rec["suite"]} task{rec["evaluation_task_id"]:02d} trial{rec["trial_id"]:02d} | E{rec["episode_index"]}',
        f'OSC step {steps} | original step {source_step} | compensation={extra}',
        f'goal error {error*1000:.2f}mm | LIBERO success={success}',
    ]):
        draw.text((5, 260+18*n), text, fill='white')
    return np.asarray(canvas)


def run_episode(rec, old_eid, output, settings=None):
    global SETTINGS
    if settings is not None:
        SETTINGS = settings
    eid = rec['episode_index']
    dest = output/f'episode_{eid:06d}'
    if (dest/'report.json').exists():
        report = json.loads((dest/'report.json').read_text())
        assert report['settings'] == SETTINGS and report['source_record'] == rec
        assert json.loads((dest/'trajectory/manifest.json').read_text())['complete']
        return report
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path.insert(0, str(HELPERS))
    import probe_conversion as probe
    probe.SOURCE = SOURCE/'sources/capx_recovery17'
    ep = rec['source_record']
    data = probe.source_cache(ep, output/'source_cache')
    spec = importlib.util.spec_from_file_location('matched_recorder', SOURCE/'read_trajectory.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    import imageio.v2 as imageio
    labels_path = OFFLINE/f'offline_labels/episode_{old_eid:06d}/labels_and_tcp.npz'
    with np.load(labels_path, allow_pickle=False) as f:
        label = {k:f[k].copy() for k in f.files}
    details = rows(labels_path.parent/'action_provenance.jsonl')
    handoff = ep['handoff_step']
    assert list(data['owners'][:handoff]) == ['pi05']*handoff
    assert list(data['owners'][handoff:]) == ['capx']*(len(data['actions'])-handoff)
    source_inputs = {str(p.relative_to(SOURCE)):sha(p) for p in
                     [SOURCE/'index.jsonl', SOURCE/rec['trajectory_path']/'manifest.json',
                      SOURCE/rec['trajectory_path']/'initial.npz', SOURCE/rec['trajectory_path']/'model.xml']}
    programs = dest/'programs'
    programs.mkdir()
    for p in (SOURCE/rec['path']/'capx').glob('turn_*.py'):
        shutil.copy2(p, programs/p.name)
        assert sha(p) == sha(programs/p.name)
        source_inputs[str(p.relative_to(SOURCE))] = sha(p)
    env = probe.environment(ep, images=True)
    recorder = writer = trace = None
    states, actions, source_steps, inserted, position_errors, rotation_errors = [], [], [], [], [], []
    success = done = False
    prefix_error = 0.
    split_targets, blocked = [], None
    goal_api_p = goal_api_r = 0.
    try:
        probe.restore_initial(env, data)
        for name in list(env.env._observables):
            if name.endswith('_depth'):
                env.env.modify_observable(name, 'enabled', False)
        before = env.env._get_observations()
        kin = Kinematics(env)
        fk_checks = verify_goals(kin, data, label, handoff)
        np.testing.assert_array_equal(env.sim.get_state().flatten(), data['sim/state'][0])
        controller = env.robots[0].controller
        np.testing.assert_array_equal(controller.output_max, [.05, .05, .05, .5, .5, .5])
        assert controller.use_delta and controller.use_ori and controller.position_limits is None
        recorder = reader.EpisodeRecorder(dest/'trajectory', dict(
            experiment='matched17_fixed_original_fk_goal_split', source_record=rec, settings=SETTINGS,
            suite=ep['suite'], task_id=ep['task_id'], trial_id=ep['trial_id'], instruction=ep['instruction'],
            controller='OSC_POSE', control_hz=20, wcm_reevaluated=False,
            source_images_reused=False, training_started=False), shard_steps=16)
        initial = env.sim.get_state().flatten().copy()
        recorder.begin(before, policy_observation={'states':state8(before)}, sim_state=initial)
        states.append(initial)
        writer = imageio.get_writer(str(dest/'execution.mp4'), fps=20, codec='libx264', quality=7, macro_block_size=16)
        trace = (dest/'action_provenance.jsonl').open('x')
        writer.append_data(frame(before, rec, 0, 0, False, 0., False))
        stop_reason = 'source_sequence_exhausted'
        for i in range(len(data['actions'])):
            owner = 'pi05' if i < handoff else 'capx_converted'
            p, r = kin.pose(states[-1])
            goal_p, goal_r = None, None
            if i < handoff:
                first = data['actions'][i, :7].astype(np.float32)
                np.testing.assert_array_equal(first.astype(np.float64), data['actions'][i, :7])
                split = False
            else:
                goal_p = label['target_position'][i]
                goal_r = label['target_rotation_recomputed'][i]
                first, split = command(p, r, goal_p, goal_r, data['actions'][i, 7])
            confirms = 0
            split_start = len(actions)
            budget = SETTINGS['max_controls_per_split_target'] if split else 1
            converged = False
            for j in range(budget):
                previous = states[-1]
                p, r = kin.pose(previous)
                if j == 0:
                    action, action_clipped = first, split
                else:
                    action, action_clipped = command(p, r, goal_p, goal_r, data['actions'][i, 7])
                assert action.shape == (7,) and np.isfinite(action).all()
                if i >= handoff:
                    assert np.max(np.abs(action)) <= 1
                after, reward, done, _ = env.step(action.astype(np.float64))
                # Preserve LIBERO's own post-step predicate timing. No sim.forward here.
                success = bool(env.check_success())
                state = env.sim.get_state().flatten().copy()
                assert np.isfinite(state).all() and abs(state[0]-previous[0]-.05) < 1e-8
                ap, ar = kin.pose(state)
                if i < handoff:
                    drift = float(np.max(np.abs(state-data['sim/state'][i+1])))
                    prefix_error = max(prefix_error, drift)
                    assert drift < 1e-8, f'VLA prefix diverged at {i+1}: {drift}'
                    err_p = err_r = 0.
                else:
                    err_p = float(np.linalg.norm(goal_p-ap))
                    err_r = float(Rotation.from_matrix(goal_r @ ar.T).magnitude())
                    # Check the actual official OSC goal, including orientation convention.
                    goal_api_p = max(goal_api_p, float(np.linalg.norm(controller.goal_pos-(p+action[:3].astype(float)*.05))))
                    decoded_r = Rotation.from_rotvec(action[3:6].astype(float)*.5).as_matrix() @ r
                    goal_api_r = max(goal_api_r, float(Rotation.from_matrix(controller.goal_ori @ decoded_r.T).magnitude()))
                    assert goal_api_p < 1e-7 and goal_api_r < 2e-6, (goal_api_p, goal_api_r)
                recorder.append(action, after, owner=owner, action_space='osc_pose',
                    reward=reward, success=success, terminated=done, truncated=False,
                    policy_observation={'states':state8(after)}, sim_state=state)
                detail = details[i]
                program_name = Path(detail['source_program']).name if detail['source_program'] else None
                item = dict(frame_index=len(actions), source_action_step=i+1, compensation_index=j,
                    inserted_compensation_action=j > 0, split_target=split,
                    complete_goal_encoded=not action_clipped if i >= handoff else None,
                    original_target_exceeded_range=bool(label['clipped'][i].any()), owner=owner,
                    original_action_space='osc_pose' if i < handoff else 'joint_position',
                    action_space='osc_pose', action=action.tolist(),
                    source_program=str(SOURCE/rec['path']/'capx'/program_name) if program_name else None,
                    copied_program='programs/'+program_name if program_name else None,
                    source_turn=detail['source_turn'], original_joint_action=data['actions'][i].tolist() if i >= handoff else None,
                    absolute_goal_position=goal_p.tolist() if goal_p is not None else None,
                    absolute_goal_quat_xyzw=Rotation.from_matrix(goal_r).as_quat().tolist() if i >= handoff else None,
                    actual_tcp_position_before=p.tolist(), actual_tcp_position_after=ap.tolist(),
                    goal_position_error_after_m=err_p, goal_rotation_error_after_rad=err_r,
                    success_after_action=success, reward=float(reward), terminated=bool(done),
                    observation_index=len(actions), next_observation_index=len(actions)+1,
                    wcm_reevaluated=False)
                trace.write(json.dumps(item, allow_nan=False)+'\n')
                states.append(state)
                actions.append(action)
                source_steps.append(i+1)
                inserted.append(j > 0)
                position_errors.append(err_p)
                rotation_errors.append(err_r)
                writer.append_data(frame(after, rec, len(actions), i+1, j > 0, err_p, success))
                before = after
                within = err_p <= SETTINGS['position_tolerance_m'] and err_r <= np.deg2rad(SETTINGS['rotation_tolerance_deg'])
                confirms = confirms+1 if within else 0
                converged = confirms >= SETTINGS['consecutive_confirmations']
                completed = not action_clipped if SETTINGS.get('completion_rule')=='goal_encoding' else converged
                if len(actions) % 100 == 0:
                    trace.flush()
                    print('SPLIT_PROGRESS '+json.dumps(dict(episode=eid, steps=len(actions), source_step=i+1,
                        source_total=len(data['actions']), split=split, compensation_index=j,
                        error_mm=err_p*1000, angle_deg=float(np.degrees(err_r)))), flush=True)
                if success or done or (split and completed):
                    break
            if split:
                result = dict(source_action_step=i+1, first_output_step=split_start+1,
                    last_output_step=len(actions), controls=len(actions)-split_start,
                    converged=converged, task_success=success, final_error_m=err_p,
                    complete_goal_encoded=not action_clipped, compensation_complete=completed,
                    final_angle_deg=float(np.degrees(err_r)), source_turn=detail['source_turn'])
                split_targets.append(result)
                recorder.event('split_target', **result)
            if success:
                stop_reason = 'libero_success'
                break
            if done:
                stop_reason = 'environment_terminated'
                break
            if split and not completed:
                blocked = result
                original_p, original_r = kin.pose(data['sim/state'][i+1])
                blocked['original_next_position_error_to_goal_m'] = float(np.linalg.norm(original_p-goal_p))
                blocked['original_next_rotation_error_to_goal_deg'] = float(np.degrees(Rotation.from_matrix(goal_r @ original_r.T).magnitude()))
                # Diagnose geometry in a separate data object, preserving the real
                # env's integration state, observations and success-predicate timing.
                kin.pose(states[-1])
                mujoco.mj_forward(kin.model, kin.data)
                contacts = []
                for c in kin.data.contact:
                    if c.dist > .002:
                        continue
                    names = [mujoco.mj_id2name(kin.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(int(g)) for g in (c.geom1,c.geom2)]
                    if any('robot0' in n or 'gripper0' in n for n in names):
                        contacts.append(dict(geom_names=names, distance_m=float(c.dist)))
                blocked['robot_contacts_at_stop'] = contacts
                stop_reason = 'fixed_original_target_not_reached'
                break
        recorder.finish(stop_reason, success=success)
        trace.close()
        trace = None
        writer.close()
        writer = None
        np.savez_compressed(dest/'states_actions.npz', sim_states=np.stack(states), actions=np.stack(actions),
            source_action_steps=source_steps, inserted_compensation_actions=inserted,
            goal_position_errors_m=position_errors, goal_rotation_errors_rad=rotation_errors)
        assert all(sha(SOURCE/p) == h for p,h in source_inputs.items()), 'Original source was modified'
        report = dict(state='complete', settings=SETTINGS, source_record=rec,
            episode_index=eid, original_offline_episode_index=old_eid,
            suite=ep['suite'], task_id=ep['task_id'], trial_id=ep['trial_id'], instruction=ep['instruction'],
            source_steps=ep['steps'], source_capx_steps=ep['steps']-handoff,
            executed_steps=len(actions), last_source_action_step=source_steps[-1],
            added_compensation_steps=int(sum(inserted)), split_targets=split_targets,
            success=success, accepted_for_training=success, stop_reason=stop_reason,
            blocked_target=blocked, prefix_steps=handoff, prefix_state_error_max=prefix_error,
            fk_checks=fk_checks, official_osc_goal_max_position_error_m=goal_api_p,
            official_osc_goal_max_rotation_error_rad=goal_api_r,
            source_input_sha256=source_inputs, labels_sha256=sha(labels_path),
            continuous_execution=True, intermediate_resets=0, original_source_unchanged=True,
            source_images_reused=False, wcm_reevaluated=False, training_started=False,
            elapsed_seconds=time.monotonic()-started)
        save_json(dest/'report.json', report)
        print('SPLIT_EPISODE_DONE '+json.dumps({k:v for k,v in report.items() if k not in
            ('source_record','source_input_sha256','settings','split_targets')}), flush=True)
        return report
    except BaseException as exc:
        if recorder is not None:
            recorder.finish('technical_error', success=False, error=repr(exc))
        save_json(dest/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if writer is not None:
            writer.close()
        if trace is not None:
            trace.close()
        env.close()


def main():
    global SETTINGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=OUTPUT)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--episodes', type=int, nargs='*')
    ap.add_argument('--completion', choices=['convergence','goal_encoding'],default='convergence')
    args = ap.parse_args()
    if args.completion=='goal_encoding':
        if args.output==OUTPUT:
            args.output = OUTPUT.with_name(OUTPUT.name+'_goal_encoding')
        SETTINGS = dict(SETTINGS,version=2,completion_rule='goal_encoding',
            split_completion='the final bounded action encodes the full original absolute target without saturation; physical convergence is recorded separately',
            blocked_target='stop and reject if the target still cannot be encoded within 200 controls')
    output = args.output.resolve()
    assert output != SOURCE and SOURCE not in output.parents and output != OFFLINE
    output.mkdir(parents=True, exist_ok=True)
    records = [r for r in rows(SOURCE/'index.jsonl') if r['source']=='capx_recovery_success']
    assert len(records) == 17 and [r['episode_index'] for r in records] == list(range(57,74))
    old = [r for r in rows(OFFLINE/'meta/provenance.jsonl') if r['source']=='capx_recovery_success']
    oldmap = {(r['suite'],r['evaluation_task_id'],r['source_record']['trial_id']):r['exported_episode_index'] for r in old}
    assert len(oldmap) == 17
    if args.episodes:
        assert set(args.episodes) <= {r['episode_index'] for r in records}
        records = [r for r in records if r['episode_index'] in args.episodes]
    plan = dict(settings=SETTINGS, source=str(SOURCE), output=str(output), total=17,
        source_index_sha256=sha(SOURCE/'index.jsonl'), training_started=False,
        selected_episode_indices=[r['episode_index'] for r in records])
    if (output/'plan.json').exists():
        previous = json.loads((output/'plan.json').read_text())
        assert previous['settings'] == SETTINGS and previous['source_index_sha256'] == plan['source_index_sha256']
    save_json(output/'plan.json', plan)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_episode, rec, oldmap[(rec['suite'],rec['evaluation_task_id'],rec['trial_id'])], output, SETTINGS):rec for rec in records}
        for future in as_completed(pending):
            rec = pending[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(dict(episode_index=rec['episode_index'], error=repr(exc), traceback=traceback.format_exc()))
                print('SPLIT_TECHNICAL_ERROR '+json.dumps(errors[-1]), flush=True)
            save_json(output/'status.json', dict(state='running', selected=len(records), completed=len(results),
                successful=sum(r['success'] for r in results), failed=sum(not r['success'] for r in results),
                technical_errors=errors, training_started=False))
    results.sort(key=lambda r:r['episode_index'])
    summary = dict(state='complete' if not errors else 'technical_error', selected=len(records),
        requested_recoveries=17, completed=len(results), successful=sum(r['success'] for r in results),
        failed=sum(not r['success'] for r in results), added_compensation_steps=sum(r['added_compensation_steps'] for r in results),
        full17_validated=len(results)==17 and all(r['success'] for r in results),
        technical_errors=errors, training_started=False, results=results)
    save_json(output/'summary.json', summary)
    save_json(output/'status.json', {k:v for k,v in summary.items() if k!='results'})
    print('SPLIT_ALL_DONE '+json.dumps({k:v for k,v in summary.items() if k!='results'}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
