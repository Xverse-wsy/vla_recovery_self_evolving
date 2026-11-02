"""Read-only source audit and native/OSC replay experiments for CAP-X recordings."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
SOURCE = ROOT / 'dataset_subsets/pi05_capx_mixed_success34_20261021'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_cache(ep, cache):
    name = f'{ep["suite"]}_task{ep["task_id"]:02d}_trial{ep["trial_id"]:02d}'
    out = cache / (name + '.npz')
    if out.exists():
        with np.load(out, allow_pickle=False) as d:
            return {k: d[k].copy() for k in d.files}
    p = SOURCE / ep['trajectory_path']
    meta = json.loads((p / 'manifest.json').read_text())
    assert meta['complete'] and meta['success'] and ep['used_capx']
    keys = ['sim/state', 'raw/robot0_joint_pos', 'raw/robot0_joint_vel',
            'raw/robot0_eef_pos', 'raw/robot0_eef_quat', 'raw/robot0_gripper_qpos']
    parts = {k: [] for k in keys}
    with np.load(p / 'initial.npz', allow_pickle=False) as d:
        for k in keys:
            parts[k].append(d[k][None].copy())
    acts, owners = [], []
    for shard in meta['shards']:
        f = p / shard['file']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == shard['sha256']
        with np.load(f, allow_pickle=False) as d:
            for k in keys:
                parts[k].append(d[k].copy())
            acts.append(d['transition/action'].copy())
            owners.append(d['transition/owner'].copy())
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    arrays['actions'] = np.concatenate(acts)
    arrays['owners'] = np.concatenate(owners)
    assert len(arrays['actions']) == ep['steps']
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return arrays


def environment(ep, images=False):
    import sys
    frozen = ROOT/'wcm_pi05_monitor/outputs/hybrid_independent_gpu1_20261019_172047/runtime'
    sys.path.insert(0,str(frozen))
    from hybrid.environment import SharedEnvironment
    np.random.seed(ep['seed'])
    random.seed(ep['seed'])
    shared = SharedEnvironment(ep['suite'],ep['task_id'],ep['trial_id'],ep['seed'],
                               limit=10000,capx_limit=10000)
    env = shared.env
    env._bootstrap_shared = shared
    c = env.robots[0].controller
    env._bootstrap = dict(state=env.sim.get_state().flatten().copy(),
                          goal_pos=c.goal_pos.copy(),goal_ori=c.goal_ori.copy(),
                          initial_joint=c.initial_joint.copy(),
                          gripper=env.robots[0].gripper.current_action.copy(),
                          physics={k:getattr(env.sim.data,k).copy() for k in
                                   ['ctrl','qacc_warmstart','qfrc_applied','xfrc_applied']})
    if not images:
        for name in list(env.env._observables):
            if name.endswith('_image') or name.endswith('_depth'):
                env.env.modify_observable(name,'enabled',False)
    return env


def set_controller(env, kind):
    from robosuite import load_controller_config
    robot = env.robots[0]
    robot.controller_config = load_controller_config(default_controller=kind)
    robot._load_controller()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    env.env._action_dim = sum(r.action_dim for r in env.robots)


def restore_initial(env, data):
    boot = env._bootstrap
    np.testing.assert_array_equal(boot['state'],data['sim/state'][0],
                                  err_msg='Original environment bootstrap differs from recorded state')
    env.sim.set_state_from_flattened(data['sim/state'][0])
    for k,v in boot['physics'].items():
        getattr(env.sim.data,k)[:] = v
    env.sim.forward()
    env.env.timestep = 15
    env.env.done = False
    env.env.cur_time = 15 / 20
    robot = env.robots[0]
    robot.controller.update(force=True)
    robot.controller.goal_pos = boot['goal_pos'].copy()
    robot.controller.goal_ori = boot['goal_ori'].copy()
    robot.controller.initial_joint = boot['initial_joint'].copy()
    robot.gripper.current_action = boot['gripper'].copy()


def torque_action(env, target, native_action):
    """Project joint impedance torque onto the six controllable OSC dimensions."""
    c = env.robots[0].controller
    c.update(force=True)
    pos, ori = c.ee_pos.copy(), c.ee_ori_mat.copy()
    c.goal_pos, c.goal_ori = pos.copy(), ori.copy()
    base = c.run_controller().copy()
    matrix = np.empty((7, 6))
    for k in range(6):
        c.goal_pos, c.goal_ori = pos.copy(), ori.copy()
        if k < 3:
            c.goal_pos[k] += .001
            matrix[:, k] = (c.run_controller().copy() - base) / .001
        else:
            v = np.zeros(3)
            v[k - 3] = .001
            c.goal_ori = Rotation.from_rotvec(v).as_matrix() @ ori
            matrix[:, k] = (c.run_controller().copy() - base) / np.sin(.001)
    desired = 50. * (target - c.joint_pos) - 2 * np.sqrt(50.) * c.joint_vel + c.torque_compensation
    errors = np.linalg.lstsq(matrix, desired - base, rcond=1e-6)[0]
    delta = errors.copy()
    norm = np.linalg.norm(errors[3:])
    if norm > 1e-10:
        delta[3:] *= np.arcsin(min(norm, .999)) / norm
    return np.r_[np.clip(delta / np.r_[[.05] * 3, [.5] * 3], -1, 1), native_action[-1]]


def tracking_action(env, data, index, lead=.15):
    c = env.robots[0].controller
    c.update(force=True)
    j = min(index + 1, len(data['actions']))
    p = data['raw/robot0_eef_pos']
    q = data['raw/robot0_eef_quat']
    vel = (p[j] - p[index]) * 20
    goal = p[j] + lead * vel
    r0, r1 = Rotation.from_quat(q[index]), Rotation.from_quat(q[j])
    omega = (r1 * r0.inv()).as_rotvec() * 20
    goal_r = Rotation.from_rotvec(lead * omega).as_matrix() @ r1.as_matrix() @ env._body_to_control_site
    delta_r = Rotation.from_matrix(goal_r @ c.ee_ori_mat.T).as_rotvec()
    # set_goal keeps the old orientation for exactly-zero rotation commands.
    if np.linalg.norm(delta_r) < 1e-12:
        delta_r[0] = 1e-10
    return np.r_[np.clip((goal - c.ee_pos) / .05, -1, 1),
                 np.clip(delta_r / .5, -1, 1), data['actions'][index, 7]]


def exact_reference(env, data):
    """Use saved MuJoCo state, avoiding the sensor's one-substep sampling lag."""
    sid = env.robots[0].eef_site_id
    ps, rs = [], []
    for state in data['sim/state']:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        ps.append(env.sim.data.site_xpos[sid].copy())
        rs.append(env.sim.data.site_xmat[sid].reshape(3, 3).copy())
    restore_initial(env, data)
    data['reference_pos'], data['reference_rot'] = np.stack(ps), np.stack(rs)


def endpoint_action(env, data, index):
    c = env.robots[0].controller
    c.update(force=True)
    # Semi-implicit Euler response of the configured critically damped OSC.
    # Used as an initial inverse model; full-simulation replay decides validity.
    dt = env.env.model_timestep
    n = int(env.env.control_timestep / dt)
    goal_response, goal_velocity, velocity_response, velocity_velocity = 0., 0., 0., 1.
    for _ in range(n):
        goal_velocity += dt * (150. * (1. - goal_response) - 2 * np.sqrt(150.) * goal_velocity)
        goal_response += dt * goal_velocity
        velocity_velocity += dt * (-150. * velocity_response - 2 * np.sqrt(150.) * velocity_velocity)
        velocity_response += dt * velocity_velocity
    goal = c.ee_pos + (data['reference_pos'][index + 1] - c.ee_pos - velocity_response * c.ee_pos_vel) / goal_response
    dr = Rotation.from_matrix(data['reference_rot'][index + 1] @ c.ee_ori_mat.T).as_rotvec()
    goal_dr = (dr - velocity_response * c.ee_ori_vel) / goal_response
    if np.linalg.norm(goal_dr) < 1e-12:
        goal_dr[0] = 1e-10
    return np.r_[np.clip((goal-c.ee_pos)/.05,-1,1),np.clip(goal_dr/.5,-1,1),data['actions'][index,7]]


def numeric_action(env, data, index):
    """Fit a standard 7D OSC action using reversible low-level simulation probes.

    Predictions never call env.step, its reward hooks, or mutate object/task
    state. All physics state, warm starts, actuator controls and gripper state
    are restored before the one authoritative action is executed.
    """
    sim, robot = env.sim, env.robots[0]
    c = robot.controller
    state = sim.get_state().flatten().copy()
    snapshots = {k: getattr(sim.data, k).copy() for k in
                 ['ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied']}
    grip = robot.gripper.current_action.copy()
    old_pos, old_ori = c.goal_pos.copy(), c.goal_ori.copy()
    target_p, target_r = data['reference_pos'][index + 1], data['reference_rot'][index + 1]
    n = int(env.env.control_timestep / env.env.model_timestep)
    sid = robot.eef_site_id

    def restore():
        sim.set_state_from_flattened(state)
        for k, v in snapshots.items():
            getattr(sim.data, k)[:] = v
        sim.forward()
        robot.gripper.current_action = grip.copy()
        c.goal_pos, c.goal_ori = old_pos.copy(), old_ori.copy()
        c.update(force=True)
        np.testing.assert_array_equal(sim.get_state().flatten(), state)

    def residual(action):
        restore()
        for j in range(n):
            sim.forward()
            if j == 0:
                c.set_goal(action[:6])
            torques = c.run_controller()
            low, high = robot.torque_limits
            robot.grip_action(gripper=robot.gripper, gripper_action=action[6:])
            sim.data.ctrl[robot._ref_joint_actuator_indexes] = np.clip(torques, low, high)
            sim.step()
        sim.forward()
        p = sim.data.site_xpos[sid]
        r = sim.data.site_xmat[sid].reshape(3, 3)
        return np.r_[p-target_p, .08 * Rotation.from_matrix(r @ target_r.T).as_rotvec()]

    try:
        action = endpoint_action(env, data, index)
        best_r = residual(action)
        for iteration in range(2):
            if np.linalg.norm(best_r[:3]) < .0001 and np.linalg.norm(best_r[3:]) < .00008:
                break
            jac = np.empty((6, 6))
            for k in range(6):
                a = action.copy()
                d = -.01 if a[k] > .98 else .01
                a[k] += d
                jac[:, k] = (residual(a)-best_r)/d
            update = np.linalg.lstsq(np.vstack([jac, np.eye(6)*.0001]),
                                      np.r_[-best_r, np.zeros(6)],rcond=1e-6)[0]
            candidate = action.copy()
            candidate[:6] = np.clip(action[:6]+update, -1, 1)
            new_r = residual(candidate)
            if np.linalg.norm(new_r) >= np.linalg.norm(best_r):
                break
            action, best_r = candidate, new_r
        return action
    finally:
        restore()


def retime(data, handoff, factor):
    """Slow only the recovery; keep policy prefix and source-step provenance."""
    if factor == 1:
        return data
    result = {}
    count = len(data['actions'])
    times = np.r_[np.arange(handoff+1),
                  handoff + np.arange(1,(count-handoff)*factor+1)/factor]
    for key, value in data.items():
        if key in ('actions','owners'):
            result[key] = np.concatenate([value[:handoff],np.repeat(value[handoff:],factor,axis=0)])
        elif key in ('raw/robot0_eef_quat','reference_rot'):
            rot = Rotation.from_quat(value) if key.endswith('quat') else Rotation.from_matrix(value)
            interpolated = Slerp(np.arange(len(value)),rot)(times)
            result[key] = interpolated.as_quat() if key.endswith('quat') else interpolated.as_matrix()
        elif len(value) == count+1:
            flat = value.reshape(len(value),-1)
            result[key] = np.stack([np.interp(times,np.arange(len(value)),flat[:,k])
                                    for k in range(flat.shape[1])],axis=1).reshape((len(times),)+value.shape[1:])
        else:
            result[key] = value
    result['source_step'] = np.r_[np.arange(1,handoff+1),
                                 np.repeat(np.arange(handoff+1,count+1),factor)]
    return result


def replay(ep, data, mode, out, lead=.15, tail=0, speed_factor=1):
    started = time.monotonic()
    env = environment(ep)
    restore_initial(env, data)
    if mode in ('endpoint', 'numeric'):
        exact_reference(env, data)
    data = retime(data, ep['handoff_step'], speed_factor)
    c = env.robots[0].controller
    c.update(force=True)
    # Raw eef_quat is the hand BODY orientation, whereas OSC controls grip SITE.
    # eef_pos is already the grip-site position. Derive the fixed rotation from
    # the restored XML/state instead of mixing these two frame conventions.
    env._body_to_control_site = Rotation.from_quat(data['raw/robot0_eef_quat'][0]).as_matrix().T @ c.ee_ori_mat
    print('BODY_TO_CONTROL_SITE ' + json.dumps(env._body_to_control_site.tolist()), flush=True)
    errors, rotations, state_errors, actions = [], [], [], []
    successes = []
    handoff = ep['handoff_step']
    try:
        for i, native in enumerate(data['actions']):
            if i == handoff and mode == 'native':
                set_controller(env, 'JOINT_POSITION')
            if i < handoff:
                action = native[:7].copy()
            elif mode == 'native':
                action = native.copy()
            elif mode == 'torque':
                target = data['raw/robot0_joint_pos'][i] + .05 * np.clip(native[:7], -1, 1)
                action = torque_action(env, target, native)
            elif mode == 'track':
                action = tracking_action(env, data, i, lead)
            elif mode == 'endpoint':
                action = endpoint_action(env, data, i)
            elif mode == 'numeric':
                action = numeric_action(env, data, i)
            else:
                raise ValueError(mode)
            obs, _, _, _ = env.step(action)
            actions.append(action)
            success = bool(env.check_success())
            errors.append(float(np.linalg.norm(obs['robot0_eef_pos'] - data['raw/robot0_eef_pos'][i + 1])))
            rotations.append(float((Rotation.from_quat(obs['robot0_eef_quat']) *
                                    Rotation.from_quat(data['raw/robot0_eef_quat'][i + 1]).inv()).magnitude()))
            state_errors.append(float(np.max(np.abs(env.sim.get_state().flatten()[1:] - data['sim/state'][i + 1, 1:]))))
            if i < handoff:
                assert state_errors[-1] < 1e-9, f'Unconverted policy prefix drift at step {i+1}: {state_errors[-1]}'
            if success:
                successes.append(i + 1)
            if (i + 1) % 200 == 0:
                print(json.dumps(dict(event='replay_progress', suite=ep['suite'], task=ep['task_id'],
                                      mode=mode, step=i + 1, total=len(data['actions']),
                                      eef_error=errors[-1], success=success)), flush=True)
        tail_success = None
        if mode in ('track','endpoint','numeric') and tail:
            for k in range(tail):
                c.update(force=True)
                goal = data['raw/robot0_eef_pos'][-1]
                goal_r = Rotation.from_quat(data['raw/robot0_eef_quat'][-1]).as_matrix() @ env._body_to_control_site
                dr = Rotation.from_matrix(goal_r @ c.ee_ori_mat.T).as_rotvec()
                if np.linalg.norm(dr) < 1e-12:
                    dr[0] = 1e-10
                action = np.r_[np.clip((goal-c.ee_pos)/.05,-1,1),np.clip(dr/.5,-1,1),data['actions'][-1,7]]
                env.step(action)
                if env.check_success():
                    tail_success = k + 1
                    break
        result = dict(suite=ep['suite'], task_id=ep['task_id'], trial_id=ep['trial_id'], mode=mode, lead=lead,
                      replay_version=2,initialization='original_bootstrap_preserve_controller_goals',
                      speed_factor=speed_factor,
                      success=bool(successes), success_steps=successes, steps=len(actions),
                      tail_success=tail_success,
                      position_error_mean=float(np.mean(errors)), position_error_max=max(errors),
                      position_error_p95=float(np.quantile(errors, .95)),
                      rotation_error_p95=float(np.quantile(rotations, .95)),
                      state_error_max=max(state_errors), prefix_state_error_max=max(state_errors[:handoff]),
                      elapsed_seconds=time.monotonic() - started,
                      final_sim_state=env.sim.get_state().flatten().tolist(),
                      original_final_sim_state=data['sim/state'][-1].tolist())
        name = f'{ep["suite"]}_task{ep["task_id"]:02d}_{mode}_lead{lead}_speed{speed_factor}'
        save_json(out / (name + '.json'), result)
        np.savez_compressed(out / (name + '_diagnostics.npz'), eef_errors=errors,
                            rotation_errors=rotations, state_errors=state_errors)
        if mode != 'native':
            np.savez_compressed(out / (name + '_actions.npz'), actions=np.stack(actions))
        print('RESULT ' + json.dumps(result), flush=True)
        return result
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--suite', default='libero_goal')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--mode', choices=['native', 'track', 'torque', 'endpoint', 'numeric'], default='native')
    ap.add_argument('--lead', type=float, default=.15)
    ap.add_argument('--tail', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--speed-factor',type=int,default=1)
    ap.add_argument('--failed-only',action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((SOURCE / 'manifest.json').read_text())['episodes']
    episodes = [e for e in records if e['used_capx'] and
                (args.all or (e['suite'] == args.suite and e['task_id'] == args.task))]
    if args.failed_only:
        unresolved=json.loads((Path(__file__).resolve().parent/'conversion_plan.json').read_text())['unresolved']
        keys={(r['episode']['suite'],r['episode']['task_id']) for r in unresolved}
        episodes=[e for e in episodes if (e['suite'],e['task_id']) in keys]
    results = []
    for ep in episodes:
        data = source_cache(ep, args.output / 'source_cache')
        results.append(replay(ep, data, args.mode, args.output, args.lead, args.tail,args.speed_factor))
        save_json(args.output / ('summary_' + args.mode + '.json'),
                  dict(completed=len(results), total=len(episodes),
                       successes=sum(r['success'] for r in results), results=results))


if __name__ == '__main__':
    main()
