"""Import the explicitly selected original demonstrations for FM training only.

Do not feed these records to any VLA matching bank. Sparse episode IDs, raw
pre-action RGB/state and action ordering are retained; no replay/resampling occurs.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import QuantileAssets, action_spec, quaternion_from_rotvec, read_jsonl, write_jsonl
from .dense_fm import DENSE_POLICY
from .features import sha256
from .schema import Window, stable_id


def expert_windows(dataset, config, norm_stats, output, *, expected_episodes=40):
    from .positive_fm import transform_window
    root, out = Path(dataset).resolve(), Path(output).resolve()
    if out == root or root in out.parents:
        raise ValueError('Expert output must be outside original dataset')
    metadata = [root/'episodes.json', root/'meta/info.json', root/'meta/tasks.jsonl']
    metadata += [root/p for p in ('manifest.json','selection.json') if (root/p).is_file()]
    hashes = {str(p):sha256(p) for p in metadata}
    ids = json.loads((root/'episodes.json').read_text())
    if (not isinstance(ids,list) or any(type(x) is not int or x<0 for x in ids) or
            len(ids)!=expected_episodes or len(set(ids))!=len(ids)):
        raise ValueError('Expert episodes.json must explicitly select unique expected episode IDs')
    info = json.loads((root/'meta/info.json').read_text())
    tasks = {int(t['task_index']):t['task'] for t in read_jsonl(root/'meta/tasks.jsonl')}
    assets = QuantileAssets(norm_stats)
    spec = action_spec(assets,config)
    recorded_hz = float(info['fps'])
    if not np.isfinite(recorded_hz) or recorded_hz<=0:
        raise ValueError('Invalid original expert recorded fps')
    source = out/'expert_sources'
    (source/'actions').mkdir(parents=True,exist_ok=False)
    manifest, numerical, seen_tasks = [], {}, set()
    for eid in ids:
        path = root/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid)
        table = pq.read_table(path,columns=['state','actions','task_index','episode_index','frame_index'])
        n = len(table)
        states = np.asarray(table['state'].to_pylist(),np.float32)
        actions = np.asarray(table['actions'].to_pylist(),np.float32)
        tids = np.asarray(table['task_index']).reshape(-1)
        if (not n or states.shape!=(n,8) or actions.shape!=(n,7) or
                not np.isfinite(states).all() or not np.isfinite(actions).all() or
                not np.all(np.asarray(table['episode_index'])==eid) or
                not np.array_equal(np.asarray(table['frame_index']),np.arange(n)) or len(set(tids.tolist()))!=1):
            raise ValueError('Original expert pre-action state/action/episode/frame layout mismatch')
        task_index = int(tids[0])
        if task_index in seen_tasks or task_index not in tasks:
            raise ValueError('Expected one original expert demonstration per task')
        seen_tasks.add(task_index)
        segment = stable_id('original_expert',str(root),eid)
        manifest.append(dict(segment_id=segment,parquet=str(path),length=n,parent_start_step=0,
            controller='expert',episode_id=f'expert:{root}:episode{eid}',task_id=f'expert_task:{task_index}',
            instruction=tasks[task_index],recorded_hz=recorded_hz,source_episode_index=eid))
        numerical[segment] = (states,assets.normalize('actions',actions))
    manifest_path = source/'input_manifest.jsonl'
    write_jsonl(manifest_path,manifest)
    manifest_sha = sha256(manifest_path)
    scopes=[]
    for e in manifest:
        segment,n = e['segment_id'],e['length']
        states,normalized = numerical[segment]
        anchor_id = stable_id(segment,'source_anchor',spec.identity)
        anchor_path = source/'actions'/(anchor_id+'.npz')
        anchor_actions = np.zeros((spec.horizon,spec.model_dim),np.float32)
        anchor_mask = np.zeros_like(anchor_actions,bool)
        real = min(n,spec.horizon)
        anchor_actions[:real,:7],anchor_mask[:real,:7] = normalized[:real],True
        np.savez_compressed(anchor_path,actions=anchor_actions,valid_mask=anchor_mask,
                            trusted_error_mask=np.zeros_like(anchor_mask))
        anchor_sha = sha256(anchor_path)
        anchor = None
        for row in range(n):
            length = min(spec.horizon,n-row)
            raw_actions = np.zeros_like(anchor_actions)
            valid = np.zeros_like(anchor_mask)
            raw_actions[:length,:7],valid[:length,:7] = normalized[row:row+length],True
            record = Window(
                sample_id=stable_id(segment,row,spec.identity,DENSE_POLICY),task_id=e['task_id'],
                episode_id=e['episode_id'],initial_group_id='expert_initial_unknown:'+segment,event_id=segment,
                controller='expert',local_quality='unknown',start_step=row,start_time_s=row/recorded_hz,
                instruction=e['instruction'],phase=None,target_object=None,
                image_refs={v:dict(parquet=e['parquet'],column=c,row=row)
                            for v,c in [('front','image'),('wrist','wrist_image')]},
                proprio=states[row].tolist(),standardized_proprio=assets.normalize('state',states[row]).tolist(),
                eef_position_m=states[row,:3].tolist(),eef_quaternion_xyzw=quaternion_from_rotvec(states[row,3:6]),
                gripper_state=None,held_object=None,contact_state=None,progress=None,
                action_path=str(anchor_path),spec=spec,split='train',uncertain_initialization=True,
                bank_class='D_expert_positive',bank_label=dict(source_category='original_expert_demonstration',
                    basis='original_selected_demonstration',source_episode_index=e['source_episode_index']),
                source=dict(dataset=str(root),segment_id=segment,segment_row=row,action_range=[row,row+length],
                    normalization_asset=assets.path,expert_training_only=True,recorded_hz=recorded_hz,
                    expert_source_metadata_sha256=hashes,source_episode_index=e['source_episode_index'],
                    fm_start_stride=1,fm_sampling_policy=DENSE_POLICY,fm_scope_range=[0,n],fm_label_boundary=[0,n],
                    fm_source_manifest=str(manifest_path),fm_source_manifest_sha256=manifest_sha,
                    fm_source_anchor_id=anchor_id,fm_source_anchor_action_sha256=anchor_sha,
                    matching_performed=False,image_rotation_applied=False,resampling_applied=False))
            record.validate()
            if anchor is None:
                anchor = Window.from_dict(dict(record.to_dict(),sample_id=anchor_id))
            yield transform_window(record,out,payload=dict(actions=raw_actions,valid_mask=valid,
                trusted_error_mask=np.zeros_like(valid)),original=anchor)
        scopes.append(dict(segment_id=segment,episode_id=e['episode_id'],controller='expert',
            source_category='original_expert_demonstration',scope_range=[0,n],label_boundary=[0,n],
            anchor_id=anchor_id,tail_mode='repeat_last',eligible_starts=n,excluded_short_tail_starts=0))
    if any(sha256(p)!=h for p,h in hashes.items()):
        raise ValueError('Original expert metadata changed during import')
    write_jsonl(out/'expert_scopes.jsonl',scopes)
