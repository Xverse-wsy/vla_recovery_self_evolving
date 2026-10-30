"""Read-only import of existing segmented data; explicit local reviews enrich labels.

WCM warnings and program completion flags remain weak annotations. No rollout,
replay, interpolation, future observation, or new action is produced here.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .features import sha256
from .schema import ActionSpec, Window, stable_id, save_windows
from .splits import audit_split
from .libraries import selection_labels, split_manifest, write_window_libraries


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in rows))


class QuantileAssets:
    """OpenPI quantile normalization: no clipping, no added delta transform."""
    def __init__(self, path):
        self.path, self.sha = str(Path(path).resolve()), sha256(path)
        root = json.loads(Path(path).read_text())
        self.stats = root.get('norm_stats', root)
        for key in ('actions', 'state'):
            for quantile in ('q01', 'q99'):
                if quantile not in self.stats[key]:
                    raise ValueError('pi05 quantile assets must contain q01 and q99')

    def normalize(self, key, values):
        x = np.asarray(values, dtype=np.float32)
        a = np.asarray(self.stats[key]['q01'], np.float32)[:x.shape[-1]]
        b = np.asarray(self.stats[key]['q99'], np.float32)[:x.shape[-1]]
        if a.shape[-1] != x.shape[-1] or not np.isfinite(x).all():
            raise ValueError('Normalization dimension/finite-value mismatch')
        return (x - a) / (b - a + 1e-6) * 2 - 1


def action_spec(assets, config):
    if config.environment_action_dim != 7:
        raise ValueError('This import adapter supports seven-dimensional LIBERO OSC only')
    pad = config.action_dim - 7
    return ActionSpec('libero_osc_normalized_command_quantile',
        tuple(['dx', 'dy', 'dz', 'dRx', 'dRy', 'dRz', 'gripper'] + [f'pad{i}' for i in range(pad)]),
        tuple(['normalized_control'] * 7 + ['padding'] * pad), 'robot_base',
        config.native_control_hz, assets.sha, config.horizon, config.action_dim,
        tuple([True] * 7 + [False] * pad)).validate()


def quaternion_from_rotvec(v):
    v = np.asarray(v, float)
    angle = np.linalg.norm(v)
    return np.r_[v * (np.sin(angle / 2) / angle if angle > 1e-12 else .5), np.cos(angle / 2)].tolist()


def manifest_segmented(dataset):
    """Reuses existing program/WCM boundaries; parent episode identity is preserved."""
    root = Path(dataset).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    out = []
    provenance=read_jsonl(root / 'meta/provenance.jsonl')
    labels=selection_labels(root,provenance)
    for p in provenance:
        eid = int(p['episode_index'])
        path = root / info['data_path'].format(episode_chunk=eid // info['chunks_size'], episode_index=eid)
        # Evaluation slots may be shuffled; match by canonical LIBERO task ID.
        task = f"{p['suite']}/task{int(p['canonical_task_index']):02d}"
        parent = p['original_episode']
        out.append(dict(segment_id=stable_id(str(root), eid), parquet=str(path),
            task_id=task, episode_id=parent,
            initial_group_id=f"libero_default_init:{task}:trial{p['trial_id']}",
            initialization_verified=False,  # set true only after checking actual init-state provenance
            event_id=stable_id(parent, p['category'], p.get('turn')), controller='VLA' if p['owner'] == 'pi05' else 'code',
            instruction=p['instruction'], parent_start_step=int(p['start']),
            length=int(p['end'] - p['start']), state_layout='eef_xyz_axisangle_gripper2',
            handoff_start=p['owner'] != 'pi05' and int(p['start']) == int(pq_prefix(root, p)),
            **labels.get(eid,dict(bank_class=None,bank_label={'source_category':'vla_boundary_uncertain'})),
            annotations=[], start_context={}, source=dict(dataset=str(root), provenance=p,
                weak_category=p['category'], weak_completion=p.get('completion_label'),
                action_mapping=str(root / p['action_mapping']))))
    return out


def manifest_autonomous(dataset):
    """Import the saved autonomous20 schema without upgrading episode labels."""
    root=Path(dataset).resolve()
    info=json.loads((root/'meta/info.json').read_text())
    out=[]
    for p in read_jsonl(root/'meta/provenance.jsonl'):
        rec=p['source_record']
        if rec.get('used_capx') or not rec.get('success') or rec.get('action_dim') != 7:
            raise ValueError('Expected verified autonomous seven-dimensional success source')
        eid=int(p['exported_episode_index'])
        path=root/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid)
        n=pq.ParquetFile(path).metadata.num_rows
        task=f"{p['suite']}/task{int(p['canonical_task_index']):02d}"
        episode=rec.get('original_episode')
        if not episode:
            raise ValueError('Autonomous data must preserve original full episode identity')
        out.append(dict(segment_id=stable_id(str(root),eid),parquet=str(path),length=n,
            task_id=task,episode_id=episode,
            initial_group_id=f"libero_default_init:{task}:trial{rec['trial_id']}",initialization_verified=False,
            event_id=stable_id(episode,'autonomous_rollout'),controller='VLA',instruction=p['instruction'],
            parent_start_step=0,state_layout='eef_xyz_axisangle_gripper2',handoff_start=False,
            bank_class='D_vla_positive',bank_label=dict(source_category='autonomous_episode_success',
                label_scope='existing_episode',basis='autonomous_LIBERO_success',
                provenance_path=str(root/'meta/provenance.jsonl'),provenance_sha256=sha256(root/'meta/provenance.jsonl'),
                source_episode_index=eid,local_actions_verified=False),
            annotations=[],start_context={},source=dict(dataset=str(root),provenance=p,
                parent_success=True,weak_category='autonomous_episode_success')))
    return out


def pq_prefix(root, p):
    # Existing parent summary is copied alongside each segment's action mapping.
    summary = Path(root) / Path(p['action_mapping']).parent / 'segments.json'
    return json.loads(summary.read_text())['prefix_steps']


def build_windows(manifest, assets, config, output):
    """Manifest ranges are segment-local [start,end); quality boundaries cut windows."""
    config.validate()
    if not manifest:
        raise ValueError('Cannot build windows from an empty segment library')
    assignments={entry.get('split','unassigned')!='unassigned' for entry in manifest}
    if assignments=={False}:
        # Even custom manifests are grouped BEFORE any action window is built.
        manifest=split_manifest(manifest,config)
    elif assignments!={True}:
        raise ValueError('Manifest mixes preassigned and unassigned splits')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'actions').mkdir()
    spec, records, review = action_spec(assets, config), [], []
    for entry in manifest:
        if entry['controller'] not in ('VLA', 'code'):
            raise ValueError('Expert demonstrations require a separate role, not fake VLA ownership')
        if entry.get('state_layout') != 'eef_xyz_axisangle_gripper2':
            raise ValueError('Verify LIBERO eight-dimensional state layout explicitly')
        path = str(Path(entry['parquet']).resolve())
        table = pq.read_table(path, columns=['state', 'actions'])
        state = np.asarray(table['state'].to_pylist(), np.float32)
        raw = np.asarray(table['actions'].to_pylist(), np.float32)
        n = len(raw)
        if n != entry['length'] or state.shape != (n, 8) or raw.shape != (n, 7):
            raise ValueError('Recorded length/state/action layout mismatch')
        annotations = sorted(entry.get('annotations', []), key=lambda a: a['start'])
        labels = ['unknown'] * n
        evidence = [{} for _ in range(n)]
        event = [entry['event_id']] * n
        covered = np.zeros(n, bool)
        cuts = {0, n}
        for a in annotations:
            lo, hi, quality = int(a['start']), int(a['end']), a['local_quality']
            if not 0 <= lo < hi <= n or covered[lo:hi].any() or quality not in ('positive', 'negative', 'unknown'):
                raise ValueError('Invalid/overlapping local action review')
            covered[lo:hi] = True
            cuts.update([lo, hi])
            for j in range(lo, hi):
                labels[j], evidence[j], event[j] = quality, a.get('quality_evidence', {}), a.get('event_id', entry['event_id'])
        normalized = assets.normalize('actions', raw)
        boundaries = sorted(cuts)
        for lo, hi in zip(boundaries, boundaries[1:]):
            # Rare verified negative windows dense; other states every ~10 controls.
            stride = 1 if labels[lo] == 'negative' else config.state_stride
            starts = sorted(set(range(lo, hi, stride)) | {lo} |
                {int(x) for x in entry.get('event_starts', []) if lo <= int(x) < hi})
            for start in starts:
                length = min(config.horizon, hi - start)
                sid = stable_id(entry['segment_id'], start, spec.identity)
                actions = np.zeros((config.horizon, config.action_dim), np.float32)
                valid = np.zeros_like(actions, bool)
                actions[:length, :7], valid[:length, :7] = normalized[start:start + length], True
                trusted = valid.copy() if labels[start] == 'negative' else np.zeros_like(valid)
                payload = output / 'actions' / (sid + '.npz')
                np.savez_compressed(payload, actions=actions, valid_mask=valid, trusted_error_mask=trusted)
                context = entry.get('start_context', {}).get(str(start), {})
                record = Window(sid, entry['task_id'], entry['episode_id'],
                    entry.get('initial_group_id') or 'unknown:' + entry['task_id'], event[start],
                    entry['controller'], labels[start], start + entry['parent_start_step'],
                    (start + entry['parent_start_step']) / config.native_control_hz,
                    entry['instruction'], context.get('phase'), context.get('target_object'),
                    {v: dict(parquet=path, column=c, row=start) for v, c in [('front','image'),('wrist','wrist_image')]},
                    state[start].tolist(), assets.normalize('state', state[start]).tolist(),
                    state[start, :3].tolist(), quaternion_from_rotvec(state[start, 3:6]),
                    context.get('gripper_state'), context.get('held_object'), context.get('contact_state'),
                    context.get('progress'), str(payload), spec,
                    split=entry.get('split','unassigned'),
                    quality_evidence=evidence[start], object_poses=context.get('object_poses', {}),
                    bank_class=entry.get('bank_class'),bank_label=entry.get('bank_label',{}),
                    uncertain_initialization=not entry.get('initialization_verified', False),
                    source=dict(entry.get('source', {}), segment_id=entry['segment_id'], segment_row=start,
                        action_range=[start, start + length], normalization_asset=assets.path,
                        critical_state_verified=context.get('critical_state_verified', False),
                        progress_scorer_id=context.get('progress_scorer_id'),
                        state_evidence=context.get('evidence', {}), manifest_entry=entry['segment_id']))
                record.validate().payload()
                records.append(record)
                review.append(dict(sample_id=sid, segment_id=entry['segment_id'], row=start,
                    bank_class=record.bank_class,bank_label=record.bank_label,
                    training_quality=record.training_quality,
                    suggested_local_quality='unknown', image_refs=record.image_refs,
                    fields_needed=[] if record.bank_class else ['verified local behavior']))
        if entry.get('handoff_start'):
            # Exact same observed start belongs to S_vla_all too: pi05 visited it
            # immediately before yielding, though the following action is code.
            r = next(r for r in records if r.source['segment_id'] == entry['segment_id'] and r.source['segment_row'] == 0)
            d = r.to_dict()
            d.update(sample_id=stable_id(r.sample_id, 'vla_handoff_state'), controller='VLA',
                     local_quality='unknown', quality_evidence={}, bank_class=None,bank_label={},
                     action_path=str(output/'actions'/(stable_id(r.sample_id,'state')+'.npz')))
            d['source'] = dict(r.source, state_only=True, handoff_observation=True,handoff_from_bank_class=r.bank_class)
            zeros = np.zeros((config.horizon, config.action_dim), np.float32)
            np.savez_compressed(d['action_path'], actions=zeros, valid_mask=zeros.astype(bool), trusted_error_mask=zeros.astype(bool))
            records.append(Window.from_dict(d))
    audit_split(records)
    save_windows(output / 'windows.jsonl', records)
    write_jsonl(output / 'review_queue.jsonl', review)
    write_jsonl(output / 'input_manifest.jsonl', manifest)
    (output / 'config.json').write_text(json.dumps(config.to_dict(), indent=2))
    write_window_libraries(records,output)
    return records
