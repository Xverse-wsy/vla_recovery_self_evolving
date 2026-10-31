"""Every-frame FM windows inside immutable, already-admitted positive scopes.

This is deliberately downstream of matching. It never runs a matcher or admits a
new program. Source manifests define label/event cuts; CAP anchors define suffixes.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import QuantileAssets, quaternion_from_rotvec, read_jsonl, write_jsonl
from .features import sha256
from .schema import Window, stable_id

DENSE_POLICY = 'positive_fm_dense_source_tail_v2'


def dense_windows(records, admission, output):
    from .positive_fm import tail_mode, transform_window
    suffixes = {(s['episode_id'], s['segment_id']): s
                for s in read_jsonl(Path(admission) / 'cap_success_suffixes.jsonl')}
    manifests, assets, groups = {}, {}, defaultdict(list)
    for r in records:
        path = Path(r.action_path).parent.parent / 'input_manifest.jsonl'
        if path not in manifests:
            if not path.is_file():
                raise ValueError('Dense training requires original input_manifest.jsonl')
            manifests[path] = (sha256(path), {e['segment_id']: e for e in read_jsonl(path)})
        entry = manifests[path][1][r.source['segment_id']]
        parquet = str(Path(entry['parquet']).resolve())
        if (parquet != r.image_refs['front']['parquet'] or
                r.episode_id != entry['episode_id'] or r.controller != entry['controller']):
            raise ValueError('Dense source manifest identity mismatch')
        cuts = sorted({0, entry['length']} |
                      {int(a[k]) for a in entry.get('annotations', []) for k in ('start', 'end')})
        row = int(r.source['segment_row'])
        bounds = next(((lo, hi) for lo, hi in zip(cuts, cuts[1:]) if lo <= row < hi), None)
        if bounds is None:
            raise ValueError('Dense source row outside label boundaries')
        groups[(path, r.source['segment_id'], *bounds)].append(r)

    scopes = []
    for (manifest, segment, boundary_lo, boundary_hi), sources in groups.items():
        sources.sort(key=lambda r: r.source['segment_row'])
        anchor = sources[0]
        entry = manifests[manifest][1][segment]
        lo, hi = boundary_lo, boundary_hi
        suffix = None
        if anchor.controller == 'code':
            suffix = suffixes.get((anchor.episode_id, segment))
            if not suffix or suffix['admission_kind'] != 'successful_segment_suffix':
                raise ValueError('Dense CAP training requires an existing successful suffix admission')
            lo, hi = suffix['segment_row_range']
            if not boundary_lo <= lo < hi <= boundary_hi or suffix['anchor_id'] != anchor.sample_id:
                raise ValueError('CAP suffix would cross label boundaries or change its admitted anchor')
            if set(suffix['admitted_window_ids']) != {r.sample_id for r in sources}:
                raise ValueError('CAP suffix admission inventory changed')
            suffix = {k: suffix[k] for k in ('episode_id', 'segment_id', 'anchor_id',
                'segment_row_range', 'admission_kind', 'anchor_vla_reference', 'frozen_bank_id') if k in suffix}
        if (sources[0].source['segment_row'] != lo or
                max(r.source['action_range'][1] for r in sources) != hi):
            raise ValueError('Admitted windows do not reach the original positive scope boundaries')
        identity = {(r.split, r.event_id, r.local_quality, r.bank_class,
                     r.instruction, r.spec.identity, tail_mode(r)) for r in sources}
        if len(identity) != 1:
            raise ValueError('Dense scope mixes split, event, label or action specification')
        table = pq.read_table(entry['parquet'], columns=['state', 'actions'])
        states = np.asarray(table['state'].to_pylist(), np.float32)
        raw = np.asarray(table['actions'].to_pylist(), np.float32)
        if raw.shape != (entry['length'], 7) or states.shape != (len(raw), 8):
            raise ValueError('Dense source state/action layout mismatch')
        norm = anchor.source['normalization_asset']
        if norm not in assets:
            assets[norm] = QuantileAssets(norm)
        asset = assets[norm]
        if asset.sha != anchor.spec.normalization_sha256:
            raise ValueError('Dense source normalization changed')
        normalized = asset.normalize('actions', raw)
        # Verify old sparse payloads against the same real action rows before expansion.
        for r in sources:
            a, b = r.source['action_range']
            if not lo <= a < b <= hi or not np.allclose(
                    r.payload()['actions'][:b-a, :7], normalized[a:b], atol=2e-6, rtol=0):
                raise ValueError('Original admitted actions differ from source Parquet')
        full_only = tail_mode(anchor) == 'full_real_only'
        anchor_sha = sha256(anchor.action_path)
        stop = hi - anchor.spec.horizon + 1 if full_only else hi
        scopes.append(dict(segment_id=segment, episode_id=anchor.episode_id,
            controller=anchor.controller, source_category=anchor.bank_label.get('source_category'),
            scope_range=[lo,hi], label_boundary=[boundary_lo,boundary_hi],
            anchor_id=anchor.sample_id, tail_mode=tail_mode(anchor),
            eligible_starts=max(0,stop-lo), excluded_short_tail_starts=(hi-lo)-max(0,stop-lo)))
        # Empty short intermediate segments produce no windows, and no invented actions.
        for row in range(lo, max(lo, stop)):
            n = min(anchor.spec.horizon, hi-row)
            data = anchor.to_dict()
            context = entry.get('start_context', {}).get(str(row), {})
            data.update(sample_id=stable_id(segment, lo, hi, row, anchor.spec.identity, DENSE_POLICY),
                start_step=row + entry['parent_start_step'],
                start_time_s=(row + entry['parent_start_step']) / anchor.spec.control_hz,
                image_refs={v: dict(parquet=str(Path(entry['parquet']).resolve()), column=c, row=row)
                            for v, c in [('front', 'image'), ('wrist', 'wrist_image')]},
                proprio=states[row].tolist(), standardized_proprio=asset.normalize('state', states[row]).tolist(),
                eef_position_m=states[row, :3].tolist(),
                eef_quaternion_xyzw=quaternion_from_rotvec(states[row, 3:6]), visual_features={},
                phase=context.get('phase'), target_object=context.get('target_object'),
                gripper_state=context.get('gripper_state'), held_object=context.get('held_object'),
                contact_state=context.get('contact_state'), progress=context.get('progress'),
                object_poses=context.get('object_poses', {}))
            data['source'] = dict(anchor.source, segment_row=row, action_range=[row, row+n],
                fm_start_stride=1, fm_sampling_policy=DENSE_POLICY, fm_scope_range=[lo, hi],
                fm_label_boundary=[boundary_lo, boundary_hi],
                fm_source_manifest=str(manifest), fm_source_manifest_sha256=manifests[manifest][0],
                fm_admission_anchor_id=anchor.sample_id,
                fm_admission_anchor_action_path=anchor.action_path,
                fm_admission_anchor_action_sha256=anchor_sha,
                fm_matching_reference_applies_to='admitted_scope_anchor',
                fm_cap_suffix=suffix,
                critical_state_verified=context.get('critical_state_verified', False),
                progress_scorer_id=context.get('progress_scorer_id'), state_evidence=context.get('evidence', {}))
            actions = np.zeros((anchor.spec.horizon, anchor.spec.model_dim), np.float32)
            valid = np.zeros_like(actions, bool)
            actions[:n, :7], valid[:n, :7] = normalized[row:row+n], True
            # Reuse the single tail-policy implementation without writing temporary source payloads.
            result = transform_window(Window.from_dict(data), output,
                payload=dict(actions=actions, valid_mask=valid, trusted_error_mask=np.zeros_like(valid)),
                original=anchor)
            yield result
    for path, (digest, _) in manifests.items():
        if sha256(path) != digest:
            raise ValueError('Original source manifest changed during dense preparation')
    write_jsonl(Path(output)/'dense_scopes.jsonl',scopes)


def audit_dense_starts(records):
    groups = defaultdict(list)
    for r in records:
        if r.source.get('fm_sampling_policy') != DENSE_POLICY:
            raise ValueError('Dense artifact contains a non-dense window')
        lo, hi = r.source['fm_scope_range']
        blo, bhi = r.source['fm_label_boundary']
        if not blo <= lo < hi <= bhi or r.source['fm_start_stride'] != 1:
            raise ValueError('Invalid dense source scope')
        groups[(r.source['fm_source_manifest'], r.source['segment_id'], lo, hi)].append(r)
    for (_, _, lo, hi), group in groups.items():
        from .positive_fm import tail_mode
        r = group[0]
        stop = hi-r.spec.horizon+1 if tail_mode(r) == 'full_real_only' else hi
        if sorted(x.source['segment_row'] for x in group) != list(range(lo, max(lo, stop))):
            raise ValueError('Dense starts are missing, duplicated or outside admitted scope')
        if r.controller == 'code':
            suffix = r.source['fm_cap_suffix']
            if suffix['segment_row_range'] != [lo, hi] or suffix['anchor_id'] != r.source['fm_admission_anchor_id']:
                raise ValueError('Dense CAP scope differs from admitted successful suffix')


def supervision_counts(records):
    """Unique source rows are distinct from overlapping window supervision slots."""
    unique, windows, slots, repeats = defaultdict(set), defaultdict(int), defaultdict(int), defaultdict(int)
    for r in records:
        category = r.bank_label.get('source_category') or r.source.get('weak_category')
        row, n = r.source['segment_row'], r.source['fm_real_steps']
        path = r.image_refs['front']['parquet']
        unique[category].update((path, j) for j in range(row, row+n))
        windows[category] += 1
        slots[category] += n
        repeats[category] += r.source['fm_repeated_steps']
    return dict(windows_by_category=dict(windows), unique_real_frames_by_category={k: len(v) for k,v in unique.items()},
                unique_real_frames=sum(map(len, unique.values())), real_action_slots=sum(slots.values()),
                repeated_action_slots=sum(repeats.values()), supervised_action_slots=sum(slots.values())+sum(repeats.values()))
