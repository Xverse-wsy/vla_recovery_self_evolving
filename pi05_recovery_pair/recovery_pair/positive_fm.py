"""Prepare an admitted positive pool for source-aware FM only; never run training."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .config import Config
from .data import write_jsonl, read_jsonl
from .features import sha256
from .matching import OBSERVATION_POLICY, NEGATIVE_QUERY_POLICY
from .schema import Window, load_windows, save_windows, stable_id
from .splits import audit_split
from .dense_fm import DENSE_POLICY, dense_windows, audit_dense_starts, supervision_counts

POLICY = 'positive_fm_source_tail_v1'


def tail_mode(record):
    if record.training_quality != 'positive':
        raise ValueError('Positive-only FM rejects negative, unknown and state-only records')
    if record.controller == 'code':
        return 'repeat_last'
    category = record.bank_label.get('source_category') or record.source.get('weak_category')
    if record.controller == 'expert':
        if category != 'original_expert_demonstration' or not record.source.get('expert_training_only'):
            raise ValueError('Expert positives must preserve original demonstration provenance')
        return 'repeat_last'
    if category == 'vla_progress_candidate':
        return 'full_real_only'
    if category == 'autonomous_episode_success':
        return 'repeat_last'
    raise ValueError('VLA positive source must identify intermediate progress or autonomous success')


def transform_window(record, output, *, payload=None, original=None):
    mode = tail_mode(record)
    payload = record.payload() if payload is None else payload
    original = record if original is None else original
    dims = np.asarray(record.spec.valid_dimensions, bool)
    steps = payload['valid_mask'][:, dims]
    n = int(steps.any(-1).sum())
    expected = np.zeros_like(payload['valid_mask'])
    expected[:n, dims] = True
    if not n or not np.array_equal(expected, payload['valid_mask']):
        raise ValueError('Expected a nonempty contiguous real action prefix with all environment dimensions')
    if n < record.spec.horizon and mode == 'full_real_only':
        return None
    actions = np.zeros_like(payload['actions'])
    actions[:n, dims] = payload['actions'][:n, dims]
    if n < record.spec.horizon:
        actions[n:, dims] = payload['actions'][n - 1, dims]
    valid = np.broadcast_to(dims, actions.shape).copy()
    sid = stable_id(record.sample_id, POLICY)
    dest = Path(output) / 'actions' / (sid + '.npz')
    np.savez_compressed(dest, actions=actions, valid_mask=valid,
                        trusted_error_mask=np.zeros_like(valid), real_action_mask=expected)
    data = record.to_dict()
    data.update(sample_id=sid, action_path=str(dest), visual_features={})
    data['source'] = dict(record.source, fm_tail_policy=POLICY, fm_tail_mode=mode,
        fm_real_steps=n, fm_repeated_steps=record.spec.horizon - n,
        original_window_id=original.sample_id, original_action_path=original.action_path,
        original_action_sha256=sha256(original.action_path))
    return Window.from_dict(data)


def audit_positive_fm(records, pairs, config):
    if pairs or config.pair_batch != 0 or config.eta != 0:
        raise ValueError('Positive-only artifact requires no pairs, pair_batch=0 and eta=0')
    for r in records:
        p = r.payload()
        mode = tail_mode(r)
        if r.source.get('fm_split_policy') == 'all_train_v1':
            if r.split != 'train' or r.source.get('fm_original_split') not in ('train', 'validation', 'test'):
                raise ValueError('All-train window split/provenance is invalid')
        n = r.source.get('fm_real_steps', 0)
        if (r.source.get('fm_tail_policy') != POLICY or r.source.get('fm_tail_mode') != mode
                or not 1 <= n <= r.spec.horizon
                or r.source.get('fm_repeated_steps') != r.spec.horizon - n):
            raise ValueError('Invalid source-aware FM tail provenance')
        dims = np.asarray(r.spec.valid_dimensions, bool)
        valid = np.broadcast_to(dims, p['actions'].shape)
        real = np.zeros_like(valid)
        real[:n, dims] = True
        if (not np.array_equal(p['valid_mask'], valid) or p['trusted_error_mask'].any()
                or not np.array_equal(p.get('real_action_mask'), real)):
            raise ValueError('Invalid FM tail supervision masks')
        if mode == 'full_real_only' and n != r.spec.horizon:
            raise ValueError('Intermediate VLA windows must contain a full real horizon')
        if n < r.spec.horizon and not np.all(p['actions'][n:, dims] == p['actions'][n - 1, dims]):
            raise ValueError('Successful terminal windows must repeat the last real action')
    if any(r.source.get('fm_sampling_policy') == DENSE_POLICY for r in records):
        audit_dense_starts(records)


def check_training_mode(prepared, mode):
    report = json.loads((Path(prepared) / 'report.json').read_text())
    if report.get('training_policy') in (POLICY, DENSE_POLICY) and mode != 'fm_only':
        raise ValueError('Source-aware positive artifact must use --mode fm_only')


def audit_source_alignment(records, norm_stats):
    """Deep read-only audit against original segment Parquet, not copied NPZ alone."""
    import pyarrow.parquet as pq
    from .data import QuantileAssets, read_jsonl, quaternion_from_rotvec
    assets = QuantileAssets(norm_stats)
    tables, original_hashes, manifest_hashes, manifest_entries, source_metadata_hashes = {}, {}, {}, {}, {}
    seen = set()
    repeated, max_action_error = 0, 0.
    for r in records:
        if r.sample_id in seen:
            raise ValueError('Duplicate FM window')
        seen.add(r.sample_id)
        if r.spec.normalization_sha256 != assets.sha:
            raise ValueError('Source alignment normalization mismatch')
        p, n = r.payload(), r.source['fm_real_steps']
        front, wrist = r.image_refs['front'], r.image_refs['wrist']
        if not isinstance(front, dict) or not isinstance(wrist, dict):
            raise ValueError('Deep source alignment requires segment Parquet image references')
        path, row = front['parquet'], int(front['row'])
        if (wrist['parquet'] != path or int(wrist['row']) != row
                or front.get('column') != 'image' or wrist.get('column') != 'wrist_image'
                or r.source['segment_row'] != row or r.source['action_range'] != [row, row+n]):
            raise ValueError('Starting observations and action range are misaligned')
        if path not in tables:
            table = pq.read_table(path, columns=['state', 'actions'])
            tables[path] = (np.asarray(table['state'].to_pylist(), np.float32),
                            np.asarray(table['actions'].to_pylist(), np.float32))
        state, raw = tables[path]
        if row < 0 or row+n > len(raw) or raw.shape[1] != 7:
            raise ValueError('FM window crosses the original segment boundary')
        scope = r.source.get('fm_scope_range', [0, len(raw)])
        if row < scope[0] or row+n > scope[1]:
            raise ValueError('FM window leaves admitted positive scope')
        if r.source['fm_tail_mode'] == 'repeat_last' and n < r.spec.horizon and row+n != scope[1]:
            raise ValueError('Repeated actions must begin only at original successful segment end')
        if not np.array_equal(np.asarray(r.proprio, np.float32), state[row]):
            raise ValueError('FM state is not the original pre-action observation')
        normalized = assets.normalize('actions', raw[row:row+n])
        error = float(np.max(np.abs(normalized-p['actions'][:n, :7])))
        max_action_error = max(max_action_error, error)
        if error > 2e-6 or not np.allclose(assets.normalize('state', state[row]),
                                          r.standardized_proprio, atol=2e-6, rtol=0):
            raise ValueError('FM real actions/state differ from original normalization')
        original = r.source['original_action_path']
        if original not in original_hashes:
            original_hashes[original] = sha256(original)
        if original_hashes[original] != r.source['original_action_sha256']:
            raise ValueError('Original source action payload changed')
        for path, digest in r.source.get('expert_source_metadata_sha256',{}).items():
            if path not in source_metadata_hashes:
                source_metadata_hashes[path] = sha256(path)
            if source_metadata_hashes[path] != digest:
                raise ValueError('Original expert source metadata changed')
        if r.source.get('fm_sampling_policy') == DENSE_POLICY:
            manifest = r.source['fm_source_manifest']
            if manifest not in manifest_hashes:
                manifest_hashes[manifest] = sha256(manifest)
                manifest_entries[manifest] = {e['segment_id']:e for e in read_jsonl(manifest)}
            if manifest_hashes[manifest] != r.source['fm_source_manifest_sha256']:
                raise ValueError('Original source manifest changed')
            entry = manifest_entries[manifest][r.source['segment_id']]
            step = row + entry['parent_start_step']
            hz = r.source.get('recorded_hz',r.spec.control_hz)
            if (r.start_step != step or r.start_time_s != step/hz or
                    not np.array_equal(np.asarray(r.eef_position_m,np.float32),state[row,:3]) or
                    not np.allclose(r.eef_quaternion_xyzw,quaternion_from_rotvec(state[row,3:6]),atol=1e-7)):
                raise ValueError('Dense start time or EEF state differs from source observation')
        repeated += r.source['fm_repeated_steps']
    return dict(windows_checked=len(records), original_parquets=len(tables),
        original_action_payloads_checked=len(original_hashes), max_real_action_error=max_action_error,
        repeated_action_steps=repeated, observation_action_alignment=True,
        original_segment_boundaries_respected=True, normalization_sha256=assets.sha)


def prepare_positive_fm(admission, config, output, *, all_train=False, dense=True,
                        expert_dataset=None, expected_expert_episodes=40):
    config.validate()
    if config.pair_batch != 0 or config.eta != 0:
        raise ValueError('Use the positive_fm config: pair_batch=0, eta=0')
    if config.fm_epochs is not None and not all_train:
        raise ValueError('Full positive epochs require --all-train')
    if expert_dataset is not None and (not dense or not all_train):
        raise ValueError('Expert demonstrations require dense positive-only --all-train preparation')
    previous = Path(admission).resolve()
    info = json.loads((previous / 'report.json').read_text())
    if info.get('state') != 'complete' or info.get('observation_policy') != OBSERVATION_POLICY:
        raise ValueError('Expected completed policy-input-only positive admission')
    saved = Config.load(previous / 'config.json')
    for key in ('horizon', 'action_dim', 'environment_action_dim', 'native_control_hz',
                'coverage', 'cap_success_admission', 'state_stride', 'seed', 'split_fractions', 'conservative_task_split'):
        if getattr(saved, key) != getattr(config, key):
            raise ValueError('Positive admission config differs: ' + key)
    source = previous / 'positive_pool.jsonl'
    digest = sha256(source)
    records = load_windows(source)
    if not records or len({r.sample_id for r in records}) != len(records):
        raise ValueError('Positive admission must contain unique nonempty windows')
    if info.get('positive_pool_windows') != len(records):
        raise ValueError('Positive admission inventory changed')
    audit_split(records)
    for r in records:
        tail_mode(r)
        r.payload()
    out = Path(output).resolve()
    # Never create generated files inside any original dataset or admission output.
    inputs = [previous] + [Path(r.source['dataset']).resolve() for r in records if r.source.get('dataset')]
    if expert_dataset is not None:
        inputs.append(Path(expert_dataset).resolve(strict=True))
    if any(out == p or p in out.parents for p in inputs):
        raise ValueError('Output must be outside source/admission folders')
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    status.write_text(json.dumps({'state': 'building_positive_fm'}))
    try:
        (out / 'actions').mkdir()
        accepted, dropped = [], []
        if dense:
            dropped = [dict(sample_id=r.sample_id, episode_id=r.episode_id,
                segment_id=r.source.get('segment_id'), split=r.split,
                reason='original_sparse_start_has_less_than_full_real_horizon')
                for r in records if tail_mode(r)=='full_real_only' and
                r.source['action_range'][1]-r.source['action_range'][0]<r.spec.horizon]
        candidates = dense_windows(records, previous, out) if dense else records
        for r in candidates:
            result = r if dense else transform_window(r, out)
            if result is None:
                dropped.append(dict(sample_id=r.sample_id, episode_id=r.episode_id,
                    segment_id=r.source.get('segment_id'), split=r.split,
                    reason='intermediate_vla_has_less_than_full_real_horizon'))
            else:
                if all_train:
                    result.source = dict(result.source, fm_original_split=result.split,
                                         fm_split_policy='all_train_v1')
                    result.split = 'train'
                accepted.append(result)
        expert_records = []
        if expert_dataset is not None:
            from .expert_fm import expert_windows
            norm_stats = records[0].source['normalization_asset']
            expert_records = list(expert_windows(expert_dataset,config,norm_stats,out,
                                                expected_episodes=expected_expert_episodes))
            for result in expert_records:
                result.source = dict(result.source,fm_original_split='train',fm_split_policy='all_train_v1')
            accepted.extend(expert_records)
        if not any(r.split == 'train' and r.controller == 'VLA' for r in accepted):
            raise ValueError('No train VLA positives remain')
        audit_positive_fm(accepted, [], config)
        audit_split(accepted)
        save_windows(out / 'all_windows.jsonl', accepted)
        save_windows(out / 'training_pool.jsonl', accepted)
        write_jsonl(out / 'pairs.jsonl', [])
        write_jsonl(out / 'dropped_windows.jsonl', dropped)
        write_jsonl(out / 'source_windows.jsonl', [dict(sample_id=r.sample_id,
            action_path=r.action_path, action_sha256=sha256(r.action_path)) for r in records])
        (out / 'config.json').write_text(json.dumps(config.to_dict(), indent=2))
        counts = Counter((r.split, r.controller) for r in accepted)
        report = dict(state='complete', training_policy=DENSE_POLICY if dense else POLICY, training_mode='fm_only',
            observation_policy=OBSERVATION_POLICY, negative_query_policy=NEGATIVE_QUERY_POLICY,
            source_admission=str(previous), source_positive_pool_sha256=digest,
            source_positive_windows=len(records), positive_windows=len(accepted),
            dropped_intermediate_vla_windows=len(dropped), pairs=0, negative_windows=0,
            counts_by_split_controller={'/'.join(k): v for k, v in counts.items()},
            tail_policies=dict(Counter(tail_mode(r) for r in accepted)),
            repeated_tail_windows=sum(r.source['fm_repeated_steps'] > 0 for r in accepted),
            repeated_action_steps=sum(r.source['fm_repeated_steps'] for r in accepted),
            split_policy='all_train_v1' if all_train else 'preserve_source_v1',
            validation_enabled=any(r.split == 'validation' for r in accepted),
            state_stride_unchanged=True, coverage_and_suffix_scope_unchanged=True,
            matching_state_stride=config.state_stride, training_start_stride=1 if dense else config.state_stride,
            expert_episodes=len({r.episode_id for r in expert_records}),expert_windows=len(expert_records),
            expert_dataset=str(Path(expert_dataset).resolve()) if expert_dataset is not None else None,
            expert_in_matching=False,
            fm_sampling_policy=('positive_epoch_permutation_v1' if config.fm_epochs is not None else
                                'code_quota_then_uniform_expert_plus_vla_windows_v1'),
            source_original_actions_unchanged=True, training_started=False,
            fm_only_updates=config.fm_only_updates,
            fm_epochs=config.fm_epochs,fm_only_updates_active=config.fm_epochs is None,
            note='Matching/suffix scopes unchanged; dense starts inside admitted positives plus explicitly selected expert demonstrations. Intermediate VLA requires full real H actions; successful/expert tails repeat and ARE supervised. Experts never enter matching. No pair branch.')
        if dense:
            report.update(supervision_counts(accepted))
            scopes = read_jsonl(out/'dense_scopes.jsonl')
            expert_scopes = read_jsonl(out/'expert_scopes.jsonl') if expert_records else []
            report.update(admitted_positive_scopes=len(scopes),
                expert_positive_scopes=len(expert_scopes),total_positive_scopes=len(scopes)+len(expert_scopes),
                admitted_cap_success_segments=sum(s['controller']=='code' for s in scopes),
                excluded_intermediate_tail_starts=sum(s['excluded_short_tail_starts'] for s in scopes
                                                     if s['tail_mode']=='full_real_only'))
        if config.fm_epochs is not None:
            from .batches import PositiveEpochSampler
            report['training_budget']=PositiveEpochSampler(accepted,[],config).budget_report()
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        if sha256(source) != digest:
            raise ValueError('Positive admission changed during preparation')
        status.write_text(json.dumps({'state': 'complete'}))
        checksums = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
        checksums['payloads'] = {r.sample_id: sha256(r.action_path) for r in accepted}
        (out / 'checksums.json').write_text(json.dumps(checksums, indent=2))
        return report
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admission', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--all-train', action='store_true',
                        help='Use every already-admitted positive window for training; no heldout or early stopping')
    parser.add_argument('--legacy-sparse', action='store_true', help='Reproduce historical sparse training starts')
    parser.add_argument('--expert-dataset', help='Original expert dataset with explicit episodes.json, training only')
    parser.add_argument('--expected-expert-episodes', type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(prepare_positive_fm(args.admission, Config.load(args.config), args.output,
        all_train=args.all_train,dense=not args.legacy_sparse,expert_dataset=args.expert_dataset,
        expected_expert_episodes=args.expected_expert_episodes), indent=2))


if __name__ == '__main__':
    main()
