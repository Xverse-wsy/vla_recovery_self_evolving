"""Prepare an admitted positive pool for source-aware FM only; never run training."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .config import Config
from .data import write_jsonl
from .features import sha256
from .matching import OBSERVATION_POLICY, NEGATIVE_QUERY_POLICY
from .schema import Window, load_windows, save_windows, stable_id
from .splits import audit_split

POLICY = 'positive_fm_source_tail_v1'


def tail_mode(record):
    if record.training_quality != 'positive':
        raise ValueError('Positive-only FM rejects negative, unknown and state-only records')
    if record.controller == 'code':
        return 'repeat_last'
    category = record.bank_label.get('source_category') or record.source.get('weak_category')
    if category == 'vla_progress_candidate':
        return 'full_real_only'
    if category == 'autonomous_episode_success':
        return 'repeat_last'
    raise ValueError('VLA positive source must identify intermediate progress or autonomous success')


def transform_window(record, output):
    mode = tail_mode(record)
    payload = record.payload()
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
        original_window_id=record.sample_id, original_action_path=record.action_path,
        original_action_sha256=sha256(record.action_path))
    return Window.from_dict(data)


def audit_positive_fm(records, pairs, config):
    if pairs or config.pair_batch != 0 or config.eta != 0:
        raise ValueError('Positive-only artifact requires no pairs, pair_batch=0 and eta=0')
    for r in records:
        p = r.payload()
        mode = tail_mode(r)
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


def check_training_mode(prepared, mode):
    report = json.loads((Path(prepared) / 'report.json').read_text())
    if report.get('training_policy') == POLICY and mode != 'fm_only':
        raise ValueError('Source-aware positive artifact must use --mode fm_only')


def prepare_positive_fm(admission, config, output):
    config.validate()
    if config.pair_batch != 0 or config.eta != 0:
        raise ValueError('Use the positive_fm config: pair_batch=0, eta=0')
    previous = Path(admission).resolve()
    info = json.loads((previous / 'report.json').read_text())
    if info.get('state') != 'complete' or info.get('observation_policy') != OBSERVATION_POLICY:
        raise ValueError('Expected completed policy-input-only positive admission')
    saved = Config.load(previous / 'config.json')
    for key in ('horizon', 'action_dim', 'environment_action_dim', 'native_control_hz',
                'coverage', 'cap_success_admission', 'seed', 'split_fractions', 'conservative_task_split'):
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
    if any(out == p or p in out.parents for p in inputs):
        raise ValueError('Output must be outside source/admission folders')
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    status.write_text(json.dumps({'state': 'building_positive_fm'}))
    try:
        (out / 'actions').mkdir()
        accepted, dropped = [], []
        for r in records:
            result = transform_window(r, out)
            if result is None:
                dropped.append(dict(sample_id=r.sample_id, episode_id=r.episode_id,
                    segment_id=r.source.get('segment_id'), split=r.split,
                    reason='intermediate_vla_has_less_than_full_real_horizon'))
            else:
                accepted.append(result)
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
        report = dict(state='complete', training_policy=POLICY, training_mode='fm_only',
            observation_policy=OBSERVATION_POLICY, negative_query_policy=NEGATIVE_QUERY_POLICY,
            source_admission=str(previous), source_positive_pool_sha256=digest,
            source_positive_windows=len(records), positive_windows=len(accepted),
            dropped_intermediate_vla_windows=len(dropped), pairs=0, negative_windows=0,
            counts_by_split_controller={'/'.join(k): v for k, v in counts.items()},
            tail_policies=dict(Counter(tail_mode(r) for r in accepted)),
            repeated_tail_windows=sum(r.source['fm_repeated_steps'] > 0 for r in accepted),
            repeated_action_steps=sum(r.source['fm_repeated_steps'] for r in accepted),
            state_stride_unchanged=True, coverage_and_suffix_scope_unchanged=True,
            source_original_actions_unchanged=True, training_started=False,
            fm_only_updates=config.fm_only_updates,
            note='Retains existing window starts and split; repeated terminal actions ARE supervised. No pair branch or expert40 added.')
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
    args = parser.parse_args()
    print(json.dumps(prepare_positive_fm(args.admission, Config.load(args.config), args.output), indent=2))


if __name__ == '__main__':
    main()
