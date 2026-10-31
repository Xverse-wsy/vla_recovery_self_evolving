"""Immutable split-local artifacts, calibration, audit and provenance."""
from collections import Counter
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
from .config import MatchThresholds
from .data import read_jsonl, write_jsonl
from .schema import load_windows, save_windows, stable_id
from .features import sha256
from .matching import FrozenMatcher, image_distance, numeric_match, pair_mask, OBSERVATION_POLICY, NEGATIVE_QUERY_POLICY
from .splits import audit_split


def prepare(records, config, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    matcher = FrozenMatcher(records, config)
    pairs = matcher.build_pairs()
    accepted = {r.sample_id for r in matcher.positive + matcher.negative}
    save_windows(out / 'all_windows.jsonl', records)
    save_windows(out / 'training_pool.jsonl', [r for r in records if r.sample_id in accepted])
    write_jsonl(out / 'pairs.jsonl', pairs)
    write_jsonl(out / 'excluded_negative_windows.jsonl', [dict(sample_id=r.sample_id,
        episode_id=r.episode_id, controller=r.controller, bank_class=r.bank_class,
        reason='cap_failure_excluded_from_negative_matching') for r in matcher.excluded_negative])
    write_jsonl(out / 'coverage.jsonl', [dict(sample_id=k, **v) for k, v in matcher.coverage.items()])
    write_jsonl(out / 'success_matches.jsonl', matcher.success_matches)
    write_jsonl(out / 'cap_success_suffixes.jsonl', matcher.success_segments)
    save_windows(out / 'admitted_cap_success_windows.jsonl',
                 [r for r in matcher.positive if r.controller=='code'])
    direct_ids = {r['sample_id'] for r in matcher.success_matches}
    save_windows(out / 'matched_cap_success_windows.jsonl',
                 [r for r in matcher.positive if r.sample_id in direct_ids])
    save_windows(out / 'directly_matched_cap_success_windows.jsonl',
                 [r for r in matcher.positive if r.sample_id in direct_ids])
    write_jsonl(out / 'rejections.jsonl', matcher.rejections)
    write_jsonl(out / 'frozen_vla_states.jsonl', [dict(sample_id=r.sample_id, split=r.split,
        episode_id=r.episode_id, quality=r.training_quality) for r in matcher.vla_states])
    save_windows(out / 'frozen_vla_success.jsonl', matcher.vla_success)
    # These are AFTER matching/coverage. Existing libraries are already exported
    # by the independent libraries/build stages, even with unset thresholds.
    for split in ('train','validation','test'):
        bank = out/'eligible_banks'/split
        state_records = [r for r in matcher.vla_states if r.split==split]
        save_windows(bank/'S_vla_all.jsonl',state_records)
        for quality,sign in [('positive','positive'),('negative','negative')]:
            save_windows(bank/f'B_{sign}.jsonl',[r for r in state_records if r.training_quality==quality])
        for controller,name in [('VLA','vla'),('code','code')]:
            for quality in ('positive','negative'):
                pool = matcher.positive if quality=='positive' else matcher.negative
                save_windows(bank/f'D_{name}_{quality}.jsonl',
                    [r for r in pool if r.split==split and r.controller==controller])
    counts = Counter((r.split, r.controller, r.training_quality) for r in records)
    report = dict(classes={'/'.join(k): v for k, v in counts.items()},
        accepted_positive=len(matcher.positive), accepted_negative=len(matcher.negative),
        paired_windows=len(pairs), paired_negative_episodes=len({p['negative_episode_id'] for p in pairs if p['split']=='train'}),
        frozen_bank_id=matcher.coverage_bank_id, code_coverage_is_frozen=True,
        matching_policy='cap_success_to_frozen_vla_success_v1',
        observation_policy=OBSERVATION_POLICY,
        negative_query_policy=NEGATIVE_QUERY_POLICY,
        excluded_cap_negative_windows=len(matcher.excluded_negative),
        matched_cap_success_windows=len(matcher.success_matches),
        admitted_cap_success_windows=len(matcher.coverage),
        admitted_cap_success_suffixes=len(matcher.success_segments),
        cap_success_admission=config.cap_success_admission,
        frozen_vla_success_windows=len(matcher.vla_success),
        label_policy='Existing four segment classes are training labels; optional local evidence stays separate.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'config.json').write_text(json.dumps(config.to_dict(), indent=2))
    checksums = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
    checksums['payloads'] = {r.sample_id: sha256(r.action_path) for r in records}
    (out / 'checksums.json').write_text(json.dumps(checksums, indent=2))
    return report


def audit_prepared(path, config=None):
    path = Path(path)
    from .config import Config
    stored_config = Config.load(path / 'config.json')
    report = json.loads((path/'report.json').read_text())
    if report.get('observation_policy') != OBSERVATION_POLICY:
        raise ValueError('Rebuild prepared matches using policy inputs only; old matching rules are unsupported')
    if report.get('negative_query_policy') != NEGATIVE_QUERY_POLICY:
        raise ValueError('Rebuild prepared matches using VLA-only negative queries; old negative matching rules are unsupported')
    # Microbatches/optimizer may change. Sampling/masks/split/matching may not.
    if config:
        for key in ('horizon','action_dim','environment_action_dim','native_control_hz','flow_steps',
                    'min_pair_steps','shared_prefix_tolerance','identical_action_tolerance',
                    'coverage','pairing','cap_success_admission','negative_query_policy','split_fractions','conservative_task_split'):
            if getattr(config,key) != getattr(stored_config,key):
                raise ValueError(f'Prepared artifact config mismatch: {key}')
    hashes = json.loads((path / 'checksums.json').read_text())
    for name, digest in hashes.items():
        if name != 'payloads' and sha256(path / name) != digest:
            raise ValueError(f'Prepared artifact changed: {name}')
    all_records = load_windows(path / 'all_windows.jsonl')
    audit_split(all_records)
    records = load_windows(path / 'training_pool.jsonl')
    if any(r.training_quality == 'negative' and r.controller != 'VLA' for r in records):
        raise ValueError('Eligible negative training pool must contain VLA failures only')
    by_id = {r.sample_id:r for r in records}
    for r in all_records:
        r.validate().payload()
        if sha256(r.action_path) != hashes['payloads'][r.sample_id]:
            raise ValueError('Action payload changed')
    pairs = read_jsonl(path / 'pairs.jsonl')
    if report.get('training_policy') in ('positive_fm_source_tail_v1', 'positive_fm_dense_source_tail_v2'):
        from .positive_fm import audit_positive_fm
        audit_positive_fm(records, pairs, config or stored_config)
        if {r.sample_id for r in all_records} != set(by_id):
            raise ValueError('Positive-only all_windows must equal the training pool')
    for p in pairs:
        neg = by_id[p['negative_id']]
        if (neg.training_quality != 'negative' or neg.controller != 'VLA' or
                p.get('negative_query_policy') != NEGATIVE_QUERY_POLICY or
                p.get('negative_controller') != 'VLA' or not 1 <= len(p['references']) <= 3):
            raise ValueError('Invalid negative/ref inventory')
        for ref in p['references']:
            pos = by_id[ref['sample_id']]
            if pos.training_quality != 'positive' or pos.split != neg.split:
                raise ValueError('Pair label/split mismatch')
            mask, reason = pair_mask(pos, neg, stored_config)
            if reason or not np.array_equal(mask, ref['mask']):
                raise ValueError('Pair mask no longer matches provenance')
    return records, pairs, sha256(path / 'checksums.json')


def calibrate(records, reviews, config, output):
    """Reviewed SAME-STATE compatibility examples, from TRAIN only.

    Thresholds are candidate upper bounds, not an assertion of measured accuracy.
    Rejected reviewed examples and resulting false admissions are reported.
    """
    by_id = {r.sample_id:r for r in records}
    accepted = []
    for row in reviews:
        a, b = by_id[row['query_id']], by_id[row['reference_id']]
        if a.split != 'train' or b.split != 'train':
            raise ValueError('Calibration cannot use validation/test candidates')
        if not row.get('reviewer') or not isinstance(row.get('compatible'), bool):
            raise ValueError('State compatibility review needs reviewer and boolean label')
        d, reason = numeric_match(a,b, replace(config.coverage, proprio_rms=float('inf')))
        im, error = image_distance(a,b)
        if reason or error:
            raise ValueError(f'Calibration candidate fails observable-input gates: {reason or error}')
        accepted.append(dict(**row, proprio_rms=d['proprio_rms'], image_cosine=max(im.values())))
    positive = [r for r in accepted if r['compatible']]
    if not positive:
        raise ValueError('Need reviewed compatible TRAIN examples for calibration')
    # 95th percentile starting calibration; real use requires reviewing report.
    q = {k:max(1e-8,float(np.quantile([r[k] for r in positive], .95))) for k in ['proprio_rms','image_cosine']}
    calibrated = replace(config, coverage=replace(config.coverage,**q), pairing=replace(config.pairing,**q))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(calibrated.to_dict(),indent=2))
    false_admissions = [r for r in accepted if not r['compatible'] and all(r[k]<=q[k] for k in q)]
    report = dict(calibration_split='train', observation_policy=OBSERVATION_POLICY, reviews=accepted, thresholds=q,
        false_admissions=false_admissions, automatic_accuracy_claim=False,
        recommendation='Review false admissions; tighten pairing thresholds if needed. No heldout data was used.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report
