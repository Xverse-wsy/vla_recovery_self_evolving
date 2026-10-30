"""Portable success-only admission using cached pi05 RGB features; no negative pairing."""
import argparse
import json
from pathlib import Path

from .config import Config
from .data import write_jsonl
from .features import sha256
from .matching import FrozenMatcher, OBSERVATION_POLICY, image_distance, instruction_key, numeric_match
from .schema import load_windows, save_windows


def check_coverage_features(records, config):
    groups = {}
    for r in records:
        if r.controller == 'VLA' and r.training_quality == 'positive':
            groups.setdefault((r.split, instruction_key(r)), []).append(r)
    checked = 0
    for q in records:
        if q.controller != 'code' or q.training_quality != 'positive':
            continue
        candidates = []
        for ref in groups.get((q.split, instruction_key(q)), []):
            d, reason = numeric_match(q, ref, config.coverage)
            if reason is None:
                rank = sum(d[k] / getattr(config.coverage, k)
                           for k in ('position_m', 'rotation_deg', 'proprio_rms'))
                candidates.append((rank, ref.sample_id, ref))
        for _, _, ref in sorted(candidates)[:config.coverage.max_candidates]:
            _, reason = image_distance(q, ref)
            if reason:
                raise ValueError(f'Incomplete coverage candidate features: {q.sample_id} / {ref.sample_id}: {reason}')
            checked += 1
    return checked


def admit_success(windows, feature_trial, config, output):
    config.validate(matching_ready=True)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    try:
        status.write_text(json.dumps({'state': 'checking_frozen_candidate_features'}))
        source, trial = Path(windows).resolve(), Path(feature_trial).resolve()
        digest = sha256(source)
        info = json.loads((trial / 'report.json').read_text())
        if (info.get('observation_policy') != OBSERVATION_POLICY or
                info['inputs']['input_sha256'] != digest or info.get('state') != 'complete'):
            raise ValueError('Feature trial source/policy/completion mismatch')
        records = load_windows(source)
        by_id = {r.sample_id: r for r in records}
        if len(records) != len(by_id):
            raise ValueError('Duplicate source windows')
        for featured in load_windows(trial / 'windows_with_features.jsonl'):
            original = by_id.get(featured.sample_id)
            a, b = featured.to_dict(), original.to_dict() if original else {}
            a.pop('visual_features', None)
            b.pop('visual_features', None)
            if a != b:
                raise ValueError('Feature index differs from original window')
            original.visual_features = featured.visual_features
        checked = check_coverage_features(records, config)
        status.write_text(json.dumps({'state': 'matching_and_admitting_success_suffixes'}))
        matcher = FrozenMatcher(records, config)
        # Check whole-segment completion as an index audit, never a similarity input.
        import pyarrow.parquet as pq
        for suffix in matcher.success_segments:
            anchor = by_id[suffix['anchor_id']]
            ref = anchor.image_refs['front']
            if isinstance(ref, dict) and 'parquet' in ref:
                length = pq.ParquetFile(ref['parquet']).metadata.num_rows
                if suffix['segment_row_range'][1] != length:
                    raise ValueError('Admitted suffix does not reach the original program segment end')
        admitted = [r for r in matcher.positive if r.controller == 'code']
        save_windows(out / 'admitted_cap_success_windows.jsonl', admitted)
        save_windows(out / 'positive_pool.jsonl', matcher.positive)
        write_jsonl(out / 'success_matches.jsonl', matcher.success_matches)
        write_jsonl(out / 'cap_success_suffixes.jsonl', matcher.success_segments)
        write_jsonl(out / 'coverage.jsonl', [dict(sample_id=k, **v) for k, v in matcher.coverage.items()])
        write_jsonl(out / 'rejections.jsonl', matcher.rejections)
        (out / 'config.json').write_text(json.dumps(config.to_dict(), indent=2))
        if sha256(source) != digest:
            raise RuntimeError('Original source window index changed')
        report = dict(state='complete', observation_policy=OBSERVATION_POLICY,
            negative_query_policy=config.negative_query_policy,
            negative_query_windows=len(matcher.negative), excluded_cap_negative_windows=len(matcher.excluded_negative),
            cap_success_admission=config.cap_success_admission, coverage=config.to_dict()['coverage'],
            numerical_candidates_checked=checked,
            direct_match_relations=sum(r['candidate_count'] for r in matcher.success_matches),
            directly_matched_cap_windows=len(matcher.success_matches),
            admitted_cap_windows=len(admitted), admitted_cap_success_segments=len(matcher.success_segments),
            admitted_cap_action_steps=sum(r['action_steps'] for r in matcher.success_segments),
            vla_positive_windows=len(matcher.vla_success), positive_pool_windows=len(matcher.positive),
            admitted_by_split={s: len([r for r in admitted if r.split == s])
                               for s in ('train', 'validation', 'test')},
            source_windows_sha256=digest, source_windows=str(source), feature_trial=str(trial),
            encoder_sha256=info['feature_report']['encoder_sha256'],
            negative_pairing_evaluated=False, training_started=False,
            original_datasets_modified=False, formal_prepared_pool_created=False,
            note='Downstream suffix windows are admitted by successful-program provenance, not claimed to be individually state-matched. Negative pairing still requires features for its full candidate inventory.')
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        status.write_text(json.dumps({'state': 'complete'}))
        return report
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('windows', 'feature-trial', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(admit_success(args.windows, args.feature_trial, Config.load(args.config), args.output), indent=2))


if __name__ == '__main__':
    main()
