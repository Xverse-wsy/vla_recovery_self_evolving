"""Complete VLA-negative matching against an existing admitted positive pool; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .config import Config
from .data import write_jsonl
from .features import sha256
from .matching import FrozenMatcher, OBSERVATION_POLICY, NEGATIVE_QUERY_POLICY, numeric_match, image_distance, pair_mask
from .schema import load_windows, save_windows
from .success_admission import check_coverage_features


def numeric_candidates(matcher, negative):
    candidates = []
    for ref in matcher.positive:
        d, reason = numeric_match(negative, ref, matcher.config.pairing)
        if reason is None:
            rank = sum(d[k] / getattr(matcher.config.pairing, k)
                       for k in ('position_m', 'rotation_deg', 'proprio_rms'))
            candidates.append((rank, ref.sample_id, ref, d))
    return [(r, d) for _, _, r, d in sorted(candidates)[:matcher.config.pairing.max_candidates]]


def evaluate(matcher):
    pairs = matcher.build_pairs()
    selected = {p['negative_id']: p for p in pairs}
    rows = []
    for neg in matcher.negative:
        numeric = numeric_candidates(matcher, neg)
        for ref, _ in numeric:
            _, reason = image_distance(neg, ref)
            if reason:
                raise ValueError(f'Missing candidate features: {reason}')
        visual = matcher.match(neg, [r for r, _ in numeric], matcher.config.pairing)
        rejected = Counter()
        for ref in visual:
            positive = matcher.records[ref['sample_id']]
            if positive.episode_id == neg.episode_id and positive.start_step == neg.start_step:
                rejected['same_start_opposing_labels'] += 1
            else:
                _, reason = pair_mask(positive, neg, matcher.config)
                if reason:
                    rejected[reason] += 1
        pair = selected.get(neg.sample_id)
        rows.append(dict(negative_id=neg.sample_id, episode_id=neg.episode_id,
            segment_id=neg.source.get('segment_id', neg.sample_id), split=neg.split,
            instruction=neg.instruction, start_step=neg.start_step,
            valid_action_steps=int(neg.payload()['valid_mask'].any(axis=-1).sum()),
            numerical_references=len(numeric), visual_references=len(visual),
            training_references=len(pair['references']) if pair else 0,
            action_rejections=dict(rejected),
            selected_positive_ids=[r['sample_id'] for r in pair['references']] if pair else []))
    paired = [r for r in rows if r['training_references']]
    summary = dict(negative_query_windows=len(rows),
        negative_query_episodes=len({r['episode_id'] for r in rows}),
        numerical_match_windows=sum(r['numerical_references'] > 0 for r in rows),
        numerical_relations=sum(r['numerical_references'] for r in rows),
        visual_match_windows=sum(r['visual_references'] > 0 for r in rows),
        visual_match_episodes=len({r['episode_id'] for r in rows if r['visual_references']}),
        visual_relations=sum(r['visual_references'] for r in rows),
        paired_negative_windows=len(paired), paired_negative_episodes=len({r['episode_id'] for r in paired}),
        paired_negative_segments=len({r['segment_id'] for r in paired}),
        selected_positive_references=sum(r['training_references'] for r in rows),
        action_rejections=dict(sum((Counter(r['action_rejections']) for r in rows), Counter())),
        paired_episodes_by_split={s: len({r['episode_id'] for r in paired if r['split'] == s})
                                  for s in ('train', 'validation', 'test')},
        positive_references_by_controller=dict(Counter(matcher.records[r['sample_id']].controller
            for p in pairs for r in p['references'])))
    return pairs, rows, summary


def match_negatives(windows, admission, checkpoint, config, output, device='cuda:0', memory_gib=8., batch_size=2):
    config.validate(matching_ready=True)
    source, previous, out = Path(windows).resolve(), Path(admission).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    try:
        status.write_text(json.dumps({'state': 'checking_positive_pool'}))
        digest = sha256(source)
        report = json.loads((previous / 'report.json').read_text())
        if (report.get('state') != 'complete' or report.get('observation_policy') != OBSERVATION_POLICY
                or report.get('source_windows_sha256') != digest):
            raise ValueError('Positive admission source/policy/completion mismatch')
        saved = Config.load(previous / 'config.json')
        for key in ('coverage', 'cap_success_admission', 'seed', 'split_fractions', 'conservative_task_split'):
            if getattr(saved, key) != getattr(config, key):
                raise ValueError(f'Positive admission configuration changed: {key}')
        trial = Path(report['feature_trial'])
        trial_info = json.loads((trial / 'report.json').read_text())
        if (trial_info.get('observation_policy') != OBSERVATION_POLICY or
                trial_info['inputs']['input_sha256'] != digest):
            raise ValueError('Cached feature source/policy mismatch')
        records = load_windows(source)
        by_id = {r.sample_id: r for r in records}
        for featured in load_windows(trial / 'windows_with_features.jsonl'):
            a, b = featured.to_dict(), by_id[featured.sample_id].to_dict()
            a.pop('visual_features', None)
            b.pop('visual_features', None)
            if a != b:
                raise ValueError('Cached feature window differs from source')
            by_id[featured.sample_id].visual_features = featured.visual_features
        check_coverage_features(records, config)
        matcher = FrozenMatcher(records, config)
        expected = {r.sample_id for r in load_windows(previous / 'positive_pool.jsonl')}
        if {r.sample_id for r in matcher.positive} != expected:
            raise ValueError('Positive pool changed while preparing negative matching')
        candidates = [(n, r) for n in matcher.negative for r, _ in numeric_candidates(matcher, n)]
        needed = {r.sample_id for pair in candidates for r in pair}
        missing = [by_id[sid] for sid in sorted(needed) if not by_id[sid].visual_features]
        feature_report = None
        if missing:
            status.write_text(json.dumps({'state': 'encoding_candidate_rgb', 'missing_windows': len(missing),
                                          'numerical_relations': len(candidates)}))
            from .pi05_vision import Pi05Vision
            encoder = Pi05Vision(checkpoint, device, memory_gib)
            if encoder.encoder_sha != report['encoder_sha256']:
                raise ValueError('Negative encoder differs from positive-pool encoder')
            if any(r.visual_features and r.visual_features.get('calibration_id') != encoder.preprocess_id
                   for r in matcher.positive):
                raise ValueError('Negative image preprocessing differs from positive-pool preprocessing')
            feature_report = encoder.extract(missing, out / 'feature_cache', batch_size)
        status.write_text(json.dumps({'state': 'visual_and_action_pairing'}))
        pairs, rows, summary = evaluate(matcher)
        save_windows(out / 'windows_with_features.jsonl', records)
        write_jsonl(out / 'pairs.jsonl', pairs)
        write_jsonl(out / 'negative_window_results.jsonl', rows)
        save_windows(out / 'matched_negative_windows.jsonl',
                     [by_id[r['negative_id']] for r in rows if r['training_references']])
        (out / 'config.json').write_text(json.dumps(config.to_dict(), indent=2))
        if sha256(source) != digest:
            raise ValueError('Source window index changed')
        result = dict(state='complete', observation_policy=OBSERVATION_POLICY,
            negative_query_policy=NEGATIVE_QUERY_POLICY, pairing=config.to_dict()['pairing'], **summary,
            positive_pool_windows=len(matcher.positive), excluded_cap_negative_windows=len(matcher.excluded_negative),
            newly_encoded_windows=len(missing), feature_report=feature_report,
            source_windows_sha256=digest, positive_admission=str(previous),
            training_started=False, original_datasets_modified=False, formal_prepared_pool_created=False,
            note='All counts use fixed current thresholds. State matches and action-valid training pairs are reported separately; no accuracy or task-success claim.')
        (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        status.write_text(json.dumps({'state': 'complete'}))
        return result
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('windows', 'admission', 'checkpoint', 'config', 'output'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-cuda-memory-gb', type=float, default=8.)
    p.add_argument('--batch-size', type=int, default=2)
    a = p.parse_args()
    print(json.dumps(match_negatives(a.windows, a.admission, a.checkpoint, Config.load(a.config),
        a.output, a.device, a.max_cuda_memory_gb, a.batch_size), indent=2))


if __name__ == '__main__':
    main()
