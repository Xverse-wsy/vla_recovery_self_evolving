"""Audit a visual pilot against explicit start-observation reviews, without admission."""
import argparse
import hashlib
import json
from pathlib import Path

from .vision_trial import threshold_grid


def audit_reviews(rows, reviews):
    by_id = {r['pair_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate candidate pair IDs')
    seen, labeled, unknown = set(), [], 0
    for review in reviews:
        key = review['pair_id']
        if key in seen or key not in by_id:
            raise ValueError('Unknown or duplicate reviewed pair')
        seen.add(key)
        row = by_id[key]
        if row['split'] != 'train':
            raise ValueError('Threshold review must use train candidates only')
        if any(review.get(k) != row[k] for k in ('query_id', 'reference_id')):
            raise ValueError('Reviewed pair identity mismatch')
        label = review.get('compatible')
        if label is not None and type(label) is not bool:
            raise ValueError('compatible must be true, false, or null')
        if not review.get('reviewer') or not review.get('evidence'):
            raise ValueError('Record reviewer and visible evidence, including uncertainty')
        if label is None:
            unknown += 1
        else:
            labeled.append((row, label))
    positives = sum(label for _, label in labeled)
    negatives = len(labeled) - positives
    grid = []
    for candidate in threshold_grid(rows):
        def accepted(row):
            return ((candidate['proprio_rms'] is None or row['proprio_rms'] <= candidate['proprio_rms'])
                    and (candidate['image_cosine'] is None or row['image_worst'] <= candidate['image_cosine']))
        pos = sum(accepted(row) for row, label in labeled if label)
        neg = sum(accepted(row) for row, label in labeled if not label)
        grid.append(dict(candidate, reviewed_compatible_accepted=pos,
                         reviewed_incompatible_accepted=neg,
                         compatible_retention=pos / positives if positives else None,
                         incompatible_acceptance=neg / negatives if negatives else None))
    enough = positives >= 5 and negatives >= 5
    return dict(reviewed_pairs=len(seen), compatible=positives, incompatible=negatives,
                unknown=unknown, unreviewed_pairs=len(rows) - len(seen),
                calibration_status='manual_threshold_assessment_required' if enough else 'insufficient_reviewed_classes',
                selected_thresholds=None, threshold_grid=grid,
                limitations=['Start-observation review is not action-transfer or LIBERO success ground truth.',
                             'Rank-stratified, correlated candidate pairs do not estimate population accuracy.',
                             'Unknown reviews are excluded from compatible/incompatible rates.',
                             'No held-out threshold validation or formal pool admission was performed.'],
                training_started=False, accepted_pool_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    trial, review_path = Path(args.trial).resolve(), Path(args.reviews).resolve()
    report = json.loads((trial / 'report.json').read_text())
    if report.get('observation_policy') != 'policy_inputs_only_v1' or not report.get('pilot_only'):
        raise ValueError('Expected a public-input visual pilot')
    distances = trial / 'joint_distances.jsonl'
    rows = [json.loads(s) for s in distances.read_text().splitlines() if s.strip()]
    reviews = [json.loads(s) for s in review_path.read_text().splitlines() if s.strip()]
    result = audit_reviews(rows, reviews)
    result.update(observation_policy=report['observation_policy'],
                  encoder_sha256=report['feature_report']['encoder_sha256'],
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (trial / 'report.json', distances, review_path)})
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'threshold_grid'}, indent=2))


if __name__ == '__main__':
    main()
