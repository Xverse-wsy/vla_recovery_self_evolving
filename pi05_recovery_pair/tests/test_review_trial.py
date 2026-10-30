import pytest

from recovery_pair.review_trial import audit_reviews


def fixtures():
    rows = [dict(pair_id=str(i), query_id=str(i), reference_id='vla', split='train',
                 query_segment='segment', proprio_rms=.1, image_worst=.001) for i in range(3)]
    reviews = [dict(pair_id=r['pair_id'], query_id=r['query_id'], reference_id='vla',
                    compatible=label, reviewer='visible_input_review', evidence='starting images')
               for r, label in zip(rows, (False, None, False))]
    return rows, reviews


def test_unknown_review_does_not_become_negative_or_imply_calibration():
    rows, reviews = fixtures()
    report = audit_reviews(rows, reviews)
    assert (report['compatible'], report['incompatible'], report['unknown']) == (0, 2, 1)
    assert report['selected_thresholds'] is None
    assert report['calibration_status'] == 'insufficient_reviewed_classes'
    assert all(r['compatible_retention'] is None for r in report['threshold_grid'])
    unfiltered = report['threshold_grid'][-1]
    assert unfiltered['incompatible_acceptance'] == 1.


def test_threshold_review_rejects_leaked_or_mismatched_evidence():
    rows, reviews = fixtures()
    with pytest.raises(ValueError, match='duplicate'):
        audit_reviews(rows, reviews + reviews[:1])
    reviews[0]['reference_id'] = 'different'
    with pytest.raises(ValueError, match='identity'):
        audit_reviews(rows, reviews)
    rows, reviews = fixtures()
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='train'):
        audit_reviews(rows, reviews)
