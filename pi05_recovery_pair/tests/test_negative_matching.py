import numpy as np
import pytest

from recovery_pair.matching import FrozenMatcher
from recovery_pair.negative_matching import evaluate, numeric_candidates


def test_negative_report_separates_state_matches_from_trainable_pairs(window, config):
    positive = window('positive')
    good = window('good', quality='negative')
    short = window('short', quality='negative')
    payload = short.payload()
    payload['valid_mask'][1:] = False
    payload['trusted_error_mask'][1:] = False
    payload['actions'][1:] = 0
    np.savez(short.action_path, **payload)
    same = window('same', quality='negative', value=.5)
    far = window('far', quality='negative', position=.2)
    cap_failure = window('cap_failure', 'code', 'negative')
    matcher = FrozenMatcher([positive, good, short, same, far, cap_failure], config)
    pairs, rows, report = evaluate(matcher)
    assert report['negative_query_windows'] == 4
    assert report['numerical_match_windows'] == report['visual_match_windows'] == 3
    assert report['paired_negative_windows'] == report['paired_negative_episodes'] == 1
    assert pairs[0]['negative_id'] == 'good'
    assert {r['negative_id'] for r in rows} == {'good', 'short', 'same', 'far'}
    assert report['action_rejections']['insufficient_trusted_overlap'] == 2
    assert report['positive_references_by_controller'] == {'VLA': 1}


def test_missing_visual_features_cannot_be_reported_as_zero_matches(window, config):
    positive, negative = window('positive'), window('negative', quality='negative')
    matcher = FrozenMatcher([positive, negative], config)
    assert len(numeric_candidates(matcher, negative)) == 1
    negative.visual_features = {}
    with pytest.raises(ValueError, match='Missing candidate features'):
        evaluate(matcher)


def test_no_numerical_candidates_returns_zero_without_image_features(window, config):
    negative = window('negative', quality='negative', position=1.)
    negative.visual_features = {}
    matcher = FrozenMatcher([window('positive'), negative], config)
    pairs, _, report = evaluate(matcher)
    assert pairs == [] and report['numerical_relations'] == report['paired_negative_episodes'] == 0
