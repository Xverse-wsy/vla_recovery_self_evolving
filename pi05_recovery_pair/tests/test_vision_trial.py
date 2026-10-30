import torch
import pytest

from recovery_pair.pi05_vision import pooled_unit_features
from recovery_pair.vision_trial import threshold_grid


def test_pooled_image_features_are_finite_unit_vectors():
    tokens=torch.tensor([[[1.,0.],[0.,1.]],[[0.,2.],[0.,4.]]])
    pooled=pooled_unit_features(tokens)
    torch.testing.assert_close(pooled.norm(dim=-1),torch.ones(2))
    torch.testing.assert_close(pooled[1],torch.tensor([0.,1.]))
    with pytest.raises(ValueError):
        pooled_unit_features(torch.zeros(1,2,3))


def test_joint_threshold_counts_windows_once_and_preserves_missing_review_status():
    rows=[dict(query_id='a',query_segment='s',proprio_rms=.1,image_worst=.01),
          dict(query_id='a',query_segment='s',proprio_rms=.15,image_worst=.02),
          dict(query_id='b',query_segment='t',proprio_rms=.4,image_worst=.02)]
    grid=threshold_grid(rows)
    result=next(r for r in grid if r['proprio_rms']==.2 and r['image_cosine']==.02)
    assert result['candidate_pairs']==2 and result['cap_windows']==1 and result['cap_segments']==1
    unfiltered=next(r for r in grid if r['proprio_rms'] is None and r['image_cosine'] is None)
    assert unfiltered['candidate_pairs']==3 and unfiltered['cap_windows']==2
