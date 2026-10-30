from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from recovery_pair.config import Config,MatchThresholds
from recovery_pair.features import extract_features
from recovery_pair.openpi_backend import ObservationLoader
from recovery_pair.data import QuantileAssets


class FrozenMean(torch.nn.Module):
    def forward(self,x):
        return x.mean(dim=(2,3))


def test_feature_cache_is_frozen_multiview_and_no_image_rotation(window,tmp_path):
    image=np.zeros((8,8,3),np.uint8)
    image[0,0]=[255,20,30]
    path=tmp_path/'image.png'
    Image.fromarray(image).save(path)
    record=window('r')
    record.image_refs={'front':str(path),'wrist':str(path),'oracle_segmentation':'must-not-open'}
    encoder=tmp_path/'encoder.pt'
    torch.jit.script(FrozenMean()).save(str(encoder))
    spec=tmp_path/'preprocess.json'
    spec.write_text(json.dumps(dict(size_wh=[8,8],mean=[0.,0.,0.],std=[1.,1.,1.],resize_mode='bilinear_stretch')))
    extract_features([record],encoder,spec,tmp_path/'cache')
    assert record.visual_features['views']['front']==record.visual_features['views']['wrist']
    first=record.visual_features.copy()
    assert set(first['views'])=={'front','wrist'}
    extract_features([record],encoder,spec,tmp_path/'cache')
    assert record.visual_features==first
    from recovery_pair.features import read_start_image
    np.testing.assert_array_equal(read_start_image(str(path)),image)


def test_native_env_keys_and_no_label_or_future_inputs(window,tmp_path):
    path=tmp_path/'rgb.png'
    Image.new('RGB',(8,8)).save(path)
    record=window('r')
    record.image_refs={'front':str(path),'wrist':str(path)}
    class Policy:
        def env_obs_to_observation(self,env):
            assert set(env)=={'states','main_images','wrist_images','task_descriptions'}
            assert env['task_descriptions']==[record.instruction]
            return env
    result=ObservationLoader(SimpleNamespace(policy=Policy()))([record])
    assert result['main_images'].shape==(1,8,8,3)


def test_quantile_normalization_keeps_out_of_range_values(tmp_path):
    path=tmp_path/'norm.json'
    path.write_text(json.dumps({'norm_stats':{key:{'q01':[-2.,0.],'q99':[2.,1.]} for key in ['state','actions']}}))
    normalizer=QuantileAssets(path)
    raw=np.array([[8.,1.5]],np.float32)
    expected=(raw-np.array([-2.,0.]))/(np.array([4.,1.])+1e-6)*2-1
    np.testing.assert_allclose(normalizer.normalize('actions',raw),expected,rtol=1e-6)
    assert (normalizer.normalize('actions',raw)>1).all()


@pytest.mark.parametrize('lam',[1.,1.01,0.,-1.])
def test_configuration_rejects_unbounded_pair_coefficients(lam):
    with pytest.raises(ValueError):
        replace(Config(),lambda_push=lam).validate()


def test_pairing_must_be_at_least_as_strict_as_coverage(config):
    with pytest.raises(ValueError,match='wider'):
        replace(config,pairing=replace(config.pairing,position_m=.04)).validate(matching_ready=True)


def test_legacy_progress_thresholds_are_ignored_and_not_reexported(config,tmp_path):
    legacy=config.to_dict()
    legacy['coverage']['progress']=0.01
    legacy['pairing']['progress']=0.9
    legacy['coverage']['object_position_m']=.01
    legacy['pairing']['object_position_m']=.9
    path=tmp_path/'old_config.json'
    path.write_text(json.dumps(legacy))
    loaded=Config.load(path).validate(matching_ready=True)
    assert loaded==config
    assert all('progress' not in loaded.to_dict()[k] and 'object_position_m' not in loaded.to_dict()[k]
               for k in ('coverage','pairing'))
