from dataclasses import replace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from recovery_pair.config import Config,MatchThresholds
from recovery_pair.schema import ActionSpec,Window


@pytest.fixture
def config():
    return replace(Config(),coverage=MatchThresholds(proprio_rms=.5,image_cosine=.1),
        pairing=MatchThresholds(position_m=.01,rotation_deg=5.,proprio_rms=.2,image_cosine=.05), max_episode_reuses=3)


@pytest.fixture
def window(tmp_path,config):
    def make(sid,controller='VLA',quality='positive',*,value=None,episode=None,position=0.,split='train',event=None):
        spec=ActionSpec('normalized_osc',tuple('abcdefg')+tuple(f'pad{i}' for i in range(25)),
            tuple(['normalized']*32),'robot_base',20,'norm_v1',10,32,tuple([True]*7+[False]*25))
        value=(.5 if quality=='positive' else -.5) if value is None else value
        actions=np.full((10,32),value,np.float32)
        actions[:,7:]=0
        mask=np.zeros((10,32),bool)
        mask[:,:7]=True
        path=tmp_path/(sid+'.npz')
        np.savez(path,actions=actions,valid_mask=mask,trusted_error_mask=mask if quality=='negative' else np.zeros_like(mask))
        r=Window(sid,'task',episode or sid,'initial:'+str(episode or sid),event or 'event:'+sid,
            controller,quality,0,0.,'place block','approach','block',{'front':'a.png','wrist':'b.png'},
            [position]+[0.]*7,[position]+[0.]*7,[position,0.,0.],[0.,0.,0.,1.],'open','empty','none',.2,str(path),spec,
            split=split,quality_evidence=dict(verified_local_actions=True,negative_behavior_confirmed=quality=='negative'),
            visual_features=dict(encoder_sha256='frozen1',calibration_id='prep1',views={'front':[1.,0.],'wrist':[1.,0.]}),
            source=dict(critical_state_verified=True,progress_scorer_id='score1'))
        return r.validate()
    return make
