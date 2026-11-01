from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from recovery_pair.config import MatchThresholds
from recovery_pair.features import sha256
from recovery_pair.schema import save_windows
from recovery_pair.stage2_data import SELECTED, prepare, audit
from recovery_pair.stage2_matching import run
from recovery_pair.stage2_sampler import Stage2Sampler
from recovery_pair.stage2_train import check_stage1, train_loop, apply_trainable
from recovery_pair.toy import ToyFlow
from test_dense_fm import dense_fixture
from test_expert_fm import expert_fixture
from test_stage2_matching import negative_fixture, feature
from recovery_pair.positive_fm import prepare_positive_fm


def selected(config):
    return replace(config,pairing=MatchThresholds(**SELECTED),fm_batch=8,pair_batch=2)


def test_prepare_pairs_keeps_fm_repeats_but_real_only_refs_and_sources_immutable(tmp_path,config):
    c,admission,norm=dense_fixture(tmp_path,selected(config))
    stage1=tmp_path/'stage1'
    experts=expert_fixture(tmp_path)
    prepare_positive_fm(admission,c,stage1,all_train=True,expert_dataset=experts,expected_expert_episodes=2)
    sparse=negative_fixture(tmp_path,c,norm)
    index=tmp_path/'negative.jsonl';save_windows(index,sparse)
    class Encoder:
        def __init__(self,*args):pass
        def extract(self,records,*args):
            for r in records:feature(r)
            return {'synthetic_only':True}
    matching=tmp_path/'matching'
    run(stage1,index,'unused',c,matching,device='cpu',encoder_factory=Encoder,expected_expert_episodes=2)
    before={str(p):sha256(p) for root in [stage1,matching] for p in root.rglob('*') if p.is_file()}
    out=tmp_path/'stage2'
    report=prepare(stage1,matching,c,out)
    records,pairs,fm_ids,_=audit(out,c)
    assert report['negative_windows']==len(pairs)>0
    assert any(r.controller=='expert' for r in records if r.sample_id in fm_ids)
    assert any(r.payload().get('real_action_mask',r.payload()['valid_mask']).sum()<r.payload()['valid_mask'].sum()
               for r in records if r.sample_id in fm_ids)
    refs=[r for r in records if r.source['stage2_training_role']=='reference']
    assert refs and all(not r.payload()['valid_mask'][r.source['fm_real_steps']:].any() for r in refs)
    assert {p:sha256(p) for p in before}==before
    victim=records[0]
    with open(victim.action_path,'ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='payload changed'):audit(out,c)


def simple_pool(window,config):
    c=selected(config)
    p=window('p');p.controller='expert';p.bank_class='D_expert_positive';p.source={'expert_training_only':True}
    code=window('code',controller='code')
    records=[p,code]
    pairs=[]
    for episode,number in [('one',5),('two',1)]:
        for j in range(number):
            n=window(f'{episode}{j}',quality='negative',episode=episode)
            records.append(n)
            pairs.append(dict(pair_id=n.sample_id,negative_id=n.sample_id,negative_episode_id=episode,
                event_id=n.event_id,match_inherited=False,references=[dict(sample_id=p.sample_id,mask=n.payload()['valid_mask'].tolist())]))
    return c,records,pairs,[p.sample_id,code.sample_id]


def test_sampler_balances_episodes_and_resume_even_with_unequal_window_counts(window,config):
    c,records,pairs,fm=simple_pool(window,config)
    c=replace(c,pair_batch=4,max_episode_reuses=5)
    sampler=Stage2Sampler(records,pairs,fm,c)
    first=sampler.draw()
    assert len(first['pairs'])==2 and len(first['fm_ids'])==8
    resumed=Stage2Sampler(records,pairs,fm,c);resumed.load_state_dict(sampler.state_dict())
    while not sampler.done:
        assert sampler.draw()==resumed.draw()
    assert sampler.visits=={'one':5,'two':5} and sampler.updates==sampler.nominal_updates==5
    assert resumed.done


def test_empty_pair_pool_falls_back_to_fm(window,config):
    c,records,_,fm=simple_pool(window,config)
    sampler=Stage2Sampler(records,[],fm,replace(c,fm_only_updates=2))
    for _ in range(2):
        plan=sampler.draw();assert not plan['pairs'] and len(plan['fm_ids'])==8
    assert sampler.done


def test_stage_boundary_rejects_partial_or_wrong_data_and_accepts_relocated_identical_base():
    base=dict(weights_sha256='weights',norm_sha256='norm',checkpoint='/new/base')
    state=dict(update=2,identity=dict(mode='fm_only',objective_version='native_fm_masked_window_mean_v1',
        base=dict(base,checkpoint='/old/base'),config={'fm_epochs':50},prepared_sha256='data'),sampler={'epochs_completed':0})
    with pytest.raises(ValueError,match='incomplete'):check_stage1(state,base)
    assert check_stage1(state,base,allow_partial=True,max_updates=2,prepared_sha='data')['partial_stage1_smoke']
    with pytest.raises(ValueError,match='different prepared'):check_stage1(state,base,allow_partial=True,max_updates=1,prepared_sha='other')
    state['sampler']['epochs_completed']=50
    assert not check_stage1(state,base,prepared_sha='data')['partial_stage1_smoke']


def test_export_refuses_to_write_into_original_stage1_directory(tmp_path,monkeypatch):
    from recovery_pair import stage2_train
    original=tmp_path/'stage1';original.mkdir()
    base={'checkpoint':str(tmp_path/'base'),'norm_stats':'norm'}
    identity=dict(objective_version=stage2_train.OBJECTIVE,base=base,
                  stage1_checkpoint=str(original/'update.pt'))
    checkpoint=tmp_path/'stage2.pt';torch.save({'identity':identity},checkpoint)
    monkeypatch.setattr(stage2_train,'base_identity',lambda *_:base)
    with pytest.raises(ValueError,match='outside the original'):
        stage2_train.export(checkpoint,original/'exported')
    assert list(original.iterdir())==[]


def test_stage2_real_optimizer_resume_matches_continuous_toy_and_frozen_weights(window,config,tmp_path):
    c,records,pairs,fm=simple_pool(window,config)
    c=replace(c,max_episode_reuses=3)
    def model():
        torch.manual_seed(42);return ToyFlow()
    def loader(rows):return {'state':torch.tensor([r.proprio for r in rows],dtype=torch.float32)}
    identity={'test':'isolated-stage2'}
    short=tmp_path/'short';short.mkdir()
    backend=model();frozen=backend.frozen_prefix.weight.detach().clone()
    train_loop(backend,loader,records,pairs,fm,c,short,identity,max_updates=1,save_every=1)
    checkpoint=short/'update_000001.pt'
    resumed=model()
    result=train_loop(resumed,loader,records,pairs,fm,c,short,identity,resume=checkpoint,max_updates=2,save_every=1)
    continuous=tmp_path/'continuous';continuous.mkdir()
    baseline=model()
    train_loop(baseline,loader,records,pairs,fm,c,continuous,identity,max_updates=2,save_every=1)
    for a,b in zip(resumed.parameters(),baseline.parameters()):assert torch.equal(a,b)
    assert torch.equal(resumed.frozen_prefix.weight,frozen)
    rows=[json.loads(s) for s in (short/'metrics.jsonl').read_text().splitlines()]
    ref=[json.loads(s) for s in (continuous/'metrics.jsonl').read_text().splitlines()]
    assert rows==ref and all(r['pair_count']==2 and r['fm_count']==8 for r in rows)
    with pytest.raises(ValueError,match='inventory'):apply_trainable(resumed,{})
