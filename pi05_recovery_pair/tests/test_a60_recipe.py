from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from recovery_pair.a60_recipe import RECIPE, full_sft_mask, build_optimizer_and_scheduler, validate_training_settings
from recovery_pair.config import Config
from recovery_pair.engine import accumulate_update
from recovery_pair.openpi_backend import OpenPiBackend
from recovery_pair.synthetic import loader


def native(config):
    return replace(config,training_recipe=RECIPE,pair_batch=0,eta=0,
        fm_epochs=50,learning_rate=5e-6,flow_steps=5,fm_batch=2)


def test_native_recipe_checks_shared_model_and_optimizer_defaults(config):
    pytest.importorskip('omegaconf')
    cfg=native(config)
    report=validate_training_settings(cfg)
    assert report['full_model_trainable'] and report['training_preprocessing']
    assert report['optimizer']['adam_beta2']==.95
    for key,value in [('flow_steps',3),('learning_rate',1e-5),('seed',43)]:
        with pytest.raises(ValueError,match=key):
            validate_training_settings(replace(cfg,**{key:value}))


def test_legacy_config_identity_is_preserved_and_native_is_explicit(config):
    assert 'training_recipe' not in config.to_dict()
    assert native(config).validate().to_dict()['training_recipe']==RECIPE
    with pytest.raises(ValueError,match='positive-only'):
        replace(native(config),pair_batch=1).validate()


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img=torch.nn.Linear(2,2)
        self.action_out_proj=torch.nn.Linear(2,2)
        self.pi05=True
        self.action_horizon=10
        self.action_dim=32
        self.rlt_cfg=type('Rlt',(),{'use_rlt':False})()
        self.freeze_calls=0
    def freeze_vlm(self,freeze_action_expert=False):
        self.freeze_calls+=1
        self.img.requires_grad_(False)
    def build_prefix_cache(self,*a,**kw):pass
    def run_suffix(self,*a,**kw):pass
    def velocity_from_suffix(self,*a,**kw):pass
    def sample_actions(self,*a,**kw):pass
    def compute_loss(self,observation,actions,*,train,noise,time):
        assert train
        return (self.action_out_proj(self.img(observation))-actions).square()


def test_native_backend_trains_vision_and_uses_training_preprocess():
    policy=Policy()
    backend=OpenPiBackend(policy,full_model_sft=True)
    assert not policy.freeze_calls and policy.training and backend.fm_augmentation
    loss=backend.fm_loss_elements(torch.ones(1,2),torch.zeros(1,2),noise=None,time=None).mean()
    loss.backward()
    assert policy.img.weight.grad is not None and policy.img.weight.grad.abs().sum()>0
    legacy=OpenPiBackend(Policy())
    assert legacy.policy.freeze_calls==1 and not legacy.policy.training
    assert not any(p.requires_grad for p in legacy.policy.img.parameters())


def test_native_mask_rejects_short_vla_windows(config):
    mask=torch.zeros(1,10,32,dtype=torch.bool);mask[:,:,:7]=True
    assert full_sft_mask(mask,native(config)).all()
    mask[:,-1,:7]=False
    with pytest.raises(ValueError,match='full legal'):
        full_sft_mask(mask,native(config))


def test_native_loss_includes_padding_dims_and_matches_accumulation(window,config):
    class PaddingLoss(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def sample_flow_time(self,batch,device):return torch.full((batch,),.5,device=device)
        def fm_loss_elements(self,observation,target,*,noise,time):
            weights=torch.zeros_like(target);weights[:,:,7:]=1
            return weights*self.weight.square()
    records=[window('a'),window('b')];plan=dict(fm_ids=['a','b'],pairs=[])
    cfg=native(config);a=PaddingLoss();b=copy.deepcopy(a)
    metrics=accumulate_update(a,loader,records,plan,cfg,update_index=0,total_updates=50,mode='fm_only')
    metrics_b=accumulate_update(b,loader,records,plan,replace(cfg,fm_micro_batch=2),update_index=0,total_updates=50,mode='fm_only')
    assert metrics['fm']==pytest.approx(25/32)
    assert metrics_b['fm']==pytest.approx(metrics['fm'])
    assert a.weight.grad.item()==pytest.approx(50/32)
    torch.testing.assert_close(a.weight.grad,b.weight.grad)
    legacy=PaddingLoss()
    metrics=accumulate_update(legacy,loader,records,plan,
        replace(cfg,training_recipe='legacy_frozen_masked_v1'),update_index=0,total_updates=50,mode='fm_only')
    assert metrics['fm']==0


def test_actual_rlinf_optimizer_schedule_and_resume(config):
    pytest.importorskip('rlinf')
    pytest.importorskip('omegaconf')
    cfg=native(config)
    start=torch.nn.Linear(2,1)
    a=copy.deepcopy(start);b=copy.deepcopy(start)
    opt,sch,recipe=build_optimizer_and_scheduler(a,cfg,200)
    assert opt.param_groups[0]['betas']==(.9,.95)
    assert recipe['lr_warmup_steps']==100 and recipe['min_lr']==5e-7
    # The pinned openpi_cosine schedule starts at peak/(warmup+1).
    assert opt.param_groups[0]['lr']==pytest.approx(5e-6/101)
    def step(model,optimizer,scheduler):
        optimizer.zero_grad();model(torch.ones(1,2)).square().mean().backward()
        optimizer.step();scheduler.step()
    for _ in range(110):step(a,opt,sch)
    saved=copy.deepcopy((a.state_dict(),opt.state_dict(),sch.state_dict()))
    step(a,opt,sch)
    opt_b,sch_b,_=build_optimizer_and_scheduler(b,cfg,200)
    b.load_state_dict(saved[0]);opt_b.load_state_dict(saved[1]);sch_b.load_state_dict(saved[2])
    step(b,opt_b,sch_b)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert opt.param_groups[0]['lr']==opt_b.param_groups[0]['lr']
    for _ in range(89):step(a,opt,sch)
    assert opt.param_groups[0]['lr']==pytest.approx(5e-7)


def test_native_trainer_restores_optimizer_scheduler_and_rng(window, config, tmp_path):
    pytest.importorskip('rlinf')
    pytest.importorskip('omegaconf')
    import json
    from recovery_pair.positive_fm import prepare_positive_fm
    from recovery_pair.trainer import train
    from recovery_pair.toy import ToyFlow
    from test_positive_fm import source, admission

    cfg=native(config)
    records=[source(window,'native'+str(i)) for i in range(5)]
    prepared=tmp_path/'prepared'
    prepare_positive_fm(admission(tmp_path,records,cfg),cfg,prepared,dense=False,all_train=True)

    class StochasticFM(ToyFlow):
        def fm_loss_elements(self, observation, target, *, noise, time):
            # Exercise RNG restoration as well as the scheduled optimizer.
            feature=torch.rand_like(target)
            return (self.expert(torch.cat([target,feature[:,:,:5]],-1))-noise).square()

    torch.manual_seed(19)
    initial=StochasticFM()
    def run(model, path, **kwargs):
        return train(model,loader,prepared,cfg,path,mode='fm_only',save_every=1,keep_last=3,**kwargs)
    torch.manual_seed(37)
    full=copy.deepcopy(initial)
    run(full,tmp_path/'full',max_updates=4)
    torch.manual_seed(37)
    run(copy.deepcopy(initial),tmp_path/'split',max_updates=2)
    saved=torch.load(tmp_path/'split/update_000002.pt',weights_only=False)
    assert 'lr_scheduler' in saved
    assert saved['identity']['objective_version']=='native_sft_full_action_mean_a60_v1'
    assert saved['identity']['optimizer_recipe']['adam_beta2']==.95
    restored=copy.deepcopy(initial)
    run(restored,tmp_path/'split',resume=tmp_path/'split/update_000002.pt',max_updates=4)
    for p,q in zip(full.parameters(),restored.parameters()):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert (tmp_path/'full/metrics.jsonl').read_text()==(tmp_path/'split/metrics.jsonl').read_text()
    logs=[json.loads(x) for x in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert logs[0]['learning_rate_used']==pytest.approx(5e-6/101)
    assert logs[1]['learning_rate_used']>logs[0]['learning_rate_used']
