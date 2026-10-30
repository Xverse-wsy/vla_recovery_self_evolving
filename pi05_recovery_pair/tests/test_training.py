import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from recovery_pair.batches import EpisodeEventSampler
from recovery_pair.engine import accumulate_update
from recovery_pair.matching import FrozenMatcher
from recovery_pair.toy import ToyFlow
from recovery_pair.synthetic import loader,fixture
from recovery_pair.trainer import train


def make_plan(window,config):
    records=[window('p0'),window('p1','code'),window('n0',quality='negative'),window('n1',quality='negative')]
    matcher=FrozenMatcher(records,config)
    sampler=EpisodeEventSampler(records,matcher.build_pairs(),config)
    return records,sampler,sampler.draw()


def test_microbatch_accumulation_matches_logical_batch(window,config):
    config=replace(config,fm_batch=8,pair_batch=2)
    records,sampler,plan=make_plan(window,config)
    torch.manual_seed(5)
    a=ToyFlow()
    b=copy.deepcopy(a)
    metrics_a=accumulate_update(a,loader,records,plan,config,update_index=1,total_updates=20)
    metrics_b=accumulate_update(b,loader,records,plan,replace(config,fm_micro_batch=8,pair_micro_batch=2),
                               update_index=1,total_updates=20)
    assert metrics_a['loss']==pytest.approx(metrics_b['loss'],rel=1e-6)
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-7,rtol=2e-5)


def test_fm_only_empty_pair_and_missing_code_backfill(window,config):
    records=[window('p0')]
    config=replace(config,fm_batch=8,pair_batch=2)
    sampler=EpisodeEventSampler(records,[],config)
    plan=sampler.draw()
    assert plan['pairs']==[] and plan['fm_ids']==['p0']*8
    backend=ToyFlow()
    metrics=accumulate_update(backend,loader,records,plan,config,update_index=0,total_updates=10)
    assert metrics['pair']==0 and np.isfinite(metrics['loss'])


def test_invalid_recorded_fm_targets_cannot_change_prediction_or_loss(window,config):
    r=window('p0')
    config=replace(config,fm_batch=8,pair_batch=0)
    plan=EpisodeEventSampler([r],[],config).draw()
    a=ToyFlow()
    b=copy.deepcopy(a)
    before=accumulate_update(a,loader,[r],plan,config,update_index=1,total_updates=10)
    payload=r.payload()
    payload['actions'][:,7:]=1e6
    np.savez(r.action_path,**payload)
    after=accumulate_update(b,loader,[r],plan,config,update_index=1,total_updates=10)
    assert before['loss']==after['loss']
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)


def test_episode_budget_and_resume_independent_of_windows(window,config):
    records,sampler,plan=make_plan(window,replace(config,fm_batch=8,pair_batch=1))
    state=sampler.state_dict()
    resumed=EpisodeEventSampler(records,FrozenMatcher(records,config).build_pairs(),sampler.config)
    resumed.load_state_dict(state)
    assert sampler.draw()==resumed.draw()
    while not sampler.done:
        plan=sampler.draw()
        episodes=[p['episode_id'] for p in plan['pairs']]
        assert len(episodes)==len(set(episodes))
        assert max(plan['episode_visits'].values())<=config.max_episode_reuses
    assert set(sampler.visits.values())=={config.max_episode_reuses}


def test_end_to_end_training_checkpoint_resume_and_fixed_validation(tmp_path):
    config,records,report=fixture(tmp_path/'fixture')
    config=replace(config,fm_batch=8,pair_batch=2)
    model=ToyFlow()
    result=train(model,loader,tmp_path/'fixture/prepared',config,tmp_path/'run',
                 max_updates=1,save_every=1,validate_every=1)
    assert result['updates']==1 and result['validation_enabled']
    restored=ToyFlow()
    # Frozen parameters belong to base checkpoint, and are not reinitialized on resume.
    restored.frozen_prefix.load_state_dict(model.frozen_prefix.state_dict())
    result2=train(restored,loader,tmp_path/'fixture/prepared',config,tmp_path/'run',
        resume=tmp_path/'run/update_000001.pt',max_updates=2,save_every=1,validate_every=1)
    assert result2['updates']==2
    checkpoint=torch.load(tmp_path/'run/update_000002.pt',weights_only=False)
    assert all('frozen_prefix' not in key for key in checkpoint['trainable'])
    assert checkpoint['sampler']['updates']==2


def _distributed_worker(rank, rendezvous, output, records, plan, config, weights):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    model=ToyFlow()
    model.load_state_dict(weights)
    accumulate_update(model,loader,records,plan,config,update_index=1,total_updates=20)
    if rank==0:
        torch.save({k:p.grad for k,p in model.named_parameters() if p.requires_grad},output)
    dist.destroy_process_group()


def test_two_rank_sum_reduction_matches_global_mean_even_rank_without_pairs(window,config,tmp_path):
    import torch.multiprocessing as mp
    config=replace(config,fm_batch=6,pair_batch=1)
    records,sampler,plan=make_plan(window,config)
    torch.manual_seed(50)
    model=ToyFlow()
    weights=copy.deepcopy(model.state_dict())
    accumulate_update(model,loader,records,plan,config,update_index=1,total_updates=20)
    output=tmp_path/'gradients.pt'
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),str(output),records,plan,config,weights),
             nprocs=2,join=True)
    grads=torch.load(output,weights_only=True)
    for name,p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad,grads[name],atol=1e-7,rtol=2e-5)
