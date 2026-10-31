"""Exact positive epochs, deterministic cursors, tail weights and restart parity."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from recovery_pair.batches import PositiveEpochSampler, training_sampler
from recovery_pair.engine import accumulate_update
from recovery_pair.positive_fm import prepare_positive_fm
from recovery_pair.synthetic import loader
from recovery_pair.toy import ToyFlow
from recovery_pair.trainer import train
from test_positive_fm import source, admission
from test_training import _distributed_worker


def pool(n):
    return [SimpleNamespace(sample_id=str(i), split='train', training_quality='positive',
                            controller=['expert','VLA','code'][i%3]) for i in range(n)]


@pytest.mark.parametrize('n,batch', [(1,64),(4,4),(7,4),(16324,64)])
def test_fifty_epochs_visit_each_window_once_without_dropped_or_padded_tails(config,n,batch):
    c=replace(config,fm_epochs=50,pair_batch=0,eta=0,fm_batch=batch)
    records=pool(n)
    s=training_sampler(records,[],c,mode='fm_only')
    report=s.budget_report()
    assert report['expected_updates']==((n+batch-1)//batch)*50
    assert report['expected_samples']==n*50
    if n==16324:
        assert report['expected_updates']==12800 and report['expected_samples']==816200
        assert report['tail_batch_size']==4
    for epoch in range(1,51):
        seen=[]
        while True:
            p=s.draw()
            assert p['epoch']==epoch and not p['pairs']
            assert 0<len(p['fm_ids'])<=batch
            seen.extend(p['fm_ids'])
            if p['epoch_complete']:
                assert len(p['fm_ids'])==(n-1)%batch+1
                break
        assert len(seen)==n and set(seen)=={r.sample_id for r in records}
        assert s.samples_seen==epoch*n
    assert s.done and s.updates==report['expected_updates']
    with pytest.raises(StopIteration): s.draw()


@pytest.mark.parametrize('draws',[0,1,2,3,6])
def test_restart_reconstructs_same_order_at_middle_boundary_and_completion(config,draws):
    c=replace(config,fm_epochs=3,pair_batch=0,eta=0,fm_batch=4)
    s=PositiveEpochSampler(pool(7),[],c)
    for _ in range(draws): s.draw()
    resumed=PositiveEpochSampler(list(reversed(pool(7))),[],c)
    resumed.load_state_dict(s.state_dict())
    while not s.done: assert resumed.draw()==s.draw()
    assert resumed.state_dict()==s.state_dict()


def test_epoch_config_and_checkpoint_fail_closed(config):
    c=replace(config,fm_epochs=2,pair_batch=0,eta=0,fm_batch=4)
    s=PositiveEpochSampler(pool(7),[],c)
    s.draw()
    state=s.state_dict()
    for key in ('inventory','policy','seed','fm_batch','fm_epochs','updates','samples_seen','position','epochs_completed'):
        bad=dict(state)
        bad[key]=999 if isinstance(bad[key],int) else 'wrong'
        with pytest.raises(ValueError): PositiveEpochSampler(pool(7),[],c).load_state_dict(bad)
    for epochs in (0,-1,True,1.5):
        with pytest.raises(ValueError): replace(c,fm_epochs=epochs).validate()
    with pytest.raises(ValueError): replace(c,pair_batch=1).validate()
    with pytest.raises(ValueError): training_sampler(pool(7),[],c,mode='continuous_pair')
    with pytest.raises(ValueError): PositiveEpochSampler(pool(7),[{}],c)
    heldout=pool(7);heldout[0].split='validation'
    with pytest.raises(ValueError,match='all-train'): PositiveEpochSampler(heldout,[],c)


def test_full_epoch_training_and_resume_are_exact(window,config,tmp_path):
    c=replace(config,fm_epochs=3,pair_batch=0,eta=0,fm_batch=4)
    records=[source(window,'v'+str(i)) for i in range(7)]
    prepared=tmp_path/'prepared'
    report=prepare_positive_fm(admission(tmp_path,records,c),c,prepared,dense=False,all_train=True)
    assert report['training_budget']['expected_updates']==6
    torch.manual_seed(17)
    initial=ToyFlow()
    full=copy.deepcopy(initial)
    result=train(full,loader,prepared,c,tmp_path/'full',mode='fm_only',save_every=1,keep_last=3)
    assert result['reason']=='epochs_complete' and result['epochs_completed']==3
    assert result['samples_seen']==21 and result['updates']==6 and not result['validation_enabled']
    assert len(list((tmp_path/'full').glob('update_*.pt')))==3
    train(copy.deepcopy(initial),loader,prepared,c,tmp_path/'split',mode='fm_only',
          max_updates=3,save_every=1,keep_last=3)
    restored=copy.deepcopy(initial)
    resumed=train(restored,loader,prepared,c,tmp_path/'split',mode='fm_only',
          resume=tmp_path/'split/update_000003.pt',save_every=1,keep_last=3)
    assert resumed==result
    for a,b in zip(full.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert (tmp_path/'full/metrics.jsonl').read_text()==(tmp_path/'split/metrics.jsonl').read_text()
    metrics=[json.loads(x) for x in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert [m['fm_count'] for m in metrics]==[4,3]*3
    assert metrics[-1]['epochs_completed']==3


def test_epoch_preparation_rejects_heldout_recipe_before_writing(window,config,tmp_path):
    c=replace(config,fm_epochs=50,pair_batch=0,eta=0)
    previous=admission(tmp_path,[source(window,'v')],c)
    with pytest.raises(ValueError,match='--all-train'):
        prepare_positive_fm(previous,c,tmp_path/'wrong',dense=False)
    assert not (tmp_path/'wrong').exists()


def test_tail_smaller_than_world_size_matches_true_global_mean(window,config,tmp_path):
    import torch.multiprocessing as mp
    c=replace(config,fm_epochs=1,pair_batch=0,eta=0,fm_batch=4)
    records=[window('tail')]
    plan=PositiveEpochSampler(records,[],c).draw()
    torch.manual_seed(50)
    model=ToyFlow()
    weights=copy.deepcopy(model.state_dict())
    accumulate_update(model,loader,records,plan,c,update_index=1,total_updates=20)
    output=tmp_path/'grads.pt'
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),str(output),records,plan,c,weights),nprocs=2,join=True)
    grads=torch.load(output,weights_only=True)
    for name,p in model.named_parameters():
        if p.requires_grad: torch.testing.assert_close(p.grad,grads[name],atol=1e-7,rtol=2e-5)
