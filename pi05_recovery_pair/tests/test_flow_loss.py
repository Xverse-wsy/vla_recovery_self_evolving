import inspect
import pytest
import torch
from recovery_pair.flow import differentiable_sample,fm_prediction
from recovery_pair.toy import ToyFlow
from recovery_pair.losses import pair_loss,distance_per_window


@pytest.mark.parametrize('steps',[1,3,7])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_deployed_sampler_parity_and_every_step_gradient(steps,checkpoint):
    torch.manual_seed(31)
    backend=ToyFlow(steps=steps).double()
    obs={'state':torch.randn(2,8,dtype=torch.float64)}
    noise=torch.randn(2,10,32,dtype=torch.float64)
    expected=backend.deployment_sample(obs,noise)
    out,chain=differentiable_sample(backend,backend.prepare_condition(obs),noise,steps,
        checkpoint_steps=checkpoint,return_chain=True)
    torch.testing.assert_close(out,expected,atol=0,rtol=0)
    out.square().mean().backward()
    assert len(chain)==steps+1
    assert all(x.grad is not None and x.grad.abs().sum()>0 for x in chain[1:])
    assert all(p.grad is None for p in backend.frozen_prefix.parameters())
    assert backend.expert.weight.grad.abs().sum()>0


def test_finite_difference_full_sampler():
    torch.manual_seed(4)
    backend=ToyFlow(horizon=4,action_dim=3,steps=3).double()
    condition=backend.prepare_condition({'state':torch.randn(1,8,dtype=torch.float64)})
    noise=torch.randn(1,4,3,dtype=torch.float64)
    def objective():
        return differentiable_sample(backend,condition,noise,3).square().mean()
    objective().backward()
    automatic=backend.expert.weight.grad[0,0].item()
    with torch.no_grad():
        original=backend.expert.weight[0,0].item()
        backend.expert.weight[0,0]=original+1e-5
    plus=objective().item()
    with torch.no_grad():
        backend.expert.weight[0,0]=original-1e-5
    minus=objective().item()
    assert automatic==pytest.approx((plus-minus)/2e-5,rel=1e-5,abs=1e-7)


def test_records_cannot_initialize_sampler():
    assert not {'positive','negative','actions','target'} & inspect.signature(differentiable_sample).parameters.keys()
    backend=ToyFlow()
    obs={'state':torch.zeros(1,8)}
    noise=torch.randn(1,10,32)
    out,chain=differentiable_sample(backend,backend.prepare_condition(obs),noise,3,return_chain=True)
    torch.testing.assert_close(chain[0],noise)
    positive=torch.full_like(noise,100.)
    negative=torch.full_like(noise,-100.)
    pair_loss(out,positive,negative,torch.ones_like(noise,dtype=torch.bool))[0].backward()
    torch.testing.assert_close(chain[0],noise)
    with torch.no_grad(),pytest.raises(RuntimeError):
        differentiable_sample(backend,backend.prepare_condition(obs),noise,3)


def test_invalid_values_empty_mask_and_window_mean():
    x=torch.tensor([[[2.,float('nan')],[4.,float('inf')]],[[6.,0.],[7.,0.]]],requires_grad=True)
    y=torch.zeros_like(x)
    mask=torch.tensor([[[True,False],[True,False]],[[True,False],[False,False]]])
    values,valid=distance_per_window(x,y,mask)
    torch.testing.assert_close(values,torch.tensor([10.,36.]))
    values.mean().backward()
    assert torch.isfinite(x.grad).all()
    empty=torch.zeros((0,10,32),requires_grad=True)
    loss,_=pair_loss(empty,empty,empty,empty.bool())
    assert loss.item()==0
    loss.backward()


def test_continuous_repulsion_even_far_and_negative_loss():
    generated=torch.tensor([[[0.]]],requires_grad=True)
    positive=torch.zeros_like(generated)
    negative=torch.full_like(generated,-100.)
    loss,_=pair_loss(generated,positive,negative,torch.ones_like(generated,dtype=torch.bool),.1)
    assert loss.item()<0
    loss.backward()
    assert generated.grad.item()==pytest.approx(-20.)
    for lam in [0.,1.,1.1]:
        with pytest.raises(ValueError):
            pair_loss(generated,positive,negative,torch.ones_like(generated,dtype=torch.bool),lam)


def test_fm_uses_own_state_and_time_convention():
    backend=ToyFlow()
    obs={'state':torch.ones(2,8)}
    actions=torch.randn(2,10,32)
    noise=torch.randn_like(actions)
    time=torch.tensor([.2,.7])
    v,u=fm_prediction(backend,obs,actions,noise=noise,time=time)
    torch.testing.assert_close(u,noise-actions)
    expected=backend.velocity(backend.prepare_condition(obs),(1-time[:,None,None])*actions+time[:,None,None]*noise,time)
    torch.testing.assert_close(v,expected)
