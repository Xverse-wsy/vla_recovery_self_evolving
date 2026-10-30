from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from recovery_pair.checkpoint_io import restore_full_checkpoint_after_dtype_setup


def fake_checkpoint_api(monkeypatch, resolver, loader):
    path='rlinf.models.embodiment.openpi.checkpoint'
    for i in range(1,len(path.split('.'))+1):
        name='.'.join(path.split('.')[:i])
        mod=ModuleType(name)
        mod.__path__=[]
        monkeypatch.setitem(sys.modules,name,mod)
    module=sys.modules[path]
    module.resolve_full_weights=resolver
    module.load_full_weights=loader


def test_full_checkpoint_preserves_fp32_training_updates_after_upstream_bf16_rounding(tmp_path,monkeypatch):
    model=torch.nn.Linear(2,1,bias=False)
    model.rlt_cfg=SimpleNamespace(use_rlt=False)
    trained=torch.tensor([[1.00013, .50017]],dtype=torch.float32)
    model.weight.data.copy_(trained.to(torch.bfloat16).float())
    assert not torch.equal(model.weight,trained)
    path=tmp_path/'full_weights.pt'
    torch.save({'weight':trained},path)
    def load(policy,weights,*,expect_rlt):
        assert not expect_rlt
        policy.load_state_dict(torch.load(weights,weights_only=True))
    fake_checkpoint_api(monkeypatch,lambda _:path,load)
    assert restore_full_checkpoint_after_dtype_setup(model,tmp_path)
    torch.testing.assert_close(model.weight,trained,atol=0,rtol=0)
    assert model.weight.dtype==torch.float32


def test_safetensors_base_is_not_reloaded_or_changed(monkeypatch):
    model=torch.nn.Linear(2,1)
    before={k:v.clone() for k,v in model.state_dict().items()}
    def forbidden(*args,**kwargs):
        pytest.fail('Safetensors base must retain the pinned upstream initialization')
    fake_checkpoint_api(monkeypatch,lambda _:None,forbidden)
    assert not restore_full_checkpoint_after_dtype_setup(model,'base')
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],atol=0,rtol=0)
