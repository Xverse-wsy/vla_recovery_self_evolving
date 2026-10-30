"""Bounded episode-budget trainer. Explicit base checkpoint; no warm-up branch."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import torch
import numpy as np
from .batches import EpisodeEventSampler
from .engine import optimizer_update, accumulate_update, distributed_info
from .features import sha256
from .flow import differentiable_sample
from .pipeline import audit_prepared


def check_sampler(backend, loader, record, tolerance=2e-5, checkpoint_steps=False):
    """Use identical real model, start observation, noise, K and time grid."""
    device = next(backend.parameters()).device
    obs = loader([record])
    generator = torch.Generator(device=device).manual_seed(781)
    noise = torch.randn(1, backend.horizon, backend.action_dim, device=device, generator=generator)
    reference = backend.deployment_sample(obs, noise)
    with torch.enable_grad():
        condition = backend.prepare_condition(obs, train=False)
        generated, chain = differentiable_sample(backend, condition, noise,
            backend.deployment_steps, return_chain=True,checkpoint_steps=checkpoint_steps)
        if not torch.allclose(generated, reference, atol=tolerance, rtol=tolerance):
            raise RuntimeError('Differentiable/deployed sampler parity failed')
        generated.square().mean().backward()
    steps = [x.grad is not None and torch.isfinite(x.grad).all().item() for x in chain[1:]]
    if not all(steps) or not any(p.grad is not None and p.grad.abs().sum()>0 for p in backend.parameters() if p.requires_grad):
        raise RuntimeError('Sampler gradients did not traverse every step')
    backend.zero_grad(set_to_none=True)
    return dict(native_forward_parity=True, max_abs_error=float((generated-reference).abs().max().detach()),
                gradient_steps=steps, H=backend.horizon, K=backend.deployment_steps,
                checkpoint_steps=checkpoint_steps)


def base_identity(checkpoint, norm):
    root = Path(checkpoint).resolve()
    candidates = [root/'full_weights.pt', root/'model/full_weights.pt', root/'model.safetensors']
    weights = next((p for p in candidates if p.is_file()), None)
    if weights is None:
        raise FileNotFoundError('Expected pinned RLinf base model weights')
    return dict(checkpoint=str(root), weight_file=str(weights), weights_sha256=sha256(weights),
                norm_stats=str(Path(norm).resolve()), norm_sha256=sha256(norm))


def train(backend, loader, prepared, config, output, *, base=None, resume=None,
          mode='continuous_pair', max_updates=None, save_every=50,
          validate_every=25, patience=5, keep_last=3):
    config.validate()
    if save_every<1 or validate_every<1 or patience<1 or keep_last<1 or (max_updates is not None and max_updates<1):
        raise ValueError('Training intervals/bounds must be positive')
    records, pairs, digest = audit_prepared(prepared, config)
    from .positive_fm import check_training_mode
    check_training_mode(prepared, mode)
    if len({r.spec.identity for r in records}) != 1:
        raise ValueError('FM and pair pools must use one identical action specification')
    for r in records:
        if (r.spec.horizon,r.spec.model_dim,r.spec.control_hz)!=(config.horizon,config.action_dim,config.native_control_hz):
            raise ValueError('Data/model action specification mismatch')
        if base and r.spec.normalization_sha256 != base['norm_sha256']:
            raise ValueError('Base checkpoint normalization differs from data')
    sampler = EpisodeEventSampler(records,pairs,config)
    rank, world = distributed_info()
    out = Path(output).resolve()
    if rank==0:
        if out.exists() and not resume:
            raise FileExistsError('New run needs an empty output directory')
        out.mkdir(parents=True,exist_ok=bool(resume))
    if world>1:
        torch.distributed.barrier()
    optimizer = torch.optim.AdamW([p for p in backend.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=0.0)
    config_identity = config.to_dict()
    # Changing microbatch/world size is valid on resume; logical batches stay fixed.
    for k in ('fm_micro_batch','pair_micro_batch'):
        config_identity.pop(k)
    identity = dict(base=base,prepared_sha256=digest,config=config_identity,mode=mode)
    start, best, bad = 0, float('inf'), 0
    if resume:
        state = torch.load(resume,map_location='cpu',weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume base/data/config/mode differs')
        named = dict(backend.named_parameters())
        with torch.no_grad():
            for name, value in state['trainable'].items():
                named[name].copy_(value.to(named[name]))
        optimizer.load_state_dict(state['optimizer'])
        sampler.load_state_dict(state['sampler'])
        start,best,bad = state['update'], state['best_validation'],state['bad_validations']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng') is not None:
            # Current device only; supports changing number of ranks on resume.
            torch.cuda.set_rng_state(state['cuda_rng'])
    validation = [r for r in records if r.split=='validation' and r.training_quality=='positive' and r.controller=='VLA']
    validation_plan = None
    if validation:
        val_config = replace(config, max_episode_reuses=1)
        val_sampler = EpisodeEventSampler(records,pairs,val_config,split='validation')
        validation_plan = val_sampler.draw()  # fixed windows/reference/noise across checks
    # Automatically enforce real native-vs-differentiable parity before updating.
    report = check_sampler(backend,loader,next(r for r in records if r.split=='train' and r.training_quality=='positive'))
    if rank==0:
        (out/'sampler_audit.json').write_text(json.dumps(report,indent=2))
        (out/'run.json').write_text(json.dumps(dict(identity=identity,world_size=world,
            freeze_report=backend.freeze_report() if hasattr(backend,'freeze_report') else {'backend':'toy'},
            no_positive_warmup=True,validation_enabled=bool(validation_plan)),indent=2))
    stop_reason='episode_budget_complete'

    def checkpoint(update):
        if rank!=0:
            return
        state = dict(identity=identity,update=update,
            trainable={n:p.detach().cpu().clone() for n,p in backend.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
            best_validation=best,bad_validations=bad,torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
        dest=out/f'update_{update:06d}.pt'
        tmp=dest.with_suffix('.tmp')
        torch.save(state,tmp)
        tmp.replace(dest)
        for old in sorted(out.glob('update_*.pt'))[:-keep_last]:
            old.unlink()

    update = start
    while not sampler.done:
        if max_updates is not None and update >= max_updates:
            stop_reason='requested_update_limit'
            break
        plan=sampler.draw()
        metrics=optimizer_update(backend,optimizer,loader,records,plan,config,
            update_index=update,total_updates=sampler.nominal_updates,mode=mode)
        update+=1
        metrics.update(update=update,episode_visits=plan['episode_visits'],
                       fm_count=len(plan['fm_ids']),pair_count=len(plan['pairs']))
        if validation_plan and update%validate_every==0:
            validation_metrics=accumulate_update(backend,loader,records,validation_plan,config,
                update_index=781,total_updates=1,mode=mode,backward=False)
            score=validation_metrics['loss']
            metrics['validation']=validation_metrics
            if score<best:
                best,bad=score,0
            else:
                bad+=1
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics,allow_nan=False)+'\n')
            print(json.dumps(metrics,ensure_ascii=False),flush=True)
        if update%save_every==0:
            checkpoint(update)
        if validation_plan and bad>=patience:
            stop_reason='validation_early_stop'
            break
    checkpoint(update)
    result=dict(state='stopped',reason=stop_reason,updates=update,episode_visits=sampler.visits,
                max_episode_reuses=config.max_episode_reuses,validation_enabled=bool(validation_plan))
    if rank==0:
        (out/'status.json').write_text(json.dumps(result,indent=2))
    return result


def init_distributed():
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        torch.distributed.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    if torch.cuda.is_available():
        device=int(os.environ.get('LOCAL_RANK','0'))
        torch.cuda.set_device(device)
        return f'cuda:{device}'
    return 'cpu'
