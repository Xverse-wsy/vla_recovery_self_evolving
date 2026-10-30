"""Global FM and pair means, microbatch accumulation, optional manual all-reduce.

Manual SUM reduction is deliberate: each rank scales by the GLOBAL branch size.
Ranks without pair examples still participate with zero gradients. No DDP hooks
are bypassed by the adapter's direct suffix calls.
"""
import torch
from .flow import differentiable_sample, fm_prediction
from .losses import distance_per_window, eta_at_update, pair_loss_per_window


def distributed_info():
    dist = torch.distributed
    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


def synchronize_gradients(model):
    rank, world = distributed_info()
    if world == 1:
        return
    for param in model.parameters():
        if param.requires_grad:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)


def accumulate_update(backend, observation_loader, records, plan, config, *, update_index,
                      total_updates, draws=None, mode='continuous_pair', backward=True):
    if mode not in ('continuous_pair', 'pull_only', 'fm_only'):
        raise ValueError('Unknown ablation mode')
    config.validate()
    by_id = {r.sample_id: r for r in records}
    device = next(backend.parameters()).device
    rank, world = distributed_info()
    n_fm, n_pair = len(plan['fm_ids']), len(plan['pairs'])
    if not n_fm:
        raise ValueError('FM branch requires positive windows')
    if draws is None:
        # Generate complete logical-batch draws before slicing into microbatches.
        # All ranks use an update-specific CPU generator, then take disjoint rows.
        generator = torch.Generator().manual_seed(config.seed + 1000003 * update_index)
        fm_noise = torch.randn(n_fm, config.horizon, config.action_dim, generator=generator)
        pair_noise = torch.randn(n_pair, config.horizon, config.action_dim, generator=generator)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 2000003 * update_index)
            time = backend.sample_flow_time(n_fm, torch.device('cpu'))
        draws = dict(fm_noise=fm_noise, pair_noise=pair_noise, time=time)
    eta = eta_at_update(update_index, total_updates, config.eta, config.eta_warmup_fraction)
    metrics = dict(fm=0.0, pair=0.0, pull=0.0, push=0.0, eta=eta)

    def chunks(indices, size):
        for j in range(0, len(indices), size):
            yield indices[j:j + size]

    for ids in chunks(list(range(rank, n_fm, world)), config.fm_micro_batch):
        selected = [by_id[plan['fm_ids'][i]] for i in ids]
        if any(r.training_quality != 'positive' for r in selected):
            raise ValueError('Negative/unknown actions may not enter ordinary FM')
        payload = [r.payload() for r in selected]
        target = torch.as_tensor(__import__('numpy').stack([p['actions'] for p in payload]), device=device)
        mask = torch.as_tensor(__import__('numpy').stack([p['valid_mask'] for p in payload]), device=device)
        # Invalid recorded targets must not influence valid predictions through
        # action-token attention before the masked loss is computed.
        target = torch.where(mask, target, torch.zeros_like(target))
        prediction, velocity_target = fm_prediction(backend, observation_loader(selected), target,
            noise=draws['fm_noise'][ids].to(device), time=draws['time'][ids].to(device))
        loss, valid = distance_per_window(prediction, velocity_target, mask)
        if not valid.all():
            raise ValueError('Empty positive windows must be removed before batching')
        objective = loss.sum() / n_fm
        if backward:
            objective.backward()
        metrics['fm'] += objective.detach().item()
    if n_pair and mode != 'fm_only':
        for ids in chunks(list(range(rank, n_pair, world)), config.pair_micro_batch):
            selected = [plan['pairs'][i] for i in ids]
            neg = [by_id[x['negative_id']] for x in selected]
            pos = [by_id[x['positive_id']] for x in selected]
            if any(r.training_quality != 'negative' for r in neg) or any(r.training_quality != 'positive' for r in pos):
                raise ValueError('Pair labels invalid')
            for a, b in zip(pos, neg):
                if (a.spec.identity != b.spec.identity or a.split != b.split or
                        ' '.join(a.instruction.split()) != ' '.join(b.instruction.split())):
                    raise ValueError('Pair provenance/action representation mismatch')
            np = __import__('numpy')
            positive = torch.as_tensor(np.stack([r.payload()['actions'] for r in pos]), device=device)
            negative = torch.as_tensor(np.stack([r.payload()['actions'] for r in neg]), device=device)
            mask = torch.as_tensor(np.asarray([x['mask'] for x in selected], dtype=bool), device=device)
            condition = backend.prepare_condition(observation_loader(neg), train=False)
            generated = differentiable_sample(backend, condition, draws['pair_noise'][ids].to(device),
                config.flow_steps, checkpoint_steps=config.checkpoint_sampling)
            loss, valid, terms = pair_loss_per_window(generated, positive, negative, mask, config.lambda_push)
            if not valid.all():
                raise ValueError('Empty pair mask must be excluded before training')
            if mode == 'pull_only':
                loss = terms['pull']
            objective = loss.sum() / n_pair
            if backward:
                (eta * objective).backward()
            metrics['pair'] += objective.detach().item()
            metrics['pull'] += terms['pull'].detach().sum().item() / n_pair
            metrics['push'] += terms['push'].detach().sum().item() / n_pair
    if backward:
        synchronize_gradients(backend)
    if world > 1:
        names = ['fm', 'pair', 'pull', 'push']
        tensor = torch.tensor([metrics[k] for k in names], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        metrics.update(zip(names, tensor.tolist()))
    metrics['loss'] = metrics['fm'] + eta * metrics['pair']
    if not all(__import__('math').isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Nonfinite loss; negative pair loss itself is allowed')
    return metrics


def optimizer_update(backend, optimizer, observation_loader, records, plan, config, **kwargs):
    optimizer.zero_grad(set_to_none=True)
    metrics = accumulate_update(backend, observation_loader, records, plan, config, **kwargs)
    trainable = [p for p in backend.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    metrics['gradient_norm_before_clip'] = float(norm)
    return metrics
