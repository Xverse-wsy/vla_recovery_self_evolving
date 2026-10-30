"""Differentiable fixed-length Euler chain from pure noise, t=1 -> t=0."""
import torch
from torch.utils.checkpoint import checkpoint


def deployment_time_grid(steps):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Flow integration steps must be a positive integer')
    # Preserve the deployed Python-float accumulation before conversion to fp32.
    t, dt = 1.0, -1.0 / steps
    grid = []
    for _ in range(steps):
        grid.append(t)
        t += dt
    return grid


def differentiable_sample(backend, condition, noise, steps, *, checkpoint_steps=False, return_chain=False):
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise RuntimeError('Pair sampler requires grad mode; do not call inside inference_mode/no_grad')
    if noise.ndim != 3 or noise.shape[1:] != (backend.horizon, backend.action_dim):
        raise ValueError('Noise shape must match model H,D')
    if getattr(backend, 'deployment_steps', steps) != steps:
        raise ValueError('Training sampler K differs from deployment K')
    x = noise.clone()
    chain = [x]
    for t in deployment_time_grid(steps):
        time = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)

        def velocity(value, time=time):
            return backend.velocity(condition, value, time)

        v = checkpoint(velocity, x, use_reentrant=False) if checkpoint_steps else velocity(x)
        x = x + (-1.0 / steps) * v
        if return_chain and x.requires_grad:
            x.retain_grad()
        chain.append(x)
    return (x, chain) if return_chain else x


def fm_prediction(backend, observation, actions, *, noise=None, time=None):
    """Use the backbone's own flow-time distribution, own positive start state."""
    if noise is None:
        noise = torch.randn_like(actions)
    if time is None:
        time = backend.sample_flow_time(actions.shape[0], actions.device)
    condition = backend.prepare_condition(observation, train=True)
    x_t = (1 - time[:, None, None]) * actions + time[:, None, None] * noise
    return backend.velocity(condition, x_t, time), noise - actions
