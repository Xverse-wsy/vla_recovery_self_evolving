"""Adapter for RLinf c70606f Pi0.5; frozen prefix, differentiable expert.

No monkeypatch of inference decorators, no calls to sft_forward (which disables
checkpointing). Native sample_actions remains the forward-parity reference.
"""
from dataclasses import dataclass
from pathlib import Path
import torch


RLINF_COMMIT = 'c70606f08cdca259b8dec03d4430926b5b8fac9d'


@dataclass
class PrefixCondition:
    raw: object
    observation: object
    prefix_mask: torch.Tensor
    kv_cache: tuple


class OpenPiBackend(torch.nn.Module):
    def __init__(self, policy, steps=3, fm_augmentation=False):
        super().__init__()
        required = ['freeze_vlm', 'build_prefix_cache', 'run_suffix', 'velocity_from_suffix', 'sample_actions', 'compute_loss']
        if not all(callable(getattr(policy, name, None)) for name in required):
            raise TypeError('Expected pinned RLinf Pi0 interface; official OpenPI has a different API')
        if not policy.pi05 or getattr(policy.rlt_cfg, 'use_rlt', False):
            raise ValueError('First version requires pi05=True, use_rlt=False')
        self.policy = policy
        self.horizon, self.action_dim, self.deployment_steps = policy.action_horizon, policy.action_dim, steps
        self.fm_augmentation = fm_augmentation
        # Use upstream's explicit expert0/expert1 freeze implementation.
        policy.requires_grad_(True)
        policy.freeze_vlm(freeze_action_expert=False)
        policy.eval()  # Match deployment and avoid dropout across the K-step chain.
        if any(p.requires_grad for p in policy.img.parameters()):
            raise RuntimeError('Vision encoder was not frozen')
        if not any(p.requires_grad for p in policy.action_out_proj.parameters()):
            raise RuntimeError('Action projections must remain trainable')

    def prepare_condition(self, observation, train=False):
        from rlinf.models.embodiment.openpi.modules import model
        # no_grad, not inference_mode: cache tensors are consumed by autograd later.
        with torch.no_grad():
            prepared = model.preprocess_observation(observation, train=bool(train and self.fm_augmentation))
            _, mask, cache = self.policy.build_prefix_cache(prepared)
        return PrefixCondition(observation, prepared, mask, cache)

    def velocity(self, condition, action, time):
        suffix = self.policy.run_suffix(condition.observation, action, time,
                                        condition.kv_cache, condition.prefix_mask)
        return self.policy.velocity_from_suffix(suffix)

    def sample_flow_time(self, batch_size, device):
        # Exactly the pinned compute_loss distribution; H is unrelated to K.
        return torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((batch_size,)).to(device) * .999 + .001

    def fm_loss_elements(self, observation, actions, *, noise, time):
        # FM uses the SAME joint prefix/suffix forward as upstream SFT. The
        # cached-prefix path remains specific to deployment/pair sampling.
        # Do not call sft_forward: that wrapper disables gradient checkpointing.
        return self.policy.compute_loss(observation, actions, train=self.fm_augmentation,
                                        noise=noise, time=time)

    @torch.no_grad()
    def deployment_sample(self, observation, noise):
        return self.policy.sample_actions(observation, num_steps=self.deployment_steps, noise=noise.clone())

    def freeze_report(self):
        trainable = {n: p.numel() for n, p in self.policy.named_parameters() if p.requires_grad}
        return dict(trainable_parameters=sum(trainable.values()), trainable_names=trainable,
                    frozen_parameters=sum(p.numel() for p in self.policy.parameters() if not p.requires_grad),
                    model_family='RLinf Pi0.5', pinned_rlinf_commit=RLINF_COMMIT,
                    source_identity=getattr(self,'source_identity',None))


def load_backend(checkpoint, norm_stats, config, device='cpu'):
    import rlinf
    from .source_identity import verify_installed_rlinf
    identity=verify_installed_rlinf(rlinf,RLINF_COMMIT)
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    if not Path(norm_stats).is_file():
        raise FileNotFoundError(norm_stats)
    model_config = OmegaConf.create(dict(
        model_path=str(checkpoint), precision=None, pi05=True, action_dim=7,
        num_action_chunks=config.horizon, num_steps=config.flow_steps, add_value_head=False,
        openpi_data=dict(norm_stats_path=str(norm_stats), extra_delta_transform=False),
        openpi=dict(task='eval', config_name='pi05_libero', model_action_dim=config.action_dim,
                    action_env_dim=7, action_horizon=config.horizon, action_chunk=config.horizon,
                    paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                    max_token_len=200, discrete_state_input=False, torch_compile=False, use_rlt=False)))
    policy = get_model(model_config).to(device)
    if (policy.action_horizon, policy.action_dim) != (config.horizon, config.action_dim):
        raise ValueError('Checkpoint model shape disagrees with window specification')
    backend=OpenPiBackend(policy, config.flow_steps)
    backend.source_identity=identity
    return backend


class ObservationLoader:
    """Only starting RGB, raw state, and original task text reach the base model."""
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, records):
        import numpy as np
        from .features import read_start_image
        policy = self.backend.policy
        # EnvIO uses the same original LiberoInputs/normalization/tokenizer assets
        # as deployment. No labels, future image, event or progress is included.
        env = dict(
            states=np.asarray([r.proprio for r in records], dtype=np.float32),
            main_images=np.stack([read_start_image(r.image_refs['front']) for r in records]),
            wrist_images=np.stack([read_start_image(r.image_refs['wrist']) for r in records]),
            task_descriptions=[r.instruction for r in records])
        return policy.env_obs_to_observation(env)
