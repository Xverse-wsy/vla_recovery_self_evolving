"""Use the original A60 optimizer configuration and RLinf LR scheduler for Stage1."""
from pathlib import Path
import hashlib

import torch

RECIPE = 'stage1_a60_sft_v1'
OBJECTIVE = 'native_sft_full_action_mean_a60_v1'


def settings():
    from omegaconf import OmegaConf
    path = Path(__file__).resolve().parents[2]/'config/base.yaml'
    if not path.is_file():
        raise FileNotFoundError('Clone the complete adaptation repository: missing A60 config/base.yaml')
    base = OmegaConf.load(path)
    return OmegaConf.to_container(base.actor.optim, resolve=False)


def validate_training_settings(config):
    """Read model/epoch defaults from A60; global/micro batch remain configurable."""
    from omegaconf import OmegaConf
    path = Path(__file__).resolve().parents[2]/'config/base.yaml'
    if not path.is_file():
        raise FileNotFoundError('Use the COMPLETE adaptation repository: missing A60 config/base.yaml')
    base = OmegaConf.load(path)
    expected = dict(horizon=int(base.actor.model.openpi.action_horizon),
        action_dim=int(base.actor.model.openpi.model_action_dim),
        environment_action_dim=int(base.actor.model.openpi.action_env_dim),
        flow_steps=int(base.actor.model.num_steps), learning_rate=float(base.actor.optim.lr),
        gradient_clip=float(base.actor.optim.clip_grad), fm_epochs=int(base.runner.max_epochs),
        seed=int(base.actor.seed))
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError('Stage1 differs from the shared A60 recipe: '+key)
    return dict(recipe=RECIPE, objective=OBJECTIVE, shared_config=str(path),
        shared_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_and_budget=expected, optimizer=settings(), full_model_trainable=True,
        training_preprocessing=True, loss_reduction='mean_B_H_model_dim',
        global_batch=config.fm_batch, micro_batch=config.fm_micro_batch,
        global_batch_source='explicit_stage1_config; A60 batch is also launcher-overridable')


def build_optimizer_and_scheduler(backend, config, total_updates):
    # Use the same shared YAML and upstream builders as FSDPModelManager.
    from rlinf.hybrid_engines.fsdp.optim import build_adamw
    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
    from rlinf.utils.utils import warmup_optimizer_state
    validate_training_settings(config)
    opt = settings()
    if config.learning_rate != float(opt['lr']) or config.gradient_clip != float(opt['clip_grad']):
        raise ValueError('Stage1 LR/gradient clip differs from the shared A60 recipe')
    params = [p for p in backend.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    use_master = bool(opt.get('use_fp32_master_params', False))
    fused = (not use_master and all(p.device.type=='cuda' and p.dim()>0 for p in params))
    optimizer = build_adamw([dict(params=params,lr=float(opt['lr']),
        betas=(float(opt['adam_beta1']),float(opt['adam_beta2'])))],
        eps=float(opt['adam_eps']),weight_decay=float(opt['weight_decay']),
        use_fp32_master_params=use_master,fused=fused)
    warmup_optimizer_state(optimizer)
    scheduler = get_lr_scheduler(lr_scheduler=opt['lr_scheduler'], optimizer=optimizer,
        num_warmup_steps=int(opt['lr_warmup_steps']),num_training_steps=total_updates,
        num_cycles=opt.get('num_cycles',0.5),min_lr=float(opt['min_lr']),
        min_lr_rate=opt.get('min_lr_rate'),last_epoch=-1)
    return optimizer, scheduler, dict(opt, total_training_steps=total_updates)


def full_sft_mask(mask, config):
    """Native SFT includes model padding dims; segment-tail rules remain upstream."""
    if (mask.ndim!=3 or mask.shape[1:]!=(config.horizon,config.action_dim)
            or not mask[:,:,:config.environment_action_dim].all()
            or mask[:,:,config.environment_action_dim:].any()):
        raise ValueError('Native Stage1 requires full legal real/repeated action windows before model padding')
    return torch.ones_like(mask,dtype=torch.bool)
