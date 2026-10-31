"""Bounded real-GPU FM review: source alignment, native loss, resume/continuous parity."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import Config
from .flow import fm_prediction
from .engine import fm_loss_elements
from .openpi_backend import load_backend, ObservationLoader
from .pipeline import audit_prepared
from .positive_fm import POLICY, DENSE_POLICY, check_training_mode, audit_source_alignment
from .trainer import base_identity, train


def parameter_hash(backend, trainable):
    digest = hashlib.sha256()
    for name, p in backend.named_parameters():
        if p.requires_grad != trainable:
            continue
        digest.update(name.encode())
        # Bound CPU copies even for the large frozen token embedding.
        flat = p.detach().reshape(-1)
        for offset in range(0, flat.numel(), 4_000_000):
            part = flat[offset:offset + 4_000_000].contiguous().cpu().view(torch.uint8)
            digest.update(part.numpy().tobytes())
    return digest.hexdigest()


def probe_native_fm(backend, loader, record):
    device = next(backend.parameters()).device
    payload = record.payload()
    target = torch.as_tensor(payload['actions'][None], device=device)
    mask = torch.as_tensor(payload['valid_mask'][None], device=device)
    generator = torch.Generator(device=device).manual_seed(12345)
    noise = torch.randn(target.shape, device=device, generator=generator)
    t = torch.tensor([.65], device=device)
    observation = loader([record])
    with torch.no_grad():
        native = backend.policy.compute_loss(observation, target, train=False, noise=noise, time=t)
        old_prediction, velocity_target = fm_prediction(backend, observation, target, noise=noise, time=t)
        cached_difference = float(((old_prediction-velocity_target).square()[mask]-native[mask]).abs().max())
    element = fm_loss_elements(backend, observation, target, noise=noise, time=t)
    if not torch.allclose(element[mask], native[mask], atol=2e-5, rtol=2e-5):
        raise RuntimeError('Positive FM differs from native compute_loss')
    loss = element[mask].mean()
    if not mask.any() or not torch.isfinite(loss):
        raise RuntimeError('Invalid native FM probe loss')
    loss.backward()
    grads = [p.grad for p in backend.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
        raise RuntimeError('Native FM probe has invalid or zero trainable gradients')
    if any(p.grad is not None for p in backend.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen VLM unexpectedly received gradients')
    result = dict(sample_id=record.sample_id, category=record.bank_label.get('source_category'),
        real_steps=record.source['fm_real_steps'], repeated_steps=record.source['fm_repeated_steps'],
        loss=float(loss.detach()), native_max_abs_error=float((element[mask]-native[mask]).abs().max().detach()),
        cached_path_max_abs_difference=cached_difference, fm_path='native_compute_loss',
        finite_nonzero_gradients=True, frozen_gradients_absent=True)
    backend.zero_grad(set_to_none=True)
    return result


def validate(prepared, config_path, checkpoint, norm_stats, output, device='cuda:0', memory_gib=20.):
    config = Config.load(config_path)
    records, pairs, digest = audit_prepared(prepared, config)
    check_training_mode(prepared, 'fm_only')
    info = json.loads((Path(prepared) / 'report.json').read_text())
    if info.get('training_policy') not in (POLICY, DENSE_POLICY) or pairs:
        raise ValueError('Expected source-aware positive-only prepared data')
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('This validation explicitly requires a real CUDA model')
    if config.fm_only_updates < 2:
        raise ValueError('Validation needs room for two updates')
    source_audit = audit_source_alignment(records, norm_stats)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    started = time.monotonic()
    try:
        torch.cuda.set_device(torch.device(device))
        free, total = torch.cuda.mem_get_info()
        limit = int(memory_gib * 1024**3)
        if not 0 < limit or limit + 4*1024**3 > free:
            raise RuntimeError('Insufficient free GPU memory for the cap plus 4GiB margin')
        torch.cuda.set_per_process_memory_fraction(limit / total)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        status.write_text(json.dumps({'state': 'loading_base', 'free_gib': free / 1024**3}))
        base = base_identity(checkpoint, norm_stats)
        backend = load_backend(checkpoint, norm_stats, config, device)
        loader = ObservationLoader(backend)
        frozen_before = parameter_hash(backend, False)
        base_trainable = parameter_hash(backend, True)
        frozen_report = backend.freeze_report()
        status.write_text(json.dumps({'state': 'probing_native_fm'}))
        probes = []
        for category in ('vla_progress_candidate', 'autonomous_episode_success', 'capx_completed'):
            candidates = [r for r in records if r.split == 'train' and
                          r.bank_label.get('source_category') == category and
                          (category == 'vla_progress_candidate' or r.source['fm_repeated_steps'] > 0)]
            if not candidates:
                raise ValueError('No training probe for ' + category)
            probe = probe_native_fm(backend, loader, candidates[0])
            probes.append(probe)
            print(json.dumps({'native_fm_probe': probe}), flush=True)
        status.write_text(json.dumps({'state': 'first_update'}))
        first = train(backend, loader, prepared, config, out / 'run', base=base, mode='fm_only',
                      max_updates=1, save_every=1, validate_every=1, keep_last=2)
        first_hash = parameter_hash(backend, True)
        if first_hash == base_trainable or parameter_hash(backend, False) != frozen_before:
            raise RuntimeError('First update did not change expert or modified frozen weights')
        status.write_text(json.dumps({'state': 'reloading_base_for_resume'}))
        del loader, backend
        gc.collect()
        torch.cuda.empty_cache()
        backend = load_backend(checkpoint, norm_stats, config, device)
        if parameter_hash(backend, True) != base_trainable or parameter_hash(backend, False) != frozen_before:
            raise RuntimeError('Fresh base reload differs from initialization')
        loader = ObservationLoader(backend)
        status.write_text(json.dumps({'state': 'resuming_second_update'}))
        second = train(backend, loader, prepared, config, out / 'run', base=base, mode='fm_only',
            resume=out / 'run/update_000001.pt', max_updates=2, save_every=1, validate_every=1, keep_last=2)
        final_hash = parameter_hash(backend, True)
        if final_hash in (base_trainable, first_hash) or parameter_hash(backend, False) != frozen_before:
            raise RuntimeError('Resume did not update expert or modified frozen weights')
        frozen_after_resume = parameter_hash(backend, False)
        status.write_text(json.dumps({'state': 'checking_uninterrupted_equivalence'}))
        del loader, backend
        gc.collect()
        torch.cuda.empty_cache()
        backend = load_backend(checkpoint, norm_stats, config, device)
        loader = ObservationLoader(backend)
        continuous = train(backend, loader, prepared, config, out / 'uninterrupted', base=base,
            mode='fm_only', max_updates=2, save_every=1, validate_every=1, keep_last=2)
        if parameter_hash(backend, True) != final_hash:
            raise RuntimeError('Resumed and uninterrupted two-update parameters differ')
        if parameter_hash(backend, False) != frozen_after_resume:
            raise RuntimeError('Uninterrupted run changed frozen parameters')
        metrics = [json.loads(x) for x in (out / 'run/metrics.jsonl').read_text().splitlines()]
        if ([m['update'] for m in metrics] != [1, 2] or second['updates'] != 2
                or any(m['fm_count'] != config.fm_batch or m['pair_count'] or m['pair'] or m['eta']
                       or not np.isfinite(m['loss']) or not m['gradient_norm_before_clip'] > 0 for m in metrics)):
            raise RuntimeError('Invalid real FM update metrics')
        sampler = json.loads((out / 'run/sampler_audit.json').read_text())
        continuous_metrics = [json.loads(x) for x in (out / 'uninterrupted/metrics.jsonl').read_text().splitlines()]
        if continuous_metrics != metrics:
            raise RuntimeError('Resumed and uninterrupted sampling/loss metrics differ')
        audit_prepared(prepared, config)
        if base_identity(checkpoint, norm_stats) != base:
            raise RuntimeError('Original base checkpoint changed')
        result = dict(state='complete', real_model=True, optimizer_updates=4,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device=device,
            prepared_sha256=digest, prepared=str(Path(prepared).resolve()),
            base=base, config=config.to_dict(), native_fm_probes=probes, metrics=metrics,
            sampler_audit=sampler, freeze_report=frozen_report,
            frozen_weights_unchanged=True, trainable_weights_changed=True,
            fresh_base_reload_verified=True, checkpoint_resume_verified=True,
            uninterrupted_equivalence_verified=True, source_alignment_audit=source_audit,
            validation_fm_windows=json.loads((out/'run/run.json').read_text())['validation_fm_windows'],
            original_base_unchanged=True, prepared_audit_after_passed=True,
            frozen_sha256=frozen_before, base_trainable_sha256=base_trainable,
            first_trainable_sha256=first_hash, final_trainable_sha256=final_hash,
            cuda_cap_gib=memory_gib, cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            elapsed_seconds=time.monotonic()-started,
            scope='Two resumed and two uninterrupted FM64 updates with exact final-weight/metric equivalence; no long training, LIBERO evaluation or API.')
        (out / 'report.json').write_text(json.dumps(result, indent=2))
        status.write_text(json.dumps({'state': 'complete', 'optimizer_updates': 4}))
        return result
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'config', 'checkpoint', 'norm-stats', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-cuda-memory-gb', type=float, default=20.)
    args = parser.parse_args()
    print(json.dumps(validate(args.prepared, args.config, args.checkpoint, args.norm_stats,
        args.output, args.device, args.max_cuda_memory_gb), indent=2))


if __name__ == '__main__':
    main()
