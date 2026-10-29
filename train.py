"""Independent A/B SFT runs, exact epochs, and isolated end-to-end short tests."""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import traceback

import ray
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from direct_data import DirectEpisodes
from epoch_plan import EpochPlan
from settings import load_config
from training_runtime import RetainingRunner, TrainingWorker, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, choices=['a60', 'b80'])
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--gpus', required=True, help='Allocated physical GPUs, e.g. 1,3')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--micro-batch-size', type=int)
    ap.add_argument('--global-batch-size', type=int)
    ap.add_argument('--gradient-accumulation-steps', type=int)
    ap.add_argument('--num-workers', type=int)
    ap.add_argument('--gpu-memory-limit-gib', type=float)
    ap.add_argument('--gpu-headroom-gib', type=float)
    ap.add_argument('--num-nodes', type=int, default=1)
    ap.add_argument('--ray-address', default=None, help='Existing multi-node Ray cluster; omit for private local tests')
    args = ap.parse_args()
    assert not (args.smoke and args.resume)
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(level=logging.INFO)
    gpus = [int(x) for x in args.gpus.split(',')]
    assert gpus == sorted(set(gpus)) and gpus
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists() and not args.resume:
        raise FileExistsError(f'Use a fresh output directory: {output}')
    cfg = load_config(args.experiment)
    if args.micro_batch_size is not None:
        cfg.actor.micro_batch_size = args.micro_batch_size
    if args.global_batch_size is not None:
        cfg.actor.global_batch_size = args.global_batch_size
    if args.gradient_accumulation_steps is not None:
        assert args.gradient_accumulation_steps > 0
        calculated = cfg.actor.micro_batch_size * len(gpus) * args.gradient_accumulation_steps
        assert args.global_batch_size is None or args.global_batch_size == calculated, 'Global batch and accumulation disagree'
        cfg.actor.global_batch_size = calculated
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.gpu_memory_limit_gib is not None:
        cfg.actor.gpu_memory_limit_gib = args.gpu_memory_limit_gib
    if args.gpu_headroom_gib is not None:
        cfg.actor.gpu_headroom_gib = args.gpu_headroom_gib
    assert args.num_nodes > 0 and (args.num_nodes == 1 or args.ray_address), 'Multi-node training needs an existing Ray cluster'
    cfg.cluster.num_nodes = args.num_nodes
    assert cfg.actor.micro_batch_size > 0 and cfg.actor.global_batch_size > 0 and cfg.data.num_workers >= 0
    assert cfg.actor.global_batch_size % (len(gpus) * cfg.actor.micro_batch_size) == 0, 'Global batch must divide evenly across GPUs and microbatches'
    for component in ('actor', 'env', 'rollout'):
        cfg.cluster.component_placement[component] = args.gpus
    cfg.runner.logger.log_path = str(output)
    OmegaConf.update(cfg, 'validation_world_size', len(gpus), force_add=True)
    raw = DirectEpisodes(OmegaConf.to_container(cfg.data.sources, resolve=True))
    expected_episodes = 60 if args.experiment == 'a60' else 80
    assert len(raw.episodes) == expected_episodes
    write_json(output / 'dataset_manifest.json', raw.summary())
    if args.smoke:
        # 15 examples exercise uneven last microbatches, partial accumulation,
        # a complete epoch boundary, and every selected data source.
        indices = sorted(set([0, 6631, 6632, 9863, len(raw) - 1] +
                             [int(i * (len(raw) - 1) / 13) for i in range(14)]))[:15]
        if args.experiment == 'b80' and not any(i >= 9864 for i in indices):
            raise AssertionError('Recovery examples absent from smoke')
        cfg.data.validation_indices = indices
        cfg.actor.global_batch_size = len(gpus) * cfg.actor.micro_batch_size * 3
        cfg.runner.max_epochs = 2
        cfg.runner.save_interval = 2
        cfg.actor.optim.lr_warmup_steps = 1
    samples = len(cfg.data.validation_indices) if args.smoke else len(raw)
    plan = EpochPlan(samples, len(gpus), cfg.actor.micro_batch_size,
                     cfg.actor.global_batch_size // (len(gpus) * cfg.actor.micro_batch_size))
    max_updates = plan.updates_per_epoch * cfg.runner.max_epochs
    if args.resume:
        resume = args.resume.resolve()
        assert (resume / 'COMPLETE.json').is_file()
        prior_config = resume.parents[2] / 'resolved_config.yaml'
        # Worker also verifies the complete dataset identity and epoch plan.
        assert prior_config.exists(), prior_config
        cfg.runner.resume_dir = str(resume)
    weights = Path(cfg.actor.model.model_path) / 'model.safetensors'
    original_stat = (weights.stat().st_size, weights.stat().st_mtime_ns)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                         '--format=csv,noheader,nounits'], text=True)
    devices = {int(v[0]): (v[1].strip(), int(v[2])) for line in gpu_rows.splitlines()
               if (v := line.split(','))}
    required_gib = cfg.actor.gpu_memory_limit_gib + cfg.actor.gpu_headroom_gib + 1
    if args.num_nodes == 1:
        assert all(devices[i][1] >= required_gib * 1024 for i in gpus), f'Insufficient free memory: need {required_gib} GiB, found {devices}'
        expected_uuids = {devices[i][0] for i in gpus}
    else:
        expected_uuids = None  # Each remote worker independently checks its actual device before loading.
    status = dict(state='initializing', experiment=args.experiment, smoke=args.smoke,
                  physical_gpus=gpus, max_updates=max_updates, epochs=cfg.runner.max_epochs,
                  epoch_plan=plan.describe(), global_batch=cfg.actor.global_batch_size,
                  micro_batch=cfg.actor.micro_batch_size, started=time.time(), pid=os.getpid())
    status.update(gpu_allocator_limit_gib=cfg.actor.gpu_memory_limit_gib,
                  gpu_free_headroom_gib=cfg.actor.gpu_headroom_gib)
    status.update(cpu_offload=bool(cfg.actor.fsdp_config.cpu_offload), num_nodes=args.num_nodes)
    write_json(output / 'status.json', status)
    original_init = ray.init
    ray_path = f'/tmp/pi05ab_{os.getuid()}_{os.getpid()}'

    def private_init(*a, **kw):
        if args.ray_address:
            kw.update(address=args.ray_address)
            for field in ('_temp_dir','num_cpus','object_store_memory','include_dashboard'):
                kw.pop(field, None)
        else:
            kw.update(address='local', _temp_dir=ray_path, include_dashboard=False,
                      num_cpus=12, object_store_memory=256 * 1024**2)
        return original_init(*a, **kw)

    try:
        ray.init = private_init
        cfg = validate_cfg(cfg)
        cluster = Cluster(cluster_cfg=cfg.cluster)
        ray.init = original_init
        OmegaConf.save(cfg, output / 'resolved_config.yaml')
        placement = HybridComponentPlacement(cfg, cluster)
        actor = TrainingWorker.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
                                         placement_strategy=placement.get_strategy('actor'))
        runner = RetainingRunner(cfg=cfg, actor=actor)
        assert runner.max_steps == max_updates
        runner.init_workers()
        workers = [json.loads((output / f'worker_{rank}.json').read_text()) for rank in range(len(gpus))]
        actual_uuids = {'GPU-' + w['device_uuid'].removeprefix('GPU-') for w in workers}
        assert len(actual_uuids) == len(gpus), 'Ranks unexpectedly share a GPU'
        if expected_uuids is not None:
            assert actual_uuids == expected_uuids, actual_uuids
        assert all(w['epoch_plan'] == plan.describe() for w in workers)
        initial = actor.audit_state().wait()
        if args.resume:
            # Preserve the original step-zero audit across interruptions, so
            # the A/B experiment can still verify identical initialization.
            original_initial = json.loads((output / 'initial_weights.json').read_text())
            assert all(item['optimizer_steps'] == 0 for item in original_initial)
            write_json(output / 'resumed_weights.json', initial)
        else:
            write_json(output / 'initial_weights.json', initial)
        status.update(state='running', workers=workers)
        write_json(output / 'status.json', status)
        runner.run()
        final = actor.audit_state(include_optimizer=args.smoke).wait()
        checkpoint_root = output / cfg.runner.logger.experiment_name / 'checkpoints'
        checkpoint = checkpoint_root / f'global_step_{max_updates}'
        assert (checkpoint / 'COMPLETE.json').is_file(), 'Final checkpoint was not saved'
        if args.smoke:
            resume_ckpt = checkpoint_root / 'global_step_2' / 'actor'
            actor.load_checkpoint(str(resume_ckpt)).wait()
            for step in range(2, max_updates):
                actor.set_global_step(step).wait()
                actor.run_training().wait()
            replayed = actor.audit_state(include_optimizer=True).wait()
            assert final == replayed, 'Resume does not reproduce weights/optimizer/input counters'
            assert final != initial
            write_json(output / 'roundtrip_report.json', dict(state='complete',
                       checkpoint_restored=str(resume_ckpt), final=final, replayed=replayed))
        progress = [json.loads((output / f'progress_rank_{i}.json').read_text()) for i in range(len(gpus))]
        assert all(p['metrics']['completed_epochs'] == cfg.runner.max_epochs for p in progress)
        assert all(p['metrics']['samples_seen'] == samples * cfg.runner.max_epochs for p in progress)
        raw.assert_unchanged()
        assert original_stat == (weights.stat().st_size, weights.stat().st_mtime_ns)
        status.update(state='complete', finished=time.time(), checkpoint=str(checkpoint),
                      exact_epochs_verified=True, source_data_unchanged=True, base_checkpoint_unchanged=True,
                      progress=progress)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        ray.init = original_init
        write_json(output / 'status.json', status)
        ray.shutdown()


if __name__ == '__main__':
    main()
