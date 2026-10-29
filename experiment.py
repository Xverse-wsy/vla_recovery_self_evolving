"""Plan/execute the two independent SFT runs and three full LIBERO evaluations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from omegaconf import OmegaConf
from direct_data import DirectEpisodes
from epoch_plan import EpochPlan
from eval_jobs import lock, save
from settings import load_config, load_paths


def build_plan(smoke=False):
    package = Path(__file__).resolve().parent
    output = Path(os.environ['PI05_EXPERIMENT_OUTPUT']).expanduser().resolve()
    if smoke:
        output /= 'smoke'
    gpu_text = os.environ['PI05_GPUS']
    gpus = [int(value) for value in gpu_text.split(',')]
    assert gpus == sorted(set(gpus)) and gpus and min(gpus) >= 0
    nodes = int(os.environ.get('PI05_TRAIN_NUM_NODES', '1'))
    address = os.environ.get('PI05_RAY_ADDRESS', '')
    assert nodes > 0 and (nodes == 1 or address), 'Multi-node training needs an existing Ray address'
    eval_gpus = os.environ['PI05_EVAL_GPUS']
    eval_ids = [int(value) for value in eval_gpus.split(',')]
    assert eval_ids and len(eval_ids) == len(set(eval_ids)) and min(eval_ids) >= 0
    paths = OmegaConf.to_container(load_paths(), resolve=True)
    base = Path(paths['base_checkpoint']).expanduser().resolve()
    assert (base / 'model.safetensors').is_file(), 'Both runs must start from the original safetensors checkpoint'
    assert Path(paths['norm_stats']).expanduser().is_file()
    plan = dict(output=str(output), smoke=smoke, paths=paths, training={},
                evaluation=dict(gpus=eval_gpus, output=str(output / 'evaluation'),
                                episodes_per_model=4 if smoke else 2000,
                                models=['fewshot', 'a60', 'b80']),
                training_python=os.environ['PI05_TRAIN_PYTHON'],
                evaluation_python=os.environ['PI05_EVAL_PYTHON'])
    for label in ['a60', 'b80']:
        cfg = load_config(label)
        raw = DirectEpisodes(OmegaConf.to_container(cfg.data.sources, resolve=True))
        assert len(raw.episodes) == (60 if label == 'a60' else 80)
        micro = int(os.environ.get('PI05_MICRO_BATCH') or cfg.actor.micro_batch_size)
        accum = os.environ.get('PI05_ACCUMULATION')
        global_batch = os.environ.get('PI05_GLOBAL_BATCH')
        if accum:
            computed = len(gpus) * micro * int(accum)
            assert not global_batch or int(global_batch) == computed, 'Global batch and accumulation disagree'
            global_batch = computed
        else:
            global_batch = int(global_batch or cfg.actor.global_batch_size)
        assert micro > 0 and global_batch > 0 and global_batch % (len(gpus) * micro) == 0
        epochs = 2 if smoke else int(cfg.runner.max_epochs)
        assert epochs == (2 if smoke else 50), 'This experiment requires 50 complete epochs per dataset'
        effective_global = len(gpus) * micro * 3 if smoke else global_batch
        epoch = EpochPlan(15 if smoke else len(raw), len(gpus), micro,
                          effective_global // (len(gpus) * micro))
        updates = epochs * epoch.updates_per_epoch
        directory = output / 'training' / label
        command = ['bash', str(package / 'run_train.sh'), '--experiment', label,
                   '--output', str(directory), '--micro-batch-size', str(micro),
                   '--global-batch-size', str(global_batch), '--num-nodes', str(nodes)]
        for flag, key in [('--gpu-memory-limit-gib', 'PI05_GPU_MEMORY_LIMIT_GIB'),
                          ('--gpu-headroom-gib', 'PI05_GPU_HEADROOM_GIB'),
                          ('--num-workers', 'PI05_DATA_WORKERS')]:
            if os.environ.get(key):
                command += [flag, os.environ[key]]
        if address:
            command += ['--ray-address', address]
        if smoke:
            command += ['--smoke']
        checkpoint = directory / cfg.runner.logger.experiment_name / 'checkpoints' / f'global_step_{updates}'
        plan['training'][label] = dict(gpus=gpu_text, num_nodes=nodes, ray_address=address,
                                      initial_checkpoint=str(base), dataset=raw.summary(),
                                      micro_batch=micro, global_batch=effective_global, epochs=epochs,
                                      max_updates=updates, output=str(directory), checkpoint=str(checkpoint),
                                      command=command)
    return plan


def final_status(request):
    path = Path(request['output']) / 'status.json'
    assert path.is_file(), f'Training has not completed: {path}'
    status = json.loads(path.read_text())
    assert status['state'] == 'complete' and status['exact_epochs_verified'], path
    for key in ['epochs', 'global_batch', 'micro_batch', 'max_updates']:
        assert status[key] == request[key], f'Training settings changed: {key}'
    checkpoint = Path(status['checkpoint'])
    assert checkpoint.resolve() == Path(request['checkpoint']).resolve(), 'Wrong final checkpoint'
    assert (checkpoint / 'COMPLETE.json').is_file(), 'Checkpoint save did not complete'
    dataset = json.loads((Path(request['output']) / 'dataset_manifest.json').read_text())
    assert dataset == request['dataset'], 'Training dataset differs from the experiment plan'
    cfg = OmegaConf.load(Path(request['output']) / 'resolved_config.yaml')
    assert Path(cfg.actor.model.model_path).resolve() == Path(request['initial_checkpoint']).resolve(), 'Wrong initialization model'
    return checkpoint / 'actor'


def execute(plan, stage='all', resume=False, run_command=None):
    """run_command injection is used by the CPU orchestration integration check."""
    package = Path(__file__).resolve().parent
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if run_command is None:
        def run_command(command, env):
            print('RUN', shlex.join(command), flush=True)
            subprocess.run(command, env=env, check=True)
    env = dict(os.environ, PI05_TRAIN_PYTHON=plan['training_python'], PI05_EVAL_PYTHON=plan['evaluation_python'])
    with lock(output / '.experiment.lock'):
        manifest = output / 'experiment.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == plan, 'Experiment paths/data/batch changed; use a new output directory'
        else:
            save(manifest, plan)
        state = dict(state='running', stage=stage, smoke=plan['smoke'], started=time.time())
        save(output / 'status.json', state)
        try:
            if stage in ['train', 'all']:
                for label, request in plan['training'].items():
                    status_path = Path(request['output']) / 'status.json'
                    if status_path.exists() and json.loads(status_path.read_text())['state'] == 'complete':
                        final_status(request)
                        print('SKIP_COMPLETED_TRAINING', label, flush=True)
                        continue
                    command = list(request['command'])
                    if status_path.exists():
                        assert resume, f'{label} was interrupted; use --resume'
                        assert not plan['smoke'], 'Use a fresh smoke output after interrupted smoke training'
                        checkpoints = [p for p in Path(request['checkpoint']).parent.glob('global_step_*')
                                       if (p / 'COMPLETE.json').is_file()]
                        assert checkpoints, f'No complete checkpoint for {label}; use a fresh experiment output'
                        latest = max(checkpoints, key=lambda p: int(p.name.removeprefix('global_step_')))
                        command += ['--resume', str(latest)]
                    run_command(command, env | {'PI05_GPUS': request['gpus']})
                    final_status(request)
            if stage in ['eval', 'all']:
                checkpoints = {label: str(final_status(request)) for label, request in plan['training'].items()}
                # Different GPU counts are allowed, but both models must have
                # exactly the same initial model parameters before any update.
                digests = []
                for request in plan['training'].values():
                    initial = json.loads((Path(request['output']) / 'initial_weights.json').read_text())
                    assert all(entry['optimizer_steps'] == 0 for entry in initial), 'Step-zero initialization audit is missing'
                    digests.append({entry['model_digest'] for entry in initial})
                assert len(digests[0]) == 1 and digests[0] == digests[1], 'A/B started from different models'
                for label in ['fewshot', 'a60', 'b80']:
                    checkpoint = plan['paths']['base_checkpoint'] if label == 'fewshot' else checkpoints[label]
                    destination = Path(plan['evaluation']['output']) / label
                    command = ['bash', str(package / 'run_eval_parallel.sh'), '--gpus', plan['evaluation']['gpus'],
                               '--label', label, '--checkpoint', checkpoint, '--output', str(destination)]
                    if plan['smoke']:
                        command += ['--tasks', '0', '--trials', '1']
                    if (destination / 'parallel.json').exists():
                        command += ['--resume']
                    run_command(command, env)
                command = [plan['training_python'], str(package / 'compare_results.py'),
                           '--fewshot', str(Path(plan['evaluation']['output']) / 'fewshot'),
                           '--a60', str(Path(plan['evaluation']['output']) / 'a60'),
                           '--b80', str(Path(plan['evaluation']['output']) / 'b80'),
                           '--output', str(output / 'comparison')]
                run_command(command, env)
            state.update(state='complete', finished=time.time())
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), finished=time.time())
            raise
        finally:
            save(output / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'train', 'eval', 'all'], nargs='?', default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two-epoch/15-sample training plus four eval episodes/model')
    args = parser.parse_args()
    plan = build_plan(args.smoke)
    if args.stage == 'plan':
        print('OUTPUT', plan['output'])
        print('PATHS', json.dumps(plan['paths'], indent=2))
        for label, request in plan['training'].items():
            print(label, json.dumps({key: request[key] for key in ['gpus', 'epochs', 'micro_batch', 'global_batch', 'max_updates']}))
            print('DATA', request['dataset']['source_episodes'], 'FRAMES', request['dataset']['frames'])
            print('COMMAND', shlex.join(request['command']))
        print('EVALUATION', json.dumps(plan['evaluation']))
        return
    execute(plan, args.stage, args.resume)


if __name__ == '__main__':
    main()
