"""Check the two pinned environments without allocating model/GPU memory."""
import argparse
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


RLINF_COMMIT = 'c70606f08cdca259b8dec03d4430926b5b8fac9d'
OPENPI_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--root', type=Path, default=os.environ.get('PI05_ROOT'))
    parser.add_argument('--resources', action='store_true', help='Also require data, norm stats, tokenizer and LIBERO resources')
    parser.add_argument('--cuda', action='store_true', help='Also check driver/device access; initializes a CUDA context')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.root is None:
        parser.error('Set PI05_ROOT or pass --root')
    root = args.root.expanduser().resolve(strict=True)
    os.environ['PI05_ROOT'] = str(root)
    assert sys.version_info[:2] == (3, 12), sys.version
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    os.environ.update(JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                      TOKENIZERS_PARALLELISM='false', LIBERO_TYPE='standard', ROBOT_PLATFORM='LIBERO')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    sys.path.insert(0, str(root / 'RLinf'))
    source_commits = {}
    repos = {'RLinf': RLINF_COMMIT}
    if args.mode == 'eval':
        repos['openpi'] = OPENPI_COMMIT
        # Same source lookup order as run_eval.sh; no rlinf-openpi in eval.
        sys.path[1:1] = [str(root / 'openpi/src'), str(root / 'openpi/packages/openpi-client/src')]
        config = root / 'libero-resources/config'
        assert (config / 'config.yaml').is_file(), 'Run configure_resources.py before the eval import check'
        os.environ['LIBERO_CONFIG_PATH'] = str(config)
    for name, expected in repos.items():
        actual = subprocess.check_output(['git', '-C', str(root / name), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == expected, f'{name} commit: expected {expected}, got {actual}'
        source_commits[name] = actual
    versions = dict(numpy='1.26.4', transformers='4.53.2', jax='0.5.3',
                    jaxlib='0.5.3', flax='0.10.2', ray='2.59.0', torchdata='0.11.0',
                    **{'hydra-core': '1.3.7', 'omegaconf': '2.3.1'})
    if args.mode == 'train':
        versions.update(torch='2.7.1', torchvision='0.22.1', datasets='3.6.0', pyarrow='19.0.1',
                        **{'rlinf-openpi': '0.1.1', 'rlinf-transformer-openpi': '4.53.2', 'lerobot': '0.1.0'})
        dist = metadata.distribution('rlinf-transformer-openpi')
        item = next(p for p in dist.files if str(p) == 'transformers/models/gemma/modeling_gemma.py')
        digest = base64.urlsafe_b64encode(hashlib.sha256(dist.locate_file(item).read_bytes()).digest()).decode().rstrip('=')
        assert digest == item.hash.value, 'OpenPI Transformers files were overwritten; rerun install_environment.sh train'
        lerobot = json.loads(metadata.distribution('lerobot').read_text('direct_url.json'))
        assert lerobot['vcs_info']['commit_id'] == '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
    else:
        versions.update(torch='2.6.0+cu124', torchvision='0.21.0+cu124',
                        **{'rlinf-libero': '0.1.3', 'robosuite': '1.4.1', 'mujoco': '3.3.7'})
    actual_versions = {}
    for name, expected in versions.items():
        actual = metadata.version(name)
        assert actual.split('+')[0] == expected.split('+')[0], f'{name}: expected {expected}, got {actual}'
        actual_versions[name] = actual
    imports = ['torch', 'torchvision', 'openpi.training.config', 'rlinf.config',
               'rlinf.models.embodiment.openpi.pi0', 'transformers.models.gemma.modeling_gemma']
    if args.mode == 'train':
        imports += ['lerobot.common.datasets.lerobot_dataset', 'openpi.training.data_loader',
                    'rlinf.workers.sft.fsdp_vla_sft_worker', 'training_runtime']
    else:
        imports += ['libero.libero.benchmark', 'robosuite', 'mujoco', 'imageio.v2',
                    'OpenGL.EGL', 'rlinf.envs.sim.libero.libero_env']
    for name in imports:
        importlib.import_module(name)
    if args.resources:
        from settings import load_paths
        norm = Path(load_paths().norm_stats)
        assert norm.is_file(), norm
        cache = Path(os.environ.get('OPENPI_DATA_HOME', str(Path.home() / '.cache/openpi'))).expanduser()
        assert (cache / 'big_vision/paligemma_tokenizer.model').is_file(), cache
        if args.mode == 'train':
            from settings import load_config
            for experiment in ['a60', 'b80']:
                for source in load_config(experiment).data.sources:
                    folder = Path(source.path)
                    for name in ['info.json', 'tasks.jsonl', 'episodes.jsonl']:
                        assert (folder / 'meta' / name).is_file(), folder / 'meta' / name
                    if source.get('episodes_file'):
                        assert (folder / source.episodes_file).is_file()
        else:
            import yaml
            config = yaml.safe_load((root / 'libero-resources/config/config.yaml').read_text())
            for name, folder in [('assets', 'assets'), ('bddl_files', 'bddl_files'), ('init_states', 'init_files')]:
                path = root / 'libero-resources' / folder
                assert path.is_dir() and any(path.iterdir()), path
                assert Path(config[name]).resolve() == path.resolve(), f'Stale LIBERO {name} path'
            from libero.libero import benchmark
            for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
                tasks = benchmark.get_benchmark_dict()[suite]()
                assert tasks.get_num_tasks() == 10
                for task in range(10):
                    assert len(tasks.get_task_init_states(task)) >= 50
    import torch
    device_count = None
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA driver/device unavailable'
        device_count = torch.cuda.device_count()
        assert torch.distributed.is_nccl_available(), 'NCCL backend unavailable'
    report = dict(state='complete', mode=args.mode, python=sys.version.split()[0],
                  versions=actual_versions, source_commits=source_commits, imports=imports,
                  resources_checked=args.resources, cuda_checked=args.cuda,
                  cuda_device_count=device_count, torch_cuda_build=torch.version.cuda,
                  model_loaded=False, training_started=False, rollout_started=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
