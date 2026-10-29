"""Configure a copied LIBERO resource tree inside the active evaluation venv."""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    args = parser.parse_args()
    resources = args.resources.expanduser().resolve(strict=True)
    for folder in ['assets', 'bddl_files', 'init_files']:
        if not (resources/folder).is_dir():
            raise FileNotFoundError(resources/folder)
    spec = importlib.util.find_spec('libero')
    if spec is None:
        raise ModuleNotFoundError('Install requirements-eval.txt in the active venv first')
    core = Path(next(iter(spec.submodule_search_locations)))/'libero'
    if not (core/'__init__.py').is_file():
        raise RuntimeError(f'Unexpected rlinf-libero package layout: {core}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    package_assets = core/'assets'
    target = resources/'assets'
    if package_assets.resolve() != target:
        if package_assets.exists() or package_assets.is_symlink():
            backup = core/f'assets.before_pi05_eval_{stamp}'
            package_assets.rename(backup)
            print(f'Preserved previous venv assets at {backup}')
        package_assets.symlink_to(target, target_is_directory=True)
    config_dir = resources/'config'
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir/'config.yaml'
    if config_file.is_file():
        shutil.copy2(config_file, config_dir/f'config.before_pi05_eval_{stamp}.yaml')
    config = dict(benchmark_root=str(core), assets=str(target),
                  bddl_files=str(resources/'bddl_files'),
                  init_states=str(resources/'init_files'),
                  datasets=str(resources.parent/'dataset'))
    config_file.write_text(yaml.safe_dump(config, sort_keys=True))
    print(f'Wrote {config_file}')
    print(f'export LIBERO_CONFIG_PATH={config_dir}')


if __name__ == '__main__':
    main()
