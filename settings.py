"""One configuration entry for A/B; no baked-in server root."""
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_config(experiment):
    if experiment not in ('a60', 'b80'):
        raise ValueError(experiment)
    with initialize_config_dir(version_base='1.1', config_dir=str(Path(__file__).resolve().parent / 'config')):
        cfg = compose(config_name=experiment)
    OmegaConf.resolve(cfg)
    return cfg
