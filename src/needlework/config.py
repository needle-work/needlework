"""Compose, check, save and reload run configs.

Configs are composed with Hydra's compose API from ``needlework/configs``; the run
directory, not Hydra, owns outputs. A resumed training run reads the config saved in its
run directory and accepts no overrides; the stitch resume's permitted overrides are
handled in ``stitch.py``.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose as hydra_compose
from hydra import initialize_config_module
from omegaconf import DictConfig, OmegaConf

from needlework.training.run_dir import CONFIG

CONFIG_MODULE = "needlework.configs"
HYDRA_VERSION_BASE = "1.2"


def _compose(root: str, overrides: list[str]) -> DictConfig:
    with initialize_config_module(CONFIG_MODULE, version_base=HYDRA_VERSION_BASE):
        return hydra_compose(root, overrides=overrides)


def compose(overrides: list[str]) -> DictConfig:
    """A training run's config (``configs/train.yaml``)."""
    cfg = _compose("train", overrides)
    check(cfg)
    return cfg


def compose_stitch(overrides: list[str]) -> DictConfig:
    """A stitching run's config (``configs/stitch.yaml``); checked by ``stitch.py``."""
    return _compose("stitch", overrides)


def check(cfg: DictConfig) -> None:
    """The few cross-group rules; every value set and resolvable."""
    if cfg.sampler.name != "normal" and cfg.component.name != "policy":
        raise ValueError(f"sampler={cfg.sampler.name} applies to component=policy only")
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if "/" in cfg.run.name:
        raise ValueError(f"run.name must not contain '/': {cfg.run.name}")


def save(cfg: DictConfig, run_dir: Path) -> None:
    OmegaConf.save(cfg, run_dir / CONFIG, resolve=True)


def load(run_dir: Path) -> DictConfig:
    cfg = OmegaConf.load(run_dir / CONFIG)
    check(cfg)
    return cfg
