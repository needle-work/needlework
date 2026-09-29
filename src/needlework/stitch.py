"""Offline stitching: python -m needlework.stitch task=... idm_checkpoint=... ...

Resume: python -m needlework.stitch --resume /path/to/run ['gpus=[0,1]']
    [threshold_offsets.stageN=D ...] [keep_fraction.stageN=F ...]
GPU assignment may change on any resume. Threshold offsets and keep fractions apply
after scoring, so they may change while bridges.zarr does not exist yet. The overrides
are checked, then recorded in the run config. Candidate batches stay fixed.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from needlework import config
from needlework.constants import STAGES
from needlework.stitching.candidates import max_candidates, targets_per_source
from needlework.stitching.pipeline import run
from needlework.stitching.selection import keep_fraction, threshold_offset
from needlework.training import determinism, run_dir


def check(cfg: DictConfig) -> None:
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not cfg.gpus or len(set(cfg.gpus)) != len(cfg.gpus):
        raise ValueError("gpus must be nonempty and unique")
    if any(g < 0 or g >= torch.cuda.device_count() for g in cfg.gpus):
        raise ValueError("GPU ordinal outside visible devices")
    if (
        not cfg.stages
        or len(set(cfg.stages)) != len(cfg.stages)
        or not set(cfg.stages) <= set(STAGES)
    ):
        raise ValueError(f"stages must be a nonempty unique subset of {list(STAGES)}")
    targets = [targets_per_source(cfg, stage) for stage in cfg.stages]
    if (
        min(
            cfg.batch_size,
            cfg.sources_per_stage,
            cfg.candidate_spacing,
            *targets,
            cfg.min_steps_saved,
        )
        < 1
    ):
        raise ValueError(
            "batch size, sources_per_stage, candidate_spacing, targets_per_source and "
            "min_steps_saved must be positive"
        )
    for stage in cfg.stages:
        cap = max_candidates(cfg, stage)
        if cap is not None and cap < 1:
            raise ValueError(
                f"max_candidates_per_stage.stage{stage} must be null or positive: {cap}"
            )
        if not math.isfinite(threshold_offset(cfg, stage)):
            raise ValueError(f"threshold_offsets.stage{stage} must be finite")
        if not 0 < keep_fraction(cfg, stage) <= 1:
            raise ValueError(f"keep_fraction.stage{stage} must be in (0, 1]")
    if not 1 <= cfg.min_votes <= cfg.proposals:
        raise ValueError("need 1 <= min_votes <= proposals")
    if not 0 < cfg.recovery_quantile < 1:
        raise ValueError("recovery_quantile must be in (0, 1)")
    if "/" in cfg.run.name:
        raise ValueError("run.name cannot contain /")


RESUME_USAGE = (
    "usage: --resume /path/to/run [gpus=[0,1]] [threshold_offsets.stageN=D ...] "
    "[keep_fraction.stageN=F ...]"
)
SELECTION_OVERRIDE = re.compile(r"(threshold_offsets|keep_fraction)\.stage(\d+)=\S+")


def resume_config(argv: list[str]) -> tuple[DictConfig, Path]:
    """The saved config of a run with the overrides a resume may change, checked
    before the run config records them."""
    if len(argv) < 2 or argv[0] != "--resume":
        raise ValueError(RESUME_USAGE)
    overrides = argv[2:]
    selection = [o for o in overrides if SELECTION_OVERRIDE.fullmatch(o)]
    if any(not (o.startswith("gpus=") or o in selection) for o in overrides):
        raise ValueError(RESUME_USAGE)
    directory = Path(argv[1]).resolve()
    cfg = OmegaConf.load(directory / run_dir.CONFIG)
    if overrides:
        # a selection override also updates the task entry the setting reads
        task_entries = [f"task.stitch_{o}" for o in selection]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides + task_entries))
    for override in selection:
        stage = int(SELECTION_OVERRIDE.fullmatch(override).group(2))
        if stage not in cfg.stages:
            raise ValueError(
                f"{override}: stage{stage} is not in stages {list(cfg.stages)}"
            )
    check(cfg)
    if selection and (directory / "bridges.zarr").exists():
        raise ValueError(
            f"{directory}: bridges.zarr exists; remove it and summary.json to "
            "select again with new threshold offsets or keep fractions"
        )
    if overrides:
        OmegaConf.save(cfg, directory / run_dir.CONFIG, resolve=True)
    return cfg, directory


def main(argv: list[str]) -> None:
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism.CUBLAS_WORKSPACE
    if not torch.cuda.is_available():
        raise RuntimeError("stitching requires CUDA")
    if argv[:1] == ["--resume"]:
        cfg, directory = resume_config(argv)
    else:
        cfg = config.compose_stitch(argv)
        check(cfg)
        directory = None
    determinism.configure(seed=cfg.seed, strict=True)
    if directory is None:
        directory = run_dir.new_run_dir(
            cfg.task.domain, cfg.task.name, "stitch", cfg.run.name
        )
        OmegaConf.save(cfg, directory / run_dir.CONFIG, resolve=True)
    run_dir.tee_output(directory)
    print(f"run directory: {directory}", flush=True)
    run(cfg, directory)


if __name__ == "__main__":
    main(sys.argv[1:])
