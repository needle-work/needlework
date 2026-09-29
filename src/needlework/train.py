"""Train a component, or resume a run in place.

    python -m needlework.train component=policy task=robomimic/square run.name=square_dp
    python -m needlework.train component=policy task=umi task.name=sweater run.name=sw
    python -m needlework.train component=idm task=robomimic/square run.name=square_idm
    python -m needlework.train component=verifier task=robomimic/square run.name=sq_ver
    python -m needlework.train --resume <run directory>

A resume continues from the run's checkpoint with the run's saved config; a run that
stopped before its first checkpoint restarts from epoch 0 in the same directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from needlework import config, constants
from needlework.training import checkpoint, determinism, run_dir, wandb_log
from needlework.training.engine import Trainer
from needlework.training.idm_task import IdmTask
from needlework.training.policy_task import PolicyTask
from needlework.training.verifier_task import VerifierTask

TASKS = {"policy": PolicyTask, "idm": IdmTask, "verifier": VerifierTask}


def main(argv: list[str]) -> None:
    # Read by cuBLAS when its first handle is created; a different value is an error.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism.CUBLAS_WORKSPACE
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA device")
    device = torch.device("cuda")
    if argv[:1] == ["--resume"]:
        if len(argv) != 2:
            raise SystemExit("usage: train.py --resume <run directory>")
        directory = Path(argv[1]).resolve()
        cfg = config.load(directory)
        ckpt_path = checkpoint.path_in(directory)
        payload = checkpoint.load(ckpt_path) if ckpt_path.exists() else None
        if payload is not None:
            if "constants" not in payload:
                raise ValueError(
                    f"{ckpt_path} has no constants snapshot; cannot resume"
                )
            constants.check_snapshot(payload["constants"], str(ckpt_path))
    else:
        cfg = config.compose(argv)
        directory, payload = None, None
    if cfg.logging.wandb.enabled:
        wandb_log.require_login()
    if directory is None:
        directory = run_dir.new_run_dir(
            cfg.task.domain, cfg.task.name, cfg.component.name, cfg.run.name
        )
        config.save(cfg, directory)
    run_dir.tee_output(directory)
    print(f"run directory: {directory}", flush=True)

    generator = determinism.configure(
        seed=cfg.seed.training, strict=cfg.determinism.strict
    )
    task = TASKS[cfg.component.name](cfg, device=device, fit=payload is None)
    trainer = Trainer(task, cfg, device=device, data_generator=generator)
    if payload is not None:
        if payload["data_identity"] != task.identity:
            raise ValueError("the data under NEEDLEWORK_ROOT differs from this run's")
        trainer.load_state(payload)
        print(f"resumed after epoch {trainer.epoch}", flush=True)
    run_dir.truncate_metrics(directory, trainer.epoch)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    wandb_run = None
    if cfg.logging.wandb.enabled:
        wandb_run = wandb_log.start(
            directory,
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=directory.name,
            config=resolved,
        )

    def log(record: dict) -> None:
        run_dir.append_metrics(directory, record)
        if wandb_run is not None:
            wandb_run.log(record)  # epoch and global_step are fields of the record
        print(record, flush=True)

    def save(state: dict, metrics: dict) -> None:
        payload = {
            **state,
            "config": resolved,
            "data_identity": task.identity,
            "constants": constants.snapshot(),
        }
        if cfg.component.name == "verifier":
            payload["thresholds"] = task.thresholds
        path = checkpoint.save(directory, payload)
        checkpoint.retain_policy(directory, payload, metrics)
        print(f"saved {path} after epoch {state['epoch']}", flush=True)

    trainer.run(log=log, save=save)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
