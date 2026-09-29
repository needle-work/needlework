"""Final component weights and batched source/goal proposal scoring.

Components are read from their runs' ``checkpoints/last.ckpt`` through
``checkpoint.load_inference``: memory-mapped, keeping only the EMA weights, config, data
identity and verifier thresholds, so optimizer state is never read. Feature caches are
memory-mapped too, so every worker process on a host shares one copy of them in the page
cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from needlework.constants import (
    DINOV3_FEATURE_DIM,
    PATCHES,
    POOLED_GRID,
    SPATIAL,
)
from needlework.data import features
from needlework.data.store import EpisodeStore
from needlework.models.dinov3 import POOLINGS
from needlework.models.idm import IdmPolicy
from needlework.models.policy import PolicyShape
from needlework.models.verifier import Verifier
from needlework.sampling.observations import RobomimicObservations, UmiObservations
from needlework.stitching.candidates import max_prefix_lengths
from needlework.training import checkpoint
from needlework.training.batches import to_device


def load_component(
    path: Path, name: str, store: EpisodeStore, device: torch.device
) -> tuple[torch.nn.Module, DictConfig, torch.Tensor | None]:
    """Model, config and, for the verifier, its calibrated per-step thresholds."""
    payload = checkpoint.load_inference(path)
    cfg = OmegaConf.create(payload["config"])
    if cfg.component.name != name or payload["epoch"] != cfg.component.train.epochs:
        raise ValueError(
            f"{path}: expected a completed {name} run, got {cfg.component.name} at "
            f"epoch {payload['epoch']} of {cfg.component.train.epochs}"
        )
    if (cfg.task.domain, cfg.task.name) != (store.domain, store.task):
        raise ValueError(f"{path}: checkpoint task differs from input store")
    if payload["data_identity"] != store.identity:
        raise ValueError(f"{path}: checkpoint data identity differs from input store")
    state = payload["ema"]
    widths = {k: state[f"normalizer.{k}__scale"].numel() for k in cfg.task.obs.proprio}
    h = cfg.horizon.prediction - cfg.horizon.obs + 1
    if name == "idm":
        shape = PolicyShape(
            tuple(cfg.task.obs.cameras),
            POOLINGS[SPATIAL][0],
            widths,
            store.spec.action_dim,
            cfg.horizon.prediction,
            cfg.horizon.obs,
            h,
        )
        model = IdmPolicy(
            shape=shape,
            unet=dict(cfg.component.unet),
            scheduler=dict(cfg.component.scheduler),
            inference_steps=cfg.component.inference_steps,
            hidden=tuple(cfg.component.hidden),
        )
    elif name == "verifier":
        model = Verifier(
            n_cameras=len(cfg.task.obs.cameras),
            patch_grid=POOLED_GRID,
            feature_dim=DINOV3_FEATURE_DIM,
            proprio=widths,
            action_dim=store.spec.action_dim,
            horizon=h,
            n_obs=cfg.horizon.obs,
            **cfg.component.model,
        )
    else:
        raise ValueError(name)
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    if name != "verifier":
        return model, cfg, None
    thresholds = payload["thresholds"].to(device)
    if not torch.isfinite(thresholds).all():
        raise ValueError(f"{path}: verifier thresholds are not finite")
    return model, cfg, thresholds


def check_compatibility(
    idm: DictConfig, verifier: DictConfig, stitch: DictConfig
) -> None:
    """Compare inference semantics, not component training/evaluation recipes."""
    for key in (
        "horizon.prediction",
        "horizon.obs",
        "task.stride",
        "task.relative_actions",
        "task.obs.cameras",
        "task.obs.proprio",
        "task.proprio_identity",
        "task.action_identity",
    ):
        if OmegaConf.select(idm, key, throw_on_missing=True) != OmegaConf.select(
            verifier, key, throw_on_missing=True
        ):
            raise ValueError(f"component checkpoint mismatch: {key}")
    for key in ("domain", "name", "stride", "relative_actions", "obs"):
        if idm.task[key] != stitch.task[key]:
            raise ValueError(f"stitch task differs from model task: {key}")


def proposal_seed(seed: int, row: np.ndarray, sample: int) -> int:
    # Endpoint identity, not candidate enumeration, GPU, batch, or completion order.
    return int(
        np.random.SeedSequence([seed, *row[1:].tolist(), sample]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


class Scorer:
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.store = store = EpisodeStore.open(cfg.task.domain, cfg.task.name)
        self.idm, self.model_cfg, _ = load_component(
            Path(cfg.idm_checkpoint), "idm", store, device
        )
        self.verifier, verifier_cfg, self.thresholds = load_component(
            Path(cfg.verifier_checkpoint), "verifier", store, device
        )
        check_compatibility(self.model_cfg, verifier_cfg, cfg)
        task = self.model_cfg.task
        self.spatial = features.open_rows(store, SPATIAL)
        self.patches = features.open_rows(store, PATCHES)
        keys = (*task.obs.cameras, *task.obs.proprio)
        self.observations = (
            RobomimicObservations(store, self.spatial, keys)
            if task.domain == "robomimic"
            else UmiObservations(store, self.spatial, keys, task.start_noise)
        )
        self.horizon = (
            self.model_cfg.horizon.prediction - self.model_cfg.horizon.obs + 1
        )

    @torch.inference_mode()
    def score(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Every proposal's per-prefix logits and executable actions, and each row's
        longest eligible prefix."""
        source, goal = rows[:, 1], rows[:, 2]
        shape, task = self.model_cfg.horizon, self.model_cfg.task
        episodes = self.store.episode_of_frame(source)
        frames = source[:, None] + (np.arange(shape.obs) - shape.obs + 1) * task.stride
        frames = np.maximum(frames, self.store.episode_starts[episodes, None])
        obs = self.observations(frames, episodes, False)
        inputs = to_device(
            {
                "obs": {k: torch.from_numpy(v) for k, v in obs.items()},
                "goal": {
                    c: torch.from_numpy(self.spatial[c][goal]) for c in task.obs.cameras
                },
            },
            self.device,
        )
        patch = {
            kind: torch.from_numpy(
                np.stack([self.patches[c][idx] for c in task.obs.cameras], 1)
            ).to(self.device)
            for kind, idx in (("source_patches", source), ("goal_patches", goal))
        }
        actions, logits = [], []
        for sample in range(self.cfg.proposals):
            generators = [
                torch.Generator(device=self.device).manual_seed(
                    proposal_seed(self.cfg.seed, row, sample)
                )
                for row in rows
            ]
            action = self.idm.predict(inputs, generators)
            output = self.verifier({"obs": inputs["obs"], "action": action, **patch})
            actions.append(action[:, shape.obs - 1 :].cpu().numpy())
            logits.append(output.cpu().numpy())
        return {
            "logits": np.stack(logits, 1).astype(np.float32),
            "actions": np.stack(actions, 1).astype(np.float32),
            "max_length": max_prefix_lengths(
                self.store,
                rows,
                horizon=self.horizon,
                stride=task.stride,
                min_steps_saved=self.cfg.min_steps_saved,
            ),
        }
