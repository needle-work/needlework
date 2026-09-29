"""component=policy: data, model and per-epoch metrics of a diffusion policy.

Training and validation windows come from success episodes. The normalizer is fit on
the training windows of a new run and restored from the checkpoint on resume.
Robomimic policies are also rolled out in simulation every ``task.eval.every`` epochs.
With ``sampler=augmented`` the training windows also include stitched bridges; the
normalizer and validation still use logged success windows only.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from needlework import paths
from needlework.constants import SPATIAL
from needlework.data import bridges
from needlework.data.download import HDF5_NAME
from needlework.models.dinov3 import POOLINGS, Dinov3
from needlework.models.policy import DiffusionPolicy, PolicyShape
from needlework.sampling import roles
from needlework.sampling.augmented_dataset import AugmentedPolicyDataset, SamplerOptions
from needlework.sampling.policy_dataset import PolicyDataset
from needlework.sim import rollout
from needlework.training import validation
from needlework.training.common import Windows


def policy_shape(
    cfg: DictConfig, windows: Windows, proprio: dict, n_execute: int
) -> PolicyShape:
    return PolicyShape(
        cameras=tuple(cfg.task.obs.cameras),
        camera_dim=POOLINGS[SPATIAL][0],
        proprio=proprio,
        action_dim=windows.store.spec.action_dim,
        horizon=cfg.horizon.prediction,
        n_obs=cfg.horizon.obs,
        n_execute=n_execute,
    )


def augmented(cfg: DictConfig, logged: PolicyDataset) -> AugmentedPolicyDataset:
    table = bridges.load(
        Path(cfg.sampler.bridges),
        logged.store,
        horizon=logged.shape.horizon - logged.shape.n_obs + 1,
        stride=logged.shape.stride,
    )
    return AugmentedPolicyDataset(
        logged,
        table,
        weight=cfg.sampler.bridge_weight,
        role_weights=roles.RoleWeights(**cfg.sampler.role_weights),
        epoch_length=cfg.sampler.epoch_length,
        options=SamplerOptions(**cfg.sampler.options),
        seed=cfg.seed.training,
    )


class PolicyTask:
    def __init__(self, cfg: DictConfig, *, device: torch.device, fit: bool) -> None:
        self.cfg, self.device = cfg, device
        comp = cfg.component
        self.windows = w = Windows(cfg, cameras=True, n_execute=cfg.horizon.execute)
        episodes = w.episodes(successes_only=True)
        self.train_data = w.dataset(episodes.train, train=True)
        if cfg.sampler.name == "augmented":
            self.train_data = augmented(cfg, self.train_data)
        self.val_data = w.dataset(episodes.val, train=False)
        train_probe = w.dataset(episodes.train, train=False)  # metrics: no noise
        self.model = DiffusionPolicy(
            shape=policy_shape(
                cfg, w, w.proprio_widths(train_probe), cfg.horizon.execute
            ),
            unet=dict(comp.unet),
            scheduler=dict(comp.scheduler),
            inference_steps=comp.inference_steps,
        )
        if fit:
            self.model.normalizer.set(w.normalizer(train_probe))
        size = comp.validation.action_mse_windows
        self.probes = validation.probes(train=train_probe, val=self.val_data, size=size)
        self.encoder: Dinov3 | None = None  # loaded at the first evaluation

    @property
    def identity(self) -> dict[str, str]:
        identity = dict(self.windows.store.identity)
        if isinstance(self.train_data, AugmentedPolicyDataset):
            identity["bridges"] = self.train_data.bridges.digest()
        return identity

    @staticmethod
    def inputs(batch: dict) -> dict:
        return batch["obs"]

    def loss(self, model: DiffusionPolicy, batch: dict) -> torch.Tensor:
        return model.loss(self.inputs(batch), batch["action"], batch["valid"])

    def epoch_metrics(self, ema_model: DiffusionPolicy, epoch: int) -> dict[str, float]:
        cfg = self.cfg
        out = {
            "val_loss": validation.mean_loss(
                lambda batch: self.loss(ema_model, batch),
                self.val_data,
                batch_size=cfg.component.train.batch_size,
                seed=cfg.seed.training,
                device=self.device,
            )
        }
        if epoch % cfg.component.validation.action_mse_every == 0:
            out |= validation.action_mse_metrics(
                ema_model,
                self.probes,
                self.inputs,
                batch_size=cfg.component.train.batch_size,
                seed=cfg.seed.training,
                device=self.device,
            )
        if cfg.task.domain == "robomimic" and epoch % cfg.task.eval.every == 0:
            out |= rollout.summarize(self.rollouts(ema_model))
        return out

    def rollouts(self, policy: DiffusionPolicy) -> list[dict]:
        task = self.cfg.task
        if self.encoder is None:
            self.encoder = Dinov3(self.device)
        return rollout.evaluate(
            policy,
            self.encoder,
            pooling=SPATIAL,
            grid=rollout.EvalGrid(
                seeds=tuple(task.eval.seeds),
                episodes_per_seed=task.eval.episodes_per_seed,
                n_envs=task.eval.n_envs,
                max_steps=task.eval.max_steps,
                construction_seed=task.eval.construction_seed,
            ),
            hdf5=paths.data_dir() / "robomimic" / task.name / HDF5_NAME,
            proprio=tuple(task.obs.proprio),
            arms=self.windows.store.spec.arms,
            device=self.device,
        )
