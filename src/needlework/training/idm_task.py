"""component=idm: goal-conditioned diffusion over action chunks.

Pairs come from all training episodes, failures included; ``val_ratio`` of all episodes
is held out. Each epoch draws a new goal per window (``sampling/pairs.py``); validation
pairs are drawn once. The IDM proposes the ``H`` executable rows after the current step.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from needlework.models.idm import IdmPolicy
from needlework.sampling.pairs import IdmPairs, action_horizon
from needlework.sampling.policy_dataset import PolicyDataset
from needlework.training import validation
from needlework.training.common import Windows
from needlework.training.policy_task import policy_shape


class IdmTask:
    def __init__(self, cfg: DictConfig, *, device: torch.device, fit: bool) -> None:
        self.cfg, self.device = cfg, device
        comp, seed = cfg.component, cfg.seed.dataset
        # windows are padded like the policy's; the IDM proposes every row after the
        # current step
        self.windows = w = Windows(cfg, cameras=True, n_execute=cfg.horizon.execute)
        episodes = w.episodes(successes_only=False)
        train_windows = w.dataset(episodes.train, train=True)
        train_probe = w.dataset(episodes.train, train=False)  # metrics: no noise
        val_windows = w.dataset(episodes.val, train=False)
        n_execute = action_horizon(train_windows)

        def pairs(data: PolicyDataset, resample: bool) -> IdmPairs:
            return IdmPairs(data, w.features, seed=seed, resample=resample)

        self.train_data = pairs(train_windows, resample=True)
        self.val_data = pairs(val_windows, resample=False)
        probe = pairs(train_probe, resample=False)
        self.model = IdmPolicy(
            shape=policy_shape(cfg, w, w.proprio_widths(train_probe), n_execute),
            unet=dict(comp.unet),
            scheduler=dict(comp.scheduler),
            inference_steps=comp.inference_steps,
            hidden=tuple(comp.hidden),
        )
        if fit:
            self.model.normalizer.set(w.normalizer(train_probe))
        size = comp.validation.action_mse_windows
        self.probes = validation.probes(train=probe, val=self.val_data, size=size)

    @property
    def identity(self) -> dict[str, str]:
        return self.windows.store.identity

    @staticmethod
    def inputs(batch: dict) -> dict:
        return {"obs": batch["obs"], "goal": batch["goal"]}

    def loss(self, model: IdmPolicy, batch: dict) -> torch.Tensor:
        return model.loss(self.inputs(batch), batch["action"], batch["valid"])

    def epoch_metrics(self, ema_model: IdmPolicy, epoch: int) -> dict[str, float]:
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
        return out
