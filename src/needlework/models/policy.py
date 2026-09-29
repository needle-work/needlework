"""Diffusion over action chunks, conditioned on observations.

Observations are a dict of [B, n_obs, D] tensors: one DINOv3 feature per camera and the
proprioception keys. Per observation step the features are cameras in sorted order
(not normalized) then proprioception in sorted order (normalized). ``DiffusionPolicy``
conditions on the steps concatenated; the IDM (``models/idm.py``) conditions on a
projection of them and of a goal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from needlework.models import diffusion
from needlework.models.normalizer import Normalizer
from needlework.models.unet import ConditionalUnet1D


@dataclass(frozen=True)
class PolicyShape:
    cameras: tuple[str, ...]  # feature keys, each [camera_dim]
    camera_dim: int
    proprio: dict[str, int]  # key -> width
    action_dim: int
    horizon: int  # predicted action rows
    n_obs: int  # observation steps
    n_execute: int  # rows executed per prediction, starting at row n_obs - 1

    @property
    def step_dim(self) -> int:
        return len(self.cameras) * self.camera_dim + sum(self.proprio.values())


class ConditionedDiffusion(nn.Module):
    """Normalizer, U-Net denoiser and DDIM; subclasses define ``condition``."""

    def __init__(
        self,
        *,
        shape: PolicyShape,
        unet: dict,
        scheduler: dict,
        inference_steps: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.inference_steps = inference_steps
        self.normalizer = Normalizer({"action": shape.action_dim, **shape.proprio})
        self.model = ConditionalUnet1D(
            input_dim=shape.action_dim, global_cond_dim=cond_dim, **unet
        )
        self.scheduler = diffusion.make_scheduler(**scheduler)

    def step_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """{key: [B, >= n_obs, D]} -> [B, n_obs, step_dim]."""
        n = self.shape.n_obs
        parts = [obs[key][:, :n] for key in sorted(self.shape.cameras)]
        parts += [
            self.normalizer.normalize(key, obs[key][:, :n])
            for key in sorted(self.shape.proprio)
        ]
        return torch.cat(parts, dim=-1)

    def condition(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self, inputs: Any, action: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        cond = self.condition(inputs)
        return diffusion.masked_epsilon_loss(
            denoiser=lambda x, t: self.model(x, t, cond),
            scheduler=self.scheduler,
            target=self.normalizer.normalize("action", action),
            valid=valid,
        )

    @torch.inference_mode()
    def predict(
        self, inputs: Any, generators: Sequence[torch.Generator]
    ) -> torch.Tensor:
        """All ``horizon`` predicted rows, unnormalized: [B, horizon, action_dim]."""
        cond = self.condition(inputs)
        normalized = diffusion.sample(
            denoiser=lambda x, t: self.model(x, t, cond),
            scheduler=self.scheduler,
            shape=(cond.shape[0], self.shape.horizon, self.shape.action_dim),
            inference_steps=self.inference_steps,
            generators=generators,
            device=cond.device,
        )
        return self.normalizer.unnormalize("action", normalized)

    def executable(self, actions: torch.Tensor) -> torch.Tensor:
        """The rows a controller executes: n_execute rows from the current step on."""
        start = self.shape.n_obs - 1
        return actions[:, start : start + self.shape.n_execute]


class DiffusionPolicy(ConditionedDiffusion):
    def __init__(
        self, *, shape: PolicyShape, unet: dict, scheduler: dict, inference_steps: int
    ) -> None:
        super().__init__(
            shape=shape,
            unet=unet,
            scheduler=scheduler,
            inference_steps=inference_steps,
            cond_dim=shape.n_obs * shape.step_dim,
        )

    def condition(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Observation steps concatenated: [B, n_obs * step_dim]."""
        return self.step_features(obs).flatten(1)
