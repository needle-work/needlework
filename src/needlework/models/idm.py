"""Inverse dynamics model: (observation history, goal frame) -> action chunk.

The diffusion policy's U-Net and DDIM, conditioned on
    cond_mlp([obs_mlp(observation steps concatenated), goal_mlp(goal camera features)])
where the goal contributes camera features only (sorted cameras, not normalized), never
proprioception. Each MLP is Linear -> ReLU -> ... -> Linear over ``hidden``.
"""

from __future__ import annotations

import torch
from torch import nn

from needlework.models.policy import ConditionedDiffusion, PolicyShape


def mlp(input_dim: int, dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, dim in enumerate(dims):
        layers.append(nn.Linear(input_dim, dim))
        if index < len(dims) - 1:
            layers.append(nn.ReLU())
        input_dim = dim
    return nn.Sequential(*layers)


class IdmPolicy(ConditionedDiffusion):
    def __init__(
        self,
        *,
        shape: PolicyShape,
        unet: dict,
        scheduler: dict,
        inference_steps: int,
        hidden: tuple[int, ...],
    ) -> None:
        super().__init__(
            shape=shape,
            unet=unet,
            scheduler=scheduler,
            inference_steps=inference_steps,
            cond_dim=hidden[-1],
        )
        goal_dim = len(shape.cameras) * shape.camera_dim
        self.obs_mlp = mlp(shape.n_obs * shape.step_dim, hidden)
        self.goal_mlp = mlp(goal_dim, hidden)
        self.cond_mlp = mlp(2 * hidden[-1], hidden)

    def condition(self, inputs: dict) -> torch.Tensor:
        """inputs: {"obs": {key: [B, n_obs, D]}, "goal": {camera: [B, D]}}."""
        goal = torch.cat(
            [inputs["goal"][key] for key in sorted(self.shape.cameras)], dim=-1
        )
        obs = self.obs_mlp(self.step_features(inputs["obs"]).flatten(1))
        return self.cond_mlp(torch.cat([obs, self.goal_mlp(goal)], dim=-1))
