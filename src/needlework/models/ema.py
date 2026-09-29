"""Exponential moving average of model weights with warmup.

decay(step) = clamp(1 - (1 + step / inv_gamma) ** -power, min_value, max_value) with
step = updates so far - 1, and 0 for the first two updates (as in diffusers'
``EMAModel``). Parameters that do not require gradients (buffers such as the
normalizer are not parameters and are copied with the state) are copied, not averaged.
"""

from __future__ import annotations

import torch
from torch import nn


class Ema:
    def __init__(
        self,
        model: nn.Module,
        *,
        inv_gamma: float,
        power: float,
        min_value: float,
        max_value: float,
    ) -> None:
        self.model = model.eval().requires_grad_(False)
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.step_count = 0

    def decay(self) -> float:
        step = max(0, self.step_count - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        decay = self.decay()
        for param, ema_param in zip(
            online.parameters(), self.model.parameters(), strict=True
        ):
            if param.requires_grad:
                ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)
            else:
                ema_param.copy_(param.detach())
        for buffer, ema_buffer in zip(
            online.buffers(), self.model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)
        self.step_count += 1
