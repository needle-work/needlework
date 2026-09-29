"""DDIM noise schedule, masked epsilon loss, and seeded sampling (shared by the policy
and the IDM)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def make_scheduler(
    *, train_steps: int, beta_start: float, beta_end: float, beta_schedule: str
) -> DDIMScheduler:
    """Epsilon-prediction DDIM with sample clipping."""
    return DDIMScheduler(
        num_train_timesteps=train_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=True,
        prediction_type="epsilon",
    )


def masked_epsilon_loss(
    *,
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler: DDIMScheduler,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Diffusion loss on normalized targets [B, H, D] with a validity mask [B, H].

    Masked rows get pure noise as input and no loss. The loss is the mean over each
    window's valid elements, then the mean over windows, so every window weighs the same
    however many of its rows are valid.
    """
    if valid.shape != target.shape[:2] or valid.dtype != torch.bool:
        raise ValueError(f"valid must be bool {tuple(target.shape[:2])}")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every window needs at least one valid action row")
    noise = torch.randn(target.shape, device=target.device)
    steps = scheduler.config.num_train_timesteps
    timesteps = torch.randint(0, steps, (target.shape[0],), device=target.device)
    noisy = scheduler.add_noise(target, noise, timesteps)
    noisy = torch.where(valid[..., None], noisy, noise)
    error = (denoiser(noisy, timesteps) - noise) ** 2
    mask = valid[..., None].expand_as(error).to(error.dtype)
    per_window = (error * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
    return per_window.mean()


def sample(
    *,
    denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler: DDIMScheduler,
    shape: tuple[int, ...],
    inference_steps: int,
    generators: Sequence[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Denoise from Gaussian noise. Row i's noise comes only from ``generators[i]``,
    so a row's sample does not depend on what else is in the batch."""
    if len(generators) != shape[0]:
        raise ValueError(f"need one generator per row: {len(generators)} != {shape[0]}")
    x = torch.cat(
        [torch.randn((1, *shape[1:]), device=device, generator=g) for g in generators]
    )
    scheduler.set_timesteps(inference_steps)
    for t in scheduler.timesteps:
        x = scheduler.step(denoiser(x, t), t, x).prev_sample
    return x
