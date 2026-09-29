"""Conditional 1-D U-Net over action sequences (Chi et al., Diffusion Policy).

The global condition (diffusion-step embedding + observation features) modulates every
residual block through FiLM: a per-channel scale and bias.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn

SINUSOIDAL_BASE = 10000.0  # diffusion-step embedding base


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(SINUSOIDAL_BASE) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        angles = x[:, None] * freqs[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int, n_groups: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
                Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
            ]
        )
        self.out_channels = out_ch
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 2 * out_ch),  # FiLM scale and bias
            nn.Unflatten(-1, (2 * out_ch, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B, in_ch, T], cond [B, cond_dim] -> [B, out_ch, T]."""
        out = self.blocks[0](x)
        film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        out = film[:, 0] * out + film[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        global_cond_dim: int,
        step_embed_dim: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *down_dims]
        in_out = list(pairwise(all_dims))
        cond_dim = step_embed_dim + global_cond_dim

        def block(i: int, o: int) -> ConditionalResidualBlock1D:
            return ConditionalResidualBlock1D(i, o, cond_dim, kernel_size, n_groups)

        # step MLP expansion factor 4, as in Diffusion Policy
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(step_embed_dim),
            nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(step_embed_dim * 4, step_embed_dim),
        )
        self.down_modules = nn.ModuleList(
            nn.ModuleList(
                [
                    block(dim_in, dim_out),
                    block(dim_out, dim_out),
                    Downsample1d(dim_out) if i < len(in_out) - 1 else nn.Identity(),
                ]
            )
            for i, (dim_in, dim_out) in enumerate(in_out)
        )
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [block(mid_dim, mid_dim), block(mid_dim, mid_dim)]
        )
        # Every up level upsamples.
        self.up_modules = nn.ModuleList(
            nn.ModuleList(
                [
                    block(dim_out * 2, dim_in),
                    block(dim_in, dim_in),
                    Upsample1d(dim_in),
                ]
            )
            for dim_in, dim_out in reversed(in_out[1:])
        )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self, sample: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """sample [B, T, input_dim], timesteps [B] or [], global_cond [B, D]."""
        x = sample.transpose(1, 2)
        timesteps = timesteps.to(sample.device).reshape(-1).expand(sample.shape[0])
        cond = torch.cat([self.diffusion_step_encoder(timesteps), global_cond], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = upsample(resnet2(resnet(x, cond), cond))
        return self.final_conv(x).transpose(1, 2)
