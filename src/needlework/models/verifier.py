"""Verifier: whether an action chunk takes the source to the goal, and by which step.

Tokens: the source and goal frames' DINOv3 7x7 patch grids (every camera), one source
proprioception token (current step, normalized, keys in config order), then one token
per executable action row (normalized). Context tokens attend to context only; action
token ``h`` attends to the context and action tokens ``<= h``. Action tokens carry 1-D
RoPE over their step; patch tokens carry 2-D axial RoPE over their (row, col), the same
for source and goal and for every camera. The output is one logit per action step:
"the goal is reached within the first ``h`` actions".

Per-step decision thresholds are calibrated on validation logits (``calibrate``) and
stored in the verifier's ``last.ckpt``, not in this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from needlework.models.normalizer import Normalizer

SOURCE, GOAL, PROPRIO = 0, 1, 2
ROPE_BASE = 10000.0  # rotary position encoding base


def rope_frequencies(half: int, device: torch.device | None = None) -> torch.Tensor:
    steps = torch.arange(half, device=device, dtype=torch.float32)
    return torch.exp(-math.log(ROPE_BASE) * steps / max(half, 1))


def rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x [B, heads, seq, head_dim] rotated pairwise by angles [seq, head_dim // 2]."""
    half = x.shape[-1] // 2
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half :]], -1)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, ff_mult: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ff_mult),
            nn.GELU(),
            nn.Linear(hidden * ff_mult, hidden),
        )

    def forward(
        self, x: torch.Tensor, allowed: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, hidden = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (batch, seq, self.heads, hidden // self.heads)
        q, k, v = (t.reshape(shape).transpose(1, 2) for t in (q, k, v))
        q, k = rotate(q, angles), rotate(k, angles)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, seq, hidden))
        return x + self.ff(self.norm2(x))


class Verifier(nn.Module):
    def __init__(
        self,
        *,
        n_cameras: int,
        patch_grid: int,
        feature_dim: int,
        proprio: dict[str, int],
        action_dim: int,
        horizon: int,
        n_obs: int,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
    ) -> None:
        super().__init__()
        self.proprio, self.n_obs, self.horizon = proprio, n_obs, horizon
        self.normalizer = Normalizer({"action": action_dim, **proprio})
        self.dino_proj = nn.Linear(feature_dim, hidden)
        self.camera_embedding = nn.Embedding(n_cameras, hidden)
        self.token_type_embedding = nn.Embedding(3, hidden)
        self.proprio_proj = nn.Linear(sum(proprio.values()), hidden)
        self.action_proj = nn.Linear(action_dim, hidden)
        self.action_step_embedding = nn.Embedding(horizon, hidden)
        self.blocks = nn.ModuleList(
            [Block(hidden, heads, ff_mult) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.logit_head = nn.Linear(hidden, 1)
        n_patches = patch_grid * patch_grid
        context = 2 * n_cameras * n_patches + 1
        angles = self._context_angles(context, hidden // heads, patch_grid, n_cameras)
        cameras = torch.arange(n_cameras).repeat_interleave(n_patches)
        self.register_buffer("angles", angles, persistent=False)
        allowed = self._allowed(context, horizon)
        self.register_buffer("allowed", allowed, persistent=False)
        self.register_buffer("camera_ids", cameras, persistent=False)

    @staticmethod
    def _context_angles(
        context: int, head_dim: int, grid: int, cameras: int
    ) -> torch.Tensor:
        """RoPE angles of the context tokens [context, head_dim // 2]: axial over each
        patch's (row, col); zero (no rotation) for the proprioception token."""
        half = head_dim // 2
        angles = torch.zeros(context, half)
        rows = torch.arange(grid).repeat_interleave(grid).float()
        cols = torch.arange(grid).repeat(grid).float()
        row_pairs = half // 2
        per_patch = torch.cat(
            [
                rows[:, None] * rope_frequencies(row_pairs)[None, :],
                cols[:, None] * rope_frequencies(half - row_pairs)[None, :],
            ],
            dim=1,
        )
        scene = per_patch.repeat(cameras, 1)
        angles[: 2 * len(scene)] = scene.repeat(2, 1)
        return angles

    @staticmethod
    def _allowed(context: int, horizon: int) -> torch.Tensor:
        seq = context + horizon
        allowed = torch.ones(seq, seq, dtype=torch.bool)
        allowed[:context, context:] = False
        allowed[context:, context:] = torch.ones(horizon, horizon).tril().bool()
        return allowed

    def _scene(self, patches: torch.Tensor, kind: int) -> torch.Tensor:
        """[B, cameras, patches, feature_dim] -> [B, cameras * patches, hidden]."""
        tokens = self.dino_proj(patches.flatten(1, 2))
        tokens = tokens + self.camera_embedding(self.camera_ids)[None]
        return tokens + self.token_type_embedding.weight[kind]

    def forward(self, batch: dict) -> torch.Tensor:
        """Logits [B, horizon]."""
        step = self.n_obs - 1
        proprio = torch.cat(
            [
                self.normalizer.normalize(key, batch["obs"][key][:, step])
                for key in self.proprio
            ],
            dim=-1,
        )
        proprio = self.proprio_proj(proprio) + self.token_type_embedding.weight[PROPRIO]
        context = torch.cat(
            [
                self._scene(batch["source_patches"], SOURCE),
                self._scene(batch["goal_patches"], GOAL),
                proprio[:, None],
            ],
            dim=1,
        )
        actions = self.normalizer.normalize("action", batch["action"][:, step:])
        steps = torch.arange(self.horizon, device=actions.device)
        actions = self.action_proj(actions) + self.action_step_embedding(steps)
        half = self.angles.shape[1]
        step_angles = steps[:, None].float() * rope_frequencies(half, steps.device)
        angles = torch.cat([self.angles, step_angles])
        x = torch.cat([context, actions], dim=1)
        for block in self.blocks:
            x = block(x, self.allowed, angles)
        out = self.final_norm(x[:, context.shape[1] :])
        return self.logit_head(F.gelu(out)).squeeze(-1)


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean BCE over positive entries, averaged with the mean over rows of each row's
    mean BCE over its negative entries (a term that is absent is left out)."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    positive, negative = labels == 1.0, labels == 0.0
    terms = []
    if positive.any():
        terms.append(bce[positive].mean())
    rows = negative.any(1)
    if rows.any():
        per_row = (bce * negative).sum(1) / negative.sum(1).clamp(min=1)
        terms.append(per_row[rows].mean())
    return torch.stack(terms).mean()


def calibrate(
    logits: torch.Tensor, labels: torch.Tensor, *, target_fpr: float, min_negatives: int
) -> torch.Tensor:
    """Per step: the (1 - target_fpr) quantile of the negatives' logits."""
    thresholds = []
    for h in range(logits.shape[1]):
        negatives = logits[labels[:, h] == 0.0, h]
        if len(negatives) < min_negatives:
            raise ValueError(
                f"step {h + 1}: {len(negatives)} negatives < {min_negatives}"
            )
        thresholds.append(torch.quantile(negatives.float(), 1.0 - target_fpr))
    return torch.stack(thresholds)
