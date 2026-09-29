"""Per-key affine normalization, fit on training data and stored with the model.

normalized = x * scale + offset, per feature dimension. ``minmax`` maps the training
range to [-1, 1]; a dimension whose range is below 1e-7 is shifted to 0 instead of
scaled. ``identity`` leaves the dimension unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

RANGE_EPS = 1e-7


def minmax_params(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(scale, offset) mapping each column of [N, D] values to [-1, 1]."""
    low, high = values.min(axis=0), values.max(axis=0)
    span = high - low
    flat = span < RANGE_EPS
    scale = 2.0 / np.where(flat, 2.0, span)
    offset = np.where(flat, -low, -1.0 - scale * low)
    return scale.astype(np.float32), offset.astype(np.float32)


def fit_params(values: np.ndarray, identity_dims: np.ndarray) -> dict[str, np.ndarray]:
    """Min-max on every column except ``identity_dims`` (bool [D])."""
    scale, offset = minmax_params(values)
    scale[identity_dims] = 1.0
    offset[identity_dims] = 0.0
    return {"scale": scale, "offset": offset}


class Normalizer(nn.Module):
    """Holds (scale, offset) per key as non-trainable buffers."""

    def __init__(self, dims: dict[str, int]) -> None:
        super().__init__()
        self.keys = tuple(dims)
        for key, dim in dims.items():
            self.register_buffer(f"{key}__scale", torch.ones(dim))
            self.register_buffer(f"{key}__offset", torch.zeros(dim))

    def set(self, params: dict[str, dict[str, np.ndarray]]) -> None:
        if set(params) != set(self.keys):
            raise ValueError(f"normalizer keys {sorted(params)} != {sorted(self.keys)}")
        for key, value in params.items():
            getattr(self, f"{key}__scale").copy_(torch.as_tensor(value["scale"]))
            getattr(self, f"{key}__offset").copy_(torch.as_tensor(value["offset"]))

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return x * getattr(self, f"{key}__scale") + getattr(self, f"{key}__offset")

    def unnormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return (x - getattr(self, f"{key}__offset")) / getattr(self, f"{key}__scale")
