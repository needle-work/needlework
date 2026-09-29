"""Frozen DINOv3 ViT-B/16 and the two poolings of its patch tokens.

uint8 image / 255 -> ImageNet normalization -> ``forward_features`` in float32 ->
``x_norm_patchtokens`` (PATCH_GRID x PATCH_GRID x 768 at 224 x 224) -> pooling.

Features are computed the same way for the training caches and during simulator
rollouts: the HWC images viewed as NCHW, without a copy. A contiguous copy selects
other convolution kernels and shifts the features slightly, so the view is used in both
places.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from needlework import paths
from needlework.constants import (
    DINOV3_FEATURE_DIM,
    DINOV3_MODEL,
    DINOV3_WEIGHTS_FILE,
    IMAGE_SHAPE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_GRID,
    PATCHES,
    POOLED_GRID,
    SPATIAL,
)

# Feature shape per frame and camera.
POOLINGS = {
    SPATIAL: (2 * DINOV3_FEATURE_DIM,),
    PATCHES: (POOLED_GRID * POOLED_GRID, DINOV3_FEATURE_DIM),
}


def spatial_softmax(tokens: torch.Tensor) -> torch.Tensor:
    """[N, PATCH_GRID ** 2, C] -> [N, 2C]: per channel, softmax over the patch
    positions, then the expected (x, y) position in [-1, 1]. All x values first."""
    weights = torch.softmax(tokens.permute(0, 2, 1), dim=-1)  # [N, C, 196]
    axis = torch.linspace(
        -1.0, 1.0, PATCH_GRID, device=tokens.device, dtype=tokens.dtype
    )
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    expected_x = (weights * grid_x.reshape(-1)).sum(dim=-1)
    expected_y = (weights * grid_y.reshape(-1)).sum(dim=-1)
    return torch.cat([expected_x, expected_y], dim=-1)


def patch_grid(tokens: torch.Tensor, size: int = POOLED_GRID) -> torch.Tensor:
    """[N, PATCH_GRID ** 2, C] -> [N, size * size, C] by average pooling the grid."""
    n, _, channels = tokens.shape
    grid = tokens.reshape(n, PATCH_GRID, PATCH_GRID, channels).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, output_size=(size, size))
    return pooled.permute(0, 2, 3, 1).reshape(n, size * size, channels)


class Dinov3:
    """Frozen encoder. Holds no trainable state and is never part of a checkpoint."""

    def __init__(self, device: torch.device) -> None:
        source = paths.cache_dir() / "dinov3" / "src"
        weights = paths.cache_dir() / "dinov3" / DINOV3_WEIGHTS_FILE
        for required in (source / "hubconf.py", weights):
            if not required.is_file():
                raise FileNotFoundError(f"{required} is missing; run install_deps.sh.")
        net = torch.hub.load(
            str(source),
            DINOV3_MODEL,
            source="local",
            pretrained=True,
            weights=str(weights),
        )
        self.net = net.to(device).eval().requires_grad_(False)
        self.normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.device = device

    def encode(self, images: torch.Tensor, poolings: tuple[str, ...]) -> dict:
        """uint8 images [N, 224, 224, 3] -> {pooling: float32 features}."""
        return self._encode(self._nchw(images), poolings)

    def _nchw(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.uint8 or tuple(images.shape[1:]) != IMAGE_SHAPE:
            raise ValueError(f"expected uint8 [N, *{IMAGE_SHAPE}], got {images.shape}")
        return images.to(self.device, torch.float32).permute(0, 3, 1, 2)

    @torch.inference_mode()
    def _encode(self, x: torch.Tensor, poolings: tuple[str, ...]) -> dict:
        x = x.div(255.0)
        tokens = self.net.forward_features(self.normalize(x))["x_norm_patchtokens"]
        out = {}
        for pooling in poolings:
            if pooling == SPATIAL:
                out[pooling] = spatial_softmax(tokens)
            elif pooling == PATCHES:
                out[pooling] = patch_grid(tokens)
            else:
                raise ValueError(f"unknown pooling {pooling}; known {sorted(POOLINGS)}")
        return out
