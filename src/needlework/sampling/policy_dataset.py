"""Policy training examples from logged episodes (normal sampling).

Each example is a window of ``horizon`` frames: observations at the first ``n_obs``
positions, actions at every position, and a validity mask (all true here; bridge
windows use it). ``pad_before = n_obs - 1`` and ``pad_after = n_execute - 1`` (edge
repeat). With ``relative_actions`` (UMI) each window's actions
are expressed relative to the pose at its current step, position ``n_obs - 1``.

Batches are assembled in the training process from in-memory arrays, far cheaper than
one U-Net step, so there are no loader workers and the draw order is exactly the seeded
permutation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from needlework.data.store import EpisodeStore
from needlework.geometry.actions import action_poses, to_relative
from needlework.models.normalizer import fit_params
from needlework.sampling.windows import enumerate_windows, window_frames

Observations = Callable[[np.ndarray, np.ndarray, bool], dict[str, np.ndarray]]
FIT_CHUNK = 8192


@dataclass(frozen=True)
class WindowShape:
    horizon: int
    n_obs: int
    n_execute: int
    stride: int


class PolicyDataset:
    def __init__(
        self,
        *,
        store: EpisodeStore,
        observations: Observations,
        episodes: np.ndarray,
        shape: WindowShape,
        relative_actions: bool,
        train: bool,
    ) -> None:
        self.store = store
        self.observations = observations
        self.shape = shape
        self.relative_actions = relative_actions
        self.train = train
        self.windows = enumerate_windows(
            store.episode_ends,
            episodes,
            horizon=shape.horizon,
            stride=shape.stride,
            pad_before=shape.n_obs - 1,
            pad_after=shape.n_execute - 1,
        )

    def __len__(self) -> int:
        return len(self.windows.episode)

    def arrays(self, indices: np.ndarray, train: bool) -> tuple[dict, np.ndarray]:
        """Observations {key: [B, n_obs, D]} and actions [B, horizon, A] of windows."""
        frames = window_frames(self.windows, indices, self.store.episode_ends)
        episodes = self.windows.episode[indices]
        obs = self.observations(frames[:, : self.shape.n_obs], episodes, train)
        action = self.store.arrays["action"][frames]
        if self.relative_actions:
            layout = self.store.layout
            base = action_poses(layout, action[:, self.shape.n_obs - 1])
            action = to_relative(layout, action, base)
        return obs, action

    def set_epoch(self, epoch: int) -> None:
        """Windows are the same every epoch; only their order changes."""

    def batch(self, indices: np.ndarray) -> dict:
        """Windows as CPU tensors: obs {key: [B, n_obs, D]}, action, valid."""
        obs, action = self.arrays(indices, self.train)
        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs.items()},
            "action": torch.from_numpy(action),
            "valid": torch.ones(action.shape[:2], dtype=torch.bool),
        }


def fit_normalizer(
    dataset: PolicyDataset,
    *,
    proprio_identity: tuple[str, ...],
    action_identity_dims: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Min/max over every window of ``dataset`` (as the model sees them, no noise):
    actions over all rows, proprioception over all observation rows."""
    low: dict[str, np.ndarray] = {}
    high: dict[str, np.ndarray] = {}
    cameras = set(dataset.store.spec.cameras)  # features are not normalized
    for start in tqdm(
        range(0, len(dataset), FIT_CHUNK), desc="normalizer", leave=False
    ):
        indices = np.arange(start, min(start + FIT_CHUNK, len(dataset)))
        obs, action = dataset.arrays(indices, train=False)
        rows = {"action": action, **{k: v for k, v in obs.items() if k not in cameras}}
        for key, value in rows.items():
            flat = value.reshape(-1, value.shape[-1])
            low[key] = np.minimum(low[key], flat.min(0)) if key in low else flat.min(0)
            high[key] = (
                np.maximum(high[key], flat.max(0)) if key in high else flat.max(0)
            )
    params = {}
    for key in low:
        if key == "action":
            identity = np.asarray(action_identity_dims, dtype=bool)
        elif key in proprio_identity:
            identity = np.ones_like(low[key], dtype=bool)
        else:
            identity = np.zeros_like(low[key], dtype=bool)
        params[key] = fit_params(np.stack([low[key], high[key]]), identity)
    return params
