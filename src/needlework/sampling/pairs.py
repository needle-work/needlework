"""Source-goal pairs over policy windows, shared by the IDM and the verifier.

The source of a window is its current frame (position ``n_obs - 1``). A goal ``k`` steps
ahead is frame ``current + k * stride``; ``max_future`` is the largest ``k`` still
inside the episode. For a goal reached at step ``k`` of the action horizon ``H``, the
logged actions are held from row ``n_obs - 1 + k`` on ("subgoal terminal" padding): the
label depends on the goal, which is what makes the IDM use it.

IDM: one pair per window per epoch, ``k`` uniform in ``[1, min(max_future, H)]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from needlework.sampling.policy_dataset import PolicyDataset
from needlework.sampling.windows import window_frames


@dataclass(frozen=True)
class Anchors:
    windows: np.ndarray  # window index of each anchor
    current: np.ndarray  # source frame
    episode: np.ndarray
    max_future: np.ndarray  # steps (of ``stride`` frames) to the episode's last frame


def current_frames(data: PolicyDataset) -> np.ndarray:
    """The current frame (position ``n_obs - 1``) of every window."""
    frames = window_frames(data.windows, np.arange(len(data)), data.store.episode_ends)
    return frames[:, data.shape.n_obs - 1]


def anchors(data: PolicyDataset) -> Anchors:
    """Every window whose source frame has at least one future step."""
    all_windows = np.arange(len(data))
    current = current_frames(data)
    episode = data.windows.episode
    last = data.store.episode_ends[episode] - 1
    max_future = (last - current) // data.shape.stride
    keep = max_future >= 1
    return Anchors(all_windows[keep], current[keep], episode[keep], max_future[keep])


def hold_from(action: np.ndarray, offset: np.ndarray, n_obs: int) -> np.ndarray:
    """Rows from ``n_obs - 1 + offset`` on repeat row ``n_obs - 2 + offset``."""
    rows = np.arange(action.shape[1])[None, :]
    last = (n_obs - 2 + offset)[:, None]
    source = np.where(rows > last, last, rows)
    return np.take_along_axis(action, source[..., None], axis=1)


def action_horizon(data: PolicyDataset) -> int:
    """Executable rows from the current step to the end of the window."""
    return data.shape.horizon - data.shape.n_obs + 1


class IdmPairs:
    def __init__(
        self,
        data: PolicyDataset,
        goal_features: dict[str, np.ndarray],
        *,
        seed: int,
        resample: bool,
    ) -> None:
        self.data = data
        self.goal_features = goal_features
        self.anchors = anchors(data)
        self.horizon = action_horizon(data)
        self.seed = seed
        self.resample = resample
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.anchors.windows)

    def set_epoch(self, epoch: int) -> None:
        """Draw every pair's goal offset for ``epoch`` (fixed if not resampling)."""
        rng = np.random.default_rng([self.seed, epoch if self.resample else 0])
        high = np.minimum(self.anchors.max_future, self.horizon)
        self.offset = rng.integers(1, high + 1)

    def batch(self, indices: np.ndarray) -> dict:
        a, offset = self.anchors, self.offset[indices]
        obs, action = self.data.arrays(a.windows[indices], self.data.train)
        action = hold_from(action, offset, self.data.shape.n_obs)
        goal_frame = a.current[indices] + offset * self.data.shape.stride
        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs.items()},
            "goal": {
                key: torch.from_numpy(features[goal_frame])
                for key, features in self.goal_features.items()
            },
            "action": torch.from_numpy(action),
            "valid": torch.ones(action.shape[:2], dtype=torch.bool),
        }
