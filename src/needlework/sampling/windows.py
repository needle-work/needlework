"""Fixed-length windows over episodes, with edge padding and a frame stride.

A window is (episode e, start s) with s possibly negative. Its position k reads frame
``episode_start[e] + clip(s + k * stride, 0, L_e - 1)``: positions before the episode
repeat its first frame and positions after it repeat its last frame. Starts run from
``-pad_before * stride`` to ``L_e - span + pad_after * stride`` where
``span = (horizon - 1) * stride + 1``. The same rule serves stride 1 (Robomimic) and
stride 3 (UMI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Windows:
    episode: np.ndarray  # int64 [N]
    start: np.ndarray  # int64 [N], local frame index, may be negative
    horizon: int
    stride: int


def enumerate_windows(
    episode_ends: np.ndarray,
    episodes: np.ndarray,
    *,
    horizon: int,
    stride: int,
    pad_before: int,
    pad_after: int,
) -> Windows:
    """All windows of the listed episodes, episode by episode, start ascending."""
    if not 0 <= pad_before < horizon or not 0 <= pad_after < horizon:
        raise ValueError(f"padding must be in [0, horizon): {pad_before}, {pad_after}")
    lengths = np.diff(np.concatenate([[0], episode_ends]))
    span = (horizon - 1) * stride + 1
    episode_ids, starts = [], []
    for e in np.asarray(episodes, dtype=np.int64):
        first = -pad_before * stride
        last = int(lengths[e]) - span + pad_after * stride
        if last < first:
            continue
        s = np.arange(first, last + 1, dtype=np.int64)
        episode_ids.append(np.full(len(s), e, dtype=np.int64))
        starts.append(s)
    if not starts:
        raise ValueError("no episode is long enough for a single window")
    return Windows(
        episode=np.concatenate(episode_ids),
        start=np.concatenate(starts),
        horizon=horizon,
        stride=stride,
    )


def window_frames(
    windows: Windows, indices: np.ndarray, episode_ends: np.ndarray
) -> np.ndarray:
    """Global frame indices [len(indices), horizon] of the given windows."""
    episodes = windows.episode[indices]
    begins = np.concatenate([[0], episode_ends[:-1]])[episodes]
    lengths = episode_ends[episodes] - begins
    positions = np.arange(windows.horizon) * windows.stride
    local = windows.start[indices][:, None] + positions[None, :]
    return begins[:, None] + np.clip(local, 0, lengths[:, None] - 1)
