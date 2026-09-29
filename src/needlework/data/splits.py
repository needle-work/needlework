"""Train/validation split over episodes, fixed by ``seed.dataset``.

Validation episodes are drawn either from successes only (the policy, so every policy
variant holds out the same demonstrations) or from all episodes (IDM, verifier). Every
other episode is a training episode; which of its windows are used is the sampler's
business.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAL_FROM = ("success", "all")


@dataclass(frozen=True)
class Split:
    train: np.ndarray  # bool [E]
    val: np.ndarray  # bool [E]


def _val_draw(n_episodes: int, val_ratio: float, seed: int) -> np.ndarray:
    """At least one validation and one training episode."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    mask = np.zeros(n_episodes, dtype=bool)
    rng = np.random.default_rng(seed)
    mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return mask


def split_episodes(
    episode_success: np.ndarray, *, val_ratio: float, seed: int, val_from: str
) -> Split:
    if val_from not in VAL_FROM:
        raise ValueError(f"val_from must be one of {VAL_FROM}, got {val_from}")
    episode_success = np.asarray(episode_success, dtype=bool)
    eligible = (
        np.flatnonzero(episode_success)
        if val_from == "success"
        else np.arange(len(episode_success))
    )
    if len(eligible) < 2:
        raise ValueError(f"need at least two eligible episodes, got {len(eligible)}")
    val = np.zeros(len(episode_success), dtype=bool)
    val[eligible[_val_draw(len(eligible), val_ratio, seed)]] = True
    return Split(train=~val, val=val)
