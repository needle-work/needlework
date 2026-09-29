"""Sampling weight of a training window by the role of its current frame on the
stitched routes.

A success bridge re-routes its source episode: after a stage-1 source the route skips
to the target in the same episode, after a stage-2 source it leaves for another
episode's target and continues along that episode's suffix. Each frame of a success
episode is:

    TARGET_SUFFIX  from a stage-2 target to its episode's end (the route continues)
    SKIPPED        after a stage-1 source up to its target, or after a stage-2 source
                   up to its episode's end, and not a TARGET_SUFFIX frame; weighs
                   ``skipped``
    CANONICAL      any other frame

A window weighs its current frame's role weight. At a departure (the window whose
current frame is a source) the crossing window weighs the task's ``bridge_weight`` and
its logged twin stays in at ``twin``; a failure bridge's single window weighs
``failure_departure``. A crossing window before its departure (an approach) weighs its
current frame's role weight times ``approach``. ``skipped``, ``twin``,
``failure_departure`` and ``approach`` come from the task (``sampler.role_weights``); a
weight of 0 leaves the role's windows out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from needlework.constants import STAGE_CROSS, STAGE_RECOVERY, STAGE_WITHIN
from needlework.data.bridges import Bridges
from needlework.data.store import EpisodeStore

CANONICAL = 1.0
TARGET_SUFFIX = 1.0


@dataclass(frozen=True)
class RoleWeights:
    skipped: float
    twin: float
    failure_departure: float
    approach: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not (np.isfinite(value) and value >= 0):
                raise ValueError(f"{name} weight must be finite and >= 0, got {value}")


def frame_weights(
    store: EpisodeStore, bridges: Bridges, weights: RoleWeights
) -> np.ndarray:
    """[n_frames] the role weight of the window whose current frame is each frame."""
    is_skipped, suffix = (np.zeros(store.n_frames, dtype=bool) for _ in range(2))
    source_episode = store.episode_of_frame(bridges.source_frame)
    target_episode = store.episode_of_frame(bridges.target_frame)
    for source, target, stage, s_ep, t_ep in zip(
        bridges.source_frame,
        bridges.target_frame,
        bridges.stage,
        source_episode,
        target_episode,
        strict=True,
    ):
        s_end, t_end = store.episode_ends[s_ep], store.episode_ends[t_ep]
        if stage == STAGE_WITHIN:
            if s_ep != t_ep or not source < target:
                raise ValueError(
                    f"stage-1 bridge {source} -> {target} leaves its episode"
                )
            is_skipped[source + 1 : target] = True
        elif stage == STAGE_CROSS:
            if s_ep == t_ep:
                raise ValueError(
                    f"stage-2 bridge {source} -> {target} stays in its episode"
                )
            is_skipped[source + 1 : s_end] = True
            suffix[target:t_end] = True
        elif stage != STAGE_RECOVERY:
            raise ValueError(f"bridge {source} -> {target} has unknown stage {stage}")
    return np.where(
        suffix, TARGET_SUFFIX, np.where(is_skipped, weights.skipped, CANONICAL)
    )


def crossing_weights(
    frame: np.ndarray,
    departure: np.ndarray,
    success: np.ndarray,
    bridge_weight: float,
    weights: RoleWeights,
) -> np.ndarray:
    """Crossing windows' weights from their current frames' role weights ``frame``,
    whether each is its bridge's departure, and whether its episode succeeded."""
    return np.where(
        departure,
        np.where(success, bridge_weight, weights.failure_departure),
        weights.approach * frame,
    )
