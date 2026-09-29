"""Compact bridge stores: immutable logged-frame references and packed action chunks.

Actions are absolute for Robomimic, source-relative for UMI. Lengths are policy
steps, frame indices are raw rows. No images or base trajectories are copied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from needlework.constants import STAGE_CROSS, STAGE_RECOVERY, STAGE_WITHIN, STAGES
from needlework.data.store import EpisodeStore

FORMAT = 1  # bridge store layout version

ARRAYS = (
    "source_frame",
    "target_frame",
    "stage",
    "verifier_margin",
    "action_start",
    "action_len",
    "actions",
)


@dataclass(frozen=True)
class Bridges:
    source_frame: np.ndarray
    target_frame: np.ndarray
    stage: np.ndarray
    verifier_margin: np.ndarray
    action_start: np.ndarray
    action_len: np.ndarray
    actions: np.ndarray

    def __len__(self) -> int:
        return len(self.stage)

    def subset(self, keep: np.ndarray) -> Bridges:
        return Bridges(
            **{k: getattr(self, k)[keep] for k in ARRAYS[:-1]}, actions=self.actions
        )

    def rows(self, index: int) -> np.ndarray:
        start = self.action_start[index]
        return self.actions[start : start + self.action_len[index]]

    @classmethod
    def empty(cls, action_dim: int) -> Bridges:
        return cls(
            **{
                k: np.empty(0, np.float32 if k == "verifier_margin" else np.int64)
                for k in ARRAYS[:-1]
            },
            actions=np.empty((0, action_dim), np.float32),
        )

    def digest(self) -> str:
        value = hashlib.sha256()
        for name in ARRAYS:
            array = getattr(self, name)
            value.update(str((name, array.shape, array.dtype.str)).encode())
            value.update(np.ascontiguousarray(array).tobytes())
        return value.hexdigest()


def validate(b: Bridges, store: EpisodeStore, *, horizon: int, stride: int) -> None:
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    for name in ARRAYS[:-1]:
        a = getattr(b, name)
        dtype = np.float32 if name == "verifier_margin" else np.int64
        if a.shape != (len(b),) or a.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype} [{len(b)}], got {a.shape}/{a.dtype}"
            )
    if (
        b.actions.ndim != 2
        or b.actions.shape[1] != store.spec.action_dim
        or b.actions.dtype != np.float32
    ):
        raise ValueError("actions have incorrect shape or dtype")
    if not np.isfinite(b.actions).all() or not np.isfinite(b.verifier_margin).all():
        raise ValueError("bridge actions and margins must be finite")
    for frame in (b.source_frame, b.target_frame):
        if np.any((frame < 0) | (frame >= store.n_frames)):
            raise ValueError("bridge frame outside the store")
    if len(np.unique(b.source_frame)) != len(b):
        raise ValueError("bridge sources must be unique")
    if not np.isin(b.stage, STAGES).all():
        raise ValueError(f"bridge stage must be one of {STAGES}")
    if np.any((b.action_len < 1) | (b.action_len > horizon)):
        raise ValueError("bridge action length outside the horizon")
    if np.any(b.action_start < 0) or np.any(
        b.action_start + b.action_len > len(b.actions)
    ):
        raise ValueError("bridge action slice outside packed actions")
    src, dst = (
        store.episode_of_frame(b.source_frame),
        store.episode_of_frame(b.target_frame),
    )
    success = store.episode_success
    within = b.stage == STAGE_WITHIN
    cross = b.stage == STAGE_CROSS
    recovery = b.stage == STAGE_RECOVERY
    if (
        np.any(~success[dst])
        or np.any(~success[src[within | cross]])
        or np.any(success[src[recovery]])
        or np.any(src[within] != dst[within])
        or np.any(src[cross] == dst[cross])
    ):
        raise ValueError("bridge stage disagrees with episode endpoints/outcomes")
    remaining = store.episode_ends[src] - b.source_frame
    target_remaining = store.episode_ends[dst] - b.target_frame
    if np.any((remaining - target_remaining - stride * b.action_len)[~recovery] <= 0):
        raise ValueError("success bridge must save steps along the complete route")


def _identity(store: EpisodeStore, horizon: int, stride: int) -> dict:
    return {
        "format": FORMAT,
        "domain": store.domain,
        "task": store.task,
        "store": store.identity,
        "horizon": horizon,
        "stride": stride,
        "action_frame": "source_relative" if store.domain == "umi" else "absolute",
    }


def save(
    path: Path,
    b: Bridges,
    store: EpisodeStore,
    *,
    horizon: int,
    stride: int,
    recipe: dict,
) -> None:
    validate(b, store, horizon=horizon, stride=stride)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    # A partial write is never reused or overwritten silently.
    tmp.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(tmp), mode="w")
    group.attrs.update(
        {
            **_identity(store, horizon, stride),
            "recipe": recipe,
            "content_sha256": b.digest(),
        }
    )
    for name in ARRAYS:
        array = getattr(b, name)
        group.array(name, array, chunks=tuple(max(1, n) for n in array.shape))
    tmp.rename(path)


def load(path: Path, store: EpisodeStore, *, horizon: int, stride: int) -> Bridges:
    group = zarr.open_group(str(path), mode="r")
    for key, expected in _identity(store, horizon, stride).items():
        if group.attrs[key] != expected:
            raise ValueError(
                f"{path}: {key} differs from expected {json.dumps(expected)}"
            )
    b = Bridges(**{name: group[name][:] for name in ARRAYS})
    validate(b, store, horizon=horizon, stride=stride)
    if b.digest() != group.attrs["content_sha256"]:
        raise ValueError(f"{path}: bridge content changed")
    return b
