"""One task's episodes, ``success.zarr`` then ``failure.zarr``, as a single view.

Episode ``e`` of the view is episode ``e`` of the success store for ``e < n_success``,
else episode ``e - n_success`` of the failure store. Frames are numbered the same way.
Numeric arrays (actions, proprioception) are read into memory at open; images stay on
disk and are only read when building feature caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from needlework import paths
from needlework.constants import OUTCOMES
from needlework.data.schema import StoreSpec, spec_for, validate_store
from needlework.geometry.actions import ActionLayout


@dataclass(frozen=True)
class EpisodeStore:
    domain: str
    task: str
    spec: StoreSpec
    layout: ActionLayout
    store_paths: dict[str, Path]  # outcome -> store directory
    identity: dict[str, str]  # outcome -> sha256 of the archive it was unpacked from
    episode_ends: np.ndarray  # int64 [E], exclusive, over the concatenated view
    episode_success: np.ndarray  # bool [E]
    arrays: dict[str, np.ndarray]  # action and proprioception, float32 [T, D]

    @classmethod
    def open(cls, domain: str, task: str) -> EpisodeStore:
        """Open ``$NEEDLEWORK_ROOT/data/<domain>/<task>/{success,failure}.zarr``."""
        task_dir = paths.data_dir() / domain / task
        return cls.open_paths(
            domain=domain,
            task=task,
            store_paths={outcome: task_dir / f"{outcome}.zarr" for outcome in OUTCOMES},
        )

    @classmethod
    def open_paths(
        cls, *, domain: str, task: str, store_paths: dict[str, Path]
    ) -> EpisodeStore:
        if tuple(store_paths) != OUTCOMES:
            raise ValueError(f"store_paths must be ordered {OUTCOMES}")
        spec = spec_for(domain, task)
        ends, success, identity = [], [], {}
        # Simulator state is for resets, never for training, so it is not loaded.
        numeric: dict[str, list[np.ndarray]] = {
            key: [] for key in spec.data_widths() if key != "state"
        }
        offset = 0
        for outcome, path in store_paths.items():
            validate_store(path, domain=domain, task=task)
            marker = path.with_name(path.name + ".sha256")
            if not marker.is_file():
                raise FileNotFoundError(
                    f"{marker} is missing; unpack stores with needlework.data.download."
                )
            identity[outcome] = marker.read_text().strip()
            root = zarr.open(str(path), mode="r")
            store_ends = root["meta/episode_ends"][:]
            ends.append(store_ends + offset)
            success.append(root["meta/episode_success"][:])
            offset += int(store_ends[-1])
            for key in numeric:
                numeric[key].append(root[f"data/{key}"][:])
        episode_success = np.concatenate(success)
        n_success = int(episode_success.sum())
        if not episode_success[:n_success].all():
            raise ValueError("success episodes must precede failure episodes")
        return cls(
            domain=domain,
            task=task,
            spec=spec,
            layout=ActionLayout.from_spec(
                dict(zarr.open(str(store_paths["success"]), mode="r").attrs["action"])
            ),
            store_paths=dict(store_paths),
            identity=identity,
            episode_ends=np.concatenate(ends).astype(np.int64),
            episode_success=episode_success,
            arrays={key: np.concatenate(parts) for key, parts in numeric.items()},
        )

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def n_frames(self) -> int:
        return int(self.episode_ends[-1])

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64)

    def episode_of_frame(self, frames: np.ndarray) -> np.ndarray:
        """Episode index of each global frame index."""
        return np.searchsorted(self.episode_ends, frames, side="right")
