"""Observations for a batch of windows: frames [B, n_obs] -> {key: [B, n_obs, D]}.

Camera keys carry DINOv3 features. Robomimic proprioception is read as stored. UMI
proprioception is derived from the logged gripper poses:

    gripper_<s>_eef_pos, _eef_rot_6d          pose relative to the window's last frame
    gripper_<s>_gripper_width                 as logged
    gripper_<s>_eef_rot_6d_wrt_start          rotation relative to the episode start
    gripper_<s>_eef_{pos,rot_6d}_wrt_gripper_<o>
                                              pose relative to the other gripper's
                                              pose at the window's last frame

During training the episode-start pose is perturbed by Gaussian noise (``start_noise``).

Camera features are any array indexable by frame: in memory, or ``data.features.Rows``
over memory-mapped caches.
"""

from __future__ import annotations

import numpy as np

from needlework.data.features import Rows
from needlework.data.store import EpisodeStore
from needlework.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rot6d,
    pose_matrix,
)


def _pose(position: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    return pose_matrix(position, axis_angle_to_matrix(rotvec))


def _relative(poses: np.ndarray, base: np.ndarray) -> np.ndarray:
    """poses [B, n, 4, 4] in base [B, 4, 4]: inv(base) @ poses, in float32."""
    return np.linalg.inv(base).astype(np.float32)[:, None] @ poses


def _position(poses: np.ndarray) -> np.ndarray:
    return poses[..., :3, 3]


def _rot6d(poses: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(poses[..., :3, :3])


class RobomimicObservations:
    def __init__(
        self,
        store: EpisodeStore,
        features: dict[str, np.ndarray | Rows],
        keys: tuple[str, ...],
    ) -> None:
        self.sources = {}
        for key in keys:
            if key in features:
                self.sources[key] = features[key]
            elif key in store.arrays:
                self.sources[key] = store.arrays[key]
            else:
                raise KeyError(f"unknown observation key {key}")

    def __call__(self, frames: np.ndarray, episodes: np.ndarray, train: bool) -> dict:
        return {key: source[frames] for key, source in self.sources.items()}


class UmiObservations:
    SIDES = ("left", "right")

    def __init__(
        self,
        store: EpisodeStore,
        features: dict[str, np.ndarray | Rows],
        keys: tuple[str, ...],
        start_noise: float,
    ) -> None:
        self.arrays = store.arrays
        self.starts = store.episode_starts
        self.features = {key: features[key] for key in keys if key in features}
        self.proprio_keys = tuple(key for key in keys if key not in features)
        self.start_noise = start_noise
        probe = self._derive(np.zeros((1, 1), np.int64), np.zeros(1, np.int64), False)
        unknown = set(self.proprio_keys) - set(probe)
        if unknown:
            raise KeyError(f"unknown UMI observation keys {sorted(unknown)}")

    def _noise(self, shape: tuple[int, ...]) -> np.ndarray:
        """Start-pose noise from numpy's global generator, whose state every
        checkpoint saves and every resume restores."""
        return np.random.normal(scale=self.start_noise, size=shape).astype(np.float32)

    def _derive(self, frames: np.ndarray, episodes: np.ndarray, train: bool) -> dict:
        start_frames = self.starts[episodes]
        poses, start_poses = {}, {}
        for side in self.SIDES:
            position = self.arrays[f"gripper_{side}_eef_pos"]
            rotvec = self.arrays[f"gripper_{side}_eef_rot_axis_angle"]
            poses[side] = _pose(position[frames], rotvec[frames])
            start_position = position[start_frames]
            start_rotvec = rotvec[start_frames]
            if train and self.start_noise > 0:
                start_position = start_position + self._noise(start_position.shape)
                start_rotvec = start_rotvec + self._noise(start_rotvec.shape)
            start_poses[side] = _pose(start_position, start_rotvec)
        out = {}
        for side in self.SIDES:
            prefix = f"gripper_{side}"
            latest = _relative(poses[side], poses[side][:, -1])
            out[f"{prefix}_eef_pos"] = _position(latest)
            out[f"{prefix}_eef_rot_6d"] = _rot6d(latest)
            width = self.arrays[f"{prefix}_gripper_width"]
            out[f"{prefix}_gripper_width"] = width[frames]
            out[f"{prefix}_eef_rot_6d_wrt_start"] = _rot6d(
                _relative(poses[side], start_poses[side])
            )
            for other in self.SIDES:
                if other != side:
                    rel = _relative(poses[side], poses[other][:, -1])
                    out[f"{prefix}_eef_pos_wrt_gripper_{other}"] = _position(rel)
                    out[f"{prefix}_eef_rot_6d_wrt_gripper_{other}"] = _rot6d(rel)
        return out

    def __call__(self, frames: np.ndarray, episodes: np.ndarray, train: bool) -> dict:
        derived = self._derive(frames, episodes, train)
        out = {key: derived[key].astype(np.float32) for key in self.proprio_keys}
        out.update({key: source[frames] for key, source in self.features.items()})
        return out
