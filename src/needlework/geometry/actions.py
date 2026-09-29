"""Action layout and the transforms applied to actions.

Every stored action is, per arm: position (3) | rotation_6d (6) | gripper (1),
absolute in the store's frame (see needlework.data.schema). Two transforms exist:

- ``to_relative``: express each arm's pose relative to a base pose (UMI trains on
  actions relative to the current gripper pose). The gripper value passes through.
- ``to_env_command``: rotation_6d -> axis-angle, the robosuite OSC absolute command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from needlework.constants import ACTION_DIM_PER_ARM, ACTION_LAYOUT
from needlework.geometry.rotations import (
    matrix_to_axis_angle,
    matrix_to_rot6d,
    pose_matrix,
    rot6d_to_matrix,
)

_POSITION = slice(*ACTION_LAYOUT["position"])
_ROTATION = slice(*ACTION_LAYOUT["rotation_6d"])
_GRIPPER = slice(*ACTION_LAYOUT["gripper"])


@dataclass(frozen=True)
class ActionLayout:
    arms: tuple[str, ...]

    @classmethod
    def from_spec(cls, action_spec: dict) -> ActionLayout:
        """Build from a store's ``action`` attribute; rejects any other layout."""
        if dict(action_spec["layout_per_arm"]) != ACTION_LAYOUT:
            layout = action_spec["layout_per_arm"]
            raise ValueError(f"unsupported action layout {layout}")
        return cls(arms=tuple(str(arm) for arm in action_spec["arms"]))

    @property
    def dim(self) -> int:
        return ACTION_DIM_PER_ARM * len(self.arms)

    def per_arm(self, actions: np.ndarray) -> np.ndarray:
        """[..., dim] -> [..., n_arms, 10]."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[-1] != self.dim:
            raise ValueError(f"expected action dim {self.dim}, got {actions.shape}")
        return actions.reshape(*actions.shape[:-1], len(self.arms), ACTION_DIM_PER_ARM)


def _checked_base(layout: ActionLayout, base_poses: np.ndarray) -> np.ndarray:
    """Base poses [..., n_arms, 4, 4]: one per arm, never broadcast across arms."""
    base_poses = np.asarray(base_poses, dtype=np.float32)
    if base_poses.shape[-3:] != (len(layout.arms), 4, 4):
        raise ValueError(f"expected base poses [..., {len(layout.arms)}, 4, 4]")
    return base_poses


def action_poses(layout: ActionLayout, actions: np.ndarray) -> np.ndarray:
    """[..., dim] -> per-arm homogeneous poses [..., n_arms, 4, 4]."""
    arms = layout.per_arm(actions)
    return pose_matrix(arms[..., _POSITION], rot6d_to_matrix(arms[..., _ROTATION]))


def to_relative(
    layout: ActionLayout, actions: np.ndarray, base_poses: np.ndarray
) -> np.ndarray:
    """Express actions [..., T, dim] relative to base poses [..., n_arms, 4, 4].

    Leading dimensions (e.g. a batch of windows) must match between the two.
    """
    arms = layout.per_arm(actions)
    base_poses = _checked_base(layout, base_poses)
    inverse = np.expand_dims(np.linalg.inv(base_poses).astype(np.float32), axis=-4)
    relative = inverse @ action_poses(layout, actions)
    out = np.concatenate(
        [
            relative[..., :3, 3],
            matrix_to_rot6d(relative[..., :3, :3]),
            arms[..., _GRIPPER],
        ],
        axis=-1,
    )
    return out.reshape(*arms.shape[:-2], layout.dim).astype(np.float32)


def to_env_command(layout: ActionLayout, actions: np.ndarray) -> np.ndarray:
    """[..., dim] -> [..., 7 * n_arms]: position | axis-angle | gripper per arm."""
    arms = layout.per_arm(actions)
    rotvec = matrix_to_axis_angle(rot6d_to_matrix(arms[..., _ROTATION]))
    out = np.concatenate([arms[..., _POSITION], rotvec, arms[..., _GRIPPER]], axis=-1)
    return out.reshape(*arms.shape[:-2], 7 * len(layout.arms)).astype(np.float32)


def from_relative(
    layout: ActionLayout, actions: np.ndarray, base_poses: np.ndarray
) -> np.ndarray:
    """Source-relative actions [..., T, A] to absolute poses in the source episode."""
    arms = layout.per_arm(actions)
    base_poses = _checked_base(layout, base_poses)
    poses = np.expand_dims(base_poses, axis=-4) @ action_poses(layout, actions)
    out = np.concatenate(
        [poses[..., :3, 3], matrix_to_rot6d(poses[..., :3, :3]), arms[..., _GRIPPER]],
        axis=-1,
    )
    return out.reshape(*actions.shape).astype(np.float32)
