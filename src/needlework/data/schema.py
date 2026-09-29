"""On-disk dataset contract (zarr v2).

A store holds episodes of one task. Released stores are split by outcome
(``success.zarr``, ``failure.zarr``) but carry a per-episode success flag, so stores
concatenate along episodes without losing outcomes.

    <store>.zarr/
    ├── .zattrs                 {"domain", "task", "action": ACTION_SPEC}
    ├── meta/
    │   ├── episode_ends        int64  [E]   exclusive end frame of each episode
    │   └── episode_success     bool   [E]
    └── data/
        ├── action              float32 [T, 10 * arms]
        ├── <camera>            uint8   [T, 224, 224, 3]
        ├── <proprio>           float32 [T, D]
        └── state               float32 [T, S]   robomimic only; simulator resets only

``domain`` fixes the arrays and the action convention (robomimic: simulated Franka arms;
umi: handheld grippers). ``task`` names the dataset within a domain; for UMI it changes
nothing about the arrays.

Action, per arm, in ``ACTION_SPEC["arms"]`` order: position (3) | rotation (6) |
gripper (1). Rotations are rotation_6d: the first two rows of the rotation matrix,
row-major. The rotation they encode is their Gram-Schmidt orthonormalization, which is
also what the controller executes: demonstration actions are orthonormal, but actions
logged from policy rollouts (Robomimic failures) are the policy's raw outputs and only
approximately so. Poses are absolute in the domain's frame; any relative representation
is computed at load time. ``action[t]`` is the target commanded at frame ``t``
(robomimic), or the gripper pose logged at frame ``t`` (UMI).

``state`` is the MuJoCo state ``[time, qpos, qvel]``: robot and object joint positions
and velocities. It is privileged and exists only to reset the simulator.

Images: one frame per chunk, lossless JPEG-XL. Numeric arrays: Blosc-zstd.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from imagecodecs.numcodecs import register_codecs

from needlework.constants import ACTION_DIM_PER_ARM, ACTION_LAYOUT, IMAGE_SHAPE

ROTATION_6D = (
    "first two rows of the rotation matrix, row-major; "
    "the rotation is their Gram-Schmidt orthonormalization"
)
_DEGENERATE_TOL = 1e-6

# Images are JPEG-XL chunks; the codec must be registered before they can be read.
register_codecs()


@dataclass(frozen=True)
class StoreSpec:
    """Arrays and conventions of one domain/task: nothing more, nothing less."""

    arms: tuple[str, ...]
    frame: str
    gripper: str
    cameras: tuple[str, ...]
    proprio: dict[str, int]  # key -> width
    state_dim: int | None  # None: no simulator state

    @property
    def action_dim(self) -> int:
        return ACTION_DIM_PER_ARM * len(self.arms)

    def action_spec(self) -> dict:
        return {
            "arms": list(self.arms),
            "layout_per_arm": ACTION_LAYOUT,
            "rotation_6d": ROTATION_6D,
            "frame": self.frame,
            "gripper": self.gripper,
        }

    def data_widths(self) -> dict[str, int]:
        widths = {"action": self.action_dim, **self.proprio}
        if self.state_dim is not None:
            widths["state"] = self.state_dim
        return widths


def _robomimic(
    cameras: tuple[str, ...], arms: tuple[str, ...], state_dim: int
) -> StoreSpec:
    proprio: dict[str, int] = {}
    for arm in arms:
        proprio |= {f"{arm}_eef_pos": 3, f"{arm}_eef_quat": 4, f"{arm}_gripper_qpos": 2}
    return StoreSpec(
        arms=arms,
        frame="simulator world frame, meters; OSC absolute pose targets",
        gripper="command: -1 open, +1 close",
        cameras=cameras,
        proprio=proprio,
        state_dim=state_dim,
    )


_UMI = StoreSpec(
    arms=("left", "right"),
    frame="capture tracking frame, meters; logged gripper poses",
    gripper="jaw width, meters",
    cameras=("camera_left_main_rgb", "camera_right_main_rgb"),
    proprio={
        f"gripper_{side}_{key}": width
        for side in ("left", "right")
        for key, width in (
            ("eef_pos", 3),
            ("eef_rot_axis_angle", 3),
            ("gripper_width", 1),
        )
    },
    state_dim=None,
)

_ROBOMIMIC_SINGLE_ARM = ("agentview_image", "robot0_eye_in_hand_image")
SPECS: dict[tuple[str, str], StoreSpec] = {
    ("robomimic", "can"): _robomimic(_ROBOMIMIC_SINGLE_ARM, ("robot0",), 71),
    ("robomimic", "square"): _robomimic(_ROBOMIMIC_SINGLE_ARM, ("robot0",), 45),
    ("robomimic", "transport"): _robomimic(
        (
            "shouldercamera0_image",
            "shouldercamera1_image",
            "robot0_eye_in_hand_image",
            "robot1_eye_in_hand_image",
        ),
        ("robot0", "robot1"),
        115,
    ),
}


def spec_for(domain: str, task: str) -> StoreSpec:
    """UMI shares one spec across tasks; robomimic tasks differ in cameras and arms."""
    return _UMI if domain == "umi" else SPECS[(domain, task)]


def store_attrs(domain: str, task: str) -> dict:
    return {
        "domain": domain,
        "task": task,
        "action": spec_for(domain, task).action_spec(),
    }


def _check_rotation_6d(path: Path, action: np.ndarray, num_arms: int) -> None:
    """Both rows must be nonzero and not parallel, so Gram-Schmidt is defined."""
    rotation = slice(*ACTION_LAYOUT["rotation_6d"])
    rot = action.reshape(len(action), num_arms, ACTION_DIM_PER_ARM)[..., rotation]
    rows = rot.reshape(len(action), num_arms, 2, 3).astype(np.float64)
    norms = np.linalg.norm(rows, axis=-1)
    if not np.isfinite(rows).all() or norms.min() < _DEGENERATE_TOL:
        raise ValueError(f"{path}: action rotation_6d has a zero or non-finite row")
    cosine = np.abs((rows[..., 0, :] * rows[..., 1, :]).sum(-1)) / norms.prod(-1)
    if cosine.max() > 1 - _DEGENERATE_TOL:
        raise ValueError(f"{path}: action rotation_6d rows are parallel")


def validate_store(path: Path, *, domain: str, task: str) -> tuple[int, int]:
    """Check ``path`` against the contract; return (num_episodes, num_success).

    Raises on any missing, extra, or malformed array or attribute.
    """
    spec = spec_for(domain, task)
    root = zarr.open(str(path), mode="r")
    if dict(root.attrs) != store_attrs(domain, task):
        raise ValueError(f"{path}: attrs {dict(root.attrs)} do not match the spec")
    if set(root.group_keys()) != {"meta", "data"} or set(root.array_keys()):
        raise ValueError(f"{path}: top level must be exactly meta/ and data/")
    if set(root["meta"].keys()) != {"episode_ends", "episode_success"}:
        raise ValueError(f"{path}: meta/ must hold episode_ends and episode_success")
    expected = set(spec.cameras) | set(spec.data_widths())
    if set(root["data"].keys()) != expected:
        raise ValueError(
            f"{path}: data/ keys {sorted(root['data'].keys())} != {sorted(expected)}"
        )
    ends = root["meta/episode_ends"][:]
    success = root["meta/episode_success"][:]
    if ends.dtype != np.int64 or len(ends) == 0 or ends[0] <= 0:
        raise ValueError(f"{path}: episode_ends must be positive int64")
    if np.any(np.diff(ends) <= 0):
        raise ValueError(f"{path}: episode_ends must be strictly increasing")
    if success.dtype != np.bool_ or success.shape != ends.shape:
        raise ValueError(f"{path}: episode_success must be bool [{len(ends)}]")
    num_frames = int(ends[-1])
    for key in spec.cameras:
        array = root[f"data/{key}"]
        if array.shape != (num_frames, *IMAGE_SHAPE) or array.dtype != np.uint8:
            raise ValueError(f"{path}: {key} {array.shape} {array.dtype}")
        if array.chunks != (1, *IMAGE_SHAPE):
            raise ValueError(f"{path}: {key} must be chunked one frame per chunk")
    for key, width in spec.data_widths().items():
        array = root[f"data/{key}"]
        if array.shape != (num_frames, width) or array.dtype != np.float32:
            raise ValueError(f"{path}: {key} {array.shape} {array.dtype}")
    _check_rotation_6d(path, root["data/action"][:], len(spec.arms))
    return len(ends), int(success.sum())
