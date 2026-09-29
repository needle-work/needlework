"""Build one robomimic simulator for evaluation.

- Env arguments come from the task's HDF5, with
  the OSC arm controllers switched to absolute world-frame pose targets.
- ``construction_seed`` seeds draws made while the env is built (Transport's hammer
  size); the rollout gives each worker slot its own.
- ``hard_reset`` is off: the model is built once per worker and reused.
- Gripper debug sites are hidden, as in the training images. The first sim is destroyed
  explicitly before ``VisualizationWrapper`` builds its own: left to the garbage
  collector, its EGL context is freed while the live one is current, which blinds every
  later render.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import robomimic.utils.env_utils as env_utils
import robomimic.utils.file_utils as file_utils
import robomimic.utils.obs_utils as obs_utils
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper

from needlework.constants import IMAGE_SIZE


def camera_name(image_key: str) -> str:
    """Store key ``agentview_image`` -> robosuite camera ``agentview``."""
    if not image_key.endswith("_image"):
        raise ValueError(f"not a robomimic image key: {image_key}")
    return image_key.removesuffix("_image")


def make_env(
    *,
    hdf5: Path,
    cameras: tuple[str, ...],
    proprio: tuple[str, ...],
    construction_seed: int,
    egl_index: int,
) -> Any:
    meta = file_utils.get_env_metadata_from_dataset(dataset_path=str(hdf5))
    kwargs = meta["env_kwargs"]
    for part in kwargs["controller_configs"]["body_parts"].values():
        if part["type"] == "OSC_POSE":
            part["input_type"] = "absolute"
            part["input_ref_frame"] = "world"
    kwargs.update(
        render_gpu_device_id=egl_index,
        hard_reset=False,
        seed=construction_seed,
        camera_names=[camera_name(key) for key in cameras],
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )
    obs_utils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": list(proprio), "rgb": list(cameras)}}
    )
    # Read when the render context is created. robosuite asserts at import time that
    # this value is one of CUDA_VISIBLE_DEVICES (it conflates the two), so it is set
    # here, after robosuite is imported, and never inherited by a worker.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_index)
    env = env_utils.create_env_from_metadata(
        env_meta=meta,
        env_name=meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
    )
    env.env._destroy_sim()  # the bare robosuite env, before any wrapper
    env.env = VisualizationWrapper(env.env, indicator_configs=None)
    env.env.set_visualization_setting(setting="grippers", visible=False)
    env.env.env.visualize(vis_settings=env.env._vis_settings)
    return env
