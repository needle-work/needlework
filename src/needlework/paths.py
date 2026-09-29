"""Filesystem layout: the one place that reads ``NEEDLEWORK_ROOT``.

The only other environment variables the package touches are process settings of
other libraries, each set where it matters: ``CUBLAS_WORKSPACE_CONFIG`` (``train.py``,
``stitch.py``, ``training/determinism.py``) and ``MUJOCO_EGL_DEVICE_ID``
(``sim/factory.py``).

Every path the package touches is derived from one variable, ``NEEDLEWORK_ROOT``:

    $NEEDLEWORK_ROOT/data     downloaded inputs, read-only during runs
    $NEEDLEWORK_ROOT/outputs  training runs and stitched stores
    $NEEDLEWORK_ROOT/cache    re-creatable assets (DINOv3 source and weights)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "NEEDLEWORK_ROOT"


def root() -> Path:
    """Return ``$NEEDLEWORK_ROOT``; raise if it is unset or relative."""
    if ROOT_ENV not in os.environ:
        raise KeyError(f"{ROOT_ENV} is not set. Export it and `source set_env.sh`.")
    path = Path(os.environ[ROOT_ENV])
    if not path.is_absolute():
        raise ValueError(f"{ROOT_ENV} must be an absolute path, got {path}.")
    return path


def data_dir() -> Path:
    return root() / "data"


def output_dir() -> Path:
    return root() / "outputs"


def cache_dir() -> Path:
    return root() / "cache"
