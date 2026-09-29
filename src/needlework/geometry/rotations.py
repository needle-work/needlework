"""Rotation conversions. float32 in, float32 out, like the data they operate on.

rotation_6d is the first two rows of a rotation matrix, row-major. A 6-vector whose rows
are not orthonormal (for example a policy's raw output) encodes the rotation given by
Gram-Schmidt on those rows, which is also what the controller executes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_DEGENERATE_NORM = 1e-8


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] -> [..., 3, 3] by Gram-Schmidt on the two rows."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must have 6 values, got shape {rot6d.shape}")
    if not np.all(np.isfinite(rot6d)):
        raise ValueError("rotation_6d contains non-finite values")
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    norm_a1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if not np.all(norm_a1 > _DEGENERATE_NORM):
        raise ValueError("rotation_6d has a zero first row")
    b1 = a1 / norm_a1
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm_b2 = np.linalg.norm(b2, axis=-1, keepdims=True)
    if not np.all(norm_b2 > _DEGENERATE_NORM):
        raise ValueError("rotation_6d rows are parallel")
    b2 = b2 / norm_b2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 6]: the first two rows."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected [..., 3, 3], got {matrix.shape}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """[..., 3] -> [..., 3, 3]."""
    rotvec = np.asarray(rotvec, dtype=np.float32)
    flat = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return flat.reshape(*rotvec.shape[:-1], 3, 3).astype(np.float32)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 3]."""
    matrix = np.asarray(matrix)
    flat = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec()
    return flat.reshape(*matrix.shape[:-2], 3).astype(np.float32)


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transforms [..., 4, 4] from positions [..., 3] and rotations [..., 3, 3]."""
    position = np.asarray(position, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    out = np.zeros((*position.shape[:-1], 4, 4), dtype=np.float32)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = position
    out[..., 3, 3] = 1.0
    return out
