"""Fixed values of the data contract, the frozen encoder and the evaluation protocol.

Each value is defined here and nowhere else. Method settings (sampling probabilities,
proximity shells and percentiles, proposal and vote counts, quantiles, snapshot counts)
are Hydra config fields (``needlework/configs``), so they are recorded in every run's
config. Everything here except the process-plumbing block is captured by ``snapshot()``
in every checkpoint and stitching run; a resume refuses if any of it changed. File names
of the run directory stay with the module that owns that layout, and a numerical
tolerance used by one function stays next to it.
"""

import json

# Data contract (``data/schema.py``).
OUTCOMES = ("success", "failure")  # store order within a task: successes first
IMAGE_SIZE = 224  # pixels; every camera frame is square RGB
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
ACTION_DIM_PER_ARM = 10
ACTION_LAYOUT = {"position": [0, 3], "rotation_6d": [3, 9], "gripper": [9, 10]}

# Frozen image encoder: DINOv3 ViT-B/16, pinned source and weights. install_deps.sh
# reads these four values from this file.
DINOV3_MODEL = "dinov3_vitb16"
DINOV3_SOURCE_COMMIT = "31703e4cbf1ccb7c4a72daa1350405f86754b6d1"
DINOV3_WEIGHTS_FILE = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINOV3_WEIGHTS_SHA256 = (
    "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
)
DINOV3_FEATURE_DIM = 768
DINOV3_PATCH_SIZE = 16  # pixels per patch side
PATCH_GRID = IMAGE_SIZE // DINOV3_PATCH_SIZE  # 14 patch tokens per image side
POOLED_GRID = 7  # verifier scene tokens: the patch grid average-pooled to 7 x 7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPATIAL = f"spatial_softmax_{PATCH_GRID}"  # policy, IDM, proximity, stitching
PATCHES = f"patch_grid_{POOLED_GRID}"  # verifier source and goal tokens

# Stitching (``stitching/``).
STAGE_WITHIN = 1  # success source -> later frame of the same success episode
STAGE_CROSS = 2  # success source -> frame of another success episode
STAGE_RECOVERY = 3  # failure source -> frame of a success episode
STAGES = (STAGE_WITHIN, STAGE_CROSS, STAGE_RECOVERY)

# Robomimic evaluation: episode i of eval seed s resets with seed s * stride + i.
EPISODE_SEED_STRIDE = 1000

# Random-stream identifiers. Arbitrary, but fixed: every sampled table and example
# depends on them.
HOLD_STREAM = 8191
SHELL_ORDER_STREAM = 991
CROSS_POOL_STREAM = 7331
UNIFORM_SHELL_STREAM = 4409
DRAW_BLOCK = 65536

# Process plumbing: speed, memory and failure detection, not the algorithm. Not part of
# ``snapshot()``, so changing one never blocks a resume.
SIMILARITY_BATCH = 512  # source frames per similarity matrix product
QUEUE_DEPTH = 2  # stitch batches queued per GPU worker
REPLY_TIMEOUT_S = 900  # a worker silent this long is treated as hung
SHUTDOWN_TIMEOUT_S = 60  # grace period for a worker process to exit
_PLUMBING = ("SIMILARITY_BATCH", "QUEUE_DEPTH", "REPLY_TIMEOUT_S", "SHUTDOWN_TIMEOUT_S")


def snapshot() -> dict:
    """Every constant above except process plumbing, as plain JSON values."""
    values = {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_") and name not in _PLUMBING
    }
    return json.loads(json.dumps(values))


def differing_keys(a: dict, b: dict) -> list[str]:
    """Keys present in only one of ``a`` and ``b``, or mapped to different values."""
    return sorted(
        k for k in a.keys() | b.keys() if k not in a or k not in b or a[k] != b[k]
    )


def check_snapshot(saved: dict, source: str) -> None:
    """Raise, naming every differing key, unless ``saved`` equals ``snapshot()``."""
    differ = differing_keys(saved, snapshot())
    if differ:
        raise ValueError(f"{source}: constants changed since this run began: {differ}")
