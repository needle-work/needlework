"""Seeds, strict determinism, and the random state saved with every checkpoint.

Strict mode requires ``CUBLAS_WORKSPACE_CONFIG`` before cuBLAS starts;
``needlework.train`` sets it first thing.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

CUBLAS_WORKSPACE = ":4096:8"


def configure(*, seed: int, strict: bool) -> torch.Generator:
    """Seed python, numpy and torch; return the generator that drives data order."""
    if strict:
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != CUBLAS_WORKSPACE:
            raise RuntimeError("strict determinism needs CUBLAS_WORKSPACE_CONFIG set")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def capture(data_generator: torch.Generator) -> dict:
    """Every random stream, as tensors and plain values (loadable with weights_only)."""
    version, keys, gauss = random.getstate()
    name, np_keys, np_pos, has_gauss, cached = np.random.get_state()
    return {
        "python": {"version": version, "keys": list(keys), "gauss": gauss},
        "numpy": {
            "name": name,
            "keys": torch.from_numpy(np_keys.astype(np.int64)),
            "pos": int(np_pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "data": data_generator.get_state(),
    }


def restore(state: dict, data_generator: torch.Generator) -> None:
    py = state["python"]
    random.setstate((py["version"], tuple(py["keys"]), py["gauss"]))
    nps = state["numpy"]
    np.random.set_state(
        (
            nps["name"],
            nps["keys"].numpy().astype(np.uint32),
            nps["pos"],
            nps["has_gauss"],
            nps["cached"],
        )
    )
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    data_generator.set_state(state["data"])
