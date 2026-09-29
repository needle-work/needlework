"""Validation metrics, computed on the EMA model in fp32.

``val_loss`` is the loss over the whole validation set: for the policy and the IDM the
mean of per-window losses (batches weighted by their size), for the verifier its
balanced BCE over all validation examples at once, from the same forward pass that
calibrates its thresholds. Neither is a mean of batch means.

Each metric draws its noise from a fixed seed inside a forked RNG, so values are
comparable across epochs and computing them does not shift the training random stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from needlework.models.policy import ConditionedDiffusion
from needlework.training.batches import BatchSource, to_device


@torch.no_grad()
def mean_loss(
    loss: Callable[[dict], torch.Tensor],
    data: BatchSource,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    """Mean over every example of ``data`` of a loss that ``loss`` averages per batch:
    each batch's value is weighted by its size, so a short last batch counts once per
    example, not once per batch."""
    total = torch.zeros((), device=device)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        for start in tqdm(range(0, len(data), batch_size), desc="val", leave=False):
            indices = np.arange(start, min(start + batch_size, len(data)))
            total += loss(to_device(data.batch(indices), device)) * len(indices)
    return float(total / len(data))


PROBE_STREAMS = {"train": 0, "val": 1}  # random streams of the per-split probes


def fixed_sample(n: int, size: int, seed: int) -> np.ndarray:
    """``size`` distinct indices out of ``n``, the same every epoch."""
    return np.random.RandomState(seed).choice(n, size=min(size, n), replace=False)


def probes(
    *, train: BatchSource, val: BatchSource, size: int
) -> dict[str, tuple[BatchSource, np.ndarray]]:
    """Fixed examples of each split for the sampled-action metrics."""
    return {
        "train": (train, fixed_sample(len(train), size, PROBE_STREAMS["train"])),
        "val": (val, fixed_sample(len(val), size, PROBE_STREAMS["val"])),
    }


def action_mse(
    policy: ConditionedDiffusion,
    data: BatchSource,
    indices: np.ndarray,
    inputs: Callable[[dict], Any],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """MSE between sampled and logged actions (unnormalized), pooled over valid rows:
    all ``horizon`` rows, and the rows a controller executes."""
    shape = policy.shape
    execute = np.zeros(shape.horizon, dtype=bool)
    execute[shape.n_obs - 1 : shape.n_obs - 1 + shape.n_execute] = True
    execute = torch.from_numpy(execute).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    sums = {"all": 0.0, "executed": 0.0}
    rows = {"all": 0, "executed": 0}
    for start in range(0, len(indices), batch_size):
        batch = to_device(data.batch(indices[start : start + batch_size]), device)
        generators = [generator] * len(batch["action"])
        predicted = policy.predict(inputs(batch), generators)
        error = ((predicted - batch["action"]) ** 2).mean(-1)
        for name, mask in (
            ("all", batch["valid"]),
            ("executed", batch["valid"] & execute),
        ):
            sums[name] += float(error[mask].sum())
            rows[name] += int(mask.sum())
    return {name: sums[name] / rows[name] for name in sums}


def action_mse_metrics(
    policy: ConditionedDiffusion,
    probes: dict[str, tuple[BatchSource, np.ndarray]],
    inputs: Callable[[dict], Any],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """``{split}_action_mse`` and ``{split}_action_mse_executed`` per probe split."""
    out = {}
    for split, (data, indices) in probes.items():
        mse = action_mse(
            policy,
            data,
            indices,
            inputs,
            batch_size=batch_size,
            seed=seed + PROBE_STREAMS[split],
            device=device,
        )
        out[f"{split}_action_mse"] = mse["all"]
        out[f"{split}_action_mse_executed"] = mse["executed"]
    return out
