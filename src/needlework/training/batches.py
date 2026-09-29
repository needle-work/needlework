"""Batch sources and the epoch draw order."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class BatchSource(Protocol):
    def __len__(self) -> int: ...

    def batch(self, indices: np.ndarray) -> dict: ...

    def set_epoch(self, epoch: int) -> None:
        """Fix the examples drawn for ``epoch`` (resume replays the same draws)."""
        ...


def epoch_order(
    n: int, batch_size: int, generator: torch.Generator
) -> list[np.ndarray]:
    """A seeded permutation of ``range(n)`` cut into batches; the last may be short."""
    order = torch.randperm(n, generator=generator).numpy()
    return [order[start : start + batch_size] for start in range(0, n, batch_size)]


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: to_device(value, device)
        if isinstance(value, dict)
        else value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
