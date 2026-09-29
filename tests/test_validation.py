"""Validation loss is the mean over validation examples, not over batches."""

import numpy as np
import pytest
import torch

from needlework.training.validation import mean_loss


class _Examples:
    """Five examples whose loss is their own value: batches of 4 and 1 at size 4."""

    values = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])

    def __len__(self) -> int:
        return len(self.values)

    def batch(self, indices: np.ndarray) -> dict:
        return {"x": self.values[torch.from_numpy(indices)]}

    def set_epoch(self, epoch: int) -> None:
        pass


def test_val_loss_weights_a_short_last_batch_by_its_size() -> None:
    loss = mean_loss(
        lambda batch: batch["x"].mean(),
        _Examples(),
        batch_size=4,
        seed=0,
        device=torch.device("cuda"),
    )
    assert loss == pytest.approx(9.0 / 5.0)  # the mean of batch means would be 3.0
