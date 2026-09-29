"""Verifier model contracts: label/action alignment, causal action attention, threshold
calibration and the balanced BCE."""

import numpy as np
import pytest
import torch

from needlework.models.verifier import Verifier, balanced_bce, calibrate
from needlework.sampling.pairs import hold_from

N_OBS, HORIZON = 2, 23  # 24 window rows: one history row, then 23 executable rows


def test_hold_and_labels_align_with_the_goal_step() -> None:
    """A positive whose goal is k steps ahead: rows from n_obs - 1 + k repeat row
    n_obs - 2 + k (the k-th executable action), and "reached by step h" is 1 iff
    h >= k."""
    k = 5
    action = np.arange(N_OBS - 1 + HORIZON, dtype=np.float32)[None, :, None]
    held = hold_from(action, np.array([k]), N_OBS)[0, :, 0]
    np.testing.assert_array_equal(held[: N_OBS - 1 + k], action[0, : N_OBS - 1 + k, 0])
    assert (held[N_OBS - 1 + k :] == N_OBS - 2 + k).all()
    labels = np.arange(1, HORIZON + 1) >= k  # as VerifierPairs.batch builds them
    assert labels[k - 1] and not labels[k - 2]


def _verifier() -> Verifier:
    torch.manual_seed(0)
    return (
        Verifier(
            n_cameras=2,
            patch_grid=7,
            feature_dim=16,
            proprio={"p": 3},
            action_dim=4,
            horizon=HORIZON,
            n_obs=N_OBS,
            hidden=32,
            layers=2,
            heads=4,
            ff_mult=2,
        )
        .cuda()
        .eval()
    )


def _batch() -> dict:
    g = torch.Generator().manual_seed(1)
    return {
        "obs": {"p": torch.randn(3, N_OBS, 3, generator=g)},
        "source_patches": torch.randn(3, 2, 49, 16, generator=g),
        "goal_patches": torch.randn(3, 2, 49, 16, generator=g),
        "action": torch.randn(3, N_OBS - 1 + HORIZON, 4, generator=g),
    }


def _cuda(batch: dict) -> dict:
    return {
        "obs": {k: v.cuda() for k, v in batch["obs"].items()},
        **{k: v.cuda() for k, v in batch.items() if k != "obs"},
    }


@torch.no_grad()
def test_logit_h_sees_only_the_first_h_executable_rows() -> None:
    """Step h's logit reads window rows n_obs-1 .. n_obs-1+h: changing executable row j
    leaves the logits of earlier steps unchanged and changes step j; the history row is
    not read at all."""
    model, batch = _verifier(), _batch()
    base = model(_cuda(batch))
    for j in (0, 7, HORIZON - 1):
        changed = {**batch, "action": batch["action"].clone()}
        changed["action"][:, N_OBS - 1 + j] += 5.0
        out = model(_cuda(changed))
        torch.testing.assert_close(out[:, :j], base[:, :j], rtol=0, atol=1e-6)
        assert (out[:, j] - base[:, j]).abs().min() > 1e-4
    history = {**batch, "action": batch["action"].clone()}
    history["action"][:, 0] += 5.0
    torch.testing.assert_close(model(_cuda(history)), base, rtol=0, atol=1e-6)


def test_calibrate_is_the_negative_quantile_per_step() -> None:
    g = torch.Generator().manual_seed(2)
    logits, labels = torch.randn(400, 3, generator=g), torch.zeros(400, 3)
    labels[:100, 1] = 1.0
    thresholds = calibrate(logits, labels, target_fpr=0.1, min_negatives=200)
    for h in range(3):
        negatives = logits[labels[:, h] == 0.0, h]
        torch.testing.assert_close(thresholds[h], torch.quantile(negatives, 0.9))
    with pytest.raises(ValueError, match="negatives"):
        calibrate(logits, labels, target_fpr=0.1, min_negatives=301)


def test_balanced_bce_averages_positives_and_per_row_negatives() -> None:
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(6, 4, generator=g)
    labels = (torch.rand(6, 4, generator=g) > 0.6).float()
    labels[0] = 1.0  # a row with no negatives is left out of the negative term
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    positive, negative = labels == 1.0, labels == 0.0
    rows = negative.any(1)
    per_row = (bce * negative).sum(1)[rows] / negative.sum(1)[rows]
    expected = (bce[positive].mean() + per_row.mean()) / 2
    torch.testing.assert_close(balanced_bce(logits, labels), expected)
    assert not torch.isclose(balanced_bce(logits, 1.0 - labels), expected)
