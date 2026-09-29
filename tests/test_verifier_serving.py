"""Verifier action serving, set per domain by the task config.

- ``label_dependent`` (Robomimic): every positive holds from its
  goal step, every negative continues its logged actions.
- ``independent`` (UMI): each example holds with ``hold_probability``, independently of
  its label.
"""

import numpy as np
import pytest
import torch

from needlework import config
from needlework.constants import HOLD_STREAM, PATCHES, SPATIAL
from needlework.data import features
from needlework.sampling import proximity
from needlework.sampling.pairs import hold_from
from needlework.sampling.verifier_pairs import VerifierPairs
from needlework.training.common import Windows


def _pairs(serving: dict) -> VerifierPairs:
    cfg = config.compose(["component=verifier", "task=robomimic/square", "run.name=t"])
    w = Windows(cfg, cameras=False, n_execute=cfg.horizon.execute)
    success = w.store.episode_success
    train = w.episodes(successes_only=False).train
    episodes = np.concatenate([train[success[train]][:3], train[~success[train]][:3]])
    spatial = features.load(w.store, SPATIAL)
    rows = features.open_rows(w.store, PATCHES)
    device = torch.device("cuda")
    sampling = cfg.component.sampling
    tau = proximity.calibrate_tau(
        proximity.embeddings(spatial, device),
        w.store.episode_starts,
        w.store.episode_ends,
        23,
        percentile=sampling.proximity.tau_percentile,
    )
    return VerifierPairs(
        w.dataset(np.sort(episodes), train=True),
        spatial,
        {camera: rows[camera] for camera in cfg.task.obs.cameras},
        {**dict(sampling), "serving": serving},
        tau=tau,
        seed=sampling.seed.train,
        resample=True,
        device=device,
    )


def _served(p: VerifierPairs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(served actions, logged actions, positive) for every example of the epoch."""
    idx = np.arange(len(p))
    _, logged = p.data.arrays(p.row[idx], False)
    return p.batch(idx)["action"].numpy(), logged, p.offset[idx] > 0


def test_label_dependent_positives_hold_negatives_continue() -> None:
    p = _pairs({"mode": "label_dependent"})
    p.set_epoch(3)
    served, logged, positive = _served(p)
    n_obs = p.data.shape.n_obs
    held = hold_from(logged, p.offset, n_obs)
    np.testing.assert_array_equal(served[positive], held[positive])
    np.testing.assert_array_equal(served[~positive], logged[~positive])
    assert positive.any() and (~positive).any()


def test_independent_serving_holds_with_its_probability() -> None:
    p = _pairs({"mode": "independent", "hold_probability": 0.5})
    p.set_epoch(3)
    served, logged, _ = _served(p)
    rng = np.random.default_rng([p.seed, 3, HOLD_STREAM])
    hold = rng.random(len(p.row)) < 0.5
    random_offset = rng.integers(1, np.minimum(p.max_future[p.row], p.horizon) + 1)
    offset = np.where(p.offset > 0, p.offset, random_offset)
    expected = np.where(
        hold[:, None, None], hold_from(logged, offset, p.data.shape.n_obs), logged
    )
    np.testing.assert_array_equal(served, expected)


@pytest.mark.parametrize(
    "serving",
    [
        {"mode": "label_dependent", "hold_probability": 0.5},  # would be ignored
        {"mode": "independent"},  # no silent default probability
        {"mode": "sometimes"},
    ],
)
def test_bad_serving_raises(serving) -> None:
    with pytest.raises(ValueError, match="serving"):
        VerifierPairs.check_serving(serving)


def test_task_configs_choose_the_serving() -> None:
    def serving(*task: str) -> dict:
        cfg = config.compose(["component=verifier", *task, "run.name=t"])
        return dict(cfg.component.sampling.serving)

    for task in ("robomimic/can", "robomimic/square", "robomimic/transport"):
        assert serving(f"task={task}") == {"mode": "label_dependent"}
    umi = serving("task=umi", "task.name=sweater")
    assert umi == {"mode": "independent", "hold_probability": 0.5}
