"""Logged actions within one policy step of a selected bridge source are never
supervised, in logged windows and in the history rows of bridge windows alike."""

import numpy as np
import pytest

from needlework.data import bridges
from needlework.data.store import EpisodeStore
from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    AugmentedPolicyDataset,
    SamplerOptions,
)
from needlework.sampling.observations import RobomimicObservations
from needlework.sampling.policy_dataset import PolicyDataset, WindowShape
from needlework.sampling.windows import window_frames

ROLE_WEIGHTS = roles.RoleWeights(
    skipped=0.5, twin=0.5, failure_departure=0.5, approach=1.0
)
OPTIONS = SamplerOptions(
    path_start="aligned",
    twin_rows="own_source",
    draw="even_passes",
    source_rows="masked",
    after_departure="all",
)


@pytest.fixture(scope="module")
def square() -> EpisodeStore:
    return EpisodeStore.open("robomimic", "square")


def _data(store: EpisodeStore) -> AugmentedPolicyDataset:
    episode = int(np.flatnonzero(store.episode_success)[0])
    first = int(store.episode_starts[episode])
    logged = PolicyDataset(
        store=store,
        observations=RobomimicObservations(store, {}, ("robot0_eef_pos",)),
        episodes=np.array([episode]),
        shape=WindowShape(24, 2, 12, 1),
        relative_actions=False,
        train=False,
    )
    sources = np.array([first + 40, first + 44])  # the second crossing sees the first
    table = bridges.Bridges(
        source_frame=sources,
        target_frame=sources + 80,
        stage=np.array([1, 1]),
        verifier_margin=np.array([1.0, 1.0], np.float32),
        action_start=np.array([0, 3]),
        action_len=np.array([3, 3]),
        actions=store.arrays["action"][first + 100 : first + 106],
    )
    return AugmentedPolicyDataset(
        logged,
        table,
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=42,
    )


def test_sources_are_masked_in_logged_and_bridge_windows(square) -> None:
    data = _data(square)
    windows = np.arange(len(data.data))
    frames = window_frames(data.data.windows, windows, square.episode_ends)
    near = data.near_sources(frames)
    np.testing.assert_array_equal(near, np.isin(frames, data.selected_sources))
    np.testing.assert_array_equal(data.logged_arrays(windows)[2], ~near)
    crossings = np.arange(len(data.crossings))
    _, _, valid = data.bridge_arrays(crossings)
    for i, (_, current, row) in enumerate(data.crossings):
        start = data.data.shape.n_obs - 1 + (data.bridges.source_frame[row] - current)
        history = current + np.arange(start) - (data.data.shape.n_obs - 1)
        np.testing.assert_array_equal(
            valid[i, :start], ~np.isin(history, data.selected_sources)
        )
        assert valid[i, start : start + 3].all()
