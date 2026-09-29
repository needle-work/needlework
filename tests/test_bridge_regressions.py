"""Edge cases of the bridge store, packing and the augmented sampler."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from needlework.data import bridges
from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    AugmentedPolicyDataset,
    SamplerOptions,
    _splice_bridge,
)
from needlework.sampling.observations import RobomimicObservations
from needlework.sampling.policy_dataset import PolicyDataset, WindowShape
from needlework.stitching.selection import pack

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


def test_empty_selection() -> None:
    assert len(pack([], 10)) == 0


def test_negative_frame_rejected(tmp_path: Path, can_store) -> None:
    table = bridges.Bridges(
        np.array([-1]),
        np.array([10]),
        np.array([1]),
        np.array([1.0], np.float32),
        np.array([0]),
        np.array([1]),
        can_store.arrays["action"][:1],
    )
    with pytest.raises(ValueError, match="frame"):
        bridges.save(
            tmp_path / "bad.zarr", table, can_store, horizon=23, stride=1, recipe={}
        )


def test_neighbor_window_masks_source(can_store) -> None:
    logged = PolicyDataset(
        store=can_store,
        observations=RobomimicObservations(can_store, {}, ("robot0_eef_pos",)),
        episodes=np.array([0]),
        shape=WindowShape(24, 2, 12, 1),
        relative_actions=False,
        train=False,
    )
    b = bridges.Bridges(
        np.array([20]),
        np.array([50]),
        np.array([1]),
        np.array([1.0], np.float32),
        np.array([0]),
        np.array([2]),
        can_store.arrays["action"][20:22],
    )
    data = AugmentedPolicyDataset(
        logged,
        b,
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=42,
    )
    index = int(np.flatnonzero(logged.windows.start == 18)[0])
    assert not data.logged_arrays(np.array([index]))[2][0, 2]


def test_failure_bridge_never_supervises_failure_actions(can_store) -> None:
    """A stage-3 bridge departs a failure episode: the failing policy's logged actions
    are never supervised. It gets one window, at its source; a success bridge keeps all
    its crossing windows."""
    s = can_store
    success = np.flatnonzero(s.episode_success)
    failure = int(np.flatnonzero(~s.episode_success)[0])
    logged = PolicyDataset(
        store=s,
        observations=RobomimicObservations(s, {}, ("robot0_eef_pos",)),
        episodes=success[:2],
        shape=WindowShape(24, 2, 12, 1),
        relative_actions=False,
        train=False,
    )
    recovery = int(s.episode_starts[failure]) + 200
    within = int(s.episode_starts[success[0]]) + 60
    target = int(s.episode_starts[success[1]]) + 50
    b = pack(
        [
            {
                "candidate_id": 0,
                "source": recovery,
                "target": target,
                "stage": 3,
                "margin": 1.0,
                "actions": s.arrays["action"][recovery : recovery + 10],
            },
            {
                "candidate_id": 1,
                "source": within,
                "target": within + 30,
                "stage": 1,
                "margin": 1.0,
                "actions": s.arrays["action"][within : within + 10],
            },
        ],
        s.spec.action_dim,
    )
    data = AugmentedPolicyDataset(
        logged,
        b,
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    episode_of = s.episode_of_frame(data.crossings[:, 1])
    failure_rows = np.flatnonzero(episode_of == failure)
    assert data.crossings[failure_rows, 1].tolist() == [recovery]
    assert (episode_of == success[0]).sum() == 23
    # a failure crossing has no logged twin; each success crossing replaces one
    assert len(data.logged_windows) + len(data.crossings) == len(logged) + 1
    _, _, valid = data.bridge_arrays(failure_rows)
    start = 1  # row n_obs - 1 is the source
    assert not valid[0, :start].any()
    assert valid[0, start : start + 10].all() and not valid[0, start + 10 :].any()


def test_empty_augmentation_preserves_logged_epoch(can_store):
    logged = PolicyDataset(
        store=can_store,
        observations=RobomimicObservations(can_store, {}, ("robot0_eef_pos",)),
        episodes=np.array([0]),
        shape=WindowShape(24, 2, 12, 1),
        relative_actions=False,
        train=False,
    )
    data = AugmentedPolicyDataset(
        logged,
        bridges.Bridges.empty(10),
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=42,
    )
    data.set_epoch(3)
    assert not data.is_bridge.any()
    np.testing.assert_array_equal(np.sort(data.draw), np.arange(len(logged)))


@pytest.mark.parametrize(("start", "length"), [(4, 3), (1, 0)])
def test_a_splice_needs_at_least_one_bridge_row(start: int, length: int) -> None:
    action, valid = np.zeros((4, 2), np.float32), np.ones(4, bool)
    with pytest.raises(ValueError, match="bridge"):
        _splice_bridge(action, valid, np.ones((length, 2), np.float32), start)


def test_crossings_follow_the_lattice_and_the_earlier_source_owns(can_store) -> None:
    """A success source is entered from each of the horizon - n_obs + 1 window positions
    at or before it inside its episode, a failure source from itself only; where two
    sources reach the same (episode, current frame), the earlier source owns it."""
    s = can_store
    success = np.flatnonzero(s.episode_success)
    failure = int(np.flatnonzero(~s.episode_success)[0])
    shape = WindowShape(24, 2, 12, 1)
    logged = PolicyDataset(
        store=s,
        observations=RobomimicObservations(s, {}, ("robot0_eef_pos",)),
        episodes=success[:2],
        shape=shape,
        relative_actions=False,
        train=False,
    )
    first = int(s.episode_starts[success[0]])
    sources = [first + 5, first + 30, first + 40, int(s.episode_starts[failure]) + 50]
    b = pack(
        [
            {
                "candidate_id": i,
                "source": src,
                "target": src + 50,
                "stage": 3 if i == 3 else 1,
                "margin": 1.0,
                "actions": s.arrays["action"][src : src + 4],
            }
            for i, src in enumerate(sources)
        ],
        s.spec.action_dim,
    )
    data = AugmentedPolicyDataset(
        logged,
        b,
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    expected: dict[tuple[int, int], int] = {}
    for row, src in enumerate(b.source_frame.tolist()):
        episode = int(s.episode_of_frame(np.array([src]))[0])
        leads = shape.horizon - shape.n_obs + 1 if s.episode_success[episode] else 1
        for lead in range(leads):
            current = src - lead * shape.stride
            if (
                current >= s.episode_starts[episode]
                and (episode, current) not in expected
            ):
                expected[(episode, current)] = row
    got = {(int(e), int(c)): int(r) for e, c, r in data.crossings}
    assert got == expected
    assert len(got) == 6 + 23 + 10 + 1  # clipped at the start, full, overlap, failure


def test_crossing_rows_and_first_epoch_draws_are_fixed(can_store) -> None:
    """The crossings table in row order and epoch 0's draws on a fixed fixture: a
    refactor that reorders crossings or draws changes these literals."""
    s = can_store
    success = np.flatnonzero(s.episode_success)
    failure = int(np.flatnonzero(~s.episode_success)[0])
    logged = PolicyDataset(
        store=s,
        observations=RobomimicObservations(s, {}, ("robot0_eef_pos",)),
        episodes=success[:2],
        shape=WindowShape(24, 2, 12, 1),
        relative_actions=False,
        train=False,
    )
    first = int(s.episode_starts[success[0]])
    sources = [first + 2, first + 10, int(s.episode_starts[failure]) + 50]
    b = pack(
        [
            {
                "candidate_id": i,
                "source": src,
                "target": src + 50,
                "stage": 3 if i == 2 else 1,
                "margin": 1.0,
                "actions": s.arrays["action"][src : src + 4],
            }
            for i, src in enumerate(sources)
        ],
        s.spec.action_dim,
    )
    data = AugmentedPolicyDataset(
        logged,
        b,
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    episode, current, row = data.crossings.T
    rows = list(
        zip(
            (episode == success[0]).tolist(),
            (current - s.episode_starts[episode]).tolist(),
            row.tolist(),
            strict=True,
        )
    )
    # (in the first success episode, frame offset in its episode, bridge row)
    assert rows == [
        (True, 2, 0),
        (True, 1, 0),
        (True, 0, 0),
        (True, 10, 1),
        (True, 9, 1),
        (True, 8, 1),
        (True, 7, 1),
        (True, 6, 1),
        (True, 5, 1),
        (True, 4, 1),
        (True, 3, 1),
        (False, 50, 2),
    ]
    data.set_epoch(0)
    # every window plus the two success departures' twins
    assert (len(data), int(data.is_bridge.sum()), int(data.is_twin.sum())) == (
        189,
        9,
        2,
    )
    assert hashlib.sha256(data.draw.astype(np.int64).tobytes()).hexdigest() == (
        "a0a2e0dc49fc37c693f933a3fe386903a88d9c98d42b471a886bf1b71413d22b"
    )
    assert hashlib.sha256(data.is_bridge.astype(np.bool_).tobytes()).hexdigest() == (
        "0e0d3325d26d77de1884855f4753b877b4c6f8ed719c1c0e1a87ec645b2ffc61"
    )
    assert hashlib.sha256(data.is_twin.astype(np.bool_).tobytes()).hexdigest() == (
        "22e37eea8088c692a3b0fea1e38cd33161640753dcf6ad05120acdac8ead25db"
    )
