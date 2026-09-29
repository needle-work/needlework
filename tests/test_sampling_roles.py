"""Window weights by role on the stitched routes (sampling/roles.py) and their use in
the augmented epoch."""

import numpy as np
import pytest
from test_stitches import logged, record

from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    AugmentedPolicyDataset,
    SamplerOptions,
)
from needlework.sampling.pairs import current_frames
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


def _episodes(store):
    e0, e1 = (int(e) for e in np.flatnonzero(store.episode_success)[:2])
    failure = int(np.flatnonzero(~store.episode_success)[0])
    return e0, e1, failure


def _bridges(store):
    """Stage 1 inside e0 (+60 -> +90), stage 2 from e1 (+50) into e0 (+70), stage 3 from
    a failure episode into e1 (+30)."""
    e0, e1, failure = _episodes(store)
    s0, s1, sf = (int(store.episode_starts[e]) for e in (e0, e1, failure))
    rows = [
        record(store, s0 + 60, s0 + 90, 1, candidate=0),
        record(store, s1 + 50, s0 + 70, 2, candidate=1),
        record(store, sf + 20, s1 + 30, 3, candidate=2),
    ]
    return pack(rows, store.spec.action_dim), (s0, s1, sf)


@pytest.mark.parametrize("skipped", [0.5, 1.0])
def test_frame_weights_by_role(can_store, skipped):
    """(Episode 0 of the Can store has 101 frames.) Frames a route skips (after a
    stage-1 source up to its target, after a stage-2 source to its episode end) weigh
    the task's skipped weight; a stage-2 target's suffix weighs TARGET_SUFFIX even where
    it was skipped; every other frame CANONICAL, including frames past a stage-1 target.
    """
    s = can_store
    b, (s0, s1, _) = _bridges(s)
    e0, e1, _ = _episodes(s)
    end0, end1 = int(s.episode_ends[e0]), int(s.episode_ends[e1])
    want = np.full(s.n_frames, roles.CANONICAL)
    want[s0 + 61 : s0 + 90] = skipped
    want[s1 + 51 : end1] = skipped
    want[s0 + 70 : end0] = roles.TARGET_SUFFIX
    np.testing.assert_array_equal(
        roles.frame_weights(s, b, roles.RoleWeights(skipped, 0.5, 0.5, 1.0)), want
    )


def test_window_weights_departures_and_twins(can_store):
    """A crossing window weighs its current frame's role weight, except the departure
    itself (current frame = source), which weighs bridge_weight (success) or
    ``failure_departure`` (failure). Each success departure keeps its logged twin at
    ``twin``, whose own logged action at the source is supervised."""
    s = can_store
    b, (s0, s1, sf) = _bridges(s)
    e0, e1, _ = _episodes(s)
    train = logged(s, [e0, e1])
    data = AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    frame = roles.frame_weights(s, data.bridges, ROLE_WEIGHTS)
    by_current = dict(
        zip(data.crossings[:, 1].tolist(), data.crossing_weight.tolist(), strict=True)
    )
    assert by_current[s0 + 60] == 0.2 and by_current[s1 + 50] == 0.2
    assert by_current[sf + 20] == ROLE_WEIGHTS.failure_departure
    assert by_current[s0 + 57] == frame[s0 + 57] == roles.CANONICAL
    assert by_current[s1 + 47] == frame[s1 + 47]
    current = current_frames(train)
    np.testing.assert_array_equal(
        data.logged_weight, frame[current[data.logged_windows]]
    )
    np.testing.assert_array_equal(np.sort(current[data.twins]), [s0 + 60, s1 + 50])
    valid = data.twin_arrays(data.twins)[2]
    assert valid[:, train.shape.n_obs - 1].all()  # the logged action at the source
    assert not data.logged_arrays(data.twins)[2][:, train.shape.n_obs - 1].any()


def test_epoch_draws_follow_mass(can_store):
    """An epoch holds len(logged) + one draw per twin; each (kind, weight) group gets
    its share of the total mass (rounded along the cumulative mass) as even shuffled
    passes."""
    s = can_store
    b, _ = _bridges(s)
    e0, e1, _ = _episodes(s)
    train = logged(s, [e0, e1])
    data = AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=5,
    )
    positive = (data.logged_weight > 0).sum() + (data.crossing_weight > 0).sum()
    assert len(data) == positive + len(data.twins) == len(data.draw)
    mass = {
        "logged": data.logged_weight.sum(),
        "twin": ROLE_WEIGHTS.twin * len(data.twins),
        "bridge": data.crossing_weight.sum(),
    }
    total = sum(mass.values())
    groups = (
        len(np.unique(data.logged_weight)) + 1 + len(np.unique(data.crossing_weight))
    )
    assert not np.isin(
        data.draw[~data.is_twin & ~data.is_bridge],
        data.logged_windows[data.logged_weight == 0],
    ).any()
    for kind, flags in (
        ("twin", data.is_twin),
        ("bridge", data.is_bridge),
        ("logged", ~data.is_twin & ~data.is_bridge),
    ):
        assert abs(flags.sum() - len(data) * mass[kind] / total) <= groups
    logged_draws = np.bincount(
        data.draw[~data.is_twin & ~data.is_bridge], minlength=len(train)
    )
    for value in np.unique(data.logged_weight):
        per = logged_draws[data.logged_windows[data.logged_weight == value]]
        assert per.max() - per.min() <= 1


def test_frame_weights_reject_malformed_bridges(can_store):
    """A stage-1 bridge must stay in its episode, a stage-2 bridge must leave it, and
    every stage is 1, 2 or 3."""
    s = can_store
    e0, e1, _ = _episodes(s)
    s0, s1 = int(s.episode_starts[e0]), int(s.episode_starts[e1])
    for row in (
        record(s, s0 + 10, s1 + 10, 1),  # stage 1 into another episode
        record(s, s0 + 10, s0 + 50, 2),  # stage 2 inside its own episode
        record(s, s0 + 10, s0 + 50, 4),  # no such stage
    ):
        with pytest.raises(ValueError, match="bridge"):
            roles.frame_weights(s, pack([row], s.spec.action_dim), ROLE_WEIGHTS)


def test_small_groups_are_drawn_in_expectation(can_store):
    """Draw counts round the cumulative mass at a random offset per epoch: a group whose
    expected count is below one draw is drawn in that fraction of epochs (not never),
    and every group gets the floor or ceiling of its expected count."""
    s = can_store
    e0, _, _ = _episodes(s)
    s0 = int(s.episode_starts[e0])
    b = pack([record(s, s0 + 60, s0 + 90, 1)], s.spec.action_dim)
    data = AugmentedPolicyDataset(
        logged(s, [e0]),
        b,
        weight=0.05,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    departure = int(np.flatnonzero(data.crossing_weight == 0.05)[0])
    total = (
        data.logged_weight.sum()
        + data.crossing_weight.sum()
        + ROLE_WEIGHTS.twin * len(data.twins)
    )
    expected = len(data) * 0.05 / total
    assert expected < 1
    draws = []
    for epoch in range(400):
        data.set_epoch(epoch)
        n = int((data.draw[data.is_bridge] == departure).sum())
        assert n in (0, 1)
        draws.append(n)
    assert abs(np.mean(draws) - expected) < 0.05


def test_epoch_refuses_zero_total_weight(can_store, monkeypatch):
    s = can_store
    e0, _, _ = _episodes(s)
    monkeypatch.setattr(roles, "CANONICAL", 0.0)
    with pytest.raises(ValueError, match="positive weight"):
        AugmentedPolicyDataset(
            logged(s, [e0]),
            pack([], s.spec.action_dim),
            weight=1.0,
            role_weights=ROLE_WEIGHTS,
            epoch_length="windows",
            options=OPTIONS,
            seed=0,
        )


def test_replacement_setting(can_store):
    """Skipped 1, twin 0, failure departure and approach at the bridge weight, epoch the
    logged length: every remaining logged window weighs 1, every crossing window the
    bridge weight, no twin is drawn, and an epoch is len(logged) draws."""
    s = can_store
    b, _ = _bridges(s)
    e0, e1, _ = _episodes(s)
    train = logged(s, [e0, e1])
    weights = roles.RoleWeights(
        skipped=1.0, twin=0.0, failure_departure=0.2, approach=0.2
    )
    data = AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=weights,
        epoch_length="logged",
        options=OPTIONS,
        seed=5,
    )
    assert len(data.twins) and not np.count_nonzero(data.twin_weight)
    np.testing.assert_array_equal(data.logged_weight, np.ones(len(data.logged_windows)))
    np.testing.assert_array_equal(
        data.crossing_weight, np.full(len(data.crossings), 0.2)
    )
    assert len(data) == len(train) == len(data.draw)
    assert not data.is_twin.any()
    mass = 0.2 * len(data.crossings)
    expected = len(train) * mass / (mass + len(data.logged_windows))
    assert data.is_bridge.sum() in (np.floor(expected), np.ceil(expected))


def test_approach_weight_scales_the_frame_weight(can_store):
    s = can_store
    b, _ = _bridges(s)
    e0, e1, _ = _episodes(s)
    train = logged(s, [e0, e1])
    base = AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    weights = roles.RoleWeights(
        skipped=0.5, twin=0.5, failure_departure=0.5, approach=3.0
    )
    scaled = AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=weights,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    departure = base.crossings[:, 1] == base.bridges.source_frame[base.crossings[:, 2]]
    np.testing.assert_array_equal(
        scaled.crossing_weight[~departure], 3.0 * base.crossing_weight[~departure]
    )
    np.testing.assert_array_equal(
        scaled.crossing_weight[departure], base.crossing_weight[departure]
    )


@pytest.mark.parametrize("bad", ["skipped", "twin", "failure_departure", "approach"])
def test_role_weights_reject_negative(bad):
    values = {
        "skipped": 0.5,
        "twin": 0.5,
        "failure_departure": 0.5,
        "approach": 1.0,
        bad: -0.1,
    }
    with pytest.raises(ValueError, match=bad):
        roles.RoleWeights(**values)


def test_unknown_epoch_length_raises(can_store):
    s = can_store
    e0, _, _ = _episodes(s)
    with pytest.raises(ValueError, match="epoch_length"):
        AugmentedPolicyDataset(
            logged(s, [e0]),
            pack([], s.spec.action_dim),
            weight=1.0,
            role_weights=ROLE_WEIGHTS,
            epoch_length="stitched",
            options=OPTIONS,
            seed=0,
        )
