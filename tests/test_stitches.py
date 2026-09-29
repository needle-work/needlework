"""Compact bridge format and augmented windows, using real stores in both domains."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import zarr

from needlework.data import bridges
from needlework.geometry.actions import action_poses, from_relative, to_relative
from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    AugmentedPolicyDataset,
    SamplerOptions,
)
from needlework.sampling.pairs import current_frames
from needlework.sampling.policy_dataset import PolicyDataset, WindowShape
from needlework.sampling.windows import window_frames
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


def logged(store, episodes):
    # Geometry is real; camera reads are unnecessary for this focused sampler test.
    def obs(frames, episodes, train):
        return {"frame": frames[..., None].astype(np.float32)}

    return PolicyDataset(
        store=store,
        observations=obs,
        episodes=np.asarray(episodes),
        shape=WindowShape(24, 2, 12, 3 if store.domain == "umi" else 1),
        relative_actions=store.domain == "umi",
        train=False,
    )


def record(store, source, target, stage=1, length=3, candidate=0):
    action = store.arrays["action"][source : source + length].copy()
    if store.domain == "umi":
        action = to_relative(
            store.layout,
            action,
            action_poses(store.layout, store.arrays["action"][source]),
        )
    return {
        "candidate_id": candidate,
        "source": source,
        "target": target,
        "stage": stage,
        "margin": 1.0,
        "actions": action,
    }


@pytest.mark.parametrize("fixture", ["can_store", "sweater_store"])
def test_roundtrip_empty_and_integrity(fixture, request, tmp_path):
    store = request.getfixturevalue(fixture)
    source = int(store.episode_starts[np.flatnonzero(store.episode_success)[0]]) + 30
    stride = 3 if store.domain == "umi" else 1
    for name, b in [
        ("empty", bridges.Bridges.empty(store.spec.action_dim)),
        ("one", pack([record(store, source, source + 20)], store.spec.action_dim)),
    ]:
        path = tmp_path / name
        bridges.save(path, b, store, horizon=23, stride=stride, recipe={"test": True})
        other = bridges.load(path, store, horizon=23, stride=stride)
        assert other.digest() == b.digest()
        with pytest.raises(ValueError, match="stride"):
            bridges.load(path, store, horizon=23, stride=stride + 1)
        with pytest.raises(FileExistsError):
            bridges.save(path, b, store, horizon=23, stride=stride, recipe={})
        if len(b):
            zarr.open_group(str(path), mode="a")["actions"][0, 0] += 0.1
            with pytest.raises(ValueError, match="content changed"):
                bridges.load(path, store, horizon=23, stride=stride)
    b = pack([record(store, source, source + 20)], store.spec.action_dim)
    for bad in [
        replace(b, stage=np.array([3])),
        replace(b, action_len=np.array([24])),
        replace(b, action_start=np.array([1000])),
        replace(b, target_frame=np.array([store.n_frames])),
    ]:
        with pytest.raises(ValueError):
            bridges.validate(bad, store, horizon=23, stride=stride)


@pytest.mark.parametrize("fixture", ["can_store", "sweater_store"])
def test_crossing_frames_masks_and_mixture(fixture, request):
    store = request.getfixturevalue(fixture)
    e = int(np.flatnonzero(store.episode_success)[0])
    src = int(store.episode_starts[e]) + 60
    train = logged(store, [e])
    stride = train.shape.stride
    b = pack([record(store, src, src + 20)], store.spec.action_dim)
    data = AugmentedPolicyDataset(
        train,
        b,
        weight=0.25,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=42,
    )
    positive = (data.logged_weight > 0).sum() + (data.crossing_weight > 0).sum()
    assert len(data) == positive + len(data.twins) and len(data.twins) == 1
    mass = data.crossing_weight.sum()
    total = mass + data.logged_weight.sum() + ROLE_WEIGHTS.twin * len(data.twins)
    assert (
        abs(data.is_bridge.sum() - len(data) * mass / total) <= 4
    )  # per-group rounding
    first = data.draw.copy()
    flags = data.is_bridge.copy()
    data.set_epoch(0)
    np.testing.assert_array_equal(first, data.draw)
    data.set_epoch(1)
    assert not np.array_equal(flags, data.is_bridge)
    crossing = int(np.flatnonzero(data.crossings[:, 1] == src - 2 * stride)[0])
    _, action, valid = data.bridge_arrays(np.array([crossing]))
    expected = b.rows(0)
    if store.domain == "umi":
        absolute = from_relative(
            store.layout,
            expected,
            action_poses(store.layout, store.arrays["action"][src]),
        )
        expected = to_relative(
            store.layout,
            absolute,
            action_poses(store.layout, store.arrays["action"][src - 2 * stride]),
        )
    np.testing.assert_allclose(action[0, 3:6], expected, atol=2e-6)
    assert valid[0, :6].all() and not valid[0, 6:].any()
    np.testing.assert_allclose(
        action[0, 6:], np.repeat(action[0, 5:6], 18, axis=0), atol=2e-6
    )
    idx = np.flatnonzero(current_frames(train) == src - stride)
    assert not data.logged_arrays(idx)[2][0, 2]
    empty = AugmentedPolicyDataset(
        train,
        bridges.Bridges.empty(store.spec.action_dim),
        weight=0.25,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=42,
    )
    assert not empty.is_bridge.any()


def test_source_only_isolation(can_store):
    s = can_store
    successes = np.flatnonzero(s.episode_success)
    failure = int(np.flatnonzero(~s.episode_success)[0])
    train, held = int(successes[0]), int(successes[1])
    starts = s.episode_starts
    rows = [
        record(s, int(starts[train]) + 20, int(starts[held]) + 100, 2, candidate=0),
        record(s, int(starts[held]) + 20, int(starts[held]) + 100, 1, candidate=1),
        record(s, int(starts[failure]) + 20, int(starts[held]) + 100, 3, candidate=2),
    ]
    data = AugmentedPolicyDataset(
        logged(s, [train]),
        pack(rows, s.spec.action_dim),
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=1,
    )
    assert set(s.episode_of_frame(data.bridges.source_frame)) == {train, failure}
    assert held not in data.crossings[:, 0]
    assert (s.episode_of_frame(data.bridges.target_frame) == held).all()


def test_bridge_weight_epoch_draws(can_store):
    """Each epoch draws len(logged) + one per twin windows as shuffled passes: every
    window in proportion to its weight (role weight, bridge_weight at a departure,
    ``twin`` for a departure's logged twin), each equal-weight group spread evenly.
    Weight 0 is the logged epoch."""
    s = can_store
    e0, e1 = (int(e) for e in np.flatnonzero(s.episode_success)[:2])
    train = logged(s, [e0, e1])
    rows = [
        record(s, int(s.episode_starts[e]) + 60, int(s.episode_starts[e]) + 90, 1, 3, i)
        for i, e in enumerate((e0, e1))
    ]
    b = pack(rows, s.spec.action_dim)
    zero = AugmentedPolicyDataset(
        train,
        b,
        weight=0.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=7,
    )
    for epoch in (0, 5):
        zero.set_epoch(epoch)
        assert not zero.is_bridge.any() and not zero.is_twin.any()
        np.testing.assert_array_equal(np.sort(zero.draw), np.arange(len(train)))
    assert zero.logged_arrays(np.arange(len(train)))[2].all()  # no source masks
    data = AugmentedPolicyDataset(
        train,
        b,
        weight=0.5,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=7,
    )
    c, n = len(data.crossings), len(data.logged_windows)
    assert c == 2 * 23 and n == len(train) - c  # each crossing replaces its twin
    positive = (data.logged_weight > 0).sum() + (data.crossing_weight > 0).sum()
    assert len(data.twins) == 2 and len(data) == positive + 2
    data.set_epoch(0)
    total = (
        data.logged_weight.sum() + data.crossing_weight.sum() + 2 * ROLE_WEIGHTS.twin
    )
    expected = len(data) * data.crossing_weight.sum() / total
    assert abs(data.is_bridge.sum() - expected) <= 4
    logged_draws = np.bincount(
        data.draw[~data.is_bridge & ~data.is_twin], minlength=len(train)
    )
    for value in np.unique(data.logged_weight):
        per = logged_draws[data.logged_windows[data.logged_weight == value]]
        assert per.max() - per.min() <= 1
    per_crossing = np.bincount(data.draw[data.is_bridge], minlength=c)
    for value in np.unique(data.crossing_weight):
        per = per_crossing[data.crossing_weight == value]
        assert per.max() - per.min() <= 1
    first = (data.draw.copy(), data.is_bridge.copy())
    data.set_epoch(1)
    assert not np.array_equal(first[0], data.draw)
    data.set_epoch(0)
    np.testing.assert_array_equal(first[0], data.draw)
    np.testing.assert_array_equal(first[1], data.is_bridge)


def test_bridge_weight_zero_batches_equal_the_plain_dataset(can_store):
    """Weight 0 ignores the bridges: an epoch's batches hold exactly the plain dataset's
    windows (sampler=normal), every action supervised."""
    s = can_store
    e0, e1 = (int(e) for e in np.flatnonzero(s.episode_success)[:2])
    train = logged(s, [e0, e1])
    rows = [
        record(s, int(s.episode_starts[e]) + 60, int(s.episode_starts[e]) + 90, 1, 3, i)
        for i, e in enumerate((e0, e1))
    ]
    zero = AugmentedPolicyDataset(
        train,
        pack(rows, s.spec.action_dim),
        weight=0.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=7,
    )
    zero.set_epoch(3)
    got = zero.batch(np.argsort(zero.draw))  # positions holding windows 0, 1, ...
    want = train.batch(np.arange(len(train)))
    for key in want["obs"]:
        assert torch.equal(got["obs"][key], want["obs"][key])
    assert torch.equal(got["action"], want["action"])
    assert torch.equal(got["valid"], want["valid"])


@pytest.mark.parametrize("fixture", ["can_store", "sweater_store"])
def test_source_mask_covers_one_policy_step(fixture, request):
    """A logged row is masked when it lies within one policy step (stride raw frames)
    of a selected source in the same episode. On UMI (stride 3) logged windows whose
    lattice skips the source still land 1-2 frames from it: those rows are masked.
    On Robomimic (stride 1) only the source frame itself."""
    store = request.getfixturevalue(fixture)
    e = int(np.flatnonzero(store.episode_success)[0])
    train = logged(store, [e])
    stride = train.shape.stride
    src = int(store.episode_starts[e]) + 60
    data = AugmentedPolicyDataset(
        train,
        pack([record(store, src, src + 20)], store.spec.action_dim),
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    current = current_frames(train)
    frames = window_frames(train.windows, np.arange(len(train)), store.episode_ends)
    for lead in range(1, stride):  # lattices that skip the source, UMI only
        index = int(np.flatnonzero(current == src - lead)[0])
        valid = data.logged_arrays(np.array([index]))[2][0]
        near = np.abs(frames[index] - src) < stride
        assert near.any() and not valid[near].any() and valid[~near].all()
    index = int(np.flatnonzero(current == src - stride)[0])
    valid = data.logged_arrays(np.array([index]))[2][0]
    np.testing.assert_array_equal(valid, frames[index] != src)


@pytest.mark.parametrize("fixture", ["can_store", "sweater_store"])
def test_a_crossing_window_replaces_its_logged_twin(fixture, request):
    """A logged window whose action rows reach a selected success source (a row at the
    source, at or after its current frame) is replaced by that source's crossing window:
    the replaced windows and the crossings are the same current frames, also for a
    source so near the episode end that some of its lead positions have no logged
    window."""
    store = request.getfixturevalue(fixture)
    e = int(np.flatnonzero(store.episode_success)[0])
    train = logged(store, [e])
    shape = train.shape
    begin, end = int(store.episode_starts[e]), int(store.episode_ends[e])
    sources = [begin + 60 * shape.stride, end - 4 * shape.stride]
    rows = [
        record(store, src, src + 2 * shape.stride, candidate=i)
        for i, src in enumerate(sources)
    ]
    data = AugmentedPolicyDataset(
        train,
        pack(rows, store.spec.action_dim),
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=0,
    )
    frames = window_frames(train.windows, np.arange(len(train)), store.episode_ends)
    reaches = np.isin(frames[:, shape.n_obs - 1 :], sources).any(1)
    kept = np.zeros(len(train), dtype=bool)
    kept[data.logged_windows] = True
    np.testing.assert_array_equal(kept, ~reaches)
    np.testing.assert_array_equal(
        np.sort(current_frames(train)[reaches]), np.sort(data.crossings[:, 1])
    )
    leads = shape.horizon - shape.n_obs + 1
    assert (
        leads < len(data.crossings) < 2 * leads
    )  # the late source loses lead positions


def test_weight_one_epoch_is_one_pass(can_store):
    """At bridge weight 1 with nothing skipped (each stage-2 bridge departs
    from its episode's last window into the other episode, whose frames are all target
    suffix) every window but the twins weighs 1: an epoch is one pass over the kept
    logged windows and the crossing windows, plus the twins."""
    s = can_store
    e0, e1 = (int(e) for e in np.flatnonzero(s.episode_success)[:2])
    train = logged(s, [e0, e1])
    last = {
        e: int(
            current_frames(train)[s.episode_of_frame(current_frames(train)) == e].max()
        )
        for e in (e0, e1)
    }
    rows = [
        record(s, last[e0], int(s.episode_starts[e1]), 2, 3, 0),
        record(s, last[e1], int(s.episode_starts[e0]), 2, 3, 1),
    ]
    data = AugmentedPolicyDataset(
        train,
        pack(rows, s.spec.action_dim),
        weight=1.0,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=OPTIONS,
        seed=3,
    )
    assert (data.logged_weight == 1).all() and (data.crossing_weight == 1).all()
    assert len(data.logged_windows) + len(data.crossings) == len(train)
    assert len(data.twins) == 2 and len(data) == len(train) + 2
    for epoch in (0, 4):
        data.set_epoch(epoch)
        assert len(data.draw) == len(train) + 2
        logged_draws = np.sort(data.draw[~data.is_bridge & ~data.is_twin])
        crossing_draws = np.sort(data.draw[data.is_bridge])
        # the two half-weight twins take one draw between them; the pass keeps the rest
        assert data.is_twin.sum() == 1
        assert len(logged_draws) + len(crossing_draws) == len(train) + 1
        assert (
            np.isin(data.logged_windows, logged_draws).sum()
            >= len(data.logged_windows) - 1
        )
        assert np.isin(np.arange(len(data.crossings)), crossing_draws).all()
