"""Sampler options (``sampler.options``), each one behaviour:

path_start  aligned | repeat_current
    repeat_current: at a training episode's first frame and the frame after a selected
    success source, the history repeats the current frame and the action chunk starts
    at the current frame (one step later than aligned)
twin_rows   own_source | all
    all: a departure's twin also supervises its logged rows at other selected sources
draw        even_passes | with_replacement
    with_replacement: each epoch is drawn independently, with probability by weight
"""

import numpy as np
import pytest
from test_stitches import logged, record

from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    PATH_STARTS,
    AugmentedPolicyDataset,
    SamplerOptions,
)
from needlework.sampling.pairs import current_frames
from needlework.stitching.selection import pack

ROLE_WEIGHTS = roles.RoleWeights(
    skipped=0.5, twin=0.5, failure_departure=0.5, approach=1.0
)
BASE = {
    "path_start": "aligned",
    "twin_rows": "own_source",
    "draw": "even_passes",
    "source_rows": "masked",
    "after_departure": "all",
}


def _data(train, b, **options):
    return AugmentedPolicyDataset(
        train,
        b,
        weight=0.2,
        role_weights=ROLE_WEIGHTS,
        epoch_length="windows",
        options=SamplerOptions(**{**BASE, **options}),
        seed=0,
    )


@pytest.mark.parametrize(
    "key", ["path_start", "twin_rows", "draw", "source_rows", "after_departure"]
)
def test_unknown_values_raise(key) -> None:
    with pytest.raises(ValueError, match=key):
        SamplerOptions(**{**BASE, key: "nope"})


def test_repeat_current_path_start_shifts_one_step(can_store):
    s = can_store
    e = int(np.flatnonzero(s.episode_success)[0])
    start, end = int(s.episode_starts[e]), int(s.episode_ends[e])
    src = start + 40
    b = pack([record(s, src, src + 30, 1)], s.spec.action_dim)
    train = logged(s, [e])
    data = _data(train, b, path_start="repeat_current")
    assert data.path_starts.tolist() == [start, src + 1]
    current = current_frames(train)
    action = s.arrays["action"]
    for f in (start, src + 1):  # logged path-start windows
        k = int(np.flatnonzero(current == f)[0])
        obs, act, valid = data.logged_arrays(np.array([k]))
        np.testing.assert_array_equal(obs["frame"][0, :, 0], [f, f])
        np.testing.assert_array_equal(
            act[0], action[np.minimum(f + np.arange(24), end - 1)]
        )
        assert valid[0].all()
    k = int(np.flatnonzero(current == src + 2)[0])  # not a path start: unchanged
    obs, act, _ = data.logged_arrays(np.array([k]))
    np.testing.assert_array_equal(obs["frame"][0, :, 0], [src + 1, src + 2])
    np.testing.assert_array_equal(act[0], action[src + 1 + np.arange(24)])
    # a lead crossing whose current frame is a path start carries the bridge from its
    # current row
    b2 = pack([record(s, start + 1, start + 30, 1, 3)], s.spec.action_dim)
    data2 = _data(train, b2, path_start="repeat_current")
    c = int(np.flatnonzero(data2.crossings[:, 1] == start)[0])
    obs, act, valid = data2.bridge_arrays(np.array([c]))
    np.testing.assert_array_equal(obs["frame"][0, :, 0], [start, start])
    np.testing.assert_array_equal(act[0, 0], action[start])
    np.testing.assert_allclose(act[0, 1:4], b2.rows(0), atol=1e-6)
    assert valid[0, :4].all() and not valid[0, 4:].any()
    # aligned: no path starts, every window as logged
    assert _data(train, b2).path_starts.size == 0


@pytest.mark.parametrize("stride, relative", [(2, False), (1, True)])
def test_repeat_current_refuses_what_it_does_not_define(stride, relative) -> None:
    """Defined for stride-1 absolute actions only."""
    from types import SimpleNamespace

    data = SimpleNamespace(
        shape=SimpleNamespace(stride=stride), relative_actions=relative
    )
    with pytest.raises(ValueError, match="path_start"):
        PATH_STARTS["repeat_current"](data, None, np.empty(0, dtype=np.int64))


def test_all_twin_rows_supervise_other_sources(can_store):
    s = can_store
    e = int(np.flatnonzero(s.episode_success)[0])
    start = int(s.episode_starts[e])
    a, later = (
        start + 40,
        start + 50,
    )  # the second source lies inside the first twin's horizon
    b = pack(
        [record(s, a, a + 5, 1), record(s, later, later + 5, 1, candidate=1)],
        s.spec.action_dim,
    )
    train = logged(s, [e])
    current = current_frames(train)
    twin = int(np.flatnonzero(current == a)[0])
    row = 1 + (later - a)  # n_obs - 1 + offset
    own = _data(train, b)
    every = _data(train, b, twin_rows="all")
    assert twin in own.twins and twin in every.twins
    assert not own.twin_arrays(np.array([twin]))[2][0, row]
    _, _, valid = every.twin_arrays(np.array([twin]))
    assert valid[0, row] and valid[0].all()
    # logged (non-twin) windows keep their mask either way
    np.testing.assert_array_equal(
        every.logged_arrays(np.array([twin]))[2],
        own.logged_arrays(np.array([twin]))[2],
    )


def test_with_replacement_draws_follow_weights(can_store):
    s = can_store
    e0, e1 = (int(e) for e in np.flatnonzero(s.episode_success)[:2])
    start = int(s.episode_starts[e0])
    b = pack([record(s, start + 40, start + 60, 1)], s.spec.action_dim)
    data = _data(logged(s, [e0, e1]), b, draw="with_replacement")
    n_logged, n_twin = len(data.logged_windows), len(data.twins)
    weights = np.concatenate(
        [data.logged_weight, data.twin_weight, data.crossing_weight]
    )
    counts = np.zeros(len(weights))
    epochs = 400
    for epoch in range(epochs):
        data.set_epoch(epoch)
        assert len(data.draw) == len(data)
        kind = np.where(data.is_bridge, 2, np.where(data.is_twin, 1, 0))
        logged_pos = np.searchsorted(data.logged_windows, data.draw[kind == 0])
        twin_pos = n_logged + np.searchsorted(data.twins, data.draw[kind == 1])
        crossing_pos = n_logged + n_twin + data.draw[kind == 2]
        np.add.at(counts, np.concatenate([logged_pos, twin_pos, crossing_pos]), 1)
    expected = weights / weights.sum() * len(data) * epochs
    positive = weights > 0
    assert np.abs(counts[positive] / expected[positive] - 1).max() < 0.25
    assert counts[~positive].sum() == 0
    data.set_epoch(3)
    first = data.draw.copy()
    data.set_epoch(3)
    np.testing.assert_array_equal(first, data.draw)  # reproducible from (seed, epoch)


def test_supervised_source_rows_keep_logged_rows_at_sources(can_store):
    """source_rows=supervised: logged rows at a selected source, and a failure
    episode's logged history row in its departure window, stay supervised; masked
    (the default elsewhere) drops them. Rows after a bridge stay masked either way."""
    s = can_store
    e = int(np.flatnonzero(s.episode_success)[0])
    f = int(np.flatnonzero(~s.episode_success)[0])
    a0, fs = int(s.episode_starts[e]), int(s.episode_starts[f]) + 20
    b = pack(
        [
            record(s, a0 + 40, a0 + 60, 1),
            record(s, a0 + 41, a0 + 70, 1, candidate=1),
            record(s, fs, a0 + 80, 3, candidate=2),
        ],
        s.spec.action_dim,
    )
    train = logged(s, [e])
    for rows, expected in (("masked", False), ("supervised", True)):
        data = _data(train, b, source_rows=rows)
        for source in (a0 + 41, fs):  # history row = the other source / a failure frame
            c = int(
                np.flatnonzero(
                    (data.crossings[:, 1] == source)
                    & (data.bridges.source_frame[data.crossings[:, 2]] == source)
                )[0]
            )
            valid = data.bridge_arrays(np.array([c]))[2][0]
            assert bool(valid[0]) is expected, (rows, source)
            assert not valid[-1]  # after the 3-row bridge: padding, never supervised


def test_after_departure_skipped_keeps_only_skipped_frames(can_store):
    """after_departure=skipped: after an episode's first departure, a logged window
    stays only where a success bridge skips its frame; before it every window stays.
    all keeps every logged window."""
    s = can_store
    e = int(np.flatnonzero(s.episode_success)[0])
    a0 = int(s.episode_starts[e])
    b = pack([record(s, a0 + 40, a0 + 60, 1)], s.spec.action_dim)
    train = logged(s, [e])
    current = current_frames(train)
    kept = {
        rows: set(
            current[_data(train, b, after_departure=rows).logged_windows].tolist()
        )
        for rows in ("all", "skipped")
    }
    assert kept["skipped"] < kept["all"]
    dropped = kept["all"] - kept["skipped"]
    assert min(dropped) >= a0 + 60  # at and after the target: no bridge skips them
    assert all(f < a0 + 40 or a0 + 40 < f < a0 + 60 for f in kept["skipped"])


def test_repeat_current_window_reaching_a_source_carries_its_bridge(can_store):
    """A path-start window's shifted action rows can end on a success source that its
    unshifted rows did not reach; that row carries the bridge's first action, as the
    route does there."""
    s = can_store
    e = int(np.flatnonzero(s.episode_success)[0])
    a0 = int(s.episode_starts[e])
    bridge = record(s, a0 + 23, a0 + 60, 1)
    bridge["actions"] = bridge["actions"] + 0.5  # distinguishable from the logged rows
    b = pack([bridge], s.spec.action_dim)
    train = logged(s, [e])
    data = _data(train, b, path_start="repeat_current", source_rows="supervised")
    k = int(np.flatnonzero(current_frames(train) == a0)[0])
    assert k in data.logged_windows  # not a crossing: unshifted rows stop short
    _, action, valid = data.logged_arrays(np.array([k]))
    np.testing.assert_allclose(action[0, 23], b.rows(0)[0], atol=1e-6)
    assert valid[0, 23]
    np.testing.assert_array_equal(action[0, 22], s.arrays["action"][a0 + 22])
