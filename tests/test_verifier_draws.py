"""Verifier negative goals, per ``sampling.negative_draw``:
- ``epoch_wide``: a source drawn as a negative source in many groups walks through its
  table (an epoch-wide running count per family and source);
- ``within_group``: the count restarts in every group, so such a
  source reuses its first candidates."""

from collections import Counter

import numpy as np
import pytest
import torch

from needlework import config
from needlework.constants import PATCHES, SPATIAL
from needlework.data import features
from needlework.sampling import proximity
from needlework.sampling.proximity import _round_robin
from needlework.sampling.verifier_pairs import NEGATIVES, VerifierPairs, _seed
from needlework.training.common import Windows


def _build(negative_draw: str) -> VerifierPairs:
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
        {**dict(sampling), "negative_draw": negative_draw},
        tau=tau,
        seed=sampling.seed.train,
        resample=True,
        device=device,
    )


@pytest.fixture(scope="module")
def pairs() -> VerifierPairs:
    return _build("epoch_wide")


def _negatives(p: VerifierPairs) -> list[tuple[str, int, int]]:
    episode_of = p.data.store.episode_of_frame
    out = []
    for row, goal, offset in zip(p.row, p.goal, p.offset, strict=True):
        if offset > 0:
            continue
        same = (
            episode_of(np.array([p.current[row]]))[0] == episode_of(np.array([goal]))[0]
        )
        out.append((NEGATIVES[0] if same else NEGATIVES[1], int(row), int(goal)))
    return out


def _table_size(p: VerifierPairs, family: str, row: int) -> int:
    table = p.tables[family]
    return int(
        table.offsets[(row + 1) * table.n_shells] - table.offsets[row * table.n_shells]
    )


def test_negative_goals_do_not_repeat_within_an_epoch(pairs) -> None:
    pairs.set_epoch(0)
    negatives = _negatives(pairs)
    draws = Counter((family, row) for family, row, _ in negatives)
    goals = Counter(negatives)
    repeated = [
        key
        for key, count in goals.items()
        if count > 1 and draws[key[:2]] <= _table_size(pairs, *key[:2])
    ]
    assert not repeated, f"{len(repeated)} negative (source, goal) pairs repeat"
    assert max(draws.values()) > 1  # sources are drawn in several groups


def _replayed_goals(p: VerifierPairs, epoch: int, epoch_wide: bool) -> np.ndarray:
    """set_epoch's negative goals, replayed with either draw-index rule."""
    goals, counts, orders = [], Counter(), {}
    for anchor in p.groups.tolist():
        within = Counter()
        for slot in range(1, 1 + p.n_negatives):
            rng = np.random.default_rng(_seed(p.seed, epoch, anchor, slot))
            t = int(rng.choice(len(NEGATIVES), p=p.negative_p))
            sources = p.sources[NEGATIVES[t]]
            source = int(sources[rng.integers(0, len(sources))])
            table = p.tables[NEGATIVES[t]]
            if (t, source) not in orders:
                order_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [p.seed + epoch, int(table.sources[source]), 0]
                    )
                )
                queues = {
                    s: order_rng.permutation(table.shell(source, s)).tolist()
                    for s in range(table.n_shells)
                    if len(table.shell(source, s))
                }
                shells = order_rng.permutation(np.asarray(list(queues), dtype=np.int64))
                orders[(t, source)] = _round_robin([queues[s] for s in shells.tolist()])
            order = orders[(t, source)]
            index = counts[(t, source)] if epoch_wide else within[t]
            goal = order[index % len(order)] if order else p._fallback(t, source, rng)
            goals.append(goal)
            within[t] += 1
            counts[(t, source)] += 1
    return np.asarray(goals)


def test_within_group_repeats_goals_on_this_fixture(pairs) -> None:
    """The replay reproduces the sampler; the within-group count, replayed on the same
    slots, repeats goals the epoch-wide count does not."""
    pairs.set_epoch(0)
    sampled = pairs.goal.reshape(len(pairs.groups), -1)[:, 1:].ravel()
    np.testing.assert_array_equal(_replayed_goals(pairs, 0, epoch_wide=True), sampled)
    within = _replayed_goals(pairs, 0, epoch_wide=False)
    rows = pairs.row.reshape(len(pairs.groups), -1)[:, 1:].ravel()

    def repeats(goals: np.ndarray) -> int:
        return sum(
            c - 1
            for c in Counter(zip(rows.tolist(), goals.tolist(), strict=False)).values()
        )

    assert repeats(within) > repeats(sampled)


def test_within_group_draw_matches_its_replay(pairs) -> None:
    within = _build("within_group")
    within.set_epoch(0)
    sampled = within.goal.reshape(len(within.groups), -1)[:, 1:].ravel()
    np.testing.assert_array_equal(_replayed_goals(pairs, 0, epoch_wide=False), sampled)
    np.testing.assert_array_equal(within.row, pairs.row)  # only the goals differ


def test_unknown_draw_rule_raises() -> None:
    with pytest.raises(ValueError, match="negative_draw"):
        VerifierPairs.check_negative_draw("per_source")


def test_boundary_band_is_boundary_horizons_wide(pairs) -> None:
    """With probability ``boundary_prob`` a beyond-horizon fallback goal lies within
    ``boundary_horizons`` proposal horizons of the source; the default is two."""
    cfg = config.compose(["component=verifier", "task=robomimic/square", "run.name=t"])
    assert cfg.component.sampling.boundary_horizons == 2
    h, t = pairs.horizon, NEGATIVES.index("beyond_hor_neg")
    sources = np.flatnonzero(pairs.max_future > 4 * h)[:50]
    prob, width = pairs.boundary_prob, pairs.boundary_horizons
    try:
        pairs.boundary_prob = 1.0
        for width_now in (2, 3):
            pairs.boundary_horizons = width_now
            rng = np.random.default_rng(0)
            steps = [
                (pairs._fallback(t, int(s), rng) - int(pairs.current[s]))
                // pairs.stride
                for s in sources
                for _ in range(20)
            ]
            assert min(steps) > h and max(steps) <= width_now * h
            assert max(steps) > (width_now - 1) * h  # the whole band is reachable
    finally:
        pairs.boundary_prob, pairs.boundary_horizons = prob, width


def test_shipped_draw_rule_is_within_group() -> None:
    for task, extra in (("robomimic/square", []), ("umi", ["task.name=sweater"])):
        overrides = ["component=verifier", f"task={task}", *extra, "run.name=t"]
        cfg = config.compose(overrides)
        assert cfg.component.sampling.negative_draw == "within_group"
