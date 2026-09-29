"""Stitch candidate budgets: ``candidate_spacing`` (raw frames between sources and
between the targets of one source), per-stage ``targets_per_source``, and an optional
source-uniform ``max_candidates_per_stage`` cap."""

from collections import Counter

import numpy as np
import pytest
import torch
from omegaconf import open_dict

from needlework import config, stitch
from needlework.data.store import EpisodeStore
from needlework.stitching.candidates import generate, source_uniform_cap


def _cfg(*overrides: str, task: str = "robomimic/square"):
    return config.compose_stitch(
        [
            f"task={task}",
            "run.name=t",
            "idm_checkpoint=unused",
            "verifier_checkpoint=unused",
            *overrides,
        ]
    )


def _generate(cfg) -> np.ndarray:
    store = EpisodeStore.open(cfg.task.domain, cfg.task.name)
    return generate(store, cfg, horizon=23, stride=1, device=torch.device("cuda"))


def _rows(counts: list[int]) -> np.ndarray:
    source = np.repeat(np.arange(len(counts)), counts)
    target = np.arange(len(source)) + 1000
    return np.stack([np.zeros_like(source), source, target, np.ones_like(source)], 1)


def test_cap_is_source_uniform_and_seeded() -> None:
    rows = _rows([1, 5, 10, 50])
    # Rounds take one target per source still holding one: 4 + 3*4 + 2*4 = 24 after
    # round 8, so the 25th slot goes to one of the two sources left in round 9.
    kept = source_uniform_cap(rows, 25, seed=42, stage=1)
    assert len(kept) == 25
    per = Counter(kept[:, 1].tolist())
    assert per[0] == 1 and per[1] == 5  # small sources are used up
    assert sorted([per[2], per[3]]) == [9, 10]  # the rest share the budget evenly
    # A subsequence of the input, in its original order.
    position = {tuple(r): i for i, r in enumerate(rows.tolist())}
    order = [position[tuple(r)] for r in kept.tolist()]
    assert order == sorted(order)
    again = source_uniform_cap(rows, 25, seed=42, stage=1)
    np.testing.assert_array_equal(kept, again)
    other = source_uniform_cap(rows, 25, seed=43, stage=1)
    assert not np.array_equal(kept, other)
    np.testing.assert_array_equal(source_uniform_cap(rows, 100, seed=42, stage=1), rows)


def test_full_budgets_cap_each_stage() -> None:
    # Square's shipped budget with fewer sources (3,000 per stage).
    rows = _generate(_cfg("sources_per_stage=3000"))
    counts = {s: int((rows[:, 3] == s).sum()) for s in (1, 2, 3)}
    assert counts[1] == counts[2] == 100000
    assert counts[3] < 100000  # the cap, then the recovery gate
    assert max(Counter(rows[rows[:, 3] == 3, 1].tolist()).values()) <= 8
    np.testing.assert_array_equal(rows[:, 0], np.arange(len(rows)))
    for stage, limit in ((1, 100), (2, 200)):
        per = Counter(rows[rows[:, 3] == stage, 1].tolist())
        assert max(per.values()) <= limit


def test_per_stage_caps() -> None:
    rows = _generate(
        _cfg(
            "candidate_spacing=1",
            "targets_per_source={stage1: 100, stage2: 200, stage3: 8}",
            "max_candidates_per_stage={stage1: null, stage2: 2800, stage3: null}",
            "sources_per_stage=3000",
        )
    )
    counts = {s: int((rows[:, 3] == s).sum()) for s in (1, 2, 3)}
    assert counts[2] == 2800
    assert counts[1] > 100000  # uncapped: 3,000 sources x up to 100 targets
    assert counts[3] < 100000  # uncapped: up to 8 targets per source, then the gate
    per_source = Counter(rows[rows[:, 3] == 2, 1].tolist())
    assert max(per_source.values()) - min(per_source.values()) <= 1  # source-uniform


def test_shipped_caps() -> None:
    for task in ("robomimic/square", "robomimic/can", "robomimic/transport"):
        caps = _cfg(task=task).max_candidates_per_stage
        stage3 = 1000000 if task == "robomimic/can" else 100000
        assert dict(caps) == {"stage1": 100000, "stage2": 100000, "stage3": stage3}


@pytest.mark.parametrize(
    "override",
    [
        "~targets_per_source.stage2",
        "task.stitch_max_candidates_per_stage.stage2=0",
        "task.stitch_max_candidates_per_stage.stage1=-5",
        "candidate_spacing=0",
    ],
)
def test_bad_budgets_raise(override: str) -> None:
    with pytest.raises(ValueError, match=r"targets_per_source|max_candidates|spacing"):
        stitch.check(_cfg(override))


def test_task_configs_record_the_spacing() -> None:
    assert _cfg().candidate_spacing == 1
    assert _cfg(task="robomimic/can").candidate_spacing == 1
    assert _cfg(task="robomimic/transport").candidate_spacing == 1
    umi = config.compose_stitch(
        [
            "task=umi",
            "task.name=sweater",
            "run.name=t",
            "idm_checkpoint=unused",
            "verifier_checkpoint=unused",
        ]
    )
    assert umi.candidate_spacing == 69


def test_missing_stage_cap_raises() -> None:
    cfg = _cfg()
    with open_dict(cfg):
        del cfg.max_candidates_per_stage["stage3"]
    with pytest.raises(ValueError, match="max_candidates_per_stage"):
        stitch.check(cfg)


def test_recovery_candidates_are_capped_before_the_recovery_gate() -> None:
    """Stage 3 draws targets for every failure frame, caps them source-uniformly, then
    keeps the sources that pass the recovery gate: on Square, 2,237 sources with one or
    two targets each (the cap spreads 100,000 over about 75,000 failure frames)."""
    cfg = _cfg("stages=[3]")
    rows = _generate(cfg)
    per_source = Counter(rows[:, 1].tolist())
    assert len(per_source) == 2237
    assert set(per_source.values()) == {1, 2}
    # 2,237 + about 2,237 x 25,002 / 74,998 second targets; binomial sd about 22
    assert 2900 <= len(rows) <= 3070
    store = EpisodeStore.open("robomimic", "square")
    remaining = store.episode_ends[store.episode_of_frame(rows[:, 2])] - rows[:, 2]
    assert 300 <= np.median(remaining) <= 370  # frame-uniform draw within each shell


def test_can_recovery_cap_spreads_over_every_failure_target() -> None:
    """Can's stage-3 cap of 1,000,000 over about 75,000 failure frames x 16 targets
    leaves 13 or 14 targets per source before the recovery gate, which keeps 243 or
    244 sources."""
    rows = _generate(_cfg("stages=[3]", task="robomimic/can"))
    per_source = Counter(rows[:, 1].tolist())
    # The gate's boundary frame can flip across GPU types (float differences).
    assert len(per_source) in (243, 244)
    assert set(per_source.values()) == {13, 14}
    assert 3240 <= len(rows) <= 3254
