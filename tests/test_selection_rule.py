"""Acceptance at the offset thresholds with the ensemble gate, prefix choice, packing
and paring of the packed bridges by margin (stitching/selection.py)."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

from needlework import config, stitch
from needlework.data.bridges import Bridges
from needlework.stitching import pipeline
from needlework.stitching.selection import (
    choose_prefixes,
    keep_fraction,
    keep_top_fraction,
    margins,
    pack,
    threshold_offset,
)
from needlework.stitching.workers import read_result, write_result

PROPOSALS, HORIZON, VOTES = 8, 5, 2
FAR = np.array([10**6])  # episode ends of one long episode holding every frame


def _scores(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, (n, PROPOSALS, HORIZON)).astype(np.float32)
    thresholds = rng.normal(0.0, 0.5, HORIZON).astype(np.float32)
    max_length = rng.integers(0, HORIZON + 1, n)
    return logits, thresholds, max_length


def _margin(entries: dict[tuple[int, int], float], rest: float = -1.0) -> np.ndarray:
    """One candidate: every (sample, head) margin is ``rest`` except ``entries``."""
    m = np.full((1, PROPOSALS, HORIZON), rest, np.float32)
    for (sample, head), value in entries.items():
        m[0, sample, head] = value
    return m


def test_a_proposal_needs_the_ensemble_mean_to_clear_its_head() -> None:
    # samples 0 and 1 clear head 0, but the mean over the 8 samples does not
    assert choose_prefixes(_margin({(0, 0): 0.5, (1, 0): 0.5}), min_votes=VOTES) == []
    # with the other samples just below zero, the mean clears it too
    m = _margin({(0, 0): 0.5, (1, 0): 0.5}, rest=-0.1)
    assert [row for row, *_ in choose_prefixes(m, min_votes=VOTES)] == [0]


def test_a_candidate_needs_min_votes_distinct_samples() -> None:
    m = _margin({}, rest=0.0)  # every sample exactly at its thresholds
    assert len(choose_prefixes(m, min_votes=PROPOSALS)) == 1
    m[0, 1:, :], m[0, 0, :] = -0.01, 3.0
    assert choose_prefixes(m, min_votes=VOTES) == []  # one sample, however large


def test_prefix_is_the_shortest_accepted_head_and_the_median_sample() -> None:
    # head 0 accepts nothing; at head 1 samples 2, 5 and 6 accept with 0.9, 0.2, 0.5
    m = _margin({(2, 1): 0.9, (5, 1): 0.2, (6, 1): 0.5}, rest=0.0)
    m[0, :, 0] = -1.0
    for s in (0, 1, 3, 4, 7):
        m[0, s, 1] = -0.01
    assert choose_prefixes(m, min_votes=VOTES) == [(0, 6, 2, pytest.approx(0.5))]
    # an even count takes the lower median; equal margins go to the lower sample
    m[0, 6, 1] = -0.01
    assert choose_prefixes(m, min_votes=VOTES) == [(0, 5, 2, pytest.approx(0.2))]
    m[0, 2, 1] = 0.2
    assert choose_prefixes(m, min_votes=VOTES) == [(0, 2, 2, pytest.approx(0.2))]


def test_ineligible_prefixes_are_never_accepted() -> None:
    logits, thresholds, max_length = _scores(300)
    assert (
        choose_prefixes(
            margins(logits, thresholds, max_length - 10) + 100.0, min_votes=VOTES
        )
        == []
    )
    chosen = choose_prefixes(
        margins(logits, thresholds, max_length) + 100.0, min_votes=VOTES
    )
    assert {row for row, *_ in chosen} == set(np.flatnonzero(max_length > 0).tolist())
    assert all(length == 1 and np.isfinite(margin) for _, _, length, margin in chosen)


def test_batch_files_hold_every_proposal(tmp_path) -> None:
    rows = np.array([[0, 1, 20, 1], [1, 3, 30, 3]])
    rng = np.random.default_rng(3)
    scores = {
        "logits": rng.normal(size=(2, PROPOSALS, 23)).astype(np.float32),
        "actions": rng.normal(size=(2, PROPOSALS, 23, 10)).astype(np.float32),
        "max_length": np.array([12, 23]),
    }
    path = tmp_path / "batch.npz"
    write_result(path, rows, scores)
    back = read_result(path, rows)
    for k, value in scores.items():
        np.testing.assert_array_equal(back[k], value)
    with pytest.raises(ValueError, match="different candidates"):
        read_result(path, rows + 1)
    bad = {**scores, "logits": scores["logits"][:1]}
    write_result(path, rows, bad)
    with pytest.raises(ValueError, match="malformed"):
        read_result(path, rows)


def _table() -> Bridges:
    """Packed layout: rows sorted by source frame, actions contiguous in row order."""
    stage = np.array([1, 2, 2, 2, 2, 2, 3, 3, 3, 2])
    margin = np.array([0.1, 3.0, 1.0, 2.0, 2.0, 0.5, 4.0, 1.0, 1.0, 2.0], np.float32)
    lengths = np.array([3, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n = len(stage)
    return Bridges(
        source_frame=np.arange(n, dtype=np.int64) * 10,
        target_frame=np.arange(n, dtype=np.int64) * 10 + 500,
        stage=stage.astype(np.int64),
        verifier_margin=margin,
        action_start=(np.cumsum(lengths) - lengths).astype(np.int64),
        action_len=lengths.astype(np.int64),
        actions=np.arange(lengths.sum() * 2, dtype=np.float32).reshape(-1, 2),
    )


def test_keep_top_fraction_keeps_the_ceiling_ties_to_lower_source() -> None:
    table = _table()  # 1 / 6 / 3 bridges by stage
    # stage 2: ceil(0.3 x 6) = 2, margin 3.0 and the lowest source among the 2.0s;
    # stages 1 and 3 keep at least one
    kept = keep_top_fraction(table, {1: 0.01, 2: 0.3, 3: 0.01})
    np.testing.assert_array_equal(kept.source_frame, [0, 10, 30, 60])
    for i, source in enumerate(kept.source_frame):
        j = int(np.flatnonzero(table.source_frame == source)[0])
        np.testing.assert_array_equal(kept.rows(i), table.rows(j))
    assert len(kept.actions) == kept.action_len.sum()
    assert keep_top_fraction(table, {1: 1.0, 2: 1.0, 3: 1.0}).digest() == table.digest()
    with pytest.raises(ValueError, match="keep_fraction"):
        keep_top_fraction(table, {1: 1.0, 2: 1.5, 3: 1.0})


def test_pack_prefers_stage_then_margin_then_steps_saved_then_length() -> None:
    def row(i: int, stage: int, margin: float, saved: int, n: int) -> dict:
        return {
            "candidate_id": i,
            "source": 1,
            "target": 20 + i,
            "stage": stage,
            "margin": margin,
            "saved": saved,
            "actions": np.zeros((n, 10), np.float32),
        }

    rows = [row(0, 2, 9.0, 99, 1), row(1, 1, 1.0, 5, 9), row(2, 1, 2.0, 5, 9)]
    assert pack(rows, 10).target_frame.tolist() == [22]  # stage 1, then larger margin
    rows = [row(0, 1, 2.0, 5, 3), row(1, 1, 2.0, 7, 9), row(2, 1, 2.0, 7, 4)]
    assert pack(rows, 10).target_frame.tolist() == [22]  # more saved, then shorter
    assert pack(rows[::-1], 10).digest() == pack(rows, 10).digest()


def _cfg(task: str = "robomimic/square", *overrides: str):
    extra = ["task.name=sweater"] if task == "umi" else []
    return config.compose_stitch(
        [
            f"task={task}",
            *extra,
            "run.name=t",
            "idm_checkpoint=unused",
            "verifier_checkpoint=unused",
            *overrides,
        ]
    )


SHIPPED = {  # task: (threshold offsets, keep fractions) by stage
    "robomimic/square": ({1: 0.0, 2: 0.0, 3: -6.0}, {1: 1.0, 2: 0.05, 3: 0.4}),
    "robomimic/can": ({1: 0.0, 2: 0.0, 3: -1.5}, {1: 0.02, 2: 0.06, 3: 0.4}),
    "robomimic/transport": ({1: 0.0, 2: -6.0, 3: -8.0}, {1: 1.0, 2: 0.1, 3: 1.0}),
    "umi": ({1: 0.0, 2: 0.0, 3: 0.0}, {1: 1.0, 2: 1.0, 3: 1.0}),
}


def test_shipped_values() -> None:
    for task, (offsets, keep) in SHIPPED.items():
        cfg = _cfg(task)
        assert {s: threshold_offset(cfg, s) for s in (1, 2, 3)} == offsets
        assert {s: keep_fraction(cfg, s) for s in (1, 2, 3)} == keep


def test_can_overrides_merge_with_the_robomimic_base() -> None:
    """Can sets only its stage-3 targets, its stage-3 cap and its IDM epochs; every
    other entry of those blocks comes from robomimic/_base.yaml."""
    cfg = _cfg("robomimic/can")
    assert dict(cfg.targets_per_source) == {"stage1": 100, "stage2": 200, "stage3": 16}
    assert dict(cfg.max_candidates_per_stage) == {
        "stage1": 100000,
        "stage2": 100000,
        "stage3": 1000000,
    }
    assert dict(cfg.task.epochs) == {"policy": 300, "idm": 300, "verifier": 40}


@pytest.mark.parametrize(
    "override",
    [
        "task.stitch_threshold_offsets.stage2=.nan",
        "task.stitch_threshold_offsets.stage3=x",
        "task.stitch_keep_fraction.stage1=-1",
        "task.stitch_keep_fraction.stage2=0",
    ],
)
def test_check_rejects_bad_values(override: str) -> None:
    with pytest.raises(ValueError, match=r"threshold_offsets|keep_fraction|float"):
        stitch.check(_cfg("robomimic/square", override))


def test_check_requires_every_stage() -> None:
    cfg = _cfg()
    with open_dict(cfg):
        del cfg.threshold_offsets["stage3"]
    with pytest.raises(ValueError, match="threshold_offsets"):
        stitch.check(cfg)


def _select_cfg(offsets: dict, keep: dict) -> OmegaConf:
    return OmegaConf.create(
        {
            "batch_size": 4,
            "min_votes": VOTES,
            "stages": sorted(offsets),
            "threshold_offsets": {f"stage{s}": v for s, v in offsets.items()},
            "keep_fraction": {f"stage{s}": v for s, v in keep.items()},
            "task": {"stride": 1},
        }
    )


def test_selection_applies_the_rule_at_each_stages_offset(
    tmp_path, monkeypatch
) -> None:
    """pipeline.select accepts exactly what choose_prefixes accepts at each candidate's
    stage offset, and keeps copies of the chosen prefixes, never batch views."""
    n = 40
    stage = np.where(np.arange(n) < 20, 1, 2)
    rows = np.stack([np.arange(n), np.arange(n) * 3, np.arange(n) * 3 + 90, stage], 1)
    logits, thresholds, max_length = _scores(n, seed=5)
    paths = []
    for i in range(n // 4):
        span = slice(i * 4, (i + 1) * 4)
        paths.append(tmp_path / f"{i:06d}.npz")
        write_result(
            paths[-1],
            rows[span],
            {
                "logits": logits[span],
                "actions": np.zeros((4, PROPOSALS, HORIZON, 10), np.float32),
                "max_length": max_length[span],
            },
        )
    offsets = {1: 0.5, 2: -1.0}
    offset = np.array([offsets[s] for s in stage], np.float32)[:, None, None]
    expected = {
        r
        for r, *_ in choose_prefixes(
            margins(logits, thresholds, max_length) - offset, min_votes=VOTES
        )
    }
    seen = []

    def recording_pack(records: list[dict], action_dim: int) -> Bridges:
        seen.extend(records)
        return pack(records, action_dim)

    monkeypatch.setattr(pipeline, "pack", recording_pack)
    _, summary = pipeline.select(
        _select_cfg(offsets, {1: 1.0, 2: 1.0}), rows, paths, thresholds, 10, FAR
    )
    assert expected and {r["candidate_id"] for r in seen} == expected
    assert all(r["actions"].base is None for r in seen)
    assert summary["threshold_offsets"] == {"1": 0.5, "2": -1.0}
    assert summary["accepted_by_stage"] == {
        str(s): sum(int(stage[r] == s) for r in expected) for s in (1, 2)
    }
    json.dumps(summary, allow_nan=False)


def test_selection_without_candidates_is_empty() -> None:
    cfg = _select_cfg({1: 0.0, 2: 0.0, 3: 0.0}, {1: 1.0, 2: 0.5, 3: 0.5})
    thresholds = np.zeros(HORIZON, np.float32)
    table, summary = pipeline.select(
        cfg, np.empty((0, 4), np.int64), [], thresholds, 10, FAR
    )
    assert len(table) == 0 and table.actions.shape == (0, 10)
    assert summary["candidates"] == 0 and summary["selected_bridges"] == 0
    zero = {"1": 0, "2": 0, "3": 0}
    assert summary["accepted_by_stage"] == summary["packed_by_stage"] == zero
    assert summary["by_stage"] == zero
    json.dumps(summary, allow_nan=False)
