"""Acceptance, prefix choice, packing and paring of scored stitch candidates.

Each candidate carries ``proposals`` action samples and one verifier logit per
executable prefix (a head). A head's threshold is the verifier's calibrated threshold
for that prefix length (the ``1 - target_fpr`` quantile of that step's validation
negatives, ``models/verifier.py``) plus the candidate's stage offset
(``threshold_offsets``); a margin is a logit minus it. Prefixes longer than the
candidate's maximum length (its temporal eligibility, ``candidates.max_prefix_lengths``)
never count.

Acceptance. A sample's prefix is accepted when its margin and the mean margin over all
samples at that head both reach zero. A candidate is accepted when at least
``min_votes`` samples have an accepted prefix.

Prefix choice. The shortest accepted prefix, taken from the median sample by margin
among those accepted at it (the lower median of an even count, ties to the lower
sample). The bridge's margin is that sample's margin.

Packing. One bridge per source: stage 1 before stage 2 for a successful source (failure
sources have only stage 3), then the larger margin, the more raw frames saved on the
route, the shorter bridge and the smaller candidate id.

Paring (``keep_fraction``). Per stage, the max(1, ceil(f * n)) packed bridges with the
largest margin are kept, ties to the lower source frame.
"""

from __future__ import annotations

import math

import numpy as np
from omegaconf import DictConfig

from needlework.data.bridges import Bridges
from needlework.stitching.candidates import stage_entry


def threshold_offset(cfg: DictConfig, stage: int) -> float:
    return float(stage_entry(cfg, "threshold_offsets", stage))


def keep_fraction(cfg: DictConfig, stage: int) -> float:
    return float(stage_entry(cfg, "keep_fraction", stage))


def margins(
    logits: np.ndarray, thresholds: np.ndarray, max_length: np.ndarray
) -> np.ndarray:
    """[candidates, samples, H] logits -> margins; ineligible prefixes are -inf."""
    if logits.ndim != 3 or thresholds.shape != (logits.shape[2],):
        raise ValueError("expected [candidates, samples, H] logits and [H] thresholds")
    if max_length.shape != (len(logits),):
        raise ValueError("max_length must have one entry per candidate")
    if not np.isfinite(logits).all() or not np.isfinite(thresholds).all():
        raise ValueError("logits and thresholds must be finite")
    lengths = np.arange(1, logits.shape[2] + 1)
    eligible = lengths[None, None, :] <= max_length[:, None, None]
    return np.where(eligible, logits - thresholds[None, None, :], -np.inf).astype(
        np.float32
    )


def choose_prefixes(
    margin: np.ndarray, *, min_votes: int
) -> list[tuple[int, int, int, float]]:
    """Accepted candidates -> (candidate row, sample, prefix length, margin)."""
    if not 1 <= min_votes <= margin.shape[1]:
        raise ValueError("need 1 <= min_votes <= samples")
    accepted = (margin >= 0) & (margin.mean(1) >= 0)[:, None, :]
    out = []
    for row in np.flatnonzero(accepted.any(2).sum(1) >= min_votes):
        h = int(np.flatnonzero(accepted[row].any(0))[0])
        samples = np.flatnonzero(accepted[row, :, h])
        samples = samples[np.argsort(margin[row, samples, h], kind="stable")]
        s = int(samples[(len(samples) - 1) // 2])
        out.append((int(row), s, h + 1, float(margin[row, s, h])))
    return out


def keep_top_fraction(b: Bridges, fractions: dict[int, float]) -> Bridges:
    """Per stage, keep max(1, ceil(fraction * n)) bridges with the highest verifier
    margin, ties to the lower source frame. Rows stay in source order; only the kept
    actions are stored."""
    for stage in np.unique(b.stage).tolist():
        if stage not in fractions:
            raise ValueError(f"keep_fraction has no entry for stage{stage}")
    for stage, fraction in fractions.items():
        if not 0 < fraction <= 1:
            raise ValueError(
                f"keep_fraction.stage{stage} must be in (0, 1]: {fraction}"
            )
    keep = []
    for stage in np.unique(b.stage).tolist():
        rows = np.flatnonzero(b.stage == stage)
        order = np.lexsort((b.source_frame[rows], -b.verifier_margin[rows]))
        keep.append(rows[order[: max(1, math.ceil(fractions[stage] * len(rows)))]])
    keep = np.sort(np.concatenate(keep)) if keep else np.empty(0, np.int64)
    lengths = b.action_len[keep]
    actions = [b.rows(i) for i in keep]
    return Bridges(
        source_frame=b.source_frame[keep],
        target_frame=b.target_frame[keep],
        stage=b.stage[keep],
        verifier_margin=b.verifier_margin[keep],
        action_start=np.cumsum(lengths) - lengths,
        action_len=lengths,
        actions=np.concatenate(actions) if actions else b.actions[:0],
    )


def pack(records: list[dict], action_dim: int) -> Bridges:
    """One bridge per source, in the order stated in the module docstring."""
    best: dict[int, dict] = {}

    def key(row: dict) -> tuple:
        return (
            row["stage"],
            -row["margin"],
            -row["saved"],
            len(row["actions"]),
            row["candidate_id"],
        )

    for row in records:
        source = row["source"]
        if source not in best or key(row) < key(best[source]):
            best[source] = row
    rows = [best[s] for s in sorted(best)]
    if not rows:
        return Bridges.empty(action_dim)
    lengths = np.array([len(r["actions"]) for r in rows], np.int64)
    return Bridges(
        source_frame=np.array([r["source"] for r in rows], np.int64),
        target_frame=np.array([r["target"] for r in rows], np.int64),
        stage=np.array([r["stage"] for r in rows], np.int64),
        verifier_margin=np.array([r["margin"] for r in rows], np.float32),
        action_start=np.cumsum(lengths) - lengths,
        action_len=lengths,
        actions=np.concatenate([r["actions"] for r in rows]).astype(np.float32),
    )
