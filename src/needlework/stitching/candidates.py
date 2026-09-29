"""Source-balanced visual candidates for the three bridge stages.

Each row is [candidate_id, source_frame, target_frame, stage]. Eligibility is checked
before diffusion inference. Raw frame units are converted using the policy stride.
Budgets: ``candidate_spacing`` raw frames between selected sources (per episode) and
between one source's targets, ``targets_per_source`` per stage, and an optional
source-uniform cap per stage, ``max_candidates_per_stage``.

Stage 3 draws its sources from the failure frames (subject to ``sources_per_stage`` and
``candidate_spacing``), caps the candidates, and then keeps the sources that pass the
recovery gate, so a capped budget spreads over the drawn failure frames before the gate.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from needlework.constants import (
    SIMILARITY_BATCH,
    SPATIAL,
    STAGE_CROSS,
    STAGE_RECOVERY,
    STAGE_WITHIN,
)
from needlework.data import features
from needlework.data.store import EpisodeStore
from needlework.sampling import proximity
from needlework.stitching.recovery import recovery_sources


def min_savings_frames(*, stride: int, min_steps_saved: int) -> int:
    """Raw frames a stage-1/2 route must save: ``min_steps_saved`` policy steps, plus
    one more so that at least one prefix length stays eligible."""
    return stride * (min_steps_saved + 1)


def max_prefix_lengths(
    store: EpisodeStore,
    rows: np.ndarray,
    *,
    horizon: int,
    stride: int,
    min_steps_saved: int,
) -> np.ndarray:
    source, target, stage = rows[:, 1], rows[:, 2], rows[:, 3]
    se, te = store.episode_of_frame(source), store.episode_of_frame(target)
    savings = (store.episode_ends[se] - source) - (store.episode_ends[te] - target)
    # min_steps_saved counts policy steps (stride raw frames each), as the IDM does.
    maximum = np.minimum(horizon, (savings - stride * min_steps_saved) // stride)
    return np.where(stage == STAGE_RECOVERY, horizon, maximum).astype(np.int64)


def source_uniform_cap(
    rows: np.ndarray, cap: int, *, seed: int, stage: int
) -> np.ndarray:
    """At most ``cap`` rows of one stage, drawn source-uniformly: in
    rounds, every source that still has a target gives one, in a random order, until the
    cap. Seeded by (seed, stage, cap); kept rows stay in their original order."""
    if len(rows) <= cap:
        return rows
    rng = np.random.default_rng(np.random.SeedSequence([seed, stage, cap]))
    tiebreak = rng.random(len(rows))
    source = rows[:, 1]
    # Rows sorted by source, and within a source in random order.
    by_source = np.lexsort((tiebreak, source))
    group_start = np.flatnonzero(
        np.r_[True, source[by_source][1:] != source[by_source][:-1]]
    )
    group_size = np.diff(np.r_[group_start, len(rows)])
    # rank_in_source[r]: how many of r's source's rows come before it. Round 0 takes
    # every source's first row, round 1 its second, and so on, so truncating at ``cap``
    # after sorting by rank spreads the cap over the sources evenly.
    rank_in_source = np.empty(len(rows), np.int64)
    rank_in_source[by_source] = np.arange(len(rows)) - np.repeat(
        group_start, group_size
    )
    kept = np.lexsort((tiebreak, rank_in_source))[:cap]
    return rows[np.sort(kept)]


def _stage_targets(
    stage: int,
    sources: np.ndarray,
    emb: torch.Tensor,
    frames: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    tau: float,
    edges: np.ndarray,
    reach: int,
    stride: int,
    max_targets: int,
) -> list[tuple[int, int]]:
    """(source, target) pairs of one stage: each source's eligible targets, visually
    within ``tau``, drawn across the shells (episode-balanced within a shell at stages
    1 and 2, uniform over the shell's frames at stage 3) and ``candidate_spacing``
    frames apart."""
    episode, local, success, remaining = (
        frames[k] for k in ("episode", "local", "success", "remaining")
    )
    frame = np.arange(len(episode))
    draw = proximity.select_shells
    if stage == STAGE_RECOVERY:
        draw = proximity.uniform_shells
    pairs = []
    for start in tqdm(
        range(0, len(sources), SIMILARITY_BATCH), desc=f"stage {stage} candidates"
    ):
        src = sources[start : start + SIMILARITY_BATCH]
        sims = (emb[torch.as_tensor(src, device=emb.device)] @ emb.T).cpu().numpy()
        for i, source in enumerate(src):
            valid = success.copy()
            if stage == STAGE_WITHIN:
                valid &= (episode == episode[source]) & (frame - source > reach)
            elif stage == STAGE_CROSS:
                valid &= episode != episode[source]
            if stage != STAGE_RECOVERY:
                valid &= remaining[source] - remaining >= min_savings_frames(
                    stride=stride, min_steps_saved=cfg.min_steps_saved
                )
            valid &= sims[i] >= tau
            targets = np.flatnonzero(valid)
            if not len(targets):
                continue
            shells = draw(
                targets,
                sims[i, targets],
                (episode, local),
                edges=edges,
                spacing=cfg.candidate_spacing,
                max_targets=max_targets,
                seed=cfg.seed,
                source=int(source),
            )
            pairs += [(int(source), int(t)) for t in np.sort(np.concatenate(shells))]
    return pairs


def generate(
    store: EpisodeStore,
    cfg: DictConfig,
    *,
    horizon: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    spatial = features.load(store, SPATIAL)
    emb = proximity.embeddings(spatial, device)
    del spatial
    frame = np.arange(store.n_frames)
    episode = store.episode_of_frame(frame)
    frames = {
        "episode": episode,
        "local": frame - store.episode_starts[episode],
        "success": store.episode_success[episode],
        "remaining": store.episode_ends[episode] - frame,
    }
    reach = horizon * stride  # raw frames one proposal spans
    tau = proximity.calibrate_tau(
        emb,
        store.episode_starts,
        store.episode_ends,
        reach,
        percentile=cfg.proximity.tau_percentile,
    )
    edges = proximity.shell_edges(tau, cfg.proximity.shells)
    recovery = (
        recovery_sources(
            emb, episode, store.episode_success, quantile=cfg.recovery_quantile
        )
        if STAGE_RECOVERY in cfg.stages
        else None
    )
    all_rows = []
    for stage in cfg.stages:
        success, remaining = frames["success"], frames["remaining"]
        pool = np.flatnonzero(~success if stage == STAGE_RECOVERY else success)
        if stage != STAGE_RECOVERY:
            savings = min_savings_frames(
                stride=stride, min_steps_saved=cfg.min_steps_saved
            )
            # Every target has at least one remaining frame, so a source that can save
            # `savings` frames has more than `savings` remaining.
            pool = pool[remaining[pool] > savings]
        if not len(pool):
            continue
        sources = proximity.balanced_selection(
            pool,
            episode[pool],
            frames["local"][pool],
            max_rows=cfg.sources_per_stage,
            spacing=cfg.candidate_spacing,
            seed=cfg.seed,
            source=stage,
        )
        pairs = _stage_targets(
            stage,
            sources,
            emb,
            frames,
            cfg,
            tau=tau,
            edges=edges,
            reach=reach,
            stride=stride,
            max_targets=targets_per_source(cfg, stage),
        )
        rows = np.array([(0, s, t, stage) for s, t in pairs], np.int64).reshape(-1, 4)
        cap = max_candidates(cfg, stage)
        if cap is not None:
            rows = source_uniform_cap(rows, cap, seed=cfg.seed, stage=stage)
        if stage == STAGE_RECOVERY:
            rows = rows[recovery[rows[:, 1]]]
        all_rows.append(rows)
    rows = np.concatenate(all_rows) if all_rows else np.empty((0, 4), np.int64)
    rows[:, 0] = np.arange(len(rows))
    print(f"candidate count={len(rows)}, proximity tau={tau:.6g}", flush=True)
    return rows


def stage_entry(cfg: DictConfig, name: str, stage: int) -> int | float | None:
    """The ``stage<N>`` value of a per-stage setting; a missing stage is an error."""
    key = f"stage{stage}"
    if key not in cfg[name]:
        raise ValueError(f"{name} has no entry for {key}")
    return cfg[name][key]


def targets_per_source(cfg: DictConfig, stage: int) -> int:
    return int(stage_entry(cfg, "targets_per_source", stage))


def max_candidates(cfg: DictConfig, stage: int) -> int | None:
    """The stage's source-uniform candidate cap, or None for no cap."""
    cap = stage_entry(cfg, "max_candidates_per_stage", stage)
    return None if cap is None else int(cap)
