"""Visual proximity over frozen DINOv3 features: a similarity cutoff, distance shells,
and episode- and time-balanced selection within them. The verifier uses it to build
hard-negative goal tables (goals that look like the source but are not reachable), and
stitching uses it to build bridge candidates; the selection is shared, the eligibility
rules are not.

Frames are compared by the cosine similarity of their DINOv3 spatial-softmax features
(cameras in sorted order, concatenated, L2-normalized).
- ``tau`` is the ``percentile``-th percentile of the similarity between frames
  ``H`` policy steps apart within an episode; a candidate qualifies if its similarity
  to the source is >= ``tau``;
- qualified candidates fall into distance shells over ``1 - cos`` in ``[0, 1 - tau]``,
  log-spaced towards ``cos = 1`` (``shells`` halvings plus one far shell);
- per shell, candidates are ordered by a shuffled round-robin over episodes (and, within
  an episode, over time cells of ``spacing`` frames) and kept unless within ``spacing``
  frames of an already kept one in the same episode; the shells are then merged
  round-robin in shuffled order, with the same spacing rule, up to ``max_targets``;
  the recovery stage orders each shell uniformly over its frames instead;
- a draw for a source visits its candidates in a seeded shell-balanced order and takes
  the ``draw_index``-th, so draws with the same (seed, source, draw_index) coincide.
Every random choice is seeded by the source frame, so tables and draws are identical
for the same inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

from needlework.constants import (
    CROSS_POOL_STREAM,
    DRAW_BLOCK,
    SHELL_ORDER_STREAM,
    SIMILARITY_BATCH,
    UNIFORM_SHELL_STREAM,
)


def embeddings(features: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    """Per-frame unit vectors: cameras (sorted) concatenated, then L2-normalized."""
    flat = [
        torch.from_numpy(features[key].reshape(features[key].shape[0], -1))
        for key in sorted(features)
    ]
    combined = torch.cat(flat, dim=1).to(device)
    return combined / combined.norm(dim=1, keepdim=True)


def calibrate_tau(
    emb: torch.Tensor,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    frames: int,
    *,
    percentile: float,
) -> float:
    """``percentile``-th percentile of cos(frame t, frame t + ``frames``) over the
    given episodes."""
    values = []
    for start, end in zip(episode_starts.tolist(), episode_ends.tolist(), strict=True):
        if end - start > frames:
            pairs = emb[start : end - frames] * emb[start + frames : end]
            values.append(pairs.sum(1))
    return float(np.percentile(torch.cat(values).cpu().numpy(), percentile))


def shell_edges(tau: float, shells: int) -> np.ndarray:
    """``[0, (1-tau)/2^n, ..., (1-tau)/2, 1-tau]`` over ``1 - cos``."""
    far = 1.0 - tau
    near = [far / 2.0**k for k in range(shells, 0, -1)]
    return np.array([0.0, *near, far])


def _round_robin(queues: list[list[int]]) -> list[int]:
    """Pop from the end of each queue in turn until all are empty."""
    out = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop())
    return out


def _take_if_spaced(
    retained: dict[int, list[int]], *, episode: int, local: int, gap: int
) -> bool:
    """Record ``local`` as kept in ``episode`` and return True, unless it lies within
    ``gap`` frames of a frame already kept there."""
    kept = retained.setdefault(episode, [])
    if any(abs(local - other) < gap for other in kept):
        return False
    kept.append(local)
    return True


def balanced_selection(
    rows: np.ndarray,
    episode: np.ndarray,
    local: np.ndarray,
    *,
    max_rows: int,
    spacing: int,
    seed: int,
    source: int,
) -> np.ndarray:
    """Episode round-robin of time-cell round-robins, then the spacing rule."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, source]))
    by_episode = {}
    for e in np.unique(episode).tolist():
        mask = episode == e
        cells = local[mask] // spacing
        queues = {
            c: rng.permutation(rows[mask][cells == c]).tolist()
            for c in np.unique(cells).tolist()
        }
        order = rng.permutation(np.asarray(list(queues), dtype=np.int64)).tolist()
        by_episode[e] = _round_robin([queues[c] for c in order])
    order = rng.permutation(np.asarray(list(by_episode), dtype=np.int64)).tolist()
    where = dict(
        zip(
            rows.tolist(),
            zip(episode.tolist(), local.tolist(), strict=True),
            strict=True,
        )
    )
    retained: dict[int, list[int]] = {}
    selected = []
    for row in _round_robin([by_episode[e] for e in order]):
        episode_of_row, local_of_row = where[row]
        if _take_if_spaced(
            retained, episode=episode_of_row, local=local_of_row, gap=spacing
        ):
            selected.append(row)
            if len(selected) == max_rows:
                break
    return np.asarray(selected, dtype=np.int64)


def _shell_of(sims: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Shell index of each candidate from its similarity to the source."""
    n_shells = len(edges) - 1
    distance = np.maximum(0.0, 1.0 - sims).astype(np.float64)
    return np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, n_shells - 1)


def _merge_shells(
    per_shell: list[np.ndarray],
    episode: np.ndarray,
    local: np.ndarray,
    *,
    shell_order: list[int],
    spacing: int,
    max_targets: int,
) -> list[np.ndarray]:
    """Round-robin over the shells' queues in ``shell_order`` with the spacing rule, up
    to ``max_targets``: the kept candidates of each shell."""
    kept: list[list[int]] = [[] for _ in per_shell]
    cursors = [0] * len(per_shell)
    retained: dict[int, list[int]] = {}
    total = 0
    while total < max_targets:
        added = False
        for s in shell_order:
            if cursors[s] >= len(per_shell[s]):
                continue
            row = int(per_shell[s][cursors[s]])
            cursors[s] += 1
            added = True
            if _take_if_spaced(
                retained, episode=int(episode[row]), local=int(local[row]), gap=spacing
            ):
                kept[s].append(row)
                total += 1
                if total == max_targets:
                    break
        if not added:
            break
    return [np.sort(np.asarray(k, dtype=np.int64)) for k in kept]


def select_shells(
    rows: np.ndarray,
    sims: np.ndarray,
    frames: tuple[np.ndarray, np.ndarray],
    *,
    edges: np.ndarray,
    spacing: int,
    max_targets: int,
    seed: int,
    source: int,
) -> list[np.ndarray]:
    """Qualified candidates of one source -> the kept candidates of each shell, each
    shell ordered by the episode- and time-balanced round-robin."""
    episode, local = frames
    shell = _shell_of(sims, edges)
    per_shell = [
        balanced_selection(
            rows[shell == s],
            episode[rows[shell == s]],
            local[rows[shell == s]],
            max_rows=max_targets,
            spacing=spacing,
            seed=seed + s,
            source=source,
        )
        for s in range(len(edges) - 1)
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, source, SHELL_ORDER_STREAM])
    )
    shell_order = rng.permutation(len(edges) - 1).tolist()
    return _merge_shells(
        per_shell,
        episode,
        local,
        shell_order=shell_order,
        spacing=spacing,
        max_targets=max_targets,
    )


def uniform_shells(
    rows: np.ndarray,
    sims: np.ndarray,
    frames: tuple[np.ndarray, np.ndarray],
    *,
    edges: np.ndarray,
    spacing: int,
    max_targets: int,
    seed: int,
    source: int,
) -> list[np.ndarray]:
    """As ``select_shells``, each shell ordered uniformly over its frames (the recovery
    stage's draw)."""
    episode, local = frames
    shell = _shell_of(sims, edges)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, source, UNIFORM_SHELL_STREAM])
    )
    per_shell = [rng.permutation(rows[shell == s]) for s in range(len(edges) - 1)]
    shell_order = rng.permutation(len(edges) - 1).tolist()
    return _merge_shells(
        per_shell,
        episode,
        local,
        shell_order=shell_order,
        spacing=spacing,
        max_targets=max_targets,
    )


def cross_episode_candidates(
    source: int,
    source_episode: int,
    episodes: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    *,
    spacing: int,
    seed: int,
) -> Iterator[int]:
    """Frames of the other ``episodes``: a shuffled round-robin over episodes, each
    visiting its time cells in a seeded coprime-stride order, one random frame per
    cell. Any prefix is a pool of that size."""
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, source, CROSS_POOL_STREAM])
    )
    order = rng.permutation(episodes[episodes != source_episode]).tolist()
    states = {}
    for e in order:
        n_cells = (episode_ends[e] - 1 - episode_starts[e]) // spacing + 1
        start = int(rng.integers(0, n_cells))
        stride = int(rng.integers(1, n_cells + 1))
        while math.gcd(stride, n_cells) != 1:
            stride = stride % n_cells + 1
        states[e] = [n_cells, start, stride, 0]
    while True:
        added = False
        for e in order:
            n_cells, start, stride, cursor = states[e]
            if cursor == n_cells:
                continue
            states[e][3] += 1
            low = episode_starts[e] + ((start + cursor * stride) % n_cells) * spacing
            high = min(low + spacing, episode_ends[e])
            yield int(low + rng.integers(0, high - low))
            added = True
        if not added:
            return


@dataclass(frozen=True)
class ShellTable:
    """Per source row and shell, the kept candidate frames (CSR layout)."""

    sources: np.ndarray  # source frame of each row
    targets: np.ndarray
    offsets: np.ndarray  # [rows * n_shells + 1]
    n_shells: int

    def shell(self, row: int, s: int) -> np.ndarray:
        flat = row * self.n_shells + s
        return self.targets[self.offsets[flat] : self.offsets[flat + 1]]

    def draw(self, row: int, seed: int, draw_index: int) -> int | None:
        """The ``draw_index``-th candidate of the seeded shell-balanced order."""
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, int(self.sources[row]), draw_index // DRAW_BLOCK]
            )
        )
        queues = {}
        for s in range(self.n_shells):
            candidates = self.shell(row, s)
            if len(candidates):
                queues[s] = rng.permutation(candidates).tolist()
        if not queues:
            return None
        order = rng.permutation(np.asarray(list(queues), dtype=np.int64)).tolist()
        visits = _round_robin([queues[s] for s in order])
        return int(visits[draw_index % len(visits)])


def build_table(
    emb: torch.Tensor,
    sources: np.ndarray,
    candidates: Callable[[int], Iterator[int] | np.ndarray],
    frames: tuple[np.ndarray, np.ndarray],
    *,
    tau: float,
    edges: np.ndarray,
    spacing: int,
    max_targets: int,
    pool: int | None,
    seed: int,
    desc: str,
) -> ShellTable:
    """``candidates(row)``: the row's candidate frames, all at once (``pool`` None) or
    as an iterator read ``pool`` frames at a time, doubling while fewer than
    ``max_targets`` qualify."""
    per_row: list[list[np.ndarray]] = []
    for start in tqdm(range(0, len(sources), SIMILARITY_BATCH), desc=desc):
        batch = sources[start : start + SIMILARITY_BATCH]
        src = torch.from_numpy(batch).to(emb.device)
        sims_all = (emb[src] @ emb.T).cpu().numpy()
        for i, source in enumerate(batch.tolist()):
            found = candidates(start + i)
            if pool is None:
                rows = np.asarray(found, dtype=np.int64)
            else:
                rows = np.fromiter(islice(found, pool), dtype=np.int64)
                limit = pool
                qualified = (sims_all[i, rows] >= tau).sum()
                while len(rows) == limit and qualified < max_targets:
                    more = np.fromiter(islice(found, limit), dtype=np.int64)
                    if len(more) == 0:
                        break
                    rows, limit = np.concatenate([rows, more]), 2 * limit
                    qualified = (sims_all[i, rows] >= tau).sum()
            sims = sims_all[i, rows]
            keep = sims >= tau
            if not keep.any():
                per_row.append([np.empty(0, dtype=np.int64)] * (len(edges) - 1))
                continue
            per_row.append(
                select_shells(
                    rows[keep],
                    sims[keep],
                    frames,
                    edges=edges,
                    spacing=spacing,
                    max_targets=max_targets,
                    seed=seed,
                    source=source,
                )
            )
    flat = [shell for shells in per_row for shell in shells]
    lengths = np.array([len(shell) for shell in flat], dtype=np.int64)
    return ShellTable(
        sources=sources,
        targets=np.concatenate(flat),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        n_shells=len(edges) - 1,
    )
