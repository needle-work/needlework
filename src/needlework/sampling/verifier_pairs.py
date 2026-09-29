"""Verifier examples: (source window, goal frame, actions) -> reached-by-step labels.

Three types. Every window with a future step heads one group per epoch of
``1 + n_negatives`` examples:
- slot 0, ``within_hor_pos``: the window's own goal ``k`` steps ahead, ``k`` uniform in
  ``[1, min(max_future, H)]``; label ``h``
  (h = 1..H) is 1 iff ``h >= k``;
- slots 1.., negatives, all labels 0, each with its source's logged actions. A slot
  draws its type by ``negative_ratio``, then its own source window uniformly among the
  windows that type supports, then a goal from that source's proximity table
  (``sampling/proximity.py``): the ``j``-th candidate of that source's seeded
  shell-balanced order. Under ``negative_draw: epoch_wide``, ``j`` counts the earlier
  draws of the same (type, source) anywhere in the epoch, so a source drawn in many
  groups walks through its table; under ``within_group``, ``j``
  counts the earlier slots of the same type in this group only, so such a source
  restarts at its first candidates. Groups are visited in a fixed order, so the epoch
  is still reproducible from (seed, epoch):
  - ``beyond_hor_neg``: a goal later in the same episode, beyond the horizon. Sources
    need at least ``n_negatives`` such steps. Without table candidates, ``k`` in
    ``(H, max_future]``, with probability ``boundary_prob`` in the boundary band
    ``(H, boundary_horizons * H]``;
  - ``cross_traj_neg``: a goal in another episode of this split. Without table
    candidates, a uniform frame of the other episodes.
Actions are served per the task's ``serving`` mode:
- ``label_dependent`` (Robomimic): every positive holds from its
  goal step, every negative continues its logged actions. The tail reveals the label.
- ``independent`` (UMI): each example holds with ``hold_probability``, otherwise
  continues. Positive holds begin at the true goal; negative holds begin at a uniform
  admissible within-horizon offset. This choice never identifies the label.
Each slot's random draws are seeded by (seed, epoch, window, slot).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import numpy as np
import torch
from tqdm import tqdm

from needlework.constants import HOLD_STREAM
from needlework.data.features import Rows
from needlework.sampling import proximity
from needlework.sampling.pairs import action_horizon, current_frames, hold_from
from needlework.sampling.policy_dataset import PolicyDataset

POSITIVE = "within_hor_pos"
NEGATIVES = ("beyond_hor_neg", "cross_traj_neg")
NEGATIVE_DRAWS = ("epoch_wide", "within_group")
SERVING_KEYS = {
    "label_dependent": {"mode"},
    "independent": {"mode", "hold_probability"},
}


def _seed(*values: int) -> int:
    return int(np.random.SeedSequence(values).generate_state(1)[0])


class VerifierPairs:
    def __init__(
        self,
        data: PolicyDataset,
        spatial: dict[str, np.ndarray],
        patches: dict[str, Rows],
        cfg: dict,
        *,
        tau: float,
        seed: int,
        resample: bool,
        device: torch.device,
    ) -> None:
        self.data, self.patches = data, patches
        self.seed, self.resample = seed, resample
        store = data.store
        self.horizon = h = action_horizon(data)
        self.stride = data.shape.stride
        self.current = current_frames(data)  # per window
        self.episode = data.windows.episode
        last = store.episode_ends[self.episode] - 1
        self.max_future = (last - self.current) // self.stride
        self.n_negatives = cfg["n_negatives"]
        self.boundary_prob = cfg["boundary_prob"]
        self.boundary_horizons = cfg["boundary_horizons"]
        self.serving = dict(cfg["serving"])
        self.check_serving(self.serving)
        self.negative_draw = cfg["negative_draw"]
        self.check_negative_draw(self.negative_draw)
        ratio = np.array([cfg["negative_ratio"][t] for t in NEGATIVES], np.float64)
        self.negative_p = ratio / ratio.sum()
        has_future = self.max_future >= 1
        self.groups = np.flatnonzero(has_future)
        self.sources = {
            "beyond_hor_neg": np.flatnonzero(
                has_future & (self.max_future - h >= self.n_negatives)
            ),
            "cross_traj_neg": self.groups,
        }
        self.split_episodes = np.unique(self.episode)
        if len(self.split_episodes) < 2:
            raise ValueError("cross_traj_neg needs at least two episodes")
        self.tables = self._tables(spatial, cfg["proximity"], tau, device)
        self.set_epoch(0)

    def _tables(
        self,
        spatial: dict[str, np.ndarray],
        cfg: dict,
        tau: float,
        device: torch.device,
    ) -> dict[str, proximity.ShellTable]:
        store, h, stride = self.data.store, self.horizon, self.stride
        frames = np.arange(store.episode_ends[-1])
        frame_episode = store.episode_of_frame(frames)
        frame_local = frames - store.episode_starts[frame_episode]
        spacing = h * stride

        def beyond(row: int) -> np.ndarray:
            if self.max_future[row] <= h:
                return np.empty(0, dtype=np.int64)
            steps = np.arange(h + 1, self.max_future[row] + 1)
            return self.current[row] + steps * stride

        def cross(row: int) -> Iterator[int]:
            if self.max_future[row] < 1:
                return iter(())
            return proximity.cross_episode_candidates(
                int(self.current[row]),
                int(self.episode[row]),
                self.split_episodes,
                store.episode_starts,
                store.episode_ends,
                spacing=spacing,
                seed=self.seed,
            )

        emb = proximity.embeddings(spatial, device)
        common = {
            "frames": (frame_episode, frame_local),
            "tau": tau,
            "edges": proximity.shell_edges(tau, cfg["shells"]),
            "spacing": spacing,
            "max_targets": cfg["max_targets"],
            "seed": self.seed,
        }
        pools = {"beyond_hor_neg": None, "cross_traj_neg": cfg["candidate_pool"]}
        candidates = {"beyond_hor_neg": beyond, "cross_traj_neg": cross}
        return {
            name: proximity.build_table(
                emb,
                self.current,
                candidates[name],
                pool=pools[name],
                desc=name,
                **common,
            )
            for name in NEGATIVES
        }

    def __len__(self) -> int:
        return len(self.groups) * (1 + self.n_negatives)

    def set_epoch(self, epoch: int) -> None:
        step = epoch if self.resample else 0
        h, slots = self.horizon, 1 + self.n_negatives
        row = np.empty((len(self.groups), slots), dtype=np.int64)
        goal = np.empty_like(row)
        offset = np.zeros_like(row)
        fallbacks = np.zeros(len(NEGATIVES))
        drawn = np.zeros(len(NEGATIVES))
        earlier_draws: Counter[tuple[int, int]] = Counter()  # (type, source) -> count
        groups = tqdm(self.groups.tolist(), desc="verifier examples")
        epoch_wide = self.negative_draw == "epoch_wide"
        for g, anchor in enumerate(groups):
            in_group: Counter[int] = Counter()  # type -> earlier slots in this group
            rng = np.random.default_rng(_seed(self.seed, step, anchor, 0))
            k = int(rng.integers(1, min(self.max_future[anchor], h) + 1))
            row[g, 0], offset[g, 0] = anchor, k
            goal[g, 0] = self.current[anchor] + k * self.stride
            for slot in range(1, slots):
                rng = np.random.default_rng(_seed(self.seed, step, anchor, slot))
                t = int(rng.choice(len(NEGATIVES), p=self.negative_p))
                sources = self.sources[NEGATIVES[t]]
                source = int(sources[rng.integers(0, len(sources))])
                index = earlier_draws[(t, source)] if epoch_wide else in_group[t]
                target = self.tables[NEGATIVES[t]].draw(source, self.seed + step, index)
                earlier_draws[(t, source)] += 1
                in_group[t] += 1
                drawn[t] += 1
                if target is None:
                    fallbacks[t] += 1
                    target = self._fallback(t, source, rng)
                row[g, slot], goal[g, slot] = source, target
        self.fallback = {
            name: float(fallbacks[t] / max(drawn[t], 1))
            for t, name in enumerate(NEGATIVES)
        }
        self.row, self.offset, self.goal = row.ravel(), offset.ravel(), goal.ravel()
        self.hold, self.hold_offset = self._holds(step)

    @staticmethod
    def check_negative_draw(rule: str) -> None:
        if rule not in NEGATIVE_DRAWS:
            raise ValueError(f"negative_draw must be one of {NEGATIVE_DRAWS}: {rule}")

    @staticmethod
    def check_serving(serving: dict) -> None:
        if (
            "mode" not in serving
            or serving["mode"] not in SERVING_KEYS
            or set(serving) != SERVING_KEYS[serving["mode"]]
        ):
            raise ValueError(f"verifier serving must match {SERVING_KEYS}: {serving}")

    def _holds(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Which examples hold their actions, and from which step."""
        if self.serving["mode"] == "label_dependent":
            return self.offset > 0, self.offset
        # Separate stream keeps the established source/goal draw unchanged.
        rng = np.random.default_rng([self.seed, step, HOLD_STREAM])
        hold = rng.random(len(self.row)) < self.serving["hold_probability"]
        random_offset = rng.integers(
            1, np.minimum(self.max_future[self.row], self.horizon) + 1
        )
        return hold, np.where(self.offset > 0, self.offset, random_offset)

    def _fallback(self, t: int, source: int, rng: np.random.Generator) -> int:
        h, store = self.horizon, self.data.store
        if NEGATIVES[t] == "beyond_hor_neg":
            high = int(self.max_future[source])
            if float(rng.random()) < self.boundary_prob:
                high = min(high, self.boundary_horizons * h)
            k = int(rng.integers(h + 1, high + 1))
            return int(self.current[source] + k * self.stride)
        episodes = self.split_episodes
        counts = store.episode_ends[episodes] - store.episode_starts[episodes]
        cumulative = np.cumsum(counts)
        own = int(np.searchsorted(episodes, self.episode[source]))
        draw = int(rng.integers(0, cumulative[-1] - counts[own]))
        if draw >= cumulative[own] - counts[own]:
            draw += int(counts[own])
        target = int(np.searchsorted(cumulative, draw, side="right"))
        start = store.episode_starts[episodes[target]]
        return int(start + rng.integers(0, counts[target]))

    def batch(self, indices: np.ndarray) -> dict:
        rows, offset, goal = self.row[indices], self.offset[indices], self.goal[indices]
        positive = offset > 0
        obs, action = self.data.arrays(rows, self.data.train)
        held = hold_from(action, self.hold_offset[indices], self.data.shape.n_obs)
        action = np.where(self.hold[indices, None, None], held, action)
        steps = np.arange(1, self.horizon + 1)[None, :]
        labels = positive[:, None] & (steps >= offset[:, None])
        source = self.current[rows]
        cameras = list(self.patches)  # the task's camera order
        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs.items()},
            "source_patches": torch.from_numpy(
                np.stack([self.patches[c][source] for c in cameras], axis=1)
            ),
            "goal_patches": torch.from_numpy(
                np.stack([self.patches[c][goal] for c in cameras], axis=1)
            ),
            "action": torch.from_numpy(action),
            "labels": torch.from_numpy(labels.astype(np.float32)),
            "positive": torch.from_numpy(positive),
        }
