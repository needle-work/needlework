"""Logged windows plus bridge-crossing windows, each weighted by its role (roles.py).

A logged window whose action rows reach a selected success source (a row at the
source, at or after its current frame) is replaced by that source's crossing window: the
bridge is the only continuation taught there. At the departure itself (current frame =
source) the logged window also stays in, as the crossing's twin, so the action logged
there is still taught. Any other logged action within one policy step (``stride`` raw
frames) of a selected source is masked: history rows, and UMI windows whose frame
lattice skips the source.

Every window weighs by its role (roles.py): a logged window by its current frame, a
departure crossing ``bridge_weight`` (success) or the task's ``failure_departure``
(failure), a twin the task's ``twin``, an approach crossing its frame's weight times the
task's ``approach``. Each group of equal-weight windows gets its share of the total
weight, taken as consecutive shuffled passes so its draws are spread evenly; a window of
weight 0 is never drawn, and a group expecting less than one draw per epoch is drawn in
that fraction of epochs. The epoch length is ``epoch_length``: ``windows``, one draw per
sampled window (weight > 0, twins included), or ``logged``, the logged dataset's length.
Bridge weight 0 ignores the bridges entirely: every logged window once, no masks, as
``sampler=normal``.

``SamplerOptions`` (``sampler.options``) picks five behaviours by name:
``path_start`` (``aligned``: every window as logged; ``repeat_current``: at a training
episode's first frame and the frame after a selected success source, the history repeats
the current frame and the action rows start at it, one step later), ``twin_rows``
(``own_source``: a twin supervises the action logged at its source; ``all``: also its
logged rows at other selected sources) and ``draw`` (``even_passes`` as above;
``rounded_passes``: each group's share rounded to the nearest count, no random offset;
``with_replacement``: each epoch drawn independently, with probability by weight),
``source_rows`` (``masked``: logged rows near a selected source and a failure episode's
logged rows are not targets; ``supervised``: every logged row is) and
``after_departure`` (``all``: every logged window; ``skipped``: after an episode's first
success departure, only windows at frames a success bridge skips, since a route that
departs does not return there). Under ``repeat_current`` a path-start window whose last
action row lands on a success source takes that bridge from the row.

Source-only split isolation excludes held-out success departures while keeping failure
departures. Bridge windows contain source-episode history and one bridge, never
target-episode actions and never a failure episode's logged actions (a failure bridge
has one window, at its source, and replaces nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from needlework.constants import STAGE_RECOVERY, STAGE_WITHIN
from needlework.data.bridges import Bridges
from needlework.geometry.actions import action_poses, from_relative, to_relative
from needlework.sampling import roles
from needlework.sampling.policy_dataset import PolicyDataset
from needlework.sampling.windows import window_frames


def _splice_bridge(
    action: np.ndarray, valid: np.ndarray, bridge: np.ndarray, start: int
) -> None:
    """Write one bridge into rows ``start`` onward of a single window.

    Rows [start, start + count) are the bridge and are supervised. Rows after it repeat
    the bridge's last action so the array keeps its shape, and are masked out: nothing
    after a bridge ends is a target.
    """
    count = min(len(bridge), len(action) - start)
    if count < 1:
        raise ValueError(
            f"a splice needs at least one bridge row: {len(bridge)} bridge rows "
            f"from row {start} of {len(action)}"
        )
    action[start : start + count] = bridge[:count]
    action[start + count :] = bridge[count - 1]
    valid[start : start + count] = True
    valid[start + count :] = False


# Draw kinds in an epoch.
KIND_LOGGED, KIND_TWIN, KIND_CROSSING = 0, 1, 2
EPOCH_LENGTHS = ("windows", "logged")


def _passes(rng: np.random.Generator, items: np.ndarray, count: int) -> np.ndarray:
    """``count`` items as consecutive shuffled passes over ``items``."""
    if count == 0:
        return np.empty(0, dtype=np.int64)
    passes = -(-count // len(items))
    return np.concatenate([rng.permutation(items) for _ in range(passes)])[:count]


def _aligned_starts(
    data: PolicyDataset, bridges: Bridges, sources: np.ndarray
) -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _repeat_current_starts(
    data: PolicyDataset, bridges: Bridges, sources: np.ndarray
) -> np.ndarray:
    """Path starts: each training episode's first frame and the frame after each
    success source, unless that frame is itself a source."""
    if data.shape.stride != 1 or data.relative_actions:
        raise ValueError(
            "path_start=repeat_current is defined for stride-1 absolute actions only"
        )
    store = data.store
    success = store.episode_success[store.episode_of_frame(bridges.source_frame)]
    candidates = np.concatenate(
        [
            store.episode_starts[np.unique(data.windows.episode)],
            bridges.source_frame[success] + 1,
        ]
    )
    return np.setdiff1d(candidates, sources)


def _own_source_rows(valid: np.ndarray, n_obs: int) -> np.ndarray:
    valid[:, n_obs - 1] = True  # the action logged at the source
    return valid


def _all_rows(valid: np.ndarray, n_obs: int) -> np.ndarray:
    return np.ones_like(valid)  # also the logged rows at other selected sources


def _even_passes(rng: np.random.Generator, groups: list, length: int) -> tuple:
    """Each equal-weight group gets its share of the epoch as consecutive shuffled
    passes; the cumulative mass is rounded at a random offset, so each group gets the
    floor or ceiling of its expected count, unbiased over epochs, summing to
    ``length``."""
    mass = np.array([value * len(items) for _, items, value in groups])
    bounds = np.floor(np.cumsum(mass) / mass.sum() * length + rng.random())
    bounds[-1] = length
    counts = np.diff(bounds.astype(int), prepend=0)
    draws = [
        _passes(rng, items, c) for (_, items, _), c in zip(groups, counts, strict=True)
    ]
    return np.concatenate(draws), np.repeat([kind for kind, _, _ in groups], counts)


def _rounded_passes(rng: np.random.Generator, groups: list, length: int) -> tuple:
    """Each equal-weight group after the first gets its share of the epoch rounded to
    the nearest count, the first group the rest; each taken as consecutive shuffled
    passes, group by group. No other random draw, so with one logged and one crossing
    group this is the plain replacement sampler's epoch exactly."""
    mass = [value * len(items) for _, items, value in groups]
    total = sum(mass)
    counts = [0] + [round(length * m / total) for m in mass[1:]]
    counts[0] = length - sum(counts[1:])
    draws = [
        _passes(rng, items, c) for (_, items, _), c in zip(groups, counts, strict=True)
    ]
    return np.concatenate(draws), np.repeat([kind for kind, _, _ in groups], counts)


def _with_replacement(rng: np.random.Generator, groups: list, length: int) -> tuple:
    """``length`` independent draws with probability proportional to weight."""
    items = np.concatenate([items for _, items, _ in groups])
    kinds = np.concatenate([np.full(len(items), kind) for kind, items, _ in groups])
    weight = np.concatenate([np.full(len(items), value) for _, items, value in groups])
    pick = rng.choice(len(items), size=length, replace=True, p=weight / weight.sum())
    return items[pick], kinds[pick]


PATH_STARTS = {"aligned": _aligned_starts, "repeat_current": _repeat_current_starts}
TWIN_ROWS = {"own_source": _own_source_rows, "all": _all_rows}
DRAWS = {
    "even_passes": _even_passes,
    "rounded_passes": _rounded_passes,
    "with_replacement": _with_replacement,
}


def _masked_source_rows(data, frames: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """Logged rows near a selected source, and a failure episode's logged rows, are not
    targets."""
    return ~data.near_sources(frames) & data.data.store.episode_success[episode, None]


def _supervised_source_rows(
    data, frames: np.ndarray, episode: np.ndarray
) -> np.ndarray:
    return np.ones(frames.shape, dtype=bool)  # every logged row is a target


SOURCE_ROWS = {"masked": _masked_source_rows, "supervised": _supervised_source_rows}


def _all_logged(data, frames: np.ndarray) -> np.ndarray:
    return np.ones(len(frames), dtype=bool)


def _skipped_after_departure(data, frames: np.ndarray) -> np.ndarray:
    """After its episode's first success departure a logged window stays only where a
    success bridge skips its frame (a route that departs never returns there)."""
    store, b = data.data.store, data.bridges
    success = b.stage != STAGE_RECOVERY
    first = np.full(len(store.episode_starts), np.iinfo(np.int64).max)
    np.minimum.at(
        first, store.episode_of_frame(b.source_frame[success]), b.source_frame[success]
    )
    skipped = np.zeros(store.n_frames, dtype=bool)
    for s, t, stage in zip(
        b.source_frame[success], b.target_frame[success], b.stage[success], strict=True
    ):
        end = (
            t
            if stage == STAGE_WITHIN
            else store.episode_ends[store.episode_of_frame(s)]
        )
        skipped[s + 1 : end] = True
    return (frames < first[store.episode_of_frame(frames)]) | skipped[frames]


AFTER_DEPARTURE = {"all": _all_logged, "skipped": _skipped_after_departure}


@dataclass(frozen=True)
class SamplerOptions:
    path_start: str
    twin_rows: str
    draw: str
    source_rows: str
    after_departure: str

    def __post_init__(self) -> None:
        for key, known in (
            ("path_start", PATH_STARTS),
            ("twin_rows", TWIN_ROWS),
            ("draw", DRAWS),
            ("source_rows", SOURCE_ROWS),
            ("after_departure", AFTER_DEPARTURE),
        ):
            if getattr(self, key) not in known:
                value = getattr(self, key)
                raise ValueError(
                    f"sampler option {key}={value!r}; known {sorted(known)}"
                )


class AugmentedPolicyDataset:
    def __init__(
        self,
        logged: PolicyDataset,
        bridges: Bridges,
        *,
        weight: float,
        role_weights: roles.RoleWeights,
        epoch_length: str,
        options: SamplerOptions,
        seed: int,
    ) -> None:
        if not (np.isfinite(weight) and weight >= 0):
            raise ValueError(f"bridge weight must be finite and >= 0, got {weight}")
        if epoch_length not in EPOCH_LENGTHS:
            raise ValueError(
                f"epoch_length must be one of {EPOCH_LENGTHS}, got {epoch_length!r}"
            )
        self.data, self.seed = logged, seed
        self.role_weights, self.epoch_length = role_weights, epoch_length
        self.options = options
        store = logged.store
        source_episode = store.episode_of_frame(bridges.source_frame)
        if weight == 0:
            # Weight 0 is ordinary logged training: no bridges and no masks.
            eligible = np.zeros(len(bridges), dtype=bool)
        else:
            from_failure = ~store.episode_success[source_episode]
            in_training_split = np.isin(
                source_episode, np.unique(logged.windows.episode)
            )
            # Source-only split isolation: a held-out success episode contributes no
            # departures; failure episodes are never in the policy split, so they do.
            eligible = from_failure | in_training_split
        self.bridges = b = bridges.subset(np.flatnonzero(eligible))
        self.selected_sources = np.sort(b.source_frame)
        crossings = self._crossing_windows(b)
        frames = window_frames(
            logged.windows, np.arange(len(logged)), store.episode_ends
        )
        current = frames[:, logged.shape.n_obs - 1]
        # A success crossing exists where a logged window does, and replaces it.
        twin = np.isin(crossings[:, 1], current)
        self.crossings = crossings[twin | ~store.episode_success[crossings[:, 0]]]
        replaced = np.isin(current, self.crossings[:, 1])
        # Drop logged windows only when replaced or every action is masked.
        valid = ~self.near_sources(frames)
        self.logged_windows = np.flatnonzero(valid.any(1) & ~replaced)
        kept = AFTER_DEPARTURE[options.after_departure](
            self, current[self.logged_windows]
        )
        self.logged_windows = self.logged_windows[kept]
        if len(self.logged_windows) == 0:
            raise ValueError("selected sources replace or mask every logged window")
        frame = roles.frame_weights(store, b, role_weights)
        self.logged_weight = frame[current[self.logged_windows]]
        episode, crossing_current, row = self.crossings.T
        departure = crossing_current == b.source_frame[row]
        success = store.episode_success[episode]
        self.crossing_weight = roles.crossing_weights(
            frame[crossing_current], departure, success, weight, role_weights
        )
        self.twins = np.flatnonzero(
            np.isin(current, crossing_current[departure & success])
        )
        self.twin_weight = np.full(len(self.twins), role_weights.twin)
        self.path_starts = PATH_STARTS[options.path_start](
            logged, b, self.selected_sources
        )
        if not len(b):
            print(
                "No bridges in use (bridge_weight 0 or none eligible): "
                "using logged windows only.",
                flush=True,
            )
        self.set_epoch(0)

    def __len__(self) -> int:
        if self.epoch_length == "logged":
            return len(self.data)
        return int(
            np.count_nonzero(self.logged_weight)
            + np.count_nonzero(self.crossing_weight)
            + np.count_nonzero(self.twin_weight)
        )

    def _crossing_windows(self, b: Bridges) -> np.ndarray:
        """[n, 3] (episode, current frame, bridge row): the windows bridges depart from.

        A success source is entered from each of the ``horizon - n_obs + 1`` window
        positions at or before it on its stride lattice, inside its episode. A failure
        source is entered from itself only: a failure episode's logged actions are the
        failing policy's and never supervised. Where two bridges reach the same
        (episode, current frame), the earlier source frame owns it.
        """
        store, shape = self.data.store, self.data.shape
        episodes = store.episode_of_frame(b.source_frame)
        starts = store.episode_starts[episodes]
        n_leads = shape.horizon - shape.n_obs + 1
        owner: dict[tuple[int, int], int] = {}
        for row in np.argsort(b.source_frame, kind="stable"):
            episode = int(episodes[row])
            leads = n_leads if store.episode_success[episode] else 1
            for lead in range(leads):
                current = int(b.source_frame[row]) - lead * shape.stride
                if current >= starts[row]:
                    owner.setdefault((episode, current), int(row))
        return np.asarray(
            [(e, c, row) for (e, c), row in owner.items()], dtype=np.int64
        ).reshape(-1, 3)

    def near_sources(self, frames: np.ndarray) -> np.ndarray:
        """Frames whose logged actions are masked: within one policy step (``stride``
        raw frames) of a selected source in the same episode. Exactly the source frame
        at stride 1; its neighbors too on a coarser lattice, which windows starting off
        the source's lattice would otherwise supervise."""
        sources, store = self.selected_sources, self.data.store
        flat = frames.reshape(-1)
        near = np.zeros(flat.shape, dtype=bool)
        if len(sources):
            above = np.searchsorted(sources, flat)
            episode = store.episode_of_frame(flat)
            for index in (above - 1, above):  # the nearest source on either side
                source = sources[np.clip(index, 0, len(sources) - 1)]
                near |= (
                    (index >= 0)
                    & (index < len(sources))
                    & (np.abs(flat - source) < self.data.shape.stride)
                    & (store.episode_of_frame(source) == episode)
                )
        return near.reshape(frames.shape)

    def set_epoch(self, epoch: int) -> None:
        """Fix the epoch's draws: ``self.draw`` (a logged window index, for a twin too,
        or a crossing index), ``self.is_bridge`` and ``self.is_twin``, reproducible from
        (seed, epoch)."""
        rng = np.random.default_rng([self.seed, epoch])
        groups = []  # (kind, items, weight)
        for kind, items, weights in (
            (KIND_LOGGED, self.logged_windows, self.logged_weight),
            (KIND_TWIN, self.twins, self.twin_weight),
            (KIND_CROSSING, np.arange(len(self.crossings)), self.crossing_weight),
        ):
            for value in np.unique(weights[weights > 0]):
                groups.append((kind, items[weights == value], value))
        if not groups:
            raise ValueError("no window has positive weight")
        draws, kinds = DRAWS[self.options.draw](rng, groups, len(self))
        order = rng.permutation(len(self))
        self.draw = draws[order]
        self.is_bridge, self.is_twin = (
            kinds[order] == KIND_CROSSING,
            kinds[order] == KIND_TWIN,
        )

    def aligned_frames(
        self, episode: np.ndarray, frames: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(observation frames, action frames, shifted) of windows: at a path start
        (``path_starts``) the history repeats the current frame and the action rows
        start at it; elsewhere, and always under path_start=aligned, the window's own
        frames."""
        n_obs, ends = self.data.shape.n_obs, self.data.store.episode_ends
        shift = np.isin(frames[:, n_obs - 1], self.path_starts)
        obs_frames = frames[:, :n_obs].copy()
        obs_frames[shift] = frames[shift, n_obs - 1, None]
        action_frames = frames.copy()
        action_frames[shift] = np.minimum(
            frames[shift, n_obs - 1, None] + np.arange(frames.shape[1]),
            ends[episode[shift], None] - 1,
        )
        return obs_frames, action_frames, shift

    def logged_arrays(self, indices: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        data = self.data
        frames = window_frames(data.windows, indices, data.store.episode_ends)
        episode = data.windows.episode[indices]
        rows = SOURCE_ROWS[self.options.source_rows]
        if not self.path_starts.size:  # every window as logged
            obs, action = data.arrays(indices, data.train)
            return obs, action, rows(self, frames, episode)
        obs_frames, action_frames, shift = self.aligned_frames(episode, frames)
        obs = data.observations(obs_frames, episode, data.train)
        action = data.store.arrays["action"][action_frames].copy()
        valid = rows(self, action_frames, episode)
        self._splice_reached_bridges(action, valid, action_frames, shift)
        return obs, action, valid

    def _splice_reached_bridges(
        self,
        action: np.ndarray,
        valid: np.ndarray,
        frames: np.ndarray,
        shift: np.ndarray,
    ) -> None:
        """A shifted path-start window whose last action row lands on a success source
        (its unshifted rows stop one short) takes that bridge from the row, as the route
        does."""
        store, b = self.data.store, self.bridges
        success = store.episode_success[store.episode_of_frame(b.source_frame)]
        row_of = {int(f): r for r, f in enumerate(b.source_frame) if success[r]}
        for i in np.flatnonzero(shift):
            if int(frames[i, -1]) in row_of:
                row = row_of[int(frames[i, -1])]
                start = frames.shape[1] - 1
                _splice_bridge(
                    action[i], valid[i], self._bridge_in_source_frame(row), start
                )

    def twin_arrays(self, indices: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        """A departure's logged twin: its logged window, with the action logged at the
        source (the current row) supervised (``twin_rows``)."""
        obs, action, valid = self.logged_arrays(indices)
        rows = TWIN_ROWS[self.options.twin_rows]
        return obs, action, rows(valid, self.data.shape.n_obs)

    def bridge_arrays(self, indices: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        """Crossing windows: the source episode's logged history and actions, each with
        its bridge spliced in from the source's row onward."""
        data, b, shape = self.data, self.bridges, self.data.shape
        episode, current, rows = self.crossings[indices].T
        obs_frames, frames, shift = self.aligned_frames(
            episode, self._window_frames(episode, current)
        )
        obs = data.observations(obs_frames, episode, data.train)
        action = data.store.arrays["action"][frames].copy()
        # Logged rows near sources and in failure episodes: source_rows.
        valid = SOURCE_ROWS[self.options.source_rows](self, frames, episode)
        for i, row in enumerate(rows):
            start = shape.n_obs - 1 + (b.source_frame[row] - current[i]) // shape.stride
            start -= int(shift[i])
            _splice_bridge(
                action[i], valid[i], self._bridge_in_source_frame(int(row)), start
            )
        return obs, self._window_relative(action, current), valid

    def _window_frames(self, episode: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Raw frames of each window's rows, clipped to its episode."""
        store, shape = self.data.store, self.data.shape
        positions = (np.arange(shape.horizon) - shape.n_obs + 1) * shape.stride
        return np.clip(
            current[:, None] + positions,
            store.episode_starts[episode, None],
            store.episode_ends[episode, None] - 1,
        )

    def _bridge_in_source_frame(self, row: int) -> np.ndarray:
        """A bridge's actions in the store's frame (UMI bridges are stored relative to
        the source gripper pose)."""
        bridge = self.bridges.rows(row)
        if not self.data.relative_actions:
            return bridge
        store = self.data.store
        source = self.bridges.source_frame[row]
        base = action_poses(store.layout, store.arrays["action"][source])
        return from_relative(store.layout, bridge, base)

    def _window_relative(self, action: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Window actions relative to each window's current gripper pose (UMI)."""
        if not self.data.relative_actions:
            return action
        store = self.data.store
        base = action_poses(store.layout, store.arrays["action"][current])
        return to_relative(store.layout, action, base)

    def batch(self, indices: np.ndarray) -> dict:
        flags, twins = self.is_bridge[indices], self.is_twin[indices]
        draws = self.draw[indices]
        parts, positions = [], []
        for mask, builder in (
            (~flags & ~twins, self.logged_arrays),
            (twins, self.twin_arrays),
            (flags, self.bridge_arrays),
        ):
            if mask.any():
                parts.append(builder(draws[mask]))
                positions.append(np.flatnonzero(mask))
        order = np.argsort(np.concatenate(positions))
        obs = {
            key: np.concatenate([p[0][key] for p in parts])[order]
            for key in parts[0][0]
        }
        action = np.concatenate([p[1] for p in parts])[order]
        valid = np.concatenate([p[2] for p in parts])[order]
        return {
            "obs": {k: torch.from_numpy(v) for k, v in obs.items()},
            "action": torch.from_numpy(action),
            "valid": torch.from_numpy(valid),
        }
