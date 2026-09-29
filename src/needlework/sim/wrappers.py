"""The rollout stack, one concern per wrapper (innermost first):

    SeededReset       reset(seed) seeds every random stream, then resets
    StoreObservations robomimic observations -> store arrays (HWC uint8, float32)
    PolicyActions     a policy action row (rotation 6D) -> the controller's axis-angle
    EpisodeEnd        ends the episode at its first success or after max_steps
    ChunkRunner       keeps the observation history; executes an action chunk

Each exposes ``reset(seed)`` and ``step(...)``; nothing here is gym.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from needlework.geometry.actions import ActionLayout, to_env_command
from needlework.sim.seeding import PlacementSeedHook, seed_everything


class SeededReset:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.hook = PlacementSeedHook(env.env)

    def reset(self, seed: int) -> dict:
        seed_everything(self.env.env, self.hook, seed)
        return self.env.reset()

    def step(self, command: np.ndarray) -> tuple[dict, bool]:
        obs, _, _, _ = self.env.step(command)
        return obs, bool(self.env.is_success()["task"])


class StoreObservations:
    def __init__(
        self, env: SeededReset, cameras: tuple[str, ...], proprio: tuple[str, ...]
    ) -> None:
        self.env = env
        self.cameras = cameras
        self.proprio = proprio

    def _convert(self, raw: dict) -> dict[str, np.ndarray]:
        # robomimic returns images as float CHW in [0, 1]; x * 255 recovers the rendered
        # bytes exactly for all 256 values.
        out = {
            key: np.clip(raw[key] * 255.0, 0.0, 255.0)
            .astype(np.uint8)
            .transpose(1, 2, 0)
            for key in self.cameras
        }
        out.update({key: raw[key].astype(np.float32) for key in self.proprio})
        return out

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        return self._convert(self.env.reset(seed))

    def step(self, command: np.ndarray) -> tuple[dict[str, np.ndarray], bool]:
        raw, success = self.env.step(command)
        return self._convert(raw), success


class PolicyActions:
    def __init__(self, env: StoreObservations, layout: ActionLayout) -> None:
        self.env = env
        self.layout = layout

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        return self.env.reset(seed)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], bool]:
        return self.env.step(to_env_command(self.layout, action))


class EpisodeEnd:
    def __init__(self, env: PolicyActions, max_steps: int) -> None:
        self.env = env
        self.max_steps = max_steps
        self.steps = 0
        self.first_success: int | None = None

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        self.steps = 0
        self.first_success = None
        return self.env.reset(seed)

    @property
    def done(self) -> bool:
        return self.first_success is not None or self.steps >= self.max_steps

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        if self.done:
            raise RuntimeError("step after the episode ended")
        obs, success = self.env.step(action)
        self.steps += 1
        if success:
            self.first_success = self.steps
        return obs


class ChunkRunner:
    def __init__(self, env: EpisodeEnd, n_obs: int) -> None:
        self.env = env
        self.history: deque = deque(maxlen=n_obs)

    def _stacked(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack([obs[key] for obs in self.history]) for key in self.history[0]
        }

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        first = self.env.reset(seed)
        self.history.extend([first] * self.history.maxlen)  # repeat the first frame
        return self._stacked()

    def step(self, chunk: np.ndarray) -> tuple[dict[str, np.ndarray], bool]:
        """Execute rows of ``chunk`` [k, A] until they run out or the episode ends."""
        for action in chunk:
            self.history.append(self.env.step(action))
            if self.env.done:
                break
        return self._stacked(), self.env.done

    def result(self) -> dict[str, int | None]:
        """``length_at_success``: steps to the first success, None for a failure."""
        return {"length_at_success": self.env.first_success, "steps": self.env.steps}
