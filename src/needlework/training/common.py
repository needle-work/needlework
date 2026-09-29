"""What every component's task builds the same way: the store, its episode split,
observation windows over it, and the normalizer fit on training windows.

Split: the policy validates on held-out success episodes and trains on the remaining
successes; the IDM and the verifier hold out ``val_ratio`` of all episodes and train on
the rest, failures included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from needlework.constants import ACTION_DIM_PER_ARM, ACTION_LAYOUT, SPATIAL
from needlework.data import features
from needlework.data.splits import split_episodes
from needlework.data.store import EpisodeStore
from needlework.sampling import policy_dataset
from needlework.sampling.observations import RobomimicObservations, UmiObservations
from needlework.sampling.policy_dataset import PolicyDataset, WindowShape


def identity_dims(n_arms: int, parts: list[str]) -> np.ndarray:
    """Action dimensions left unnormalized: the listed parts of every arm."""
    dims = np.zeros((n_arms, ACTION_DIM_PER_ARM), dtype=bool)
    for part in parts:
        low, high = ACTION_LAYOUT[part]
        dims[:, low:high] = True
    return dims.reshape(-1)


@dataclass(frozen=True)
class Episodes:
    train: np.ndarray
    val: np.ndarray


class Windows:
    """Builds ``PolicyDataset``s over one store with one observation set."""

    def __init__(self, cfg: DictConfig, *, cameras: bool, n_execute: int) -> None:
        """Observations: the task's proprioception, and its cameras' spatial-softmax
        features if ``cameras`` (then also kept in ``self.features``)."""
        self.cfg = cfg
        task = cfg.task
        self.store = EpisodeStore.open(task.domain, task.name)
        self.features = features.load(self.store, SPATIAL) if cameras else {}
        keys = (*(task.obs.cameras if cameras else ()), *task.obs.proprio)
        if task.domain == "robomimic":
            self.observations = RobomimicObservations(self.store, self.features, keys)
        elif task.domain == "umi":
            self.observations = UmiObservations(
                self.store, self.features, keys, task.start_noise
            )
        else:
            raise ValueError(f"unknown domain {task.domain}")
        self.shape = WindowShape(
            horizon=cfg.horizon.prediction,
            n_obs=cfg.horizon.obs,
            n_execute=n_execute,
            stride=task.stride,
        )

    def episodes(self, *, successes_only: bool) -> Episodes:
        success = self.store.episode_success
        split = split_episodes(
            success,
            val_ratio=self.cfg.component.validation.val_ratio,
            seed=self.cfg.seed.dataset,
            val_from="success" if successes_only else "all",
        )
        train = split.train & success if successes_only else split.train
        return Episodes(np.flatnonzero(train), np.flatnonzero(split.val))

    def dataset(self, episodes: np.ndarray, *, train: bool) -> PolicyDataset:
        return PolicyDataset(
            store=self.store,
            observations=self.observations,
            episodes=episodes,
            shape=self.shape,
            relative_actions=self.cfg.task.relative_actions,
            train=train,
        )

    def proprio_widths(self, data: PolicyDataset) -> dict[str, int]:
        probe, _ = data.arrays(np.array([0]), train=False)
        return {key: probe[key].shape[-1] for key in self.cfg.task.obs.proprio}

    def normalizer(self, data: PolicyDataset) -> dict:
        """Min-max over ``data``'s windows, with the task's identity rules."""
        task = self.cfg.task
        return policy_dataset.fit_normalizer(
            data,
            proprio_identity=tuple(task.proprio_identity),
            action_identity_dims=identity_dims(
                len(self.store.spec.arms), list(task.action_identity)
            ),
        )
