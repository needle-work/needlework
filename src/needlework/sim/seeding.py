"""Reproducible resets: every random stream a robosuite reset draws from, seeded.

Seeding numpy's and python's global generators is not enough (identically seeded envs
then reset to different states):
- each object-placement sampler holds its own ``np.random.Generator``;
- ``env.rng`` drives the robot's joint initialization noise, and model objects hold it
  by reference (Transport's hammer flips its head with it), so it is reseeded in place;
- a hard reset rebuilds the samplers with fresh generators, so ``_load_model`` is
  wrapped to reseed them after every rebuild.
Streams: ``env.rng`` from ``default_rng([seed, 0])``; the k-th sampler of the placement
tree (breadth first, insertion order) from ``default_rng([seed, k + 1])``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler


def base_env(env: Any) -> Any:
    """The innermost robosuite env. Assignment through a robosuite wrapper lands on the
    wrapper, not the env, so seeds must be written to this object."""
    return env.unwrapped


def placement_samplers(env: Any) -> list[Any]:
    found, queue = [], [base_env(env).placement_initializer]
    while queue:
        node = queue.pop(0)
        found.append(node)
        if isinstance(node, SequentialCompositeSampler):
            queue.extend(node.samplers.values())
    return found


def seed_placement(env: Any, seed: int) -> None:
    for position, sampler in enumerate(placement_samplers(env)):
        sampler.rng = np.random.default_rng([seed, position + 1])


def seed_env_rng(env: Any, seed: int) -> None:
    holder = base_env(env).rng
    holder.bit_generator.state = np.random.default_rng([seed, 0]).bit_generator.state


class PlacementSeedHook:
    """Wraps ``_load_model``: reseeds the rebuilt samplers with the current seed."""

    def __init__(self, env: Any) -> None:
        self.seed: int | None = None
        base = base_env(env)
        load_model = base._load_model

        def load_model_then_seed() -> None:
            load_model()
            if self.seed is not None:
                seed_placement(base, self.seed)

        base._load_model = load_model_then_seed


def seed_everything(env: Any, hook: PlacementSeedHook, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    hook.seed = seed
    seed_env_rng(env, seed)
    seed_placement(env, seed)
