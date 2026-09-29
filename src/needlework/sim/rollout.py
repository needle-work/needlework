"""Evaluate a policy in simulation on the fixed grid of episodes.

Episode ``i`` of eval seed ``s`` resets with seed ``EPISODE_SEED_STRIDE * s + i`` and
draws its diffusion noise from a generator seeded with the same value, so any two
checkpoints meet the same initial states and outcomes are paired.

Each worker slot runs a fixed queue: episodes ``i`` with ``i % n_envs == slot``, seed by
seed, and slot ``k`` is built with construction seed ``construction_seed + k``, so
construction-time draws (Transport's hammer) and each slot's episode order are fixed.
An episode ends at its first success, and a slot starts its next episode as soon as its
current one ends. Success rate and length at success are reported; episode return is
not.

Every policy call is batched over all ``n_envs`` slots, idle ones included, so the batch
shape (and with it the GPU kernels) never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from needlework.constants import EPISODE_SEED_STRIDE
from needlework.models.dinov3 import Dinov3
from needlework.models.policy import DiffusionPolicy
from needlework.sim.egl import egl_index_for_cuda_ordinal
from needlework.sim.vector_env import SimPool, WorkerSpec


@dataclass(frozen=True)
class EvalGrid:
    seeds: tuple[int, ...]
    episodes_per_seed: int
    n_envs: int
    max_steps: int
    construction_seed: int


def episode_seed(eval_seed: int, index: int) -> int:
    if not 0 <= index < EPISODE_SEED_STRIDE:
        raise ValueError(f"episode index {index} out of range")
    return eval_seed * EPISODE_SEED_STRIDE + index


def slot_queues(grid: EvalGrid) -> list[list[tuple[int, int]]]:
    return [
        [
            (seed, index)
            for seed in grid.seeds
            for index in range(slot, grid.episodes_per_seed, grid.n_envs)
        ]
        for slot in range(grid.n_envs)
    ]


def _policy_obs(
    obs: list[dict[str, np.ndarray]],
    encoder: Dinov3,
    pooling: str,
    policy: DiffusionPolicy,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stacked per-slot observations -> policy inputs; one encoder call for all."""
    cameras = policy.shape.cameras
    images = np.stack([np.stack([o[cam] for o in obs]) for cam in cameras])
    n_cams, n_slots, n_obs = images.shape[:3]
    flat = torch.from_numpy(images.reshape(-1, *images.shape[3:]))
    features = encoder.encode(flat, (pooling,))[pooling]
    features = features.reshape(n_cams, n_slots, n_obs, -1)
    out = {cam: features[i] for i, cam in enumerate(cameras)}
    for key in policy.shape.proprio:
        out[key] = torch.from_numpy(np.stack([o[key] for o in obs])).to(device)
    return out


@torch.inference_mode()
def evaluate(
    policy: DiffusionPolicy,
    encoder: Dinov3,
    *,
    pooling: str,
    grid: EvalGrid,
    hdf5: Path,
    proprio: tuple[str, ...],
    arms: tuple[str, ...],
    device: torch.device,
) -> list[dict]:
    """One record per episode: eval_seed, index, length_at_success (None = failure)."""
    # Render on the GPU this process computes on.
    egl_index = egl_index_for_cuda_ordinal(torch.cuda.current_device())
    specs = [
        WorkerSpec(
            hdf5=hdf5,
            cameras=policy.shape.cameras,
            proprio=proprio,
            arms=arms,
            max_steps=grid.max_steps,
            n_obs=policy.shape.n_obs,
            construction_seed=grid.construction_seed + slot,
            egl_index=egl_index,
        )
        for slot in range(grid.n_envs)
    ]
    queues = slot_queues(grid)
    total = sum(len(queue) for queue in queues)
    current: list[tuple[int, int] | None] = [None] * grid.n_envs
    obs: list[dict | None] = [None] * grid.n_envs
    generators = [torch.Generator(device=device) for _ in range(grid.n_envs)]
    records: list[dict] = []
    pool = SimPool(specs)
    progress = tqdm(total=total, desc="eval episodes", leave=False)

    def start(slots: list[int]) -> None:
        started = [slot for slot in slots if queues[slot]]
        for slot in started:
            current[slot] = queues[slot].pop(0)
            seed = episode_seed(*current[slot])
            generators[slot].manual_seed(seed)
            pool.send(slot, "reset", seed)
        for slot in started:
            obs[slot] = pool.recv(slot)

    try:
        start(list(range(grid.n_envs)))
        while any(episode is not None for episode in current):
            active = [slot for slot in range(grid.n_envs) if current[slot] is not None]
            filler = obs[active[0]]
            batch = [o if o is not None else filler for o in obs]
            inputs = _policy_obs(batch, encoder, pooling, policy, device)
            actions = policy.executable(policy.predict(inputs, generators))
            actions = actions.cpu().numpy()
            for slot in active:
                pool.send(slot, "step", actions[slot])
            ended = []
            for slot in active:
                obs[slot], done, result = pool.recv(slot)
                if done:
                    eval_seed, index = current[slot]
                    records.append({"eval_seed": eval_seed, "index": index, **result})
                    current[slot] = None
                    ended.append(slot)
                    progress.update(1)
            start(ended)
    finally:
        progress.close()
        pool.close()
    return sorted(records, key=lambda r: (r["eval_seed"], r["index"]))


def summarize(records: list[dict]) -> dict[str, float]:
    """Success rate (pooled, per seed, sample std across seeds) and mean length at
    success over successful episodes (absent when there are none)."""
    seeds = sorted({r["eval_seed"] for r in records})
    if len(seeds) < 2:
        raise ValueError("the spread over seeds needs at least two eval seeds")
    per_seed = [
        np.mean(
            [r["length_at_success"] is not None for r in records if r["eval_seed"] == s]
        )
        for s in seeds
    ]
    lengths = [
        r["length_at_success"] for r in records if r["length_at_success"] is not None
    ]
    out = {
        "eval/success_rate": float(
            np.mean([r["length_at_success"] is not None for r in records])
        ),
        "eval/success_rate_std_over_seeds": float(np.std(per_seed, ddof=1)),
        "eval/episodes": float(len(records)),
    }
    out.update(
        {
            f"eval/seed_{s}/success_rate": float(v)
            for s, v in zip(seeds, per_seed, strict=True)
        }
    )
    if lengths:
        out["eval/length_mean_at_success"] = float(np.mean(lengths))
    return out
