"""Keep failure sources visually outside the successful-demonstration distribution."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from needlework.constants import SIMILARITY_BATCH


def recovery_sources(
    emb: torch.Tensor, episode: np.ndarray, success: np.ndarray, *, quantile: float
) -> np.ndarray:
    """Failure frames at least as far from every success frame as the
    ``quantile`` of success frames' nearest other-episode success frame (cosine
    distance of visual embeddings; no robot or object state)."""
    success_rows = np.flatnonzero(success[episode])
    if len(np.unique(episode[success_rows])) < 2:
        raise ValueError("recovery calibration needs two success episodes")
    reference = emb[torch.as_tensor(success_rows, device=emb.device)]
    reference_episode = torch.as_tensor(episode[success_rows], device=emb.device)
    distances = []
    for start in tqdm(
        range(0, len(reference), SIMILARITY_BATCH), desc="recovery calibration"
    ):
        stop = start + SIMILARITY_BATCH
        sims = reference[start:stop] @ reference.T
        same = reference_episode[start:stop, None] == reference_episode[None, :]
        sims.masked_fill_(same, -torch.inf)
        distances.append((1 - sims.max(1).values).cpu().numpy())
    threshold = float(np.quantile(np.concatenate(distances), quantile))
    failure_rows = np.flatnonzero(~success[episode])
    keep = np.zeros(len(emb), bool)
    for start in tqdm(
        range(0, len(failure_rows), SIMILARITY_BATCH), desc="recovery sources"
    ):
        rows = failure_rows[start : start + SIMILARITY_BATCH]
        sims = emb[torch.as_tensor(rows, device=emb.device)] @ reference.T
        keep[rows] = (1 - sims.max(1).values).cpu().numpy() >= threshold
    print(f"recovery distance threshold={threshold:.6g}, kept={keep.sum()}", flush=True)
    return keep
