"""The proximity cutoff ``tau`` is calibrated over the episodes it is given: the
verifier's cutoff comes from its training episodes only."""

import numpy as np
import torch

from needlework.sampling import proximity


def test_tau_uses_only_the_given_episodes() -> None:
    g = torch.Generator().manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(60, 8, generator=g), dim=1).cuda()
    starts = np.array([0, 20, 40])
    ends = np.array([20, 40, 60])
    frames = 5
    both = proximity.calibrate_tau(emb, starts[:2], ends[:2], frames, percentile=50.0)
    first = proximity.calibrate_tau(emb, starts[:1], ends[:1], frames, percentile=50.0)
    cos = (emb[:15] * emb[5:20]).sum(1).cpu().numpy()
    assert first == float(np.percentile(cos, 50.0))
    assert first != both
    every = proximity.calibrate_tau(emb, starts, ends, frames, percentile=50.0)
    assert every != both
