"""component=verifier: reached-by-step classifier over (source, goal, action chunk).

Examples come from all training episodes, failures included; ``val_ratio`` of all
episodes is held out (``sampling/verifier_pairs.py``). After every epoch one forward
pass of the EMA model over the validation examples gives ``val_loss`` (balanced BCE over
the full set, not a mean of batch means) and the per-step thresholds calibrated to
``target_fpr`` (``self.thresholds``), which the checkpoints carry for stitching.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from needlework.constants import DINOV3_FEATURE_DIM, PATCHES, POOLED_GRID, SPATIAL
from needlework.data import features
from needlework.models.verifier import Verifier, balanced_bce, calibrate
from needlework.sampling import proximity
from needlework.sampling.verifier_pairs import VerifierPairs
from needlework.training.batches import to_device
from needlework.training.common import Windows


class VerifierTask:
    def __init__(self, cfg: DictConfig, *, device: torch.device, fit: bool) -> None:
        self.cfg, self.device = cfg, device
        comp, task = cfg.component, cfg.task
        horizon = cfg.horizon.prediction - cfg.horizon.obs + 1  # rows judged
        self.windows = w = Windows(cfg, cameras=False, n_execute=cfg.horizon.execute)
        episodes = w.episodes(successes_only=False)
        spatial = features.load(w.store, SPATIAL)
        rows = features.open_rows(w.store, PATCHES)
        patches = {camera: rows[camera] for camera in task.obs.cameras}
        emb = proximity.embeddings(spatial, device)
        tau = proximity.calibrate_tau(  # training episodes only
            emb,
            w.store.episode_starts[episodes.train],
            w.store.episode_ends[episodes.train],
            horizon * task.stride,
            percentile=comp.sampling.proximity.tau_percentile,
        )
        del emb
        sampling = dict(comp.sampling)

        def pairs(episode_ids: np.ndarray, *, train: bool) -> VerifierPairs:
            return VerifierPairs(
                w.dataset(episode_ids, train=train),
                spatial,
                patches,
                sampling,
                tau=tau,
                seed=comp.sampling.seed.train if train else comp.sampling.seed.val,
                resample=train,
                device=device,
            )

        self.train_data = pairs(episodes.train, train=True)
        self.val_data = pairs(episodes.val, train=False)
        train_probe = w.dataset(episodes.train, train=False)
        self.model = Verifier(
            n_cameras=len(patches),
            patch_grid=POOLED_GRID,
            feature_dim=DINOV3_FEATURE_DIM,
            proprio=w.proprio_widths(train_probe),
            action_dim=w.store.spec.action_dim,
            horizon=horizon,
            n_obs=cfg.horizon.obs,
            **comp.model,
        )
        if fit:
            self.model.normalizer.set(w.normalizer(train_probe))
        self.tau = tau
        self.thresholds: torch.Tensor | None = None  # set by every epoch's metrics

    @property
    def identity(self) -> dict[str, str]:
        return self.windows.store.identity

    def loss(self, model: Verifier, batch: dict) -> torch.Tensor:
        return balanced_bce(model(batch), batch["labels"])

    @torch.no_grad()
    def epoch_metrics(self, ema_model: Verifier, epoch: int) -> dict[str, float]:
        comp = self.cfg.component
        logits, labels = [], []
        for start in range(0, len(self.val_data), comp.train.batch_size):
            end = min(start + comp.train.batch_size, len(self.val_data))
            indices = np.arange(start, end)
            batch = to_device(self.val_data.batch(indices), self.device)
            logits.append(ema_model(batch))
            labels.append(batch["labels"])
        logits, labels = torch.cat(logits), torch.cat(labels)
        thresholds = calibrate(logits, labels, **comp.calibration)
        self.thresholds = thresholds.cpu()
        accepted = logits >= thresholds
        positive, negative = labels == 1.0, labels == 0.0
        return {
            "val_loss": float(balanced_bce(logits, labels)),  # over the full set
            "verifier/recall": float(accepted[positive].float().mean()),
            "verifier/fpr": float(accepted[negative].float().mean()),
            "verifier/mean_threshold": float(thresholds.mean()),
            "verifier/tau": self.tau,
            **{
                f"verifier/fallback_{split}_{name}": value
                for split, data in (("train", self.train_data), ("val", self.val_data))
                for name, value in data.fallback.items()
            },
        }
