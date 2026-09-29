"""Config-driven stitching and input-checked batch resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from needlework import constants
from needlework.data import bridges
from needlework.data.download import sha256_of
from needlework.data.store import EpisodeStore
from needlework.stitching.candidates import generate
from needlework.stitching.inference import check_compatibility, load_component
from needlework.stitching.selection import (
    choose_prefixes,
    keep_fraction,
    keep_top_fraction,
    margins,
    pack,
    threshold_offset,
)
from needlework.stitching.workers import batch_rows, read_result, score_batches

# Settings that define the candidates and their scores. Checkpoints enter through
# their content hashes; gpus, run.name and the settings applied after scoring
# (threshold_offsets, keep_fraction) may change on resume.
CONTRACT = (
    "seed",
    "stages",
    "sources_per_stage",
    "candidate_spacing",
    "targets_per_source",
    "max_candidates_per_stage",
    "batch_size",
    "min_steps_saved",
    "proposals",
    "min_votes",
    "recovery_quantile",
    "proximity",
)
NOT_CONTRACT = (
    "task",
    "idm_checkpoint",
    "verifier_checkpoint",
    "gpus",
    "run",
    "threshold_offsets",
    "keep_fraction",
)
TASK_CONTRACT = ("domain", "name", "stride", "relative_actions", "obs")


def contract_config(resolved: dict) -> dict:
    """The resume contract of a resolved stitch config; every key is classified."""
    unclassified = sorted(set(resolved) - set(CONTRACT) - set(NOT_CONTRACT))
    if unclassified:
        raise ValueError(
            f"stitch config keys outside the resume contract: {unclassified}"
        )
    return {
        **{k: resolved[k] for k in CONTRACT},
        "task": {k: resolved["task"][k] for k in TASK_CONTRACT},
    }


def check_resume(path: Path, identity: dict) -> None:
    """Record a new run's inputs at ``path``, or refuse a resume whose inputs differ,
    naming every differing top-level key and constant."""
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(identity, indent=2) + "\n")
        tmp.replace(path)
        return
    saved = json.loads(path.read_text())
    differ = constants.differing_keys(saved, identity)
    if not differ:
        return
    if "constants" in differ:
        differ += [
            f"constants.{k}"
            for k in constants.differing_keys(saved["constants"], identity["constants"])
        ]
    raise ValueError(f"stitch resume inputs differ from the original run: {differ}")


def run(cfg: DictConfig, directory: Path) -> Path:
    store = EpisodeStore.open(cfg.task.domain, cfg.task.name)
    # Check final checkpoint contracts even when there are no candidates to score.
    device = torch.device("cuda", cfg.gpus[0])
    idm, model_cfg, _ = load_component(Path(cfg.idm_checkpoint), "idm", store, device)
    verifier, verifier_cfg, thresholds = load_component(
        Path(cfg.verifier_checkpoint), "verifier", store, device
    )
    check_compatibility(model_cfg, verifier_cfg, cfg)
    horizon = model_cfg.horizon.prediction - model_cfg.horizon.obs + 1
    thresholds = thresholds.cpu().numpy()
    del idm, verifier
    torch.cuda.empty_cache()
    identity = {
        "config": contract_config(OmegaConf.to_container(cfg, resolve=True)),
        "data": store.identity,
        "idm_sha256": sha256_of(Path(cfg.idm_checkpoint)),
        "verifier_sha256": sha256_of(Path(cfg.verifier_checkpoint)),
        "constants": constants.snapshot(),
    }
    check_resume(directory / "inputs.json", identity)
    candidate_path = directory / "candidates.npy"
    if not candidate_path.exists():
        rows = generate(
            store, cfg, horizon=horizon, stride=cfg.task.stride, device=device
        )
        with candidate_path.with_suffix(".tmp").open("wb") as handle:
            np.save(handle, rows)
        candidate_path.with_suffix(".tmp").replace(candidate_path)
        del rows
        torch.cuda.empty_cache()
    output = directory / "bridges.zarr"
    if output.exists():
        bridges.load(output, store, horizon=horizon, stride=cfg.task.stride)
        print(f"already complete: {output}", flush=True)
        return output
    paths = score_batches(cfg, candidate_path, directory / "batches")
    rows = np.load(candidate_path)
    selected, summary = select(
        cfg, rows, paths, thresholds, store.spec.action_dim, store.episode_ends
    )
    bridges.save(
        output,
        selected,
        store,
        horizon=horizon,
        stride=cfg.task.stride,
        recipe=identity,
    )
    summary["lengths"] = {
        str(h): int((selected.action_len == h).sum()) for h in range(1, horizon + 1)
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(summary, flush=True)
    print(f"bridge store: {output}", flush=True)
    return output


def select(
    cfg: DictConfig,
    rows: np.ndarray,
    paths: list[Path],
    thresholds: np.ndarray,
    action_dim: int,
    episode_ends: np.ndarray,
) -> tuple[bridges.Bridges, dict]:
    """Apply stitching/selection.py to the scored batches, one batch in memory at a
    time: acceptance and prefix choice at each stage's thresholds, packing, paring."""
    offsets = {s: threshold_offset(cfg, s) for s in cfg.stages}
    records = []
    for i, path in enumerate(paths):
        batch = batch_rows(rows, i, cfg.batch_size)
        scores = read_result(path, batch)
        offset = np.array([offsets[s] for s in batch[:, 3].tolist()], np.float32)
        m = margins(scores["logits"], thresholds, scores["max_length"])
        for j, sample, length, margin in choose_prefixes(
            m - offset[:, None, None], min_votes=cfg.min_votes
        ):
            candidate, source, target, stage = batch[j].tolist()
            end = episode_ends[np.searchsorted(episode_ends, [source, target], "right")]
            records.append(
                {
                    "candidate_id": candidate,
                    "source": source,
                    "target": target,
                    "stage": stage,
                    "margin": margin,
                    # raw frames the route saves: source's remainder minus the bridge
                    # and the target's remainder
                    "saved": int(end[0] - source - cfg.task.stride * length)
                    - int(end[1] - target),
                    "actions": scores["actions"][j, sample, :length].copy(),
                }
            )
    packed = pack(records, action_dim)
    fractions = {s: keep_fraction(cfg, s) for s in cfg.stages}
    selected = keep_top_fraction(packed, fractions)
    stages = [str(s) for s in cfg.stages]
    summary = {
        "candidates": len(rows),
        "threshold_offsets": {str(s): o for s, o in offsets.items()},
        "accepted_by_stage": {
            k: int(sum(r["stage"] == int(k) for r in records)) for k in stages
        },
        "accepted_candidates": len(records),
        "packed_bridges": len(packed),
        "packed_by_stage": {k: int((packed.stage == int(k)).sum()) for k in stages},
        "keep_fraction": {str(s): f for s, f in fractions.items()},
        "selected_bridges": len(selected),
        "by_stage": {k: int((selected.stage == int(k)).sum()) for k in stages},
    }
    return selected, summary
