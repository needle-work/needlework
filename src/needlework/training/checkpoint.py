"""Checkpoint files of a run directory.

    checkpoints/last.ckpt    full resume state (weights, EMA, optimizer, scheduler,
                             RNG), every ``checkpoint.every`` epochs and after the last
    checkpoints/epoch_NNNN.ckpt, selected.json
                             policy only: ``checkpoint.policy_snapshots`` EMA
                             snapshots, the best rollout scores (Robomimic) or the
                             newest epochs (UMI)

Every ``last.ckpt`` carries ``constants`` (``needlework.constants.snapshot()``), which
a resume must match. A verifier's ``last.ckpt`` also carries ``thresholds``, calibrated
on its validation logits at the end of each epoch; they are not model weights, so the
EMA never touches them. Inference (stitching) reads ``last.ckpt`` through
``load_inference``, which memory-maps the file and keeps only the EMA weights, config,
data identity, epoch and thresholds, so optimizer state is never read into memory.

Payloads hold only tensors and plain values, so they load with ``weights_only=True``.
Writes go to a temporary sibling that is fsynced and then renamed over the old file, so
an interrupted save leaves the previous file intact.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

FORMAT = 1
NAME = "last.ckpt"
SELECTED = "selected.json"
SNAPSHOT_KEYS = ("epoch", "ema", "config", "data_identity")  # policy snapshots


def path_in(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / NAME


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save({"format": FORMAT, **payload}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save(run_dir: Path, payload: dict) -> Path:
    target = path_in(run_dir)
    _write(target, payload)
    return target


def load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT:
        raise ValueError(f"{path}: checkpoint format {payload['format']} != {FORMAT}")
    return payload


def load_inference(path: Path) -> dict:
    """What inference uses from a ``last.ckpt``: epoch, EMA weights, config, data
    identity and, for a verifier, its thresholds. The file is memory-mapped and nothing
    else is kept, so the online weights and optimizer state are never read."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload["format"] != FORMAT:
        raise ValueError(f"{path}: checkpoint format {payload['format']} != {FORMAT}")
    keys = SNAPSHOT_KEYS
    if payload["config"]["component"]["name"] == "verifier":
        if "thresholds" not in payload:
            raise ValueError(f"{path}: verifier checkpoint has no thresholds")
        keys += ("thresholds",)
    return {key: payload[key] for key in keys}


def retain_policy(run_dir: Path, payload: dict, metrics: dict) -> None:
    """``checkpoint.policy_snapshots`` policy EMA snapshots: best rollout scores (sim),
    newest epochs (UMI). Ties keep the earlier epoch."""
    cfg = payload["config"]
    if cfg["component"]["name"] != "policy":
        return
    domain = cfg["task"]["domain"]
    if domain == "robomimic" and "eval/success_rate" not in metrics:
        return
    epoch = int(payload["epoch"])
    score = (
        float(metrics["eval/success_rate"]) if domain == "robomimic" else float(epoch)
    )
    if not math.isfinite(score):
        raise ValueError("checkpoint selection score must be finite")
    index_path = run_dir / "checkpoints" / SELECTED
    entries = json.loads(index_path.read_text()) if index_path.exists() else []
    entry = {"epoch": epoch, "score": score, "file": f"epoch_{epoch:04d}.ckpt"}
    entries = [e for e in entries if e["epoch"] != epoch] + [entry]
    ranked = sorted(entries, key=lambda e: (-e["score"], e["epoch"]))
    keep = cfg["checkpoint"]["policy_snapshots"]
    kept = ranked[:keep]
    if entry in kept:
        snapshot = {key: payload[key] for key in SNAPSHOT_KEYS}
        _write(index_path.parent / entry["file"], snapshot)
    tmp_index = index_path.with_suffix(".tmp")
    tmp_index.write_text(json.dumps(kept, indent=2) + "\n")
    os.replace(tmp_index, index_path)
    for entry in ranked[keep:]:
        (index_path.parent / entry["file"]).unlink(missing_ok=True)
