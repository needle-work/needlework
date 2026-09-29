"""Bounded dynamic multi-GPU scoring with atomic, GPU-count-independent resume."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from needlework.constants import QUEUE_DEPTH, REPLY_TIMEOUT_S, SHUTDOWN_TIMEOUT_S
from needlework.stitching.inference import Scorer


def batch_digest(rows: np.ndarray) -> str:
    return hashlib.sha256(rows.tobytes()).hexdigest()


def write_result(path: Path, rows: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    """Write one batch's scores atomically, tagged with its candidates' digest."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, identity=np.array(batch_digest(rows)), **scores)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_result(path: Path, rows: np.ndarray) -> dict[str, np.ndarray]:
    """One batch's logits [B, samples, H], actions [B, samples, H, action dim] and
    max_length [B], checked against the batch's candidates."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["identity"]) != batch_digest(rows):
            raise ValueError(f"{path}: result belongs to different candidates")
        scores = {k: data[k] for k in ("logits", "actions", "max_length")}
    logits, actions, max_length = (
        scores[k] for k in ("logits", "actions", "max_length")
    )
    if (
        logits.ndim != 3
        or len(logits) != len(rows)
        or actions.ndim != 4
        or actions.shape[:3] != logits.shape
        or max_length.shape != (len(rows),)
        or np.any((max_length < 0) | (max_length > logits.shape[2]))
        or not np.isfinite(logits).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"{path}: malformed batch result")
    return scores


def batch_rows(candidates: np.ndarray, batch: int, batch_size: int) -> np.ndarray:
    return candidates[batch * batch_size : (batch + 1) * batch_size]


def _worker(
    gpu: int,
    config: dict,
    candidate_path: Path,
    directory: Path,
    tasks: mp.Queue,
    results: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        torch.use_deterministic_algorithms(True)
        cfg = OmegaConf.create(config)
        scorer = Scorer(cfg, torch.device("cuda", gpu))
        candidates = np.load(candidate_path, mmap_mode="r")
        while True:
            batch = tasks.get()
            if batch is None:
                return
            rows = batch_rows(candidates, batch, cfg.batch_size)
            write_result(directory / f"{batch:06d}.npz", rows, scorer.score(rows))
            results.put(("done", batch), timeout=REPLY_TIMEOUT_S)
    except Exception:
        # a coordinator that stopped reading cannot hold this worker past the timeout
        results.put(("error", traceback.format_exc()), timeout=REPLY_TIMEOUT_S)


def stop_workers(procs: list, gpus: list[int], *, grace_s: float) -> None:
    """Wait for workers that were sent the stop signal. One still running after
    ``grace_s`` is killed and reported as hung; one that exited nonzero is reported with
    its exit code. A slow but clean exit within the grace period is not an error."""
    for gpu, proc in zip(gpus, procs, strict=True):
        proc.join(timeout=grace_s)
        if proc.is_alive():
            proc.kill()
            proc.join()
            raise RuntimeError(
                f"stitch worker on GPU {gpu} still running after {grace_s} s "
                "once its work was done; killed"
            )
        if proc.exitcode != 0:
            raise RuntimeError(
                f"stitch worker on GPU {gpu} exited with code {proc.exitcode}"
            )


def score_batches(cfg: DictConfig, candidate_path: Path, directory: Path) -> list[Path]:
    """Score every candidate batch on ``cfg.gpus`` (validated by ``stitch.check``) and
    return the batch files in candidate order."""
    candidates = np.load(candidate_path, mmap_mode="r")
    directory.mkdir(exist_ok=True)
    count = (len(candidates) + cfg.batch_size - 1) // cfg.batch_size
    paths = [directory / f"{batch:06d}.npz" for batch in range(count)]
    pending = []
    for batch, path in enumerate(paths):
        if path.exists():
            read_result(path, batch_rows(candidates, batch, cfg.batch_size))
        else:
            pending.append(batch)
    if pending:
        _run_workers(cfg, candidate_path, directory, pending)
    return paths


def _run_workers(
    cfg: DictConfig, candidate_path: Path, directory: Path, pending: list[int]
) -> None:
    ctx = mp.get_context("spawn")
    gpus = list(cfg.gpus)[: len(pending)]
    depth = QUEUE_DEPTH * len(gpus)
    tasks, results = ctx.Queue(maxsize=depth), ctx.Queue(maxsize=depth)
    procs = []
    progress = tqdm(total=len(pending), desc="stitch batches")
    completed: set[int] = set()
    try:
        for gpu in gpus:
            proc = ctx.Process(
                target=_worker,
                args=(
                    gpu,
                    OmegaConf.to_container(cfg, resolve=True),
                    candidate_path,
                    directory,
                    tasks,
                    results,
                ),
            )
            proc.start()
            procs.append(proc)
        submitted = min(depth, len(pending))
        for batch in pending[:submitted]:
            tasks.put(batch)
        last_reply = time.monotonic()
        while len(completed) < len(pending):
            try:
                status, payload = results.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode is not None for p in procs):
                    raise RuntimeError(
                        "stitch worker exited without completing queued work"
                    ) from None
                if time.monotonic() - last_reply > REPLY_TIMEOUT_S:
                    raise TimeoutError(
                        f"no stitch worker replied within {REPLY_TIMEOUT_S} s"
                    ) from None
                continue
            last_reply = time.monotonic()
            if status == "error":
                raise RuntimeError(f"stitch worker failed:\n{payload}")
            if payload not in pending[:submitted] or payload in completed:
                raise RuntimeError(f"unexpected batch completion {payload}")
            completed.add(payload)
            progress.update(1)
            if submitted < len(pending):
                tasks.put(pending[submitted])
                submitted += 1
        for _ in procs:
            tasks.put(None)
        stop_workers(procs, gpus, grace_s=SHUTDOWN_TIMEOUT_S)
    finally:
        progress.close()
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=SHUTDOWN_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
                proc.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.cancel_join_thread()
        results.close()
