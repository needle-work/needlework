"""DINOv3 feature caches, one per (task, outcome, pooling), outside the data stores.

    $NEEDLEWORK_ROOT/cache/features/<domain>/<task>/<outcome>/<pooling>/
        <camera>.npy     float32 [T, *pooling shape], row t = frame t of that store
        identity.json    what the features were computed from

A cache whose identity does not match its store and encoder is rejected, never reused.
Arrays are plain ``.npy`` files so large ones (patch grids: 150 KB per frame and camera)
are read row by row through a memory map instead of being loaded.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

from needlework import paths
from needlework.constants import DINOV3_SOURCE_COMMIT, DINOV3_WEIGHTS_SHA256
from needlework.data.store import EpisodeStore
from needlework.models.dinov3 import POOLINGS, Dinov3

IDENTITY = "identity.json"


def cache_path(store: EpisodeStore, outcome: str, pooling: str) -> Path:
    root = paths.cache_dir() / "features" / store.domain / store.task
    return root / outcome / pooling


def expected_identity(store: EpisodeStore, outcome: str, pooling: str) -> dict:
    return {
        "store_sha256": store.identity[outcome],
        "dinov3_source_commit": DINOV3_SOURCE_COMMIT,
        "dinov3_weights_sha256": DINOV3_WEIGHTS_SHA256,
        "pooling": pooling,
        "cameras": list(store.spec.cameras),
    }


def _decode(
    images: zarr.Array, start: int, stop: int, pool: ThreadPoolExecutor
) -> np.ndarray:
    """Frames [start, stop) as uint8; one JPEG-XL chunk per frame, threaded."""
    return np.stack(list(pool.map(images.__getitem__, range(start, stop))))


def build(
    store: EpisodeStore,
    poolings: tuple[str, ...],
    *,
    device: torch.device,
    batch_size: int,
    decode_threads: int,
) -> None:
    """Compute every missing cache for ``poolings``; existing ones must match."""
    pending = []
    for outcome in store.store_paths:
        for pooling in poolings:
            path = cache_path(store, outcome, pooling)
            if path.exists():
                _check(path, expected_identity(store, outcome, pooling))
                print(f"[ok]    {path}")
            else:
                pending.append((outcome, pooling))
    if not pending:
        return
    encoder = Dinov3(device)
    for outcome in dict.fromkeys(outcome for outcome, _ in pending):
        wanted = tuple(pooling for o, pooling in pending if o == outcome)
        source = zarr.open(str(store.store_paths[outcome]), mode="r")
        n_frames = int(source["meta/episode_ends"][-1])
        tmps = {}
        for pooling in wanted:
            tmp = cache_path(store, outcome, pooling).with_suffix(".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            tmps[pooling] = tmp
        with ThreadPoolExecutor(decode_threads) as pool:
            for camera in store.spec.cameras:
                outputs = {
                    pooling: np.lib.format.open_memmap(
                        tmps[pooling] / f"{camera}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n_frames, *POOLINGS[pooling]),
                    )
                    for pooling in wanted
                }
                images = source["data"][camera]
                for start in tqdm(
                    range(0, n_frames, batch_size), desc=f"{outcome}/{camera}"
                ):
                    stop = min(start + batch_size, n_frames)
                    batch = torch.from_numpy(_decode(images, start, stop, pool))
                    features = encoder.encode(batch, wanted)
                    for pooling, array in outputs.items():
                        array[start:stop] = features[pooling].cpu().numpy()
                for array in outputs.values():
                    array.flush()
        for pooling, tmp in tmps.items():
            identity = expected_identity(store, outcome, pooling)
            (tmp / IDENTITY).write_text(json.dumps(identity, indent=1) + "\n")
            tmp.rename(cache_path(store, outcome, pooling))
            print(f"[built] {cache_path(store, outcome, pooling)}")


def _check(path: Path, identity: dict) -> None:
    actual = json.loads((path / IDENTITY).read_text())
    if actual != identity:
        raise ValueError(
            f"{path} was computed from different inputs: {actual} != {identity}. "
            "Delete it and rebuild."
        )


def _memmaps(store: EpisodeStore, pooling: str) -> dict[str, list[np.ndarray]]:
    """{camera: [one read-only memory map per outcome, in episode-view order]}."""
    out: dict[str, list[np.ndarray]] = {camera: [] for camera in store.spec.cameras}
    for outcome in store.store_paths:
        path = cache_path(store, outcome, pooling)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run python -m needlework.build_features for "
                f"{store.domain}/{store.task} with pooling {pooling}."
            )
        _check(path, expected_identity(store, outcome, pooling))
        for camera in store.spec.cameras:
            out[camera].append(np.load(path / f"{camera}.npy", mmap_mode="r"))
    for camera, parts in out.items():
        rows = sum(part.shape[0] for part in parts)
        if rows != store.n_frames:
            raise ValueError(f"{camera}: {rows} rows != {store.n_frames}")
    return out


def load(store: EpisodeStore, pooling: str) -> dict[str, np.ndarray]:
    """Features over the store's episode view, in memory: {camera: float32 [T, ...]}."""
    memmaps = _memmaps(store, pooling)
    return {camera: np.concatenate(parts) for camera, parts in memmaps.items()}


class Rows:
    """Rows of one camera's cache over the episode view, read on demand."""

    def __init__(self, parts: list[np.ndarray]) -> None:
        self.parts = parts
        self.starts = np.cumsum([0] + [part.shape[0] for part in parts])

    def __getitem__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        flat = rows.reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.starts[-1]):
            raise IndexError(
                f"feature rows {flat.min()}..{flat.max()} outside "
                f"[0, {self.starts[-1]})"
            )
        out = np.empty((flat.size, *self.parts[0].shape[1:]), dtype=np.float32)
        part_of = np.searchsorted(self.starts, flat, side="right") - 1
        for index, part in enumerate(self.parts):
            mask = part_of == index
            out[mask] = part[flat[mask] - self.starts[index]]
        return out.reshape(*rows.shape, *out.shape[1:])


def open_rows(store: EpisodeStore, pooling: str) -> dict[str, Rows]:
    """Features over the store's episode view, read lazily: {camera: Rows}."""
    return {camera: Rows(parts) for camera, parts in _memmaps(store, pooling).items()}
