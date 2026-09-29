"""Build the released dataset zips from a store holding success and failure episodes.

For each outcome, writes ``<domain>_<task>_<outcome>.zarr.zip`` to ``--out-dir``
containing ``<outcome>.zarr`` in the format of ``needlework.data.schema``, and records
its sha256, size, and counts in the dataset table (``--table``, JSON).

Episodes keep their order in the source. Image chunks (one frame each) are hard-linked
under their new frame index, so image bytes are copied exactly, never re-encoded. Only
the arrays the schema lists are kept. UMI sources have no action array; it is built here
from the logged gripper poses (``action[t]`` = pose and width at frame ``t``).

Usage:
    python tools/build_dataset_zips.py --source /path/to/combined.zarr \
        --domain robomimic --task can --out-dir /path/to/release \
        --table src/needlework/data/datasets.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc, blosc
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from needlework.constants import OUTCOMES
from needlework.data.schema import (
    StoreSpec,
    spec_for,
    store_attrs,
    validate_store,
)

NUMERIC_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
# Threaded Blosc writes a chunk's blocks in completion order. With one thread the
# bytes, and so each archive's sha256, depend only on the data.
blosc.use_threads = False

_READ_BYTES = 1 << 22
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _frames_of(ends: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    starts = np.r_[0, ends[:-1]]
    return np.concatenate([np.arange(starts[e], ends[e]) for e in episodes])


def _link_image_chunks(
    src_dir: Path, dst_dir: Path, frames: np.ndarray, shape: tuple[int, ...]
) -> None:
    meta = json.loads((src_dir / ".zarray").read_text())
    if meta["chunks"][0] != 1:
        raise ValueError(f"{src_dir}: images must be chunked one frame per chunk")
    sep = "."  # zarr v2 omits the key for its default separator
    if "dimension_separator" in meta:
        sep = meta["dimension_separator"]
    if sep not in (".", "/"):
        raise ValueError(f"{src_dir}: unknown zarr dimension separator {sep!r}")
    suffix = sep.join(["0"] * (len(shape) - 1))
    meta["shape"] = [len(frames), *shape[1:]]
    dst_dir.mkdir(parents=True)
    (dst_dir / ".zarray").write_text(json.dumps(meta, indent=4))
    for new, old in enumerate(tqdm(frames, desc=dst_dir.name, leave=False)):
        src = src_dir / f"{old}{sep}{suffix}"
        dst = dst_dir / f"{new}{sep}{suffix}"
        if sep == "/":
            dst.parent.mkdir(parents=True, exist_ok=True)
        # Every frame must have its chunk in the source store; a missing one fails
        # here and never becomes a fill-value frame.
        os.link(src, dst)


def _umi_action(source: zarr.Group, spec: StoreSpec) -> np.ndarray:
    """Per arm: position | rotation_6d (rows, from the logged axis-angle) | width."""
    blocks = []
    for side in spec.arms:
        rotvec = source[f"data/gripper_{side}_eef_rot_axis_angle"][:].astype(np.float64)
        rows = Rotation.from_rotvec(rotvec).as_matrix()[:, :2, :].reshape(-1, 6)
        blocks += [
            source[f"data/gripper_{side}_eef_pos"][:],
            rows,
            source[f"data/gripper_{side}_gripper_width"][:],
        ]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_store(
    source: zarr.Group,
    source_path: Path,
    dest: Path,
    *,
    domain: str,
    task: str,
    outcome: str,
) -> int:
    spec = spec_for(domain, task)
    ends = source["meta/episode_ends"][:]
    success = source["meta/episode_success"][:].astype(bool)
    episodes = np.flatnonzero(success if outcome == "success" else ~success)
    if len(episodes) == 0:
        raise ValueError(f"{source_path}: no {outcome} episodes")
    frames = _frames_of(ends, episodes)
    lengths = np.diff(np.r_[0, ends])[episodes]

    root = zarr.open_group(str(dest), mode="w")
    root.attrs.update(store_attrs(domain, task))
    meta = root.create_group("meta")
    meta.array("episode_ends", np.cumsum(lengths).astype(np.int64), compressor=None)
    meta.array(
        "episode_success",
        np.full(len(episodes), outcome == "success"),
        compressor=None,
    )
    data = root.create_group("data")
    for key in spec.data_widths():
        if key == "action" and domain == "umi":
            values = _umi_action(source, spec)
            chunks = (2048, spec.action_dim)
        else:
            values = source[f"data/{key}"][:]
            chunks = source[f"data/{key}"].chunks
        data.array(
            key,
            values[frames].astype(np.float32, copy=False),
            chunks=chunks,
            compressor=NUMERIC_COMPRESSOR,
        )
    for key in spec.cameras:
        _link_image_chunks(
            source_path / "data" / key,
            dest / "data" / key,
            frames,
            source[f"data/{key}"].shape,
        )
    num_episodes, num_success = validate_store(dest, domain=domain, task=task)
    assert num_success == (num_episodes if outcome == "success" else 0)
    return num_episodes


def zip_store(store: Path, zip_path: Path) -> str:
    """Zip ``store`` uncompressed, rooted at ``store.name``; return the sha256."""
    files = sorted(p for p in store.rglob("*") if p.is_file())
    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in tqdm(files, desc=zip_path.name, leave=False):
            # Fixed timestamp and mode: the same store always gives the same bytes.
            info = zipfile.ZipInfo(str(path.relative_to(store.parent)), _ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            with (
                path.open("rb") as src,
                archive.open(info, "w", force_zip64=True) as dst,
            ):
                shutil.copyfileobj(src, dst, _READ_BYTES)
    digest = hashlib.sha256()
    with part.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BYTES), b""):
            digest.update(block)
    part.rename(zip_path)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--domain", choices=["robomimic", "umi"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    source = zarr.open_group(str(args.source), mode="r")
    stage = args.out_dir / "stage"
    table = json.loads(args.table.read_text()) if args.table.exists() else {}
    for outcome in OUTCOMES:
        name = f"{args.domain}_{args.task}_{outcome}.zarr.zip"
        store = stage / f"{args.domain}_{args.task}" / f"{outcome}.zarr"
        if store.exists():
            raise FileExistsError(f"{store} exists from an earlier run; remove it.")
        num_episodes = build_store(
            source,
            args.source,
            store,
            domain=args.domain,
            task=args.task,
            outcome=outcome,
        )
        zip_path = args.out_dir / name
        sha256 = zip_store(store, zip_path)
        num_frames = int(zarr.open(str(store), mode="r")["meta/episode_ends"][-1])
        shutil.rmtree(store)
        table[name] = {
            "domain": args.domain,
            "task": args.task,
            "outcome": outcome,
            "num_episodes": num_episodes,
            "num_frames": num_frames,
            "size": zip_path.stat().st_size,
            "sha256": sha256,
        }
        print(f"{name}: {num_episodes} episodes, {num_frames} frames, {sha256}")
    shutil.rmtree(stage)
    args.table.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n")


if __name__ == "__main__":
    main()
