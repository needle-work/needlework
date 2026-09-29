"""Prepare the datasets under ``$NEEDLEWORK_ROOT/data``.

1. Downloads the robomimic v1.5 multi-human low-dim HDF5 of each Robomimic task from
   Hugging Face (pinned revision) to ``robomimic/<task>/low_dim_v15.hdf5``.
2. Unpacks the released store archives, which you download separately into
   ``$NEEDLEWORK_ROOT/data/downloads/`` (see README), to
   ``<domain>/<task>/success.zarr`` and ``<domain>/<task>/failure.zarr``.

Every file is checked against its pinned sha256 before it is used, and every unpacked
store is checked against ``needlework.data.schema``.

Usage:
    python -m needlework.data.download                        # everything
    python -m needlework.data.download robomimic/square umi/sweater
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from needlework import paths
from needlework.data.schema import validate_store

HDF5_REPO = "robomimic/robomimic_datasets"
HDF5_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
HDF5_NAME = "low_dim_v15.hdf5"
_READ_BYTES = 1 << 22


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size: int
    sha256: str


def _hdf5(task: str, size: int, sha256: str) -> RemoteFile:
    url = (
        f"https://huggingface.co/datasets/{HDF5_REPO}/resolve/{HDF5_REVISION}/"
        f"v1.5/{task}/mh/{HDF5_NAME}"
    )
    return RemoteFile(url, size, sha256)


HDF5_FILES: dict[str, RemoteFile] = {
    "can": _hdf5(
        "can",
        112_792_088,
        "c4a34c837913446d295c6ca38f2b31888d69812198478af470ecb84a99e9f7d7",
    ),
    "square": _hdf5(
        "square",
        123_300_120,
        "da7dee0b6da49feae81dbde8fc6508660788e2131098f72ca44f60528405172d",
    ),
    "transport": _hdf5(
        "transport",
        621_187_696,
        "b54d650e1048b132f160b349b3b36bb9e008ddb4093913700bbee422c6ae4347",
    ),
}


def store_zips() -> dict[str, dict]:
    """The released store zips, as written by tools/build_dataset_zips.py."""
    text = resources.files("needlework.data").joinpath("datasets.json").read_text()
    return json.loads(text)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, remote: RemoteFile) -> None:
    actual = sha256_of(path)
    if actual != remote.sha256:
        raise ValueError(
            f"{path} has sha256 {actual}, expected {remote.sha256}. "
            "Delete or move it aside and re-run."
        )


def download_hdf5(remote: RemoteFile, dest: Path) -> None:
    """Ensure ``dest`` holds ``remote``: verify an existing file or download it.

    Downloads go to a ``.part`` sibling that is renamed only after verification, so
    ``dest`` is never partial or unverified.
    """
    if dest.exists():
        _verify(dest, remote)
        print(f"[ok]       {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with (
        urllib.request.urlopen(remote.url) as response,
        part.open("wb") as out,
        tqdm(total=remote.size, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for block in iter(lambda: response.read(_READ_BYTES), b""):
            out.write(block)
            bar.update(len(block))
    _verify(part, remote)
    part.rename(dest)
    print(f"[download] {dest}")


def _safe_members(archive: zipfile.ZipFile, root: str) -> list[zipfile.ZipInfo]:
    """Reject entries that would land outside ``root/`` (absolute paths, ``..``)."""
    members = archive.infolist()
    for info in members:
        parts = PurePosixPath(info.filename).parts
        if parts[0] != root or ".." in parts or info.filename.startswith("/"):
            raise ValueError(f"Unsafe or unexpected zip entry: {info.filename}")
    return members


def unpack_store(name: str, entry: dict) -> None:
    """Verify ``downloads/<name>`` and unpack it to ``<domain>/<task>/<outcome>.zarr``.

    A ``<outcome>.zarr.sha256`` file next to the store records the archive it came from,
    so re-runs skip it and a store from anywhere else is never overwritten.
    """
    domain, task, outcome = entry["domain"], entry["task"], entry["outcome"]
    zip_path = paths.data_dir() / "downloads" / name
    target = paths.data_dir() / domain / task / f"{outcome}.zarr"
    marker = target.with_name(target.name + ".sha256")
    if target.exists():
        if not marker.exists() or marker.read_text().strip() != entry["sha256"]:
            raise FileExistsError(
                f"{target} exists but was not unpacked from {name}; move it aside."
            )
        validate_store(target, domain=domain, task=task)
        print(f"[ok]       {target}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} is missing. Download {name} into {zip_path.parent} "
            "(see README) and re-run."
        )
    _verify(zip_path, RemoteFile("", entry["size"], entry["sha256"]))
    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = _safe_members(archive, target.name)
        for info in tqdm(members, desc=f"unpack {name}", leave=False):
            archive.extract(info, tmp)
    num_episodes, num_success = validate_store(
        tmp / target.name, domain=domain, task=task
    )
    if num_success != (num_episodes if outcome == "success" else 0):
        raise ValueError(f"{name}: episode_success does not match '{outcome}'")
    (tmp / target.name).rename(target)
    tmp.rmdir()
    marker.write_text(entry["sha256"] + "\n")
    print(f"[unpack]   {target}")


def main() -> None:
    zips = store_zips()
    datasets = sorted({f"{e['domain']}/{e['task']}" for e in zips.values()})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "datasets", nargs="*", help=f"any of {datasets}; default: all of them"
    )
    requested = parser.parse_args().datasets or datasets
    unknown = sorted(set(requested) - set(datasets))
    if unknown:
        parser.error(f"unknown datasets {unknown}; choose from {datasets}")
    for dataset in requested:
        domain, task = dataset.split("/")
        if domain == "robomimic":
            dest = paths.data_dir() / "robomimic" / task / HDF5_NAME
            download_hdf5(HDF5_FILES[task], dest)
        for name, entry in sorted(zips.items()):
            if (entry["domain"], entry["task"]) == (domain, task):
                unpack_store(name, entry)


if __name__ == "__main__":
    main()
