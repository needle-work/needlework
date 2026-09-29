"""Run directories: created once, resumed in place, however many launches a run takes.

outputs/<domain>/<task>/<component>/<YYYY-MM-DD_HH-MM-SS>_<name>/
    config.yaml        fully resolved config
    metrics.jsonl      one record per epoch
    logs/<launch>.log  stdout and stderr of each launch
    checkpoints/last.ckpt  (see training/checkpoint.py)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from needlework import paths

CONFIG = "config.yaml"
METRICS = "metrics.jsonl"


def new_run_dir(domain: str, task: str, component: str, name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = paths.output_dir() / domain / task / component / f"{stamp}_{name}"
    run_dir.mkdir(parents=True)
    return run_dir


def truncate_metrics(run_dir: Path, completed_epochs: int) -> None:
    """Drop records of epochs a resumed run is about to redo."""
    path = run_dir / METRICS
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    kept = [r for r in records if r["epoch"] < completed_epochs]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, path)


def append_metrics(run_dir: Path, record: dict) -> None:
    with (run_dir / METRICS).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


class _Tee:
    """Writes to every stream; anything else (``isatty``, ``fileno``, ``encoding``)
    is answered by the first, the terminal."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def __getattr__(self, name: str) -> object:
        return getattr(self.streams[0], name)

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def tee_output(run_dir: Path) -> None:
    """Copy this launch's stdout and stderr into ``logs/<launch time>.log``."""
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    handle = (logs / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log").open("a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
