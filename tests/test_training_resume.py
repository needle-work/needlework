"""Resume is exact: a run killed mid-epoch and resumed ends bit-identical to one that
was never interrupted (metrics and every model and EMA tensor). About 3 minutes on one
GPU; the two runs write about 13 GB of checkpoints under $NEEDLEWORK_ROOT/outputs,
deleted at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from needlework import paths

TRAIN = (
    Path(__file__).resolve().parents[2]
    / "needlework"
    / "src"
    / "needlework"
    / "train.py"
)
ARGS = [
    "component=policy",
    "task=robomimic/square",
    "task.epochs.policy=4",
    "checkpoint.every=2",
    "logging.wandb.enabled=false",
    "component.validation.action_mse_every=1",
]


def _run_dir(log: str) -> Path:
    for line in log.splitlines():
        if line.startswith("run directory: "):
            return Path(line.removeprefix("run directory: "))
    raise AssertionError(f"no run directory in:\n{log}")


def _metrics(run_dir: Path) -> list[dict]:
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    return [{k: v for k, v in r.items() if not k.startswith("time/")} for r in records]


def test_resume_is_exact(tmp_path: Path) -> None:
    env = {**os.environ, "NEEDLEWORK_ROOT": str(paths.root())}
    a = subprocess.run(
        [sys.executable, str(TRAIN), *ARGS, "run.name=resume_test_a"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    run_a = _run_dir(a.stdout)
    log_b = tmp_path / "b.log"
    with log_b.open("w") as handle:
        b = subprocess.Popen(
            [sys.executable, str(TRAIN), *ARGS, "run.name=resume_test_b"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while "after epoch 2" not in log_b.read_text():
            assert b.poll() is None, log_b.read_text()
            time.sleep(1)
        time.sleep(5)  # into epoch 3
        b.send_signal(signal.SIGKILL)
        b.wait()
    run_b = _run_dir(log_b.read_text())
    subprocess.run(
        [sys.executable, str(TRAIN), "--resume", str(run_b)],
        env=env,
        capture_output=True,
        check=True,
    )
    try:
        assert _metrics(run_a) == _metrics(run_b)
        ckpt_a = torch.load(run_a / "checkpoints/last.ckpt", weights_only=True)
        ckpt_b = torch.load(run_b / "checkpoints/last.ckpt", weights_only=True)
        for part in ("model", "ema"):
            for key, value in ckpt_a[part].items():
                assert torch.equal(value, ckpt_b[part][key]), (part, key)
    finally:
        shutil.rmtree(run_a)
        shutil.rmtree(run_b)
