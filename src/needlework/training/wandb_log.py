"""Weights & Biases: required credentials, one W&B run per run directory.

The W&B run id is kept in the run directory, so every launch of a multi-day run logs to
the same W&B run. W&B files are written inside the run directory.
"""

from __future__ import annotations

from pathlib import Path

import wandb
from wandb.sdk.lib import apikey

ID_FILE = "wandb_id.txt"


def require_login() -> None:
    """Fail before any data or model work if there are no W&B credentials."""
    if apikey.api_key() is None:
        raise RuntimeError(
            "no W&B credentials: run `wandb login`, or pass logging.wandb.enabled=false"
        )


def start(
    run_dir: Path, *, project: str, entity: str | None, name: str, config: dict
) -> wandb.sdk.wandb_run.Run:
    """Start a new W&B run, or continue the one recorded in ``run_dir``.

    ``entity: null`` in the config means the logged-in account's default entity."""
    id_path = run_dir / ID_FILE
    resuming = id_path.exists()
    run_id = id_path.read_text().strip() if resuming else wandb.util.generate_id()
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        id=run_id,
        resume="must" if resuming else "never",
        config=config,
        dir=run_dir,
    )
    id_path.write_text(run_id + "\n")
    return run
