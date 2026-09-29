"""Tests of the stitching, sampling, verifier and training contracts.

They read the unpacked datasets under ``$NEEDLEWORK_ROOT/data`` and need a CUDA GPU; the
pipeline tests need two. From the package root, with the environment active:

    python -m pytest tests -q

Every test is skipped when ``NEEDLEWORK_ROOT`` is unset or holds no ``data`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from needlework import paths
from needlework.data.store import EpisodeStore


def _data_missing() -> str | None:
    if paths.ROOT_ENV not in os.environ:
        return f"{paths.ROOT_ENV} is not set"
    if not (Path(os.environ[paths.ROOT_ENV]) / "data").is_dir():
        return f"${paths.ROOT_ENV}/data does not exist"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    reason = _data_missing()
    if reason is not None:
        for item in items:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def can_store() -> EpisodeStore:
    return EpisodeStore.open("robomimic", "can")


@pytest.fixture(scope="session")
def transport_store() -> EpisodeStore:
    return EpisodeStore.open("robomimic", "transport")


@pytest.fixture(scope="session")
def sweater_store() -> EpisodeStore:
    return EpisodeStore.open("umi", "sweater")
