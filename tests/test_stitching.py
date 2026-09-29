"""Selector, worker file integrity and checkpoint retention contracts."""

import json

import numpy as np
import pytest
import torch

from needlework.stitching.candidates import max_prefix_lengths
from needlework.stitching.inference import proposal_seed
from needlework.training import checkpoint


def test_proposal_seed_is_endpoint_identity():
    rows = np.array([[0, 1, 20, 1], [1, 3, 30, 1]])
    assert proposal_seed(42, rows[0], 0) == proposal_seed(
        42, np.array([999, 1, 20, 1]), 0
    )
    assert proposal_seed(42, rows[0], 0) != proposal_seed(42, rows[0], 1)


def test_min_steps_saved_is_in_policy_steps(can_store):
    """The route must shorten by min_steps_saved POLICY steps (stride raw frames each)
    after paying for the bridge (stride raw frames per bridge step). Stage 3 has no
    route requirement."""
    s = can_store
    long = s.episode_success & (s.episode_ends - s.episode_starts > 150)
    src = int(s.episode_starts[int(np.flatnonzero(long)[0])]) + 20

    def longest(gap: int, stride: int, stage: int = 1) -> int:
        rows = np.array([[0, src, src + gap, stage]])
        return int(
            max_prefix_lengths(s, rows, horizon=23, stride=stride, min_steps_saved=25)[
                0
            ]
        )

    assert longest(30, stride=3) < 1  # 30 raw frames < 25 steps * 3 frames
    assert longest(76, stride=3) < 1  # a 1-step bridge would leave 73 < 75
    assert longest(81, stride=3) == 2  # 81 - 3 * 2 = 75 frames saved
    assert longest(40, stride=1) == 15  # stride 1: raw frames = policy steps, unchanged
    assert longest(30, stride=3, stage=3) == 23


@pytest.mark.parametrize("domain,expected", [("robomimic", [2, 4]), ("umi", [4, 3])])
def test_retention(tmp_path, domain, expected):
    assert torch.cuda.is_available()
    for epoch, score in enumerate([0.1, 0.9, 0.2, 0.8], 1):
        payload = {
            "epoch": epoch,
            "ema": {"w": torch.tensor([epoch], device="cuda")},
            "config": {
                "component": {"name": "policy"},
                "task": {"domain": domain},
                "checkpoint": {"policy_snapshots": 2},
            },
            "data_identity": {"x": "y"},
        }
        checkpoint.save(tmp_path, payload)
        checkpoint.retain_policy(tmp_path, payload, {"eval/success_rate": score})
    index = json.loads((tmp_path / "checkpoints/selected.json").read_text())
    assert [e["epoch"] for e in index] == expected
    assert len(list((tmp_path / "checkpoints").glob("epoch_*.ckpt"))) == 2
    assert checkpoint.load(checkpoint.path_in(tmp_path))["epoch"] == 4
    payload["config"]["component"]["name"] = "idm"
    checkpoint.retain_policy(tmp_path, payload, {})
    assert json.loads((tmp_path / "checkpoints/selected.json").read_text()) == index


def test_recovery_gate_uses_cross_episode_reference():
    from needlework.stitching.recovery import recovery_sources

    # Successes in different episodes are near each other; one failure is far.
    x = torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [1.0, 0.0], [-1.0, 0.0]], device="cuda"
    )
    x = x / x.norm(dim=1, keepdim=True)
    episode = np.array([0, 0, 1, 2, 2])
    success = np.array([True, True, False])
    result = recovery_sources(x, episode, success, quantile=0.9)
    assert not result[:4].any() and result[4]


def _mapped(path) -> bool:
    with open("/proc/self/maps") as handle:
        return str(path) in handle.read()


def test_stitching_reads_last_ckpt_without_optimizer_state(tmp_path, can_store):
    """Stitching reads last.ckpt memory-mapped and keeps only what inference uses, so
    optimizer state is never read into memory; an incomplete run is rejected. The test
    checks the mapping, since resident memory is too noisy to assert on."""
    from needlework.stitching.inference import load_component

    config = {"component": {"name": "idm", "train": {"epochs": 2}}}
    kept = {"ema": {"w": torch.zeros(4)}, "config": config, "data_identity": {}}
    moments = {"state": {0: {"exp_avg": torch.ones(1024)}}}
    path = checkpoint.save(
        tmp_path, {**kept, "epoch": 2, "optimizer": moments, "model": {}, "rng": {}}
    )
    loaded = checkpoint.load_inference(path)
    assert set(loaded) == set(checkpoint.SNAPSHOT_KEYS)
    assert _mapped(path)  # tensors are views of the file, read only when touched
    del loaded
    plain = checkpoint.load(path)  # control: a plain load copies and closes the file
    assert not _mapped(path)
    del plain
    kept["data_identity"] = can_store.identity
    incomplete = checkpoint.save(tmp_path, {**kept, "epoch": 1})
    with pytest.raises(ValueError, match="completed idm run"):
        load_component(incomplete, "idm", can_store, torch.device("cuda"))


def test_verifier_thresholds_are_not_model_state(tmp_path):
    """Thresholds are a calibration result, not weights: not a module buffer (the EMA
    copies every buffer from the uncalibrated online model), but a field of the
    verifier's last.ckpt, required when it is loaded for inference."""
    from needlework.models.verifier import Verifier

    verifier = Verifier(
        n_cameras=2,
        patch_grid=7,
        feature_dim=768,
        proprio={"p": 3},
        action_dim=10,
        horizon=23,
        n_obs=2,
        hidden=32,
        layers=1,
        heads=4,
        ff_mult=2,
    ).cuda()
    assert "thresholds" not in dict(verifier.named_buffers())
    config = {"component": {"name": "verifier", "train": {"epochs": 1}}}
    payload = {"epoch": 1, "ema": {}, "config": config, "data_identity": {}}
    with pytest.raises(ValueError, match="no thresholds"):
        checkpoint.load_inference(checkpoint.save(tmp_path, payload))
    thresholds = torch.linspace(-1, 1, 23)
    path = checkpoint.save(tmp_path, {**payload, "thresholds": thresholds})
    torch.testing.assert_close(
        checkpoint.load_inference(path)["thresholds"], thresholds
    )


def test_simulator_initialization_failure_cleans_children(tmp_path):
    import multiprocessing as mp

    from needlework.sim.vector_env import SimPool, WorkerSpec

    spec = WorkerSpec(
        hdf5=tmp_path / "missing.hdf5",
        cameras=("agentview_image",),
        proprio=("robot0_eef_pos",),
        arms=("robot0",),
        max_steps=1,
        n_obs=2,
        construction_seed=42,
        egl_index=0,
    )
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="simulator slot"):
        SimPool([spec, spec])
    assert {p.pid for p in mp.active_children()} == before
