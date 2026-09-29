"""Real-data GPU smoke: three components, two domains, persistent multi-GPU workers.

Small models and deterministic subsets of the fixed splits are test-only. Their
checkpoints test interfaces, not policy performance or verifier quality.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from needlework import config
from needlework.data import bridges
from needlework.sampling import roles
from needlework.sampling.augmented_dataset import (
    AugmentedPolicyDataset,
    SamplerOptions,
)
from needlework.sampling.pairs import hold_from
from needlework.stitching import pipeline
from needlework.stitching.inference import load_component
from needlework.stitching.selection import pack
from needlework.stitching.workers import batch_rows, read_result, score_batches
from needlework.train import TASKS
from needlework.training import checkpoint, determinism
from needlework.training.batches import to_device
from needlework.training.common import Episodes, Windows
from needlework.training.engine import Trainer

ROLE_WEIGHTS = roles.RoleWeights(
    skipped=0.5, twin=0.5, failure_departure=0.5, approach=1.0
)
OPTIONS = SamplerOptions(
    path_start="aligned",
    twin_rows="own_source",
    draw="even_passes",
    source_rows="masked",
    after_departure="all",
)


class Limited:
    def __init__(self, data, count):
        self.data, self.count = data, min(count, len(data))

    @property
    def fallback(self):
        return self.data.fallback

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.data.set_epoch(epoch)

    def batch(self, indices):
        return self.data.batch(indices)


@pytest.fixture(scope="module")
def components(tmp_path_factory):
    assert torch.cuda.is_available()
    # as train.py and stitch.py set it, before cuBLAS starts
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism.CUBLAS_WORKSPACE
    root = tmp_path_factory.mktemp("real_gpu_components")
    result = {}
    original = Windows.episodes

    def subset(self, *, successes_only):
        ids = original(self, successes_only=successes_only)
        return Episodes(ids.train[:2], ids.val[:2])

    # Only this fixture reduces data and model size; no shipped code path does.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Windows, "episodes", subset)
        for domain, _task in [("robomimic", "square"), ("umi", "dish")]:
            paths = {}
            for component in ("idm", "verifier", "policy"):
                overrides = [
                    f"component={component}",
                    f"task={'umi' if domain == 'umi' else 'robomimic/square'}",
                    f"run.name=smoke_{domain}_{component}",
                    "logging.wandb.enabled=false",
                    "component.train.epochs=1",
                    "component.train.batch_size=8",
                    "optim.warmup_steps=0",
                ]
                if domain == "umi":
                    overrides += ["task.name=dish"]
                if component == "verifier":
                    overrides += [
                        "component.model.hidden=32",
                        "component.model.layers=1",
                        "component.model.ff_mult=2",
                        "component.calibration.min_negatives=20",
                        "component.sampling.proximity.max_targets=8",
                        "component.sampling.proximity.candidate_pool=32",
                    ]
                else:
                    overrides += [
                        "component.unet.down_dims=[32,64,128]",
                        "component.unet.step_embed_dim=32",
                        "component.validation.action_mse_windows=4",
                        "component.validation.action_mse_every=1",
                    ]
                    if component == "idm":
                        overrides += ["component.hidden=[64,32]"]
                cfg = config.compose(overrides)
                OmegaConf.update(
                    cfg,
                    "smoke",
                    {"episodes_per_split": 2, "train_examples": 32, "val_examples": 64},
                    force_add=True,
                )
                generator = determinism.configure(seed=42, strict=True)
                instance = TASKS[component](cfg, device=torch.device("cuda"), fit=True)
                store = instance.windows.store
                if component == "verifier":
                    pairs = instance.train_data
                    labels = pairs.offset > 0
                    # Each domain's shipped serving: Robomimic holds exactly the
                    # positives; UMI holds independently of the label.
                    independent = cfg.task.verifier_serving.mode == "independent"
                    if independent:
                        for positive in (True, False):
                            frac = pairs.hold[labels == positive].mean()
                            assert 0.4 < frac < 0.6
                    else:
                        np.testing.assert_array_equal(pairs.hold, labels)
                    ids = np.arange(32)
                    _, raw = pairs.data.arrays(pairs.row[ids], False)
                    held = hold_from(
                        raw, pairs.hold_offset[ids], pairs.data.shape.n_obs
                    )
                    # Disable UMI start noise for observation comparison only; actions
                    # never use it.
                    expected = np.where(pairs.hold[ids, None, None], held, raw)
                    np.testing.assert_array_equal(
                        pairs.batch(ids)["action"].numpy(), expected
                    )
                    first = pairs.hold.copy()
                    pairs.set_epoch(0)
                    np.testing.assert_array_equal(first, pairs.hold)
                    pairs.set_epoch(1)
                    if independent:
                        assert not np.array_equal(first, pairs.hold)
                if component == "policy":
                    data = instance.train_data
                    ep = int(data.windows.episode[0])
                    src = int(store.episode_starts[ep]) + 10
                    actions = store.arrays["action"][src : src + 3].copy()
                    if domain == "umi":
                        from needlework.geometry.actions import (
                            action_poses,
                            to_relative,
                        )

                        actions = to_relative(
                            store.layout,
                            actions,
                            action_poses(store.layout, store.arrays["action"][src]),
                        )
                    b = pack(
                        [
                            {
                                "candidate_id": 0,
                                "source": src,
                                "target": int(store.episode_ends[ep]) - 2,
                                "stage": 1,
                                "margin": 1.0,
                                "actions": actions,
                            }
                        ],
                        store.spec.action_dim,
                    )
                    path = root / f"{domain}_controlled.zarr"
                    bridges.save(
                        path,
                        b,
                        store,
                        horizon=23,
                        stride=cfg.task.stride,
                        recipe={"controlled_smoke": True},
                    )
                    instance.train_data = AugmentedPolicyDataset(
                        data,
                        bridges.load(path, store, horizon=23, stride=cfg.task.stride),
                        weight=1.0,
                        role_weights=ROLE_WEIGHTS,
                        epoch_length="windows",
                        options=OPTIONS,
                        seed=42,
                    )
                    # Deliberately sample an actual crossing for a GPU forward/backward.
                    obs, action, valid = instance.train_data.bridge_arrays(
                        np.array([0])
                    )
                    batch = to_device(
                        {
                            "obs": {k: torch.from_numpy(v) for k, v in obs.items()},
                            "action": torch.from_numpy(action),
                            "valid": torch.from_numpy(valid),
                        },
                        torch.device("cuda"),
                    )
                    instance.model.cuda()
                    loss = instance.loss(instance.model, batch)
                    assert torch.isfinite(loss)
                    loss.backward()
                    instance.model.zero_grad(set_to_none=True)
                instance.train_data = Limited(instance.train_data, 32)
                instance.val_data = Limited(instance.val_data, 64)
                trainer = Trainer(
                    instance, cfg, device=torch.device("cuda"), data_generator=generator
                )
                directory = root / f"{domain}_{component}"
                directory.mkdir()
                OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
                records = []

                def save(
                    state,
                    metrics,
                    *,
                    cfg=cfg,
                    store=store,
                    component=component,
                    instance=instance,
                    directory=directory,
                ):
                    payload = {
                        **state,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "data_identity": store.identity,
                    }
                    if component == "verifier":
                        payload["thresholds"] = instance.thresholds
                    checkpoint.save(directory, payload)
                    checkpoint.retain_policy(directory, payload, metrics)

                trainer.run(log=records.append, save=save)
                assert all(
                    np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"])
                    for r in records
                )
                (directory / "metrics.json").write_text(json.dumps(records, indent=2))
                paths[component] = checkpoint.path_in(directory)
                if component != "policy":
                    model, _, thresholds = load_component(
                        paths[component], component, store, torch.device("cuda")
                    )
                    if component == "verifier":
                        assert torch.isfinite(thresholds).all()
                    del model
                del trainer, instance
                gc.collect()
                torch.cuda.empty_cache()
            stitch = config.compose_stitch(
                [
                    f"task={'umi' if domain == 'umi' else 'robomimic/square'}",
                    f"idm_checkpoint={paths['idm']}",
                    f"verifier_checkpoint={paths['verifier']}",
                    "run.name=smoke",
                    "sources_per_stage=2",
                    "targets_per_source={stage1: 2, stage2: 2, stage3: 2}",
                    "batch_size=1",
                    "stages=[1,2]",
                    "task.stitch_threshold_offsets="
                    "{stage1: -1000.0, stage2: -1000.0, stage3: -1000.0}",
                    *(["task.name=dish"] if domain == "umi" else []),
                ]
            )
            result[domain] = (stitch, store)
    print(f"SMOKE_ARTIFACTS={root}", flush=True)
    return result


@pytest.mark.parametrize("domain", ["robomimic", "umi"])
def test_multigpu_and_resume(components, domain, tmp_path):
    assert torch.cuda.device_count() >= 2, "requires two visible GPUs"
    cfg, store = components[domain]
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    e = int(
        np.flatnonzero(
            store.episode_success & ((store.episode_ends - store.episode_starts) > 160)
        )[0]
    )
    s = int(store.episode_starts[e])
    rows = np.array(
        [[i, s + 20 + i * 4, s + 120 + i * 4, 1] for i in range(4)], np.int64
    )
    path = tmp_path / "candidates.npy"
    np.save(path, rows)

    def scores(paths: list) -> list:
        return [
            read_result(p, batch_rows(rows, i, cfg.batch_size))
            for i, p in enumerate(paths)
        ]

    def same(x: list, y: list) -> bool:
        return all(
            np.array_equal(a[k], b[k]) for a, b in zip(x, y, strict=True) for k in a
        )

    cfg.gpus = [0]
    one = scores(score_batches(cfg, path, tmp_path / "one"))
    cfg.gpus = [0, 1]
    two = scores(score_batches(cfg, path, tmp_path / "two"))
    assert same(one, two)
    # Resume on another GPU, skipping all completed work.
    cfg.gpus = [1]
    assert same(two, scores(score_batches(cfg, path, tmp_path / "two")))
    # Resume with one missing batch and changed GPU count.
    (tmp_path / "two/000001.npz").unlink()
    assert same(two, scores(score_batches(cfg, path, tmp_path / "two")))
    np.save(path, rows + 1)
    with pytest.raises(ValueError, match="different candidates"):
        score_batches(cfg, path, tmp_path / "two")


@pytest.mark.parametrize("domain", ["robomimic", "umi"])
def test_pipeline_real_candidates(components, domain, tmp_path):
    cfg, store = components[domain]
    cfg.gpus = [0, 1]
    output = pipeline.run(cfg, tmp_path)
    b = bridges.load(output, store, horizon=23, stride=cfg.task.stride)
    assert len(b) <= 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["threshold_offsets"] == {"1": -1000.0, "2": -1000.0}
    assert summary["by_stage"] == {
        k: int((b.stage == int(k)).sum()) for k in ("1", "2")
    }
    cfg.gpus = [1]
    assert pipeline.run(cfg, tmp_path) == output
    cfg.seed += 1
    with pytest.raises(ValueError, match="resume inputs"):
        pipeline.run(cfg, tmp_path)
    cfg.seed -= 1


def test_pipeline_without_candidates_writes_an_empty_store(components, tmp_path):
    cfg, store = components["robomimic"]
    cfg.gpus = [0]
    np.save(tmp_path / "candidates.npy", np.empty((0, 4), np.int64))
    output = pipeline.run(cfg, tmp_path)
    assert len(bridges.load(output, store, horizon=23, stride=cfg.task.stride)) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["candidates"] == 0 and summary["selected_bridges"] == 0
    assert summary["accepted_by_stage"] == {"1": 0, "2": 0}


def test_worker_failure_cleanup(components, tmp_path):
    cfg, _ = components["robomimic"]
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.gpus = [0, 1]
    cfg.idm_checkpoint = str(tmp_path / "missing.ckpt")
    path = tmp_path / "candidates.npy"
    np.save(path, np.array([[0, 20, 120, 1], [1, 30, 130, 1]]))
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="worker"):
        score_batches(cfg, path, tmp_path / "bad")
    assert {p.pid for p in mp.active_children()} == before
