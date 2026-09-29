"""Constants that define a run are recorded with it; a resume refuses when one of them
differs, naming it, and refuses a checkpoint without the record."""

import pytest

from needlework import config, constants, train
from needlework.stitching import pipeline
from needlework.training import checkpoint


def test_training_resume_refuses_changed_constants(tmp_path) -> None:
    cfg = config.compose(
        [
            "component=policy",
            "task=robomimic/square",
            "run.name=t",
            "logging.wandb.enabled=false",
        ]
    )
    config.save(cfg, tmp_path)
    changed = {**constants.snapshot(), "POOLED_GRID": 8}
    checkpoint.save(tmp_path, {"epoch": 1, "constants": changed})
    with pytest.raises(ValueError, match="POOLED_GRID"):
        train.main(["--resume", str(tmp_path)])
    checkpoint.save(tmp_path, {"epoch": 1})
    with pytest.raises(ValueError, match="no constants snapshot"):
        train.main(["--resume", str(tmp_path)])


def test_stitch_resume_refuses_changed_constants(tmp_path) -> None:
    path = tmp_path / "inputs.json"
    identity = {"config": {"seed": 42}, "constants": constants.snapshot()}
    pipeline.check_resume(path, identity)  # a new run records its inputs
    pipeline.check_resume(path, identity)  # the same inputs resume
    changed = {**identity, "constants": {**identity["constants"], "HOLD_STREAM": 1}}
    with pytest.raises(ValueError, match=r"constants\.HOLD_STREAM"):
        pipeline.check_resume(path, changed)
    assert constants.snapshot()["HOLD_STREAM"] != 1
    assert "SIMILARITY_BATCH" not in constants.snapshot()  # plumbing never blocks
