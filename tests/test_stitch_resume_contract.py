"""The stitch resume contract holds the settings that define candidates, scoring and
the kept bridges; task settings the stitch never reads do not refuse a resume."""

import pytest
from omegaconf import OmegaConf

from needlework import config
from needlework.stitching import pipeline


def _resolved(*overrides: str) -> dict:
    cfg = config.compose_stitch(
        [
            "task=robomimic/square",
            "run.name=t",
            "idm_checkpoint=/a/idm.ckpt",
            "verifier_checkpoint=/a/verifier.ckpt",
            *overrides,
        ]
    )
    return OmegaConf.to_container(cfg, resolve=True)


def test_unread_task_settings_paths_and_selection_leave_the_contract_unchanged() -> (
    None
):
    """Thresholds and paring apply after scoring, so a resume may change them."""
    base = pipeline.contract_config(_resolved())
    for override in (
        "task.epochs.verifier=7",
        "task.eval.max_steps=9",
        "task.verifier_serving.mode=independent",
        "idm_checkpoint=/b/idm.ckpt",
        "gpus=[1]",
        "run.name=u",
        "task.stitch_threshold_offsets.stage2=-2.0",
        "task.stitch_keep_fraction.stage2=0.1",
    ):
        assert pipeline.contract_config(_resolved(override)) == base, override
    assert {"threshold_offsets", "keep_fraction"} <= set(pipeline.NOT_CONTRACT)


@pytest.mark.parametrize(
    "override",
    [
        "seed=43",
        "batch_size=16",
        "task.stitch_targets_per_source.stage2=50",
        "task.stride=2",
    ],
)
def test_semantic_settings_change_the_contract(override: str) -> None:
    assert pipeline.contract_config(_resolved(override)) != pipeline.contract_config(
        _resolved()
    )


def test_unclassified_key_raises() -> None:
    with pytest.raises(ValueError, match="outside the resume contract"):
        pipeline.contract_config({**_resolved(), "new_setting": 1})


def _saved_run(tmp_path):
    from needlework.training import run_dir

    cfg = config.compose_stitch(
        [
            "task=robomimic/square",
            "run.name=t",
            "idm_checkpoint=/a/idm.ckpt",
            "verifier_checkpoint=/a/verifier.ckpt",
        ]
    )
    OmegaConf.save(cfg, tmp_path / run_dir.CONFIG, resolve=True)
    return tmp_path


def test_resume_takes_selection_overrides_and_records_them(tmp_path) -> None:
    from needlework import stitch
    from needlework.training import run_dir

    run = _saved_run(tmp_path)
    cfg, directory = stitch.resume_config(
        [
            "--resume",
            str(run),
            "gpus=[1]",
            "threshold_offsets.stage2=-2.0",
            "keep_fraction.stage1=0.3",
        ]
    )
    assert directory == run.resolve()
    assert list(cfg.gpus) == [1]
    assert cfg.threshold_offsets.stage2 == -2.0 and cfg.keep_fraction.stage1 == 0.3
    saved = OmegaConf.load(run / run_dir.CONFIG)  # the run records what selected it
    assert saved.threshold_offsets.stage2 == -2.0 and saved.keep_fraction.stage1 == 0.3


@pytest.mark.parametrize(
    "override", ["seed=43", "batch_size=16", "threshold_offsets=0.5", "task.stride=2"]
)
def test_resume_refuses_other_overrides(tmp_path, override: str) -> None:
    from needlework import stitch

    with pytest.raises(ValueError, match="usage"):
        stitch.resume_config(["--resume", str(_saved_run(tmp_path)), override])


def test_resume_selection_override_needs_no_bridge_store(tmp_path) -> None:
    from needlework import stitch

    run = _saved_run(tmp_path)
    (run / "bridges.zarr").mkdir()
    with pytest.raises(ValueError, match=r"bridges\.zarr"):
        stitch.resume_config(["--resume", str(run), "threshold_offsets.stage2=-2.0"])
    stitch.resume_config(["--resume", str(run), "gpus=[0]"])  # GPUs alone may change


@pytest.mark.parametrize(
    "override",
    [
        "threshold_offsets.stage2=.nan",
        "keep_fraction.stage1=0",
        "threshold_offsets.stage3=x",
    ],
)
def test_resume_checks_a_selection_override_before_recording_it(
    tmp_path, override: str
) -> None:
    from needlework import stitch
    from needlework.training import run_dir

    run = _saved_run(tmp_path)
    before = (run / run_dir.CONFIG).read_text()
    with pytest.raises(ValueError):
        stitch.resume_config(["--resume", str(run), override])
    assert (run / run_dir.CONFIG).read_text() == before


@pytest.mark.parametrize(
    "override",
    [
        "threshold_offsets.stage4=0.3",
        "keep_fraction.stage7=0.5",
        "threshold_offsets.stage2=",
    ],
)
def test_resume_refuses_a_selection_override_it_would_not_read(
    tmp_path, override: str
) -> None:
    from needlework import stitch
    from needlework.training import run_dir

    run = _saved_run(tmp_path)
    before = (run / run_dir.CONFIG).read_text()
    with pytest.raises(ValueError, match=r"stage4|stage7|usage"):
        stitch.resume_config(["--resume", str(run), override])
    assert (run / run_dir.CONFIG).read_text() == before


def test_resume_records_a_gpu_override(tmp_path) -> None:
    from needlework import stitch
    from needlework.training import run_dir

    run = _saved_run(tmp_path)
    stitch.resume_config(["--resume", str(run), "gpus=[1]"])
    assert list(OmegaConf.load(run / run_dir.CONFIG).gpus) == [1]


def test_a_recorded_override_agrees_with_its_task_entry(tmp_path) -> None:
    """The run config records a selection override in the stitch setting and in the
    task entry it reads, so composing the recorded task block gives the same values."""
    from needlework import stitch
    from needlework.training import run_dir

    run = _saved_run(tmp_path)
    stitch.resume_config(
        [
            "--resume",
            str(run),
            "threshold_offsets.stage2=-0.25",
            "keep_fraction.stage3=0.2",
        ]
    )
    saved = OmegaConf.load(run / run_dir.CONFIG)
    for name in ("threshold_offsets", "keep_fraction"):
        assert saved[name] == saved.task[f"stitch_{name}"]
        entries = [
            f"task.stitch_{name}.{k}={v}"
            for k, v in saved.task[f"stitch_{name}"].items()
        ]
        assert _resolved(*entries)[name] == OmegaConf.to_container(saved[name])
    assert saved.threshold_offsets.stage2 == -0.25 and saved.keep_fraction.stage3 == 0.2
