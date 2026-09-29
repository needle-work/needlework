"""The composed task and stitch configs carry exactly these keys."""

import pytest

from needlework import config

TASKS = (
    ["task=robomimic/can"],
    ["task=robomimic/square"],
    ["task=robomimic/transport"],
    ["task=umi", "task.name=sweater"],
)
STITCH_TASK_KEYS = {
    "stitch_sources_per_stage",
    "stitch_candidate_spacing",
    "stitch_targets_per_source",
    "stitch_max_candidates_per_stage",
    "stitch_threshold_offsets",
    "stitch_keep_fraction",
}
TASK_KEYS = {
    "domain",
    "name",
    "stride",
    "relative_actions",
    "bridge_weight",
    "role_weights",
    "epoch_length",
    "sampler_options",
    "action_identity",
    "verifier_serving",
    "epochs",
    "obs",
    "proprio_identity",
    *STITCH_TASK_KEYS,
}
DOMAIN_KEYS = {"robomimic": {"eval"}, "umi": {"start_noise"}}
STITCH_KEYS = {
    "task",
    "idm_checkpoint",
    "verifier_checkpoint",
    "run",
    "gpus",
    "seed",
    "stages",
    "sources_per_stage",
    "candidate_spacing",
    "targets_per_source",
    "max_candidates_per_stage",
    "threshold_offsets",
    "keep_fraction",
    "batch_size",
    "min_steps_saved",
    "proposals",
    "min_votes",
    "recovery_quantile",
    "proximity",
}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t[0])
@pytest.mark.parametrize("component", ["idm", "verifier", "policy"])
def test_task_configs_carry_exactly_the_task_keys(task: list, component: str) -> None:
    cfg = config.compose([f"component={component}", *task, "run.name=t"])
    assert set(cfg.task) == TASK_KEYS | DOMAIN_KEYS[cfg.task.domain]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t[0])
def test_stitch_config_carries_exactly_the_stitch_keys(task: list) -> None:
    cfg = config.compose_stitch(
        [*task, "run.name=t", "idm_checkpoint=a", "verifier_checkpoint=b"]
    )
    assert set(cfg) == STITCH_KEYS
    for key in STITCH_TASK_KEYS:  # each stitch setting reads its task entry
        assert cfg[key.removeprefix("stitch_")] == cfg.task[key]


ROLE_DEFAULTS = {"skipped": 0.5, "twin": 0.5, "failure_departure": 0.5, "approach": 1.0}
PATH_OPTIONS = {
    "path_start": "repeat_current",
    "twin_rows": "all",
    "draw": "with_replacement",
    "source_rows": "supervised",
    "after_departure": "skipped",
}
# repeat_current is defined for stride-1 absolute actions only; UMI keeps aligned
# starts.
UMI_OPTIONS = {
    **PATH_OPTIONS,
    "path_start": "aligned",
    "source_rows": "masked",
    "after_departure": "all",
}
# Transport keeps the logged-replacement sampler: every logged window 1, every
# crossing window the bridge weight, no twins, an epoch the logged dataset's length.
REPLACEMENT_OPTIONS = {
    "path_start": "aligned",
    "twin_rows": "own_source",
    "draw": "rounded_passes",
    "source_rows": "masked",
    "after_departure": "all",
}
REPLACEMENT_ROLES = {
    "skipped": 1.0,
    "twin": 0.0,
    "failure_departure": 0.967,
    "approach": 0.967,
}


@pytest.mark.parametrize(
    ("task", "departure", "role_weights", "epoch_length", "options"),
    [
        (TASKS[0], 0.18, ROLE_DEFAULTS, "windows", PATH_OPTIONS),
        (TASKS[1], 0.14, ROLE_DEFAULTS, "windows", PATH_OPTIONS),
        (TASKS[2], 0.967, REPLACEMENT_ROLES, "logged", REPLACEMENT_OPTIONS),
        (TASKS[3], 0.25, ROLE_DEFAULTS, "windows", UMI_OPTIONS),
    ],
    ids=["can", "square", "transport", "umi"],
)
def test_sampling_weights_per_task(
    task: list,
    departure: float,
    role_weights: dict,
    epoch_length: str,
    options: dict,
) -> None:
    cfg = config.compose(
        [
            "component=policy",
            *task,
            "run.name=t",
            "sampler=augmented",
            "sampler.bridges=x",
        ]
    )
    assert cfg.sampler.bridge_weight == cfg.task.bridge_weight == departure
    assert dict(cfg.sampler.role_weights) == dict(cfg.task.role_weights) == role_weights
    assert cfg.sampler.epoch_length == cfg.task.epoch_length == epoch_length
    assert dict(cfg.sampler.options) == dict(cfg.task.sampler_options) == options
