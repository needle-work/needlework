# Training the components

needlework trains three models with one entry point, `python -m needlework.train`.
The IDM and verifier are trained once per task and used only to generate bridges. The
policy is trained twice: on logged data alone (the baseline) and with bridges. The
[README quick start](../README.md#quick-start-rocket) gives the commands.

Configuration composes as `train.yaml < component < task < sampler < command line`
([configs/](../src/needlework/configs/)). Every run writes its resolved configuration,
`metrics.jsonl` (one record per epoch) and `checkpoints/last.ckpt` under
`NEEDLEWORK_ROOT/outputs/<domain>/<task>/<component>/`.

## Contents

- [Shared settings](#shared-settings)
- [IDM](#idm)
- [Verifier](#verifier)
- [Policy](#policy)
  - [Training with bridges](#training-with-bridges)



## Shared settings

| Key | Meaning |
| --- | ------- |
| `seed.dataset` | Train/validation episode split. Keep it fixed when comparing runs. |
| `seed.training` | Initialization, batch order and diffusion noise. Vary this across repeated runs. |
| `horizon.prediction`, `horizon.obs`, `horizon.execute` | Rows predicted per call, observation steps, rows executed per call. The IDM, verifier and policy of one task must agree. |
| `optim.*`, `ema.*` | AdamW with warmup and cosine decay, and the EMA weights used for checkpoint inference. Shared by all components. |
| `task.epochs.{policy,idm,verifier}` | Epoch counts per task. |
| `determinism.strict` | Deterministic kernels. Set `false` to trade exact repeatability for speed. |

All components read the same cached DINOv3 features, so build them once per task
before training (see the [README](../README.md#quick-start-rocket)).

## IDM

The inverse dynamics model ([component/idm.yaml](../src/needlework/configs/component/idm.yaml))
is the policy's diffusion U-Net with an extra goal frame as input. Given an observation
history and a goal frame, it proposes the executable action rows that move from the
current frame toward the goal.

- **Data.** Pairs come from all training episodes, failures included. Each epoch
  draws a new goal for every window, `k` steps ahead with `k` uniform in
  `[1, H]` (clipped at the episode end). `validation.val_ratio` of all episodes is
  held out.
- **Targets.** For a goal reached at step `k`, the logged actions are held from that
  row on ("subgoal terminal" padding). The target therefore depends on the goal, which
  is what makes the IDM use it.
- **Output.** `checkpoints/last.ckpt`. Stitching loads the final epoch's EMA weights.

## Verifier

The verifier ([component/verifier.yaml](../src/needlework/configs/component/verifier.yaml))
is a transformer over source and goal patch tokens, source proprioception and an
action chunk. It predicts, for every executable step, whether the goal has been
reached by that step.

- **Positives** are within-horizon goals with their logged actions.
- **Negatives** (`sampling.n_negatives` per positive, mixed by
  `sampling.negative_ratio`) are goals beyond the horizon in the same episode
  (`beyond_hor_neg`, with a boundary band set by `boundary_prob` and
  `boundary_horizons`) and visually close goals in other episodes
  (`cross_traj_neg`, drawn from proximity tables).
- **Serving** (`task.verifier_serving`) sets how the action chunk is presented for
  each label; it differs by domain.
- **Calibration.** After every epoch the EMA model scores the validation set, and
  per-step thresholds are calibrated to `calibration.target_fpr`. The checkpoint
  carries these thresholds; stitching adds `task.stitch_threshold_offsets` to them.
- **Output.** `checkpoints/last.ckpt`, with thresholds.

## Policy

The policy ([component/policy.yaml](../src/needlework/configs/component/policy.yaml))
is a diffusion U-Net over spatial-softmax DINOv3 features and proprioception.

- **Baseline.** `sampler=normal` (the default) draws every window of every training
  success episode uniformly.
- **Augmented.** `sampler=augmented sampler.bridges=/path/to/bridges.zarr` mixes
  bridge windows with logged windows; see [Training with bridges](#training-with-bridges).
- **Output.** `checkpoints/last.ckpt`, plus `checkpoint.policy_snapshots` EMA
  snapshots listed in `checkpoints/selected.json`.

### Training with bridges

The policy sampler keeps a bridge if its source is in a training success episode or
in a failure episode. Its target can be in any success episode. A policy window uses
history from the source episode and the generated bridge actions; it never includes
the target episode's logged actions.

Sampling terms (`sampler=augmented`):

| Term | Meaning |
| ---- | ------- |
| **Logged** | A window of recorded actions from a training success episode. |
| **Crossing** | A window that takes a generated bridge from a selected source; it may start earlier and include logged action rows before departure. A failure-source bridge has only a departure window. Rows after the bridge are padding, excluded from the loss. |
| **Twin (original source actions)** | At a success departure, a paired window teaches the logged continuation from the same observation. The appendix's **Remove Original Source Actions** omits this window; **Sample Original Source Actions More** increases its draw weight. Failure departures have no twin. |
| `bridge_weight`, `role_weights` | `bridge_weight` weights success-departure crossings; `role_weights` separately weights skipped logged windows, twins, failure departures, and approach crossings. Zero weight excludes a role. `bridge_weight=0` uses ordinary logged training. |
| `sampler.options.path_start` | `aligned` keeps the logged window alignment; `repeat_current` repeats the current observation at an episode start or just after a success departure and starts actions there. |
| `sampler.options.twin_rows` | `own_source` supervises the twin's own logged source action; `all` also allows its other logged action rows. |
| `sampler.options.source_rows` | `masked` excludes logged actions near selected sources and in failure episodes; `supervised` allows logged rows, including those *before* a failure-source bridge. The bridge replaces the action at the failure departure. |
| `sampler.options.after_departure` | `all` keeps eligible logged windows; `skipped` keeps post-departure logged windows only at frames skipped by a success bridge. |
| `sampler.options.draw` | `even_passes` and `rounded_passes` allocate draws by role weight using shuffled passes; `with_replacement` draws independently with probability proportional to weight. |
| `sampler.epoch_length` | `logged` uses the original logged-window count; `windows` uses the number of positive-weight windows, including twins. |

Paired twin and departure-crossing windows have equal draw weight only when
`role_weights.twin=bridge_weight`. Task-specific values live in the
[task configs](../src/needlework/configs/task/).

The draw order is reproducible from the seed and epoch. Each window's loss is averaged
over its valid action elements, then over windows. UMI bridge poses are converted to
the current window's frame.

For a new bridge set, follow the ordered settings in [tuning.md](tuning.md).
