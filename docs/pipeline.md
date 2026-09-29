# Stitching pipeline

Run the pipeline with `python -m needlework.stitch`. The root configuration is
[`stitch.yaml`](../src/needlework/configs/stitch.yaml); task configurations set candidate
budgets, stage threshold offsets, and keep fractions. The pipeline reads logged
stores, cached DINOv3 features, and final IDM and verifier checkpoints. It writes
`bridges.zarr` for `sampler=augmented`.

> [!IMPORTANT] Stitching scores bridges between logged observations without stepping a simulator or using object state. Use final-epoch IDM and verifier checkpoints.



## Contents

- [Pipeline at a glance](#pipeline-at-a-glance)
- [1. Validate inputs](#1-validate-inputs)
- [2. Build visual candidates](#2-build-visual-candidates)
- [3. Propose and select bridges](#3-propose-and-select-bridges)
- [4. Run workers and save](#4-run-workers-and-save)
- [Code map](#code-map)



## Pipeline at a glance


| Step     | What happens                                            | Main output       |
| -------- | ------------------------------------------------------- | ----------------- |
| Validate | Check input stores and final component checkpoints      | `inputs.json`     |
| Search   | Find visually close source-target pairs in three stages | `candidates.npy`  |
| Score    | Sample IDM actions and score prefixes with the verifier | `batches/*.npz`   |
| Select   | Apply thresholds, vote, choose one bridge per source    | `bridges.zarr`    |
| Train    | Mix bridge windows with logged policy windows           | Policy checkpoint |


Training a policy with the bridges is described in [training.md](training.md#training-with-bridges).



## 1. Validate inputs

Both components load from `checkpoints/last.ckpt` at their configured final epoch.
The files are memory-mapped so the loader reads only EMA weights, configuration,
data identity, and verifier thresholds, without loading optimizer state. It checks
component type, task, source-store identity, observation keys, action convention,
stride, and observation and prediction horizons. Verifier thresholds must be finite.
The components may have different epoch counts. The IDM uses the DDIM step count
from its configuration.

`inputs.json` records the resolved generation settings, source-store identities,
and SHA-256 hashes of both checkpoints. Resume rejects changed settings or inputs.
GPU assignment and run name do not affect candidate generation. The loader verifies
that a checkpoint is complete.

## 2. Build visual candidates

Frames use frozen spatial-softmax DINOv3 features. The proximity sampler concatenates
camera features and L2-normalizes them, then compares them by cosine similarity.
The cutoff is the `proximity.tau_percentile` percentile (5th by default) of
similarities between frames one executable horizon apart within an episode.
`proximity.shells` sets the distance bands. Stitching and verifier training each
use their own proximity settings.

Stages are:

1. Successful source to a later state in the same successful episode.
2. Successful source to a state in another successful episode.
3. Failure source to a successful episode.

Sources are episode-balanced and temporally spaced. Eligible targets are distributed
across distance bands and episodes, also respecting temporal spacing. A distance band
is a range of `1 - cosine_similarity`; round-robin selection across bands avoids spending
all candidates on almost identical frames. The verifier builds its hard-negative goal
tables with the same selection and its own eligibility rules. Stage 3 draws a failure
source's targets across distance bands in random order, uniformly over frames in
each band. Proximity scores are not policy loss weights.

For stages 1/2, compare the entire remaining route:

```text
saved raw frames = source remaining - bridge length * stride - target remaining
```

Require at least `min_steps_saved` policy steps saved, i.e. `stride * min_steps_saved`
raw frames (the IDM also works in policy steps), and restrict eligible accepted prefix
lengths accordingly. Stage 1 additionally requires the target beyond the executable
horizon. Stage 3 has no route-shortening requirement.

Failure-source gating uses visual embeddings. For each successful frame, find its
closest frame in another successful episode. Take the `recovery_quantile` quantile
(0.9 by default) of those distances, and retain failure sources at least that far
from their closest successful frame. This favors failure observations with limited
coverage in the success data. Stage 3 draws failure sources and targets subject to `sources_per_stage`
and `candidate_spacing`, applies the stage cap, then retains sources that pass the
gate. With a cap, the candidate budget is spread across the drawn failure frames.

Candidate limits and task budgets

`candidates.npy` holds int64 rows `[candidate_id, source_frame, target_frame, stage]`.
Generation scores bounded source blocks without storing a full similarity matrix.
`sources_per_stage` and the per-stage `targets_per_source` (`{stage1, stage2, stage3}`,
one entry for every stage run) are upper budgets; eligibility and spacing can produce
fewer candidates. `candidate_spacing` is the raw-frame gap kept between selected sources
of one episode and between the targets of one source. `max_candidates_per_stage`
(`{stage1, stage2, stage3}`, null for no cap) caps each stage source-uniformly: rounds
take one target from every source that still has one, in a seeded random order, until
the cap. Each task sets the four budgets (`task.stitch_sources_per_stage`,
`task.stitch_candidate_spacing`, `task.stitch_targets_per_source`,
`task.stitch_max_candidates_per_stage`). The stage-3 recovery gate runs before its
candidate cap. Stage 3's reference-distance calibration still examines the
successful dataset even with a small inference budget. The
[task configs](../src/needlework/configs/task/) contain the budgets for each dataset.



## 3. Propose and select bridges

Each worker loads the models' EMA weights once and memory-maps the feature caches.
Workers on one host share a page-cache copy. Each worker constructs source
observation history with episode-boundary padding and visual target features. The IDM
uses its configured DDIM step count (ten by default) and produces `proposals` action
samples per pair (eight by default). Initial noise is seeded by run seed, source,
target, stage, and sample number; GPU index and completion order do not affect it.
UMI actions are relative to the source gripper pose.

The causal verifier scores each executable prefix using source/goal patch tokens,
source proprioception and the proposed actions. It never reads target proprioception,
object state, or simulation outcomes. A prefix's margin is its logit minus the
verifier's calibrated per-step threshold.

`selection.py` applies two settings per stage:

1. **Threshold offset** (`task.stitch_threshold_offsets`) is added to each calibrated
  verifier threshold. A prefix passes when both its own logit and the mean logit
   across samples clear that threshold. A candidate needs passing prefixes from at
   least `min_votes` different action samples (two by default). Prefixes beyond the
   candidate's eligible maximum never pass.
2. **Keep fraction** (`task.stitch_keep_fraction`, in (0, 1]) retains the
  `max(1, ceil(f × n))` highest-margin packed bridges for each stage. Margin ties
   favor the lower source frame.

For each accepted candidate, the shortest passing prefix is selected. Among samples
accepted at that length, the median by margin supplies the bridge actions and margin
(lower median for an even count; sample-index tie break). When targets compete for
the same source, stage 1 precedes stage 2. Within a stage, selection favors higher
margin, more raw frames saved, shorter bridge, then lower candidate ID. Failure
sources use stage 3 only. Each source contributes at most one bridge.


Task-specific target budgets, threshold offsets, and keep fractions live in the
[task configs](../src/needlework/configs/task/). Negative offsets admit more
candidates than the calibrated thresholds; positive offsets admit fewer. `summary.json`
records each stage's offset, accepted candidates, packed bridges, keep fraction,
and kept bridges.

## 4. Run workers and save

`gpus=[0,1]` uses visible CUDA ordinals. A bounded queue assigns fixed candidate
batches to persistent GPU workers. Faster workers receive more batches, while
candidate order and proposal seeds stay fixed.

Each completed batch writes `batches/NNNNNN.npz` atomically. It contains each
proposal's per-prefix logits and executable actions, each candidate's longest
eligible prefix, and a candidate identity check. After all batches finish, the
pipeline reads these files to apply acceptance, select prefixes, and pack and
filter bridges. Before `bridges.zarr` is written, resume may change
`threshold_offsets.stageN=` and `keep_fraction.stageN=`; the run configuration records
the changes. Resume validates existing batches and computes missing ones. Worker
errors propagate and workers are cleaned up. A worker that does not reply within
900 seconds raises an error.

> [!TIP]
> You can change GPU count when resuming a stitching run. Batch size must stay fixed.

`bridges.zarr` is written through a temporary sibling directory. Inspect and remove
a temporary store left by an interrupted final write before retrying. A completed
store is validated and reused. Do not edit a run's candidate or batch files.

Final arrays are:


| Array                          | Type/shape                 | Meaning                                                 |
| ------------------------------ | -------------------------- | ------------------------------------------------------- |
| `source_frame`, `target_frame` | int64 `[N]`                | Global raw-frame references                             |
| `stage`                        | int64 `[N]`                | 1, 2 or 3                                               |
| `verifier_margin`              | float32 `[N]`              | Chosen prefix's margin at its stage's offset thresholds |
| `action_start`, `action_len`   | int64 `[N]`                | Offset and length in packed policy steps                |
| `actions`                      | float32 `[sum(length), A]` | Absolute Robomimic or source-relative UMI               |


Group attributes specify format version, task, source-store identity, horizon, stride,
action convention, recipe and content hash. Validation checks endpoints, outcomes,
lengths, slices, uniqueness and finite values. Zero bridges is valid.

Storage estimates

Additional uncompressed storage is `44*N + 4*A*sum(length)` bytes plus small metadata.
For 1,000 bridges averaging 10 steps, that is about 444 KB for a one-arm task (`A=10`),
or 844 KB for bimanual data (`A=20`). There are no duplicated images, feature caches or
logged trajectories. Temporary candidates cost 32 bytes per pair. Batch files hold `4 * proposals * H * (1 + A) + 8` bytes per candidate: about 8 KB for a one-arm task (`A=10`,
eight proposals, `H=23`), 1.8 GB for 218,000 candidates, and 15 KB for bimanual data.
They are intermediate files; `bridges.zarr` is the result.



## Code map

Paths are relative to `src/needlework/`.


| File                               | Responsibility                                        |
| ---------------------------------- | ----------------------------------------------------- |
| `stitch.py`                        | Configure the run and handle resume                   |
| `stitching/pipeline.py`            | Validate inputs and coordinate generation and scoring |
| `stitching/candidates.py`          | Search for eligible visual candidates                 |
| `stitching/recovery.py`            | Gate failure sources by visual distance               |
| `stitching/inference.py`           | Generate IDM proposals and score them                 |
| `stitching/selection.py`           | Apply thresholds, votes, and source conflicts         |
| `stitching/workers.py`             | Schedule GPU batches and save results                 |
| `data/bridges.py`                  | Validate the bridge store                             |
| `sampling/augmented_dataset.py`    | Build policy windows with bridges                     |


`bridges.zarr` contains selected bridges; scoring details remain in the batch files.
