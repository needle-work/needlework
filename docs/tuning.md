# Tuning a new bridge set

The shipped task configurations accompany the released IDM, verifier, and bridge stores.
Use those artifacts for the paper's policy runs. A newly trained IDM or verifier can
produce a different bridge set; tune the selection and sampler for that set. Keep
`seed.dataset` fixed across the logged baseline and augmented policies, and compare
with the same `seed.training` values. Change one setting at a time so its effect is
clear. [pipeline.md](pipeline.md) defines bridge selection;
[training.md](training.md#training-with-bridges) defines the policy windows.

## 1. Check the bridge set by stage

Read candidate counts by stage from `candidates.npy`, and accepted, packed, and kept
counts by stage from `summary.json`. Stage 1 skips ahead within a successful episode; stage 2 crosses
successful episodes; stage 3 starts in a failure episode. Compare the stage mix and
policy success with the logged baseline before adjusting settings. Start with the
stage contributing the most bridges, or the stage you intend to add.

| What the new data shows | First setting to change | Effect |
| --- | --- | --- |
| One stage dominates the kept set and a smaller, higher-margin subset is desired | Lower `task.stitch_keep_fraction.stageN` | Keeps fewer bridges, choosing the highest margins within that stage. |
| Too few retained bridges, but many candidates pass | Raise `task.stitch_keep_fraction.stageN` | Keeps more of the already accepted bridges; no new scoring needed. |
| Few candidates pass the verifier, or passing margins cluster near the threshold | Lower `task.stitch_threshold_offsets.stageN` to admit more prefixes; raise it for a stricter set | Changes which proposals pass before one bridge per source is chosen. |
| Stage 3 has few *candidates* because its cap binds | Raise `task.stitch_max_candidates_per_stage.stage3` | Scores more targets from eligible failure sources. |

Threshold offsets and keep fractions can be changed when resuming a scored stitching
run **before** `bridges.zarr` is written. They act separately by stage. The offset
controls which proposals pass; the keep fraction ranks the resulting one-per-source
bridges. Inspect the new stage counts after each change rather than carrying over a
bridge count from a different IDM or verifier.

## 2. Set policy exposure to those bridges

Train an augmented policy on the chosen store. Change the sampler only after choosing
which bridges it contains. These settings require policy retraining, not stitching.

| What you want to change | Setting and direction |
| --- | --- |
| A new bridge set needs more or less exposure at successful departures | Raise or lower `task.bridge_weight`. This is the first sampler setting to tune. |
| More or less training on approaches to a success bridge | Raise or lower `task.role_weights.approach`. |
| More or less recovery training at failure departures | Raise or lower `task.role_weights.failure_departure`. |
| Logged actions at selected success sources are especially useful, or less useful than the new bridge | Test a higher or lower `task.role_weights.twin` while holding the bridge set and `bridge_weight` fixed. The appendix calls these choices **Sample Original Source Actions More** and **Remove Original Source Actions**. |
| More or less logged data at frames skipped by a bridge | Raise or lower `task.role_weights.skipped`. |

The weights determine *relative draw frequency*, not a scalar multiplier on the loss.
At a success departure with a positive `bridge_weight`, the paired
logged-to-bridge draw-weight ratio is `role_weights.twin / bridge_weight`.
A higher `twin` gives more exposure to the recorded continuation; a higher
`bridge_weight` gives more to the bridge. Even when
logged actions are high quality, compare candidate ratios through the trained
policy. With
`epoch_length=windows`, setting `twin=0` also shortens an epoch by excluding twin
windows.
A larger bridge set also increases bridge exposure at the same weights. In Can and
Square, `bridge_weight` controls success departures while the other roles have their
own configured weights. Transport links `failure_departure` and `approach` to
`bridge_weight`; changing it affects all three roles there. Check the active task
config before interpreting a weight sweep. Keep `task.epoch_length` and
`task.sampler_options` at the shipped values when comparing doses: they also change
the training schedule or which action rows are supervised.

For a *new raw dataset*, decide whether logged actions in failure episodes are useful
targets. `task.sampler_options.source_rows=supervised` retains logged action rows
before a failure-source bridge; `masked` excludes them. The action at departure comes
from the bridge in either case. This changes the training targets, so make the choice
before comparing sampler weights. The shipped task configs specify the choice for the
released datasets.

## 3. Expand candidate search if selection has too little to work with

When a stage has too few candidates *before* verifier scoring, changing its threshold
or keep fraction cannot create new source-target pairs. Rerun stitching with the
smallest search change that addresses the shortfall:

| Data condition | Setting and direction |
| --- | --- |
| Eligible sources are exhausted by a budget or spacing | Raise `task.stitch_sources_per_stage` or lower `task.stitch_candidate_spacing`. |
| Sources have too few targets to try | Raise `task.stitch_targets_per_source.stageN`; scoring cost grows with targets. |
| A stage cap discards eligible pairs | Raise `task.stitch_max_candidates_per_stage.stageN` or set it to `null`. |
| Few failure sources pass the visual coverage gate | Lower `recovery_quantile`; this admits sources closer to successful frames. |
| Target search is too narrow in visual distance | Lower `proximity.tau_percentile` to include more distant targets, or adjust `proximity.shells` to redistribute targets across distance bands. |

`proposals` adds IDM samples per pair at proportional inference cost. Increase it
when the vote requirement is the limiting factor; `min_votes` raises or lowers the
number of distinct passing samples required. `min_steps_saved` sets the minimum
route shortening for stages 1 and 2; raising it focuses on longer skips. These
changes need a new scoring run.

## 4. Retrain components only when the bridge set calls for it

Once candidate search and policy exposure are set, component training offers another
way to change the bridge set. For IDM proposals, adjust its training data and recipe.
For verifier acceptance, adjust its balance of within-episode and cross-episode
negative goals (`sampling.negative_ratio`) or its near-horizon band
(`sampling.boundary_prob`, `sampling.boundary_horizons`). A new checkpoint produces a
new bridge set: repeat the selection and policy exposure steps above. The stitch
threshold offset adjusts acceptance while keeping the verifier checkpoint fixed.
