# Candidate sequence GuitarSet partition gradient conflict v1 audit plan

Status: frozen before implementation and before any gradient extraction. This
is a training-only diagnostic. It performs no optimizer step, writes no
checkpoint, and does not read governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result reports.

## Question

At the frozen string-v1 checkpoint, does the v5 gradient from the 384
non-calibration GuitarSet tracks conflict with the preservation gradient from
the 96 withheld GuitarSet calibration tracks?

The audit distinguishes four possible explanations:

1. a stable partition-wide direction conflict;
2. conflict isolated to a model module, player, or audio variant;
3. an apparent conflict caused by the training-side supervision weights;
4. gradient magnitude or replacement-sampling drift without stable directional
   conflict.

This audit does not estimate generalization, select a checkpoint, tune a loss,
or authorize another training run.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| v5 input audit | `e086978cef0cabd4783f34ac3d21d94b364f52b0a237ae203cc456800d15ed77` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The candidate frontend remains Basic Pitch onset `0.35`, frame `0.25`, and
minimum note length `70 ms`.

The 384-track gradient cache set contains 113,710 candidates, 823 dropped rows,
and has set SHA-256
`4aedd3c1e181b7d2470e42d8a5f1c748e3ad629536d1df3f648c0e6111b1dfb5`.
The 96-track calibration cache set contains 27,445 candidates, 230 dropped
rows, and has set SHA-256
`57c7b74f47cb7f21fd9bffe192bbd25ccb1fb4145c1ad868eb567a6a10efa70c`.

## Frozen partition

The v5 partition is reproduced with fraction `0.20` and seed `20261122`.
Only `split=train`, `dataset=GuitarSet` tracks are eligible. P3 and all other
datasets are excluded.

| Set | Tracks | Original | Derived | Track identity SHA-256 |
|---|---:|---:|---:|---|
| Gradient | 384 | 185 | 199 | `e1839751cee5f6e57149d9c93871211b833a9d714bac649df0d4113e00258709` |
| Calibration | 96 | 55 | 41 | `f5ad4b495801fca64848029e209ce7bad5a056aaccef44e5e088b067ea231335` |
| All GuitarSet train | 480 | 240 | 240 | `b36882afc7c5a8e407aea1f467aac29cd5ea7d46682f31f8a8e0302e4e653639` |

Each of the four player groups contributes 96 gradient tracks and 24
calibration tracks.

The 480 tracks form 240 recording families. A derived track is assigned to
the family named by `provenance.derived_from_track_id`; an original track is
its own family root. Every family must contain exactly one original and one
derived track with identical annotation SHA-256 values. The family-root
identity SHA-256 is
`c9c1a1535103a523e3fdd9127108e04e160082bf626c549d6af95a43057999bd`.

The frozen family allocation is:

| Allocation | Families |
|---|---:|
| both tracks in gradient | 150 |
| original calibration, derived gradient | 49 |
| original gradient, derived calibration | 35 |
| both tracks in calibration | 6 |

Thus 84 families cross the partition. They are paired diagnostics, not 168
independent recordings.

## Frozen objectives

The primary `operational` comparison preserves the v5 objective asymmetry:

- gradient tracks use ambiguity weight `1.25`, rare-pair weight capped at
  `1.5`, neighboring cross-string weight `2.0` within two seconds, and final
  supervision weight capped at `4.0`;
- calibration tracks use the calibration objective's unit target weights.

Weights are computed in separate source-local GuitarSet datasets containing
exactly the 384 and 96 selected tracks. This avoids contamination from P3,
AG-PT, or Guitar-TECHS P1 while preserving the weighting policy applied to
each side.

The `common_unweighted` sensitivity comparison replaces every target weight
with one on both sides while keeping candidates, windows, logits, targets, and
model parameters unchanged. It tests whether an operational result survives
removal of the loss-weight asymmetry.

No result from one objective may be substituted for the other. A conflict
present only in `operational` is attributed to objective weighting, not to a
stable GuitarSet partition conflict.

## Frozen gradient definition

The complete 93,190-parameter candidate sequence string model is loaded from
the string-v1 checkpoint. The position/activity model is loaded only to
precompute candidate sequence groups. Both models run in evaluation mode.

Every sequence window for every selected track is used once with sequence
length `32` and stride `16`. For each track and objective, every batch mean
loss is multiplied by its number of active candidate-window exposures before
backward accumulation. The final gradient is divided by the track's total
active candidate-window exposures.

The resulting 480 candidate-count-normalized track gradients are the atomic
observations. Source point estimates use equal track weights. The audit also
reports an active-candidate-exposure-weighted cosine as a full-coverage
approximation to the replay objective.

There is no optimizer, gradient clipping, parameter update, stochastic
sampling, source weighting, or gradient surgery. Every parameter tensor is
compared before and after extraction; any mutation fails the audit.

## Frozen module partition

The parameter partition is disjoint and exhaustive:

| Module | Parameter prefix |
|---|---|
| token encoder | `token_encoder.` |
| sequence encoder | `sequence_encoder.` |
| residual hidden | `residual_head.0.` |
| residual output | `residual_head.3.` |
| all | concatenation of all four modules |

## Comparisons

For every module and objective, the primary point estimate is the cosine
between the equal mean of 384 gradient-track gradients and the equal mean of
96 calibration-track gradients.

The audit also reports:

- source mean-gradient norms and the calibration/gradient norm ratio;
- active-candidate-exposure-weighted source cosine and norm ratio;
- four player-specific gradient-versus-calibration cosines;
- original-only and derived-only gradient-versus-calibration cosines;
- the cosine between operational and unweighted gradient-source means;
- within-family gradient-versus-calibration cosine distributions for the 84
  split families.

## Recording-family cluster bootstrap

For each module and objective, track-gradient Gram matrices are computed once
in float64. Using seed `20261124`, 10,000 bootstrap replicates resample the 240
recording families with replacement. The same family counts are expanded to
both source track sets, preserving original/derived correlation and the 84
families shared across the partition.

The reported interval is the percentile 95% interval. Classification is
mechanical:

- `conflict`: the complete interval is below zero;
- `aligned`: the complete interval is above zero;
- `inconclusive`: the interval includes zero.

Player, variant, exposure-weighted, objective-agreement, and split-family
results are diagnostics rather than additional confirmatory tests.

## Decision rules

- Stable intra-GuitarSet conflict requires `conflict` for all parameters in
  both `operational` and `common_unweighted`.
- `operational=conflict` with `common_unweighted` not conflicting indicates
  loss-weight conflict.
- A conflict confined to one module, player, or variant is reported as local,
  not partition-wide.
- If direction is not stably negative but norms differ materially, the result
  is magnitude drift.
- Replacement drift remains plausible when equal-track and
  exposure-weighted directions differ materially; this audit does not
  reconstruct individual v5 sampler draws.

## Fixed outputs

- machine-readable report:
  `reports/candidate-sequence-guitarset-gradient-conflict-v1.json`;
- interpreted result:
  `docs/candidate-sequence-guitarset-gradient-conflict-v1-result.md`.

The report must include all frozen input identities, partition and family
identities, cache identities, candidate/window exposure counts, module
parameter counts, point estimates, bootstrap intervals, classifications, and
a zero-parameter-mutation assertion. It must not contain raw gradients.
