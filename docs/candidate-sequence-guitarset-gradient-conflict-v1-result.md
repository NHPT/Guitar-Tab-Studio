# Candidate sequence GuitarSet partition gradient conflict v1 result

Status: complete. The frozen training-only audit passed its input,
recording-family, cache, and zero-mutation checks. It did not create an
optimizer, update a parameter, write a checkpoint, or read governed GuitarSet,
AG-PT held-out, or Guitar-TECHS P2 result reports.

Machine-readable report:
`reports/candidate-sequence-guitarset-gradient-conflict-v1.json`

Report SHA-256:
`ed6c9a95e58f1e3da2f0777070a884d2ebb1640c1b90660cf0740f6e36342e7f`

## Audit integrity

| Check | Result |
|---|---|
| Frozen plan SHA-256 | `345932175548254502be20049ec222149901a136508ef0685b0cf833ab157866` |
| Gradient units | 384 tracks, 96 per player |
| Calibration units | 96 tracks, 24 per player |
| Recording-family clusters | 240 |
| Families split across both sets | 84 |
| Gradient groups / windows | 55,946 / 3,288 |
| Calibration groups / windows | 13,625 / 801 |
| Sequence parameters | 93,190 in 16 tensors |
| Bootstrap | 10,000 family-cluster replicates, seed `20261124` |
| Parameter mutation | none |

Feature precomputation used MPS and created no new feature-cache entries.
Every sequence gradient was computed on CPU with both frozen models in
evaluation mode.

## Primary result

The audit does not find stable conflict between the 384 gradient tracks and
the 96 calibration tracks. Every preregistered 95% interval crosses zero under
both objective definitions.

| Module | Operational cosine | 95% CI | Common-unweighted cosine | 95% CI |
|---|---:|---:|---:|---:|
| Token encoder | `-0.126` | `[-0.881, 0.963]` | `0.541` | `[-0.870, 0.979]` |
| Sequence encoder | `-0.187` | `[-0.740, 0.888]` | `0.615` | `[-0.692, 0.939]` |
| Residual hidden | `0.000` | `[-0.446, 0.783]` | `0.622` | `[-0.149, 0.852]` |
| Residual output | `0.584` | `[-0.378, 0.868]` | `0.643` | `[-0.244, 0.865]` |
| All parameters | `0.480` | `[-0.404, 0.843]` | `0.608` | `[-0.328, 0.864]` |

All ten classifications are `inconclusive`; none meets the frozen `conflict`
criterion. The positive complete-model point estimates do not prove stable
alignment because the intervals remain wide, but they rule out describing the
v5 regression as demonstrated partition-wide antagonism.

The complete-model calibration/gradient mean-norm ratio is `1.256x` under
both objectives. This is not a source-scale failure comparable to the earlier
P3/GuitarSet ratio of `59.6x`.

## Weighting and aggregation

The operational training weights materially rotate the 384-track mean
gradient:

| Module | Operational vs unweighted cosine | Unweighted/operational norm |
|---|---:|---:|
| Token encoder | `-0.115` | `1.784x` |
| Sequence encoder | `0.148` | `1.701x` |
| Residual hidden | `0.340` | `0.778x` |
| Residual output | `0.573` | `0.869x` |
| All parameters | `0.430` | `1.000x` |

The weighted and unweighted complete-model means have almost identical norms
but substantially different directions. The token-encoder means are slightly
opposed. This is objective-induced rotation, not evidence that the two track
partitions inherently disagree.

Aggregation also changes the trunk interpretation. Under the operational
objective, equal-track token and sequence cosines are `-0.126` and `-0.187`,
but active-candidate-exposure weighting changes them to `0.698` and `0.525`.
The complete-model exposure-weighted cosine is `0.558`; under common unit
weights it is `0.608`.

| Source/objective | Active candidate exposures | Mean loss |
|---|---:|---:|
| Gradient, operational | 109,007 | `0.1527` |
| Gradient, common unweighted | 109,007 | `0.0940` |
| Calibration, unweighted | 26,342 | `0.0957` |

The training weights increase the gradient-side mean loss by about 62% while
leaving candidate identity unchanged.

## Stratified diagnostics

Every player-specific complete-model point estimate is positive:

| Objective | Player 00 | Player 01 | Player 02 | Player 03 |
|---|---:|---:|---:|---:|
| Operational | `0.816` | `0.549` | `0.539` | `0.761` |
| Common unweighted | `0.703` | `0.713` | `0.700` | `0.774` |

Original-only and derived-only pooled estimates are unstable and all of their
family-bootstrap intervals cross zero. They do not establish variant-specific
conflict.

The 84 split recording families mostly align across the partition:

| Objective | Mean within-family cosine | Median | Range |
|---|---:|---:|---:|
| Operational | `0.731` | `0.854` | `[-0.290, 0.997]` |
| Common unweighted | `0.757` | `0.883` | `[-0.351, 0.996]` |

This paired evidence argues against original/derived duplication being the
cause of a systematic 384-versus-96 conflict.

## Interpretation

V5 did not fail because the 384 GuitarSet tracks consistently optimize
against the 96 withheld GuitarSet tracks. The audit also does not show a large
partition norm imbalance.

The remaining supported mechanisms are:

- replacement replay uses a finite random multiset rather than every one of
  the 3,288 windows exactly once;
- candidate exposure and track weighting can materially change trunk
  direction;
- the operational supervision weights rotate the string-v1 gradient even
  when their complete-model norm stays almost unchanged;
- minimizing weighted cross-entropy need not preserve the hard accuracy and
  repeat-count gate at an already strong checkpoint.

PCGrad, module freezing, or another P3 ratio is not supported as the next
response. No stable negative interval was observed, and P3 was already absent
from v5.

## Next experiment

Run one `v6` GuitarSet-only control with exactly one change from v5:

1. keep the same string-v1 initialization, 0% P3 optimizer exposure, objective,
   optimizer, learning-rate schedule, three epochs, and training-only gate;
2. replace weighted sampling with deterministic nonreplacement traversal;
3. use each of the 3,288 GuitarSet windows exactly once per epoch;
4. log exact window, track, player, variant, and family coverage;
5. retain zero-update source-gradient diagnostics and do not read external
   evaluation unless an epoch passes the unchanged gate.

If v6 still regresses, replacement coverage is excluded and continued
fine-tuning itself should stop. The next mechanism would then be explicit
string-v1 anchoring or a trust-region update, not another replay ratio.

If v6 passes, reintroduce 1% P3 only after preserving deterministic
full-coverage GuitarSet replay.

## Verification

Before the audit, the worker suite passed 187 tests. Ruff, Python compilation,
input-contract validation, and whitespace checks passed. The generated report
records `optimizer_created=false` and
`parameter_mutation_detected=false`.
