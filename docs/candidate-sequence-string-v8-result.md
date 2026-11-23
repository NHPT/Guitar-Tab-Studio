# Candidate sequence string v8 result

Status: complete. The frozen 1% P3 injection control finished normally.
Epochs 1 and 3 prevented drift and passed the unchanged production gate.
Epoch 1 was selected and checkpointed. No governed GuitarSet, AG-PT
held-out, or Guitar-TECHS P2 result report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v8 preregistration | `97014e726ec595ef9f689adba76fe69340afb2639a9e4b2edce338b67d7cb0ba` |
| inherited v7 anchor calibration | `460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f` |
| v8 input audit | `417760f62cb5576bde0393c3015715867ff444fdb97bafd9d83ae528ac96a901` |
| v8 training log | `30850a58f3a8847e390303da69ecc5e1f4356cd01cfc84e1e3ff73f327e49a63` |
| selected v8 checkpoint | `0bf00091daaaf6c8428c982d1bf0e30165460eeae821a96a7552bbad83aad726` |

The input audit passed and locked the unchanged v7 calibration hash and
radii before training. The selected checkpoint identifies epoch 1,
string-v1 initialization, the same four trust regions, exact GuitarSet
full-coverage replay, and the frozen `33 / 33 / 34` P3 schedule.

Before training, the StringTrace ML suite passed 195 tests. Ruff, Python
compilation, and whitespace checks also passed.

## Frozen single-variable boundary

V8 did not recalibrate or change the anchor:

| Module | Radius |
|---|---:|
| Token encoder | `0.0035978438540137402` |
| Sequence encoder | `0.013503442576068494` |
| Residual hidden | `0.005545074699143554` |
| Residual output | `0.0009873639570436095` |

Relative to v7, the only policy change was injecting P3 windows. The
GuitarSet permutation used the unchanged seed and a separate random stream
from P3. A sampler regression test confirmed that removing the injected P3
indexes reproduces all three v7 GuitarSet permutations exactly.

## Replay integrity

Every epoch retained all 3,288 GuitarSet windows exactly once. P3 windows
were selected uniformly without replacement within each epoch:

| Epoch | GuitarSet | GSet duplicate / omitted | P3 | P3 duplicate | Total | Batches |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,288 | `0 / 0` | 33 | 0 | 3,321 | 208 |
| 2 | 3,288 | `0 / 0` | 33 | 0 | 3,321 | 208 |
| 3 | 3,288 | `0 / 0` | 34 | 0 | 3,322 | 208 |
| Total | 9,864 | `0 / 0` | 100 | 0 | 9,964 | 624 |

All three observed GuitarSet identities equal the frozen eligible identity:

`560d99d445b0daa163dc764611ba4f0ea6086bac4039815b26c338c208d60b6a`.

The cumulative P3 window share was `1.003613%`, the closest error-balanced
integer realization to 1%. Candidate-window exposures were:

| Source | Epoch 1 | Epoch 2 | Epoch 3 | Total |
|---|---:|---:|---:|---:|
| GuitarSet | 109,007 | 109,007 | 109,007 | 327,021 |
| P3 | 754 | 680 | 744 | 2,178 |

P3 therefore contributed `0.661606%` of active candidate-window exposures.
This difference from the window share is descriptive; no source-specific
loss multiplier was applied.

## Trust-region integrity

All 624 optimizer steps completed. Every maximum post-projection
displacement remained within `radius + max(1e-12, radius * 1e-6)`.

| Module | Projections | Fraction | Maximum before | Maximum after |
|---|---:|---:|---:|---:|
| Token encoder | 610 / 624 | `97.76%` | `0.005540721` | `0.003597847` |
| Sequence encoder | 624 / 624 | `100%` | `0.021189904` | `0.013503448` |
| Residual hidden | 623 / 624 | `99.84%` | `0.008572065` | `0.005545079` |
| Residual output | 621 / 624 | `99.52%` | `0.001749920` | `0.000987365` |

Optimizer moments were never reset or projected.

## Frozen decisions

Metric deltas are relative to the exact string-v1 epoch-zero baseline:

| Metric | Baseline | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815060604` | `+0.000274228` | `+0.000109691` | `+0.000109691` |
| Wrong strings | `3372` | `-5` | `-2` | `-2` |
| Repeat true positives | `7552` | `+2` | `-1` | `0` |
| GuitarSet accuracy | `0.963257430` | `+0.000154381` | `+0.000077190` | `0` |
| AG-PT accuracy | `0.462737561` | `+0.000640615` | `0` | `+0.000427077` |
| Guitar-TECHS P1 accuracy | `0.440909091` | `0` | `+0.004545455` | `0` |
| AG-PT/P1 macro accuracy | `0.451823326` | `+0.000320307` | `+0.002272727` | `+0.000213538` |
| Drift prevented | - | **yes** | no | **yes** |
| Production eligible | - | **yes** | no | **yes** |

Epoch 2 fails both decisions only because it loses one repeated-string true
positive. Epoch 1 has the highest aggregate accuracy and the most repeat true
positives, so the frozen selection order chooses epoch 1.

## Comparison with v7

The selected v8 epoch is stronger than the only drift-preventing v7 epoch on
every metric that changes:

| Delta from string-v1 | v7 epoch 3 | v8 epoch 1 |
|---|---:|---:|
| Aggregate accuracy | `+0.000164537` | `+0.000274228` |
| Wrong strings | `-3` | `-5` |
| Repeat true positives | `+1` | `+2` |
| GuitarSet accuracy | `+0.000077190` | `+0.000154381` |
| AG-PT accuracy | `+0.000427077` | `+0.000640615` |
| Guitar-TECHS P1 accuracy | `0` | `0` |
| Cross-source macro accuracy | `+0.000213538` | `+0.000320307` |

This establishes compatibility and a better selected training-only outcome.
It does not by itself prove that P3 content caused the improvement. Injecting
P3 mechanically increased the run from 618 to 624 optimizer steps. A pure
source-content attribution would require an equal-step anchored control that
injects `33 / 33 / 34` additional GuitarSet replay windows at the same
positions.

## Source-gradient trajectory

| Epoch | GuitarSet vs P3 cosine | P3/GuitarSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `0.700` | `47.7x` |
| 2 | `0.738` | `41.2x` |
| 3 | `0.511` | `55.2x` |

The diagnostic P3 gradient remains much larger than the GuitarSet mean, but
the frozen trust regions prevent that exposure from causing preservation
drift at the selected epoch.

## Causal result

The preregistered v8 question is resolved positively: nominal 1% P3 window
injection is compatible with the frozen v7 parameter anchor, exact GuitarSet
coverage, the strict no-regression decision, and the unchanged production
gate.

The selected v8 checkpoint is a stronger training-only candidate than v7.
External evaluation remains separately governed and was not opened during
this experiment. For a narrower claim that P3 source content, rather than the
six additional optimizer steps, caused the improvement, the next experiment
must be the equal-step GuitarSet injection control described above.
