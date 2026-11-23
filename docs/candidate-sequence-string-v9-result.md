# Candidate sequence string v9 result

Status: complete. The frozen equal-step GuitarSet injection control finished
normally. Epoch 3 prevented drift and epochs 2-3 passed the unchanged
production gate. Epoch 3 was selected and checkpointed. The preregistered
matched-epoch source classification is `mixed_inconclusive`. No governed
GuitarSet, AG-PT held-out, or Guitar-TECHS P2 result report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v9 preregistration | `302a77e8225a3f9d0950ebadf60d7a7c0da379db129eb1fb1900b13f35e2e66a` |
| inherited v7 anchor calibration | `460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f` |
| v9 input audit | `0ba0357da90de0f9b9aeb0bb31e45315ddf4ecbc9028eb566f3826cc621e9fec` |
| v9 training log | `e1d241db4062f825a448842d90bad4f577eb6d005ebf457dcd523e9c3f2524dc` |
| selected v9 checkpoint | `8d3fc3f953d8e96240fa0492e9e2d92bf00f5085236e7f817ba6efa7b6eb0b93` |

The input audit passed and locked the v8 epoch-1 metrics, the unchanged v7
anchor calibration, the `33 / 33 / 34` schedule, 624 optimizer steps, and
the source-attribution rule before training.

Before training, the StringTrace ML suite passed 197 tests. Ruff, Python
compilation, and whitespace checks also passed.

## Equal-step integrity

V9 changed only the injected source from P3 to GuitarSet:

- string-v1 initialization remained unchanged;
- the four v7 radii remained unchanged;
- all three base GuitarSet permutations matched v8 exactly;
- all injection positions matched v8 exactly;
- total windows, batches, epochs, and optimizer steps matched v8;
- P3 received zero optimizer windows.

| Epoch | Base GSet | Injected GSet | Combined duplicates | P3 | Total | Batches |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,288 | 33 | 33 | 0 | 3,321 | 208 |
| 2 | 3,288 | 33 | 33 | 0 | 3,321 | 208 |
| 3 | 3,288 | 34 | 34 | 0 | 3,322 | 208 |
| Total | 9,864 | 100 | 100 | 0 | 9,964 | 624 |

Every base epoch had zero duplicates, zero omissions, all 384 tracks, all
234 recording families, and the frozen eligible window identity:

`560d99d445b0daa163dc764611ba4f0ea6086bac4039815b26c338c208d60b6a`.

Every injected subset was internally unique. It covered 33, 32, and 33
tracks and 32, 29, and 32 recording families by epoch.

## Candidate exposure

| Source role | Epoch 1 | Epoch 2 | Epoch 3 | Total |
|---|---:|---:|---:|---:|
| Base GuitarSet | 109,007 | 109,007 | 109,007 | 327,021 |
| Injected GuitarSet | 1,111 | 1,210 | 1,121 | 3,442 |
| Combined | 110,118 | 110,217 | 110,128 | 330,463 |

Injected GuitarSet contributed `1.041569%` of active candidate-window
exposures. V8 P3 contributed `0.661606%`; this density difference is part of
the source-content treatment because neither experiment applied a
source-specific loss multiplier.

## Trust-region integrity

All maximum post-projection displacements remained within the frozen
tolerance.

| Module | Projections | Fraction | Maximum before | Maximum after |
|---|---:|---:|---:|---:|
| Token encoder | 616 / 624 | `98.72%` | `0.005649375` | `0.003597847` |
| Sequence encoder | 624 / 624 | `100%` | `0.021189904` | `0.013503448` |
| Residual hidden | 623 / 624 | `99.84%` | `0.008572065` | `0.005545080` |
| Residual output | 624 / 624 | `100%` | `0.001749920` | `0.000987365` |

Optimizer moments were never reset or projected.

## V9 training decisions

Metric deltas are relative to the exact string-v1 epoch-zero baseline:

| Metric | Baseline | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815060604` | `0` | `+0.000109691` | `+0.000164537` |
| Wrong strings | `3372` | `0` | `-2` | `-3` |
| Repeat true positives | `7552` | `-1` | `0` | `+1` |
| GuitarSet accuracy | `0.963257430` | `+0.000077190` | `+0.000154381` | `+0.000077190` |
| AG-PT accuracy | `0.462737561` | `-0.000427077` | `-0.000213538` | `+0.000427077` |
| Guitar-TECHS P1 accuracy | `0.440909091` | `+0.004545455` | `+0.004545455` | `0` |
| AG-PT/P1 macro accuracy | `0.451823326` | `+0.002059189` | `+0.002165958` | `+0.000213538` |
| Drift prevented | - | no | no | **yes** |
| Production eligible | - | no | **yes** | **yes** |

Epoch 1 fails because it loses one repeat true positive and decreases AG-PT.
Epoch 2 passes the production gate but does not prevent drift because AG-PT
decreases. Epoch 3 passes both decisions and is selected.

## Primary matched-epoch attribution

V8 and v9 epoch 1 have the same scheduler stage, base permutation, injection
count and positions, batch count, optimizer-step count, initialization, and
anchor.

| Metric | V8 P3 injection | V9 GSet injection | Better |
|---|---:|---:|---|
| Aggregate accuracy | `0.815334832` | `0.815060604` | P3 |
| Wrong strings | `3367` | `3372` | P3 |
| Repeat true positives | `7554` | `7551` | P3 |
| GuitarSet accuracy | `0.963411810` | `0.963334620` | P3 |
| AG-PT accuracy | `0.463378176` | `0.462310485` | P3 |
| Guitar-TECHS P1 accuracy | `0.440909091` | `0.445454545` | GSet |
| AG-PT/P1 macro accuracy | `0.452143634` | `0.453882515` | GSet |

V8 is better on five metrics, including aggregate accuracy, wrong strings,
repeat true positives, GuitarSet, and AG-PT. V9 is better on P1 and therefore
the cross-source macro. Neither epoch Pareto-dominates the other across all
seven frozen metrics, so the preregistered classification is:

`mixed_inconclusive`.

This rejects both overly broad claims:

- the v8 improvement is not explained solely by six extra optimizer steps;
- P3 is not uniformly better than equal-step GuitarSet replay across every
  preservation metric.

## Secondary selected-outcome comparison

The frozen selection rules choose v8 epoch 1 and v9 epoch 3. V8's selected
checkpoint is no worse on all seven metrics and strictly better on six:

| Delta from string-v1 | V8 selected | V9 selected |
|---|---:|---:|
| Aggregate accuracy | `+0.000274228` | `+0.000164537` |
| Wrong strings | `-5` | `-3` |
| Repeat true positives | `+2` | `+1` |
| GuitarSet accuracy | `+0.000154381` | `+0.000077190` |
| AG-PT accuracy | `+0.000640615` | `+0.000427077` |
| Guitar-TECHS P1 accuracy | `0` | `0` |
| Cross-source macro accuracy | `+0.000320307` | `+0.000213538` |

V9 selected epoch 3 is exactly equal to v7 epoch 3 on all seven count-derived
metrics. The extra equal-step GuitarSet replay therefore did not improve the
selected training-only outcome over v7, while P3 injection did.

This secondary evidence supports P3 as the source of the stronger v8
selected checkpoint. It does not override the primary
`mixed_inconclusive` classification because v9 had a matched-epoch P1/macro
advantage.

## Source-gradient trajectory

| Epoch | GuitarSet vs P3 cosine | P3/GuitarSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `0.742` | `43.2x` |
| 2 | `0.748` | `40.7x` |
| 3 | `0.653` | `48.9x` |

P3 remained diagnostic-only throughout v9.

## Attribution result

The equal-step control resolves the broad question but not every
source-specific metric:

- six additional updates alone do not reproduce v8's selected improvement;
- P3 content is the best supported source of v8's stronger aggregate,
  repeat, GuitarSet, AG-PT, and selected-checkpoint results;
- equal-step GuitarSet replay produces a matched-epoch P1 and macro advantage;
- therefore a claim that P3 is uniformly superior across all seven metrics
  is not supported.

External evaluation remains separately governed and was not opened during
this control.
