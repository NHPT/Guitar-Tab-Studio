# Candidate sequence string v6 result

Status: complete. The frozen nonreplacement full-coverage control finished
normally. No epoch passed the unchanged training-only preservation gate, so no
checkpoint was written and no governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v6 preregistration | `5be7175bb428ee48d6abd8c2f5f52c4d2b75498014ef984bf00a0f06cfcac11e` |
| v6 input audit | `81f8d0712f86076c22dc446761e706cc63d226eff321550d28da72baf9e889b9` |
| v6 training log | `cccfddda70b3ff960eea0cd09e6a57d3892ae9a69e2bf453a94077aff90ac520` |
| v6 checkpoint | not created |

The input audit passed. Before training, the worker suite passed 189 tests;
Ruff, Python compilation, and whitespace checks also passed.

## Full-coverage integrity

Every epoch traversed the complete eligible GuitarSet window set exactly once:

| Epoch | Draws | Unique | Repeated | Omitted | P3 |
|---|---:|---:|---:|---:|---:|
| 1 | 3,288 | 3,288 | 0 | 0 | 0 |
| 2 | 3,288 | 3,288 | 0 | 0 | 0 |
| 3 | 3,288 | 3,288 | 0 | 0 | 0 |
| Total | 9,864 | 9,864 | 0 | 0 | 0 |

All three observed window-set identities equal the frozen eligible identity:

`560d99d445b0daa163dc764611ba4f0ea6086bac4039815b26c338c208d60b6a`.

Each epoch covers:

- all 384 eligible GuitarSet tracks, identity
  `e1839751cee5f6e57149d9c93871211b833a9d714bac649df0d4113e00258709`;
- all 234 represented recording families, identity
  `b200319bfe1431fe63f69fc5b07f4856247b3bf054b2ad1fecc88ea5e571ea79`;
- 1,563 original-track windows and 1,725 derived-track windows;
- player window counts 867, 865, 832, and 724.

The sampler therefore satisfied the preregistered policy. Coverage failure
cannot explain the training result.

## Training-only gate

Epoch zero exactly reproduced the frozen string-v1 baseline.

| Metric | Baseline | Epoch 1 delta | Epoch 2 delta | Epoch 3 delta |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815061` | `-0.000658` | `-0.000768` | `-0.000932` |
| Wrong strings | `3372` | `+12` | `+14` | `+17` |
| Repeat true positives | `7552` | `-4` | `0` | `0` |
| GuitarSet accuracy | `0.963257` | `-0.000232` | `-0.000077` | `-0.000309` |
| AG-PT accuracy | `0.462738` | `-0.001708` | `-0.002349` | `-0.002776` |
| Guitar-TECHS P1 accuracy | `0.440909` | `-0.004545` | `-0.004545` | `0` |
| AG-PT/P1 macro accuracy | `0.451823` | `-0.003127` | `-0.003447` | `-0.001388` |
| Eligible | - | no | no | no |

Epoch 1 has the highest aggregate accuracy, but it still fails aggregate
accuracy, wrong strings, repeat true positives, GuitarSet accuracy, and
cross-source macro accuracy. No epoch is eligible for rescue.

## Comparison with v5

The best aggregate-accuracy epochs are:

| Metric delta from string-v1 | v5 epoch 2, replacement | v6 epoch 1, full coverage |
|---|---:|---:|
| Aggregate accuracy | `-0.000548` | `-0.000658` |
| Wrong strings | `+10` | `+12` |
| Repeat true positives | `-3` | `-4` |
| GuitarSet accuracy | `-0.000386` | `-0.000232` |
| AG-PT accuracy | `-0.001068` | `-0.001708` |
| Guitar-TECHS P1 accuracy | `0` | `-0.004545` |
| Cross-source macro accuracy | `-0.000534` | `-0.003127` |

V6 slightly reduces the GuitarSet accuracy regression but worsens every other
listed preservation metric. This does not support nonreplacement full coverage
as a preservation mechanism.

V5 and v6 differ mechanically in both replay multiplicity and update count:
v5 performs 3,489 draws / 219 batches per epoch, while v6 performs 3,288 draws
/ 206 batches. A passing result would not have separated those effects.
Because v6 fails, neither effect is required for the observed regression.

## Source-gradient trajectory

The complete-model zero-update diagnostics were:

| Epoch | GSet vs P3 cosine | P3/GSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `0.874` | `27.4x` |
| 2 | `0.891` | `24.7x` |
| 3 | `0.759` | `44.9x` |

P3 remains diagnostic-only. Its trajectory cannot cause the regression because
no P3 window enters the optimizer.

## Causal result

The v6 question is resolved: exact nonreplacement coverage of all 3,288
GuitarSet windows is still sufficient to regress the unchanged training-only
gate.

Across v4-v6:

- reducing P3 from 20% to 1% did not pass;
- removing P3 entirely did not pass;
- removing replacement omission and duplication did not pass;
- the 384-versus-96 audit did not demonstrate stable partition conflict.

The remaining evidence points to unconstrained movement away from an already
strong string-v1 checkpoint under a weighted cross-entropy objective whose
direction is not identical to the hard preservation gate.

## Next step

Stop replay-only and P3-ratio experiments. Do not rescue a v6 epoch.

The next experiment should add an explicit string-v1 preservation mechanism:

1. initialize from the same complete string-v1 checkpoint;
2. retain deterministic nonreplacement GuitarSet coverage and 0% P3 for the
   first anchoring control;
3. add one preregistered parameter-space anchor to the frozen string-v1
   weights, applied to all 93,190 trainable parameters;
4. report task loss, anchor loss, parameter displacement by module, gradient
   norm before/after anchoring, and the unchanged training-only gate;
5. read no external evaluation unless an epoch passes.

The anchor strength must be selected without reading governed or held-out
results. It should be calibrated from the epoch-zero task-gradient and anchor
scales, then frozen before training. P3 should be reintroduced only after the
anchoring control preserves string v1.
