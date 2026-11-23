# Candidate sequence string v5 result

Status: complete. The frozen GuitarSet-only control finished normally. No
epoch passed the unchanged training-only preservation gate, so no checkpoint
was written and no governed GuitarSet, AG-PT held-out, or Guitar-TECHS P2
result report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v5 preregistration | `e6d19577e722b4eaae0a52dbd260a64e29231439628fba0894872eb22abbd6a8` |
| v5 input audit | `e086978cef0cabd4783f34ac3d21d94b364f52b0a237ae203cc456800d15ed77` |
| v5 training log | `f01406d0fc1c97eb88ca416d8832a90991640ddb402f0f22571b15049ea22ffb` |
| v5 checkpoint | not created |

The input audit passed. The implementation and existing worker suite passed
186 tests before training.

## Control integrity

Every sampled training window was GuitarSet:

| Epoch | GuitarSet windows | P3 windows | Total |
|---|---:|---:|---:|
| 1 | 3,489 | 0 | 3,489 |
| 2 | 3,489 | 0 | 3,489 |
| 3 | 3,489 | 0 | 3,489 |
| Total | 10,467 | 0 | 10,467 |

P3 remained present only in the zero-update source-gradient diagnostic. It
never entered the optimizer.

## Training-only gate

Epoch zero exactly reproduced the frozen string-v1 baseline.

| Metric | Baseline | Epoch 1 delta | Epoch 2 delta | Epoch 3 delta |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815061` | `-0.000603` | `-0.000548` | `-0.000713` |
| Wrong strings | `3372` | `+11` | `+10` | `+13` |
| Repeat true positives | `7552` | `-1` | `-3` | `-2` |
| GuitarSet accuracy | `0.963257` | `-0.000309` | `-0.000386` | `-0.000232` |
| AG-PT accuracy | `0.462738` | `-0.001281` | `-0.001068` | `-0.001708` |
| Guitar-TECHS P1 accuracy | `0.440909` | `-0.004545` | `0` | `0` |
| AG-PT/P1 macro accuracy | `0.451823` | `-0.002913` | `-0.000534` | `-0.000854` |
| Eligible | - | no | no | no |

Epoch 2 was closest by aggregate accuracy and wrong-string count. It still
failed aggregate accuracy, wrong strings, repeat true positives, GuitarSet
accuracy, and cross-source macro accuracy.

## Causal result

The preregistered control question is resolved: continued fine-tuning without
P3 is sufficient to cause regression. The remaining v4 failure cannot be
attributed to its 1% P3 sampler mass.

The best aggregate-accuracy epochs compare as follows:

| Metric delta from string-v1 | v4 epoch 1, 1% P3 | v5 epoch 2, 0% P3 |
|---|---:|---:|
| Aggregate accuracy | `-0.000494` | `-0.000548` |
| Wrong strings | `+9` | `+10` |
| Repeat true positives | `+2` | `-3` |
| GuitarSet accuracy | `-0.000154` | `-0.000386` |
| AG-PT accuracy | `-0.001708` | `-0.001068` |
| Guitar-TECHS P1 accuracy | `0` | `0` |
| Cross-source macro accuracy | `-0.000854` | `-0.000534` |

V4 and v5 use different weighted-sampler realizations after the probability
mass changes, so these small differences do not establish that 1% P3 is
beneficial. They do establish that removing P3 does not restore the gate.

## Source-gradient trajectory

The complete-model diagnostics were:

| Epoch | GSet vs P3 cosine | P3/GSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `0.470` | `49.3x` |
| 2 | `0.888` | `26.9x` |
| 3 | `0.874` | `26.7x` |

P3 receives no update, but its diagnostic gradient changes because the shared
model changes under GuitarSet replay. The positive trajectory does not alter
the control result: calibration metrics regress without any P3 training
gradient.

## Interpretation

String v1 was already trained on all 480 GuitarSet training tracks. V4/v5
fine-tuning gradients use only the 384 non-calibration GuitarSet tracks while
the 96 selected GuitarSet tracks are withheld for epoch selection. The v5
result is consistent with the model moving toward the 384-track replay subset
and away from the withheld training tracks and cross-source calibration data.

Replacement sampling is another unresolved factor. Drawing 3,489 windows from
3,288 GuitarSet windows with replacement omits a substantial fraction of
unique windows in each epoch while repeating others. V5 shows that blindly
continuing the same optimizer schedule is not a neutral replay operation.

## Next step

Do not start another P3 sampling-ratio experiment and do not rescue a v5
epoch. The next diagnostic should compare gradients from:

- the 384 GuitarSet gradient tracks;
- the 96 GuitarSet calibration tracks;
- AG-PT/P1 calibration as a separate preservation reference.

That audit should determine whether conflict is intra-GuitarSet, cross-source,
or primarily stochastic replacement drift. A later training experiment should
then isolate one preservation mechanism, such as nonreplacement full-coverage
replay or an explicit string-v1 anchoring loss. External evaluation remains
blocked.
