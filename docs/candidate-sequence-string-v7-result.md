# Candidate sequence string v7 result

Status: complete. The frozen string-v1 module trust regions prevented drift
at epoch 3 and the unchanged production gate passed at epochs 2 and 3. Epoch
3 was selected and checkpointed. No governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v7 preregistration | `49aa18e6ab3a57ccaae13c7bda555eac770fe704f4b3be1c2d55cc6e08012f0b` |
| anchor calibration | `460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f` |
| input audit | `2974b36060b22e00ba585828d5981eb953a1a9b344054bd34baeace2acc3aeb7` |
| training log | `c53101d8f17d80fbcb7e8ec7f889900549ec16df52f74ba797061412d7b302e1` |
| selected checkpoint | `3ff0012a25fcf7e5ffa7f72e62be8a5d922e7cfff41a55299586589211afd8a3` |

The input audit passed and locked the calibration report hash before
training. The selected checkpoint identifies epoch 3, the same calibration
hash, all four frozen radii, post-AdamW projection, unchanged optimizer
moments, nonreplacement replay, and zero P3 training gradient.

Before calibration, the StringTrace ML suite passed 192 tests. The first
training attempt stopped at its first optimizer step because float32
write-back put one projected module `8.6e-8` above its radius, exceeding the
frozen `radius * 1e-6` tolerance. It produced only the epoch-zero log and no
checkpoint or epoch result. A direction-preserving numerical refinement was
added to move a quantized projection minimally inward until the frozen cap
holds. A high-dimensional nonzero-anchor regression test reproduces this
case. The final suite passed 193 tests; Ruff and Python compilation also
passed. The calibration and input-audit files were not regenerated, and
their hashes remained unchanged before the clean restart.

## Anchor calibration

Calibration used all 3,288 eligible GuitarSet windows, 55,946 unique
GuitarSet candidate groups, and 109,007 active candidate-window exposures in
206 batches. Both models ran in evaluation mode. No optimizer was created,
no parameter changed, and no checkpoint was written.

The complete objective-faithful gradient norm was `0.036312265`, versus the
source-local audit reference `0.036592013` (`0.9924x`). The global gradient
clip scale was therefore exactly `1.0`.

| Module | Raw gradient | Audit reference | Virtual AdamW radius |
|---|---:|---:|---:|
| Token encoder | `0.020399025` | `0.021608929` | `0.003597844` |
| Sequence encoder | `0.011282472` | `0.011811497` | `0.013503443` |
| Residual hidden | `0.004698044` | `0.004798117` | `0.005545075` |
| Residual output | `0.027442569` | `0.026636370` | `0.000987364` |
| All parameters | `0.036312265` | `0.036592013` | `0.015066857` |

The complete adaptive-update norm was `0.015066869`; the weight-decay update
norm was `1.2333e-7`. The small reference differences are expected because
the actual v7 dataset materializes the diagnostic-only P3 tracks before
computing frozen rare-pair weights.

## Coverage integrity

Every epoch traversed the complete eligible GuitarSet window set exactly
once:

| Epoch | Draws | Unique | Repeated | Omitted | Unexpected | P3 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,288 | 3,288 | 0 | 0 | 0 | 0 |
| 2 | 3,288 | 3,288 | 0 | 0 | 0 | 0 |
| 3 | 3,288 | 3,288 | 0 | 0 | 0 | 0 |
| Total | 9,864 | 9,864 | 0 | 0 | 0 | 0 |

All observed window-set identities equal the frozen eligible identity
`560d99d445b0daa163dc764611ba4f0ea6086bac4039815b26c338c208d60b6a`.
Each epoch contains 109,007 active candidate-window exposures.

## Trust-region integrity

All 618 optimizer steps completed. Every maximum post-projection
displacement remained within `radius + max(1e-12, radius * 1e-6)`.

| Module | Radius | Projections | Fraction | Maximum before | Maximum after |
|---|---:|---:|---:|---:|---:|
| Token encoder | `0.003597844` | 616 / 618 | `99.68%` | `0.005874400` | `0.003597847` |
| Sequence encoder | `0.013503443` | 618 / 618 | `100%` | `0.021189904` | `0.013503452` |
| Residual hidden | `0.005545075` | 618 / 618 | `100%` | `0.008572065` | `0.005545080` |
| Residual output | `0.000987364` | 618 / 618 | `100%` | `0.001749920` | `0.000987365` |

Epoch-end boundary occupancy stayed within approximately one part per
million of each radius. The optimizer moments were never reset or projected.

## Frozen decisions

Metric deltas are relative to the exact string-v1 epoch-zero baseline:

| Metric | Baseline | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815060604` | `0` | `+0.000109691` | `+0.000164537` |
| Wrong strings | `3372` | `0` | `-2` | `-3` |
| Repeat true positives | `7552` | `-2` | `0` | `+1` |
| GuitarSet accuracy | `0.963257430` | `+0.000077190` | `+0.000154381` | `+0.000077190` |
| AG-PT accuracy | `0.462737561` | `-0.000427077` | `-0.000213538` | `+0.000427077` |
| Guitar-TECHS P1 accuracy | `0.440909091` | `+0.004545455` | `+0.004545455` | `0` |
| AG-PT/P1 macro accuracy | `0.451823326` | `+0.002059189` | `+0.002165958` | `+0.000213538` |
| Drift prevented | - | no | no | **yes** |
| Production eligible | - | no | **yes** | **yes** |

Epoch 1 exactly preserves aggregate accuracy and wrong-string count but loses
two repeated-string true positives. Epoch 2 passes the production gate but
does not satisfy the stricter drift decision because AG-PT decreases. Epoch
3 is the only epoch where all seven preservation metrics are non-regressing.
It also has the highest aggregate accuracy and more repeat true positives, so
the frozen selection order chooses epoch 3.

## Source-gradient trajectory

P3 remained diagnostic-only:

| Epoch | GuitarSet vs P3 cosine | P3/GuitarSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `0.748` | `42.8x` |
| 2 | `0.746` | `40.8x` |
| 3 | `0.662` | `48.6x` |

These diagnostics do not enter optimization or radius calibration.

## Causal result

The v7 question is resolved positively. Limiting each module to one
objective-faithful virtual first-AdamW displacement from string-v1 is
sufficient to prevent the v4-v6 preservation drift under 0% P3 exact
full-coverage replay. It also produces a checkpoint that passes the unchanged
production gate.

This result supports the parameter anchor mechanism; it does not establish
that P3 replay is beneficial. The next controlled experiment may reintroduce
the frozen 1% P3 replay while retaining these exact four radii and all other
v7 mechanics. External evaluation remains separately governed and was not
opened during v7.
