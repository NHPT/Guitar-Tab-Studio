# Candidate sequence string v4 result

Status: complete. The single frozen three-epoch run finished normally, but no
epoch passed the unchanged training-only preservation gate. No checkpoint was
written and no governed GuitarSet, AG-PT held-out, or Guitar-TECHS P2 result
report was read.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v4 preregistration | `f92176cbe06069839f0099e485d8c79a6460643dc432a08d23554ec490f9f3dd` |
| v4 input audit | `87b9e4d3d7bf01c971479e66712672ab6a55d6ce5ec07ca807d9ae1dbd3761da` |
| v4 training log | `ad09d70c580a5490f0e6e45ef58cfae2913912d1236c50a71e0951571b432f0e` |
| v4 checkpoint | not created |

The input audit passed. The implementation and existing worker suite passed
185 tests before training.

## Sampler verification

The frozen P3 weight `0.1652344338911503` produced the intended 1% expected
sampler mass. Actual P3 draws were:

| Epoch | P3 windows | Total windows | Actual share |
|---|---:|---:|---:|
| 1 | 36 | 3,489 | `1.032%` |
| 2 | 36 | 3,489 | `1.032%` |
| 3 | 30 | 3,489 | `0.860%` |
| Total | 102 | 10,467 | `0.974%` |

The recording sampler was regression-tested against PyTorch's original
`WeightedRandomSampler` and produced the same seeded index sequence.

## Training-only gate

Epoch zero exactly reproduced the frozen string-v1 baseline.

| Metric | Baseline | Epoch 1 delta | Epoch 2 delta | Epoch 3 delta |
|---|---:|---:|---:|---:|
| Aggregate accuracy | `0.815061` | `-0.000494` | `-0.001097` | `-0.000878` |
| Wrong strings | `3372` | `+9` | `+20` | `+16` |
| Repeat true positives | `7552` | `+2` | `-1` | `+3` |
| GuitarSet accuracy | `0.963257` | `-0.000154` | `-0.000232` | `0` |
| AG-PT accuracy | `0.462738` | `-0.001708` | `-0.003630` | `-0.003203` |
| Guitar-TECHS P1 accuracy | `0.440909` | `0` | `0` | `+0.004545` |
| AG-PT/P1 macro accuracy | `0.451823` | `-0.000854` | `-0.001815` | `+0.000671` |
| Eligible | - | no | no | no |

Epoch 1 was closest by aggregate accuracy and wrong-string count, but still
failed four mandatory conditions: aggregate accuracy decreased, wrong strings
increased, GuitarSet accuracy decreased, and cross-source macro accuracy
decreased.

Epoch 3 had the strongest preservation shape: repeat true positives increased
by three, GuitarSet returned exactly to baseline, P1 improved, and cross-source
macro accuracy improved. It still failed because aggregate accuracy decreased
and wrong strings increased by 16. The preregistered gate therefore correctly
rejected it.

## Comparison with v3

Reducing P3 from 20% to 1% substantially reduced damage but did not reverse it.
Comparing each experiment's best aggregate-accuracy epoch:

| Metric delta from string-v1 | v3 epoch 2 | v4 epoch 1 |
|---|---:|---:|
| Aggregate accuracy | `-0.002852` | `-0.000494` |
| Wrong strings | `+52` | `+9` |
| Repeat true positives | `-22` | `+2` |
| GuitarSet accuracy | `-0.004168` | `-0.000154` |
| AG-PT accuracy | `+0.000854` | `-0.001708` |
| Guitar-TECHS P1 accuracy | `-0.013636` | `0` |

The scale-control hypothesis was directionally useful: most v3 forgetting was
removed. It was not sufficient to produce a promotable checkpoint.

## Source-gradient diagnostics

The complete-model diagnostics were:

| Epoch | GSet vs P3 cosine | P3/GSet norm ratio |
|---|---:|---:|
| 0 | `0.605` | `49.7x` |
| 1 | `-0.116` | `50.0x` |
| 2 | `0.756` | `38.4x` |
| 3 | `0.595` | `50.4x` |

At epoch 1, token-encoder and sequence-encoder cosines became `-0.859` and
`-0.771`, then returned positive in epochs 2 and 3. Because these epoch probes
do not repeat the cluster bootstrap, they are trajectory diagnostics rather
than evidence of stable source-wide conflict. The sign changes reinforce the
earlier finding that source direction is heterogeneous and state-dependent.

The P3 per-unit gradient remains much larger than GuitarSet even after sampler
mass is reduced. At 1% sampling, however, its approximate aggregate
contribution is bounded to roughly `0.38-0.51x` the GuitarSet source-mean norm
instead of dominating the update as it did at 20%.

Epoch diagnostics reuse candidate groups precomputed over all 408 gradient
tracks so they reflect the actual training objective, including full-dataset
rare-pair weights. The standalone conflict audit constructed only its 72
diagnostic tracks, so its exact epoch-zero norm ratio is not expected to match
this training-context probe.

## Conclusion and next control

V4 rejects the claim that reducing P3 to 1% is sufficient. It also shows that
the 20% v3 mixture was a major contributor to forgetting.

The next informative experiment is a GuitarSet-only replay control with P3 at
0% and every other v4 setting unchanged:

- if GuitarSet-only replay also fails, the remaining problem is continued
  fine-tuning drift or optimization scale, not P3;
- if GuitarSet-only replay passes, even 1% P3 still causes the residual
  regression and a source-aware update constraint becomes justified.

No v4 checkpoint should be rescued from epoch 1 or epoch 3, and external
evaluation should not run.
