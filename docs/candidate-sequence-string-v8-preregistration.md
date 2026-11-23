# Candidate sequence string v8 preregistration

Status: frozen before implementation, input audit, sampler realization, and
training. This is one training-only 1% P3 injection control built directly
on string v7. It cannot read governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result reports.

## Question

String v7 established that four frozen string-v1-centered module trust
regions can prevent preservation drift under 0% P3 exact GuitarSet replay.
Epoch 3 prevented drift, passed the unchanged production gate, and produced
the selected v7 checkpoint.

String v8 asks one narrower question:

> With the exact v7 anchor and GuitarSet replay retained, does adding 1% P3
> optimizer exposure preserve the v7 drift and production results?

V8 changes only P3 optimizer exposure from zero to a nominal 1% of replayed
windows. It starts again from the complete string-v1 checkpoint. It does not
fine-tune the selected v7 checkpoint.

## Frozen evidence

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v7 preregistration | `49aa18e6ab3a57ccaae13c7bda555eac770fe704f4b3be1c2d55cc6e08012f0b` |
| v7 anchor calibration | `460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f` |
| v7 input audit | `2974b36060b22e00ba585828d5981eb953a1a9b344054bd34baeace2acc3aeb7` |
| v7 training log | `c53101d8f17d80fbcb7e8ec7f889900549ec16df52f74ba797061412d7b302e1` |
| v7 result | `6c2e41e058e05ad7a194f64e799bcac41d8ebd06273984b27ee403ffdf0bae2e` |
| selected v7 checkpoint | `3ff0012a25fcf7e5ffa7f72e62be8a5d922e7cfff41a55299586589211afd8a3` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The v7 log contains epoch zero plus three complete training epochs. Epoch 3
has `drift_prevented=true` and `production_eligible=true`; epochs 2 and 3 are
production eligible. The v7 checkpoint identifies epoch 3 and the frozen
anchor calibration hash.

## Frozen anchor

V8 reuses the v7 anchor calibration report verbatim. It does not run a new
calibration and does not recalculate, rescale, or tune any radius:

| Module | Radius |
|---|---:|
| Token encoder | `0.0035978438540137402` |
| Sequence encoder | `0.013503442576068494` |
| Residual hidden | `0.005545074699143554` |
| Residual output | `0.0009873639570436095` |

After every ordinary AdamW step, each module is projected toward its
string-v1 anchor exactly as in v7. Optimizer moments remain unchanged. The
same direction-preserving float32 numerical refinement may move a quantized
projection minimally inward until:

`post_displacement <= radius + max(1e-12, radius * 1e-6)`.

Every epoch reports the unchanged v7 module displacement, projection,
boundary occupancy, and pre-clip task-gradient diagnostics.

## Unchanged GuitarSet replay

Every epoch retains all v7 GuitarSet replay properties:

- all 3,288 eligible GuitarSet windows occur exactly once;
- no GuitarSet window is duplicated or omitted;
- the GuitarSet permutation generator remains seed `20261122`;
- the relative GuitarSet permutation sequence is unchanged by P3 selection;
- P3 selection uses a separate random generator and cannot advance the
  GuitarSet generator;
- all 384 GuitarSet tracks and 234 recording families remain covered.

The eligible GuitarSet window-set identity remains:

`560d99d445b0daa163dc764611ba4f0ea6086bac4039815b26c338c208d60b6a`.

## Single changed variable: 1% P3 injection

There are 201 materialized P3 windows. P3 is added without removing or
replacing any GuitarSet window.

For target share `p = 0.01`, the expected P3 count per epoch relative to
3,288 fixed GuitarSet windows is:

`3288 * p / (1 - p) = 33.21212121212121`.

Because window counts are integral, v8 uses deterministic cumulative
nearest-integer error balancing. At epoch `e`, cumulative P3 draws equal:

`floor(e * 33.21212121212121 + 0.5)`.

The resulting frozen three-epoch schedule is:

| Epoch | GuitarSet | P3 | Total | Actual P3 share | Batches |
|---|---:|---:|---:|---:|---:|
| 1 | 3,288 | 33 | 3,321 | `0.9936766%` | 208 |
| 2 | 3,288 | 33 | 3,321 | `0.9936766%` | 208 |
| 3 | 3,288 | 34 | 3,322 | `1.0234798%` | 208 |
| Total | 9,864 | 100 | 9,964 | `1.0036130%` | 624 |

This is the closest cumulative integer realization to the exact 1% target
without omitting GuitarSet windows. The additional six optimizer steps over
v7 are a mechanical consequence of injecting P3 windows, not an independently
tuned policy.

Within each epoch:

- P3 windows are selected uniformly without replacement;
- the P3 generator seed is `20261123`;
- no P3 window repeats within an epoch;
- selected P3 windows are spread deterministically through the GuitarSet
  sequence while preserving the relative GuitarSet permutation order;
- P3 uses the unchanged candidate weights and task loss;
- there is no source-specific loss multiplier, gradient surgery,
  source-specific head, freezing, or curriculum.

## Frozen optimization

| Parameter | Value |
|---|---:|
| Initialization | complete string-v1 checkpoint |
| GuitarSet windows per epoch | 3,288, each exactly once |
| P3 windows by epoch | 33, 33, 34 |
| Batch size | 16 |
| Batches per epoch | 208 |
| Epochs | 3 |
| Trainable parameters | all 93,190 |
| Optimizer | AdamW |
| Learning rate | `0.00005` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.0000025` |
| Gradient norm cap | `5.0` |
| GuitarSet seed | `20261122` |
| P3 seed | `20261123` |
| Sequence window / stride | `32 / 16` |
| Training device | CPU |

The model architecture, data partition, candidate weights, task loss,
decoder, calibration partition, source-gradient diagnostics, learning-rate
schedule by epoch, anchor center, anchor radii, and checkpoint selection
order remain unchanged from v7.

## Frozen coverage checks

Every epoch must record and validate:

- total draws and batches;
- exact GuitarSet draw, uniqueness, duplicate, omission, track, family,
  player, variant, and identity counts;
- exact scheduled P3 draw count;
- P3 unique-window count and zero within-epoch duplicates;
- P3 selected window and track identities;
- actual per-epoch and cumulative source shares;
- 109,007 GuitarSet active candidate-window exposures plus the separately
  reported P3 active candidate-window exposures.

Any GuitarSet omission or duplicate, P3 within-epoch duplicate, schedule
deviation, source mismatch, or anchor-cap violation fails the run.

## Two frozen decisions

The v7 decisions remain unchanged relative to string v1.

An epoch has `drift_prevented=true` only if all seven conditions hold:

1. aggregate accuracy does not decrease;
2. aggregate wrong strings do not increase;
3. repeated-string true positives do not decrease;
4. GuitarSet accuracy does not decrease;
5. AG-PT accuracy does not decrease;
6. Guitar-TECHS P1 accuracy does not decrease;
7. AG-PT/P1 macro accuracy does not decrease.

An epoch has `production_eligible=true` only if it passes the original
production gate:

1. aggregate accuracy strictly increases;
2. aggregate wrong strings strictly decrease;
3. repeated-string true positives do not decrease;
4. GuitarSet accuracy does not decrease;
5. AG-PT/P1 macro accuracy does not decrease;
6. neither AG-PT nor P1 accuracy decreases by more than `0.005`.

Only a production-eligible epoch may write a checkpoint. Among eligible
epochs, retain higher aggregate accuracy, then more repeat true positives,
then higher cross-source macro accuracy, then earlier epoch.

## Causal interpretation

- If at least one epoch prevents drift and passes the production gate, 1% P3
  replay is compatible with the frozen v7 anchor.
- If v8 fails while all anchor caps and GuitarSet coverage checks hold, the
  added P3 optimizer exposure is the causal difference from v7. Do not tune
  the radii or P3 ratio post hoc.
- A passing checkpoint remains blocked from external evaluation until the
  normal governance boundary is explicitly opened.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v8-input-audit.json`;
- checkpoint, only if the production gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v8.pt`;
- training log:
  `reports/candidate-sequence-string-v8-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v8-result.md`.
