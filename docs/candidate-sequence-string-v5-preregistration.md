# Candidate sequence string v5 preregistration

Status: frozen before implementation and training. This is one training-only
GuitarSet replay control. It cannot read governed GuitarSet, AG-PT held-out,
or Guitar-TECHS P2 result reports unless the unchanged training-only gate
first selects a checkpoint.

## Hypothesis

String v4 reduced Guitar-TECHS P3 sampler mass from 20% to 1%. It reduced the
best wrong-string regression from `+52` to `+9`, but all three epochs still
failed the training-only gate.

String v5 tests whether the remaining regression is caused by even 1% P3 or by
continued fine-tuning itself. V5 changes one variable relative to v4: P3
training sampler mass becomes 0%. The P3 tracks remain available only for
zero-update source-gradient diagnostics.

The causal interpretation is frozen:

- if GuitarSet-only replay fails, continued fine-tuning or optimization drift
  is sufficient to cause regression;
- if GuitarSet-only replay passes, the residual v4 regression is attributable
  to adding P3 and a source-aware update constraint becomes justified.

## Frozen evidence and inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v3 input audit | `7dbb822c25e408acd2cfc72cbd3ee712b1b5f5bfbd0ff9e91b66bbab668a7bd4` |
| gradient conflict report | `f7ca64af7037c5e4f319f0a52ce609cb88720fd1fdaadc24e83ba3e53ec30c33` |
| v4 preregistration | `f92176cbe06069839f0099e485d8c79a6460643dc432a08d23554ec490f9f3dd` |
| v4 input audit | `87b9e4d3d7bf01c971479e66712672ab6a55d6ce5ec07ca807d9ae1dbd3761da` |
| v4 training log | `ad09d70c580a5490f0e6e45ef58cfae2913912d1236c50a71e0951571b432f0e` |
| v4 result | `d9c80902616bbca684ee44355192da5f3f9208bd7a60edfc5b9b9b39f64deb26` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The v4 log contains epoch 0 plus three training epochs. All three epochs are
ineligible, the actual P3 draws total 102 of 10,467, and no v4 checkpoint
exists.

## Unchanged data and objective

The v3/v4 partition is reproduced with fraction `20%` and seed `20261122`.
P3 remains excluded from partition construction. The 268 calibration tracks
and selected-key SHA-256 remain:

`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.

The candidate sequence dataset remains materialized with:

| Dataset | Tracks | Candidate groups | Windows |
|---|---:|---:|---:|
| GuitarSet, excluding calibration | 384 | 55,946 | 3,288 |
| Guitar-TECHS P3, diagnostic only | 24 | 3,416 | 201 |

AG-PT-set, Guitar-TECHS P1, and synthetic records receive no gradient. P3
windows receive exactly zero sampling probability and therefore no training
gradient.

The complete 93,190-parameter sequence-string model initializes from string
v1. The original per-candidate set-valued cross-entropy, candidate weights,
architecture, position/activity checkpoint, decoder, and calibration
partition remain unchanged.

## Single changed variable

| Sampling parameter | v4 | v5 |
|---|---:|---:|
| GuitarSet weight | `1.0` | `1.0` |
| P3 weight | `0.1652344338911503` | `0.0` |
| Expected P3 share | `1%` | `0%` |
| Draws per epoch | `3,489` | `3,489` |
| Replacement | yes | yes |

All 3,489 epoch draws must be GuitarSet windows. Keeping the total draws fixed
isolates source removal without changing optimizer-step count. There is no
loss multiplier, gradient surgery, source-specific head, freezing,
curriculum, or architecture change.

## Fixed optimization

| Parameter | Value |
|---|---:|
| Initialization | complete string-v1 checkpoint |
| Trainable scope | all 93,190 sequence-string parameters |
| Epochs | 3 |
| Window batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `0.00005` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.0000025` |
| Gradient norm cap | `5.0` |
| Seed | `20261122` |
| Sequence window / stride | `32 / 16` |
| Fine-tuning device | CPU |

## Source-gradient diagnostics

At epoch 0 and after epochs 1-3, run the unchanged v4 evaluation-mode,
zero-update probe over 48 GuitarSet tracks and 12 paired P3 performances.
Report module source-mean norms, norm ratios, and cosines. P3 diagnostic
gradients must not enter the optimizer or scheduler. Every probe must assert
that model parameters remain unchanged.

## Training-only selection gate

Epoch zero must exactly reproduce the frozen string-v1 baseline. An epoch is
eligible only if all unchanged conditions hold:

1. aggregate assigned-string accuracy strictly increases;
2. aggregate wrong strings strictly decrease;
3. repeated-string true positives do not decrease;
4. GuitarSet assigned-string accuracy does not decrease;
5. AG-PT/P1 unweighted macro accuracy does not decrease;
6. neither AG-PT nor P1 assigned-string accuracy decreases by more than
   `0.005` absolute.

Among eligible epochs, select higher aggregate accuracy, then more repeated
true positives, then higher AG-PT/P1 macro accuracy, then the earlier epoch.
If no epoch is eligible, write no checkpoint and stop without reading any
external result report.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v5-input-audit.json`;
- checkpoint, only if the training-only gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v5.pt`;
- training log:
  `reports/candidate-sequence-string-v5-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v5-result.md`.
