# Candidate sequence string v6 preregistration

Status: frozen before implementation and training. This is one training-only
GuitarSet replay control. It cannot read governed GuitarSet, AG-PT held-out,
or Guitar-TECHS P2 result reports unless the unchanged training-only gate
first selects a checkpoint.

## Hypothesis

String v5 showed that continued fine-tuning regresses without any P3 optimizer
exposure. The subsequent 384-versus-96 GuitarSet gradient audit did not find
stable partition-wide conflict or a large mean-gradient norm imbalance.

The remaining bounded hypothesis is that v5's finite replacement replay is
not a neutral approximation of full GuitarSet coverage. String v6 changes the
replay policy from replacement sampling to deterministic nonreplacement
coverage:

- every one of the 3,288 eligible GuitarSet windows appears exactly once per
  epoch;
- no GuitarSet window is omitted or repeated within an epoch;
- all 201 P3 windows remain diagnostic-only and receive no optimizer update.

The causal interpretation is frozen:

- if v6 passes, full-coverage replay avoids the v5 failure, but the result
  jointly reflects removal of replacement multiplicity and the mechanical
  reduction from 3,489 to 3,288 draws;
- if v6 fails, finite replacement coverage is not required for regression and
  continued fine-tuning should stop until an explicit string-v1 anchoring or
  trust-region mechanism is preregistered.

## Frozen evidence and inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| v5 preregistration | `e6d19577e722b4eaae0a52dbd260a64e29231439628fba0894872eb22abbd6a8` |
| v5 input audit | `e086978cef0cabd4783f34ac3d21d94b364f52b0a237ae203cc456800d15ed77` |
| v5 training log | `f01406d0fc1c97eb88ca416d8832a90991640ddb402f0f22571b15049ea22ffb` |
| v5 result | `8737657e7e6de870934ded990ac81a345033f34f7fe652400e8ffe99bb2126cc` |
| GuitarSet gradient audit plan | `345932175548254502be20049ec222149901a136508ef0685b0cf833ab157866` |
| GuitarSet gradient audit report | `ed6c9a95e58f1e3da2f0777070a884d2ebb1640c1b90660cf0740f6e36342e7f` |
| GuitarSet gradient audit result | `4d47faeb1f2957380f5e62cd500a3c0539c9f8d906a223b1746d0043b48ccb2f` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The v5 log contains epoch 0 plus three training epochs. All three epochs are
ineligible, every one of its 10,467 optimizer-window draws is GuitarSet, and
no v5 checkpoint exists.

The GuitarSet gradient audit passed its frozen inputs and zero-mutation checks.
All ten operational/common-unweighted module classifications are
`inconclusive`. The complete-model operational and common-unweighted cosines
are `0.48010991755300075` and `0.6083953737186203`; their
calibration/gradient norm ratios are `1.2561466720144017` and
`1.2555209333563448`.

## Unchanged data and objective

The v3-v5 partition is reproduced with fraction `20%` and seed `20261122`.
P3 remains excluded from partition construction. The 268 calibration tracks
and selected-key SHA-256 remain:

`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.

The candidate sequence dataset remains materialized with:

| Dataset | Tracks | Candidate groups | Windows |
|---|---:|---:|---:|
| GuitarSet, excluding calibration | 384 | 55,946 | 3,288 |
| Guitar-TECHS P3, diagnostic only | 24 | 3,416 | 201 |

AG-PT-set, Guitar-TECHS P1, and synthetic records receive no gradient. The
complete 93,190-parameter sequence-string model initializes from string v1.
The original per-candidate set-valued cross-entropy, candidate weights,
architecture, position/activity checkpoint, decoder, and calibration
partition remain unchanged.

## Replay policy change

| Sampling parameter | v5 | v6 |
|---|---:|---:|
| GuitarSet eligible windows | 3,288 | 3,288 |
| P3 eligible windows | 0 | 0 |
| Draws per epoch | 3,489 | 3,288 |
| Replacement | yes | no |
| Unique GuitarSet windows required | not fixed | 3,288 |
| Omitted GuitarSet windows allowed | yes | no |
| Repeated GuitarSet windows allowed | yes | no |
| Batches per epoch at batch size 16 | 219 | 206 |

V6 uses a seeded random permutation of all eligible GuitarSet window indexes
for each epoch. The sampler generator is seeded once with `20261122` and
advances across epochs. There is no source weighting after eligibility is
resolved.

The reduced draw and batch counts are a mechanical consequence of exact
one-pass coverage. Therefore a passing v6 result supports the full replay
policy as a package; it does not separately identify replacement multiplicity
or the 13 fewer batches as the sole cause.

Every epoch report must contain:

- total and unique window draws;
- duplicate and omitted eligible-window counts;
- eligible and observed window identity SHA-256 values;
- source, player, original/derived variant, recording-family, and track
  coverage;
- an assertion that all 3,288 draws are GuitarSet and all eligible windows
  appear exactly once.

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

The scheduler remains epoch-based, so learning rates at the start of epochs
1-3 remain identical to v5.

## Source-gradient diagnostics

At epoch 0 and after epochs 1-3, run the unchanged v5 evaluation-mode,
zero-update probe over 48 GuitarSet tracks and 12 paired P3 performances.
P3 diagnostic gradients must not enter the optimizer or scheduler. Every probe
must assert that model parameters remain unchanged.

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
  `reports/candidate-sequence-string-v6-input-audit.json`;
- checkpoint, only if the training-only gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v6.pt`;
- training log:
  `reports/candidate-sequence-string-v6-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v6-result.md`.
