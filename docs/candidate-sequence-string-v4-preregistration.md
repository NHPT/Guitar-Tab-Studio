# Candidate sequence string v4 preregistration

Status: frozen before implementation and training. This is one training-only
scale-control experiment. It cannot read governed GuitarSet, AG-PT held-out,
or Guitar-TECHS P2 result reports unless the unchanged training-only gate
first selects a checkpoint.

## Hypothesis

String v3 used GuitarSet replay plus Guitar-TECHS P3 at 20% expected sampler
mass. It failed every training-only preservation gate. The frozen gradient
conflict audit did not find a stable negative GuitarSet-versus-P3 gradient
interval, but found P3/GuitarSet mean-gradient norm ratios from `37.5x` to
`114.6x`, including `59.6x` over all parameters. P3 direct-input and mic+amp
gradients were strongly aligned.

String v4 tests one isolated explanation: v3 exposed the model to excessive
P3 gradient mass. V4 changes only the P3 replacement-sampler weight so P3 has
1% expected probability mass. Architecture, initialization, data partition,
objective, optimizer, scheduler, epoch count, decoder, and training-only
selection gates remain unchanged.

## Frozen evidence and inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v3 input audit | `7dbb822c25e408acd2cfc72cbd3ee712b1b5f5bfbd0ff9e91b66bbab668a7bd4` |
| gradient conflict audit plan | `3987e2f213745b9c47a4dab7b05d05bccff8e141353812b458c8a5c23dcf909e` |
| gradient conflict report | `f7ca64af7037c5e4f319f0a52ce609cb88720fd1fdaadc24e83ba3e53ec30c33` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The gradient conflict report passed its zero-parameter-mutation check, used
93,190 sequence parameters, and classified every GuitarSet-versus-P3 module
comparison as inconclusive. Every P3 direct-versus-mic module comparison was
aligned.

The candidate frontend remains Basic Pitch onset `0.35`, frame `0.25`, and
minimum note length `70 ms`. The frozen P3 candidate-cache set contains 24
audio files, 6,685 candidates, 157 dropped rows, and has SHA-256
`c592ed2656c29fc4e0c4546a683b7c500c5b8f229fb431ea20eae70f2931b91a`.

## Unchanged partition and objective

The v3 partition is reproduced with fraction `20%` and seed `20261122`.
P3 remains excluded from partition construction. The 268 calibration tracks
and selected-key SHA-256 remain:

`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.

Gradient data remains:

| Dataset | Tracks | Candidate groups | Windows |
|---|---:|---:|---:|
| GuitarSet, excluding calibration | 384 | 55,946 | 3,288 |
| Guitar-TECHS P3 | 24 | 3,416 | 201 |

AG-PT-set, Guitar-TECHS P1, and synthetic records receive no gradient. The
complete 93,190-parameter sequence-string model initializes from string v1.
The position/activity model remains frozen.

Training reuses the original per-candidate set-valued cross-entropy with
ambiguity weight `1.25`, rare `(pitch, string)` weight capped at `1.5`,
neighboring cross-string weight `2.0` within two seconds, and final
supervision weight capped at `4.0`.

## Single changed variable

GuitarSet sampling weight remains `1.0`. P3 sampling weight becomes:

`(0.01 / 0.99) * (3288 / 201) = 0.1652344338911503`.

This gives:

`201 * 0.1652344338911503 / (3288 + 201 * 0.1652344338911503) = 0.01`.

The sampler still draws 3,489 windows per epoch with replacement. Expected
draws are 3,454.11 GuitarSet windows and 34.89 P3 windows per epoch. No
per-source loss multiplier, gradient surgery, source-specific head, freezing,
curriculum, nonreplacement sampler, or architecture change is allowed.

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

There is one run and no post-result change to sampling, optimization, or
selection.

## Source-gradient diagnostics

At epoch 0 and after epochs 1-3, run an evaluation-mode, zero-update gradient
probe over the frozen conflict-audit units:

- 48 non-calibration GuitarSet tracks, 12 per player;
- 12 P3 performance units, each the equal mean of direct and mic gradients;
- candidate-count-normalized gradients over every sequence window;
- token encoder, sequence encoder, residual hidden, residual output, and all
  parameters.

Each probe reports the GuitarSet mean-gradient norm, P3 mean-gradient norm,
P3/GuitarSet norm ratio, and source-mean cosine. It also asserts that no model
parameter changed. Probes never affect the optimizer, scheduler, sampler, or
checkpoint selection. Bootstrap intervals are not repeated during training;
the frozen baseline conflict audit remains the inferential analysis.

## Training-only selection gate

Epoch zero must reproduce the frozen string-v1 calibration baseline:

| Metric | Baseline |
|---|---:|
| Aggregate assigned-string accuracy | `0.8150606043986179` |
| Aggregate wrong strings | `3372` |
| Repeated-string true positives | `7552` |
| GuitarSet assigned-string accuracy | `0.963257429563875` |
| AG-PT assigned-string accuracy | `0.4627375613922699` |
| Guitar-TECHS P1 assigned-string accuracy | `0.4409090909090909` |
| AG-PT/P1 macro accuracy | `0.45182332615068044` |

An epoch is eligible only if all unchanged v3 conditions hold:

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

## Evaluation boundary

If and only if training selects a checkpoint, the later evaluation order and
gates remain those frozen in the v3 preregistration: governed GuitarSet,
AG-PT held-out, then Guitar-TECHS P2. This v4 training run does not
automatically execute those evaluations.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v4-input-audit.json`;
- checkpoint, only if the training-only gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v4.pt`;
- training log:
  `reports/candidate-sequence-string-v4-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v4-result.md`.
