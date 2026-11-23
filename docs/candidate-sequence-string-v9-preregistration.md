# Candidate sequence string v9 preregistration

Status: frozen before implementation, input audit, sampler realization, and
training. This is one training-only equal-step GuitarSet injection control
for string v8. It cannot read governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result reports.

## Question

String v8 retained the frozen v7 trust regions and injected `33 / 33 / 34`
P3 windows over three epochs. Epoch 1 improved every changed training-only
metric relative to the drift-preventing v7 epoch 3. V8 also increased the run
from 618 to 624 optimizer steps, so that result cannot distinguish P3 source
content from the six additional updates.

String v9 asks:

> If the same six updates contain additional GuitarSet replay instead of P3,
> does the v8 epoch-1 advantage remain?

V9 changes only the source of the `33 / 33 / 34` injected windows from P3 to
GuitarSet. It starts again from the complete string-v1 checkpoint and reuses
the unchanged v7 anchor.

## Frozen evidence

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v8 preregistration | `97014e726ec595ef9f689adba76fe69340afb2639a9e4b2edce338b67d7cb0ba` |
| v8 input audit | `417760f62cb5576bde0393c3015715867ff444fdb97bafd9d83ae528ac96a901` |
| v8 training log | `30850a58f3a8847e390303da69ecc5e1f4356cd01cfc84e1e3ff73f327e49a63` |
| v8 result | `bc54b167d6d2eba7b2ab0ebdfeedaeabb8023928fef1a892818179fb41a74202` |
| selected v8 checkpoint | `0bf00091daaaf6c8428c982d1bf0e30165460eeae821a96a7552bbad83aad726` |
| v7 anchor calibration | `460bf75ad6b581ed6d162c658e4e000074269871f3415cb4e5d30b49e11c527f` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The v8 log contains epoch zero plus three complete epochs. Epochs 1 and 3
have both `drift_prevented=true` and `production_eligible=true`. The selected
v8 checkpoint identifies epoch 1, string-v1 initialization, the v7
calibration hash, and the `33 / 33 / 34` P3 schedule.

## Unchanged anchor and optimization

V9 reuses the v7 calibration report verbatim:

| Module | Radius |
|---|---:|
| Token encoder | `0.0035978438540137402` |
| Sequence encoder | `0.013503442576068494` |
| Residual hidden | `0.005545074699143554` |
| Residual output | `0.0009873639570436095` |

The v8 post-AdamW module projection, float32 numerical refinement, unchanged
optimizer moments, displacement diagnostics, and cap tolerance remain
unchanged.

| Parameter | Value |
|---|---:|
| Initialization | complete string-v1 checkpoint |
| Base GuitarSet windows per epoch | 3,288, each exactly once |
| Injected GuitarSet windows | 33, 33, 34 |
| Total windows | 3,321, 3,321, 3,322 |
| Batch size | 16 |
| Batches per epoch | 208 |
| Total optimizer steps | 624 |
| Epochs | 3 |
| Trainable parameters | all 93,190 |
| Optimizer | AdamW |
| Learning rate | `0.00005` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.0000025` |
| Gradient norm cap | `5.0` |
| Base permutation seed | `20261122` |
| Injection selection seed | `20261123` |
| Sequence window / stride | `32 / 16` |
| Training device | CPU |

The architecture, data partition, candidate weights, task loss, decoder,
calibration partition, source-gradient diagnostics, learning-rate schedule,
anchor center, radii, and checkpoint selection order remain unchanged.

## Equal-step GuitarSet injection

Every epoch first reproduces the exact v8 base GuitarSet permutation:

- all 3,288 eligible GuitarSet windows occur exactly once;
- no base window is omitted or duplicated;
- the base generator seed remains `20261122`;
- the relative base permutation is unchanged from v7 and v8.

V9 then injects additional GuitarSet replay:

- the count schedule remains exactly `33 / 33 / 34`;
- injected windows use the separate seed `20261123`;
- selection is uniform without replacement within each epoch;
- every injected window is already present once in the base full-coverage
  sequence and therefore becomes one deliberate replay duplicate;
- no injected window appears more than once in the injected subset;
- injection positions use the exact v8 spacing formula and therefore occur
  at the same sequence positions as the corresponding P3 injections;
- P3 windows receive zero optimizer exposure.

The total window counts and optimizer-step schedule are therefore identical
to v8:

| Epoch | Base GSet | Injected GSet | Total | Batches |
|---|---:|---:|---:|---:|
| 1 | 3,288 | 33 | 3,321 | 208 |
| 2 | 3,288 | 33 | 3,321 | 208 |
| 3 | 3,288 | 34 | 3,322 | 208 |
| Total | 9,864 | 100 | 9,964 | 624 |

The extra-window share is the same `1.003613%` as v8. The active-candidate
exposure count is allowed to differ because candidate density is part of the
source-content treatment. No source-specific loss multiplier is allowed.

## Frozen coverage checks

Every epoch must record and validate:

- base draw count, uniqueness, zero omission, zero duplicate, track, family,
  player, variant, and window-set identity;
- injected draw count, uniqueness within the injected subset, selected
  window identity, track count, family count, player and variant counts;
- exact combined draw and batch count;
- exactly one combined duplicate per injected window;
- zero P3 optimizer windows;
- base, injected, and combined active candidate-window exposures;
- unchanged module trust-region diagnostics and cap enforcement.

Any schedule deviation, base coverage failure, duplicate inside the injected
subset, P3 optimizer exposure, batch-count mismatch, or anchor-cap violation
fails the run.

## Unchanged training decisions

The v7-v8 decisions remain unchanged relative to string v1.

`drift_prevented=true` requires all seven metrics to be non-regressing:
aggregate accuracy, aggregate wrong strings, repeated-string true positives,
GuitarSet accuracy, AG-PT accuracy, Guitar-TECHS P1 accuracy, and AG-PT/P1
macro accuracy.

`production_eligible=true` requires the unchanged strict production gate:
aggregate accuracy increases, wrong strings decrease, repeat true positives
do not decrease, GuitarSet and cross-source macro accuracy do not decrease,
and neither AG-PT nor P1 decreases by more than `0.005`.

Only a production-eligible epoch may write a checkpoint. The unchanged
selection order is higher aggregate accuracy, then more repeat true
positives, then higher cross-source macro accuracy, then earlier epoch.

## Frozen source-attribution decision

The primary matched comparison is v8 epoch 1 versus v9 epoch 1. They have
the same scheduler stage, base permutation, injection count, insertion
positions, batch count, optimizer-step count, initialization, and anchor.

Use the seven drift metrics with their natural directions:

- higher is better for all accuracy and repeat-TP metrics;
- lower is better for aggregate wrong strings.

Classify the source-content result as:

1. `p3_content_advantage` only if v8 epoch 1 is no worse than v9 epoch 1 on
   all seven metrics and strictly better on at least one;
2. `extra_steps_sufficient` only if v9 epoch 1 is no worse than v8 epoch 1
   on all seven metrics and strictly better on at least one, or if all seven
   are exactly equal;
3. `mixed_inconclusive` otherwise.

Selected-epoch comparisons are secondary because selecting different epochs
would break the matched scheduler-stage comparison.

## Causal interpretation

- `p3_content_advantage` supports attributing the v8 matched-epoch advantage
  to P3 source content rather than merely six extra optimizer steps.
- `extra_steps_sufficient` rejects a P3-specific explanation for the v8
  improvement.
- `mixed_inconclusive` means the deterministic control does not identify one
  source as uniformly better; do not tune injection windows or radii post
  hoc.

Even a decisive result is specific to this frozen deterministic run and its
training-only metrics. External evaluation remains separately governed.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v9-input-audit.json`;
- checkpoint, only if the production gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v9.pt`;
- training log:
  `reports/candidate-sequence-string-v9-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v9-result.md`.
