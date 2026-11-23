# Candidate sequence string v7 preregistration

Status: frozen before implementation, anchor calibration, and training. This
is one training-only string-v1 parameter-anchor control. It cannot read
governed GuitarSet, AG-PT held-out, or Guitar-TECHS P2 result reports.

## Question

String v4-v6 established that:

- reducing P3 optimizer exposure to 1% does not pass the preservation gate;
- removing P3 entirely does not pass;
- exact nonreplacement coverage of all 3,288 GuitarSet windows does not pass;
- the 384-versus-96 GuitarSet audit does not demonstrate stable partition-wide
  gradient conflict or a large mean-gradient norm imbalance.

String v7 asks one narrower question:

> With 0% P3 and the exact v6 full-coverage replay, can a parameter-space
> anchor prevent movement away from string v1 from degrading any frozen
> training-only preservation metric?

V7 changes one mechanism relative to v6: after every AdamW step, each frozen
model module is projected into a string-v1-centered trust region. The trust
region sizes are determined without calibration metrics, held-out results, or
manual coefficient selection.

## Why a trust-region anchor

A conventional L2-SP penalty has zero gradient exactly at the string-v1
initialization. Its coefficient therefore cannot be identified from the
epoch-zero anchor gradient alone without choosing an arbitrary target
displacement.

V7 instead uses four hard parameter-space trust regions. Their radii are the
norms of a deterministic virtual first AdamW update computed from the complete
epoch-zero GuitarSet training gradient. This gives the anchor an
optimizer-scaled, data-derived size while avoiding validation-based tuning.

This is a parameter anchor, not gradient surgery:

- task gradients and AdamW moments remain unchanged;
- no calibration gradient enters training;
- no P3 gradient enters training;
- only the post-step distance from the frozen string-v1 parameters is capped.

## Frozen evidence

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v6 preregistration | `5be7175bb428ee48d6abd8c2f5f52c4d2b75498014ef984bf00a0f06cfcac11e` |
| v6 input audit | `81f8d0712f86076c22dc446761e706cc63d226eff321550d28da72baf9e889b9` |
| v6 training log | `cccfddda70b3ff960eea0cd09e6a57d3892ae9a69e2bf453a94077aff90ac520` |
| v6 result | `24d8101d976f202c34dca03292f70b11ad84eaff556c39918e35345ab8bd6a6d` |
| GuitarSet gradient audit plan | `345932175548254502be20049ec222149901a136508ef0685b0cf833ab157866` |
| GuitarSet gradient audit report | `ed6c9a95e58f1e3da2f0777070a884d2ebb1640c1b90660cf0740f6e36342e7f` |
| GuitarSet gradient audit result | `4d47faeb1f2957380f5e62cd500a3c0539c9f8d906a223b1746d0043b48ccb2f` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The v6 log contains epoch 0 plus three training epochs. All three epochs are
ineligible, all three cover the same 3,288 eligible windows exactly once, and
no v6 checkpoint exists.

## Epoch-zero gradient reference

The frozen GuitarSet audit measured this operational,
active-candidate-exposure-weighted epoch-zero gradient:

| Module | Task-gradient norm |
|---|---:|
| Token encoder | `0.021608929409185153` |
| Sequence encoder | `0.011811496851312856` |
| Residual hidden | `0.004798116867593484` |
| Residual output | `0.026636370019020944` |
| All parameters | `0.03659201308134063` |

These values are reference evidence, not calibration acceptance targets. The
audit computed target weights in a 384-track source-local GuitarSet dataset.
The actual v6/v7 training dataset also materializes 24 diagnostic-only P3
tracks. P3 receives no optimizer update, but its annotations participate in
the dataset-level rare-pair weight statistics. The objective-faithful v7
gradient may therefore differ from the source-local audit gradient.

V7 calibration uses the actual unchanged v6 training dataset and its frozen
weights, then selects only the 3,288 eligible GuitarSet windows for gradient
accumulation. It must contain 55,946 eligible GuitarSet candidate groups and
109,007 active candidate-window exposures. Each batch mean task loss is
multiplied by its active-candidate count before backward accumulation;
gradients are divided by 109,007 after all windows are processed.

Both frozen models run in evaluation mode during calibration. There is no
optimizer instance, clipping, stochastic sampling, parameter update, or
checkpoint write. All parameter tensors must compare bitwise equal before and
after calibration.

The calibration report records the difference from the source-local reference
for every module. No reference-value match is required, and no observed norm
may be used to alter the frozen radius formula.

## Frozen radius calibration

The parameter partition is the same disjoint and exhaustive four-module
partition used by the gradient audits:

| Module | Parameter prefix |
|---|---|
| Token encoder | `token_encoder.` |
| Sequence encoder | `sequence_encoder.` |
| Residual hidden | `residual_head.0.` |
| Residual output | `residual_head.3.` |

Let `theta0` be the complete string-v1 parameter vector and `g0` the
full-coverage epoch-zero task gradient. Apply the unchanged global norm cap:

`g_clipped = g0 * min(1, 5.0 / L2Norm(g0))`

For every scalar parameter, compute the virtual zero-state first AdamW
displacement:

`u = -lr * g_clipped / (abs(g_clipped) + epsilon) - lr * weight_decay * theta0`

with the frozen optimizer values:

- learning rate `0.00005`;
- Adam betas `(0.9, 0.999)`;
- epsilon `0.00000001`;
- weight decay `0.0001`;
- no AMSGrad and no maximize mode.

At Adam step one, bias correction reduces the moment term to the expression
above. The calibration report must retain both raw and clipped gradient norms
and the applied clip scale, even when the expected scale is one.

For module `m`, the trust radius is:

`R_m = L2Norm(u_m)`

All four radii must be finite and strictly positive. They are written once to:

`reports/candidate-sequence-string-v7-anchor-calibration.json`.

The calibration report must include:

- all frozen input and plan hashes;
- parameter names and counts by module;
- raw and clipped task-gradient norms by module and total;
- global clip scale;
- virtual update norms/radii by module and total;
- weight-decay and adaptive-update components separately;
- the 109,007 exposure denominator;
- zero-parameter-mutation confirmation.

The generated radius values become immutable training inputs. Training must
verify the calibration report hash captured by its input audit and must not
recalculate or alter a radius.

## Frozen training anchor

Training starts again from the complete string-v1 checkpoint. After every
ordinary v6 AdamW step, and before the next batch, compute each module's
displacement:

`d_m = theta_m - theta0_m`

If `L2Norm(d_m) <= R_m`, leave the module unchanged. Otherwise project it onto
the boundary:

`theta_m = theta0_m + d_m * R_m / L2Norm(d_m)`

Rules:

- projection is applied independently to all four modules;
- optimizer moments are not reset or projected;
- `theta0` never changes;
- the radius never changes;
- projection occurs after AdamW and before the next forward pass;
- the existing pre-step gradient norm cap remains `5.0`;
- no anchor loss is added to the reported task loss.

Every epoch must report, by module:

- radius;
- displacement norm before and after projection;
- maximum pre-projection and post-projection displacement;
- projection count and fraction of optimizer steps;
- boundary occupancy at epoch end;
- task-gradient norm before clipping;
- combined train loss and candidate accuracy exactly as v6.

Any post-projection module displacement greater than its radius by more than
`max(1e-12, radius * 1e-6)` fails the run.

## Unchanged replay and optimization

V7 inherits the complete v6 replay contract:

| Parameter | Value |
|---|---:|
| GuitarSet windows per epoch | 3,288, each exactly once |
| P3 optimizer windows | 0 |
| Replacement | no |
| Batch size | 16 |
| Batches per epoch | 206 |
| Epochs | 3 |
| Trainable parameters | all 93,190 |
| Optimizer | AdamW |
| Learning rate | `0.00005` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.0000025` |
| Gradient norm cap | `5.0` |
| Seed | `20261122` |
| Sequence window / stride | `32 / 16` |
| Training device | CPU |

The model architecture, candidate weights, task loss, decoder, calibration
partition, source-gradient diagnostics, sampler permutation sequence, and
epoch learning rates remain unchanged.

## Two frozen decisions

The primary control decision is drift prevention. An epoch has
`drift_prevented=true` only if all conditions hold relative to string v1:

1. aggregate accuracy does not decrease;
2. aggregate wrong strings do not increase;
3. repeated-string true positives do not decrease;
4. GuitarSet accuracy does not decrease;
5. AG-PT accuracy does not decrease;
6. Guitar-TECHS P1 accuracy does not decrease;
7. AG-PT/P1 macro accuracy does not decrease.

Equality is allowed because exact preservation is a successful result for the
question being tested. Evaluation is deterministic, so no numerical tolerance
is applied to count-derived metrics.

The unchanged production decision is also reported. An epoch has
`production_eligible=true` only if it passes the original v3-v6 gate:

1. aggregate accuracy strictly increases;
2. aggregate wrong strings strictly decrease;
3. repeated-string true positives do not decrease;
4. GuitarSet accuracy does not decrease;
5. AG-PT/P1 macro accuracy does not decrease;
6. neither AG-PT nor P1 accuracy decreases by more than `0.005`.

Only a `production_eligible` epoch may write a checkpoint. A
`drift_prevented` but non-production epoch is evidence for the anchor
mechanism, not a deployable model.

Among production-eligible epochs, retain the unchanged selection order:
higher aggregate accuracy, then more repeat true positives, then higher
cross-source macro accuracy, then earlier epoch.

## Causal interpretation

- If at least one epoch prevents drift, the frozen trust-region anchor is
  sufficient to preserve string v1 under 0% P3 full-coverage replay. The next
  experiment may introduce 1% P3 while keeping the same frozen radii.
- If no epoch prevents drift despite all module caps holding, limiting
  parameter displacement to one virtual full-gradient AdamW step is
  insufficient. The next mechanism must be direction-aware or directly
  preservation-objective-aware, not another replay or P3-ratio change.
- If an epoch is production eligible, it may be checkpointed under the
  unchanged gate; external evaluation remains separately blocked until normal
  checkpoint governance is satisfied.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v7-input-audit.json`;
- anchor calibration:
  `reports/candidate-sequence-string-v7-anchor-calibration.json`;
- checkpoint, only if the production gate passes:
  `checkpoints/guitarset-candidate-sequence-string-v7.pt`;
- training log:
  `reports/candidate-sequence-string-v7-training.jsonl`;
- result:
  `docs/candidate-sequence-string-v7-result.md`.
