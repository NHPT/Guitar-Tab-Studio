# Candidate sequence activity v1 result

Status: rejected on both fixed cross-source wrong-string gates. The governed
validation gate passed, but the checkpoint remains research-only and cannot
replace v24 or authorize a release.

## Protocol

The experiment followed the frozen
`candidate-sequence-activity-v1-preregistration.md` protocol:

- six epochs with seed `20260923`;
- activity threshold `0.50` and string margin `0`;
- no threshold scan or post-gate hyperparameter change;
- model selection only on governed GuitarSet validation;
- AG-PT and Guitar-TECHS read only after the validation gate passed.

The frozen candidate activity-v1 position model and candidate sequence string
v1 checkpoint were not updated. The new model contained 92,929 trainable
parameters and emitted one scalar activity residual per candidate.

## Training result

Epoch 6 was selected by validation candidate activity F1.

| Epoch | Validation loss | Activity F1 | Repeated-candidate recall |
|---:|---:|---:|---:|
| 0, zero-init | 0.417558 | 0.829825 | 0.851106 |
| 1 | 0.417850 | 0.829687 | 0.875522 |
| 2 | 0.410592 | 0.834686 | 0.859140 |
| 3 | 0.416311 | 0.836009 | 0.889608 |
| 4 | 0.418226 | 0.836012 | 0.894825 |
| 5 | 0.410839 | 0.837822 | 0.877817 |
| 6 | 0.412018 | 0.838140 | 0.881886 |

The selected checkpoint improved activity F1 by `0.008315` and repeated
candidate recall by `0.030780` over the zero-initialized baseline. Both
comparisons were strictly positive, so the governed validation gate passed
and the two fixed cross-source evaluations were authorized.

The research checkpoint SHA-256 is
`b90967cf7d7b6242f9a0236e9857c7723c5ab27ce4cd0f3567da89af15f668ee`.

## AG-PT fixed gate

| Metric | Baseline | Candidate | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| Onset F1 | 0.879593 | 0.948340 | +0.068748 | Strict increase | Pass |
| TAB F1 | 0.370717 | 0.389989 | +0.019272 | Non-decrease | Pass |
| Repeated TAB recall | 0.352703 | 0.387838 | +0.035135 | Non-decrease | Pass |
| Same-pitch wrong-string | 1,812 | 2,031 | +219 | No increase | **Fail** |

AG-PT failed because same-pitch wrong-string errors increased by 219.

## Guitar-TECHS fixed gate

| Metric | Baseline | Candidate | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| TAB F1 | 0.282646 | 0.302187 | +0.019541 | Strict increase | Pass |
| Onset F1 | 0.710006 | 0.759443 | +0.049438 | Non-decrease | Pass |
| Repeated TAB recall | 0.000000 | 0.166667 | +0.166667 | Non-decrease | Pass |
| Same-pitch wrong-string | 374 | 560 | +186 | No increase | **Fail** |

Guitar-TECHS failed because same-pitch wrong-string errors increased by 186.

## Evaluation correction

The first AG-PT baseline attempt stopped before writing a report. The
evaluator constructed each sequence group with the final inference batch's
complete activity-logit tensor instead of that group's sliced tensor. A group
with three candidates therefore received 40 logits.

The evaluator now pairs every group with its corresponding activity-logit
slice. A regression test crosses the 32-group inference batch boundary and
places multiple candidates in the final group. The test and Ruff check pass.
All four reports above were generated from scratch after this correction.
Training and checkpoint selection were unaffected.

## Count reconciliation

The preregistration data table reported 171,140 governed candidate event
occurrences. The checkpoint records 171,125 unique model queries:

| Count | Audit occurrences | Unique training queries | Difference |
|---|---:|---:|---:|
| Active candidates | 90,869 | 90,862 | -7 |
| Inactive candidates | 80,271 | 80,263 | -8 |
| Repeated candidates | 46,363 | 46,358 | -5 |
| Total candidates | 171,140 | 171,125 | -15 |

`candidate_position_queries` canonicalizes candidates by `(group, pitch)`, so
same-group duplicate pitch occurrences become one model query. The manifest,
cache, split, seed and checkpoint inputs remained frozen. This distinction
does not change the rejection, but successor preregistrations must report both
event occurrences and unique model-query counts.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Training log | `58ffb588209c58a114a573833d7e7638679b4858f0c1bf22044d9be0dd730c05` |
| AG-PT baseline | `ec5b83495fef9aab6e019f28f2cebfa2bdcd0e6611997e6eb5b2640fb1c37096` |
| AG-PT candidate | `8e5b9652c8cac7d14e57f89d65b13a6973c0e73571ad380f0cfa0d3221743435` |
| Guitar-TECHS baseline | `d8bef975b37501ec394304737f8e233064e3ec62027b01fd5d2e2c3a6492ca2d` |
| Guitar-TECHS candidate | `62c2e041bbae0ddb20cfa58047ceaeaa440d11a683360f0a9124354c19d61846` |

## Decision

The experiment shows that ordered candidate context can materially improve
cross-source onset, tablature and repeated-note metrics. It also increases
the number of accepted same-pitch wrong-string events. Continuing the same
loss for more epochs would optimize candidate activity without addressing the
failed decision boundary, so no further training or threshold adjustment is
authorized for v1.

The next experiment should make acceptance sensitive to string correctness,
for example by training a joint sequence decision with an explicit reject
option and a wrong-string-aware objective. It must be separately
preregistered and must preserve the same independent cross-source gates.
