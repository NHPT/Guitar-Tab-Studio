# Candidate sequence string v2 result

Status: rejected at the training-only selection gate. No checkpoint was
selected, and governed or cross-source evaluation was not started.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Assignment audit JSON | `1fad56d65486b7bd8e6c42260828e4a3c259fd3202eae4cdee0cde4cb316e815` |
| Assignment audit document | `f60eb7b32ad668779ddc53ce1959485839bb6d541e2db9bad627594a1638b181` |
| Preregistration | `a628256c0d9e99e302b2deeaff06911a5fd8712d2c97fc730fa42bc2e8f5c5cc` |
| Training log | `960353f663aced827d50e19cbdec675e70d1c5bb5c358d74e96e6b19f66b02d5` |

The audit and training selection used only the fixed formal-v2 training
partition: 268 annotations, a deterministic 20% holdout, seed `20261122`,
and selected-key SHA-256
`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.
Governed GuitarSet player-04, AG-PT validation, and Guitar-TECHS validation
were not read for model selection.

## Assignment audit

The training-only audit confirmed that string-v1 did not have a purely
polyphonic assignment problem:

- string-v1 improved the aggregate result over v4 by 27 correct strings and
  33 repeated correct strings;
- GuitarSet wrong-string decisions decreased by 49, while AG-PT-set,
  Guitar-TECHS, and synthetic wrong-string decisions increased by 14, 6,
  and 2 respectively;
- 2,767 of 3,372 string-v1 wrong-string decisions occurred in singleton
  activity-accepted groups;
- unique assignment changed 642 independent top-1 decisions, correcting 107
  but harming 317.

These findings justified testing a source-balanced objective that supervised
singletons while scoring complete feasible group assignments.

## Implemented experiment

String-v2 initialized all 93,190 trainable sequence-string parameters from
string-v1. Candidate generation and activity acceptance remained frozen.

For groups with at most six accepted candidates, the objective used the
negative log probability of all reference-valid unique-string assignments
against all feasible unique-string assignments. Activity-accepted inactive
candidates remained unlabeled competitors. Groups with more than six
candidates or no feasible unique assignment retained the deployed independent
fallback.

The single preregistered run used four epochs, AdamW, learning rate `0.0001`,
weight decay `0.0001`, CPU fine-tuning, and source sampling weights
GuitarSet `1.0`, AG-PT-set `2.0`, Guitar-TECHS `8.0`, and synthetic `0.5`.
Multi-candidate groups received weight `2.0`; repeated-reference groups
received weight `1.5`.

## Training-only result

The selection gate required every eligible epoch to have strictly higher
assignment candidate accuracy, strictly fewer wrong-string decisions, and
non-decreasing repeated-string true positives relative to epoch 0.

| Epoch | Loss | Accuracy | Correct / Total | Wrong string | Exact-group rate | Repeat TP / Total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.108977 | 0.815061 | 14,861 / 18,233 | 3,372 | 0.755954 | 7,552 / 8,065 |
| 1 | 0.878400 | 0.799210 | 14,572 / 18,233 | 3,661 | 0.736902 | 7,389 / 8,065 |
| 2 | 0.806840 | 0.802501 | 14,632 / 18,233 | 3,601 | 0.740455 | 7,422 / 8,065 |
| 3 | 0.785317 | 0.801240 | 14,609 / 18,233 | 3,624 | 0.738792 | 7,411 / 8,065 |
| 4 | 0.779968 | 0.802172 | 14,626 / 18,233 | 3,607 | 0.739170 | 7,439 / 8,065 |

All four trained epochs failed all three mandatory comparisons. Epoch 4
reduced structured loss by 29.7% relative to the frozen baseline, but it
also produced 235 fewer correct strings, 235 more wrong-string decisions,
and 113 fewer repeated-string true positives. Exact-group rate decreased by
0.016784.

The falling objective with consistently worse deployed decisions shows that
this structured likelihood is not a usable selection surrogate for the
current candidate distribution. No post-result weighting, threshold,
learning-rate, or epoch adjustment was attempted.

## Stopped evaluation

The training command returned `calibration_gate_passed=false`,
`best_epoch=0`, `selected=null`, and `output=null`. In accordance with the
preregistration:

- `guitarset-candidate-sequence-string-v2.pt` was not created;
- the governed player-04 candidate report was not created;
- the AG-PT candidate report was not created;
- the Guitar-TECHS candidate report was not created.

The governed and cross-source gates are therefore skipped rather than marked
as passed or failed.

## Verification

The final implementation passed:

- all 169 tests in `services/worker/test_stringtrace_ml.py`;
- Ruff checks for the modified Worker implementation and test files;
- Python bytecode compilation for the modified Worker implementation and
  test files;
- `git diff --check`.

## Decision

Reject candidate-sequence string-v2. It does not replace string-v1, does not
change the production model pointer, and does not alter the current v24
research-bundle status.

The evidence no longer supports another confidence threshold, margin,
loss-weight, or nearby structured-objective retry on the same labels and
features. A materially different next experiment needs either:

1. genuinely independent, licensed, real-audio string/fret supervision aligned
   to the candidate frontend; or
2. a source-invariant acoustic representation designed to reduce the observed
   GuitarSet versus AG-PT/Guitar-TECHS mapping split.
