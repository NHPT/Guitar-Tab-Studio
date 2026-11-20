# Candidate sequence joint decision v1 result

Status: rejected on the governed validation gate. The fixed AG-PT and
Guitar-TECHS cross-source reports remained unread. The checkpoint is
research-only and cannot replace v24 or authorize a release.

## Protocol

The experiment followed
`candidate-sequence-joint-v1-preregistration.md`:

- one explicit reject class and six string classes;
- frozen candidate activity-v1 and candidate sequence string v1 inputs;
- six epochs with seed `20260923`;
- wrong-string weight `2.0` and repeated-positive weight `1.5`;
- model selection only on governed GuitarSet validation;
- no threshold scan or post-result loss adjustment.

The model contained 93,319 trainable parameters. Initialization preserved the
frozen activity probabilities, conditional string probabilities and decoded
events.

## Training result

Epoch 2 had the highest validation joint position F1 and was selected.

| Epoch | Loss | Joint F1 | Wrong string | Activity F1 | Repeated-position recall |
|---:|---:|---:|---:|---:|---:|
| 0, zero-init | 0.745970 | 0.740000 | 1,536 | 0.829825 | 0.763147 |
| 1 | 0.755468 | 0.732209 | 1,636 | 0.826022 | 0.766486 |
| 2 | 0.737684 | 0.739813 | 1,599 | 0.831070 | 0.781928 |
| 3 | 0.735975 | 0.739182 | 1,609 | 0.830551 | 0.786206 |
| 4 | 0.741740 | 0.737566 | 1,623 | 0.830139 | 0.782346 |
| 5 | 0.737127 | 0.739138 | 1,608 | 0.830704 | 0.784015 |
| 6 | 0.738784 | 0.738544 | 1,615 | 0.830648 | 0.782450 |

The checkpoint SHA-256 is
`eeb546a895fc28419799c1c897899f39a646aecac9c096aba2881ce78c6ab307`.

## Governed validation gate

| Metric | Zero-init | Selected | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| Joint position F1 | 0.740000 | 0.739813 | -0.000187 | Strict increase | **Fail** |
| Wrong-string decisions | 1,536 | 1,599 | +63 | Strict decrease | **Fail** |
| Activity F1 | 0.829825 | 0.831070 | +0.001245 | Non-decrease | Pass |
| Repeated-position recall | 0.763147 | 0.781928 | +0.018781 | Non-decrease | Pass |

Two of four mandatory comparisons failed. The experiment was therefore
rejected before any cross-source evaluation.

## Numerical correction before training

The first process was stopped during feature precomputation, before epoch 0
or checkpoint creation. Static review found that non-target classes in the
set-valued loss used a finite `-30` mask. Extremely unlikely target classes
could therefore encounter an artificial loss floor.

The mask was changed to negative infinity, and a regression test now verifies
that target losses above 100 remain representable. The completed training run
started from scratch after this correction. No observed metric or
hyperparameter was used to make the change.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Training log | `47c9fdca0f6d7df2f09e61a7e2ddb35ae3af70c9d6900828cd8af623a3d9dae4` |
| Checkpoint | `eeb546a895fc28419799c1c897899f39a646aecac9c096aba2881ce78c6ab307` |

No files matching the preregistered AG-PT or Guitar-TECHS output names were
created.

## Decision

The shared seven-class residual did not produce the intended selective
behavior. It improved activity F1 and repeated-position recall, but accepted
63 more wrong-string candidates and slightly reduced joint position F1.
More epochs or a larger wrong-string weight would be post-result tuning of the
same failed objective and are not authorized.

A successor should separate two decisions:

1. retain the frozen string sequence model's top-string prediction;
2. train a selective correctness head to estimate whether accepting that
   top-string prediction is safe.

Its acceptance score should combine candidate activity with predicted string
correctness. This directly supervises abstention on likely wrong-string
events instead of asking one residual to simultaneously reject candidates and
rewrite the string distribution.
