# Candidate sequence string margin v1 result

Status: rejected on the governed validation repeated-note gate. The fixed
AG-PT and Guitar-TECHS cross-source reports were not created or read. The
calibration artifact is research-only and cannot replace v24 or authorize a
release.

## Protocol

The experiment followed
`candidate-sequence-margin-v1-preregistration.md`:

- frozen candidate activity-v1 position and sequence string-v1 checkpoints;
- Basic Pitch onset `0.35`, frame `0.25`, minimum note length `70 ms`;
- deterministic 20% training-only `(dataset, group_id)` calibration partition;
- calibration seed `20261120`;
- activity probability threshold `0.50`;
- no confidence, activity or joint sequence rejection head;
- no governed threshold scan or post-result threshold adjustment.

The preregistration SHA-256 is
`4517dae36b50a657eb196659c4cd3291cb1067f83bb3321e7f5bbdc8e90b649a`.

## Training-only calibration

The selected partition contained 220 annotations, 21,439 onset groups and
38,277 candidate queries. Its selected-key SHA-256 was
`db91c1a4f1ef744147c8393137759cf2e702ac9d2e901242d674bbeb64697ac1`.
No model parameters were trained.

The fixed selection rule evaluated 17,470 distinct acceptance sets. Six met
all three calibration constraints. It selected margin
`0.0026192963123321533`.

| Metric | Margin 0 | Selected | Change | Calibration requirement | Result |
|---|---:|---:|---:|---|---|
| Joint position F1 | 0.776775736 | 0.776844896 | +0.000069160 | Strict increase | Pass |
| Wrong-string decisions | 2,086 | 2,083 | -3 | Strict decrease | Pass |
| Repeated-position TP | 7,973 | 7,973 | 0 | Non-decrease | Pass |
| Accepted candidates | 19,228 | 19,222 | -6 | Informational | - |

The threshold removed five joint false positives and one joint true positive
on calibration data. The calibration artifact SHA-256 is
`775b671c0f52d6b523ddd26cb8d06bd44f3ac616063860520c9d3d0eea75146f`.

## Governed validation gate

Baseline and candidate were each evaluated once on the same 120-track
GuitarSet validation split. The artifact rejected nine of 17,636
baseline activity-accepted candidate events.

| Metric | Margin 0 | Selected | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| TAB F1 | 0.664778984 | 0.664830986 | +0.000052002 | Strict increase | Pass |
| Same-pitch wrong-string | 1,672 | 1,669 | -3 | Strict decrease | Pass |
| Repeated TAB recall | 0.619705185 | 0.619620468 | -0.000084717 | Non-decrease | **Fail** |

TAB counts changed from 12,678 TP, 4,958 FP and 7,828 FN to 12,676 TP,
4,951 FP and 7,830 FN. Repeated TAB true positives decreased from 7,315 to
7,314.

The margin improved precision enough to raise TAB F1, but it rejected one
correct repeated note. That violates the mandatory preservation constraint,
so the governed gate failed.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Calibration | `775b671c0f52d6b523ddd26cb8d06bd44f3ac616063860520c9d3d0eea75146f` |
| Governed baseline | `7f21f70f70da1f63baa9a448bdad34f113d920277e434ac61452efa8f6e633cc` |
| Governed candidate | `b701977dba9251ad7b81f4a25d5a822d74c128dd8df9573b2b0bf64b02f81c6c` |

No files matching the preregistered margin-v1 AG-PT or Guitar-TECHS output
names were created.

## Decision

The training-calibrated margin mechanism is technically valid and more
selective than confidence-v1, but its governed gain is too small to preserve
all required repeated-note recall. Margin-v1 is rejected and remains a
research artifact.

The threshold must not be lowered, rounded, source-tuned or rescued after this
result. A successor would require a new preregistration and a calibration
objective that protects repeated events before governed validation.
