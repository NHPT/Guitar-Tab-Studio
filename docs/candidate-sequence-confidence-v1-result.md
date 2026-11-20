# Candidate sequence confidence rejection v1 result

Status: rejected on the governed validation gate. The fixed AG-PT and
Guitar-TECHS cross-source reports were not created or read. The checkpoint is
research-only and cannot replace v24 or authorize a release.

## Protocol

The experiment followed
`candidate-sequence-confidence-v1-preregistration.md`:

- frozen candidate activity-v1 and candidate sequence string-v1 inputs;
- one scalar correctness logit without string-logit rewriting;
- active-candidate correctness supervision only;
- acceptance by the minimum of frozen activity and correctness logits;
- identity acceptance probability `0.99`;
- six epochs with seed `20260923`;
- repeated-positive weight `1.5`;
- model selection only on governed GuitarSet validation;
- no threshold scan or post-result loss adjustment.

The model contained 92,929 trainable parameters. Its initialization preserved
all baseline threshold-0 activity decisions and decoded events.

## Training result

Every trained epoch produced exactly the identity baseline's governed
validation decisions. Epoch 1 was selected because it was the first epoch in a
complete tie on joint position F1, wrong-string count and repeated-position
recall.

| Epoch | Validation loss | Correctness F1 | Joint F1 | Wrong string | Repeated-position recall |
|---:|---:|---:|---:|---:|---:|
| 0, identity-init | 0.682519 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 1, selected | 0.553610 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 2 | 0.547953 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 3 | 0.539928 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 4 | 0.588741 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 5 | 0.577543 | 0.939440 | 0.740000 | 1,536 | 0.763147 |
| 6 | 0.563500 | 0.939440 | 0.740000 | 1,536 | 0.763147 |

Validation correctness counts remained 14,675 true positives, 1,892 false
positives and zero false negatives in every epoch. The loss changed, but no
validation candidate crossed the fixed correctness-logit threshold. Joint
true positives, false positives and false negatives therefore remained
12,654, 4,979 and 3,913.

## Governed validation gate

| Metric | Identity-init | Selected | Change | Requirement | Result |
|---|---:|---:|---:|---|---|
| Joint position F1 | 0.740000 | 0.740000 | 0 | Strict increase | **Fail** |
| Wrong-string decisions | 1,536 | 1,536 | 0 | Strict decrease | **Fail** |
| Repeated-position recall | 0.763147 | 0.763147 | 0 | Non-decrease | Pass |

Two of three mandatory comparisons failed. The experiment was rejected before
cross-source evaluation.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Training log | `ececd944c9b01e09a28516c970cbbb816aaea3863b69f9dbe2a8ce6d9c88a0db` |
| Checkpoint | `bd2a9dfd5383cf750847279127999c296bbcfb6d1377079632f222dbb82ae827` |

The training log contains the identity baseline and all six epochs. No files
matching the preregistered confidence-v1 AG-PT or Guitar-TECHS output names
exist.

## Post-result separability audit

A diagnostic-only audit was run after v1 had already been rejected. It used
only the unchanged governed GuitarSet validation split and did not select a
threshold, change the checkpoint or read either cross-source set.

The audit covered all 16,567 unique active `(group, pitch)` queries:

- 14,675 frozen top-string predictions were correct;
- 1,892 frozen top-string predictions were wrong;
- the correct-class prevalence was `0.885797`.

| Score | AUROC | Average precision | Mean correct | Mean wrong | Mean gap |
|---|---:|---:|---:|---:|---:|
| Confidence-v1 logit | 0.539201 | 0.885822 | 3.528461 | 3.434196 | 0.094265 |
| Frozen top1-top2 probability margin | 0.840074 | 0.975310 | 0.853710 | 0.511148 | 0.342561 |

Confidence-v1 average precision was only `0.000025` above the positive-class
prevalence. Its correct and wrong distributions also overlapped almost
completely:

| Confidence logit | Minimum | P05 | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Correct top string | 1.105977 | 1.968014 | 3.761218 | 4.394834 | 4.724144 |
| Wrong top string | 1.304138 | 1.920805 | 3.609930 | 4.496013 | 4.741066 |

Every active validation candidate had a positive confidence logit. The fixed
zero threshold therefore retained 100% of correct predictions and rejected
0% of wrong predictions.

By contrast, the frozen string model's own probability margin preserved
substantial threshold-independent ranking information. Its median was
`0.958562` for correct predictions and `0.511982` for wrong predictions.

The audit report SHA-256 is
`e9891ae2a7d63f91946014a017b4c639d2b4a0f7256536c87cb04b81d1f20a5f`.

## Decision

The independent confidence head reduced training loss and rejected many
training wrong-string candidates, but this selective behavior did not
generalize to governed validation at the fixed threshold. It therefore
provided neither a TAB-quality gain nor a wrong-string reduction.

Changing the threshold, identity bias, class weighting or objective after
seeing this result would be post-result tuning and cannot rescue v1. Any
successor must be separately preregistered.

The threshold-independent audit does not support training confidence-v2 next:
confidence-v1 ranking was close to random, while the frozen top1-top2 margin
already reached AUROC `0.840074`. The next defensible experiment is a
margin-based rejection rule calibrated only on a new training-only calibration
split, with its threshold and coverage constraints frozen before governed
validation. Governed validation must remain a gate rather than a calibration
source.
