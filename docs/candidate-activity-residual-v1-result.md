# Candidate activity residual v1 result

Status: rejected on the source-calibration gates. The frozen
mixture-Demucs gate was not read.

## Training result

The activity residual trained for five epochs on
`governed-v1/guitarset-mixture-combined.jsonl`. Only `149,761` activity-head
parameters were trainable. Epoch 5 was selected with validation candidate
acceptance F1 `0.829825`.

The frozen v4 validation oracle string accuracy remained exactly `0.867327`
across all epochs, confirming that activity training did not alter conditional
string selection.

The checkpoint SHA-256 is
`7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e`.
Its metadata records the v4 direct initialization hash and the inherited v12
encoder root.

## Source-calibration result

Nine preregistered activity thresholds from `0.10` through `0.90` were
evaluated with the hierarchical decoder. Threshold `0.50` produced the best
TAB F1:

| Gate | Result | Required | Status |
|---|---:|---:|---|
| TAB F1 | `0.652654` | `>= 0.704668` | Fail |
| Same-pitch wrong-string errors | `993` | `<= 481` | Fail |
| Onset F1 | `0.814430` | `>= 0.815375` | Fail |
| Repeated TAB recall | `0.625042` | `>= 0.721959` | Fail |

The selected point emitted `9,206` events. Threshold `0.40` passed only the
onset gate at `0.816604`; TAB F1 was `0.646929`, repeated TAB recall was
`0.650627`, and same-pitch wrong-string errors were `1,100`. Threshold `0.90`
reduced wrong-string errors to `327`, but TAB F1 fell to `0.504903`.

Reports use the `candidate-activity-residual-v1-source-activity-*.json`
prefix. The training log is
`candidate-activity-residual-v1-training.jsonl`.

## Conclusion

The implementation fully separates candidate activity from conditional string
selection and improves the hierarchical decoder baseline TAB F1 by `0.051151`.
It still does not outperform the best v4 joint result (`0.655764`), and its
same-pitch wrong-string errors remain more than twice the v24 limit.

Independent activity training therefore fixes the architectural coupling but
does not fix the underlying candidate discrimination problem. Further
threshold tuning is not justified. A successor needs stronger temporal or
performance-context evidence and better supervised coverage of same-pitch
cross-string ambiguity. This checkpoint remains research-only; v24 remains
unchanged.
