# Candidate position head v3 result

Status: rejected on the source-calibration gates. The frozen
mixture-Demucs gate was not read.

## Training result

V3 trained the factorized candidate head for five epochs on
`governed-v1/guitarset-mixture-combined.jsonl`. The selected epoch improved
governed validation candidate string accuracy from v2's `0.681596` to
`0.723305`.

This apparent gain did not represent better conditional string
discrimination. On the 60-track original-microphone source calibration, the
oracle active-string accuracy was `0.853302`, below v2's `0.867460`.

## Source-calibration result

The best tested end-to-end point used none-logit bias `-0.5`:

| Gate | V3 | Required | Result |
|---|---:|---:|---|
| TAB F1 | `0.669006` | `>= 0.704668` | Fail |
| Same-pitch wrong-string errors | `962` | `<= 481` | Fail |
| Onset F1 | `0.823577` | `>= 0.815375` | Pass |
| Repeated TAB recall | `0.652152` | `>= 0.721959` | Fail |

The checkpoint SHA-256 is
`58021d83155fbffaa2b452cf03bca346f161b4d6b1d0cb601c050a2d7c6aa5ef`.
The candidate-level scan is in
`candidate-position-head-v3-source-none-bias-scan.json`; the selected
end-to-end report is
`candidate-position-head-v3-source-none-bias-bm050.json`.

## Conclusion

Training activity and conditional string prediction together allowed the
factorized head to improve its selected validation metric while degrading
source-domain oracle string accuracy. V3 is not a viable replacement for v2
or v24 and is not evaluated on the frozen gate.
