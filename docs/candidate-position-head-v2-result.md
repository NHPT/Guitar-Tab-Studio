# Candidate position head v2 result

## Decision

Candidate position head v2 is rejected. It remains a research checkpoint and
must not replace v24, enter a production bundle, receive additional epochs or
proceed to cross-domain evaluation.

## Governed run

- Frozen encoder: `guitarset-onset-position-mixture-v12.pt`
- Trainable parameters: `107,679`
- Training groups: `69,571`
- Validation groups: `17,400`
- Epochs: `3`
- Best validation categorical accuracy: `0.763089`
- Validation candidate acceptance F1: `0.776626`
- Validation candidate rejection F1: `0.817319`
- Validation active-candidate string accuracy: `0.681596`
- Frozen evaluation: 60 GuitarSet mixture-Demucs validation tracks
- Independent review queues: not used

The research checkpoint SHA-256 is
`37267c090204492070640f7aed10da6d90fa73b5654f2b321ddc8642928d5d0c`.

## Frozen gate

| Metric | v24 | candidate head v2 | Delta |
|---|---:|---:|---:|
| Onset F1 | 0.804544 | 0.763320 | -0.041223 |
| TAB F1 | 0.679886 | 0.608861 | -0.071025 |
| Repeated TAB recall | 0.690274 | 0.533548 | -0.156727 |
| Exact tablature events | 6,943 | 5,428 | -1,515 |
| Same-pitch wrong-string | 495 | 538 | +43 |
| Final emitted events | 10,171 | 7,577 | -2,594 |

Every promotion condition failed. V2 improved TAB F1 by `0.025097` and
reduced same-pitch wrong-string errors by `365` relative to rejected v1, but
it remained materially below v24.

## Attribution

The frontend identity change behaved as intended. Primary and supplemental
activity gating accepted `15,656` and `14,458` candidates respectively.
Pitch-and-onset identity merging suppressed `11,316` supplemental duplicates
before position assignment and produced `18,795` unique candidates. The v1
late-merge inflation therefore did not recur.

The remaining regression is candidate-head rejection. The categorical
decoder emitted only `7,577` events, compared with `10,171` for v24. Onset
precision increased to `0.898113`, but onset recall fell to `0.663708`.
TAB precision increased to `0.716379`, while TAB recall fell to `0.529406`.
The `none` decision removed too many valid candidates, and the retained
candidate string accuracy was still insufficient to reduce wrong-string
errors below v24.

Additional epochs would optimize the same over-rejecting objective and are
not justified. A successor must preserve pre-assignment candidate identity,
but must address categorical class balance and add candidate-specific evidence
before another governed run. Threshold calibration or loss changes require a
new preregistration rather than post-hoc tuning on this gate.

The v24 research bundle and Basic Pitch production default remain unchanged.
