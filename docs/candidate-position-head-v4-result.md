# Candidate position head v4 result

Status: rejected on the source-calibration gates. The frozen
mixture-Demucs gate was not read.

## Training result

V4 initialized the complete v2 categorical head and trained only the
zero-initialized spectral string residual. The selected epoch was epoch 5.

| Diagnostic | Baseline | V4 | Change | Result |
|---|---:|---:|---:|---|
| Governed validation oracle string accuracy | `0.860264` | `0.867327` | `+0.007062` | Pass |
| Source oracle string accuracy | `0.867460` | `0.876502` | `+0.009042` | Pass |

The checkpoint SHA-256 is
`a41980379cc6a1dcbc5b5217998749d25a75154ffab0569808b8b3f3bc5fd29c`.
Its metadata preserves the v12 encoder root and records the v2 direct
initialization hash.

## Source-calibration result

Six representative none-logit biases from `-1.0` through `0.25` were
evaluated end to end with the frozen v18 primary and v24 supplemental
candidate-activity heads. Bias `-0.5` had the best TAB F1:

| Gate | V4 | Required | Result |
|---|---:|---:|---|
| TAB F1 | `0.655764` | `>= 0.704668` | Fail |
| Same-pitch wrong-string errors | `710` | `<= 481` | Fail |
| Onset F1 | `0.811654` | `>= 0.815375` | Fail |
| Repeated TAB recall | `0.649949` | `>= 0.721959` | Fail |

No tested bias approached all gates. Bias `-1.0` preserved onset F1 at
`0.815742`, but TAB F1 was `0.654516`, repeated TAB recall was `0.685022`,
and same-pitch wrong-string errors rose to `815`. Bias `0.25` reduced
same-pitch wrong-string errors to `495`, still above the limit, while TAB F1
fell to `0.633190`.

The candidate-level scan is in
`candidate-position-head-v4-source-none-bias-scan.json`. End-to-end reports
use the `candidate-position-head-v4-source-none-bias-*.json` prefix.

## Architecture conclusion

Candidate-local harmonic evidence contains measurable string-position signal:
v4 improved both governed and source oracle accuracy. That improvement did
not translate to the decoder.

The categorical group assignment still compares `none` and individual string
probabilities inside one joint combination search. Although the residual
preserves total active probability, redistributing probability among string
classes changes the best joint combination and therefore changes candidate
acceptance, emitted-event count and downstream errors.

A successor should expose a genuinely hierarchical interface:

1. decide candidate activity from an independent frozen activity probability;
2. select a string from a separately normalized conditional string
   distribution only after acceptance;
3. calibrate and gate activity and string discrimination independently.

V4 is retained as a research checkpoint only. V24 remains unchanged.
