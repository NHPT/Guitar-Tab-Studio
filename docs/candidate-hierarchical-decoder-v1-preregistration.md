# Candidate hierarchical decoder v1 preregistration

Status: implementation and source-calibration evaluation authorized. The
frozen mixture-Demucs gate remains locked.

## Hypothesis

Candidate position head v4 improved source oracle string accuracy, but the
gain did not translate end to end because the categorical group search still
compared `none` against individual string probabilities. Changing the
conditional string distribution therefore changed both position selection and
candidate acceptance.

The hierarchical decoder separates those decisions:

1. derive candidate activity probability as the total probability mass of all
   playable string classes;
2. accept or reject each candidate using only that activity probability;
3. normalize the playable string classes conditionally on activity;
4. assign strings only among accepted candidates, with the existing
   one-event-per-string group constraint.

The conditional string distribution cannot reject a candidate. No
conditional-string confidence threshold is introduced.

## Frozen inputs

- Position checkpoint:
  `guitarset-candidate-position-head-v4.pt`
- Position checkpoint SHA-256:
  `a41980379cc6a1dcbc5b5217998749d25a75154ffab0569808b8b3f3bc5fd29c`
- Primary group-activity checkpoint:
  `guitarset-candidate-activity-temporal-v18.pt`
- Supplemental group-activity checkpoint:
  `guitarset-low-candidate-activity-v24.pt`
- Source-calibration manifest:
  `governed-v1/guitarset-source-calibration.jsonl`
- Primary Basic Pitch frontend: `0.35 / 0.25 / 70 ms`
- Supplemental Basic Pitch frontend: `0.25 / 0.20 / 70 ms`
- Group-activity thresholds: primary `0.20`, supplemental `0.85`
- Candidate none-logit bias: `0`
- Conditional string rejection threshold: none
- Output confidence threshold: `0`

No model parameter, candidate logit, group-activity decision, candidate merge
rule or onset tolerance is changed.

## Activity threshold scan

The following candidate activity thresholds are preregistered:

`0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`

All nine points are evaluated end to end on the 60 original-microphone source
calibration tracks. The selected point is the highest source TAB F1. Ties are
resolved by fewer same-pitch wrong-string errors, then higher repeated TAB
recall, then the higher activity threshold.

## Promotion gates

The hierarchical decoder may enter the frozen mixture-Demucs gate only if its
selected source point passes every gate:

1. TAB F1 `>= 0.704668`.
2. Same-pitch wrong-string errors `<= 481`.
3. Onset F1 `>= 0.815375`.
4. Repeated TAB recall `>= 0.721959`.

If any source gate fails, the decoder is rejected without reading the frozen
mixture-Demucs tracks. Passing all source gates authorizes the existing frozen
gate only; it does not promote a checkpoint or alter the v24 bundle.
