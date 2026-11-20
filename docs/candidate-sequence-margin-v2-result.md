# Candidate sequence string margin v2 result

Status: rejected at the training-only calibration gate. No governed or
cross-source v2 reports were created or read. The calibration artifact is
research-only and cannot replace v24 or authorize a release.

## Implemented semantics

Margin v2 now performs the required order:

1. apply activity acceptance;
2. assign the complete accepted onset group with the normal unique-string
   optimizer;
3. retain that full-group assignment;
4. mask assigned events below the frozen per-event string margin.

A rejected event can no longer cause surviving events to be reassigned.
Margin v1 remains explicitly `pre_assignment`; v2 artifacts require
`post_assignment` and assignment-aware calibration metadata.

The calibration replay also uses the actual full-group baseline assignment
instead of independent top-1 strings. A collision regression proves that
thresholding one event after assignment leaves the other event's baseline
string unchanged.

## Protocol

The experiment followed
`candidate-sequence-margin-v2-preregistration.md`:

- frozen candidate activity-v1 position and sequence string-v1 checkpoints;
- Basic Pitch onset `0.35`, frame `0.25`, minimum note length `70 ms`;
- deterministic `20%` training-only calibration with seed `20261121`;
- GuitarSet sampling stratified across all four training performers;
- activity probability threshold `0.50`;
- no repeat exemption or additional rejection head;
- no governed threshold scan or post-result threshold adjustment.

The preregistration SHA-256 is
`aa68c17ccfb6ca78d5279c7e4c65c479fbe29946b1edd3eb25734cab4559ba5f`.

## Training-only calibration

The selected partition contained 189 annotations, 17,005 onset groups, and
31,278 candidate queries. GuitarSet contributed 24 tracks from each of
players 00, 01, 02, and 03. The selected-key SHA-256 was
`646e7acf40babaf80d7facc7a1d391e789f9c4ebdab4cff325c50092d3b0efba`.

The assignment-aware baseline accepted 17,065 candidates:

| Metric | Margin 0 |
|---|---:|
| Joint position F1 | 0.773050268 |
| Joint true positives | 13,218 |
| Joint false positives | 3,847 |
| Joint false negatives | 3,914 |
| Wrong-string decisions | 2,100 |
| Repeated-position TP | 7,333 |
| Repeated-position recall | 0.828962243 |

The fixed selector evaluated 16,218 distinct acceptance sets. None
simultaneously achieved strictly higher joint F1, strictly fewer wrong-string
decisions, and non-decreasing repeated-position TP. Therefore:

- `calibration_gate_passed` is `false`;
- `eligible_thresholds` is `0`;
- `selected_threshold` falls back to `0.0` for diagnostics only;
- the artifact cannot be loaded for candidate evaluation because failed
  calibrations are rejected.

The calibration artifact SHA-256 is
`26b76cfe5a4c3d3efb97e48aafe835760cd1f8e7db818c9c6ca66fe8f55d39bd`.

## Governed boundary

Per preregistration, calibration failure stops the experiment before governed
validation. The only v2 report created is:

`reports/candidate-sequence-margin-v2-calibration.json`

No files matching the fixed governed, AG-PT, or Guitar-TECHS v2 report names
were created.

## Decision

The root-cause implementation is valid: group assignments are now fixed
before low-margin masking, calibration measures the deployed decision, and
v1/v2 artifacts cannot cross semantic modes.

The proposed scalar margin rule still lacks a training-calibrated nonzero
threshold that preserves repeated-note true positives while improving both
joint F1 and wrong-string errors. Margin v2 is therefore rejected as a model
change. Its post-assignment mechanism remains available for a separately
preregistered score or decision rule, but threshold `0.0` is the only
authorized outcome from this experiment.
