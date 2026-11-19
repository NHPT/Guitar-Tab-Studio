# Candidate temporal string v1 result

Status: rejected on the source oracle gate. The end-to-end source threshold
scan and frozen mixture-Demucs gate were not run.

## Implementation

The experiment added:

- 20 reference-free, bidirectional candidate trajectory features;
- a zero-initialized temporal conditional-string residual;
- deterministic ambiguity, rare `(pitch, string)` and same-pitch cross-string
  supervision weights;
- training, evaluation, bundle and calibration compatibility;
- candidate-context diagnostics grouped by playable positions, string pair,
  fret, inter-onset interval and same-pitch history.

The activity-v1 encoder, categorical head, v4 spectral string residual and
candidate activity residual remained frozen. Only `2,849` temporal residual
parameters were trainable. Initialization preserved both conditional string
probabilities and activity logits.

Training supervision contained `83,995` matched events:

| Diagnostic | Count |
|---|---:|
| Ambiguous playable-position events | `79,951` |
| Same-pitch cross-string events | `2,182` |
| Events with weight above 1 | `79,951` |
| Mean supervision weight | `1.409160` |
| Maximum supervision weight | `3.75` |

## Training result

Epoch 2 was selected from five epochs.

| Metric | Activity-v1 / v4 | Temporal v1 | Change | Gate |
|---|---:|---:|---:|---|
| Governed validation oracle string accuracy | `0.867327` | `0.867447` | `+0.000121` | Pass |
| Source oracle string accuracy | `0.876502` | `0.876502` | `0.000000` | Fail |

The validation gain corresponds to only a very small number of candidate
decisions and did not transfer to the 60 original-microphone source tracks.
The source model produced the same `7,367 / 8,405` correct active-candidate
string decisions as v4.

The checkpoint SHA-256 is
`6517989dfe0ec6dc142fbdc887806010d396454dc604078ba77f284baf0fa915`.
It remains research-only.

## Source diagnostics

The source failure distribution remained unchanged:

- every one of the `1,035` single-target errors occurred on an ambiguous
  playable pitch;
- string 6 accuracy remained `0.781046`;
- fret 9-12 accuracy remained `0.697222`;
- dominant adjacent-string confusions remained `3 -> 2` (`208`), `6 -> 5`
  (`134`), `5 -> 4` (`113`) and `2 -> 3` (`106`);
- short and cross-string same-pitch transitions remained materially harder
  than same-string repeats.

Evidence is recorded in:

- `candidate-temporal-string-v1-training.jsonl`;
- `candidate-temporal-string-v1-source-oracle.json`;
- `candidate-string-context-v1-governed-train.json`;
- `candidate-string-context-v1-governed-validation.json`;
- `candidate-string-context-v1-source-calibration.json`.

## Decision

The experiment is rejected before end-to-end source calibration because the
preregistered source oracle gate required a strict improvement. V24 remains
unchanged, and no frozen-gate evidence was consumed.

The result indicates that a tiny residual over hand-designed candidate
trajectory summaries cannot resolve the remaining string ambiguity. Further
weight or threshold tuning on this architecture is not justified. A successor
should change the representation rather than add another scalar heuristic:

1. encode a sequence of candidate-group embeddings with an explicit temporal
   model;
2. predict neighboring string-position hypotheses jointly, not only physical
   minimum-movement summaries;
3. add genuinely new aligned cross-string performances, especially strings
   3/6 and frets 9-12, instead of repeatedly reweighting the same GuitarSet
   events.
