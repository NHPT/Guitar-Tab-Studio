# Candidate sequence decoder calibration v1 preregistration

Status: implementation and governed-validation calibration authorized. The
frozen mixture-Demucs gate remains locked.

## Scope

Candidate sequence string v1 improved both governed-validation and independent
source oracle string accuracy, but no scalar activity threshold passed the
end-to-end source gates. This successor experiment tests one decoder-only
change: reject candidates whose conditional string distribution remains
ambiguous after sequence refinement.

This is non-blind successor evidence. The previous nine-point source activity
scan is already known. No source result may select the new policy, and this
experiment does not rewrite the sequence-v1 preregistration or relax any
production threshold.

## Frozen inputs

- Position/activity checkpoint:
  `guitarset-candidate-activity-residual-v1.pt`
- Position/activity SHA-256:
  `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e`
- Sequence checkpoint:
  `guitarset-candidate-sequence-string-v1.pt`
- Sequence SHA-256:
  `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982`
- Calibration manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`, validation split only
- Calibration manifest SHA-256:
  `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d`
- Source manifest:
  `governed-v1/guitarset-source-calibration.jsonl`
- Source manifest SHA-256:
  `7d8ce13745feca42f69dbe6df050f9d45c3f92cb3b8312f6a8d58093d85d4440`
- Candidate frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Candidate decision: hierarchical
- Onset tolerance: `50 ms`
- Model weights, candidate logits, sequence windows, activity probabilities,
  source membership and all production gates: frozen

Review queues, test splits and the frozen mixture-Demucs tracks remain locked.
No retraining is authorized.

## Decoder policy

For each candidate, let the six-way conditional string probabilities after
sequence refinement be sorted as `p1 >= p2 >= ...`. Define the conditional
string margin as `p1 - p2`.

A candidate enters the existing one-event-per-string group assignment only
when both conditions hold:

1. activity probability is at least the activity threshold;
2. conditional string margin is at least the margin threshold.

The margin is evaluated before group assignment so a rejected ambiguous
candidate cannot reserve a string. A zero margin threshold must preserve the
existing hierarchical decoder exactly.

## Governed-validation grid and selection

The fixed grid is:

- activity threshold: `0.20, 0.30, 0.40, 0.50, 0.60`;
- conditional string margin: `0.00, 0.05, 0.10, 0.15, 0.20, 0.30`.

All 30 points are evaluated on the governed GuitarSet validation split. The
in-grid baseline is activity `0.50`, margin `0.00`.

A point is eligible only when:

1. validation onset F1 is at least the in-grid baseline;
2. validation repeated TAB recall is at least the in-grid baseline.

Among eligible points, select the highest validation TAB F1. Exact ties are
resolved by fewer same-pitch wrong-string errors, then higher repeated TAB
recall, then higher onset F1, then higher margin, then higher activity
threshold. Because the baseline is in the grid, the eligible set cannot be
empty.

No source metric participates in this selection.

## Governance amendment before grid review

An identity-only overlap check performed after the baseline run, but before
reviewing the remaining grid results, found that the historical gate labels do
not describe sealed datasets:

- all 60 source-calibration track IDs occur in the 120-track calibration
  validation split;
- all 60 `guitarset-mixture-demucs-v2` validation track IDs also occur in that
  calibration split.

The overlapping mixture manifest SHA-256 is
`b40e013d75ed876b81a30256534799913a235383b2b6160fe3eb622330ebb7c6`.

The fixed grid and selection rule remain unchanged because they are still
valid for governed decoder calibration. However, the selected source rerun is
now classified only as a fixed-slice comparability check, not independent
transfer evidence. The historical mixture-Demucs set cannot be unlocked as a
frozen gate because it has already been consumed by calibration and model
selection. No additional mixture-gate run is authorized, regardless of the
source result. Production promotion remains frozen until a genuinely
non-overlapping governed gate is established.

## Fixed source comparability check

Evaluate exactly the selected policy once on the 60-track original-microphone
source calibration manifest. Classify the diagnostic as improved only if
every successor condition passes:

1. TAB F1 is strictly greater than sequence-v1's source best
   `0.6830210345007569`;
2. same-pitch wrong-string errors are strictly fewer than `785`;
3. onset F1 remains at least the existing production floor `0.815375`;
4. repeated TAB recall is at least sequence-v1's source best
   `0.6424940698068451`.

These successor conditions do not replace the production gates and cannot
authorize promotion. Production eligibility still requires all original
absolute thresholds on a future non-overlapping governed gate:

- TAB F1 `>= 0.704668`;
- same-pitch wrong-string errors `<= 481`;
- onset F1 `>= 0.815375`;
- repeated TAB recall `>= 0.721959`.

No mixture-Demucs rerun follows this check. A pass records a decoder
calibration improvement only; a failure rejects the calibrated policy.

## Fixed outputs

- validation reports:
  `reports/candidate-sequence-decoder-calibration-v1-validation-a*-m*.json`;
- selected source report:
  `reports/candidate-sequence-decoder-calibration-v1-source-selected.json`;
- result:
  `docs/candidate-sequence-decoder-calibration-v1-result.md`.
