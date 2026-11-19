# Candidate sequence decoder calibration v1 result

Status: rejected. The selected decoder is the unchanged zero-margin baseline,
and production promotion remains frozen.

## Experiment

The activity-v1 position model and sequence-v1 residual were frozen. The only
new decoder parameter was the top-1 versus top-2 conditional string
probability margin. Candidates had to pass both the activity threshold and the
string-margin threshold before the existing one-event-per-string group
assignment.

The governed-validation grid contained 30 fixed points:

- activity: `0.20, 0.30, 0.40, 0.50, 0.60`;
- string margin: `0.00, 0.05, 0.10, 0.15, 0.20, 0.30`.

The zero-margin implementation preserved every activity-accepted candidate,
and its source result exactly reproduced the previous sequence-v1 report.

## Validation selection

The in-grid baseline was activity `0.50`, margin `0.00`.

| Metric | Baseline |
|---|---:|
| TAB F1 | `0.664779` |
| Same-pitch wrong-string | `1,672` |
| Onset F1 | `0.809816` |
| Repeated TAB recall | `0.619705` |
| Emitted events | `17,636` |

Only five points preserved both baseline onset F1 and repeated TAB recall:

| Activity | Margin | TAB F1 | Wrong-string | Onset F1 | Repeat recall |
|---:|---:|---:|---:|---:|---:|
| **0.50** | **0.00** | **0.664779** | **1,672** | **0.809816** | **0.619705** |
| 0.40 | 0.00 | 0.662798 | 1,800 | 0.814998 | 0.645290 |
| 0.40 | 0.05 | 0.661252 | 1,732 | 0.811502 | 0.640037 |
| 0.30 | 0.00 | 0.657464 | 1,907 | 0.814042 | 0.667316 |
| 0.30 | 0.05 | 0.656315 | 1,833 | 0.811299 | 0.662064 |

The preregistered selector therefore chose activity `0.50`, margin `0.00`.
Every nonzero margin reduced validation TAB F1 at the same activity threshold.
For example, margin `0.05` at activity `0.50` removed 209 candidates and 65
wrong-string errors, but also reduced TAB F1 by `0.001504`, onset F1 by
`0.003447` and repeated TAB recall by `0.004744`.

The conditional margin is thus correlated with ambiguity, but it does not
separate harmful wrong-string events from correct low-margin events well
enough to improve the end-to-end boundary.

## Fixed source comparability

The selected zero-margin policy was rerun on the fixed 60-track original
microphone slice.

| Metric | Previous sequence-v1 | Calibration v1 | Change |
|---|---:|---:|---:|
| TAB F1 | 0.6830210345 | 0.6830210345 | 0 |
| Same-pitch wrong-string | 785 | 785 | 0 |
| Onset F1 | 0.8185187118 | 0.8185187118 | 0 |
| Repeated TAB recall | 0.6424940698 | 0.6424940698 | 0 |
| Emitted events | 8,906 | 8,906 | 0 |

The comparison uses the exact previous TAB baseline
`0.6830210345007569`. The shorter ten-decimal rendering is not treated as a
strict improvement.

The policy fails the successor requirement for a strict TAB improvement and
fewer than 785 wrong-string errors. It also still fails three absolute
production gates:

| Production gate | Required | Actual | Result |
|---|---:|---:|---|
| TAB F1 | >= 0.704668 | 0.683021 | Fail |
| Same-pitch wrong-string | <= 481 | 785 | Fail |
| Onset F1 | >= 0.815375 | 0.818519 | Pass |
| Repeated TAB recall | >= 0.721959 | 0.642494 | Fail |

## Gate governance correction

An identity audit found that the historical dataset names do not describe
independent sealed gates:

- all 60 source-calibration track IDs occur in the 120-track governed
  validation split;
- all 60 `guitarset-mixture-demucs-v2` validation track IDs also occur in that
  same split.

The source rerun is therefore a fixed-slice comparability check, not
independent transfer evidence. The historical mixture-Demucs set cannot be
used as an unread promotion gate, so it was not rerun. Production promotion
is administratively frozen until a genuinely non-overlapping governed gate is
created.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Selected validation report | `e8d30280eb450e46d71b456c06e03a90808334fadfe286dc032e0aae41ac5d9c` |
| Selected source report | `c4a7d81301665205da40b906c1a16fb488504a830ca7e145102fcd2fbc7a8ac9` |
| Position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| Sequence checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |

All 30 validation reports remain under
`apps/api/data/training/reports/` with the
`candidate-sequence-decoder-calibration-v1-validation-` prefix.

## Decision

The margin parameter remains available as a research diagnostic with default
`0`, which preserves old behavior. No nonzero threshold is selected, no model
is retrained, sequence-v1 remains research-only, and v24 remains unchanged.

Further scalar abstention tuning is not justified. The next useful experiment
must improve candidate activity and repeated-note recall or introduce new,
commercially usable aligned performances. Before any promotion claim, the
project also needs a new non-overlapping gate with immutable membership and a
recorded manifest hash.
