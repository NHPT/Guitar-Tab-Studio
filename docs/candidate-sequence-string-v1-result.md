# Candidate sequence string v1 result

Status: rejected on the source end-to-end gates. No separate
mixture-Demucs evaluation run was made. A later identity audit found that the
historical gate tracks were already included in governed validation, so the
set was not a sealed unread gate.

## Data decision

The source audit confirmed that all three available real aligned sources are
verified `CC-BY-4.0`, but only GuitarSet contains enough phrase-level
same-pitch cross-string context for this experiment:

| Training candidate | Same-pitch cross-string transitions within 2 s | Decision |
|---|---:|---|
| GuitarSet governed mixture | 1,674 | Included |
| Guitar-TECHS train | 2 | Excluded as insufficient sequence evidence |
| AG-PT-set train | 0 | Excluded as insufficient sequence evidence |

Candidate sequence string v1 is therefore an honest GuitarSet-only
architecture baseline. It does not claim that an independent real
cross-string dataset was added.

## Implementation

The experiment added:

- candidate tokens containing normalized pitch, confidence, duration, six
  frozen v4 conditional string logits and a playable-string mask;
- masked candidate pooling into ordered group embeddings;
- a one-layer bidirectional GRU with 64 hidden units per direction;
- a zero-initialized per-candidate six-string residual;
- deterministic 32-group windows with stride 16 and overlap averaging;
- independent sequence checkpoint loading in oracle analysis, regular
  evaluation and research bundles;
- a reproducible source-license and cross-string coverage audit.

The complete activity-v1 position and activity path remained frozen. The
sequence residual had exactly 93,190 trainable parameters. Unit tests verify
that a zero residual returns the original logits exactly and that nonzero
residuals preserve both the `none` probability and total active probability.

## Training result

Epoch 6 was selected from six epochs.

| Epoch | Validation loss | Validation oracle string accuracy | Correct / total |
|---:|---:|---:|---:|
| 0, zero-init | 0.351887 | 0.867327 | 14,369 / 16,567 |
| 1 | 0.335393 | 0.873302 | 14,468 / 16,567 |
| 2 | 0.340426 | 0.871009 | 14,430 / 16,567 |
| 3 | 0.320065 | 0.881089 | 14,597 / 16,567 |
| 4 | 0.318538 | 0.882296 | 14,617 / 16,567 |
| 5 | 0.316052 | 0.882960 | 14,628 / 16,567 |
| 6 | 0.305554 | 0.885797 | 14,675 / 16,567 |

The selected model improved governed validation oracle accuracy by
`0.0184704533` and corrected 306 additional active-candidate string
decisions. This strictly passed the preregistered validation oracle gate.

The research checkpoint SHA-256 is
`392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982`.

## Fixed source calibration oracle

The 60-track original-microphone source calibration was read only after the
validation gate passed.

| Metric | Activity-v1 / v4 | Sequence v1 | Change | Gate |
|---|---:|---:|---:|---|
| Oracle string accuracy | 0.876502 | 0.894349 | +0.017847 | Pass |
| Correct active candidates | 7,367 / 8,405 | 7,517 / 8,405 | +150 | Pass |
| All-target errors | 1,038 | 888 | -150 | Pass |

The improvement held on the original-microphone source slice, so the
hierarchical source threshold scan was authorized under the preregistered
protocol. A later identity audit found that all 60 track IDs also occur in the
120-track governed validation split. This is therefore fixed-slice
comparability evidence, not independent transfer evidence.

## Source end-to-end scan

| Activity threshold | TAB F1 | Same-pitch wrong-string | Onset F1 | Repeated TAB recall |
|---:|---:|---:|---:|---:|
| 0.10 | 0.635858 | 1,066 | 0.791362 | 0.718231 |
| 0.20 | 0.660696 | 976 | 0.813179 | 0.702982 |
| 0.30 | 0.674583 | 906 | 0.820570 | 0.686208 |
| 0.40 | 0.681185 | 854 | 0.822914 | 0.668417 |
| 0.50 | 0.683021 | 785 | 0.818519 | 0.642494 |
| 0.60 | 0.675690 | 707 | 0.802457 | 0.606913 |
| 0.70 | 0.656953 | 607 | 0.769741 | 0.557099 |
| 0.80 | 0.625640 | 475 | 0.715906 | 0.490681 |
| 0.90 | 0.520846 | 307 | 0.581228 | 0.349034 |

Threshold `0.50` had the best TAB F1. It improved materially over
activity-v1's previous best source point, including fewer same-pitch
wrong-string errors, but failed three production comparisons:

| Gate at threshold 0.50 | Required | Actual | Result |
|---|---:|---:|---|
| TAB F1 | >= 0.704668 | 0.683021 | Fail |
| Same-pitch wrong-string | <= 481 | 785 | Fail |
| Onset F1 | >= 0.815375 | 0.818519 | Pass |
| Repeated TAB recall | >= 0.721959 | 0.642494 | Fail |

The `0.80` point met the wrong-string count in isolation, but its onset,
tablature and repeated-note recall losses make it ineligible. No threshold
passed all four gates.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Training log | `d7001d99c20eb19d61d7de0ab59c305439d51f0eb347a7af039b5b419cb8d970` |
| Source audit | `56aa8a668e6b47ea89c53c011b35b0a83d5d8f7adb48a6bcabac5e0e7003aacf` |
| Source oracle | `00ef1d5b051a98c550dfe760311287544f4cec880b27f22b0b3647c0bf153a86` |
| Source threshold 0.50 | `62d2fe48a2f4b556d2d353fc3f6df2d85b9d35a0e18e0be0b094f181c3673d2a` |

The complete nine threshold reports remain under
`apps/api/data/training/reports/`.

## Decision

The experiment validates the representation change: explicit neighboring
candidate-group context produces a substantial validation gain, transfers to
the independent source oracle and improves the source end-to-end boundary.
It is still rejected for production because the gains do not meet the fixed
TAB, wrong-string and repeated-note gates.

The checkpoint remains research-only and v24 remains unchanged. No separate
mixture-Demucs evaluation was run; the later overlap audit retired its
historical designation as an unread gate. Further tuning of the same threshold
grid or loss weights is not justified.

The next model experiment should require genuinely independent, commercially
usable multi-phrase audio with aligned string/fret/onset labels. Architecture
work should separately target candidate activity and repeated-note recall;
better conditional string selection alone cannot recover the remaining
end-to-end deficit.
