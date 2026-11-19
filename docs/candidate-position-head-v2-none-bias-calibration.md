# Candidate position head v2 none-bias calibration

## Decision

None-logit calibration partially corrects candidate head v2's over-rejection,
but does not recover v24 tablature quality. The best calibrated point remains
a diagnostic result only. V2 stays rejected, v24 stays unchanged, and the next
model iteration must improve candidate-specific string evidence rather than
ship a post-hoc bias.

## Governed calibration set

- Source manifest: `governed-v1/guitarset-mixture-combined.jsonl`
- Calibration manifest: `governed-v1/guitarset-source-calibration.jsonl`
- Calibration manifest SHA-256:
  `7d8ce13745feca42f69dbe6df050f9d45c3f92cb3b8312f6a8d58093d85d4440`
- Tracks: 60 original GuitarSet microphone validation tracks
- Excluded tracks: every ID ending in `-mixture-demucs-guitar`
- Frozen mixture-Demucs gate: not used for calibration or threshold selection
- Candidate frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Checkpoint SHA-256:
  `37267c090204492070640f7aed10da6d90fa73b5654f2b321ddc8642928d5d0c`

The scanner extracted logits once for 17,481 candidate queries, then applied
each offset only to the categorical `none` logit before softmax.

## Candidate-level scan

| None bias | Categorical accuracy | Acceptance F1 | Acceptance recall | Active string accuracy | False accepts | False rejects | Wrong strings |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.761913 | 0.786004 | 0.771089 | 0.695776 | 1,605 | 1,924 | 633 |
| -0.50 | 0.754705 | 0.803979 | 0.851041 | 0.755860 | 2,236 | 1,252 | 800 |
| -0.75 | 0.748641 | 0.807738 | 0.881737 | 0.778703 | 2,534 | 994 | 866 |
| -1.00 | 0.738974 | 0.806470 | 0.904581 | 0.795836 | 2,847 | 802 | 914 |
| -1.50 | 0.712373 | 0.795841 | 0.938013 | 0.821059 | 3,524 | 521 | 983 |

`-0.75` maximized candidate acceptance F1. More negative offsets continued
to recover active candidates, but false accepts and wrong-string decisions
grew monotonically.

## End-to-end calibration

All rows use the same 60 original microphone tracks. V24 uses its frozen
direct decoder; v2 uses the frozen primary and supplemental activity heads,
candidate identity merge, and only changes the none-logit bias.

| Model | Onset F1 | TAB F1 | TAB precision | TAB recall | Repeated TAB recall | Exact TAB | Same-pitch wrong-string | Emitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v24 | 0.815375 | 0.704668 | 0.700424 | 0.708963 | 0.721959 | 7,269 | 481 | 10,378 |
| v2, 0.00 | 0.781061 | 0.643744 | 0.736829 | 0.571540 | 0.582006 | 5,860 | 526 | 7,953 |
| v2, -0.50 | 0.812336 | 0.659003 | 0.695365 | 0.626256 | 0.639105 | 6,421 | 665 | 9,234 |
| v2, -0.75 | 0.819958 | 0.662745 | 0.677745 | 0.648396 | 0.661810 | 6,648 | 726 | 9,809 |
| v2, -1.00 | 0.820667 | 0.661196 | 0.658414 | 0.664001 | 0.677736 | 6,808 | 781 | 10,340 |
| v2, -1.50 | 0.812992 | 0.654156 | 0.623091 | 0.688481 | 0.701627 | 7,059 | 861 | 11,329 |

The best v2 TAB F1 occurs at `-0.75`. Relative to bias `0`, it improves TAB
F1 by `0.019001` and repeated TAB recall by `0.079804`. Relative to v24,
however, it remains lower by `0.041923` TAB F1 and `0.060149` repeated TAB
recall, while producing 245 more same-pitch wrong-string errors.

## Attribution

The experiment confirms that v2 has a real calibration problem: lowering the
none logit recovers onset recall and event count. It also confirms that
calibration is not the root fix. As rejection is relaxed, string-selection
errors increase enough to keep TAB F1 below v24 at every representative point.

The successor should retain pre-assignment candidate identity but separate
candidate activity from string selection more cleanly and add
candidate-specific spectral or temporal evidence. No v2 bias is promoted,
and the frozen mixture-Demucs gate remains untouched.

Evidence:

- `candidate-position-head-v2-source-none-bias-scan.json`
- `candidate-position-head-v2-source-none-bias-b000.json`
- `candidate-position-head-v2-source-none-bias-bm050.json`
- `candidate-position-head-v2-source-none-bias-bm075.json`
- `candidate-position-head-v2-source-none-bias-bm100.json`
- `candidate-position-head-v2-source-none-bias-bm150.json`
- `v24-source-calibration-baseline.json`
