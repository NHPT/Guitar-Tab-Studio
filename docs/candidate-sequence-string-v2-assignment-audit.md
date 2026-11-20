# Candidate sequence string v2 assignment audit

Status: complete. The audit used only a deterministic training-only subset of
the formal-v2 manifest. Governed player-04 validation and both cross-source
validation manifests were not read.

## Scope

The audit used the same proposed string-v2 calibration partition:

- fraction `20%`, seed `20261122`;
- 24 tracks from each GuitarSet training player 00-03;
- one complete training group from AG-PT-set and Guitar-TECHS;
- 26 of 128 synthetic training groups;
- 268 annotations, 20,603 onset groups, and 38,875 candidates.

The selected-key SHA-256 is
`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.
The machine-readable audit SHA-256 is
`1fad56d65486b7bd8e6c42260828e4a3c259fd3202eae4cdee0cde4cb316e815`.

Both the frozen activity-v1/v4 conditional strings and string-v1 were replayed
through the deployed activity threshold and full-group unique-string
assignment. Metrics include only activity-accepted reference-active candidates
for string correctness.

## Aggregate result

| Metric | v4 conditional strings | string-v1 | Change |
|---|---:|---:|---:|
| Accepted reference-active candidates | 18,233 | 18,233 | 0 |
| Correct assigned strings | 14,834 | 14,861 | +27 |
| Wrong assigned strings | 3,399 | 3,372 | -27 |
| Assigned-string accuracy | 0.813580 | 0.815061 | +0.001481 |
| Repeated correct strings | 7,519 | 7,552 | +33 |
| Repeated string accuracy | 0.932300 | 0.936392 | +0.004092 |

The aggregate gain hides a strong source split:

| Source | v4 wrong | string-v1 wrong | Change |
|---|---:|---:|---:|
| GuitarSet | 525 | 476 | -49 |
| AG-PT-set | 2,502 | 2,516 | +14 |
| Guitar-TECHS | 117 | 123 | +6 |
| Synthetic | 255 | 257 | +2 |

String-v1 learned the GuitarSet domain but slightly degraded every other
training source. This matches the previously observed cross-source direction
and rules out treating the remaining problem as only a threshold issue.

## Group-assignment findings

String-v1 has 3,372 accepted wrong-string decisions. Of those, 2,767
(`82.1%`) occur in groups with exactly one activity-accepted candidate.
Consequently, a loss restricted to polyphonic or colliding groups would miss
most current errors.

There are still 483 groups whose independent top strings collide. Full-group
assignment changes 642 candidate strings relative to independent top-1:

- 107 changes correct an independently wrong string;
- 317 changes harm an independently correct string;
- the remaining changes are wrong under both decisions.

The unique-string optimizer is therefore often forced to choose among logits
that were trained independently rather than trained as a feasible set.
Accuracy also falls sharply for accepted group sizes five and six:

| Accepted candidates | string-v1 accuracy | Wrong strings |
|---:|---:|---:|
| 1 | 0.712818 | 2,767 |
| 2 | 0.891320 | 308 |
| 3 | 0.976019 | 70 |
| 4 | 0.964187 | 65 |
| 5 | 0.855499 | 113 |
| 6 | 0.775610 | 46 |

The largest confusion directions are adjacent-string shifts:
`6->5` (407), `5->4` (389), `4->3` (378), `3->2` (279), and `2->1` (256).

## Design decision

String-v2 must address both error classes:

1. include verified AG-PT-set and Guitar-TECHS training tracks with explicit
   source-balanced sampling so singleton string mapping is not GuitarSet-only;
2. optimize the probability of complete feasible unique-string assignments,
   with activity-accepted inactive candidates retained as unlabeled
   competitors;
3. preserve the current independent decoder fallback for groups with more
   than six accepted candidates or no feasible unique assignment;
4. weight multi-candidate and repeated-note groups without excluding
   singleton groups.

No confidence rejection, margin rejection, activity retraining, decoder
continuity heuristic, or validation-derived threshold is justified by this
audit.
