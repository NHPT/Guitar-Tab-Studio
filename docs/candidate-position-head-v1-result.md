# Candidate position head v1 result

## Decision

Candidate position head v1 is rejected. It remains a research checkpoint and
must not replace v24, enter a production bundle, or receive additional epochs.

## Governed run

- Frozen encoder: `guitarset-onset-position-mixture-v12.pt`
- Trainable parameters: `107,422`
- Training groups: `69,571`
- Validation groups: `17,400`
- Epochs: `3`
- Best internal candidate position F1: `0.643791`
- Frozen evaluation: 60 GuitarSet mixture validation tracks
- Independent review queues: not used

## Frozen gate

| Metric | v24 | candidate head v1 | Delta |
|---|---:|---:|---:|
| Onset F1 | 0.804544 | 0.773866 | -0.030678 |
| TAB F1 | 0.679886 | 0.583764 | -0.096122 |
| Repeated TAB recall | 0.690274 | 0.639444 | -0.050830 |
| Exact tablature events | 6,943 | 6,375 | -568 |
| Same-pitch wrong-string | 495 | 903 | +408 |
| Final emitted events | 10,171 | 11,588 | +1,417 |

Every promotion condition failed. The model increased recall by emitting more
events, but precision and string assignment deteriorated substantially.

## Attribution

The supplemental frontend amplified the regression: accepted supplemental
events increased from `177` to `1,213` because independently assigned
positions no longer merged reliably.

This was not the sole cause. With the supplemental frontend disabled,
primary-only TAB F1 was `0.590750`, repeated TAB recall was `0.606235`, and
same-pitch wrong-string errors increased to `941`. The candidate head itself
therefore failed before merge behavior was considered.

The six independent binary string outputs do not model abstention and string
selection as one decision. Group-level activity acceptance can admit an
unsupported pitch, after which one or more string probabilities may pass the
position threshold. More epochs would optimize the same mismatched objective
and are not justified by this result.

## Next architecture constraint

A successor must use one candidate-level categorical decision with seven
states: `none`, strings 1 through 6. It must:

1. learn candidate rejection and string choice jointly;
2. enforce playable pitch/string positions before softmax;
3. preserve one identity across primary and supplemental frontends before
   position assignment;
4. report candidate rejection precision/recall separately from string
   accuracy;
5. pass the same frozen 60-track gate before any cross-domain or independent
   evaluation.

The v24 research bundle and Basic Pitch production default remain unchanged.
