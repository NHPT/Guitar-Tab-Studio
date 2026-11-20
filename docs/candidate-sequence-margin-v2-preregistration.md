# Candidate sequence string margin v2 preregistration

Status: implementation and regression tests complete; training-only
calibration authorized. This is a research-only experiment. It cannot promote
a production model or authorize a release.

## Root cause and hypothesis

Margin v1 rejected nine activity-accepted governed candidates, but it removed
low-margin candidates before unique-string group assignment. That changed the
string assignments of surviving candidates. Its calibration also classified
each query by independent top-1 string instead of replaying the deployed group
assignment.

Margin v2 changes only that ordering:

1. apply the frozen activity threshold to every candidate in the onset group;
2. assign all activity-accepted candidates jointly with the normal
   unique-string search;
3. preserve those assignments without recomputing the group;
4. mask each assigned event whose own playable-string top-1 minus top-2
   conditional probability margin is below the frozen threshold.

At threshold zero, v2 must produce the same full-group assignments as the
existing hierarchical baseline. Increasing the threshold may remove an event,
but it cannot reassign any surviving event.

## Frozen inputs and governance

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v2.jsonl` | `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916` |
| `governed-v1/agpt-position-validation.jsonl` | `83abea78a2786131bdf3776def93d80c4141cb825afadf44a7908c925d67d708` |
| `governed-v1/guitar-techs-position-validation.jsonl` | `6f796466be2cae19783c177b4a821389a1f1fee3a7d1346e98dd1c4761540a4b` |
| `guitarset-candidate-activity-residual-v1.pt` | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| `guitarset-candidate-sequence-string-v1.pt` | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| manifest-isolation audit | `97cbbeb69272eafb98f976ce221061276eef102167579ce50c71ddd29ee9bbc5` |
| candidate-cache audit | `88bcfcee14b1df96e9f94cf7400e0eb3c0463f4ce4bcbf78ac314715270f96f5` |

The Basic Pitch frontend remains onset `0.35`, frame `0.25`, and minimum note
length `70 ms`. Activity probability remains `0.50`. Candidate none bias,
output-confidence rejection, adapter overrides, supplemental candidates,
onset fusion, and all sequence activity, joint, and confidence heads remain
disabled.

The calibration artifact uses schema version `2`, version
`candidate-sequence-margin-v2`, assignment order `post_assignment`, and match
mode `activity_then_full_group_assignment`. Loading fails if any of those
semantics, checkpoint hashes, frontend settings, or provenance fields differ.
Margin v1 artifacts retain `pre_assignment` behavior and cannot silently drive
the v2 path.

## Training-only calibration partition

Calibration uses only `split=train` annotations in the formal-v2 manifest.
AG-PT-set and Guitar-TECHS retain positive-only group handling. No model
parameters are updated.

The fraction is `20%` and seed is `20261121`. GuitarSet is stratified by
`group_id`, which represents the training performer: within every performer,
track IDs are ordered by ascending SHA-256 of:

`20261121:GuitarSet:{group_id}:{track_id}`

The first round-half-up `20%` tracks are selected, bounded to at least one per
performer. This ensures all four training performers contribute calibration
events instead of selecting only one performer as margin v1 did.

Other datasets preserve the v1 group-isolated rule. Their group IDs are
ordered by ascending SHA-256 of:

`20261121:{dataset}:{group_id}`

The first round-half-up `20%` groups are selected, bounded to at least one.
The artifact records all stratum counts and a digest of the complete selected
key set. This remains training-only calibration, not a generalization result.

## Assignment-aware threshold selection

For every onset group in the selected partition:

- baseline acceptance is frozen activity logit `>= 0`;
- conditional string probabilities use the frozen sequence-v1 logits after
  playable-string masking;
- all baseline-accepted candidates receive one full-group assignment before
  any margin decision;
- correctness compares that assigned string with the candidate's reference
  string set;
- the score remains the candidate's own top-1 minus top-2 conditional string
  probability;
- threshold simulation masks assignments and never reruns assignment.

Threshold zero is the baseline. Additional thresholds enumerate every
distinct acceptance set by midpoints between adjacent accepted-candidate
margins, including the midpoint from the largest margin below one to one.

A threshold is eligible only when it has strictly higher joint position F1,
strictly fewer accepted wrong-string decisions, and no loss of repeated-note
true positives relative to the assignment-aware calibration baseline.
Selection maximizes joint F1, then minimizes wrong-string decisions, then
maximizes joint true positives, then chooses the lower threshold. If no
threshold is eligible, calibration fails and no v2 governed or cross-source
candidate report may be created.

No repeated-note exemption is added at inference. Repeat protection comes
from performer-stratified training calibration and the unchanged no-loss gate,
so every event remains subject to the same frozen margin rule.

## Evaluation order and gates

1. Create one v2 artifact from the fixed training-only partition.
2. If calibration passes, freeze its threshold and artifact hash.
3. Evaluate threshold zero once on governed GuitarSet validation using
   `post_assignment`.
4. Evaluate the artifact threshold once on the same governed split.
5. Governed validation requires strictly higher TAB F1, strictly fewer
   same-pitch wrong-string errors, and non-decreasing repeated TAB recall.
6. Failure rejects v2 without creating or reading v2 cross-source reports.
7. Only after governed success, evaluate baseline and candidate once on AG-PT
   and Guitar-TECHS.
8. Each cross-source set requires strictly higher TAB F1, strictly fewer
   same-pitch wrong-string errors, non-decreasing repeated TAB recall, and no
   more than `0.01` absolute onset F1 loss.

There is no governed threshold scan, post-result adjustment, source-specific
threshold, fallback to margin v1, or repeat-specific rescue.

## Fixed outputs

- calibration artifact:
  `reports/candidate-sequence-margin-v2-calibration.json`;
- governed reports:
  `reports/candidate-sequence-margin-v2-governed-{baseline,candidate}.json`;
- AG-PT reports:
  `reports/candidate-sequence-margin-v2-agpt-{baseline,candidate}.json`;
- Guitar-TECHS reports:
  `reports/candidate-sequence-margin-v2-guitar-techs-{baseline,candidate}.json`;
- result:
  `docs/candidate-sequence-margin-v2-result.md`.
