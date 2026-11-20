# Candidate sequence string margin v1 preregistration

Status: implementation and regression tests complete; training-only
calibration authorized. This is a research-only experiment. It cannot promote
a production model or authorize a release.

## Hypothesis

Candidate sequence confidence rejection v1 did not change governed validation
decisions. Its correctness logit had AUROC `0.539201`, while the already frozen
sequence string model's top-1 versus top-2 conditional probability margin had
AUROC `0.840074`.

This experiment replaces the learned confidence rejection head with one scalar
margin threshold. It does not train another model and cannot rewrite activity
or string logits:

1. the candidate activity-v1 position checkpoint supplies the activity logit;
2. candidate sequence string v1 supplies conditional string probabilities;
3. a candidate is accepted only when its activity logit is at least zero and
   its top-1 minus top-2 string probability margin is at least the calibrated
   threshold.

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

The Basic Pitch frontend is fixed at onset threshold `0.35`, frame threshold
`0.25` and minimum note length `70 ms`. The activity probability threshold is
fixed at `0.50`. Candidate none bias, output-confidence rejection, adapter
scale overrides, supplemental candidates, onset fusion and all sequence
activity, joint and confidence heads are disabled.

The existing isolation audit found no shared track, group or audio hashes
between formal training/model selection and the two cross-source validation
manifests. The calibration process must not load governed validation or either
cross-source manifest.

## Training-only calibration partition

Calibration uses only annotations whose manifest split is `train` in the
formal-v2 manifest. AG-PT-set and Guitar-TECHS retain their positive-only group
handling. No model parameters are updated.

The partition unit is `(dataset, group_id)`. Within each dataset, unique groups
are ordered by ascending SHA-256 of:

`20261120:{dataset}:{group_id}`

The first `round-half-up(20%)` groups are selected, bounded to at least one and
at most all available groups. Duplicate annotations with the same dataset and
group ID cannot cross the partition boundary. The artifact records per-dataset
counts and the SHA-256 of the complete selected-key list.

This partition is training-only calibration data, not an independent estimate
of generalization. Only the untouched governed validation gate can establish
whether the threshold generalizes.

## Threshold selection

For every candidate query in the calibration partition:

- baseline acceptance is frozen activity logit `>= 0`;
- the predicted string is the frozen sequence-v1 top conditional class after
  playable-string masking;
- the score is top-1 probability minus top-2 probability;
- a correct joint position requires a reference-active query and a predicted
  string in its reference-string set;
- an accepted inactive query or accepted wrong-string query is a joint false
  positive;
- an active rejected or wrong-string query is a joint false negative.

Threshold zero is the baseline. Every additional evaluated threshold is the
midpoint between adjacent distinct margins among baseline-accepted candidates;
the last threshold is the midpoint between the largest margin below one and
one. This enumerates every distinct acceptance set without choosing a
threshold from governed validation.

A threshold is eligible only when, relative to calibration baseline, it has:

1. strictly higher joint position F1;
2. strictly fewer accepted wrong-string decisions;
3. non-decreasing repeated-position true positives.

Among eligible thresholds, selection maximizes joint F1, then minimizes
wrong-string decisions, then maximizes joint true positives, then chooses the
lower threshold. If no threshold is eligible, calibration fails and no
governed or cross-source candidate report may be created.

## Evaluation order and gates

1. Create one calibration artifact from the fixed training-only partition.
2. If its calibration gate passes, freeze its threshold and artifact hash.
3. Evaluate baseline margin `0` once on governed GuitarSet validation.
4. Evaluate the artifact threshold once on the same governed validation.
5. Governed validation requires strictly higher joint position F1, strictly
   fewer accepted wrong-string decisions and non-decreasing repeated-position
   recall.
6. Failure of any governed comparison rejects the experiment without creating
   or reading margin-v1 cross-source reports.
7. Only after governed validation passes, evaluate baseline and candidate once
   on AG-PT and Guitar-TECHS.
8. On each cross-source set, TAB F1 must strictly increase, same-pitch
   wrong-string errors must strictly decrease, repeated TAB recall must not
   decrease, and onset F1 may decrease by at most `0.01` absolute.

There is no governed threshold scan, threshold adjustment, source-specific
threshold or post-result rescue. Passing every gate permits only a later
production-gate preregistration; it does not replace v24.

## Fixed outputs

- calibration artifact:
  `reports/candidate-sequence-margin-v1-calibration.json`;
- governed reports:
  `reports/candidate-sequence-margin-v1-governed-{baseline,candidate}.json`;
- AG-PT reports:
  `reports/candidate-sequence-margin-v1-agpt-{baseline,candidate}.json`;
- Guitar-TECHS reports:
  `reports/candidate-sequence-margin-v1-guitar-techs-{baseline,candidate}.json`;
- result:
  `docs/candidate-sequence-margin-v1-result.md`.
