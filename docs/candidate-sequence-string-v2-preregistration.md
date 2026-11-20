# Candidate sequence string v2 preregistration

Status: implementation, assignment audit, and regression tests complete;
one fixed training run is authorized. This remains a research experiment and
cannot replace the production model without every gate below.

## Hypothesis

String-v1 improved GuitarSet conditional strings but was trained with
independent per-candidate cross entropy. The training-only assignment audit
showed two remaining failures:

- cross-source singleton mapping degraded outside GuitarSet;
- full-group uniqueness changes harmed 317 independently correct candidates
  while correcting only 107.

String-v2 fine-tunes the complete string-v1 sequence model on all verified
formal-v2 training sources with a source-balanced, assignment-aware objective.
Candidate activity remains frozen.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v2.jsonl` | `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916` |
| `governed-v1/agpt-position-validation.jsonl` | `83abea78a2786131bdf3776def93d80c4141cb825afadf44a7908c925d67d708` |
| `governed-v1/guitar-techs-position-validation.jsonl` | `6f796466be2cae19783c177b4a821389a1f1fee3a7d1346e98dd1c4761540a4b` |
| `guitarset-candidate-activity-residual-v1.pt` | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| `guitarset-candidate-sequence-string-v1.pt` | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| assignment audit | `1fad56d65486b7bd8e6c42260828e4a3c259fd3202eae4cdee0cde4cb316e815` |
| governed string-v1 baseline report | `7f21f70f70da1f63baa9a448bdad34f113d920277e434ac61452efa8f6e633cc` |
| AG-PT string-v1 baseline report | `ec5b83495fef9aab6e019f28f2cebfa2bdcd0e6611997e6eb5b2640fb1c37096` |
| Guitar-TECHS string-v1 baseline report | `d8bef975b37501ec394304737f8e233064e3ec62027b01fd5d2e2c3a6492ca2d` |

The source ledger remains version `2026-11-06.2`, SHA-256
`de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a`.
GuitarSet, AG-PT-set, and Guitar-TECHS are verified `CC-BY-4.0` sources.
Synthetic records retain their governed source policy.

The candidate frontend remains Basic Pitch onset `0.35`, frame `0.25`, and
minimum note length `70 ms`. Activity probability remains `0.50`. No activity,
joint, confidence, margin, supplemental-candidate, onset-fusion, or decoder
continuity component is enabled.

## Training-only partition

Only `split=train` records are loaded. AG-PT-set and Guitar-TECHS preserve
positive-only group filtering.

The deterministic holdout uses fraction `20%` and seed `20261122`.
GuitarSet is stratified by training performer: 24 of 120 tracks are selected
from each of players 00-03 by ascending SHA-256 rank of
`20261122:GuitarSet:{group_id}:{track_id}`. Other datasets select whole
`group_id` values by ascending SHA-256 rank of
`20261122:{dataset}:{group_id}`.

The fixed partition contains 268 annotations and has selected-key SHA-256
`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.
Selected annotations are excluded from fine-tuning and used only for epoch
selection. Governed player-04 and cross-source validation remain unread until
their gates.

## Structured objective

For each onset group:

1. retain exactly the candidates accepted by frozen activity logit `>= 0`;
2. score every playable unique-string assignment when one exists;
3. use the log-sum-exp of all feasible assignments as the denominator;
4. use the log-sum-exp of assignments where every accepted reference-active
   candidate occupies one of its reference strings as the numerator;
5. allow activity-accepted reference-inactive candidates to occupy any
   remaining string, so they compete without receiving invented labels;
6. for more than six accepted candidates or no feasible unique assignment,
   reproduce the deployed independent Cartesian fallback exactly.

The loss is denominator minus numerator. Singleton groups reduce to set-valued
conditional string likelihood and remain supervised.

Group weight is the mean existing candidate supervision weight, multiplied by
`2.0` for groups with multiple activity-accepted candidates and by `1.5` for
groups containing an accepted repeated reference. Unsatisfiable reference
groups are counted and excluded from the loss.

## Fixed optimization

| Parameter | Value |
|---|---:|
| Initialization | complete string-v1 checkpoint |
| Trainable scope | all 93,190 sequence-string parameters |
| Epochs | 4 |
| Window batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `0.0001` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.000005` |
| Gradient norm cap | `5.0` |
| Seed | `20261122` |
| Sequence window / stride | `32 / 16` |
| Precompute device | automatic |
| Fine-tuning device | CPU |

Training windows use replacement sampling with fixed source weights:

| Dataset | Weight |
|---|---:|
| GuitarSet | 1.0 |
| AG-PT-set | 2.0 |
| Guitar-TECHS | 8.0 |
| stringtrace-symbolic-synthetic | 0.5 |

These weights compensate for the audit's source imbalance without using any
validation result. There is one run and no post-result change to weights,
epochs, objective, partition, or learning rate.

## Training-only selection gate

Epoch zero is the frozen string-v1 model replayed on the fixed holdout. An
epoch is eligible only if it has:

1. strictly higher assignment candidate accuracy;
2. strictly fewer wrong-string decisions;
3. non-decreasing repeated-string true positives.

Among eligible epochs, choose higher assignment accuracy, then more repeated
true positives, then higher exact-group rate, then lower structured loss.
If no epoch is eligible, stop without governed or cross-source candidate
reports.

## Evaluation order and gates

1. Run the single fixed training job and freeze the selected checkpoint hash.
2. If the training-only gate passes, evaluate string-v2 once on governed
   GuitarSet player-04.
3. Against the frozen string-v1 governed baseline, require strictly higher
   TAB F1, strictly fewer same-pitch wrong-string errors, non-decreasing
   repeated TAB recall, and unchanged onset F1 within `1e-12`.
4. Only after governed success, evaluate AG-PT once. Apply the same four
   comparisons against its frozen string-v1 baseline.
5. Only after AG-PT success, evaluate Guitar-TECHS once with the same
   comparisons.
6. Passing all experiment gates permits a production-gate assessment only.
   Production still requires TAB F1 at least `0.75` and repeated-note recall
   at least `0.85`; no checkpoint becomes the default merely by beating
   string-v1.

There is no threshold scan, source-specific checkpoint, source-specific
decoder setting, epoch rescue, or second training run.

## Fixed outputs

- checkpoint:
  `checkpoints/guitarset-candidate-sequence-string-v2.pt`;
- training log:
  `reports/candidate-sequence-string-v2-training.jsonl`;
- governed report:
  `reports/candidate-sequence-string-v2-governed-candidate.json`;
- AG-PT report:
  `reports/candidate-sequence-string-v2-agpt-candidate.json`;
- Guitar-TECHS report:
  `reports/candidate-sequence-string-v2-guitar-techs-candidate.json`;
- result:
  `docs/candidate-sequence-string-v2-result.md`.
