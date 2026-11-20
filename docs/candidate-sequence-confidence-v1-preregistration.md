# Candidate sequence confidence rejection v1 preregistration

Status: implementation and regression tests complete; training authorized.
This is a research-only experiment. It cannot promote a production model or
authorize a release.

## Hypothesis

Candidate sequence joint decision v1 failed because one residual head had to
both rewrite the frozen string distribution and decide whether to reject a
candidate. It increased wrong-string decisions from 1,536 to 1,599 on governed
validation even though activity and repeated-position metrics improved.

This experiment freezes both predecessor decisions:

1. the candidate activity-v1 position checkpoint decides whether a pitch
   candidate is active;
2. candidate sequence string v1 chooses the top string for that candidate.

A new scalar sequence head predicts only whether the frozen top-string choice
is correct. It cannot change a string logit. At inference, the acceptance logit
is:

`minimum(frozen activity logit, predicted correctness logit)`.

The head can therefore reject an existing decision but cannot create a new
active event or move an event to another string.

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

The existing isolation audit found zero shared track, group or audio hashes
between formal training/model selection and both cross-source validation
manifests. Neither cross-source report may be read before the governed
validation gate passes.

The Basic Pitch frontend remains fixed at onset threshold `0.35`, frame
threshold `0.25` and minimum note length `70 ms`. The selected cache set
contains 1,177 unique audio files and 236,610 candidate events with aggregate
SHA-256
`cc65bed066dff338461c3dbd878cf8dd3451c268fdf17ae464c7fbfbd196a300`.

## Supervision boundary

Only reference-active unique `(group, pitch)` queries supervise the new head.
The binary target is one exactly when the frozen sequence-v1 top string is in
the valid reference-string set for that pitch. Reference-inactive candidates
receive no confidence loss; their acceptance remains controlled solely by the
frozen activity logit. They remain included in joint-position metric counts.

AG-PT-set and Guitar-TECHS retain the positive-only group rule. GuitarSet and
symbolic synthetic data retain complete positive and negative activity
supervision for the frozen input path. Sampling remains 5,151 windows with
replacement; AG-PT-set keeps weight `0.5475003196522185`, giving it an expected
10% share.

| Split | Groups | Windows | Supervised active | Inactive | Repeated | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 89,170 | 5,151 | 90,862 | 80,263 | 46,358 | 171,125 |
| Validation | 17,400 | 1,028 | 16,567 | 19,455 | 9,584 | 36,022 |

The raw supervision audit contains 15 additional same-group duplicate pitch
occurrences. `candidate_position_queries` canonicalizes them before model
input.

## Model

Each token contains normalized pitch, confidence, duration, frozen activity
logit, six frozen sequence-v1 conditional string log probabilities and six
playable-string mask values.

The architecture uses:

- token MLP: `16 -> 64 -> 64`;
- masked candidate mean and maximum pooling;
- previous and next group gaps clipped at two seconds;
- one-layer bidirectional GRU, hidden size 64 per direction;
- scalar head: `192 -> 64 -> 1`;
- sequence length 32, stride 16, overlap-average inference;
- 92,929 trainable parameters.

The final scalar weight is initialized to zero and its bias to
`logit(0.99) = 4.59511985`. Every initial correctness logit is therefore
positive. Taking the minimum with the frozen activity logit preserves every
baseline threshold-0 activity decision and decoded event.

## Objective and optimization

The objective is binary cross entropy over reference-active candidates only.
Repeated active candidates receive weight `1.5`. There is no separate
wrong-string weight because frozen wrong top-string choices are already the
negative class.

| Parameter | Value |
|---|---:|
| Epochs | 6 |
| Window batch size | 16 |
| Learning rate | `0.0003` |
| Weight decay | `0.0001` |
| Optimizer | AdamW |
| Gradient norm cap | `5.0` |
| Scheduler | cosine, minimum `0.000015` |
| Selection metric | validation joint position F1 |
| First tie-break | fewer accepted validation wrong-string decisions |
| Second tie-break | higher repeated-position recall |
| Repeat interval | `0.75 s` |
| Repeat-positive weight | `1.5` |
| Identity acceptance probability | `0.99` |
| Seed | `20260923` |

No threshold scan, onset jitter, pitch augmentation, candidate none bias,
string-margin threshold, decoder heuristic or post-result loss adjustment is
enabled.

## Evaluation order and gates

1. Train all six epochs and select only from governed GuitarSet validation.
2. Against the identity-initialized baseline, the selected epoch must have
   strictly higher joint position F1, strictly fewer accepted wrong-string
   decisions and non-decreasing repeated-position recall.
3. Failure of any governed comparison rejects the experiment without creating
   or reading confidence-v1 cross-source reports.
4. If validation passes, freeze the checkpoint and evaluate baseline and
   candidate once on AG-PT and Guitar-TECHS with activity threshold `0.50`,
   string margin `0` and the frozen Basic Pitch frontend.
5. On each cross-source set, TAB F1 must strictly increase, same-pitch
   wrong-string errors must strictly decrease, repeated TAB recall must not
   decrease, and onset F1 may decrease by at most `0.01` absolute.

The bounded onset allowance is fixed before training because selectively
rejecting same-pitch wrong-string events can remove events that were onset
true positives. TAB improvement, wrong-string reduction and repeated-note
preservation remain mandatory, preventing a reject-more strategy from passing.

Passing both cross-source gates permits only a later production-gate
preregistration; it does not change v24.

## Fixed outputs

- checkpoint:
  `checkpoints/guitarset-candidate-sequence-confidence-v1.pt`;
- training log:
  `reports/candidate-sequence-confidence-v1-training.jsonl`;
- AG-PT reports:
  `reports/candidate-sequence-confidence-v1-agpt-*.json`;
- Guitar-TECHS reports:
  `reports/candidate-sequence-confidence-v1-guitar-techs-*.json`;
- result:
  `docs/candidate-sequence-confidence-v1-result.md`.
