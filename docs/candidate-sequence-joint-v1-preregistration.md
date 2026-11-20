# Candidate sequence joint decision v1 preregistration

Status: implementation and regression tests complete; training authorized.
This is a research-only experiment. It cannot promote a production model or
authorize a release.

## Hypothesis

Candidate sequence activity v1 improved governed validation and all
cross-source onset, tablature and repeated-note metrics, but increased
same-pitch wrong-string errors on both fixed sources. Its binary activity loss
could learn whether a pitch candidate existed, but it could not directly
optimize whether accepting that candidate would emit the correct string.

This experiment replaces the separate scalar activity residual with one joint
seven-class residual per candidate:

1. reject the candidate;
2. accept it on string 1;
3. accept it on string 2;
4. accept it on string 3;
5. accept it on string 4;
6. accept it on string 5;
7. accept it on string 6.

The candidate activity-v1 position checkpoint and candidate sequence string
v1 checkpoint remain frozen. Candidate sequence activity v1 is not used as an
input or initialization source.

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

AG-PT-set and Guitar-TECHS retain the positive-only group rule from the
predecessor. GuitarSet and symbolic synthetic data retain complete positive
and negative supervision. Sampling remains 5,151 windows with replacement;
AG-PT-set keeps weight `0.5475003196522185`, giving it an expected 10% share.

The model operates on unique `(group, pitch)` queries, not raw candidate event
occurrences:

| Split | Groups | Windows | Active | Inactive | Repeated | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 89,170 | 5,151 | 90,862 | 80,263 | 46,358 | 171,125 |
| Validation | 17,400 | 1,028 | 16,567 | 19,455 | 9,584 | 36,022 |

The raw supervision audit contains 15 additional same-group duplicate pitch
occurrences. Those are canonicalized by `candidate_position_queries` before
model input and are not separate training examples.

## Model

Each candidate token contains normalized pitch, confidence, duration, frozen
base activity logit, six frozen sequence-v1 conditional string log
probabilities and six playable-string mask values.

The architecture uses:

- token MLP: `16 -> 64 -> 64`;
- masked candidate mean and maximum pooling;
- previous and next group gaps clipped at two seconds;
- one-layer bidirectional GRU, hidden size 64 per direction;
- residual head: `192 -> 64 -> 7`;
- sequence length 32, stride 16, overlap-average inference;
- 93,319 trainable parameters.

The seven base log probabilities are constructed from the frozen activity
logit and conditional string probabilities. The final residual layer is
initialized to zero, so initialization must preserve every activity
probability, conditional string probability and decoded event exactly.

## Objective and optimization

Inactive candidates target the reject class. Active candidates target the set
of valid reference strings for that pitch; the loss uses the total
probability assigned to that valid target set. A candidate whose frozen
sequence-v1 top string is not a reference string receives weight `2.0`.
Repeated positive candidates receive an additional weight `1.5`.

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
| First tie-break | fewer validation wrong-string decisions |
| Second tie-break | higher repeated-position recall |
| Repeat interval | `0.75 s` |
| Wrong-string weight | `2.0` |
| Repeat-positive weight | `1.5` |
| Seed | `20260923` |

No onset jitter, pitch augmentation, candidate none bias, string-margin
threshold, decoder heuristic or gate-driven threshold scan is enabled.

## Evaluation order and gates

1. Train all six epochs and select only from governed GuitarSet validation.
2. Against the zero-initialized baseline, the selected epoch must have:
   strictly higher joint position F1, strictly fewer wrong-string decisions,
   non-decreasing activity F1 and non-decreasing repeated-position recall.
3. Failure of any governed validation comparison rejects the experiment
   without reading cross-source reports.
4. If validation passes, freeze the checkpoint and evaluate baseline and
   candidate once on AG-PT and Guitar-TECHS with activity threshold `0.50`,
   string margin `0` and the frozen Basic Pitch frontend.
5. On AG-PT, TAB F1 must strictly increase; onset F1 and repeated TAB recall
   must not decrease; same-pitch wrong-string errors must strictly decrease.
6. On Guitar-TECHS, TAB F1 must strictly increase; onset F1 and repeated TAB
   recall must not decrease; same-pitch wrong-string errors must strictly
   decrease.

No threshold scan or post-result loss adjustment is allowed. Passing both
cross-source gates permits only a later production-gate preregistration; it
does not change v24.

## Fixed outputs

- checkpoint:
  `checkpoints/guitarset-candidate-sequence-joint-v1.pt`;
- training log:
  `reports/candidate-sequence-joint-v1-training.jsonl`;
- AG-PT reports:
  `reports/candidate-sequence-joint-v1-agpt-*.json`;
- Guitar-TECHS reports:
  `reports/candidate-sequence-joint-v1-guitar-techs-*.json`;
- result:
  `docs/candidate-sequence-joint-v1-result.md`.
