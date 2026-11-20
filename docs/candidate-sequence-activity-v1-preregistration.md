# Candidate sequence activity v1 preregistration

Status: implementation complete and training authorized. This is a
research-only experiment. It cannot promote a production model or authorize a
release.

## Hypothesis

Candidate sequence string v1 improved conditional string selection but could
not change candidate acceptance. Its best fixed source point still missed the
TAB and repeated-note gates. This experiment tests whether ordered
candidate-group context can improve the scalar candidate activity boundary,
especially repeated attacks, while preserving the existing position,
activity and sequence-string paths.

The experiment adds one zero-initialized scalar residual per candidate. The
position/activity checkpoint and candidate sequence string v1 are frozen.
String probabilities are refined by sequence string v1 before they are used as
activity-model inputs. No string residual is trained in this experiment.

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
| train supervision audit | `a16cd0acbce43d36f4bd6824d6c1d9ce76a66b9dd24264e85104b2dba40a2098` |
| validation supervision audit | `474076d70e019c4a9c11e6c98b925a0d5e989780219871b9e4376d8b2dd1f76a` |

The isolation audit found zero shared track, group or audio hashes between the
formal training/model-selection manifest and both cross-source validation
manifests. AG-PT validation and Guitar-TECHS validation are fixed research
gates. They are not used for epoch, loss, architecture or threshold selection,
and they are not production promotion gates.

The Basic Pitch candidate frontend is frozen at:

- onset threshold `0.35`;
- frame threshold `0.25`;
- minimum note length `70 ms`.

The initial cache completion reused 744 caches and generated 433 missing
caches. The finalized audit then revalidated all 1,177 unique audio files,
236,610 candidate events and 28,361 out-of-range dropped events. The selected
cache set has aggregate SHA-256
`cc65bed066dff338461c3dbd878cf8dd3451c268fdf17ae464c7fbfbd196a300`.

## Supervision boundary

AG-PT-set and Guitar-TECHS are sparse position sources. Their unmatched
candidate groups are not reliable negatives and are filtered with the
positive-only rule. GuitarSet and symbolic synthetic data retain complete
positive and negative supervision.

| Source | Groups | Windows | Active candidates | Inactive candidates | Repeated candidates |
|---|---:|---:|---:|---:|---:|
| GuitarSet | 69,571 | 4,089 | 72,093 | 69,062 | 43,423 |
| AG-PT-set, positive-only | 16,636 | 869 | 15,782 | 10,569 | 2,797 |
| Guitar-TECHS, positive-only | 1,119 | 65 | 983 | 509 | 6 |
| Symbolic synthetic | 1,844 | 128 | 2,011 | 131 | 137 |
| **Total** | **89,170** | **5,151** | **90,869** | **80,271** | **46,363** |

Positive-only filtering removes 13,354 unmatched groups. It does not remove
unmatched pitches inside a retained positive group. Validation remains the
unchanged 120-track GuitarSet split: 17,400 groups, 1,028 windows, 16,570
active candidates, 19,456 inactive candidates and 9,584 repeated candidates.

AG-PT-set would contribute 16.87% of governed windows without balancing. Its
sampling weight is frozen to:

```text
(0.10 / 0.90) * ((4089 + 65 + 128) / 869)
= 0.5475003196522185
```

This gives AG-PT-set an expected 10% of sampled windows. All other source
weights are `1`. Sampling uses replacement, 5,151 windows per epoch and seed
`20260923`.

## Model

Each candidate token contains:

1. normalized MIDI pitch;
2. Basic Pitch confidence;
3. duration clipped at two seconds;
4. frozen base candidate activity logit;
5. six frozen sequence-v1 conditional string log-probabilities;
6. six playable-string mask values.

The architecture reuses the frozen sequence-v1 dimensions and windowing:

- token MLP: `16 -> 64 -> 64`;
- masked candidate mean and maximum pooling;
- previous and next group gaps clipped at two seconds;
- one-layer bidirectional GRU, hidden size 64 per direction;
- residual head: `192 -> 64 -> 1`;
- sequence length 32, stride 16, overlap-average inference;
- 92,929 trainable parameters.

The final residual layer is initialized to zero. At initialization, every
candidate activity logit and every decoded event must be exactly unchanged.

## Objective and optimization

The loss is binary candidate activity cross entropy on
`base_activity_logit + sequence_residual`.

The active/inactive ratio does not require global positive reweighting, so the
activity-positive weight is fixed at `1.0`. Repeated candidates are 51.0% of
active training candidates; a 4x multiplier would dominate the objective.
Repeated positive candidates therefore receive a fixed `1.5` multiplier.

| Parameter | Value |
|---|---:|
| Epochs | 6 |
| Window batch size | 16 |
| Learning rate | `0.0003` |
| Weight decay | `0.0001` |
| Optimizer | AdamW |
| Gradient norm cap | `5.0` |
| Scheduler | cosine, minimum `0.000015` |
| Selection metric | validation candidate activity F1 |
| Selection tie-break | higher validation repeated-candidate recall |
| Repeat interval | `0.75 s` |
| Seed | `20260923` |

No onset jitter, pitch augmentation, candidate none bias, string-margin
threshold, decoder continuity heuristic or gate-driven threshold search is
enabled.

## Evaluation order and gates

1. Train all six epochs and select only from governed GuitarSet validation.
2. The selected epoch must strictly exceed the zero-initialized validation
   activity F1 and strictly exceed its repeated-candidate recall.
3. If either validation comparison fails, reject the experiment without
   reading the cross-source result.
4. Freeze the checkpoint. Evaluate baseline and candidate once on AG-PT
   validation and Guitar-TECHS validation with activity threshold `0.50`,
   string margin `0`, and the same Basic Pitch frontend.
5. AG-PT must have strictly higher onset F1, non-decreasing TAB F1,
   non-decreasing repeated TAB recall and no increase in same-pitch
   wrong-string errors.
6. Guitar-TECHS must have strictly higher TAB F1, non-decreasing onset F1,
   non-decreasing repeated TAB recall and no increase in same-pitch
   wrong-string errors.

No threshold scan is allowed. Cross-source failure rejects the checkpoint.
Passing both research gates permits a later, separately preregistered
production-gate design; it does not promote v1 and does not change v24.

## Fixed outputs

- checkpoint:
  `checkpoints/guitarset-candidate-sequence-activity-v1.pt`;
- training log:
  `reports/candidate-sequence-activity-v1-training.jsonl`;
- AG-PT baseline/candidate reports:
  `reports/candidate-sequence-activity-v1-agpt-*.json`;
- Guitar-TECHS baseline/candidate reports:
  `reports/candidate-sequence-activity-v1-guitar-techs-*.json`;
- result:
  `docs/candidate-sequence-activity-v1-result.md`.
