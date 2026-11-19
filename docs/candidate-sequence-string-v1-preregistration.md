# Candidate sequence string v1 preregistration

Status: implementation and training authorized. The frozen mixture-Demucs
gate remains locked.

## Hypothesis

Candidate temporal string v1 showed that a small residual over hand-designed
trajectory summaries did not change the independent source decision boundary.
This experiment tests whether jointly encoding ordered candidate groups and
their neighboring local string hypotheses improves conditional string
selection while preserving candidate activity exactly.

The data audit found no usable new phrase-level cross-string source:
AG-PT-set train contains zero same-pitch cross-string transitions within two
seconds, and Guitar-TECHS train contains only two. Candidate sequence string
v1 therefore uses only the governed GuitarSet data and is an architecture
baseline, not a claim of new independent data coverage.

## Frozen inputs

- Position/activity initialization:
  `guitarset-candidate-activity-residual-v1.pt`
- Initialization SHA-256:
  `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e`
- Conditional string root:
  `guitarset-candidate-position-head-v4.pt`
- V4 SHA-256:
  `a41980379cc6a1dcbc5b5217998749d25a75154ffab0569808b8b3f3bc5fd29c`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Manifest SHA-256:
  `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d`
- Source ledger SHA-256:
  `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a`
- Candidate frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Seed: `20260923`
- Review queues, test splits and frozen promotion gates: locked

The encoder, legacy fret output, candidate pitch embedding, categorical
candidate head, v4 spectral string residual and activity-v1 residual are
frozen. Only the new sequence string residual is trainable. Candidate activity
logits and total active probability must be numerically identical before and
after sequence refinement.

## Sequence records and split isolation

The candidate cache contains 86,971 ordered groups and 177,181 candidates.
The largest group contains 11 candidates; no candidates may be truncated.

| Split | Tracks | Candidate groups | Windows |
|---|---:|---:|---:|
| Train | 480 | 69,571 | 4,089 |
| Validation | 120 | 17,400 | 1,028 |

Windows contain 32 consecutive candidate groups with stride 16. A final
right-aligned window is added when the regular stride does not cover the end
of a track. Short tracks produce one padded window. Windows never cross track
or performer-group boundaries. Validation remains unweighted.

## Model

Each candidate token contains:

1. normalized MIDI pitch;
2. Basic Pitch confidence;
3. duration clipped at two seconds;
4. the six frozen v4 conditional string log-probabilities;
5. a six-value playable-string mask.

A two-layer MLP maps the 15-value token to 64 dimensions. Masked mean and
masked maximum pooling form a 128-value candidate-group embedding. Previous
and next group onset gaps, each clipped at two seconds, are appended.

A one-layer bidirectional GRU with hidden size 64 per direction encodes the
ordered group sequence. For each candidate, its 64-value token embedding and
the 128-value bidirectional group state feed a `192 -> 64 -> 6` residual head.
The resulting sequence module has exactly 93,190 trainable parameters. The
final layer is initialized to zero. Invalid physical string positions are
masked after adding the residual.

Training and inference use the same `32/16` overlapping windows. Inference
averages residual logits for groups covered by more than one window.

## Objective and optimization

Only candidates matched to an active reference pitch contribute conditional
six-string cross entropy. Multi-string unisons use their normalized soft
target. Training reuses the preregistered deterministic active-candidate
weights from temporal v1:

- ambiguous playable pitch: `1.25`;
- rare `(pitch, string)` pair: up to `1.5`;
- neighboring same-pitch cross-string reference within two seconds: `2.0`;
- final product capped at `4.0`.

The sequence model receives no reference strings as input. Reference strings
are used only in the loss.

| Parameter | Value |
|---|---:|
| Epochs | 6 |
| Window batch size | 16 |
| Learning rate | `0.0003` |
| Weight decay | `0.0001` |
| Optimizer | AdamW |
| Scheduler | cosine, minimum `0.000015` |
| Token size | 64 |
| GRU hidden size per direction | 64 |
| Dropout | 0.10 |
| Selection metric | unweighted validation oracle string accuracy |

No activity loss, candidate resampling, onset jitter, pitch augmentation,
none-logit bias or decoder continuity heuristic is enabled.

## Evaluation order and gates

1. The selected epoch must strictly exceed the activity-v1/v4 governed
   validation oracle string accuracy `0.8673266131`.
2. Only after that gate passes, evaluate the 60-track original-microphone
   source calibration oracle.
3. Source oracle string accuracy must strictly exceed `0.8765020821`
   (`7,367 / 8,405`).
4. Only after both oracle gates pass, run the hierarchical source activity
   thresholds `0.10` through `0.90` in steps of `0.10`.
5. The selected source point must pass all production comparisons:
   - TAB F1 `>= 0.704668`;
   - same-pitch wrong-string errors `<= 481`;
   - onset F1 `>= 0.815375`;
   - repeated TAB recall `>= 0.721959`.
6. The frozen mixture-Demucs gate may be read only if every source comparison
   passes.

Threshold ties are resolved by fewer same-pitch wrong-string errors, then
higher repeated TAB recall, then the higher activity threshold. A failed gate
ends the experiment without tuning the architecture, data composition or
threshold grid.

## Fixed outputs

- checkpoint:
  `checkpoints/guitarset-candidate-sequence-string-v1.pt`;
- training log:
  `reports/candidate-sequence-string-v1-training.jsonl`;
- source audit:
  `reports/candidate-sequence-string-v1-source-audit.json`;
- source oracle:
  `reports/candidate-sequence-string-v1-source-oracle.json`;
- result:
  `docs/candidate-sequence-string-v1-result.md`.
