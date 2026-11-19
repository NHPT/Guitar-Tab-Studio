# Candidate activity residual v1 preregistration

Status: implementation and training authorized. The frozen mixture-Demucs
gate remains locked.

## Hypothesis

Hierarchical decoder v1 failed because v2's inherited total activity
probability was not trained as an independent candidate-level classifier.
Candidate activity residual v1 adds a dedicated scalar activity residual while
preserving the validated v4 conditional string distribution.

The activity residual uses the frozen onset encoding, candidate pitch
embedding, pitch-local harmonic CQT evidence and Basic Pitch confidence. Its
final layer is zero-initialized. At initialization, activity probabilities are
exactly equal to v4.

## Frozen inputs

- Direct initialization:
  `guitarset-candidate-position-head-v4.pt`
- Direct initialization SHA-256:
  `a41980379cc6a1dcbc5b5217998749d25a75154ffab0569808b8b3f3bc5fd29c`
- Encoder root:
  `guitarset-onset-position-mixture-v12.pt`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Frozen modules: encoder, legacy position output, v2 candidate pitch
  embedding, v2 categorical candidate head and v4 spectral string residual
- Trainable module: zero-initialized scalar candidate activity residual only
- Trainable parameters: `149,761`
- Seed: `20260923`
- Independent review queues and test splits: locked

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 5 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Loss | binary candidate activity cross entropy |
| Selection metric | validation candidate acceptance F1 |
| Spectral harmonics | 5 |
| Spectral radius | 1 CQT bin |
| Spectral embedding | 128 |
| Activity hidden size | 128 |

No onset jitter, candidate augmentation, repeat oversampling, none-logit bias,
conditional string loss or decoder continuity heuristic is enabled.

## Source calibration

After training, the hierarchical decoder uses the dedicated activity
probability and the frozen v4 conditional string distribution. The following
activity thresholds are evaluated on the 60 original-microphone source tracks:

`0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`

The selected point is the highest source TAB F1. Ties are resolved by fewer
same-pitch wrong-string errors, then higher repeated TAB recall, then the
higher threshold.

## Promotion gates

The model may enter the frozen mixture-Demucs gate only if the selected source
point passes every gate:

1. TAB F1 `>= 0.704668`.
2. Same-pitch wrong-string errors `<= 481`.
3. Onset F1 `>= 0.815375`.
4. Repeated TAB recall `>= 0.721959`.

If any source gate fails, the model is rejected without reading the frozen
mixture-Demucs tracks.
