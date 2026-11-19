# Candidate position head v3 preregistration

Status: training authorized; frozen end-to-end gate remains locked until the
source-calibration gates below pass.

## Hypothesis

V2's shared onset-group encoding and one seven-class MLP do not provide enough
candidate-specific timbral evidence to distinguish the same pitch played on
different strings. Its categorical objective also couples candidate rejection
and string selection too tightly.

V3 keeps candidate identity before assignment but adds:

- onset-centered CQT trajectories around the candidate fundamental and first
  four overtones;
- one-bin context on either side of each harmonic;
- explicit candidate confidence;
- shared string embeddings and physical fret features;
- a factorized activity probability and conditional six-string probability.

The factorized loss is the sum of binary activity negative log-likelihood and
conditional string cross entropy on active references. This prevents inactive
examples from directly dominating string discrimination.

## Frozen inputs

- Encoder initialization:
  `guitarset-onset-position-mixture-v12.pt`
- Initialization SHA-256:
  `3e1e3507627b15c7ec1863022e6ae2ad8da5717a73cab7b6352d90e0d108c5dd`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Governed manifest SHA-256:
  `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Feature and candidate caches: existing governed v12 caches
- Frozen modules: spectral encoder, pitch projection, legacy output head and
  adapter
- Trainable modules: candidate pitch embedding and v3 factorized head only
- Trainable parameters: `222,458`
- Seed: `20260923`
- Independent review queues: excluded
- Test splits: locked

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 5 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Candidate activity loss weight | 1.0 |
| Conditional string loss weight | 1.0 |
| Selection metric | validation candidate string accuracy |
| Spectral harmonics | 5 |
| Spectral radius | 1 CQT bin |
| Spectral embedding | 128 |
| String embedding | 16 |
| Position hidden size | 128 |

No onset jitter, pitch candidate augmentation, repeat oversampling or
post-merge continuity heuristic is enabled.

## Calibration protocol

The first evaluation uses only
`governed-v1/guitarset-source-calibration.jsonl`: 60 original GuitarSet
microphone validation tracks. The frozen 60-track mixture-Demucs gate is not
used for model selection or none-bias calibration.

The source calibration repeats the v2 none-logit scan and then evaluates only
representative Pareto points end to end with the frozen v18 primary and v24
supplemental activity heads.

## Promotion gates

V3 may enter the frozen mixture-Demucs gate only if all conditions pass on the
source calibration:

1. Training validation candidate string accuracy must strictly exceed v2's
   `0.681596`.
2. Best source TAB F1 must be at least the same-domain v24 value `0.704668`.
3. Source same-pitch wrong-string errors must not exceed v24's `481`.
4. Source onset F1 must not decrease from v24's `0.815375`.
5. Source repeated TAB recall must not decrease from v24's `0.721959`.

If any source gate fails, v3 is rejected without reading the frozen
mixture-Demucs evaluation tracks. Passing the source gates only authorizes the
existing frozen gate; it does not promote the model or change the v24 research
bundle.
