# Candidate position head v4 preregistration

Status: training authorized; frozen mixture-Demucs gate remains locked.

## Motivation

V3 improved active-candidate coverage but reduced source-calibration oracle
string accuracy from v2's `0.867460` to `0.853302`. The factorized head
therefore did not improve string discrimination itself. V4 isolates the
string-position experiment from candidate activity:

- initialize the complete v2 categorical head and pitch embedding;
- freeze the v2 encoder, candidate head and activity behavior;
- add a zero-initialized six-string residual;
- condition the residual on candidate harmonic trajectories, confidence,
  string identity and physical fret;
- train only conditional string cross entropy on active references.

At initialization, v4 probabilities are numerically equal to v2. The residual
can only change the relative probabilities of the six string classes; it
cannot change total candidate activity probability.

## Frozen inputs

- Direct initialization:
  `guitarset-candidate-position-head-v2.pt`
- Direct initialization SHA-256:
  `37267c090204492070640f7aed10da6d90fa73b5654f2b321ddc8642928d5d0c`
- Encoder root:
  `guitarset-onset-position-mixture-v12.pt`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Frozen modules: v12 encoder, legacy position output, v2 candidate pitch
  embedding and v2 categorical candidate head
- Trainable module: zero-initialized spectral string residual only
- Trainable parameters: `152,289`
- Seed: `20260923`
- Independent review queues and test splits: locked

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 5 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Loss | conditional six-string cross entropy |
| Selection metric | validation oracle string accuracy |
| Spectral harmonics | 5 |
| Spectral radius | 1 CQT bin |
| Spectral embedding | 128 |
| String embedding | 16 |
| Position hidden size | 128 |

No candidate activity loss, none-logit loss, augmentation, oversampling or
decoder heuristic is introduced.

## Calibration and gates

The first evaluation uses only the 60 original microphone tracks in
`governed-v1/guitarset-source-calibration.jsonl`.

V4 is rejected without frozen-gate evaluation unless all conditions pass:

1. Governed validation oracle string accuracy must strictly improve over the
   v2-initialized epoch-zero baseline.
2. Source oracle string accuracy must strictly exceed v2's `0.867460`.
3. Best source TAB F1 must be at least v24's `0.704668`.
4. Source same-pitch wrong-string errors must not exceed v24's `481`.
5. Source onset F1 and repeated TAB recall must not decrease from v24's
   `0.815375` and `0.721959`.

None-logit bias may be scanned only on the source calibration after training.
It cannot be selected from the frozen mixture-Demucs gate.
