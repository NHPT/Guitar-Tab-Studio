# Candidate position head v2 preregistration

Status: rejected by the frozen 60-track gate. See
`candidate-position-head-v2-result.md`.

## Hypothesis

Candidate position head v1 failed because six independent binary outputs did
not represent rejection and string selection as one decision. V2 replaces
them with one seven-state categorical distribution for each frontend
candidate: `none`, then strings 1 through 6.

The decoder masks unplayable pitch/string states before softmax. It selects a
globally unique string assignment within each onset group while allowing any
candidate to resolve to `none`.

## Frozen inputs

- Position initialization:
  `guitarset-onset-position-mixture-v12.pt`
- Initialization SHA-256:
  `3e1e3507627b15c7ec1863022e6ae2ad8da5717a73cab7b6352d90e0d108c5dd`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Governed manifest SHA-256:
  `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Feature and candidate caches: the existing governed v12 caches
- Frozen modules: spectral encoder, pitch projection, legacy output head and
  adapter
- Trainable modules: candidate pitch embedding and categorical candidate head
  only
- Seed: `20260923`
- Independent review queues 09 and 10: excluded from training, threshold
  selection and model selection
- Test splits: locked

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Candidate loss | seven-class cross entropy |
| Candidate loss weight | 1.0 |
| Selection metric | validation categorical candidate accuracy |

An unmatched frontend pitch receives the `none` target. A candidate matching
one reference string receives that string target. If one pitch and onset map
to multiple reference strings, the valid string classes share the target
probability equally; this preserves one candidate identity without forcing a
false single-string label.

Training reports categorical accuracy, accepted-candidate precision/recall/F1,
rejected-candidate precision/recall/F1 and active-candidate string accuracy
separately.

## Frozen decoder

- Assignment mode: `candidate`
- Categorical position probability floor: `0.0`
- Rejection: categorical `none` state, not an independent threshold
- Primary activity probability: `0.20`
- Supplemental activity probability: `0.85`
- Primary frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Supplemental frontend: Basic Pitch `0.25 / 0.20 / 70 ms`
- Primary and supplemental activity gating occurs before position assignment.
- Accepted frontend candidates are merged by pitch and onset within `50 ms`.
- The merged candidate identity is classified and assigned exactly once.
- No trajectory continuity or post-merge pitch heuristic is enabled.

## Promotion gates

Candidate position head v2 is rejected if any condition fails:

1. GuitarSet mixture TAB F1 must not decrease from `0.67988641`.
2. GuitarSet same-pitch wrong-string errors must strictly decrease from `495`.
3. GuitarSet onset F1 must not decrease from `0.80454367`.
4. GuitarSet repeated TAB recall must not decrease from `0.69027448`.
5. Only if all four frozen 60-track gates pass may Guitar-TECHS and AG-PT
   cross-domain evaluation run.
6. The production gates remain `TAB F1 >= 0.75` and repeated-note recall
   `>= 0.85`; passing relative gates alone does not promote the model.

The experiment does not alter the Basic Pitch production default or the v24
research bundle.
