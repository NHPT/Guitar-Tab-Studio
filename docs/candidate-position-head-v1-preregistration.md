# Candidate position head v1 preregistration

Status: rejected by the frozen 60-track gate. See
`candidate-position-head-v1-result.md`.

## Hypothesis

The current onset-position model predicts six independent fret-state
distributions. In direct decoding this can emit a fret whose pitch differs
from the frontend candidate, while equivalent positions for the same pitch
are not compared directly.

Candidate position head v1 adds a pitch-conditioned six-string classifier on
top of the frozen v12 onset encoder. The decoder derives the fret from the
frontend pitch and selected string, so every emitted position is
pitch-consistent by construction.

## Frozen inputs

- Position initialization:
  `guitarset-onset-position-mixture-v12.pt`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Governed manifest SHA-256:
  `7646fcb2e321b48e85c8e4b4f0c2ed278cc05f0b7267cc97c4daa2e673443e2d`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Feature and candidate caches: the existing v12 caches
- Frozen modules: spectral encoder, pitch projection, legacy output head and
  adapter
- Trainable modules: candidate pitch embedding and candidate string head only
- Seed: `20260923`
- Independent review queues: excluded from training, threshold selection and
  model selection
- Test splits: locked

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Candidate loss | multi-label BCE |
| Candidate loss weight | 1.0 |
| Selection metric | validation candidate position F1 |

Duplicate pitches inside one onset group use a multi-label string target.
Inference then selects the highest-scoring globally unique string assignment
for all candidates in the group. Frontend pitches without a matching
reference event receive an all-zero string target so the head can abstain at
the frozen probability threshold. Reference pitches missing from the frontend
are excluded because they cannot be queried at inference time.

## Frozen decoder

- Assignment mode: `candidate`
- Primary position probability: `0.50`
- Primary activity probability: `0.20`
- Supplemental position probability: `0.70`
- Supplemental activity probability: `0.85`
- Primary and supplemental Basic Pitch frontends remain v24-identical.
- No trajectory continuity or pitch-identity merge heuristic is enabled.

## Promotion gates

Candidate position head v1 is rejected if any condition fails:

1. GuitarSet mixture TAB F1 must not decrease from `0.67988641`.
2. GuitarSet same-pitch wrong-string errors must strictly decrease from `495`.
3. GuitarSet onset F1 and repeated TAB recall must not decrease.
4. Frozen cross-domain Guitar-TECHS and AG-PT evaluations must not regress in
   onset F1 or TAB F1.
5. The production gates remain `TAB F1 >= 0.75` and repeated-note recall
   `>= 0.85`; passing relative gates alone does not promote the model.

This experiment does not alter the Basic Pitch production default or the v24
research bundle.
