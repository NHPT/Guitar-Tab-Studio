# Candidate temporal string v1 preregistration

Status: implementation and training authorized. The frozen mixture-Demucs
gate remains locked.

## Evidence and hypothesis

V4 improved local string discrimination, but its source oracle string
accuracy of `0.876502` still left `710` end-to-end same-pitch wrong-string
errors. A candidate-level diagnostic using the same governed Basic Pitch
frontend found:

- all `1,035` single-target source oracle errors occurred on pitches with at
  least two playable string positions;
- source accuracy was only `0.781046` on string 6 and `0.697222` at frets
  9-12;
- the dominant confusions were between adjacent playable strings, including
  `3 -> 2` (`208`), `6 -> 5` (`134`), `5 -> 4` (`113`) and `2 -> 3`
  (`106`);
- same-pitch events that changed string were rare and materially harder than
  same-string repeats;
- v4 single-target oracle accuracy was `0.967028` on governed training but
  only `0.866772` on governed validation.

The experiment tests whether reference-free performance trajectory features,
combined with explicit training emphasis on rare same-pitch string changes,
improve conditional string selection without changing candidate activity.

The diagnostic reports are:

- `candidate-string-context-v1-governed-train.json`
- `candidate-string-context-v1-governed-validation.json`
- `candidate-string-context-v1-source-calibration.json`

## Frozen inputs

- Direct initialization:
  `guitarset-candidate-activity-residual-v1.pt`
- Direct initialization SHA-256:
  `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e`
- Conditional string root:
  `guitarset-candidate-position-head-v4.pt`
- V4 SHA-256:
  `a41980379cc6a1dcbc5b5217998749d25a75154ffab0569808b8b3f3bc5fd29c`
- Encoder root:
  `guitarset-onset-position-mixture-v12.pt`
- Training and model-selection manifest:
  `governed-v1/guitarset-mixture-combined.jsonl`
- Training frontend: Basic Pitch `0.35 / 0.25 / 70 ms`
- Seed: `20260923`
- Independent review queues and test splits: locked

The encoder, legacy position output, candidate pitch embedding, categorical
candidate head, v4 spectral string residual and activity-v1 residual are
frozen. Only a new zero-initialized temporal string residual is trainable.
At initialization, both candidate activity and conditional string
probabilities must be numerically equal to activity-v1.

## Temporal features

Features are computed from the complete ordered Basic Pitch candidate groups
for one track. They do not use reference strings. Each candidate pitch and
proposed playable string receives 20 features:

1. previous candidate-group gap, clipped to 2 seconds;
2. next candidate-group gap, clipped to 2 seconds;
3. previous same-pitch gap, clipped to 2 seconds;
4. next same-pitch gap, clipped to 2 seconds;
5. previous same-pitch occurrence within 0.75 seconds;
6. next same-pitch occurrence within 0.75 seconds;
7. signed closest-pitch delta to the previous group, clipped to 24 semitones;
8. signed closest-pitch delta to the next group, clipped to 24 semitones;
9. previous-group mean candidate confidence;
10. next-group mean candidate confidence;
11. previous same-pitch candidate confidence;
12. next same-pitch candidate confidence;
13. previous-group candidate count, clipped to six;
14. next-group candidate count, clipped to six;
15. current candidate duration, clipped to 2 seconds;
16. current candidate confidence;
17. minimum normalized string movement to the closest-pitch candidate in the
    previous group;
18. corresponding normalized fret movement;
19. minimum normalized string movement to the closest-pitch candidate in the
    next group;
20. corresponding normalized fret movement.

Missing temporal neighbors use maximum normalized gaps, zero confidence and
count, and maximum movement. Previous and next context are both allowed
because transcription already processes a complete recorded track offline.

The temporal residual also receives the frozen v4 conditional string logit,
current physical fret and open-string indicator. It may change only the
relative six-string distribution.

## Same-pitch cross-string supervision

Conditional string cross entropy uses deterministic active-candidate weights.
Counts are derived only from matched active candidates in the governed
training split.

For one reference event:

`weight = min(4.0, ambiguity * rarity * cross_string)`

- `ambiguity = 1.25` when the pitch has at least two playable positions,
  otherwise `1.0`;
- `rarity = clip(sqrt(max_playable_pair_count / pair_count), 1.0, 1.5)`,
  where a pair is `(pitch, true_string)`;
- `cross_string = 2.0` when a previous or next matched same-pitch reference
  event within 2 seconds uses a different string, otherwise `1.0`.

For a multi-string unison target, the query weight is the mean event weight.
Validation remains unweighted. No sealed data, review queue or test split is
used to construct weights.

## Training recipe

| Parameter | Value |
|---|---:|
| Epochs | 5 |
| Batch size | 256 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Loss | weighted conditional six-string cross entropy |
| Selection metric | unweighted validation oracle string accuracy |
| Temporal features | 20 |
| Temporal hidden size | 64 |
| Temporal maximum interval | 2.0 s |
| Same-pitch repeat interval | 0.75 s |
| Maximum supervision weight | 4.0 |

No onset jitter, pitch augmentation, group sampler, activity loss, none-logit
bias or decoder continuity heuristic is enabled.

## Evaluation protocol and gates

1. The selected checkpoint must strictly improve governed validation oracle
   string accuracy over activity-v1/v4.
2. Source oracle string accuracy on the 60 original microphone tracks must
   strictly exceed v4's `0.876502`.
3. Only after both oracle gates pass, run the preregistered hierarchical
   source threshold scan:
   `0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`.
4. The selected source point must pass every production comparison:
   - TAB F1 `>= 0.704668`;
   - same-pitch wrong-string errors `<= 481`;
   - onset F1 `>= 0.815375`;
   - repeated TAB recall `>= 0.721959`.

Threshold ties are resolved by fewer same-pitch wrong-string errors, then
higher repeated TAB recall, then the higher activity threshold. The frozen
mixture-Demucs gate may be read only if all source gates pass.
