# Candidate source-invariant representation v1 audit plan

Status: frozen before execution. This is a training-only representation
audit, not a model-training authorization.

## Motivation

Candidate position head v4 showed that local harmonic CQT trajectories contain
string-position signal, but its GuitarSet gains did not pass source
end-to-end gates. Candidate temporal string v1, sequence string v1, and
structured string v2 did not repair the cross-source deficit.

The current v4 input contains absolute candidate-local CQT magnitude,
positive flux, and signed delta values. It can encode the recording chain,
instrument, room, and dataset together with string timbre. This audit tests a
single deterministic transformation intended to remove candidate-level
baseline and scale while preserving the onset attack pattern.

## Frozen inputs

- manifest: `governed-v1/formal-training-manifest-v2.jsonl`;
- manifest SHA-256:
  `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916`;
- position checkpoint: `guitarset-candidate-activity-residual-v1.pt`;
- checkpoint SHA-256:
  `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e`;
- split: `train` only;
- candidate frontend: Basic Pitch onset `0.35`, frame `0.25`, minimum note
  length `70 ms`;
- positive-only groups: AG-PT-set and Guitar-TECHS;
- partition fraction: `20%`;
- partition seed: `20261122`;
- GuitarSet partition: track-stratified within players 00-03;
- other datasets: complete `group_id` partition.

The partition and selected-key digest must match string-v2. Governed
player-04 and cross-source validation manifests remain unread.

## Representations

The raw control is the existing three-channel local harmonic trajectory:

1. normalized CQT magnitude;
2. positive spectral flux;
3. signed frame delta.

It uses 14 frames, five harmonics, and radius one around each harmonic.

The candidate representation changes only those values:

1. subtract the mean of the two pre-onset magnitude frames independently for
   every harmonic bin;
2. retain positive flux and signed delta;
3. divide each of the three channels by its candidate-local maximum absolute
   value, with a fixed minimum scale of `0.001`.

Pitch, string, dataset, reference, and checkpoint outputs are not inputs to
the transformation.

## Sampling

Only matched, single-target, real-source candidates are eligible. Synthetic
records and multi-string unison targets are excluded.

Each `(partition, dataset, pitch, string)` bucket retains the 64 candidates
with the smallest SHA-256 rank of
`20261122:{dataset}:{group_id}:{track_id}:{onset_ms}:{pitch}:{string}`.
The same retained identities are used for raw and transformed probes.

## Fixed probes

Both probes use a standard scaler and averaged SGD logistic classifier:

- L2 penalty;
- alpha `0.0001`;
- maximum `2,000` iterations;
- tolerance `0.00001`;
- seed `20261122`.

The source probe uses only MIDI pitches present in every real source in both
partitions. Every `(source, pitch)` cell is capped at 128 and downsampled to
the smallest source count. It reports balanced accuracy and macro F1 over
GuitarSet, AG-PT-set, and Guitar-TECHS.

The string probe appends a 49-class MIDI pitch one-hot vector, balances
training weight by source, and masks physically unplayable strings before
prediction. It reports pooled, macro-source, and per-source string accuracy.

## Qualification gate

The transformed representation qualifies for a model experiment only if all
conditions pass:

1. source balanced accuracy decreases by at least `0.05`;
2. excess source accuracy above one-third chance decreases by at least `20%`;
3. macro-source string accuracy decreases by no more than `0.005`;
4. no real source loses more than `0.01` string accuracy;
5. mean AG-PT-set and Guitar-TECHS string accuracy strictly improves.

Failure stops the architecture path without training. Passing authorizes a
separate preregistration; it does not authorize selecting probe settings or
promoting a checkpoint.

## Fixed output

`reports/candidate-source-invariant-representation-v1-audit.json`
