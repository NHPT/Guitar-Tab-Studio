# Candidate sequence gradient conflict v1 audit plan

Status: frozen before implementation and before any gradient extraction. This
is a training-only diagnostic. It performs no optimizer step, writes no
checkpoint, and does not read governed GuitarSet, AG-PT held-out, or
Guitar-TECHS P2 result reports.

## Questions

The audit tests three bounded questions at the frozen string-v1 checkpoint:

1. Do GuitarSet and Guitar-TECHS P3 optimize in opposite directions?
2. If they do, is the conflict concentrated in the shared token/sequence trunk
   or in the residual classifier head?
3. Are P3 direct-input and mic+amp views internally aligned strongly enough to
   treat their equal mean as one performance-level gradient unit?

This audit does not estimate generalization, select a model, tune a loss, or
authorize another training run.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| `governed-v1/guitar-techs-p3-music-position-train-candidate-v1.jsonl` | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| v3 input audit | `7dbb822c25e408acd2cfc72cbd3ee712b1b5f5bfbd0ff9e91b66bbab668a7bd4` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| source ledger | `de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a` |

The candidate frontend remains Basic Pitch onset `0.35`, frame `0.25`, and
minimum note length `70 ms`. The selected GuitarSet cache set contains 48
audio files, 14,691 candidates, 139 dropped rows, and has set SHA-256
`38c09ffe13d65f38c97fc7ad0ab0523664ba8ddb6907f51e2829022d0f8d07fc`.
The P3 cache set contains 24 audio files, 6,685 candidates, 157 dropped rows,
and has set SHA-256
`c592ed2656c29fc4e0c4546a683b7c500c5b8f229fb431ea20eae70f2931b91a`.

## Frozen units and panels

The v3 training partition is reproduced with fraction `0.20` and seed
`20261122`. All calibration tracks are excluded. No AG-PT, Guitar-TECHS P1,
synthetic, validation, or test track is eligible.

GuitarSet selection uses the 384 non-calibration training tracks. Within each
player group 00-03, tracks are ranked by the ascending pair
`(SHA256("20261123:<track_id>"), track_id)`. The first 12 tracks per player are
selected, producing 48 independent track units. The sorted selected-track
identity SHA-256 is
`74f07d7a2cade3fc9b448122d690568e0865a4eef0e44ec153a10cd25e4d6fed`.

Four replication panels use rank positions 1-3, 4-6, 7-9, and 10-12 from
every player. Each panel therefore contains 12 tracks, three per player:

| Panel | Track identity SHA-256 |
|---|---|
| 1 | `5c89de1d264ead52438cff322a3ab92e1f46291565b91045e728d1465c490ce7` |
| 2 | `8665ec9cf55cad5864b2044638866538899a7b7393980c0dd837ab7f9d769fae` |
| 3 | `5fbd835584832190c2c6e35153d713df715eed95c4f0a884965688606e51b1c4` |
| 4 | `a6b3473a6a0fbd293dd1eac4b7e865763201caef2fe99d930c87e1082e7bf793` |

P3 uses all 24 training tracks as 12 independent performance units. Tracks
are paired by the frozen `source_label` values `01` through `12`; every pair
must contain exactly one `directinput` and one `micamp` view. Direct and mic
track gradients are retained separately, then their equal arithmetic mean is
the performance-level P3 gradient. Candidate counts do not change this
modality weight.

## Frozen gradient definition

The complete 93,190-parameter candidate sequence string model is loaded from
the string-v1 checkpoint. The position/activity model is loaded only to
precompute the unchanged candidate sequence groups. Both models run in
evaluation mode, so dropout is disabled.

Every sequence window for a selected track is used once with sequence length
`32` and stride `16`. The objective is the original v3 per-candidate
set-valued cross-entropy with ambiguity weight `1.25`, rare-pair weight capped
at `1.5`, neighboring cross-string weight `2.0` within two seconds, and final
supervision weight capped at `4.0`.

For each track, every batch mean loss is multiplied by its number of active
candidate-window exposures before backward accumulation. The final parameter
gradient is divided by the track's total active candidate-window exposures.
Thus each track gradient is the candidate-count-normalized gradient over all
of its windows. Overlapping windows contribute exactly as they would to the
frozen window objective.

There is no optimizer, gradient clipping, parameter update, stochastic
sampling, source weighting, or gradient surgery. The implementation must
compare every parameter tensor before and after extraction and fail if any
value changes.

## Frozen module partition

The parameter partition is disjoint and exhaustive:

| Module | Parameter prefix |
|---|---|
| token encoder | `token_encoder.` |
| sequence encoder | `sequence_encoder.` |
| residual hidden | `residual_head.0.` |
| residual output | `residual_head.3.` |
| all | concatenation of all four modules |

Every trainable parameter must map to exactly one leaf module. A missing,
duplicated, or unexpected parameter fails the audit.

## Comparisons

For every module, the primary point estimate is the cosine between:

- the equal mean of 48 GuitarSet track gradients; and
- the equal mean of 12 paired P3 performance gradients.

The audit also reports:

- four GuitarSet panel-mean versus pooled P3-mean cosines;
- source mean-gradient norms and the P3/GuitarSet norm ratio;
- the pooled direct-mean versus mic-mean cosine and norm ratio;
- all 12 within-performance direct-versus-mic cosines.

Source aggregation is always equal by independent unit. No track, window, or
candidate count weights the source means after each unit gradient has been
normalized.

## Cluster bootstrap and decisions

For each module, unit-gradient Gram matrices are computed once in float64.
Using seed `20261123`, 10,000 bootstrap replicates independently resample 48
GuitarSet tracks and 12 P3 performance pairs with replacement. Each replicate
recomputes the cosine between bootstrap source means from the Gram matrices.
The reported interval is the percentile 95% interval.

The P3 direct-versus-mic bootstrap resamples the 12 performance labels with
replacement and applies the same resample counts to both modalities, preserving
the pairing.

Classification is mechanical:

- `conflict`: the complete 95% interval is below zero;
- `aligned`: the complete 95% interval is above zero;
- `inconclusive`: the interval includes zero.

Panel estimates and within-pair modality estimates are replication diagnostics,
not additional hypothesis tests. Module-level results are diagnostic and are
not corrected or promoted as independent confirmatory discoveries.

## Fixed outputs

- machine-readable report:
  `reports/candidate-sequence-gradient-conflict-v1.json`;
- interpreted result:
  `docs/candidate-sequence-gradient-conflict-v1-result.md`.

The report must include all frozen input identities, selected unit and panel
identities, candidate/window exposure counts, module parameter counts, point
estimates, bootstrap intervals, classifications, and a zero-parameter-mutation
assertion. It must not contain raw gradients.
