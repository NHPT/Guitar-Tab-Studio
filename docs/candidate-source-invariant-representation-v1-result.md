# Candidate source-invariant representation v1 result

Status: rejected at the training-only representation audit. No model
preregistration or training was authorized.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Frozen audit plan | `e894a42b77ff923929e121085c5771e04a608aa130c6a2da26771b0480e657c5` |
| Machine-readable audit | `ae4bca6f3fe761ae74623cdc5f52de1af9a446bafd7c7d4679dc0ebe9355983a` |
| Formal-v2 manifest | `630d08ede28ba7c220793715738444dab22ca53e3cde3c2311bbc6b028879916` |
| Frozen position checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |

The audit used only `split=train` records. Its deterministic 20% calibration
partition selected 268 annotations and reproduced the string-v2 selected-key
SHA-256
`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.
Governed GuitarSet player-04, AG-PT validation, and Guitar-TECHS validation
were not read.

## Tested representation

The raw control used the existing 14-frame, five-harmonic, radius-one CQT
magnitude, positive-flux, and signed-delta trajectories.

The candidate representation:

1. subtracted the two-frame pre-onset baseline from each magnitude trajectory;
2. retained positive flux and signed delta;
3. independently normalized each candidate channel by its maximum absolute
   value.

This tested whether candidate-local baseline and gain removal could suppress
recording-source identity while retaining string-position attack cues.

## Probe population

The audit observed 171,125 candidate queries and 90,862 active matched
queries. It excluded 188 multi-target unisons and 2,000 synthetic-source
examples. Deterministic per-source, pitch, string, and partition caps retained
25,212 real-source examples:

| Partition | GuitarSet | AG-PT-set | Guitar-TECHS |
|---|---:|---:|---:|
| Train | 5,443 | 8,319 | 725 |
| Calibration | 4,689 | 5,778 | 258 |

The source probe used the same 39 shared MIDI pitches for both
representations, with 676 train and 241 calibration examples per source.

## Result

| Metric | Raw | Candidate representation | Change |
|---|---:|---:|---:|
| Source balanced accuracy | 0.726141 | 0.721992 | -0.004149 |
| Source macro F1 | 0.725015 | 0.720557 | -0.004458 |
| Macro-source string accuracy | 0.508492 | 0.473756 | -0.034736 |
| Pooled string accuracy | 0.475338 | 0.443823 | -0.031515 |
| GuitarSet string accuracy | 0.524419 | 0.478354 | -0.046065 |
| AG-PT-set string accuracy | 0.431291 | 0.411907 | -0.019384 |
| Guitar-TECHS string accuracy | 0.569767 | 0.531008 | -0.038760 |

The representation reduced source balanced accuracy by only `0.004149`,
against a required reduction of at least `0.05`. Excess source accuracy above
one-third chance decreased by `1.06%`, against the required `20%`.

All three real sources also lost string-position accuracy. The mean
AG-PT-set/Guitar-TECHS change was `-0.029072`, so the transformation failed
the intended cross-source criterion directly rather than trading a small
GuitarSet loss for transfer gains.

All five frozen qualification checks failed.

## Convergence note

One fixed SGD probe reached the preregistered 2,000-iteration limit and
emitted a convergence warning. The audit was not rerun with changed probe
settings.

This prevents interpreting the final decimal places as fully optimized probe
estimates. It does not make the representation eligible: source separability
missed its absolute reduction threshold by `0.045851`, while every source had
a materially negative string-accuracy change.

## Decision

Reject candidate source-invariant representation v1. Candidate-local onset
baseline subtraction and scale normalization remove useful string timbre
without materially suppressing dataset identity. No checkpoint, training log,
governed report, or cross-source report was created.

The completed experiments now rule out another nearby threshold, loss-weight,
group-assignment, temporal-summary, or local-normalization retry on the same
features and supervision. The remaining evidence-supported path is to acquire
genuinely independent, licensed, real-audio string/fret supervision, with
priority on multi-performer phrase-level recordings and varied instruments
and recording chains.
