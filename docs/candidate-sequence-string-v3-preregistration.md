# Candidate sequence string v3 preregistration

Status: frozen before implementation and training. One implementation may be
validated against this contract, but no model training starts until those
regression checks pass. This remains a research experiment and cannot replace
the production model without every gate below.

## Hypothesis

Candidate sequence string v1 improved GuitarSet string decisions but degraded
the available non-GuitarSet training sources. String v2 then showed that a
whole-group structured likelihood was a worse optimization surrogate than the
original candidate loss.

Guitar-TECHS P3 adds the missing kind of supervision: licensed real
phrase-level performances with 250 dual-modality same-pitch cross-string
transitions within two seconds. String v3 tests one isolated change relative
to string v1: replay GuitarSet and add P3 while retaining the original
per-candidate set-valued cross-entropy.

P3 has one performer. Its direct-input and mic+amp tracks are correlated views
of the same 12 performances, not independent examples. P3 is training-only and
cannot be used for epoch selection or model promotion.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| `governed-v1/formal-training-manifest-v3.jsonl` | `cb7cc83e0a887f3413a58fc0bbd8e4258db17a94936d8472f6cb3952afc6504d` |
| formal-v3 composition report | `6d3b2731e1699df887eabf947b309e665cd196ba93955dfb395a4a76320e4cb0` |
| formal-v3 source audit | `ed171d832719ddfac766aabe3e45161e0961f0d0eb819812a052dcdebebad770` |
| formal-v3 cross-source isolation audit | `59dbc6a2d5f60ff88dc52d4980389f2be78e13128fe54067e5cc84ab103e2d17` |
| P3 position manifest | `f991cc2475ae65d558df9bfc4603b4a22ea3b1c10bbc9c6745b2704765cb22c8` |
| P3 candidate audit | `266bef7f7879198595b934ce38ec52320eb16c61f70dae61929ebb50adcaff39` |
| P3 candidate-cache report | `f167413ca2a25c6d233fddf90979bc459d3b3c14f5e9d31acedd71c0d84860eb` |
| P3 candidate-cache set | `c592ed2656c29fc4e0c4546a683b7c500c5b8f229fb431ea20eae70f2931b91a` |
| P3 sequence-coverage audit | `6c12bd0fdeb5776bed997a4d3efd8ba90bdf825d78b9127f25d393ec2fe5a728` |
| P3 frozen string-v1 audit | `48d6afde3ac03e8c7b83809dbfac19e07fb67776143039b563281b38ec01f732` |
| position/activity checkpoint | `7d255e33f22fc0f5424db7c4c90a1ea29bfb689bf30649944c6ea690c2bdc38e` |
| string-v1 checkpoint | `392d53c0f0c20cadb3e48ab048195728e22d43166fd078f64103c7cb20667982` |
| governed string-v1 baseline | `7f21f70f70da1f63baa9a448bdad34f113d920277e434ac61452efa8f6e633cc` |
| AG-PT string-v1 baseline | `ec5b83495fef9aab6e019f28f2cebfa2bdcd0e6611997e6eb5b2640fb1c37096` |
| Guitar-TECHS P2 string-v1 baseline | `d8bef975b37501ec394304737f8e233064e3ec62027b01fd5d2e2c3a6492ca2d` |

The source ledger remains version `2026-11-06.2`, SHA-256
`de5b817d884422a15bb918b094caafbef856f4b612fb67cbbd9328ef90d2610a`.
Formal-v3 contains 1,117 tracks, 131,502 reference events, 997 train tracks,
120 validation tracks, and 144 groups. Its source audit passes, it contains no
test records or split leakage, and it has zero audio-hash, group, performer,
or track overlap with the locked AG-PT and Guitar-TECHS P2 validation
manifests.

The candidate frontend remains Basic Pitch onset `0.35`, frame `0.25`, and
minimum note length `70 ms`. Activity probability remains `0.50`, conditional
string margin remains `0`, and full-group assignment remains unchanged. No
activity, joint, confidence, supplemental-candidate, onset-fusion, threshold,
margin, or continuity change is enabled.

## Training-only partition

The deterministic formal-v2 calibration partition is reproduced before P3 is
added. P3 is explicitly excluded from partition construction, so the selected
268 annotations and selected-key SHA-256 remain:

`c45b0af54438e183b5e39bd97e79185a74be2170d35ca1f91e47ef490feeaf0f`.

The partition uses fraction `20%` and seed `20261122`. GuitarSet selects 24 of
120 tracks from each training performer 00-03 by ascending SHA-256 rank.
AG-PT-set, Guitar-TECHS P1, and synthetic records select complete groups by the
existing formal-v2 rule. These 268 records are used only for epoch selection.

Gradient data contains only:

| Dataset | Tracks | Candidate groups | Windows |
|---|---:|---:|---:|
| GuitarSet, excluding calibration | 384 | 55,946 | 3,288 |
| Guitar-TECHS P3 | 24 | 3,416 | 201 |

AG-PT-set, Guitar-TECHS P1, and synthetic records receive no gradient. This
keeps the experiment attributable to one data change relative to string v1
instead of repeating the failed mixed-source or structured-loss experiments.

P3 receives sampling weight

`(0.20 / 0.80) * (3288 / 201) = 4.08955223880597`.

GuitarSet weight is `1.0`. These weights make P3 exactly 20% of the replacement
sampler probability mass; they do not claim that correlated P3 modalities are
independent. One epoch draws 3,489 windows with replacement.

## Frozen model and objective

The complete 93,190-parameter sequence-string module initializes from
string v1 and remains the only trainable component. The position encoder,
candidate pitch embedding, v4 local string residual, activity-v1 residual,
candidate features, sequence architecture, dropout, masks, and overlapping
inference are unchanged.

Training reuses the string-v1 per-candidate set-valued cross-entropy:

1. only candidates matched to an active reference pitch contribute;
2. a multi-string unison uses its normalized soft target;
3. ambiguous playable pitch weight is `1.25`;
4. rare `(pitch, string)` weight is at most `1.5`;
5. a neighboring same-pitch cross-string reference within two seconds has
   weight `2.0`;
6. the final supervision-weight product is capped at `4.0`.

The string-v2 whole-group structured likelihood, group weights, confidence
rejection, margin rejection, invented labels for inactive candidates, and
source-specific losses are prohibited.

## Fixed optimization

| Parameter | Value |
|---|---:|
| Initialization | complete string-v1 checkpoint |
| Trainable scope | all 93,190 sequence-string parameters |
| Epochs | 3 |
| Window batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `0.00005` |
| Weight decay | `0.0001` |
| Scheduler | cosine, minimum `0.0000025` |
| Gradient norm cap | `5.0` |
| Seed | `20261122` |
| Sequence window / stride | `32 / 16` |
| Fine-tuning device | CPU |

There is one run and no post-result change to the partition, sample weights,
objective, learning rate, epoch count, architecture, or decoder.

## Training-only selection gate

Epoch zero is the frozen string-v1 checkpoint replayed on the unchanged
formal-v2 calibration partition. The frozen epoch-zero values are:

| Metric | Baseline |
|---|---:|
| Aggregate assigned-string accuracy | `0.8150606043986179` |
| Aggregate wrong strings | `3372` |
| Repeated-string true positives | `7552` |
| GuitarSet assigned-string accuracy | `0.963257429563875` |
| AG-PT assigned-string accuracy | `0.4627375613922699` |
| Guitar-TECHS P1 assigned-string accuracy | `0.4409090909090909` |
| AG-PT/P1 macro accuracy | `0.45182332615068044` |

An epoch is eligible only if all conditions hold:

1. aggregate assigned-string accuracy strictly increases;
2. aggregate wrong strings strictly decrease;
3. repeated-string true positives do not decrease;
4. GuitarSet assigned-string accuracy does not decrease;
5. AG-PT/P1 unweighted macro accuracy does not decrease;
6. neither AG-PT nor P1 assigned-string accuracy decreases by more than
   `0.005` absolute.

Synthetic calibration records remain in the aggregate comparison but do not
define a real-source preservation gate. Among eligible epochs, select higher
aggregate accuracy, then more repeated true positives, then higher AG-PT/P1
macro accuracy, then the earlier epoch. If no epoch is eligible, stop without
reading governed or cross-source reports.

## Evaluation order and gates

All evaluations use activity threshold `0.50`, string margin `0`, and the
unchanged full-group assignment.

| Gate | TAB F1 must exceed | Wrong strings must be below | Repeated TAB recall must be at least | Onset F1 baseline |
|---|---:|---:|---:|---:|
| Governed GuitarSet player-04 | `0.6647789837973886` | `1672` | `0.6197051846831583` | `0.8098159509202454` |
| AG-PT held-out performer | `0.3707169395899271` | `1812` | `0.3527027027027027` | `0.8795926792348975` |
| Guitar-TECHS P2 | `0.2826455624646693` | `374` | `0` | `0.7100056529112493` |

The order is fixed:

1. run the single training job and freeze the selected checkpoint hash;
2. evaluate governed GuitarSet once;
3. only after governed success, evaluate AG-PT once;
4. only after AG-PT success, evaluate Guitar-TECHS P2 once.

At each gate, TAB F1 must strictly improve, same-pitch wrong-string errors must
strictly decrease, repeated TAB recall must not decrease, and onset F1 must
remain unchanged within `1e-12`. P2 is the key independent phrase-level
transfer gate because it is a different Guitar-TECHS performer and has zero
identity overlap with P3.

There is no P3 promotion evaluation, threshold scan, margin scan,
source-specific checkpoint, source-specific decoder setting, epoch rescue, or
second training run. Passing all experiment gates permits a production-gate
assessment only. Production still requires TAB F1 at least `0.75` and
repeated-note recall at least `0.85`.

## Fixed outputs

- input audit:
  `reports/candidate-sequence-string-v3-input-audit.json`;
- checkpoint:
  `checkpoints/guitarset-candidate-sequence-string-v3.pt`;
- training log:
  `reports/candidate-sequence-string-v3-training.jsonl`;
- governed report:
  `reports/candidate-sequence-string-v3-governed-candidate.json`;
- AG-PT report:
  `reports/candidate-sequence-string-v3-agpt-candidate.json`;
- Guitar-TECHS P2 report:
  `reports/candidate-sequence-string-v3-guitar-techs-p2-candidate.json`;
- result:
  `docs/candidate-sequence-string-v3-result.md`.
