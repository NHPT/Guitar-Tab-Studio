# Sealed evaluation intake and fingering protocol

Status: candidate discovery, blind selection, controlled acquisition and the
machine-generated Layer B lattice are complete. Rights/content/reference
review and physical-fingering collection remain open. This protocol does not
authorize any catalog record or empty assignment slot to count as sealed data.

## Roles

The data operator and model developer remain separate roles.

The model developer may define aggregate coverage requirements and receive the
final commitment hash. The developer must not receive selected work titles,
performer mappings, media, references, per-work metrics, failure examples or
candidate predictions.

The data operator controls candidate snapshots, rights evidence, acquisition,
identity resolution, annotation, adjudication and the sealed service account.
Selection is completed without running candidate models.

## Layer A intake

Select at least 100 independent real guitar recordings representing at least
100 works. The operator must verify:

- source page, creator, work and recording identity;
- item-level license or written permission for the intended operation;
- at least 30 seconds of usable guitar audio;
- real rather than synthesized or score-rendered performance;
- no duplicate, excerpt, transcode, stem or alternate-channel counting;
- an independently aligned onset/pitch reference for quantitative use;
- no identity or audio overlap with the exclusion index.

Provider metadata is only a lead. Internet Archive album items require
track-level expansion, and Wikimedia category membership requires content
review. Restricted and discovery-only candidates remain cataloged but are not
downloaded by the discovery tool.

## Layer B score lattice

Select at least 100 reviewed public scores. Pairing to Layer A uses composer,
canonical work title, movement and arrangement evidence, not filename
similarity alone.

For each aligned note, generate all legal string/fret assignments under the
declared tuning, capo and 0-24 physical-fret range. Human review checks:

- score edition and arrangement compatibility;
- transposition, tuning and capo;
- repeated attacks, ornaments and omitted or added performance notes;
- onset alignment within the benchmark tolerance;
- that the lattice is stored as a set of possibilities, not observed
  performer fingering.

## Layer C physical gold

Collect 20 to 30 independent recordings with actual physical string/fret
evidence. New recordings are one path, not the only path.

Eligible evidence can be:

- creator-supplied performance-synchronous tablature;
- a rights-cleared performance confirmed by the performer;
- clear multi-angle or close-up video adjudicated by qualified reviewers;
- another source with explicit physical string/fret ground truth and rights
  for the intended benchmark operation.

One annotator creates the onset, offset, string and absolute physical-fret
reference. A second reviewer checks audio and evidence, including repeated
attacks and same-pitch cross-string choices. Capo remains separate metadata.
Disagreements are adjudicated before sealing.

## Optional new recording capture

When a Layer C item is newly recorded, record the guitar, tuning, capo,
pickup/microphone, signal path, sample rate, channel layout, session and take.
Use lossless WAV or FLAC at 8 kHz or above, mono or stereo.

Different channels, takes, stems, excerpts, effects, transcodes or amplifier
renders of one performance share one `lineage_id` and cannot count as
independent works. Public-domain composition status does not remove the need
for performer and recording permission.

The 100-slot `services/sealed_gate/intake-plan-v1.json` is retained as a
fallback if a fully commissioned exact-fingering corpus becomes feasible. It
is not the active minimum for Layer C.

## Rights record

Before any media acquisition, record:

- provider, source URL and immutable source identity;
- declared license and rights-evidence URL;
- audited rights tier and permitted operations;
- composition, performance and recording rights basis;
- retention, withdrawal and redistribution restrictions;
- content and derivative-lineage findings.

`NC`, `ND`, BY-SA, unknown-license and YouTube records remain discoverable.
They are not silently discarded or silently upgraded. BY-SA may enter
controlled acquisition with attribution and ShareAlike obligations; the other
tiers require their own operation-specific review or permission.

## Admission and seal

The operator:

1. freezes the candidate catalog and exclusion index;
2. completes rights, content, identity and score-pair audits;
3. acquires only authorized media;
4. validates actual asset hashes and decoded media properties;
5. completes Layer A onset/pitch references;
6. completes Layer B set-valued position lattices;
7. completes Layer C exact physical-fingering adjudication;
8. rejects all prior-data collisions;
9. generates separate aggregate commitments for all three layers;
10. mounts manifests, media, caches and baseline reports read-only for the
    isolated gate.

The current `seal.py` validates only the older single-layer exact-fingering
manifest. A layer-aware schema and evaluator are required before this protocol
can produce a final seal. Until then, catalog and audit progress must be
reported separately from admitted counts.

The current isolated intake contains 100 verified Layer A audio assets, 100
verified Layer B score assets and a complete machine-generated lattice with
51,414 note events. Human-reviewed Layer B scores and admitted sealed
recordings both remain zero.

## Completion criteria

The benchmark is complete only when all three conditions hold at once:

- Layer A has at least 100 admitted real recordings with quantitative
  onset/pitch references;
- Layer B has at least 100 reviewed score lattices and records every resolved
  audio/score pair;
- Layer C has 20 to 30 admitted exact physical-fingering recordings.

A partial collection stays in isolated intake storage and cannot release
candidate-specific or per-work metrics.
