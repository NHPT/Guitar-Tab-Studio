# Sealed evaluation v1

Status: prior-data exclusion, expanded internet discovery, private blind
selection, controlled asset acquisition and machine-generated Layer B
lattices are implemented. Human reference review and the layer-aware evaluator
are not complete. No selected or acquired record counts as admitted until
item-level rights, content, identity, reference and isolation audits pass.

## Current commitments

The exclusion index was built from the 40 manifests listed in
`services/sealed_gate/exclusion-manifests.txt`.

| Artifact | Value |
|---|---|
| Exclusion index | `apps/api/data/training/sealed-evaluation-v1/exclusion-index.json` |
| Exclusion SHA-256 | `31b88dc66a4bc53c52dfbb4e797b85a3a9f108afcc5043f192f21a96b492561e` |
| Prior track identities | 1,876 |
| Prior group identities | 165 |
| Prior audio hashes | 1,733 |
| Internet candidate catalog | `apps/api/data/training/sealed-evaluation-v1/internet-candidates-v4.json` |
| Candidate catalog SHA-256 | `6953674c927b4fccea4047d128aa99e36e37274bb8819857fee728e1941b4186` |
| Discovered metadata records | 4,411 |
| Blind intake commitment | `apps/api/data/training/sealed-evaluation-v1/blind-intake-v3.commitment.json` |
| Blind intake commitment SHA-256 | `28e3d869532704132d14135ebd19d2671f51e51e2f6ec6f0c5b61146b021a243` |
| Acquisition commitment | `apps/api/data/training/sealed-evaluation-v1/blind-acquisition-v3.commitment.json` |
| Acquisition commitment SHA-256 | `6c4497b6e19822cb8721c7a5f6b5788efe91942406160b9c45d40a588f9d7875` |
| Position-lattice commitment | `apps/api/data/training/sealed-evaluation-v1/blind-position-lattice-v1.commitment.json` |
| Position-lattice commitment SHA-256 | `cb0e34d76f385d35e0a9328d086cfb0eca93c8f11dc138332fa27f136b0e75ef` |
| Layer A / B / C selected | 100 / 100 / 20 |
| Layer A / B assets acquired | 100 / 100, zero failures |
| Layer B scores / MIDI members / events | 100 / 106 / 51,414 |
| Human-reviewed Layer B scores | 0 |
| Admitted sealed recordings | 0 |

The exclusion index contains domain-separated SHA-256 identity digests, not
raw song, performer, recording or path values. A candidate is rejected if its
track, group, performer, work, recording, session, lineage or audio identity
collides with prior data.

The zero admission count is intentional. The 4,411 catalog records are leads,
not validated benchmark items. The expanded run returned 3,791 Internet
Archive records, including 1,000 collection entries and 2,791 resolved audio
files; 395 Mutopia guitar scores; and 225 Wikimedia Commons audio, video and
symbolic-score files. Openverse was unreachable from the current network; its
adapter remains covered by an offline API fixture.

## Three-layer benchmark

The benchmark no longer requires every one of 100 internet recordings to
reveal the performer's exact string choice. That requirement cannot be derived
reliably from ordinary audio or a public score.

### Layer A: real-recording domain gate

Target: at least 100 independent real guitar recordings representing at least
100 works.

Each quantitative item requires an independently aligned onset and pitch
reference. This layer measures onset F1, pitch F1, repeated-note recall and
domain robustness. A score, MIDI file or expert transcription may provide the
note reference after alignment and review. It does not establish the
performer's physical string or fret.

An internet recording without an auditable note reference may remain a
qualitative stress case, but it cannot contribute to a quantitative gate.

### Layer B: playable-position lattice

Target: at least 100 independently reviewed public scores paired to Layer A
works where possible.

For each score note, the reference stores the set of physically playable
string/fret assignments under the declared tuning, capo and fret range.
Evaluation may test pitch-correct playability against this set. It must not
describe one lattice member as the performer's observed fingering.

Mutopia provides the initial score-candidate pool. The current exact
title-and-creator matcher found no audio/score pairs, so pairing still requires
catalog expansion and human entity resolution.

### Layer C: physical-fingering gold

Target: 20 to 30 independent recordings with observed or creator-supplied
string/fret truth.

This smaller layer calibrates exact physical-fingering metrics, including TAB
F1 and repeated same-string/same-fret recall. Acceptable evidence includes a
performance-synchronous creator tablature, a rights-cleared capture with
performer confirmation, or sufficiently clear multi-angle video followed by
expert adjudication. Score-derived playable positions alone are not physical
gold.

## Rights tiers

Discovery and use are separate decisions. The catalog keeps restricted and
unclear records instead of deleting them.

| Tier | Examples | Automated policy |
|---|---|---|
| `open-production` | Public domain, CC0, CC BY, CC BY-SA | Metadata discovery; controlled acquisition with all item-level obligations |
| `restricted-research` | NC, ND, YouTube standard terms | Metadata only; no automated media download |
| `discovery-only` | Missing, unknown or conflicting rights | URL, metadata and future fingerprints only |

`restricted-research` is an operational quarantine, not a claim that every
listed license legally forbids every use. An item can move tiers only after its
actual rights evidence and intended operation are reviewed. YouTube records
enter through a local metadata importer; the tool does not scrape or download
YouTube media. BY-SA items remain production-capable, but their records retain
attribution, license-link, modification-marking and ShareAlike obligations.

## Discovery contract

`services/sealed_gate/catalog.py` normalizes:

- Wikimedia Commons API media metadata and per-file license evidence;
- Openverse audio API metadata;
- Internet Archive Advanced Search item metadata;
- Mutopia guitar-score catalog metadata;
- operator-supplied YouTube metadata without media acquisition.

Every record contains the provider, source URL, media type, creator and work
identity, declared license, rights evidence URL, tier, permitted operations,
download prohibition and score-pair state. Output is canonical sorted JSON.
All records begin with `audit_status=unreviewed` and
`qualified_for_sealed_evaluation=false`.

The expanded catalog contains 3,735 `open-production` candidates and 676
`restricted-research` candidates. It exposes zero asset or score URLs for
restricted records. It contains 2,911 open audio/video candidates of at least
30 seconds and 411 score or symbolic-score records. These counts describe
machine classification only; they do not prove solo-guitar content, aligned
references, exact license validity or benchmark eligibility.

## Blind selection

A data operator, separate from the model developer, performs:

1. a fixed provider/query snapshot and candidate-catalog commitment;
2. item-level rights and content audit without running candidate models;
3. work, recording and derivative-lineage resolution;
4. deterministic selection against preregistered coverage quotas;
5. lawful acquisition only for operations allowed by the audited tier;
6. independent onset/pitch alignment and score-lattice construction;
7. physical-fingering adjudication for the 20 to 30 gold items;
8. exclusion-index comparison and aggregate-only sealing.

The model developer must not select items from model failures or inspect the
chosen recordings, labels or per-item results. Public source metadata may be
known, but the selected subset and references remain sealed.

`services/sealed_gate/blind_set.py` assigns every catalog record to exactly one
private disposition. The initial selection used Internet Archive and
Wikimedia, but Wikimedia's asset host returned persistent HTTP 429 responses.
The recovery chain preserved every verified asset, recorded the provider
availability waiver and replaced only failed items from open overflow pools.
The final selection contains 100 Layer A Internet Archive records from 100
works, 100 lineages and 73 normalized creators; 100 Layer B Mutopia works from
22 normalized creators; and 20 separate Wikimedia videos queued as metadata
for Layer C adjudication.

## Current implementation boundary

`services/sealed_gate/seal.py` implements the exclusion index, asset hashing,
identity isolation and an earlier single-layer admission contract requiring
100 exact-fingering recordings. It remains fail-closed and must not be weakened
to make internet candidates pass.

`services/sealed_gate/acquire.py` downloads only selected open-production
assets, enforces byte limits and verifies hashes and decoded audio properties.
`services/sealed_gate/lattice.py` converts acquired MIDI into set-valued
playable positions under explicit tuning/capo assumptions. Acquisition is
complete at 100 Layer A audio assets and 100 Layer B score assets. The lattice
is complete at 100 scores, 106 MIDI members and 51,414 events, including 126
events with no standard-tuning position in the declared 0-24 fret range. Its
output remains machine-generated and unreviewed.

Before the corpus can be sealed, Layer A still needs independent onset/pitch
references, Layer B lattices need score and tuning review, and Layer C needs
observed physical-fingering adjudication. The current
`services/sealed_gate/intake-plan-v1.json` remains only a fallback commissioned
recording plan.

## Commands

Expand a frozen catalog with additional Internet Archive collections:

```bash
PYTHONPATH="services/worker:services/sealed_gate" .venv/bin/python \
  services/sealed_gate/catalog.py \
  --providers internet-archive \
  --seed-catalog \
  apps/api/data/training/sealed-evaluation-v1/internet-candidates-v3.json \
  --internet-archive-pages 10 \
  --internet-archive-expansion-limit 0 \
  --expand-seed-internet-archive \
  --output \
  apps/api/data/training/sealed-evaluation-v1/internet-candidates-v4.json \
  --report \
  apps/api/data/training/sealed-evaluation-v1/internet-candidates-v4-summary.json
```

Build or refresh the prior-data exclusion index before any new non-sealed
manifest is used:

```bash
PYTHONPATH="services/worker:services/sealed_gate" .venv/bin/python \
  services/sealed_gate/seal.py build-exclusion-index \
  --manifest-list services/sealed_gate/exclusion-manifests.txt \
  --output apps/api/data/training/sealed-evaluation-v1/exclusion-index.json
```

The catalog command downloads provider metadata and HTML only. It never
downloads discovered restricted, unknown or YouTube media.
