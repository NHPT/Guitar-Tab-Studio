# Sealed evaluation source review

Status: internet discovery is active under tiered rights controls. Public
sources can supply real-recording and score candidates; no reviewed source
alone supplies 100 exact performer-fingering references.

## Decision model

The source review now asks which benchmark layer an item can support instead
of rejecting a source unless it satisfies every requirement.

| Evidence | Layer A: real audio | Layer B: position lattice | Layer C: physical gold |
|---|---:|---:|---:|
| Real guitar recording | Required | Pairing target | Required |
| Aligned onset/pitch reference | Required for metrics | Required for pairing | Required |
| Public score or symbolic notes | Optional | Required | Insufficient alone |
| Observed performer string/fret | Not required | Not claimed | Required |
| Item-level rights audit | Required | Required | Required |
| Prior-data isolation | Required | Required | Required |

An academic paper license is not accepted as the dataset license. Search
summaries and provider-level policy are not substitutes for the item page and
its actual rights evidence.

## Rights policy

Candidates are never removed merely because they are `NC`, `ND`, BY-SA,
YouTube-hosted or unclear.

- `open-production`: public domain, CC0, CC BY and CC BY-SA candidates.
  Item-level obligations remain mandatory; BY-SA requires attribution and
  ShareAlike treatment for distributed adaptations.
- `restricted-research`: NC, ND, YouTube standard terms and other known
  limitations. Automated discovery stores metadata only.
- `discovery-only`: missing or conflicting rights. Store metadata, URL and
  future fingerprints only.

This is a project operation policy, not a universal interpretation of the
license. Separate written permission can upgrade a candidate. Failed or absent
permission leaves the metadata in the catalog and the media untouched.

## Expanded candidate inventory

The canonical inventory is
`apps/api/data/training/sealed-evaluation-v1/internet-candidates-v4.json`,
SHA-256
`6953674c927b4fccea4047d128aa99e36e37274bb8819857fee728e1941b4186`.

| Provider | Records | Open production | Restricted research | Notes |
|---|---:|---:|---:|---|
| Wikimedia Commons | 225 | 223 | 2 | 162 audio, 47 video and 16 symbolic-score records |
| Internet Archive | 3,791 | 3,117 | 674 | 1,000 collections expanded to 2,791 canonical audio files where permitted |
| Mutopia | 395 | 395 | 0 | Full guitar-score listing; public-domain, CC BY and CC BY-SA |
| Openverse | 0 | 0 | 0 | Adapter complete; endpoint unreachable during this run |
| Total | 4,411 | 3,735 | 676 | Zero records admitted without reference review |

The inventory includes 2,953 audio files, 1,000 collection records, 47
audiovisual records, 395 scores and 16 symbolic scores. It provides 2,911 open
audio/video candidates of at least 30 seconds. Exact title-and-creator matching
still produced no resolved audio/score pair.

The provider counts are not a quality claim. The next audit must reject
non-solo material, speech-heavy or mixed programs, duplicate works, derived
uploads, unusable recordings and items lacking an aligned reference.

## Source findings

### Wikimedia Commons

The Commons category supplies direct file pages with machine-readable license
metadata, duration and media URLs. Public-domain, CC BY and CC BY-SA items can
enter controlled acquisition while preserving their item-level obligations.

Use: Layer A and Layer C discovery. Download only an audited
`open-production` item. Category membership alone does not prove solo guitar,
work identity or reference availability. During the current acquisition,
Wikimedia's asset host persistently returned HTTP 429 with a 600-second
`Retry-After`; the process stopped requesting that host and replaced failed
Layer A/B selections from open Internet Archive and Mutopia overflow pools.
The 20 Layer C records remain metadata-only.

### Openverse

Openverse can broaden discovery across upstream providers and exposes license,
creator, duration and landing-page fields. It is an index, so its license data
must be verified on the upstream landing page.

Use: metadata discovery and deduplication. The current network could not reach
the API; the adapter is tested from a captured local fixture and should be
retried in a later snapshot.

### Internet Archive

The Advanced Search API exposes a large guitar-tagged audio pool. Search
results are often collection or album items rather than single performances,
and duration is not consistently available in the search response.

Use: Layer A discovery. The v4 catalog covers the first 1,000 stable search
results, expands every reachable open collection in that snapshot, chooses one
canonical encoding per track, and records duration, size, format and parent
lineage. Blind selection still permits at most one track per parent collection.

### Mutopia

Mutopia exposes guitar scores with stable piece IDs, composer/title metadata,
license links and available LilyPond, MIDI and PDF formats. The score identifies
notated pitches but not an internet performer's actual string choices.

Use: Layer B score pool and audio-pairing source. Public-domain, CC BY and
CC BY-SA entries can progress after item review. BY-SA obligations remain
attached to every selected and acquired record.

### YouTube

YouTube remains in scope for discovery. Standard-license, unknown-license and
ordinary page metadata are retained as restricted or discovery-only records.
The tool supports operator-supplied metadata and deliberately provides no media
downloader.

Use: find possible performances, creator tablatures and visible-fingering
videos. Move beyond metadata only with creator permission or separately
verified rights covering the intended operation.

### GOAT

GOAT contains 172 real DI items with tablature and physical string/fret
information. Its dataset record is `CC-BY-NC-4.0` and explicitly disallows
commercial use and internal commercial testing.

Use: restricted research comparison only. It cannot provide production-gate
evidence without separate permission.

Sources:

- Dataset terms: <https://zenodo.org/records/15690894>
- Paper and corpus description: <https://arxiv.org/html/2509.22655>
- Project repository: <https://github.com/JackJamesLoth/GOAT-Dataset>

### GAPS

GAPS provides real classical-guitar performances with aligned symbolic music,
but its access terms are non-commercial and its public references do not
provide observed physical string/fret truth.

Use: restricted Layer A research and possible Layer B methodology comparison,
not production physical gold.

Sources:

- Dataset site and access terms: <https://aim-qmul.github.io/GAPS/>
- Dataset record: <https://zenodo.org/records/13962272>

### Existing registered sources

GuitarSet, Guitar-TECHS, AG-PT-set, IDMT-SMT-GUITAR and all local reference,
review, residual, derived-stem and synthetic collections have already
influenced development. Their identities are committed to the exclusion index.

Use: legacy and development evaluation only. Renaming, resampling, stemming,
transposition or selecting an unread split does not restore blindness.

## Required next audit

The operator must now:

1. complete source-page review for the 100 selected Layer A and 100 selected
   Layer B records;
2. classify content scope, work identity, recording identity and derivatives;
3. retry Openverse and import YouTube metadata without downloading media;
4. resolve score pairs using composer/work identifiers plus human review;
5. build independent onset/pitch references for Layer A;
6. review the complete machine-generated 100-score tuning/capo position
   lattice for Layer B;
7. build 20 to 30 Layer C physical-fingering items from creator evidence,
   visible performance evidence or new contributions;
8. run exclusion and layer-aware seal checks before exposing aggregate metrics.
