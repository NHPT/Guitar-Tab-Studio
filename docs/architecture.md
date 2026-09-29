# Architecture

## Current runtime shape

```text
Browser (React)
  |-- upload / link import
  |-- job polling
  |-- synchronized, auto-following TAB editor
  |-- Web Audio stem mixer
  `-- AnalyserNode and Canvas visualizers
          |
Node API (Express)
  |-- source policy and platform adapters
  |-- upload storage
  |-- asynchronous job state
  |-- capability probing
  `-- inference provider contract
          |
External worker (optional)
  |-- media normalization (FFmpeg)
  |-- public media acquisition (yt-dlp)
  |-- source separation (Demucs)
  |-- note transcription (Basic Pitch / specialized model)
  `-- guitar fingering and technique inference
```

The browser and API run independently from inference. The repository-local
`.venv/bin/guitar-tab-worker` is detected automatically; `GTS_WORKER_COMMAND`
can override it. Import jobs fail closed when the worker is unavailable.

Production Web traffic uses `guitar.hackall.cn`. Native Nginx serves the Web build
and proxies same-origin `/api` and `/media` requests to the application service;
the community UI remains at `/community`. A separate public API subdomain is
not introduced until an external API product requires its own authentication,
rate limits and compatibility policy.

## Target multi-client topology

The target architecture keeps one React workspace and one versioned project
format while supporting browser and desktop clients:

```text
                         Shared React workspace
                      /                         \
              Browser client               Electron renderer
                    |                             |
               HTTPS client                 Secure preload API
                    |                             |
             Cloud app core               Electron main process
                    |                             |
       Cloud InferenceProvider       Local InferenceProvider
                    |                             |
       Redis queue + GPU worker      Signed Python sidecar
                    |                + FFmpeg + model packages
                    \_____________________________/
                                  |
                       versioned analysis.json
                                  |
                    shared project and revision model
```

Electron is the first desktop framework because the project already uses
React and Node.js. The desktop renderer reuses the Web UI; it is not a fork.
The main process owns native file access, application storage, sidecar
lifecycle, model packages, updates and operating-system integration.

Cloud and local inference must implement the same provider contract:

- Submit a validated local file or authorized remote source.
- Stream versioned stage and progress events.
- Support cancellation, timeout and crash recovery.
- Return the same versioned `analysis.json` schema.
- Report model identity, hashes, runtime capabilities and warnings.
- Fail closed when a required worker or model is unavailable.

The initial desktop development build may call the existing local Python
environment. A distributable release must use a frozen, signed sidecar so end
users do not install Python or invoke arbitrary shell commands.

## Community ML control plane

Community correction and training use a separate control plane from inference
and project editing:

```text
Web / desktop correction client
             |
      Contribution API
             |
    consent + rights gate
             |
     immutable task store
             |
 blind assignment + consensus + expert adjudication
             |
 versioned dataset manifests + object storage
             |
 approved recipe scheduler -> isolated training runners
             |
       MLflow model registry
             |
 sealed evaluation -> shadow/canary -> production alias
```

A project revision, annotation submission, consensus label, gold label and
dataset release are different immutable records. The annotation service never
overwrites project inference output, and the training service cannot update a
production model alias.

PostgreSQL stores operational metadata, role grants, consent versions and audit
events. Object storage holds authorized audio, features and model artifacts.
DVC-compatible manifests bind dataset releases and pipeline inputs by content
hash. MLflow tracks experiment lineage and the `candidate`, `shadow`,
`champion` and `retired` model aliases.

Entry-level tasks ask only about audio quality, event presence, missing notes,
false positives and timing. Pitch, string/fret and technique tasks require
task-specific calibration. Independent responses are hidden from one another;
disagreements escalate instead of forcing a guess.

Community training initially accepts only approved recipes and bounded
parameters. Runners have no network access, read datasets through read-only
mounts, enforce compute quotas and cannot access sealed evaluation data.
Custom code and federated learning are deferred until the centralized workflow
has demonstrated reproducibility, abuse resistance and rights compliance.

External runners authenticate with dedicated machine credentials and claim one
queued experiment under a short lease. Heartbeats extend only the current
runner's lease; expired leases return to the queue, and terminal results require
the matching one-time lease token. The claim payload contains only an approved
recipe identity, bounded parameters, immutable data and lineage hashes, and the
operator-selected runtime image. It contains no shell command, host path,
community identity or sealed-evaluation material.

The repository runner maps each recipe version to a fixed argument vector.
It does not interpolate commands supplied by users. On macOS the development
runner executes without network access and can write only its work and cache
directories. Model files are uploaded through a lease-bound endpoint; the API
streams the file, computes its digest, stores it outside public media routes and
accepts a terminal result only when the reported digest matches that upload.
Per-account active and daily quotas plus bounded retry attempts constrain the
local queue. Production replaces the macOS sandbox with a rootless GPU runtime.

Promotion automation has a separate `Gate` machine identity. Candidate creation
records lineage integrity from immutable experiment evidence. Reproduction,
public validation and robustness checks can be written only by the gate
service; maintainers retain the human shadow/canary decisions. The public gate
uses frozen player-04 validation domains and cannot access the legacy player-05
test or the future sealed evaluation service. Sealed results require a distinct
`SealedGate` credential and endpoint; a public gate credential cannot assert a
sealed pass.

The complete state model, role matrix, quality thresholds and delivery phases
are defined in the
[community annotation and open training roadmap](community-ml-roadmap.md).

Inference jobs now resolve the active promotion before execution. A stable hash
of promotion and job identity selects shadow or canary traffic without mutable
request counters. Shadow jobs return only the baseline result, run the candidate
in a separate non-public directory, retain an aggregate observation and delete
the candidate output. Canary and champion jobs run the selected artifact first;
on failure they immediately rerun the current champion, the previous champion,
or the built-in baseline in that order. Mode-specific observations are
idempotent per job. After the configured minimum sample count, a canary that
exceeds its error budget is rejected and an unhealthy champion is retired while
its previous champion is restored. Deployment policy and recent observations
are visible only to model maintainers.

The current development implementation uses an atomic repository-local JSON
state file and append-only JSONL audit log behind the same API boundary. It
already supports account bootstrap, review-package import, calibration,
consensus, withdrawals, frozen manifests, approved experiment recipes, model
promotion states, authenticated runner leases and operational metrics. This is
a development adapter:
PostgreSQL, object storage, DVC remotes, isolated GPU runners and MLflow remain
production deployment components rather than prerequisites for local product
development.

## Desktop and offline storage

Desktop metadata, project revisions, task state and installed model state are
stored in SQLite. Audio, stems, caches and exported files remain in a
user-controlled application data directory. Database records store relative
artifact identifiers rather than machine-specific absolute paths.

Offline inference packages are separate from the desktop installer and contain
the platform worker, FFmpeg, model manifest and required weights. Every package
declares:

- Application and schema compatibility.
- Operating system and CPU architecture.
- Model versions, licenses, sizes and SHA-256 hashes.
- Minimum memory and optional accelerator requirements.
- Rollback target and cleanup policy.

The desktop application must remain usable for playback and editing when no
offline model package is installed. Cloud inference is optional and may be
used only after the user understands that audio will be uploaded. Local mode
does not upload audio, features, logs or corrections without separate consent.

## Media source policy

| Source | Adapter behavior |
| --- | --- |
| Local upload | Accept supported audio/video and retain it inside project storage. |
| Bilibili | Public, non-DRM media may be delegated to an installed worker. |
| Douyin | Public, non-DRM media may be delegated to an installed worker. |
| NetEase Cloud Music | Public, non-DRM media may be delegated to an installed worker. |
| QQ Music / Qishui Music | Online import is not advertised because authenticated media acquisition is not supported. |
| Other URL | Rejected by default; allow-list in a dedicated adapter. |

The application does not bypass authentication, signatures, access controls,
paywalls, regional restrictions, or DRM. A user must have the right to process
the submitted recording. Proprietary encrypted containers such as QMC, MFLAC,
MGG, NCM, KGM, and VPR are rejected before a job is created. MFLAC/MGG names
are accepted only when signature inspection proves that the content is already
a standard FLAC, OGG, MP3, WAV, APE, or MP4-family stream.

Link input accepts either a bare URL or app-generated share text. The first
HTTP(S) URL is normalized, matched against the platform host allow-list, and
reflected in the UI before submission.

## Pipeline stages

1. Validate source and media type.
2. Normalize to a stable sample rate and channel layout.
3. Separate vocals, guitar, bass, drums, piano, and residual accompaniment;
   retain measured stem energy so silent outputs can be hidden.
4. Detect tempo, beat grid, key, chords, notes, and note boundaries.
5. Map pitches to playable string/fret sequences using movement cost.
6. Suggest note techniques from pitch, duration, velocity, onset strength,
   spectral centroid, spectral flatness, zero-crossing rate, and same-string
   transitions; classify strum, arpeggio, rasgueado, and tremolo patterns from
   within-beat onset spread and string order.
7. Quantize conservatively and retain raw timing plus confidence.
8. Compress stems to client-playable M4A and remove intermediate WAV files.
9. Produce a persistent, editable project and synchronized stem manifest.

During playback, all audible stems pass through a shared `AnalyserNode`.
Canvas renderers consume live PCM, FFT, and RMS values without putting audio
frames into React state. Real media time is the playback clock for score
positioning; the app timer is only the fallback for the built-in demo.

## Technique confidence

Technique recognition is represented as a suggestion:

```json
{
  "technique": "hammer-on",
  "confidence": 0.88,
  "positionConfidence": 0.79,
  "positionSource": "playable-optimizer-v2",
  "techniqueConfidence": 0.91,
  "techniqueSource": "transition-heuristic-v2",
  "techniqueEvidence": [
    "same-string",
    "connected-notes",
    "weak-second-onset",
    "ascending-fret"
  ],
  "techniqueCandidates": [
    { "technique": "hammer-on", "confidence": 0.91 },
    { "technique": "pick", "confidence": 0.61 }
  ],
  "relatedNoteId": "worker-17"
}
```

The note confidence and technique confidence are stored separately. Relational
techniques point to their source note so the score renderer can draw a slur or
slide between the correct fret events. This prevents uncertain hammer-ons,
pull-offs, slides, harmonics, mutes, and strum directions from becoming
irreversible notation.

String/fret position confidence is also independent. Simultaneous pitched events
are assigned as a group so one chord cannot occupy the same string twice.
Natural harmonics are remapped from sounding pitch to touch nodes such as
`<7>` and `<12>`; artificial harmonics retain both the fretted position and the
touch position, for example `4<16>`.

## Cloud production components

- Object storage for uploaded media and generated stems.
- Redis-backed queue for CPU/GPU jobs.
- Dedicated Python 3.11 inference workers.
- PostgreSQL for users, projects, revisions, and quotas.
- Signed media URLs and automatic retention cleanup.
- GPU scheduling, model version tracking, and per-stage observability.
- Contribution API, immutable task snapshots and blind assignment.
- Consensus, expert adjudication, reputation and audit services.
- Versioned dataset manifests and isolated community training runners.
- Model registry plus physically separated hidden-evaluation workers.

## Desktop production components

- Electron main, preload and shared React renderer.
- SQLite project and task database.
- User-authorized media and artifact directories.
- Frozen Python inference sidecar and bundled FFmpeg.
- Separately downloaded, signed model packages.
- IPC allowlist, context isolation and renderer sandbox.
- Signed application updates, crash recovery and model rollback.

Implementation order, acceptance gates and platform scope are defined in
[the product and platform roadmap](product-roadmap.md).
