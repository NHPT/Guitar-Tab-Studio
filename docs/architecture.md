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

`packages/core` is the environment-neutral application boundary. It owns the
project and job domain types, media signature policy, Worker analysis protocol,
schema migration and the `InferenceProvider`, `ProjectRepository` and
`MediaRepository` interfaces. The React and Express workspaces import these
types instead of maintaining parallel copies. The current API supplies a
Pipeline inference adapter and a JSON project repository adapter. The React
workspace accepts an injected `StudioApplication` and uses the HTTP adapter by
default. The core also provides in-memory project storage; Electron supplies the
same contract through an allowlisted preload bridge, SQLite and a validated
opaque-ID media repository.

Worker output is parsed as untrusted structured data before project mapping.
The parser accepts historical analysis schema versions 1 through 9, normalizes
them to v9 and rejects unknown future versions or invalid note/string fields.
Persisted projects carry `schemaVersion: 1`; unversioned local projects are
migrated on read and rewritten with the current version on their next save.

Production Web traffic uses the domain configured by the deployment environment.
Native Nginx serves the Web build and proxies same-origin `/api` and `/media`
requests to the application service; the community UI remains at `/community`.
A separate public API subdomain is not introduced until an external API product
requires its own authentication, rate limits and compatibility policy.

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

Transcription engines are selected through a second, narrower provider
registry. Basic Pitch is the bundled commercial-compatible production
baseline. Stereo spatial separation produces guitar-part candidates, not an
instrument count. Candidates are exposed as independent `guitarParts[]` only
after they pass separation-confidence, event-independence and distinct
lead/rhythm-profile gates. Failed candidates are discarded and the aggregate
guitar stem is transcribed again as one part. The top-level `tab` remains an
alias of the primary part for legacy clients.

Research providers are hidden unless `GTS_ENABLE_RESEARCH_MODELS=1` or an
explicit provider allowlist is configured. The MuScriptor research strategy
runs MuScriptor as the primary event model and Basic Pitch as a
corroborating/residual detector, then fuses same-pitch events within a bounded
onset tolerance. MuScriptor Small, Medium and Large weights are selected
automatically from a versioned device profile or fixed by the user.
The guitar-specialist strategy runs an installed StringTrace onset-position
research bundle with the playability decoder. The multitrack-research strategy
runs YourMT3+, Basic Pitch and Noise2Fret.

MuScriptor browser login uses Authorization Code + PKCE with an ephemeral
loopback callback. The public OAuth client id is configured through the desktop
runtime panel or `GTS_HF_OAUTH_CLIENT_ID`. Tokens are encrypted with Electron
`safeStorage`, remain memory-only when operating-system encryption is
unavailable, and are passed to the Worker through its environment rather than
command-line arguments. Manual `.safetensors` import remains available.
Enabling any strategy is transactional: required weights are checked,
downloaded and verified before the preference changes. The API normalizes the
selection into Worker CLI arguments, and the Worker emits the selected or
fallback model identity in `analysis.json`.

Noise2Fret and MT3-family models are separate adapter roles, not aliases for
the primary event provider. The Noise2Fret adapter runs after event fusion and
may change a string/fret position only when the predicted position has the same
MIDI pitch. Its official CC BY 4.0 checkpoint is downloaded and SHA-256
verified during high-accuracy preparation.

Medium and Large tiers also run the MIT-licensed MR-MT3 checkpoint as an
independent event verifier before Noise2Fret positioning. Small omits MR-MT3
to stay within its memory and latency budget. The noncommercial sidecar also
contains the isolated YourMT3+ adapter and downloads its pinned checkpoint only
when the multitrack-research strategy is enabled. Original Magenta MT3 and the
unlicensed MT3-PyTorch port remain excluded.

Distribution policy is enforced before runtime configuration is built. A
commercial allowlist removes noncommercial providers from the renderer and
desktop IPC, while the frozen Worker build independently omits their Python
modules. Provider-specific code therefore does not affect project storage,
analysis schemas, cloud transport or the trained StringTrace position-model
gate.

Cloud inference is configured at runtime with an HTTPS API root. The desktop
validates `/health` before persisting the endpoint and switching the project;
an invalid or unavailable endpoint never disables the local provider.

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
are visible only to model maintainers. Automatic rejection and rollback can
emit a minimal HMAC-signed HTTPS webhook; alert delivery cannot block the local
rollback path.

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

Project revisions are immutable snapshots with an expected-parent check.
Concurrent saves fail with an explicit conflict instead of overwriting another
client's current project. Restoring an old snapshot creates a new `restore`
revision, preserving both branches. The original transcription and the
pitch-preserving easy arrangement are separate score layers inside the same
versioned project snapshot.

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
| Local upload | Accept standard audio/video directly; route recognized proprietary containers through the authorized-media flow after explicit consent. |
| Bilibili | Media exposed to the configured extractor is delegated to the worker. |
| Douyin | Media exposed to the configured extractor is delegated to the worker. |
| NetEase Cloud Music | Use an exposed standard stream when available; locally decode a consented NCM container after acquisition. |
| QQ Music | Use an available standard stream when possible; locally decode a consented QMC-family container after acquisition. |
| Qishui Music | Not currently supported by the installed extractor. |
| Other URL | Rejected by default; allow-list in a dedicated adapter. |

The acquisition layer reports technical failure categories separately:
authentication or bot verification, network or regional restrictions,
unsupported URL shape, unavailable formats, and DRM. A user-provided Netscape
cookie file and proxy can be configured for authenticated non-DRM streams.
Proprietary containers such as QMC, MFLAC, MGG, NCM, KGM, and VPR require
explicit authorized-processing consent when their signatures are not already a
supported standard audio format.

Link input accepts either a bare URL or app-generated share text. The first
HTTP(S) URL is normalized, matched against the platform host allow-list, and
reflected in the UI before submission.

## Local authorized media decoder

The client submits only versioned consent. The API records the provider, fixed
processing purpose `analysis-and-score-generation`, ephemeral-output policy and
policy version on the analysis source. No decoded-media endpoint exists.

After confirmation, proprietary media is decoded inside the API Node process.
For Electron local mode this process runs on the user's computer; for browser
clients it runs on the API server:

```text
protected upload or locally acquired link
                |
       inspect container footer
                |
 embedded key or macOS/Windows desktop authorization
                |
       1 MiB bounded block reader
                |
    local Rust/WASM or Node crypto decoder
                |
       0600 ephemeral standard media
                |
       signature check -> normal Worker
                |
       unconditional finally cleanup
```

`@clamber_l/crypto 0.1.12` supplies the zero-dependency Rust/WASM primitives
for QMC1, QMC2, KWM and KGM/VPR under MIT / Apache-2.0. NCM header parsing,
AES-128-ECB key extraction and the NCM stream cipher are implemented in the API
with Node's local crypto runtime. The inline WASM build is bundled into the API
artifact, so Electron packaging does not need a separate decoder executable or
runtime download.

The decoder reads and writes bounded blocks instead of holding the whole song
in memory. It rejects malformed or oversized headers, limits decoded output to
2 GB, checks cancellation between blocks and creates output with mode `0600`.
QMC2 first uses an ekey embedded in the local footer. Newer MusicEx footers
contain an official encrypted filename and song MID instead of the ekey. The
song MID is the UTF-16LE field at footer offset `0x0c`; the encrypted resource
filename starts at `0x48` and can contain a different resource MID. The
`GetEVkey` request must send the footer song MID in `songmid` and the complete
resource filename in `filename`. Deriving both values from the filename causes
official authorization failure `101404` for files where those IDs differ.
Only the macOS and Windows Electron-local APIs may handle that variant. After
the user confirms the rights notice, macOS reads the current QQ Music login
credential from the application's sandbox preference plist.
`AutoLoginUserInfo` is a nested binary `NSKeyedArchiver`, so the API converts
both plist layers with the system `plutil`, resolves `$objects` / `CF$UID`
references and extracts only `nUserId`, `strAuthst` and `loginType`. Windows
checks the current-user QQ Music registry key and performs a bounded scan of
small login/configuration files under the current user's Roaming and Local
QQ Music roots. JSON, INI, XML, Cookie-style text and UTF-16 text are accepted;
media/cache directories, symbolic links, oversized files and unrelated user
directories are not scanned. The API then calls QQ Music's
`music.vkey.GetEVkey` endpoint using the matching macOS or Windows desktop
request profile. The returned per-file result is treated as the authorization
decision before local decryption continues.

The credential and ekey remain in request memory and are never logged,
persisted, returned to the renderer or exposed by an API route. The official
request has a fixed HTTPS destination, bounded response, ten-second timeout and
no redirects. Browser and Linux clients cannot read a local QQ Music login. No
third-party decoder or key lookup service is used. A MusicEx footer without a
usable filename is rejected rather than guessed.

macOS may deny cross-container reads until the user grants the desktop
application Full Disk Access. Permission denial, missing login state and
unsupported archive structure are reported separately; the desktop renderer
can open the exact Privacy settings page without receiving any credential
material.

KWM v2 and KGM v5 use client-side separated keys rather than the QQ Music
authorization endpoint. The Electron runtime exposes explicit native import
actions for user-selected key databases:

- Kuwo Android and iOS MMKV entries are normalized to
  `resourceId-qualityId` and matched against the parsed KWM header.
- Kugou Android MMKV and desktop encrypted SQLite entries are normalized to a
  lowercase 32-character audio hash and matched against the KGM v5 header.

MMKV data is parsed directly from the bounded upload buffer. A Kugou desktop
database is decrypted into a randomly named `0600` temporary SQLite file under
the desktop data root, queried read-only, closed and removed in `finally`.
Only validated ekey mappings remain in process memory; importing a replacement
database clears the previous map, and stopping the desktop API clears all keys.
The route is disabled outside `GTS_DESKTOP_LOCAL=1`. Browser clients cannot
import these databases. Extracting mobile MMKV files is intentionally outside
the application: Android normally requires root and iOS normally requires a
jailbroken device because of their application sandboxes.

Authorized temporary media is excluded from asynchronous shadow inference so no
background model can retain a file handle after the user-facing task settles.

For a protected link, the existing Worker first runs its configured extractor
in acquire-only mode and writes the result under the job's temporary directory.
That network step only obtains the user-requested source. If the downloaded file
is a proprietary container, local decryption follows; a standard audio stream
bypasses decryption. Both the acquired source directory and decoded file are
removed after success, failure or cancellation.

If a protected upload is sent without consent, the API deletes the uploaded
copy and returns HTTP `409` with code
`authorized_media_consent_required`. Files with proprietary extensions but a
valid MP3, FLAC, APE, OGG, WAV or MP4-family signature bypass local decryption
and follow the standard media pipeline.

## Pipeline stages

1. Validate source, media type and any required authorized-processing consent.
2. If required, acquire the source, resolve an authorized MusicEx ekey or match
   an imported KWM v2/KGM v5 separated key on a supported desktop runtime,
   decode locally and verify the ephemeral standard-media signature.
3. Normalize to a stable sample rate and channel layout.
4. Separate vocals, guitar, bass, drums, piano, and residual accompaniment;
   retain measured stem energy so silent outputs can be hidden.
5. Detect tempo, beat grid, key, chords, notes, and note boundaries.
6. Map pitches to playable string/fret sequences using movement cost.
7. Suggest note techniques from pitch, duration, velocity, onset strength,
   spectral centroid, spectral flatness, zero-crossing rate, and same-string
   transitions; classify strum, arpeggio, rasgueado, and tremolo patterns from
   within-beat onset spread and string order.
8. Quantize conservatively and retain raw timing plus confidence.
9. Compress stems to client-playable M4A and remove intermediate WAV files.
10. Produce a persistent, editable project and synchronized stem manifest.

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

The release build freezes the Python Worker with PyInstaller onedir and writes a
per-file sidecar SHA-256 inventory. Demucs and transcription weights remain
separate model packages. Model downloads are HTTPS-only, origin-confined,
resumable and bounded by manifest sizes before activation.

Implementation order, acceptance gates and platform scope are defined in
[the product and platform roadmap](product-roadmap.md).
