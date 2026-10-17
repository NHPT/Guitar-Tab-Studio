# Product Scope

## Product

Guitar Tab Studio currently runs as a browser workspace for turning authorized
audio into an editable string-instrument score and a stem-based practice
session. The target product uses the same workspace on Web, Windows and macOS,
with cloud inference and optional local offline inference. Guitar TAB is the
first fully supported notation mode; bass notation remains an explicit future
mode.

## Visual thesis

A quiet studio console: neutral surfaces, waveform-led hierarchy, compact
controls, and ten selectable color systems with vermilion as the default.

## Content plan

- Import rail: local upload, supported link sources, recent jobs.
- Main workspace: transport, waveform, complete score, selection state.
- Mixer: one stable strip per stem with mute, solo, and level controls.
- Inspector: confidence and editable guitar technique metadata.

## Platform plan

- Web remains the zero-install client and future collaboration surface.
- Windows and macOS use an Electron shell around the shared React workspace.
- The first desktop development build reuses the existing Node application
  service and Python Worker.
- Consumer desktop releases use a frozen, signed Worker sidecar; users do not
  install Python.
- Offline models are separately downloadable, versioned and hash-verified.
- Desktop projects remain editable without a network connection or installed
  inference model.
- Cloud upload and training-data contribution are separate, explicit choices.
- All registered users may complete entry-level community review tasks; pitch,
  string/fret and technique tasks unlock through task-specific calibration.
- Community experiments use approved recipes in isolated runners. Hidden
  evaluation and production promotion remain controlled platform operations.

## Interaction thesis

- Import state moves from source validation to analysis without changing pages.
- Playback position links waveform, measure, and TAB note highlighting.
- The compact score automatically follows the active measure during playback.
- Mixer changes are immediate and preserve layout while tracks mute or solo.
- The correction inspector opens only after selecting a chord or individual note.
- Community tasks present one short audio question at a time and always allow
  the user to skip an uncertain answer.

## MVP acceptance

- A browser user can open the demo project and practice without an account.
- A user can submit an audio/video file or a supported public link.
- The API exposes capability state instead of pretending unavailable models work.
- Every separated stem can be muted, soloed, and volume-adjusted independently.
- Only stems with detected signal energy are shown in the mixer.
- Playback supports seeking, speed control, and an opt-in measure loop that
  defaults to the complete score and allows selectable start and end measures.
- The complete score shows chord names, fingering diagrams, pick/strum rhythm,
  continuous tablature, rhythm beams, technique marks, sections, and optional lyrics.
- Capo recommendations prefer easier open-chord shapes and remain optional and editable.
- TAB notes expose string, fret, duration, technique, and confidence.
- Generated results remain editable because automatic transcription is imperfect.
- Personal corrections remain private unless the user separately consents and
  the audio rights permit community review and training.
- Blind consensus may enter a versioned research-training release after owner
  review. Independent expert adjudication is a stronger evidence tier, not a
  prerequisite for continuing product or model development.

## Deliberate limitations

- Qishui Music online import is not offered because a stable acquisition
  adapter is not available.
- Keyless MusicEx QMC2 files require the macOS desktop runtime, a current
  QQ Music login and a successful per-file authorization response from the
  official `music.vkey.GetEVkey` endpoint. macOS Full Disk Access may be
  required to read the nested client plist. Browser, Windows and Linux clients
  do not extract that local credential.
- The application does not use a third-party decoder or key lookup service,
  and it does not retain QQ Music credentials or returned ekeys.
- Key-separated KWM v2 and KGM v5 require a desktop user to explicitly import
  a matching Kuwo MMKV or Kugou MMKV/encrypted SQLite key database. Parsed
  mappings remain only in local process memory; mobile database extraction is
  not automated and remains subject to the operating-system sandbox.
- Proprietary QMC, MFLAC, MGG, NCM, KWM, KGM and VPR files require explicit
  rights-and-processing confirmation before the ephemeral API decoder runs.
- Public-link adapters only process media that the user is authorized to use.
- Fine guitar-technique labels combine note transitions with acoustic features,
  remain experimental suggestions, and require confirmation.
- Import jobs never fall back to demo audio or demo notation.
- The repository builds unsigned desktop candidates and defines the frozen
  offline Worker pipeline; official releases still require platform signing,
  notarization and generated sidecar artifacts.
- Desktop metadata uses SQLite. The API's local JSON adapter remains a
  development store and is not the target cloud production database.
- Commercial-song uploads are not redistributed to community reviewers merely
  because a user volunteers a correction; rights verification is mandatory.
- Community users cannot read hidden evaluation labels, run arbitrary server
  code or promote their own model to production.

## Next milestone

The immediate milestone remains transcription accuracy and data automation.
The shared application core, Electron development shell and offline packaging
pipeline are implemented without forking the Web UI. Consumer desktop and
offline inference releases still require both the model gates and the platform gates in the
[product and platform roadmap](product-roadmap.md).

Long-term annotation, review and training participation is governed by the
[community annotation and open training roadmap](community-ml-roadmap.md).

Detailed model work remains in the
[high-accuracy guitar technique recognition plan](technique-recognition-roadmap.md).
Technique labels remain experimental until its dataset, model, calibration and
acceptance gates are complete.
