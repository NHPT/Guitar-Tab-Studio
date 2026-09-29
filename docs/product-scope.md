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

- QQ Music and Qishui Music online import is not offered because authenticated
  media acquisition is not supported.
- Proprietary QMC, MFLAC, MGG, NCM, KGM, and VPR files must first be legally
  exported to a standard audio format.
- Public-link adapters only process media that the user is authorized to use.
- Fine guitar-technique labels combine note transitions with acoustic features,
  remain experimental suggestions, and require confirmation.
- Import jobs never fall back to demo audio or demo notation.
- The current repository is not yet a signed desktop application and does not
  contain a distributable offline Worker.
- The current local JSON state store is not the target database for cloud or
  desktop production releases.
- Commercial-song uploads are not redistributed to community reviewers merely
  because a user volunteers a correction; rights verification is mandatory.
- Community users cannot read hidden evaluation labels, run arbitrary server
  code or promote their own model to production.

## Next milestone

The immediate milestone remains transcription accuracy and data automation.
Shared application-core extraction and the Electron development shell follow
without forking the current Web UI. Consumer desktop and offline inference
releases require both the model gates and the platform gates in the
[product and platform roadmap](product-roadmap.md).

Long-term annotation, review and training participation is governed by the
[community annotation and open training roadmap](community-ml-roadmap.md).

Detailed model work remains in the
[high-accuracy guitar technique recognition plan](technique-recognition-roadmap.md).
Technique labels remain experimental until its dataset, model, calibration and
acceptance gates are complete.
