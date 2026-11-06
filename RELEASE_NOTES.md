# Guitar Tab Studio v0.1.0-preview.17

This preview completes the current beat-review correction workflow, tightens
the governance of model-training data and experiments, and reduces the time
before the packaged desktop window becomes usable. Basic Pitch remains the
production transcription model.

## Highlights

- Full-song beat review can add missed beats, remove false detections, correct
  tempo multipliers by phase, preview bulk changes, and undo corrections
  without rewriting submitted audit records.
- Pickup length supports quarter-beat steps with sixteenth-note labels and
  visual subdivision guides anchored to the first downbeat.
- Effective beats are recalculated from raw candidates plus persisted
  corrections, so BPM, phase, save, submit, and downstream segment inheritance
  use the same reviewed structure.
- Clip review is now a precision sample instead of a blocker for score
  generation after full-song structure has been confirmed.
- Training inputs are bound to a license ledger, source hashes, split
  identities, and checkpoint metadata. Synthetic generation and disagreement
  queues support targeted data expansion without placing generated material in
  sealed evaluation sets.
- The governed v29 activity-head experiment was rejected: it did not improve
  independent Guitar-TECHS or AG-PT results and slightly reduced GuitarSet TAB
  F1. It is retained only as an auditable research checkpoint.
- The default Basic Pitch desktop path no longer waits for a synchronous
  hardware probe before starting the local runtime. A clean packaged macOS
  smoke reached its first complete window in 9.012 seconds.

## Model Status

- Production default remains `basic-pitch / auto / built-in`.
- v24 remains a research bundle; v29 is not promoted and no equivalent v30
  activity-head training is planned.
- The production gates remain TAB F1 at least 0.75 and repeated-note recall at
  least 0.85. Generated notation still requires human review.

## Cross-Platform Builds

- macOS installer and ZIP archive for Apple Silicon.
- Windows NSIS installer and ZIP archive for x64.
- Update metadata and blockmaps are generated for both supported platforms.
- A source TAR.GZ, release notes, and SHA-256 manifest are published with the
  binary assets.

## Preview Limitations

- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- The first explicit transcription-status query can still wait for hardware
  discovery after the desktop window appears.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.

## Verification

- Core, API, desktop, and Worker suites pass with 401 automated checks.
- Promotion, sealed-evaluation, and training-runner gates pass with 11 checks.
- Web and Python lint pass, and the full application build succeeds.
- The full browser suite passes 131 checks with 17 environment-dependent checks
  skipped.
- A packaged macOS ZIP passed isolated cold-start and runtime smoke checks.
- CI rebuilds and checks macOS and Windows packages before creating the
  prerelease.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- Verify downloaded assets against `SHA256SUMS`.
