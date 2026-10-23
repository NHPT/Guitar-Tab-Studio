# Guitar Tab Studio v0.1.0-preview.15

This desktop preview improves tuner precision and weak-sustain tracking, adds
an in-tune confirmation sound, and replaces the desktop update prompt with a
resumable in-app download flow. It remains a functional preview: generated
notation still requires human review.

## Highlights

- The tuner searches a wider detune range and uses spectral harmonic evidence
  to avoid locking a low string onto an overtone. Fine pitch movement now
  follows gradual peg adjustments instead of being damped into a stale value.
- Attack and sustain use separate adaptive gates. A ringing string remains
  visible down to a much quieter input level, while a delayed release prevents
  brief signal dips from clearing the reading.
- The pitch trace retains roughly nine seconds of recent movement so tuning
  changes remain visible after the initial pick attack.
- A short confirmation sound plays once after the selected string remains
  within two cents for about 0.4 seconds. Hysteresis prevents repeated sounds
  while the reading stays near the target.
- Update availability and install readiness now use dialogs consistent with
  the application UI. Active downloads move into a compact progress controller
  instead of relying on native confirmation prompts.
- Update downloads can be paused, resumed, cancelled, or retried. HTTP Range
  requests preserve partial data across pauses and application restarts;
  explicit cancellation removes the partial file.
- Completed updates are verified against the release SHA-512 or SHA-256 digest
  before installation. Existing platform signature checks remain in the update
  path where signing is available.

## Cross-Platform Coverage

- The same update state contract and progress UI are used on packaged macOS and
  Windows builds.
- Partial downloads are stored in the application-managed user-data directory
  and moved to the updater destination only after checksum verification.
- macOS builds target Apple Silicon. Windows builds target x64.

## Preview Limitations

- The transcription model has not passed the production gate of TAB F1 0.75
  and repeated-note recall 0.85. Results still require human review.
- Tuner performance still depends on microphone placement, room noise, and
  instrument sustain, although the adaptive gates now tolerate much weaker
  signals.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.

## Verification

- Core, API, and desktop suites plus web lint pass with 192 automated checks.
- Focused browser audio coverage verifies harmonic disambiguation, gradual
  pitch tracking, weak sustain, delayed release, and one-shot confirmation.
- Desktop update coverage verifies the available, downloading, paused, resumed,
  failed, cancelled, and ready states.
- The packaged Electron preview was checked at desktop and 390-pixel responsive
  viewports with no horizontal overflow.
- Release assets include macOS DMG/ZIP, Windows EXE/ZIP, blockmaps,
  `latest-mac.yml`, `latest.yml`, this release note, and `SHA256SUMS`.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- Verify downloaded assets against `SHA256SUMS`.
