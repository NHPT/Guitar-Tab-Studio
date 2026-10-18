# Guitar Tab Studio v0.1.0-preview.5

This is the first public desktop preview for macOS Apple Silicon and Windows
x64. It is intended for functional testing and early feedback, not production
model evaluation.

## Highlights

- Offline desktop workspace with local projects, revision history, score import,
  playback, looping, count-in, rhythm accompaniment and synchronized stems.
- Local analysis sidecar with Basic Pitch, Demucs, MuScriptor and the optional
  MR-MT3 verification runtime. Model weights are downloaded separately.
- High Precision mode uses MuScriptor as the primary transcription provider.
  MR-MT3 verification is limited to CUDA Medium/Large profiles; CPU and Apple
  Silicon run a single transcription provider to avoid redundant processing.
- Manual Medium and Large selections now remain available when the optional
  MR-MT3 runtime is absent. The app clearly reports the reduced provider set
  and continues with MuScriptor, Basic Pitch and Noise2Fret.
- Analysis now reports MuScriptor chunk progress from 65% to 80%, then exposes
  the Basic Pitch and string/fret review stages instead of appearing stalled.
- Worker progress, score-import output and analysis errors now use explicit
  UTF-8 decoding on Windows, including messages split across process chunks.
- Hugging Face public OAuth with PKCE for gated model access. No client secret
  is stored in the application.
- The built-in project is now John Denver's `Take Me Home, Country Roads` from
  <https://music.163.com/#/song?id=1478037>, with generated TAB and seven
  independently controllable stems.

## Preview Limitations

- The current transcription model has not passed the production gate of TAB F1
  0.75 and repeated-note recall 0.85. Results still require human review.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- macOS builds currently support Apple Silicon only. Windows builds support x64
  only.
- First use of High Precision mode requires downloading the selected model
  package and may take several minutes.
- The built-in TAB and technique candidates are generated analysis results and
  have not been manually verified against the production quality gate.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- `latest-mac.yml`, `latest.yml` and blockmaps are included for preview update
  testing.
- Verify downloads against `SHA256SUMS`.
