# Guitar Tab Studio v0.1.0-preview.16

This preview establishes a strict boundary between the user-facing desktop
application and the Web-only administration system. It also improves beat and
downbeat analysis and streamlines the full-song review workflow used to
validate those improvements.

## Highlights

- The desktop package now uses a dedicated renderer entry containing only the
  user workspace. Admin, review, and community control pages are not included
  in the packaged renderer.
- The local desktop API is built without Admin routes, authentication,
  dashboards, review management, adjudication, training, backup, or gold
  release endpoints.
- Electron rejects Admin, review, and community navigation at the main-process
  boundary. Packaged builds ignore renderer URL overrides.
- Packaging fails if the desktop renderer or local API contains an Admin route,
  Admin session, or Admin workspace marker.
- Beat analysis now preserves stronger full-song phase evidence, changing
  meters, pickup context, tempo structure, and representative listening
  positions for human correction.
- The Web-only beat review flow starts with full-song structure confirmation,
  then opens a reduced core clip queue while retaining all original candidates.
  Confirmed structure can be applied without replacing reviewed beat times.

## Cross-Platform Builds

- macOS installer and ZIP archive for Apple Silicon.
- Windows NSIS installer and ZIP archive for x64.
- Update metadata and blockmaps are generated for both supported platforms.
- A source TAR.GZ, release notes, and SHA-256 manifest are published with the
  binary assets.

## Preview Limitations

- The transcription model has not passed the production gate of TAB F1 0.75
  and repeated-note recall 0.85. Generated notation still requires review.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.

## Verification

- Core, API, and desktop unit suites pass with 197 checks; Web lint passes
  without warnings.
- The Worker audio-analysis suite passes 78 checks.
- The Web-only beat review workflow passes 10 desktop and mobile browser tests.
- Focused score-timing and responsive mobile checks pass in an isolated
  Playwright runtime.
- The packaged macOS application loads only the user renderer, has no Admin
  navigation, and rejects a forced Admin file navigation.
- CI rebuilds and checks macOS and Windows packages before creating the
  prerelease.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- Verify downloaded assets against `SHA256SUMS`.
