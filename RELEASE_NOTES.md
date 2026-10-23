# Guitar Tab Studio v0.1.0-preview.14

This desktop preview improves live tuning, project naming, count-in playback,
and reusable isolated media authorization. It remains a functional preview:
generated notation still requires human review.

## Highlights

- The tuner now tracks quiet sustain instead of dropping the reading shortly
  after the pick attack. An adaptive noise gate keeps the note visible while
  the string is still sounding and releases it only after sustained silence.
- A continuous sound trace, input level, pitch confidence, and cents movement
  make the live signal visible while tuning.
- Guitar remains the default tuner. Bass and ukulele standard tunings are now
  available from instrument tabs, with mode-specific detection ranges and
  nearest-string selection.
- Web and desktop uploads send the original UTF-8 filename in a dedicated form
  field. The API sanitizes that value, repairs multipart Latin-1 mojibake, and
  migrates affected historical project titles and filenames.
- Local analyses use the original upload filename instead of a temporary hash.
  Online acquisition preserves bounded page title, artist, and canonical URL
  metadata while keeping the decoded media duration authoritative.
- Count-in no longer seeks past the real beginning of a recording when the
  first detected measure starts later. It schedules exactly one click per beat
  and no longer adds an extra click at song start.
- The import dialog is now the single progress surface while it is open, so an
  analysis job no longer displays duplicate progress controls.
- Isolated Bilibili and QQ Music login sessions can be reused during the current
  application run. Per-task cookie and credential exports remain temporary and
  are deleted after each task; the reusable login partition is cleared on app
  shutdown.
- Desktop preview packaging now always builds and audits the complete Worker and
  FFmpeg sidecar before launching, preventing a preview from reporting that the
  real analysis Worker is unavailable.

## Cross-Platform Coverage

- UTF-8 upload filenames use the same explicit transport and repair policy on
  macOS, Windows, Linux, Web, and the Electron local API.
- Windows Chromium App-Bound cookies still use isolated in-app authorization;
  authenticated state is retained only for the current app process.
- Packaged macOS and Windows builds contain the same frozen Worker checks,
  media normalization gates, and release-asset integrity checks.
- macOS builds target Apple Silicon. Windows builds target x64.

## Preview Limitations

- The transcription model has not passed the production gate of TAB F1 0.75
  and repeated-note recall 0.85. Results still require human review.
- Online import remains subject to platform policy, account authorization,
  regional availability, anti-automation checks, and DRM restrictions.
- Isolated web sign-in does not copy the user's full browser profile and may
  require signing in again after the app exits.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.

## Verification

- Core, API, desktop, Worker, and web test suites pass.
- Tuner E2E coverage verifies quiet sustain, release after silence, sound-trace
  updates, and guitar, bass, and ukulele switching.
- The packaged Electron preview was checked at desktop and narrow responsive
  viewports with no horizontal overflow.
- The complete 1.5 GB sidecar contains the Worker and FFmpeg and passes its
  audited-file packaging gate.
- Release assets include macOS DMG/ZIP, Windows EXE/ZIP, blockmaps,
  `latest-mac.yml`, `latest.yml`, this release note, and `SHA256SUMS`.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- Verify downloaded assets against `SHA256SUMS`.
