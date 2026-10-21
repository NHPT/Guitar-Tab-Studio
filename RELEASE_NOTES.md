# Guitar Tab Studio v0.1.0-preview.12

This desktop preview focuses on Windows reliability, authenticated media
imports, observable updates, and clearer appearance controls. It remains a
functional preview: generated notation still requires human review.

## Highlights

- Windows QQ Music authorization now accepts additional current-client
  credential formats, including Qt byte arrays, URL-encoded values, binary
  key/value records, and UIN prefixes used by recent releases.
- When a QQ Music client login cannot be read safely, the desktop app offers an
  isolated QQ Music web sign-in. The resulting credential is scoped to one
  retry and deleted when the task finishes.
- Windows Chromium browsers no longer rely on decrypting App-Bound encrypted
  cookies for Bilibili retries. An isolated in-app sign-in exports only the
  target site's cookies to a temporary file and removes it after use.
- Browser-cookie and QQ Music authorization files are validated against a
  private runtime directory, referenced by opaque one-time tokens, excluded
  from persisted jobs, and cleaned on success, failure, cancellation, or app
  shutdown.
- The packaged Worker now includes the Demucs model registry and
  `htdemucs_6s` configuration that were missing from previous sidecars.
- Every tagged macOS and Windows sidecar build now runs a frozen-runtime
  self-test, generates a real MP3 fixture, and verifies MP3 normalization before
  packaging can continue.
- Worker failures preserve a larger diagnostic tail and emit a structured
  stage-specific error. Missing Windows DLLs, damaged runtimes, unsupported
  codecs, missing Demucs data, and disk exhaustion now produce actionable
  messages instead of only `exit code 1`.
- Desktop updates now show percentage, downloaded and total size, transfer
  speed, and synchronized taskbar or Dock progress. Completion and failure
  states clear the progress indicator deterministically.
- Runtime availability is easier to scan: local API, project storage, Worker,
  and active-model states use distinct ready, pending, warning, unavailable,
  and neutral colors.
- Appearance mode and color theme are now separate settings. The header language
  button opens language choices only, while the theme button opens a dedicated
  chooser for system, light, and dark modes plus 20 color themes.
- Two additional dark themes, Nocturne and Aurora, are included. Existing theme
  preferences migrate without resetting the selected light or dark palette.
- Windows media-folder shortcuts and keyless MusicEx QMC2 support from
  Preview.11 remain included.

## Cross-Platform Coverage

- macOS and Linux retain direct browser-cookie access where supported and can
  use isolated Bilibili sign-in when browser extraction fails.
- QQ Music web authorization is available to packaged desktop builds even when
  the native client format is unavailable.
- The same frozen Worker runtime and MP3 decode gate runs on macOS and Windows
  release jobs.
- macOS builds target Apple Silicon. Windows builds target x64.

## Preview Limitations

- The transcription model has not passed the production gate of TAB F1 0.75
  and repeated-note recall 0.85. Results still require human review.
- Online import remains subject to platform policy, account authorization,
  regional availability, anti-automation checks, and DRM restrictions.
- Isolated web sign-in does not copy the user's full browser profile and may
  require signing in again inside the temporary window.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.

## Verification

- Core, API, desktop, and web test suites pass.
- Worker unit tests, Electron smoke checks, and targeted desktop E2E checks
  pass.
- The frozen sidecar runtime self-test and real MP3 decode smoke test pass.
- Release assets include macOS DMG/ZIP, Windows EXE/ZIP, blockmaps,
  `latest-mac.yml`, `latest.yml`, this release note, and `SHA256SUMS`.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- Verify downloaded assets against `SHA256SUMS`.
