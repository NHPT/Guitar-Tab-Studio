# Guitar Tab Studio v0.1.0-preview.13

This desktop preview fixes Windows media-session failures, audio-separation
encoding errors, import recovery-state leakage, and dark-theme contrast. It
remains a functional preview: generated notation still requires human review.

## Highlights

- Demucs, FFmpeg, FFprobe, and other external-process output is decoded as
  UTF-8 with replacement fallback. Windows no longer aborts stem separation
  when localized output cannot be decoded as GBK.
- QQ Music import now prioritizes the active Windows desktop session. It
  resolves the current UIN, checks bounded cookie and MMKV records, and can use
  a read-only, same-user scan of verified QQ Music processes when a current
  client keeps the named session token only in memory.
- The QQ Music runtime scan is limited by process identity, active UIN, time,
  memory, and output size. It does not scan unrelated processes or guess
  credentials from unbound Base64-looking data.
- Isolated QQ Music web authorization remains available only as a fallback.
  Its one-time credential is deleted after the retry.
- Windows Chromium App-Bound encryption is detected before Bilibili retry.
  Edge, Chrome, Brave, Opera, and Vivaldi use an isolated in-app sign-in when
  their system cookies cannot be reused safely.
- Local media, score image/PDF, and online-link import errors now have separate
  recovery state. Switching source tabs or closing the dialog clears stale
  authorization prompts and actions from the previous source.
- Native selects, options, option groups, disabled choices, placeholders,
  popovers, score selections, and primary actions now use explicit semantic
  foreground and background colors.
- All nine dark themes have automated contrast coverage for normal and disabled
  options, accent text, active surfaces, and primary-button hover states. The
  tested combinations meet a minimum 4.5:1 contrast ratio.
- Tagged Windows sidecars run an additional QQ Music runtime self-test before
  packaging. Existing frozen-runtime, real-MP3 normalization, and Demucs data
  gates remain enabled.
- Preview.12 update progress, runtime status colors, isolated media sessions,
  and separate language/appearance controls remain included.

## Cross-Platform Coverage

- macOS and Linux retain direct browser-cookie access where supported and can
  use isolated Bilibili sign-in when browser extraction fails.
- QQ Music web authorization is available to packaged desktop builds even when
  the native client format is unavailable.
- External-tool output uses the same UTF-8 decoding policy across platforms.
- The same frozen Worker runtime and MP3 decode gate runs on macOS and Windows
  release jobs, with the Windows-only QQ Music runtime gate added where needed.
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
