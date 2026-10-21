# Guitar Tab Studio v0.1.0-preview.11

This desktop preview prioritizes score timing, repeatable beat evaluation, and
practical rehearsal tools. It remains intended for functional testing and
human-reviewed transcription work, not production model evaluation.

## Highlights

- Windows media shortcuts now open user-visible download locations before
  application caches. QQ Music defaults to the Windows Music folder, with
  tested fallbacks for QQ Music, NetEase Cloud Music, MOO, Kuwo and Kugou.
- Keyless MusicEx QMC2 files are now supported by the Windows desktop build.
  The local API reads only bounded QQ Music configuration/registry data and
  uses the official Windows authorization request profile; credentials and
  returned ekeys remain in memory.
- Metronome, rhythm backing and drum-machine output now use a limited 2x gain
  bus. Acoustic, electronic, brush and wood kits have distinct synthesis
  envelopes, filters and transients instead of minor parameter variations.
- The standalone drum machine now supports editable lane steps, per-meter
  custom-pattern persistence, preset restoration and pattern clearing.
- Score timing now preserves the original performance timeline while writing
  notation against a per-beat tempo map. Each beat can select binary, triplet,
  sixteenth-note, or thirty-second-note subdivisions without altering audio.
- Repeated notes are merged only when no independent acoustic rearticulation is
  present. Pickup measures, leading rests, rubato, and tempo drift are retained
  explicitly instead of being forced onto a uniform grid.
- The built-in Internationale analysis was rebuilt as a 109-measure candidate
  with 1,462 note events, a detected pickup, and six playable stems. It remains
  quarantined pending human gold-standard review.
- A versioned beat gold-standard workflow now supports clip initialization,
  two independent submissions, disagreement comparison, third-party
  adjudication, corpus evaluation, and release gating through `beat:gold`.
- Full-score layout now reflows by available width, reaching six measures per
  row on a 1920-pixel workspace while preserving compact-window readability.
- The independent metronome and drum machine now include a two-measure count-in,
  tap tempo, meter and subdivision controls, per-part mixing, and expanded fill
  patterns. They stop cleanly when leaving the practice workspace.
- New tuner and first-stage live transcription workspaces read the microphone
  directly. Live transcription records monophonic pitch, string, fret, and
  duration without modifying reviewed project scores.
- Drum notation was removed from generated score choices; drum stems remain
  available in the mixer for listening and practice.
- Analysis can continue while the desktop window is hidden to the tray. Score
  playback controls remain available while scrolling.
- Online media import now starts anonymously and requests one-time access to
  the default browser's cookies only after an authentication failure. Cookies
  are neither persisted by the app nor uploaded.
- Local-upload mode now resets deterministically, and temporary-file cleanup is
  idempotent to avoid duplicate-removal failures.
- Chinese and English interface selection, system-language detection, expanded
  dark themes, and corrected Windows icon scaling are included.

## Beat Gold Standard

- The repository defines a staged 48/200/400+ clip corpus with recording-level
  partitioning and quotas for meter, tempo, pickup, syncopation, swing, rubato,
  instrumentation, and recording quality.
- No downloaded evaluation songs are included in this release. Copyrighted
  source audio must remain local and outside the repository and installers.
- Preview.11 provides the review and evaluation machinery, but the independent
  human annotations and adjudicated test corpus are not complete.

## Preview Limitations

- The current transcription model has not passed the production gate of TAB F1
  0.75 and repeated-note recall 0.85. Results still require human review.
- Beat timing improvements are structural and regression-tested, but the new
  300-500-clip human gold standard has not yet established production metrics.
- The built-in TAB, rhythm, and technique candidates are generated results and
  have not been manually approved as gold-standard notation.
- Existing projects retain their previous analysis until they are analyzed
  again with Preview.11.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- macOS builds currently support Apple Silicon only. Windows builds support x64
  only.
- Research providers remain experimental, hidden by default, and subject to
  their individual model licenses.
- Preview.5 and earlier must install Preview.6 or newer manually once. Their
  updater excluded prerelease releases; automatic preview updates work after
  that upgrade.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- `latest-mac.yml`, `latest.yml`, and blockmaps are included for preview update
  testing.
- Verify downloads against `SHA256SUMS`.
