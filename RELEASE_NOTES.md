# Guitar Tab Studio v0.1.0-preview.9

This desktop preview focuses on a production-oriented Basic Pitch workflow,
multi-guitar project structure, durable local data, and a more coherent visual
system. It remains intended for functional testing and early feedback, not
production model evaluation.

## Highlights

- Basic Pitch is now the only provider exposed by default. MuScriptor,
  MR-MT3, YourMT3+, Noise2Fret and StringTrace remain available only through
  explicit research-mode configuration.
- Research adapters and previously downloaded local model caches are retained
  across upgrades. Preview.9 does not silently remove model data or OAuth
  configuration.
- Projects now support a full score plus independent guitar parts with
  lead/rhythm/mixed/unknown roles. Legacy projects migrate to schema version 2.
- Stereo separation now creates guitar-part candidates rather than assuming
  every song contains two guitars. Candidates must pass spatial confidence,
  musical-role contrast, event-count and duplicate-note gates; otherwise the
  aggregate guitar stem is transcribed again as one part.
- The renderer can switch between the full score and each confirmed guitar
  part without merging multiple parts into an unplayable six-string TAB.
- The application and in-app brand marks now share the same six-string pick
  design. The in-app mark follows the selected theme.
- The theme library now contains 18 presets, including seven dark themes and
  the new red-and-black Scarlet Night preset. Amber is the first-run default.
- Desktop preview and packaged builds now use a stable user-data identity, so
  settings, OAuth state and downloaded weights survive rebuilds and upgrades.
- Frozen workers correctly re-enter bundled Demucs and yt-dlp entry points and
  initialize multiprocessing before argument parsing.
- Documentation now records the research-model retention and future removal
  criteria explicitly.

## Preview Limitations

- The current transcription model has not passed the production gate of TAB F1
  0.75 and repeated-note recall 0.85. Results still require human review.
- Installers are unsigned and not notarized. macOS Gatekeeper and Windows
  SmartScreen may display warnings.
- macOS builds currently support Apple Silicon only. Windows builds support x64
  only.
- Research providers remain experimental, non-default and subject to their
  individual model licenses.
- Conservative multi-guitar confirmation can leave centered or highly
  correlated guitar layers combined in one aggregate part.
- Existing projects retain their previous analysis until they are analyzed
  again with Preview.9.
- The built-in TAB and technique candidates are generated analysis results and
  have not been manually verified against the production quality gate.
- Preview.5 and earlier must install Preview.6 or newer manually once. Their
  updater excluded prerelease releases; automatic preview updates work after
  that upgrade.

## Downloads

- macOS: DMG installer or ZIP archive.
- Windows: NSIS installer (`.exe`) or ZIP archive.
- `latest-mac.yml`, `latest.yml` and blockmaps are included for preview update
  testing.
- Verify downloads against `SHA256SUMS`.
