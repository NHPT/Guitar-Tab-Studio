`npm run desktop:sidecar` generates platform-specific runtime files here:

- `guitar-tab-worker` (or `.exe`) with its PyInstaller runtime.
- `ffmpeg` (or `.exe`) from the pinned npm dependency.
- `LICENSE.txt` and `sidecar-manifest.json` with per-file SHA-256 hashes.

Model weights remain separate verified model packages. A complete offline package
must include the Demucs `htdemucs_6s` checkpoint with role
`separation-model`; the desktop passes its containing directory to the worker.
Generated binaries, manifests and model weights are not committed.
