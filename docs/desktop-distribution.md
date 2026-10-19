# Desktop Distribution

## Supported Targets

| Target | Architecture | Installer | Offline Worker |
| --- | --- | --- | --- |
| macOS 14+ | Apple Silicon | DMG and ZIP | PyInstaller onedir |
| Windows 11 | x64 | NSIS and ZIP | PyInstaller onedir |

The React renderer and bundled Node API are shared across both targets. Model
weights are distributed separately from the application installer.

The desktop user-data identity is pinned independently of the package version,
product display name and installation directory. Settings, OAuth state,
projects and downloaded model weights remain outside the application bundle;
an in-place update replaces binaries only. Windows uninstallers also preserve
application data unless a future explicit deletion flow is selected.

## Build Inputs

- Node.js 24 and `npm ci`.
- The pinned local protected-media Rust/WASM decoder and its third-party notice.
- Python 3.11 with `services/worker[desktop,high-accuracy]` for the open-source
  build, or `services/worker[desktop]` for a commercial build.
- The platform `ffmpeg-static` binary.
- Signing credentials supplied only by the release environment.
- A model package containing the Demucs checkpoint and optional transcription
  and technique weights.

Build the frozen sidecar and its SHA-256 inventory:

```bash
npm run desktop:sidecar
```

Then build the renderer, API bundle and installer:

```bash
npm run build
npm run package -w @gts/desktop
```

Generated sidecar files and installers are ignored by Git. The release workflow
repeats these steps independently on macOS arm64 and Windows x64.

The default desktop surface exposes only the Basic Pitch production baseline.
Research providers remain in the source tree but require
`GTS_ENABLE_RESEARCH_MODELS=1` or an explicit provider allowlist. The
open-source sidecar may include the MIT-licensed MuScriptor runtime, but it
never includes MuScriptor model weights. Those gated weights remain a separate
CC BY-NC 4.0 download accepted by the user. Browser authorization uses an
OAuth public client with Authorization Code + PKCE and a loopback callback.
Set `GTS_HF_OAUTH_CLIENT_ID` at runtime, or configure the public client ID in
the desktop runtime panel. Register
`http://127.0.0.1/oauth/callback` as a loopback redirect URI. Tokens are
encrypted with Electron `safeStorage`; if operating-system encryption is not
available, tokens remain memory-only. Manual `.safetensors` import remains
available without OAuth. A commercial build sets
`GTS_COMMERCIAL_BUILD=1` while building the sidecar and
`GTS_TRANSCRIPTION_PROVIDER_ALLOWLIST=basic-pitch` at runtime. Both controls
are enforced independently so a noncommercial provider cannot be selected or
accidentally frozen into the commercial worker.

The OAuth application must be public and have no client secret. The Client ID
is application configuration, not the authorization code returned after user
consent. Production distributions should ship the publisher-owned public
Client ID so end users only perform browser login; the manual Client ID field
is retained for development and self-hosted builds.

The MuScriptor research preparation command also downloads the CC BY 4.0
Noise2Fret checkpoint and, for Medium/Large, the MIT-licensed MR-MT3
checkpoint. The guitar-specialist strategy uses an installed StringTrace
onset-position research bundle. The multitrack-research strategy downloads a
pinned YourMT3+ checkpoint and combines it with Basic Pitch and Noise2Fret.
All downloaded checkpoints use pinned SHA-256 digests. Noncommercial desktop
builds include the MR-MT3 and YourMT3 adapters; commercial builds exclude the
research providers.

Cloud inference can be configured from the runtime panel or preconfigured with
`GTS_CLOUD_API_ROOT`. The endpoint must use HTTPS and pass `/health` before the
application stores it or switches the current project to cloud execution.

## Native File Access

The desktop renderer remains sandboxed and cannot enumerate the filesystem.
Media and score imports open an Electron main-process native file dialog, with
both supported-format and all-file filters. The selected file is read only
after the user chooses it.

- macOS may require Files and Folders permission for Desktop, Documents,
  Downloads, removable volumes, or network volumes. The setting is under
  Privacy & Security; hidden locations are available from the native dialog.
- Windows applies the current account's NTFS permissions and may hide
  unavailable OneDrive placeholders or administrator-only locations.
- Linux applies normal filesystem permissions. Flatpak and Snap packages may
  additionally require a desktop-portal grant.
- The browser build must continue to use the browser file picker and cannot
  request blanket directory access.

## Model Package Contract

Each package has a validated manifest with:

- semantic package and application compatibility versions;
- an explicit `research` or `production` release channel; legacy packages are
  treated as research packages;
- exact operating system, CPU architecture and analysis schema;
- relative file paths, roles, executable bits, sizes and SHA-256 values;
- optional Worker, FFmpeg, Demucs repository checkpoint, transcription model,
  technique model, gate and license files.

Provider weights imported from the runtime panel use a separate managed store.
The desktop streams the selected `.safetensors` file into private application
storage, records its SHA-256 digest, writes the matching architecture config
and activates it only after the complete file is in place.

Downloads require HTTPS, remain under the manifest origin and path, support
range resume, enforce declared byte limits, and are verified before an atomic
install. Activation keeps the previous package as a rollback target. The
active package cannot be removed.

## Signing And Update Policy

- macOS release builds use hardened runtime, Developer ID signing and Apple
  notarization credentials supplied through the CI secret store.
- Windows release builds use an Authenticode certificate supplied through the
  CI secret store.
- Unsigned CI directory builds are smoke-test artifacts only and are never
  published as end-user releases.
- Update metadata is fetched over HTTPS by `electron-updater`; automatic
  download is disabled until the user accepts the offered version.
- Application rollback uses the previous signed installer. Model rollback is
  independent and available from the desktop runtime panel.

## Verification Matrix

For each release candidate:

1. Install on a clean supported operating system.
2. Launch without Python, FFmpeg or model tools installed globally.
3. Open and drag a supported local media file.
4. Confirm and analyze NCM and QMC fixtures with networking disabled; verify no
   decoded media is exposed or retained after the task.
5. Cancel an active task and verify that its process group exits.
6. Complete local inference with a verified model package.
7. Restart and recover projects, revisions, recent projects and window state.
8. Create a backup, remove local state, restore it and verify revision history.
9. Switch to an unreachable cloud endpoint and verify local editing remains
   available without implicit upload.
10. Exercise 100%, 125%, 150% and 200% display scaling and keyboard navigation.
11. Upgrade over the previous signed version, then uninstall while preserving
    user data unless explicit deletion is selected.

Items requiring signing identities, clean external machines or trained model
weights are release-owner checks. The repository CI covers compilation, unit
tests, renderer E2E, Electron smoke, sidecar generation and unsigned packaging.
