import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { basename, dirname, extname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  net,
  safeStorage,
  screen,
  session,
  shell,
  Tray,
  type MenuItemConstructorOptions,
} from 'electron'
import {
  migrateStudioProject,
  type AppLanguage,
  type AppLanguagePreference,
  type AuthorizedMediaConsent,
  type DesktopApplicationSettings,
  type DesktopBackgroundJobState,
  type DesktopInvokeRequest,
  type DesktopLocalFileSource,
  type DesktopRuntimeStatus,
  type DesktopSelectedFile,
  type DesktopUpdateProgress,
  type MediaBrowserSession,
  type ProjectRevision,
  type ProjectRevisionKind,
  type StorageCleanupConfig,
  type StudioProject,
  type TranscriptionModelVariant,
  type TranscriptionProviderId,
  type TranscriptionSetupProgress,
} from '@gts/core'
import {
  LocalApiService,
  type DesktopMediaKeyProvider,
  type DesktopScoreUpload,
  type DesktopUpload,
  type StudioApiTransport,
} from './local-api.js'
import { desktopDataRoot } from './data-root.js'
import { detectDefaultMediaBrowser } from './default-browser.js'
import {
  desktopMediaSourceLabels,
  desktopMediaSources,
  localFileDefaultPath,
  type DesktopMediaSource,
} from './local-files.js'
import { DesktopMaintenance } from './maintenance.js'
import { DesktopMediaSessions } from './media-sessions.js'
import { HuggingFaceOAuth } from './huggingface-oauth.js'
import { ModelPackageManager } from './model-packages.js'
import { DesktopTranscriptionManager } from './model-providers.js'
import { proxyEnvironmentFromRule } from './network-proxy.js'
import { RemoteApiService } from './remote-api.js'
import { DesktopStateStore, type WindowState } from './state.js'
import { DesktopUpdater } from './updater.js'

const moduleDirectory = __dirname
const repositoryRoot = resolve(moduleDirectory, '../../..')
const developmentApiEntry = resolve(repositoryRoot, 'apps/api/dist/index.js')
const developmentWebEntry = resolve(repositoryRoot, 'apps/web/dist/index.html')
const developmentWorker = resolve(repositoryRoot, '.venv/bin/guitar-tab-worker')
const developmentGuitarResearchModel = resolve(
  repositoryRoot,
  'apps/api/data/training/checkpoints/stringtrace-onset-position-research-v24.bundle.pt',
)
const developmentFfmpeg = resolve(
  repositoryRoot,
  'node_modules/ffmpeg-static/ffmpeg',
)
const maximumUploadBytes = 500 * 1024 * 1024
const maximumMediaKeyDatabaseBytes = 128 * 1024 * 1024
const configuredDataRoot = process.env.GTS_DESKTOP_DATA_ROOT

app.setPath(
  'userData',
  desktopDataRoot(app.getPath('appData'), configuredDataRoot),
)

function isPackagedRuntime(): boolean {
  return app.isPackaged && process.env.GTS_DESKTOP_SMOKE !== '1'
}

let mainWindow: BrowserWindow | undefined
let stateStore: DesktopStateStore | undefined
let localApi: LocalApiService | undefined
let remoteApi: RemoteApiService | undefined
let modelPackages: ModelPackageManager | undefined
let transcriptionManager: DesktopTranscriptionManager | undefined
let huggingFaceOAuth: HuggingFaceOAuth | undefined
let maintenance: DesktopMaintenance | undefined
let mediaSessions: DesktopMediaSessions | undefined
let updater: DesktopUpdater | undefined
let tray: Tray | undefined
let quitting = false
let cloudConfigurationError: string | undefined
let backgroundJobState: DesktopBackgroundJobState = { active: false }
const localFileDirectories = new Map<string, string>()
const openedTranscriptionUrls = new Set<string>()
let latestTranscriptionAuthorizationUrl: string | undefined
const mediaBrowserIds = new Set<MediaBrowserSession['browser']>([
  'brave',
  'chrome',
  'chromium',
  'edge',
  'firefox',
  'opera',
  'safari',
  'vivaldi',
])

function systemLanguage(): AppLanguage {
  return app.getLocale().toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US'
}

function applicationSettings(): DesktopApplicationSettings {
  const languagePreference =
    stateStore?.languagePreference() ?? 'system'
  const resolvedSystemLanguage = systemLanguage()
  return {
    languagePreference,
    systemLanguage: resolvedSystemLanguage,
    language:
      languagePreference === 'system'
        ? resolvedSystemLanguage
        : languagePreference,
  }
}

function desktopIconPath(): string {
  return isPackagedRuntime()
    ? resolve(process.resourcesPath, 'icon.ico')
    : resolve(repositoryRoot, 'apps/desktop/build-resources/icon.ico')
}

function sendTranscriptionProgress(
  progress: TranscriptionSetupProgress,
): void {
  mainWindow?.webContents.send('studio:transcription-progress', progress)
  if (!progress.authorizationUrl) return
  try {
    const url = new URL(progress.authorizationUrl)
    if (
      url.protocol !== 'https:' ||
      !['huggingface.co', 'hf.co'].includes(url.hostname)
    ) {
      return
    }
    const authorizationUrl = url.toString()
    latestTranscriptionAuthorizationUrl = authorizationUrl
    if (openedTranscriptionUrls.has(authorizationUrl)) return
    openedTranscriptionUrls.add(authorizationUrl)
    void shell.openExternal(authorizationUrl)
  } catch {
    // Invalid external URLs are ignored.
  }
}

function runtimePaths() {
  if (!isPackagedRuntime()) {
    return {
      apiEntry: developmentApiEntry,
      apiWorkingDirectory: resolve(repositoryRoot, 'apps/api'),
      webEntry: developmentWebEntry,
      workerCommand: existsSync(developmentWorker)
        ? developmentWorker
        : undefined,
      ffmpegBinary: existsSync(developmentFfmpeg)
        ? developmentFfmpeg
        : undefined,
    }
  }
  const executableSuffix = process.platform === 'win32' ? '.exe' : ''
  const packagedWorker = resolve(
    process.resourcesPath,
    `sidecar/guitar-tab-worker${executableSuffix}`,
  )
  const packagedFfmpeg = resolve(
    process.resourcesPath,
    `sidecar/ffmpeg${executableSuffix}`,
  )
  return {
    apiEntry: resolve(process.resourcesPath, 'api/index.mjs'),
    apiWorkingDirectory: app.getPath('userData'),
    webEntry: resolve(process.resourcesPath, 'web/index.html'),
    workerCommand: existsSync(packagedWorker) ? packagedWorker : undefined,
    ffmpegBinary: existsSync(packagedFfmpeg) ? packagedFfmpeg : undefined,
  }
}

async function modelDownloadNetworkEnvironment(): Promise<NodeJS.ProcessEnv> {
  try {
    const rule = await session.defaultSession.resolveProxy(
      'https://huggingface.co',
    )
    return proxyEnvironmentFromRule(
      rule,
      process.env.NO_PROXY ?? process.env.no_proxy,
    )
  } catch {
    return {}
  }
}

function visibleWindowState(saved?: WindowState): WindowState {
  const fallback: WindowState = {
    width: 1360,
    height: 900,
    maximized: false,
  }
  if (!saved || saved.x === undefined || saved.y === undefined) return saved ?? fallback
  const display = screen.getDisplayMatching({
    x: saved.x,
    y: saved.y,
    width: saved.width,
    height: saved.height,
  })
  const area = display.workArea
  const visible =
    saved.x < area.x + area.width &&
    saved.x + saved.width > area.x &&
    saved.y < area.y + area.height &&
    saved.y + saved.height > area.y
  return visible ? saved : fallback
}

function payloadRecord(
  request: DesktopInvokeRequest,
): Record<string, unknown> {
  if (
    request.payload === undefined ||
    typeof request.payload !== 'object' ||
    request.payload === null ||
    Array.isArray(request.payload)
  ) {
    return {}
  }
  return request.payload as Record<string, unknown>
}

function requiredText(
  payload: Record<string, unknown>,
  key: string,
  maximum = 4096,
): string {
  const value = payload[key]
  if (typeof value !== 'string' || value.length === 0 || value.length > maximum) {
    throw new Error(`Invalid ${key}`)
  }
  return value
}

function selectedFileMediaType(path: string): string {
  const extension = extname(path).toLowerCase()
  return {
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.pdf': 'application/pdf',
  }[extension] ?? 'application/octet-stream'
}

function fileAccessError(error: unknown): Error {
  const code =
    error && typeof error === 'object' && 'code' in error
      ? String(error.code)
      : ''
  if (code !== 'EACCES' && code !== 'EPERM') {
    return error instanceof Error ? error : new Error('无法读取所选文件')
  }
  if (process.platform === 'darwin') {
    return new Error(
      'macOS 未授权访问该位置。请在“系统设置 > 隐私与安全性 > 文件与文件夹”中允许 Guitar Tab Studio 访问对应目录。',
    )
  }
  if (process.platform === 'win32') {
    return new Error(
      'Windows 拒绝访问该位置。请选择当前账号有读取权限的目录，或检查文件夹安全权限与 OneDrive 可用状态。',
    )
  }
  return new Error(
    '系统拒绝访问该位置。请检查目录读取权限以及 Flatpak、Snap 或文件门户授权。',
  )
}

function authorizedMediaConsent(
  value: unknown,
): AuthorizedMediaConsent | undefined {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value)
  ) {
    return undefined
  }
  const consent = value as Record<string, unknown>
  return consent.accepted === true && consent.policyVersion === 2
    ? {
        accepted: true,
        policyVersion: 2,
        ...(typeof consent.credentialSession === 'string' &&
        /^[0-9a-f-]{36}$/iu.test(consent.credentialSession)
          ? { credentialSession: consent.credentialSession }
          : {}),
      }
    : undefined
}

async function restartLocalApi(): Promise<void> {
  if (!modelPackages || !transcriptionManager || !stateStore) {
    throw new Error('Model runtime is not ready')
  }
  await modelPackages.recoverActive()
  const paths = runtimePaths()
  const transcription = transcriptionManager.workerConfiguration()
  await localApi?.stop()
  localApi = new LocalApiService({
    apiEntry: paths.apiEntry,
    workingDirectory: paths.apiWorkingDirectory,
    dataRoot: resolve(app.getPath('userData'), 'runtime'),
    mediaSessionRoot: resolve(
      app.getPath('userData'),
      'runtime',
      'media-sessions',
    ),
    homePath: app.getPath('home'),
    workerCommand:
      (!isPackagedRuntime() ? paths.workerCommand : undefined) ??
      modelPackages.resolveActiveFile('worker') ??
      paths.workerCommand,
    ffmpegBinary:
      modelPackages.resolveActiveFile('ffmpeg') ?? paths.ffmpegBinary,
    separationModelRepository: (() => {
      const model = modelPackages.resolveActiveFile('separation-model')
      return model ? dirname(model) : undefined
    })(),
    tabModel: transcription.tabModel,
    tabModelMode: transcription.tabModelMode,
    techniqueModel: modelPackages.resolveActiveFile('technique-model'),
    techniqueGate: modelPackages.resolveActiveFile('technique-gate'),
    transcriptionProvider: transcription.provider,
    transcriptionModel: transcription.model,
    transcriptionVariant: transcription.variant,
    positionModel: transcription.positionModel,
    verificationModel: transcription.verificationModel,
    modelCacheRoot: transcription.cacheRoot,
  })
  await localApi.start()
}

function activeApi(entityId?: string): StudioApiTransport {
  if (!localApi || !stateStore) throw new Error('Desktop runtime is not ready')
  const mode = stateStore.runtimeMode(entityId)
  if (mode === 'cloud' && remoteApi) return remoteApi
  return localApi
}

function projectForRenderer(
  project: StudioProject,
  api: StudioApiTransport,
): StudioProject {
  return {
    ...project,
    tracks: project.tracks.map((track) => ({
      ...track,
      url:
        track.url && track.url.startsWith('/')
          ? api.assetUrl(track.url)
          : track.url,
    })),
  }
}

function projectForStorage(
  project: StudioProject,
  api: StudioApiTransport,
): StudioProject {
  return {
    ...project,
    tracks: project.tracks.map((track) => ({
      ...track,
      url: track.url ? api.storageAssetUrl(track.url) : null,
    })),
  }
}

function saveLocalRevision(
  projectValue: StudioProject,
  label: string,
  kind: ProjectRevisionKind,
  expectedParentRevisionId?: string | null,
): ProjectRevision {
  if (!stateStore) throw new Error('Desktop state is unavailable')
  const project = migrateStudioProject(projectValue)
  const normalizedLabel = label.trim().slice(0, 80)
  if (!normalizedLabel) throw new Error('Revision label is required')
  const previous = stateStore.listRevisions(project.id)[0]
  if (
    expectedParentRevisionId !== undefined &&
    (previous?.id ?? null) !== expectedParentRevisionId
  ) {
    throw new Error('项目已有更新，请先载入最新修订再保存')
  }
  const revision: ProjectRevision = {
    schemaVersion: 1,
    id: randomUUID(),
    projectId: project.id,
    ...(previous ? { parentRevisionId: previous.id } : {}),
    createdAt: new Date().toISOString(),
    label: normalizedLabel,
    kind,
    project,
  }
  stateStore.saveProject(project)
  stateStore.saveRevision(revision)
  stateStore.touchRecentProject(project.id)
  return revision
}

async function invoke(request: DesktopInvokeRequest): Promise<unknown> {
  if (
    !localApi ||
    !stateStore ||
    !modelPackages ||
    !transcriptionManager ||
    !maintenance
  ) {
    throw new Error('Desktop runtime is not ready')
  }
  const payload = payloadRecord(request)
  switch (request.method) {
    case 'capabilities':
      try {
        return await activeApi().capabilities()
      } catch {
        return localApi.capabilities()
      }
    case 'platforms':
      try {
        return await activeApi().platforms()
      } catch {
        return localApi.platforms()
      }
    case 'demo-project':
      return projectForRenderer(await localApi.demoProject(), localApi)
    case 'project': {
      const id = requiredText(payload, 'id', 128)
      const api = activeApi(id)
      try {
        const project = await api.project(id)
        stateStore.saveProject(project)
        if (stateStore.listRevisions(project.id).length === 0) {
          saveLocalRevision(project, '载入项目', 'inference')
        }
        stateStore.touchRecentProject(project.id)
        return projectForRenderer(project, api)
      } catch (error) {
        const cached = stateStore.getProject(id)
        if (cached) return projectForRenderer(cached, api)
        throw error
      }
    }
    case 'job': {
      const id = requiredText(payload, 'id', 128)
      try {
        const mode = stateStore.runtimeMode(id)
        const api = activeApi(id)
        const job = await api.job(id)
        stateStore.saveJob(job)
        if (job.status === 'completed' && job.projectId) {
          const project = await api.project(job.projectId)
          stateStore.saveProject(project)
          if (stateStore.listRevisions(project.id).length === 0) {
            saveLocalRevision(project, '初始转录', 'inference')
          }
          stateStore.touchRecentProject(project.id)
          stateStore.setRuntimeMode(mode, project.id)
        }
        return job
      } catch (error) {
        const cached = stateStore.getJob(id)
        if (cached) return cached
        throw error
      }
    }
    case 'cancel-job': {
      const id = requiredText(payload, 'id', 128)
      const job = await activeApi(id).cancelJob(id)
      stateStore.saveJob(job)
      return job
    }
    case 'submit-link': {
      const mode = stateStore.runtimeMode()
      const requestedSession =
        payload.mediaSession &&
        typeof payload.mediaSession === 'object' &&
        !Array.isArray(payload.mediaSession)
          ? (payload.mediaSession as Record<string, unknown>)
          : undefined
      const browser = requestedSession?.browser
      const mediaSession: MediaBrowserSession | undefined =
        typeof browser === 'string' &&
        mediaBrowserIds.has(browser as MediaBrowserSession['browser'])
          ? { browser: browser as MediaBrowserSession['browser'] }
          : undefined
      if (requestedSession && !mediaSession) {
        throw new Error('Invalid media browser session')
      }
      if (mediaSession && mode !== 'local') {
        throw new Error('浏览器登录态仅可用于本机媒体获取')
      }
      const sessionToken =
        typeof requestedSession?.sessionToken === 'string'
          ? requestedSession.sessionToken
          : undefined
      const cookieFile = sessionToken
        ? mediaSessions?.consumeCookieSession(sessionToken)
        : undefined
      if (sessionToken && !cookieFile) {
        throw new Error('网页登录授权已过期，请重新登录')
      }
      if (mediaSession && cookieFile) mediaSession.cookieFile = cookieFile
      const job = await activeApi().submitLink(
        requiredText(payload, 'url'),
        authorizedMediaConsent(payload.authorizedMedia),
        mediaSession,
      )
      stateStore.saveJob(job)
      stateStore.setRuntimeMode(mode, job.id)
      return job
    }
    case 'select-local-file': {
      const kind = requiredText(payload, 'kind', 16)
      if (kind !== 'media' && kind !== 'score') {
        throw new Error('Invalid local file kind')
      }
      const source = (payload.source ?? 'default') as DesktopLocalFileSource
      if (
        source !== 'default' &&
        !desktopMediaSources.includes(source as DesktopMediaSource)
      ) {
        throw new Error('Invalid local file source')
      }
      const directoryKey = `${kind}:${source}`
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title:
          source !== 'default'
            ? `选择${desktopMediaSourceLabels[source]}下载文件`
            : kind === 'media'
            ? '选择音频或视频'
            : '选择吉他谱图片或 PDF',
        defaultPath:
          localFileDirectories.get(directoryKey) ??
          localFileDefaultPath({
            kind,
            source,
            platform: process.platform,
            homePath: app.getPath('home'),
            musicPath: app.getPath('music'),
            documentsPath: app.getPath('documents'),
            downloadsPath: app.getPath('downloads'),
            appDataPath: app.getPath('appData'),
            pathExists: existsSync,
          }),
        properties: ['openFile', 'showHiddenFiles'],
        filters:
          kind === 'media'
            ? [
                {
                  name: '支持的音频、视频和加密媒体',
                  extensions: [
                    'mp3',
                    'wav',
                    'flac',
                    'ape',
                    'm4a',
                    'aac',
                    'ogg',
                    'mp4',
                    'mov',
                    'webm',
                    'qmc0',
                    'qmc2',
                    'qmc3',
                    'qmcflac',
                    'qmcogg',
                    'tkm',
                    'tm0',
                    'tm2',
                    'tm3',
                    'tm6',
                    'bkcmp3',
                    'bkcflac',
                    'mflac',
                    'mflac0',
                    'mgg',
                    'mgg0',
                    'mgg1',
                    'ncm',
                    'kwm',
                    'kgg',
                    'kgm',
                    'kgma',
                    'vpr',
                  ],
                },
                { name: '所有文件', extensions: ['*'] },
              ]
            : [
                {
                  name: '吉他谱图片或 PDF',
                  extensions: ['png', 'jpg', 'jpeg', 'webp', 'pdf'],
                },
                { name: '所有文件', extensions: ['*'] },
              ],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      const path = selection.filePaths[0]
      localFileDirectories.set(directoryKey, dirname(path))
      try {
        const fileSize = statSync(path).size
        if (fileSize <= 0 || fileSize > maximumUploadBytes) {
          throw new Error('所选文件为空或超过 500 MB')
        }
        const selected: DesktopSelectedFile = {
          name: basename(path),
          mediaType: selectedFileMediaType(path),
          bytes: readFileSync(path),
        }
        return selected
      } catch (error) {
        throw fileAccessError(error)
      }
    }
    case 'import-media-key-database': {
      const provider = requiredText(payload, 'provider', 16)
      if (provider !== 'kuwo' && provider !== 'kugou') {
        throw new Error('Invalid media key provider')
      }
      const typedProvider = provider as DesktopMediaKeyProvider
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title:
          provider === 'kuwo'
            ? '导入酷我 KWM v2 密钥库'
            : '导入酷狗 KGM v5 密钥库',
        defaultPath:
          localFileDirectories.get(`media-key:${provider}`) ??
          app.getPath('downloads'),
        properties: ['openFile', 'showHiddenFiles'],
        filters: [
          {
            name:
              provider === 'kuwo'
                ? '酷我 MMKV 密钥库'
                : '酷狗 MMKV 或 SQLite 密钥库',
            extensions: ['mmkv', 'db', 'sqlite', 'dat'],
          },
          { name: '所有文件', extensions: ['*'] },
        ],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      const path = selection.filePaths[0]
      localFileDirectories.set(`media-key:${provider}`, dirname(path))
      try {
        const size = statSync(path).size
        if (size <= 0 || size > maximumMediaKeyDatabaseBytes) {
          throw new Error('所选密钥库为空或超过 128 MB')
        }
        return localApi.importMediaKeyDatabase(typedProvider, {
          name: basename(path),
          mediaType: 'application/octet-stream',
          bytes: readFileSync(path),
        })
      } catch (error) {
        throw fileAccessError(error)
      }
    }
    case 'submit-upload': {
      const bytes = payload.bytes
      if (!(bytes instanceof Uint8Array) || bytes.byteLength > maximumUploadBytes) {
        throw new Error('Invalid or oversized desktop upload')
      }
      const upload: DesktopUpload = {
        name: requiredText(payload, 'name', 255),
        mediaType:
          typeof payload.mediaType === 'string'
            ? payload.mediaType.slice(0, 128)
            : 'application/octet-stream',
        bytes,
      }
      const consent = authorizedMediaConsent(payload.authorizedMedia)
      const mode = stateStore.runtimeMode()
      if (consent?.credentialSession && mode !== 'local') {
        throw new Error('QQ 音乐网页登录授权仅可用于本机媒体处理')
      }
      const credentialFile = consent?.credentialSession
        ? mediaSessions?.consumeCredentialSession(consent.credentialSession)
        : undefined
      if (consent?.credentialSession && !credentialFile) {
        throw new Error('QQ 音乐网页登录授权已过期，请重新登录')
      }
      const job = await activeApi().submitUpload(
        upload,
        consent,
        credentialFile,
      )
      stateStore.saveJob(job)
      stateStore.setRuntimeMode(mode, job.id)
      return job
    }
    case 'submit-score': {
      const pagesValue = payload.pages
      if (!Array.isArray(pagesValue) || pagesValue.length === 0 || pagesValue.length > 24) {
        throw new Error('Invalid score pages')
      }
      let totalBytes = 0
      const pages = pagesValue.map((value) => {
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
          throw new Error('Invalid score page')
        }
        const page = value as Record<string, unknown>
        if (!(page.bytes instanceof Uint8Array)) {
          throw new Error('Invalid score page bytes')
        }
        totalBytes += page.bytes.byteLength
        return {
          name: requiredText(page, 'name', 255),
          mediaType:
            typeof page.mediaType === 'string'
              ? page.mediaType.slice(0, 128)
              : 'image/png',
          bytes: page.bytes,
        }
      })
      if (totalBytes > 256 * 1024 * 1024) {
        throw new Error('曲谱文件过大')
      }
      const upload: DesktopScoreUpload = {
        name: requiredText(payload, 'name', 255),
        pages,
      }
      const mode = stateStore.runtimeMode()
      const api = activeApi()
      const project = await api.submitScore(upload)
      stateStore.saveProject(projectForStorage(project, api))
      if (stateStore.listRevisions(project.id).length === 0) {
        saveLocalRevision(projectForStorage(project, api), '图片曲谱识别', 'import')
      }
      stateStore.touchRecentProject(project.id)
      stateStore.setRuntimeMode(mode, project.id)
      return projectForRenderer(project, api)
    }
    case 'local-projects':
      return stateStore.localProjects()
    case 'remove-project': {
      const id = requiredText(payload, 'id', 128)
      await localApi.removeProject(id)
      stateStore.removeProject(id)
      return { removed: true }
    }
    case 'recent-projects':
      return stateStore.recentProjects()
    case 'project-revisions':
      return stateStore.listRevisions(requiredText(payload, 'projectId', 128))
    case 'save-project-revision':
      {
        const project = migrateStudioProject(payload.project)
        const api = activeApi(project.id)
        const revision = saveLocalRevision(
          projectForStorage(project, api),
          requiredText(payload, 'label', 80),
          (typeof payload.kind === 'string'
            ? payload.kind
            : 'manual') as ProjectRevisionKind,
          typeof payload.expectedParentRevisionId === 'string'
            ? payload.expectedParentRevisionId
            : payload.expectedParentRevisionId === null
              ? null
              : undefined,
        )
        return {
          ...revision,
          project: projectForRenderer(revision.project, api),
        }
      }
    case 'restore-project-revision': {
      const projectId = requiredText(payload, 'projectId', 128)
      const revision = stateStore.getRevision(
        requiredText(payload, 'revisionId', 128),
      )
      if (!revision || revision.projectId !== projectId) {
        throw new Error('Project revision was not found')
      }
      const restored = saveLocalRevision(
        revision.project,
        `恢复：${revision.label}`.slice(0, 80),
        'restore',
      )
      return {
        ...restored,
        project: projectForRenderer(restored.project, activeApi(projectId)),
      }
    }
    case 'runtime-status': {
      const projectId =
        typeof payload.projectId === 'string' ? payload.projectId : undefined
      const requestedMode = stateStore.runtimeMode(projectId)
      const providerMode =
        requestedMode === 'cloud' && remoteApi ? 'cloud' : 'local'
      const localCapabilities = await localApi.capabilities()
      let cloudCapabilities: Awaited<
        ReturnType<StudioApiTransport['capabilities']>
      > | undefined
      let cloudError = cloudConfigurationError
      if (remoteApi) {
        try {
          cloudCapabilities = await remoteApi.capabilities()
        } catch (error) {
          cloudError =
            error instanceof Error ? error.message : 'Cloud API is unavailable'
        }
      }
      const capabilities =
        providerMode === 'cloud' && cloudCapabilities
          ? cloudCapabilities
          : localCapabilities
      const activeModel = modelPackages.activeManifest()
      const transcription = transcriptionManager.status()
      const status: DesktopRuntimeStatus = {
        mode: 'desktop-local',
        providerMode,
        cloudAvailable: Boolean(remoteApi),
        cloudReady: Boolean(cloudCapabilities),
        ...(stateStore.cloudApiRoot() || process.env.GTS_CLOUD_API_ROOT
          ? {
              cloudApiRoot:
                stateStore.cloudApiRoot() ??
                process.env.GTS_CLOUD_API_ROOT,
            }
          : {}),
        uploadsAudio: providerMode === 'cloud',
        apiReady: providerMode === 'local' || Boolean(cloudCapabilities),
        storageReady: true,
        workerReady: capabilities.worker,
        storageScope: 'application-data',
        estimatedSeconds:
          providerMode === 'cloud'
            ? { minimum: 20, maximum: 180 }
            : { minimum: 60, maximum: 600 },
        resourceNote:
          providerMode === 'cloud'
            ? '需要稳定网络，音频将上传到已配置服务'
            : activeModel
              ? '使用本机 CPU/GPU 与已安装模型'
              : '可编辑项目；安装离线模型包后执行本地推理',
        transcription,
        ...(cloudError ? { cloudError } : {}),
        ...(modelPackages.recoveryWarning()
          ? { modelWarning: modelPackages.recoveryWarning() }
          : {}),
        ...(activeModel
          ? {
              activeModel: {
                id: activeModel.id,
                version: activeModel.version,
                releaseChannel:
                  activeModel.releaseChannel ?? 'research',
              },
            }
          : {}),
      }
      return status
    }
    case 'set-runtime-mode': {
      const mode = requiredText(payload, 'mode', 16)
      if (mode !== 'local' && mode !== 'cloud') {
        throw new Error('Invalid runtime mode')
      }
      if (mode === 'cloud' && !remoteApi) {
        throw new Error('Cloud inference is not configured')
      }
      const projectId =
        typeof payload.projectId === 'string' ? payload.projectId : undefined
      stateStore.setRuntimeMode(mode, projectId)
      return { mode }
    }
    case 'configure-cloud-runtime': {
      const apiRoot = requiredText(payload, 'apiRoot', 2048).trim()
      const configured = new RemoteApiService(apiRoot)
      await configured.capabilities()
      remoteApi = configured
      cloudConfigurationError = undefined
      stateStore.setCloudApiRoot(apiRoot)
      const projectId =
        typeof payload.projectId === 'string' ? payload.projectId : undefined
      stateStore.setRuntimeMode('cloud', projectId)
      return { configured: true }
    }
    case 'configure-huggingface-oauth': {
      const clientId = requiredText(payload, 'clientId', 256).trim()
      if (!/^[A-Za-z0-9._-]+$/u.test(clientId)) {
        throw new Error('Hugging Face OAuth Client ID 格式无效')
      }
      stateStore.setHuggingFaceOAuthClientId(clientId)
      huggingFaceOAuth?.resetToken()
      return { configured: true }
    }
    case 'set-transcription-provider': {
      openedTranscriptionUrls.clear()
      latestTranscriptionAuthorizationUrl = undefined
      const providerId = requiredText(payload, 'providerId', 32)
      if (
        providerId !== 'basic-pitch' &&
        providerId !== 'muscriptor' &&
        providerId !== 'guitar-specialist' &&
        providerId !== 'multitrack-research'
      ) {
        throw new Error('Invalid transcription provider')
      }
      const selectionMode =
        typeof payload.selectionMode === 'string'
          ? payload.selectionMode
          : undefined
      if (
        selectionMode !== undefined &&
        selectionMode !== 'auto' &&
        selectionMode !== 'manual'
      ) {
        throw new Error('Invalid transcription selection mode')
      }
      const variant =
        typeof payload.variant === 'string' ? payload.variant : undefined
      if (
        variant !== undefined &&
        variant !== 'small' &&
        variant !== 'medium' &&
        variant !== 'large'
      ) {
        throw new Error('Invalid transcription model variant')
      }
      await transcriptionManager.prepareAndSetPreference({
        providerId: providerId as TranscriptionProviderId,
        ...(selectionMode ? { selectionMode } : {}),
        ...(variant
          ? { variant: variant as TranscriptionModelVariant }
          : {}),
        licenseAccepted: payload.licenseAccepted === true,
      })
      await restartLocalApi()
      return transcriptionManager.status()
    }
    case 'cancel-transcription-setup': {
      const cancelled = transcriptionManager.cancelPreparation()
      if (cancelled) {
        latestTranscriptionAuthorizationUrl = undefined
        openedTranscriptionUrls.clear()
      }
      return { cancelled }
    }
    case 'open-transcription-authorization':
      if (!latestTranscriptionAuthorizationUrl) {
        throw new Error('当前没有等待完成的浏览器授权')
      }
      await shell.openExternal(latestTranscriptionAuthorizationUrl)
      return { opened: true }
    case 'import-transcription-weights': {
      const variant = requiredText(payload, 'variant', 16)
      if (
        variant !== 'small' &&
        variant !== 'medium' &&
        variant !== 'large'
      ) {
        throw new Error('Invalid transcription model variant')
      }
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title: `导入 MuScriptor ${variant} 权重`,
        properties: ['openFile'],
        filters: [
          { name: 'MuScriptor weights', extensions: ['safetensors'] },
        ],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      const manifest = await transcriptionManager.importWeights(
        selection.filePaths[0],
        variant,
        payload.licenseAccepted === true,
      )
      return manifest
    }
    case 'open-transcription-provider-page': {
      const variant =
        typeof payload.variant === 'string' ? payload.variant : ''
      const url =
        variant === 'small' || variant === 'medium' || variant === 'large'
          ? `https://huggingface.co/MuScriptor/muscriptor-${variant}`
          : 'https://huggingface.co/MuScriptor'
      await shell.openExternal(url)
      return { opened: true }
    }
    case 'model-packages':
      return modelPackages.listInstalled()
    case 'install-model-package': {
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title: '安装模型包',
        properties: ['openDirectory'],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      return modelPackages.installFromDirectory(selection.filePaths[0])
    }
    case 'download-model-package':
      return modelPackages.installFromUrl(
        requiredText(payload, 'manifestUrl', 2048),
      )
    case 'activate-model-package': {
      const state = await modelPackages.activate(
        requiredText(payload, 'id', 64),
        requiredText(payload, 'version', 64),
      )
      await restartLocalApi()
      return state
    }
    case 'rollback-model-package': {
      const state = await modelPackages.rollback()
      await restartLocalApi()
      return state
    }
    case 'remove-model-package':
      modelPackages.remove(
        requiredText(payload, 'id', 64),
        requiredText(payload, 'version', 64),
      )
      return { removed: true }
    case 'export-project': {
      const id = requiredText(payload, 'id', 128)
      const project =
        stateStore.getProject(id) ?? (await localApi.project(id))
      const destination = await dialog.showSaveDialog(mainWindow!, {
        title: '导出项目',
        defaultPath: `${project.title.replaceAll(/[^\p{L}\p{N}._-]+/gu, '-')}.gts.json`,
        filters: [{ name: 'Guitar Tab Studio project', extensions: ['json'] }],
      })
      if (destination.canceled || !destination.filePath) return null
      writeFileSync(
        destination.filePath,
        `${JSON.stringify(project, null, 2)}\n`,
        'utf8',
      )
      return { exported: true }
    }
    case 'import-project': {
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title: '导入项目',
        properties: ['openFile'],
        filters: [{ name: 'Guitar Tab Studio project', extensions: ['json'] }],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      const project = migrateStudioProject(
        JSON.parse(readFileSync(selection.filePaths[0], 'utf8')),
      )
      saveLocalRevision(project, '导入项目', 'import')
      stateStore.setRuntimeMode('local', project.id)
      return projectForRenderer(project, localApi)
    }
    case 'create-backup': {
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title: '选择备份保存位置',
        properties: ['openDirectory', 'createDirectory'],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      return { name: await maintenance.createBackup(selection.filePaths[0]) }
    }
    case 'restore-backup': {
      const selection = await dialog.showOpenDialog(mainWindow!, {
        title: '选择要恢复的备份',
        properties: ['openDirectory'],
      })
      if (selection.canceled || !selection.filePaths[0]) return null
      await localApi.stop()
      try {
        await maintenance.restoreBackup(selection.filePaths[0])
      } finally {
        await restartLocalApi()
      }
      return { restored: true }
    }
    case 'clear-cache': {
      await localApi.stop()
      let removedBytes = 0
      try {
        removedBytes = maintenance.clearCache()
      } finally {
        await restartLocalApi()
      }
      return { removedBytes }
    }
    case 'open-full-disk-access-settings':
      if (process.platform !== 'darwin') {
        throw new Error('完全磁盘访问权限设置仅适用于 macOS')
      }
      await shell.openExternal(
        'x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles',
      )
      return { opened: true }
    case 'storage-cleanup-status':
      return localApi.storageCleanupStatus()
    case 'set-storage-cleanup-config':
      return localApi.updateStorageCleanup(payload as unknown as StorageCleanupConfig)
    case 'run-storage-cleanup':
      return localApi.runStorageCleanup()
    case 'export-diagnostics': {
      const destination = await dialog.showSaveDialog(mainWindow!, {
        title: '导出诊断信息',
        defaultPath: `guitar-tab-studio-diagnostics-${new Date()
          .toISOString()
          .slice(0, 10)}.json`,
        filters: [{ name: 'JSON', extensions: ['json'] }],
      })
      if (destination.canceled || !destination.filePath) return null
      const runtime = await invoke({ method: 'runtime-status' })
      const diagnostics = {
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        application: {
          version: app.getVersion(),
          packaged: isPackagedRuntime(),
          platform: process.platform,
          arch: process.arch,
        },
        runtime,
        storageCleanup: await localApi.storageCleanupStatus(),
        storage: stateStore.diagnostics(),
        modelPackages: modelPackages.listInstalled().map((item) => ({
          id: item.manifest.id,
          version: item.manifest.version,
          active: item.active,
          createdAt: item.manifest.createdAt,
          runtime: item.manifest.runtime,
        })),
      }
      writeFileSync(
        destination.filePath,
        `${JSON.stringify(diagnostics, null, 2)}\n`,
        { encoding: 'utf8', mode: 0o600 },
      )
      return { exported: true }
    }
    case 'application-settings':
      return applicationSettings()
    case 'set-language': {
      const preference = requiredText(payload, 'preference', 16)
      if (
        preference !== 'system' &&
        preference !== 'zh-CN' &&
        preference !== 'en-US'
      ) {
        throw new Error('Invalid application language')
      }
      return setLanguagePreference(preference)
    }
    case 'detect-default-media-browser':
      return detectDefaultMediaBrowser()
    case 'authorize-media-session': {
      if (!mediaSessions) throw new Error('网页登录服务尚未就绪')
      const browserId = requiredText(payload, 'browser', 32)
      if (!mediaBrowserIds.has(browserId as MediaBrowserSession['browser'])) {
        throw new Error('不支持该浏览器登录态')
      }
      const detected = await detectDefaultMediaBrowser()
      if (!detected.browser || detected.browser.id !== browserId) {
        throw new Error('默认浏览器已变化，请重新检测')
      }
      return mediaSessions.authorizeMedia(
        requiredText(payload, 'url', 4096),
        detected.browser,
      )
    }
    case 'authorize-qqmusic':
      if (!mediaSessions) throw new Error('QQ 音乐授权服务尚未就绪')
      return mediaSessions.authorizeQQMusic()
    case 'open-external-url': {
      const url = new URL(requiredText(payload, 'url', 4096))
      if (url.protocol !== 'https:' && url.protocol !== 'http:') {
        throw new Error('Only HTTP media links can be opened externally')
      }
      await shell.openExternal(url.toString())
      return { opened: true }
    }
    case 'update-status':
      return updater?.status() ?? null
    case 'update-download':
      if (!updater) throw new Error('Desktop updater is not ready')
      return updater.download()
    case 'update-pause':
      if (!updater) throw new Error('Desktop updater is not ready')
      return updater.pause()
    case 'update-resume':
      if (!updater) throw new Error('Desktop updater is not ready')
      return updater.resume()
    case 'update-cancel':
      if (!updater) throw new Error('Desktop updater is not ready')
      return updater.cancel()
    case 'update-dismiss':
      return updater?.dismiss() ?? null
    case 'update-install':
      if (!updater) throw new Error('Desktop updater is not ready')
      updater.install()
      return { installing: true }
    case 'set-background-job-state': {
      const active = payload.active === true
      backgroundJobState = active
        ? {
            active: true,
            ...(typeof payload.jobId === 'string'
              ? { jobId: payload.jobId.slice(0, 128) }
              : {}),
            ...(typeof payload.progress === 'number' &&
            Number.isFinite(payload.progress)
              ? {
                  progress: Math.max(
                    0,
                    Math.min(100, payload.progress),
                  ),
                }
              : {}),
            ...(typeof payload.stageLabel === 'string'
              ? { stageLabel: payload.stageLabel.slice(0, 160) }
              : {}),
          }
        : { active: false }
      if (mainWindow?.isVisible()) {
        if (!active) destroyTray()
      } else if (active || tray) {
        ensureTray()
      }
      updateTrayMenu()
      return backgroundJobState
    }
  }
}

function menuLabels(language = applicationSettings().language) {
  if (language === 'en-US') {
    return {
      file: 'File',
      openMedia: 'Open Media...',
      importProject: 'Import Project...',
      close: 'Close Window',
      quit: 'Quit Guitar Tab Studio',
      edit: 'Edit',
      undo: 'Undo',
      redo: 'Redo',
      cut: 'Cut',
      copy: 'Copy',
      paste: 'Paste',
      selectAll: 'Select All',
      view: 'View',
      reload: 'Reload',
      developerTools: 'Toggle Developer Tools',
      actualSize: 'Actual Size',
      zoomIn: 'Zoom In',
      zoomOut: 'Zoom Out',
      fullscreen: 'Toggle Full Screen',
      settings: 'Settings',
      openSettings: 'Open Settings...',
      language: 'Language',
      followSystem: 'Follow System',
      chinese: 'Simplified Chinese',
      english: 'English',
      help: 'Help',
      checkUpdates: 'Check for Updates',
      currentVersion: `Version ${app.getVersion()}`,
      showApplication: 'Open Guitar Tab Studio',
      backgroundTask: 'Analysis running in background',
    }
  }
  return {
    file: '文件',
    openMedia: '打开媒体…',
    importProject: '导入项目…',
    close: '关闭窗口',
    quit: '退出 Guitar Tab Studio',
    edit: '编辑',
    undo: '撤销',
    redo: '重做',
    cut: '剪切',
    copy: '复制',
    paste: '粘贴',
    selectAll: '全选',
    view: '视图',
    reload: '重新加载',
    developerTools: '切换开发者工具',
    actualSize: '实际大小',
    zoomIn: '放大',
    zoomOut: '缩小',
    fullscreen: '切换全屏',
    settings: '设置',
    openSettings: '打开设置…',
    language: '语言',
    followSystem: '跟随系统',
    chinese: '简体中文',
    english: 'English',
    help: '帮助',
    checkUpdates: '检查更新',
    currentVersion: `当前版本 ${app.getVersion()}`,
    showApplication: '打开 Guitar Tab Studio',
    backgroundTask: '分析任务正在后台运行',
  }
}

function showMainWindow(): void {
  if (!mainWindow) {
    void createWindow()
    return
  }
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
}

function updateTrayMenu(): void {
  if (!tray) return
  const labels = menuLabels()
  const progress =
    typeof backgroundJobState.progress === 'number'
      ? ` · ${Math.round(backgroundJobState.progress)}%`
      : ''
  tray.setToolTip(
    backgroundJobState.active
      ? `${labels.backgroundTask}${progress}`
      : 'Guitar Tab Studio',
  )
  tray.setContextMenu(
    Menu.buildFromTemplate([
      {
        label: labels.showApplication,
        click: showMainWindow,
      },
      ...(backgroundJobState.active
        ? [
            {
              label: `${backgroundJobState.stageLabel ?? labels.backgroundTask}${progress}`,
              enabled: false,
            } satisfies MenuItemConstructorOptions,
          ]
        : []),
      { type: 'separator' },
      {
        label: labels.quit,
        click: () => app.quit(),
      },
    ]),
  )
}

function ensureTray(): void {
  if (process.platform !== 'win32') return
  if (!tray) {
    tray = new Tray(desktopIconPath())
    tray.on('click', showMainWindow)
  }
  updateTrayMenu()
}

function destroyTray(): void {
  tray?.destroy()
  tray = undefined
}

function setLanguagePreference(
  preference: AppLanguagePreference,
): DesktopApplicationSettings {
  if (!stateStore) throw new Error('Desktop state is unavailable')
  stateStore.setLanguagePreference(preference)
  const settings = applicationSettings()
  installMenu()
  updateTrayMenu()
  mainWindow?.webContents.send('studio:language-changed', settings)
  return settings
}

function installMenu(): void {
  const settings = applicationSettings()
  const labels = menuLabels(settings.language)
  const template: MenuItemConstructorOptions[] = [
    {
      label: labels.file,
      submenu: [
        {
          label: labels.openMedia,
          accelerator: 'CmdOrCtrl+O',
          click: () => mainWindow?.webContents.send('studio:open-media'),
        },
        {
          label: labels.importProject,
          click: () =>
            void invoke({ method: 'import-project' }).then((project) => {
              if (
                project &&
                typeof project === 'object' &&
                'id' in project &&
                typeof project.id === 'string'
              ) {
                mainWindow?.webContents.send('studio:open-project', project.id)
              }
            }),
        },
        { type: 'separator' },
        {
          role: process.platform === 'darwin' ? 'close' : 'quit',
          label: process.platform === 'darwin' ? labels.close : labels.quit,
        },
      ],
    },
    {
      label: labels.edit,
      submenu: [
        { role: 'undo', label: labels.undo },
        { role: 'redo', label: labels.redo },
        { type: 'separator' },
        { role: 'cut', label: labels.cut },
        { role: 'copy', label: labels.copy },
        { role: 'paste', label: labels.paste },
        { type: 'separator' },
        { role: 'selectAll', label: labels.selectAll },
      ],
    },
    {
      label: labels.view,
      submenu: [
        { role: 'reload', label: labels.reload },
        { role: 'toggleDevTools', label: labels.developerTools },
        { type: 'separator' },
        { role: 'resetZoom', label: labels.actualSize },
        { role: 'zoomIn', label: labels.zoomIn },
        { role: 'zoomOut', label: labels.zoomOut },
        { type: 'separator' },
        { role: 'togglefullscreen', label: labels.fullscreen },
      ],
    },
    {
      label: labels.settings,
      submenu: [
        {
          label: labels.openSettings,
          accelerator: 'CmdOrCtrl+,',
          click: () => mainWindow?.webContents.send('studio:open-settings'),
        },
        { type: 'separator' },
        {
          label: labels.language,
          submenu: [
            {
              label: labels.followSystem,
              type: 'radio',
              checked: settings.languagePreference === 'system',
              click: () => setLanguagePreference('system'),
            },
            {
              label: labels.chinese,
              type: 'radio',
              checked: settings.languagePreference === 'zh-CN',
              click: () => setLanguagePreference('zh-CN'),
            },
            {
              label: labels.english,
              type: 'radio',
              checked: settings.languagePreference === 'en-US',
              click: () => setLanguagePreference('en-US'),
            },
          ],
        },
      ],
    },
    {
      label: labels.help,
      submenu: [
        {
          label: labels.checkUpdates,
          click: () => updater?.checkManually(),
        },
        { type: 'separator' },
        {
          label: labels.currentVersion,
          enabled: false,
        },
      ],
    },
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

async function createWindow(): Promise<void> {
  if (!stateStore) throw new Error('Desktop state is unavailable')
  const saved = visibleWindowState(stateStore.getWindowState())
  const window = new BrowserWindow({
    x: saved.x,
    y: saved.y,
    width: saved.width,
    height: saved.height,
    minWidth: 900,
    minHeight: 640,
    show: false,
    backgroundColor: '#f2f3f1',
    ...(process.platform === 'win32' ? { icon: desktopIconPath() } : {}),
    webPreferences: {
      preload: resolve(moduleDirectory, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  })
  mainWindow = window
  if (saved.maximized) window.maximize()
  window.once('ready-to-show', () => window.show())
  window.on('close', (event) => {
    if (stateStore) {
      const bounds = window.getNormalBounds()
      stateStore.saveWindowState({
        ...bounds,
        maximized: window.isMaximized(),
      })
    }
    if (!quitting && backgroundJobState.active) {
      event.preventDefault()
      window.hide()
      ensureTray()
    }
  })
  window.on('closed', () => {
    if (mainWindow === window) mainWindow = undefined
  })
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event, url) => {
    const allowedOrigin = process.env.GTS_DESKTOP_RENDERER_URL
    if (allowedOrigin) {
      try {
        if (new URL(url).origin === new URL(allowedOrigin).origin) return
      } catch {
        // Invalid navigation targets are denied below.
      }
    } else {
      const entryUrl = pathToFileURL(runtimePaths().webEntry)
      try {
        const target = new URL(url)
        if (
          target.protocol === 'file:' &&
          target.pathname === entryUrl.pathname
        ) {
          return
        }
      } catch {
        // Invalid navigation targets are denied below.
      }
    }
    event.preventDefault()
  })

  const rendererUrl = process.env.GTS_DESKTOP_RENDERER_URL
  if (rendererUrl) {
    await window.loadURL(rendererUrl)
  } else {
    await window.loadFile(runtimePaths().webEntry)
  }
}

async function startDesktop(): Promise<void> {
  stateStore = new DesktopStateStore(
    resolve(app.getPath('userData'), 'state', 'desktop.sqlite'),
  )
  mediaSessions = new DesktopMediaSessions(
    resolve(app.getPath('userData'), 'runtime', 'media-sessions'),
    () => mainWindow,
  )
  await mediaSessions.initialize()
  modelPackages = new ModelPackageManager(
    resolve(app.getPath('userData'), 'models'),
    app.getVersion(),
  )
  huggingFaceOAuth = new HuggingFaceOAuth(
    resolve(app.getPath('userData'), 'credentials', 'huggingface-oauth.bin'),
    () =>
      stateStore?.huggingFaceOAuthClientId() ??
      process.env.GTS_HF_OAUTH_CLIENT_ID,
    sendTranscriptionProgress,
    {
      available: () => safeStorage.isEncryptionAvailable(),
      encrypt: (value) => safeStorage.encryptString(value),
      decrypt: (value) => safeStorage.decryptString(value),
    },
    (input, init) =>
      net.fetch(input instanceof URL ? input.toString() : input, init),
  )
  transcriptionManager = new DesktopTranscriptionManager(
    resolve(app.getPath('userData'), 'transcription-providers'),
    resolve(app.getPath('userData'), 'model-cache', 'huggingface', 'hub'),
    stateStore,
    () =>
      (!isPackagedRuntime() ? runtimePaths().workerCommand : undefined) ??
      modelPackages?.resolveActiveFile('worker') ??
      runtimePaths().workerCommand,
    sendTranscriptionProgress,
    () =>
      modelPackages?.resolveActiveFile('transcription-model') ??
      (!isPackagedRuntime() && existsSync(developmentGuitarResearchModel)
        ? developmentGuitarResearchModel
        : undefined),
    (signal) => huggingFaceOAuth!.accessToken(signal),
    () =>
      stateStore?.huggingFaceOAuthClientId() ??
      process.env.GTS_HF_OAUTH_CLIENT_ID,
    modelDownloadNetworkEnvironment,
  )
  const cloudApiRoot =
    stateStore.cloudApiRoot() ?? process.env.GTS_CLOUD_API_ROOT
  if (cloudApiRoot) {
    try {
      remoteApi = new RemoteApiService(cloudApiRoot)
    } catch (error) {
      remoteApi = undefined
      cloudConfigurationError =
        error instanceof Error ? error.message : 'Invalid cloud API'
    }
  }
  maintenance = new DesktopMaintenance(
    app.getVersion(),
    stateStore,
    resolve(app.getPath('userData'), 'runtime'),
    resolve(app.getPath('userData'), 'cache'),
  )
  updater = new DesktopUpdater(() => mainWindow, isPackagedRuntime())
  await restartLocalApi()
  ipcMain.handle(
    'studio:invoke',
    (event, request: DesktopInvokeRequest) => {
      if (!mainWindow || event.sender.id !== mainWindow.webContents.id) {
        throw new Error('Desktop IPC sender is not trusted')
      }
      return invoke(request)
    },
  )
  session.defaultSession.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      const mediaTypes =
        'mediaTypes' in details ? details.mediaTypes ?? [] : []
      const trustedAudioRequest = Boolean(
        mainWindow &&
          webContents.id === mainWindow.webContents.id &&
          permission === 'media' &&
          mediaTypes.includes('audio') &&
          !mediaTypes.includes('video'),
      )
      callback(trustedAudioRequest)
    },
  )
  installMenu()
  updater.start()
  await createWindow()
  if (process.env.GTS_DESKTOP_SMOKE === '1') {
    const smokeWindow = mainWindow
    if (!smokeWindow) throw new Error('Desktop window was not created')
    await new Promise((resolveWait) => setTimeout(resolveWait, 1200))
    const scoreBridgeError =
      await smokeWindow.webContents.executeJavaScript(`
        window.gtsDesktop.invoke({
          method: 'submit-score',
          payload: {},
        }).then(
          () => '',
          (error) => String(error?.message ?? error),
        )
      `)
    if (!String(scoreBridgeError).includes('Invalid score pages')) {
      throw new Error('Desktop score import IPC method is not available')
    }
    smokeWindow.webContents.send('studio:update-progress', {
      phase: 'downloading',
      version: '0.1.0-preview.13',
      percent: 37.4,
      transferredBytes: 225 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 10 * 1024 ** 2,
    } satisfies DesktopUpdateProgress)
    await new Promise((resolveWait) => setTimeout(resolveWait, 100))
    const updateProgressState =
      await smokeWindow.webContents.executeJavaScript(`
        (() => {
          const panel = document.querySelector('.update-progress-panel')
          const bounds = panel?.getBoundingClientRect()
          return {
            visible: Boolean(
              panel &&
              bounds &&
              bounds.width > 300 &&
              bounds.height > 100 &&
              bounds.right <= window.innerWidth &&
              bounds.bottom <= window.innerHeight
            ),
            percentage: panel?.querySelector('output')?.textContent?.trim(),
            metrics: panel?.querySelector('.update-progress-metrics')
              ?.textContent?.replace(/\\s+/g, ' ').trim(),
            ariaValue: panel?.querySelector('[role="progressbar"]')
              ?.getAttribute('aria-valuenow'),
            actions: Array.from(panel?.querySelectorAll('button') ?? [])
              .map((button) => button.getAttribute('aria-label')),
          }
        })()
      `)
    if (
      !updateProgressState?.visible ||
      updateProgressState.percentage !== '37%' ||
      updateProgressState.ariaValue !== '37' ||
      !String(updateProgressState.metrics).includes('225.0 MB / 600.0 MB') ||
      !String(updateProgressState.metrics).includes('10.0 MB/s') ||
      !updateProgressState.actions?.includes('暂停下载') ||
      !updateProgressState.actions?.includes('取消下载')
    ) {
      throw new Error('Desktop update download progress did not render')
    }
    smokeWindow.webContents.send('studio:update-progress', {
      phase: 'ready',
      version: '0.1.0-preview.13',
      percent: 100,
      transferredBytes: 600 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 0,
    } satisfies DesktopUpdateProgress)
    await new Promise((resolveWait) => setTimeout(resolveWait, 50))
    const updateReadyState =
      await smokeWindow.webContents.executeJavaScript(`
        (() => {
          const dialog = document.querySelector('.update-dialog')
          const bounds = dialog?.getBoundingClientRect()
          return {
            visible: Boolean(
              dialog &&
              bounds &&
              bounds.width > 300 &&
              bounds.height > 180 &&
              bounds.right <= window.innerWidth &&
              bounds.bottom <= window.innerHeight
            ),
            text: dialog?.textContent?.replace(/\\s+/g, ' ').trim(),
          }
        })()
      `)
    if (
      !updateReadyState?.visible ||
      !String(updateReadyState.text).includes('更新已准备好') ||
      !String(updateReadyState.text).includes('重启并安装')
    ) {
      throw new Error('Completed desktop update dialog did not render')
    }
    await smokeWindow.webContents.executeJavaScript(
      `Array.from(document.querySelectorAll('button'))
        .find((button) => ['桌面运行时', 'Desktop runtime']
          .includes(button.getAttribute('aria-label')))
        ?.click()`,
    )
    await new Promise((resolveWait) => setTimeout(resolveWait, 500))
    const panelState = await smokeWindow.webContents.executeJavaScript(`
      new Promise((resolve) => setTimeout(() => {
          const panel = document.querySelector('.runtime-dialog')
          const bounds = panel?.getBoundingClientRect()
          resolve({
            text: document.body.innerText,
            visible: Boolean(
              panel &&
              bounds &&
              bounds.width > 100 &&
              bounds.height > 100 &&
              getComputedStyle(panel).visibility !== 'hidden'
            ),
          })
        }, 50))
    `)
    if (
      !panelState ||
      typeof panelState.text !== 'string' ||
      !panelState.text.includes('运行模型与存储') ||
      panelState.visible !== true
    ) {
      throw new Error('Desktop runtime panel did not load')
    }
    if (process.env.GTS_COMMERCIAL_BUILD === '1') {
      const visibleProviders =
        await smokeWindow.webContents.executeJavaScript(`
          Array.from(
            document.querySelectorAll(
              'select[aria-label="转录方案"] option'
            )
          ).map((option) => option.value)
        `)
      if (
        visibleProviders.length !== 1 ||
        visibleProviders[0] !== 'basic-pitch'
      ) {
        throw new Error('Commercial build exposed a noncommercial provider')
      }
    } else {
      const transcriptionState =
        await smokeWindow.webContents.executeJavaScript(`
          new Promise((resolve) => {
            const scheme = document.querySelector(
              'select[aria-label="转录方案"]'
            )
            if (scheme) {
              scheme.value = 'muscriptor'
              scheme.dispatchEvent(new Event('change', { bubbles: true }))
            }
            setTimeout(() => {
              const setup = document.querySelector('.runtime-transcription-setup')
              const tier = document.querySelector(
                'select[aria-label="模型档位"]'
              )
              const location = document.querySelector(
                '[role="group"][aria-label="执行位置"]'
              )
              resolve({
                visible: Boolean(setup && setup.getBoundingClientRect().height > 100),
                scheme: scheme?.value,
                selection: tier?.value,
                tierEnabled: tier?.disabled === false,
                hasLocation: Boolean(location),
                providers: Array.from(scheme?.options ?? [])
                  .map((option) => option.value),
                hasLicense: Boolean(
                  setup?.querySelector('.runtime-license-confirmation input')
                ),
                actions: Array.from(
                  setup?.querySelectorAll('.runtime-model-actions button') ?? []
                ).map((button) => button.textContent?.trim()),
              })
            }, 50)
          })
        `)
      if (
        !transcriptionState ||
        transcriptionState.visible !== true ||
        transcriptionState.scheme !== 'muscriptor' ||
        ![
          'basic-pitch',
          'muscriptor',
          'guitar-specialist',
          'multitrack-research',
        ].every((provider) =>
          transcriptionState.providers.includes(provider),
        ) ||
        transcriptionState.selection !== 'auto' ||
        transcriptionState.tierEnabled !== true ||
        transcriptionState.hasLocation !== true ||
        transcriptionState.hasLicense !== true ||
        !['官方授权页', '导入权重'].every((label) =>
          transcriptionState.actions.includes(label),
        ) ||
        !transcriptionState.actions.some((label: string) =>
          ['下载并启用', '启用研究融合'].includes(label),
        )
      ) {
        throw new Error('High-accuracy model setup did not load')
      }
      smokeWindow.webContents.send('studio:transcription-progress', {
        phase: 'authorizing',
        progress: 3,
        stageLabel: '请在浏览器完成 Hugging Face 登录',
        userCode: 'ABCD-EFGH',
      } satisfies TranscriptionSetupProgress)
      await new Promise((resolveWait) => setTimeout(resolveWait, 100))
      const progressVisible =
        await smokeWindow.webContents.executeJavaScript(`
          ({
            code: document.querySelector('.runtime-model-progress')
              ?.textContent?.includes('ABCD-EFGH') === true,
            recoveryActions: Array.from(
              document.querySelectorAll('.runtime-model-actions button')
            ).map((button) => button.textContent?.trim()),
          })
        `)
      if (
        !progressVisible?.code ||
        !['重新打开登录页', '取消等待'].every((label) =>
          progressVisible.recoveryActions.includes(label),
        )
      ) {
        throw new Error('High-accuracy login recovery controls did not render')
      }
      const runtimeScreenshot = await smokeWindow.webContents.capturePage()
      writeFileSync(
        resolve(app.getPath('userData'), 'smoke-runtime.png'),
        runtimeScreenshot.toPNG(),
      )
    }
    const pickerState = await smokeWindow.webContents.executeJavaScript(`
      new Promise((resolve) => {
        document.querySelector('.runtime-dialog button[aria-label="关闭"]')?.click()
        Array.from(document.querySelectorAll('button'))
          .find((button) => ['导入歌曲', 'Import song']
            .includes(button.textContent?.trim()))
          ?.click()
        setTimeout(() => {
          const picker = document.querySelector('.import-dialog .drop-zone')
          resolve({
            tag: picker?.tagName,
            hasBrowserInput: Boolean(picker?.querySelector('input[type="file"]')),
            text: picker?.textContent ?? '',
            sourceShortcuts: Array.from(
              document.querySelectorAll('.media-source-button'),
            ).map((button) => {
              const bounds = button.getBoundingClientRect()
              return {
                text: button.textContent?.trim() ?? '',
                width: bounds.width,
                height: bounds.height,
                visible:
                  bounds.width > 70 &&
                  bounds.height >= 32 &&
                  getComputedStyle(button).visibility !== 'hidden',
              }
            }),
          })
        }, 50)
      })
    `)
    const expectedSourceShortcuts = [
      'QQ 音乐',
      '网易云音乐',
      'MOO 音乐',
      '酷我音乐',
      '酷狗音乐',
    ]
    if (
      !pickerState ||
      pickerState.tag !== 'BUTTON' ||
      pickerState.hasBrowserInput !== false ||
      !['选择音频或视频', 'Choose audio or video'].some((label) =>
        String(pickerState.text).includes(label)
      ) ||
      !Array.isArray(pickerState.sourceShortcuts) ||
      expectedSourceShortcuts.some(
        (source) =>
          !pickerState.sourceShortcuts.some(
            (entry: { text?: string }) => entry.text === source,
          ),
      ) ||
      pickerState.sourceShortcuts.some(
        (entry: { visible?: boolean }) => entry.visible !== true,
      )
    ) {
      throw new Error('Desktop native media picker entry did not load')
    }
    const screenshot = await smokeWindow.webContents.capturePage()
    if (!screenshot || screenshot.isEmpty()) {
      throw new Error('Desktop renderer produced an empty frame')
    }
    const screenshotPath = resolve(app.getPath('userData'), 'smoke.png')
    writeFileSync(screenshotPath, screenshot.toPNG())
    const size = screenshot.getSize()
    process.stdout.write(
      `GTS_DESKTOP_READY ${size.width}x${size.height} ${screenshotPath}\n`,
    )
    setTimeout(() => app.quit(), 500)
  }
}

const hasSingleInstanceLock = app.requestSingleInstanceLock()
if (!hasSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', showMainWindow)
  app.whenReady().then(startDesktop).catch((error: unknown) => {
    const message =
      error instanceof Error ? error.stack ?? error.message : 'Desktop startup failed'
    if (process.env.GTS_DESKTOP_SMOKE === '1') {
      process.stderr.write(`${message}\n`)
      app.exit(1)
      return
    }
    dialog.showErrorBox(
      'Guitar Tab Studio',
      message,
    )
    app.quit()
  })
}

app.on('window-all-closed', () => {
  if (backgroundJobState.active) {
    ensureTray()
    return
  }
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  showMainWindow()
})

app.on('before-quit', (event) => {
  if (quitting) return
  event.preventDefault()
  quitting = true
  const stopping = Promise.all([
    localApi?.stop() ?? Promise.resolve(),
    mediaSessions?.stop() ?? Promise.resolve(),
  ])
  void stopping.finally(() => {
    destroyTray()
    stateStore?.close()
    stateStore = undefined
    localApi = undefined
    mediaSessions = undefined
    updater?.stop()
    updater = undefined
    app.quit()
  })
})
