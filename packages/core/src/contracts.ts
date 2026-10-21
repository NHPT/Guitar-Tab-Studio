import type {
  AnalysisJob,
  AuthorizedMediaConsent,
  CapabilityState,
  MediaBrowserDescriptor,
  MediaBrowserSession,
  PlatformDescriptor,
  ProjectRevision,
  ProjectRevisionKind,
  StudioProject,
} from './domain.js'
import type { DesktopTranscriptionStatus } from './transcription-provider.js'

export type RepositoryResult<T> = T | Promise<T>

export interface ProjectRepository {
  getProject(id: string): RepositoryResult<StudioProject | undefined>
  listProjects(): RepositoryResult<StudioProject[]>
  saveProject(project: StudioProject): RepositoryResult<void>
  removeProject(id: string): RepositoryResult<void>
}

export interface RevisionRepository {
  getRevision(id: string): RepositoryResult<ProjectRevision | undefined>
  listRevisions(projectId: string): RepositoryResult<ProjectRevision[]>
  saveRevision(revision: ProjectRevision): RepositoryResult<void>
}

export interface MediaObject {
  id: string
  displayName: string
  mediaType: string
  sizeBytes: number
  sha256?: string
}

export interface MediaWrite {
  displayName: string
  mediaType: string
  bytes: Uint8Array
  sha256?: string
}

export interface MediaRepository {
  getMedia(id: string): RepositoryResult<MediaObject | undefined>
  putMedia(input: MediaWrite): RepositoryResult<MediaObject>
  removeMedia(id: string): RepositoryResult<void>
}

export interface InferenceRequest<TMedia = string> {
  job: AnalysisJob
  media?: TMedia
  signal?: AbortSignal
}

export interface InferenceProgress {
  job: AnalysisJob
}

export interface InferenceProvider<TMedia = string> {
  capabilities(): RepositoryResult<CapabilityState>
  analyze(
    request: InferenceRequest<TMedia>,
    onProgress: (progress: InferenceProgress) => void,
  ): Promise<StudioProject>
  cancel?(jobId: string): RepositoryResult<boolean>
}

export interface StudioApplication<TUpload = MediaWrite> {
  getCapabilities(): Promise<CapabilityState>
  listPlatforms(): Promise<PlatformDescriptor[]>
  getDemoProject(): Promise<StudioProject>
  getProject(id: string): Promise<StudioProject>
  listProjectRevisions(projectId: string): Promise<ProjectRevision[]>
  saveProjectRevision(
    project: StudioProject,
    label: string,
    kind?: ProjectRevisionKind,
    expectedParentRevisionId?: string | null,
  ): Promise<ProjectRevision>
  restoreProjectRevision(
    projectId: string,
    revisionId: string,
  ): Promise<ProjectRevision>
  getJob(id: string): Promise<AnalysisJob>
  cancelJob(id: string): Promise<AnalysisJob>
  submitLink(
    url: string,
    authorizedMedia?: AuthorizedMediaConsent,
    mediaSession?: MediaBrowserSession,
  ): Promise<AnalysisJob>
  submitUpload(
    input: TUpload,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob>
  submitScore(input: TUpload): Promise<StudioProject>
  subscribe?(listener: (event: StudioApplicationEvent) => void): () => void
}

export type StudioApplicationEvent =
  | { type: 'open-media' }
  | { type: 'open-project'; projectId: string }

export type AppLanguage = 'zh-CN' | 'en-US'
export type AppLanguagePreference = 'system' | AppLanguage

export interface DesktopApplicationSettings {
  language: AppLanguage
  languagePreference: AppLanguagePreference
  systemLanguage: AppLanguage
}

export interface DesktopBackgroundJobState {
  active: boolean
  jobId?: string
  progress?: number
  stageLabel?: string
}

export type DesktopStudioMethod =
  | 'capabilities'
  | 'platforms'
  | 'demo-project'
  | 'project'
  | 'project-revisions'
  | 'save-project-revision'
  | 'restore-project-revision'
  | 'job'
  | 'cancel-job'
  | 'submit-link'
  | 'submit-upload'
  | 'submit-score'
  | 'select-local-file'
  | 'import-media-key-database'
  | 'local-projects'
  | 'remove-project'
  | 'recent-projects'
  | 'runtime-status'
  | 'set-runtime-mode'
  | 'configure-cloud-runtime'
  | 'configure-huggingface-oauth'
  | 'set-transcription-provider'
  | 'cancel-transcription-setup'
  | 'open-transcription-authorization'
  | 'import-transcription-weights'
  | 'open-transcription-provider-page'
  | 'model-packages'
  | 'install-model-package'
  | 'download-model-package'
  | 'activate-model-package'
  | 'rollback-model-package'
  | 'remove-model-package'
  | 'export-project'
  | 'import-project'
  | 'create-backup'
  | 'restore-backup'
  | 'clear-cache'
  | 'open-full-disk-access-settings'
  | 'storage-cleanup-status'
  | 'set-storage-cleanup-config'
  | 'run-storage-cleanup'
  | 'export-diagnostics'
  | 'application-settings'
  | 'set-language'
  | 'detect-default-media-browser'
  | 'authorize-media-session'
  | 'authorize-qqmusic'
  | 'open-external-url'
  | 'set-background-job-state'

export interface DesktopInvokeRequest {
  method: DesktopStudioMethod
  payload?: unknown
}

export type DesktopLocalFileSource =
  | 'default'
  | 'qqmusic'
  | 'netease'
  | 'moo'
  | 'kuwo'
  | 'kugou'

export interface DesktopSelectedFile {
  name: string
  mediaType: string
  bytes: Uint8Array
}

export interface DesktopDefaultMediaBrowser {
  supported: boolean
  browser?: MediaBrowserDescriptor
  detectedLabel?: string
  sessionMode?: 'browser-cookies' | 'isolated-login'
}

export interface DesktopMediaSessionAuthorization {
  browser: MediaBrowserDescriptor
  sessionToken: string
}

export interface DesktopQQMusicAuthorization {
  credentialSession: string
}

export interface DesktopRuntimeStatus {
  mode: 'desktop-local'
  providerMode: 'local' | 'cloud'
  cloudAvailable: boolean
  cloudReady: boolean
  cloudApiRoot?: string
  uploadsAudio: boolean
  apiReady: boolean
  storageReady: boolean
  workerReady: boolean
  storageScope: 'application-data'
  estimatedSeconds: {
    minimum: number
    maximum: number
  }
  resourceNote: string
  transcription: DesktopTranscriptionStatus
  cloudError?: string
  modelWarning?: string
  activeModel?: {
    id: string
    version: string
    releaseChannel: 'research' | 'production'
  }
}

export interface StorageCleanupConfig {
  enabled: boolean
  triggerFreeBytes: number
  targetFreeBytes: number
  retentionHours: number
}

export type StorageCleanupReason =
  | 'startup'
  | 'interval'
  | 'capacity'
  | 'job-settled'
  | 'manual'

export interface StorageCleanupRun {
  id: string
  reason: StorageCleanupReason
  startedAt: string
  completedAt: string
  outcome: 'completed' | 'skipped' | 'failed'
  availableBeforeBytes: number
  availableAfterBytes: number
  removedBytes: number
  removedItems: number
  targetReached: boolean
  error?: string
}

export interface StorageCleanupStatus {
  config: StorageCleanupConfig
  availableBytes: number
  reclaimableBytes: number
  reclaimableItems: number
  running: boolean
  lastRun?: StorageCleanupRun
  history: StorageCleanupRun[]
}

export interface DesktopRecentProject {
  id: string
  title: string
  sourceLabel: string
  openedAt: string
}

export interface TranscriptionSetupProgress {
  phase:
    | 'checking'
    | 'authorizing'
    | 'downloading'
    | 'verifying'
    | 'ready'
  progress: number
  stageLabel: string
  authorizationUrl?: string
  userCode?: string
}

export interface DesktopUpdateProgress {
  phase: 'downloading' | 'ready' | 'failed'
  version: string
  percent: number
  transferredBytes: number
  totalBytes: number
  bytesPerSecond: number
  error?: string
}

export interface DesktopBridge {
  invoke(request: DesktopInvokeRequest): Promise<unknown>
  onOpenMedia(listener: () => void): () => void
  onOpenProject(listener: (projectId: string) => void): () => void
  onOpenSettings(listener: () => void): () => void
  onLanguageChanged(listener: (settings: DesktopApplicationSettings) => void): () => void
  onTranscriptionProgress(
    listener: (progress: TranscriptionSetupProgress) => void,
  ): () => void
  onUpdateProgress?(
    listener: (progress: DesktopUpdateProgress) => void,
  ): () => void
}
