import {
  migrateStudioProject,
  type AnalysisJob,
  type AppLanguagePreference,
  type AuthorizedMediaConsent,
  type CapabilityState,
  type DesktopBridge,
  type DesktopApplicationSettings,
  type DesktopBackgroundJobState,
  type DesktopDefaultMediaBrowser,
  type DesktopLocalFileSource,
  type DesktopRecentProject,
  type DesktopRuntimeStatus,
  type DesktopSelectedFile,
  type InstalledModelPackage,
  type MediaBrowserSession,
  type PlatformDescriptor,
  type ProjectRevision,
  type ProjectRevisionKind,
  type StorageCleanupConfig,
  type StorageCleanupStatus,
  type StudioApplication,
  type StudioApplicationEvent,
  type StudioProject,
  type TranscriptionModelVariant,
  type TranscriptionProviderId,
  type TranscriptionSelectionMode,
  type TranscriptionSetupProgress,
} from '@gts/core'
import { prepareScorePages } from './scoreImport'

declare global {
  interface Window {
    gtsDesktop?: DesktopBridge
  }
}

export interface DesktopStudioControls {
  applicationSettings(): Promise<DesktopApplicationSettings>
  setLanguage(
    preference: AppLanguagePreference,
  ): Promise<DesktopApplicationSettings>
  detectDefaultMediaBrowser(): Promise<DesktopDefaultMediaBrowser>
  openExternalUrl(url: string): Promise<{ opened: true }>
  setBackgroundJobState(
    state: DesktopBackgroundJobState,
  ): Promise<DesktopBackgroundJobState>
  onOpenSettings(listener: () => void): () => void
  onLanguageChanged(
    listener: (settings: DesktopApplicationSettings) => void,
  ): () => void
  selectLocalFile(
    kind: 'media' | 'score',
    source?: DesktopLocalFileSource,
  ): Promise<File | null>
  importMediaKeyDatabase(
    provider: 'kuwo' | 'kugou',
  ): Promise<{ provider: 'kuwo' | 'kugou'; count: number } | null>
  runtimeStatus(projectId?: string): Promise<DesktopRuntimeStatus>
  setRuntimeMode(
    mode: 'local' | 'cloud',
    projectId?: string,
  ): Promise<void>
  configureCloudRuntime(
    apiRoot: string,
    projectId?: string,
  ): Promise<{ configured: true }>
  configureHuggingFaceOAuth(
    clientId: string,
  ): Promise<{ configured: true }>
  setTranscriptionProvider(input: {
    providerId: TranscriptionProviderId
    selectionMode?: TranscriptionSelectionMode
    variant?: TranscriptionModelVariant
    licenseAccepted?: boolean
  }): Promise<DesktopRuntimeStatus['transcription']>
  cancelTranscriptionSetup(): Promise<{ cancelled: boolean }>
  openTranscriptionAuthorization(): Promise<void>
  importTranscriptionWeights(
    variant: TranscriptionModelVariant,
    licenseAccepted: boolean,
  ): Promise<{ importedAt: string } | null>
  openTranscriptionProviderPage(
    variant?: TranscriptionModelVariant,
  ): Promise<void>
  localProjects(): Promise<DesktopRecentProject[]>
  removeProject(id: string): Promise<void>
  onTranscriptionProgress(
    listener: (progress: TranscriptionSetupProgress) => void,
  ): () => void
  modelPackages(): Promise<InstalledModelPackage[]>
  installModelPackage(): Promise<InstalledModelPackage | null>
  downloadModelPackage(manifestUrl: string): Promise<InstalledModelPackage>
  activateModelPackage(id: string, version: string): Promise<void>
  rollbackModelPackage(): Promise<void>
  removeModelPackage(id: string, version: string): Promise<void>
  exportProject(id: string): Promise<{ exported: true } | null>
  importProject(): Promise<StudioProject | null>
  createBackup(): Promise<{ name: string } | null>
  restoreBackup(): Promise<{ restored: true } | null>
  clearCache(): Promise<{ removedBytes: number }>
  openFullDiskAccessSettings(): Promise<void>
  storageCleanupStatus(): Promise<StorageCleanupStatus>
  setStorageCleanupConfig(
    config: StorageCleanupConfig,
  ): Promise<StorageCleanupStatus>
  runStorageCleanup(): Promise<StorageCleanupStatus>
  exportDiagnostics(): Promise<{ exported: true } | null>
}

export class DesktopStudioApplication
  implements StudioApplication<File>, DesktopStudioControls
{
  private readonly bridge: DesktopBridge

  constructor(bridge: DesktopBridge) {
    this.bridge = bridge
  }

  async getCapabilities(): Promise<CapabilityState> {
    return this.invoke('capabilities')
  }

  async listPlatforms(): Promise<PlatformDescriptor[]> {
    return this.invoke('platforms')
  }

  async getDemoProject(): Promise<StudioProject> {
    return migrateStudioProject(await this.invoke('demo-project'))
  }

  async getProject(id: string): Promise<StudioProject> {
    return migrateStudioProject(await this.invoke('project', { id }))
  }

  listProjectRevisions(projectId: string): Promise<ProjectRevision[]> {
    return this.invoke('project-revisions', { projectId })
  }

  saveProjectRevision(
    project: StudioProject,
    label: string,
    kind: ProjectRevisionKind = 'manual',
    expectedParentRevisionId?: string | null,
  ): Promise<ProjectRevision> {
    return this.invoke('save-project-revision', {
      project,
      label,
      kind,
      expectedParentRevisionId,
    })
  }

  restoreProjectRevision(
    projectId: string,
    revisionId: string,
  ): Promise<ProjectRevision> {
    return this.invoke('restore-project-revision', {
      projectId,
      revisionId,
    })
  }

  getJob(id: string): Promise<AnalysisJob> {
    return this.invoke('job', { id })
  }

  cancelJob(id: string): Promise<AnalysisJob> {
    return this.invoke('cancel-job', { id })
  }

  submitLink(
    url: string,
    authorizedMedia?: AuthorizedMediaConsent,
    mediaSession?: MediaBrowserSession,
  ): Promise<AnalysisJob> {
    return this.invoke('submit-link', { url, authorizedMedia, mediaSession })
  }

  async submitUpload(
    file: File,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob> {
    const bytes = new Uint8Array(await file.arrayBuffer())
    return this.invoke('submit-upload', {
      name: file.name,
      mediaType: file.type,
      bytes,
      authorizedMedia,
    })
  }

  async submitScore(file: File): Promise<StudioProject> {
    const pages = await prepareScorePages(file)
    return migrateStudioProject(
      await this.invoke('submit-score', {
        name: file.name,
        pages: await Promise.all(
          pages.map(async (page) => ({
            name: page.name,
            mediaType: page.type,
            bytes: new Uint8Array(await page.arrayBuffer()),
          })),
        ),
      }),
    )
  }

  importMediaKeyDatabase(
    provider: 'kuwo' | 'kugou',
  ): Promise<{ provider: 'kuwo' | 'kugou'; count: number } | null> {
    return this.invoke('import-media-key-database', { provider })
  }

  async selectLocalFile(
    kind: 'media' | 'score',
    source: DesktopLocalFileSource = 'default',
  ): Promise<File | null> {
    const selected = await this.invoke<DesktopSelectedFile | null>(
      'select-local-file',
      { kind, source },
    )
    if (!selected) return null
    const bytes = new Uint8Array(selected.bytes.byteLength)
    bytes.set(selected.bytes)
    return new File([bytes.buffer], selected.name, {
      type: selected.mediaType,
    })
  }

  subscribe(listener: (event: StudioApplicationEvent) => void): () => void {
    const removeOpenMedia = this.bridge.onOpenMedia(() =>
      listener({ type: 'open-media' }),
    )
    const removeOpenProject = this.bridge.onOpenProject((projectId) =>
      listener({ type: 'open-project', projectId }),
    )
    return () => {
      removeOpenMedia()
      removeOpenProject()
    }
  }

  runtimeStatus(projectId?: string): Promise<DesktopRuntimeStatus> {
    return this.invoke('runtime-status', { projectId })
  }

  applicationSettings(): Promise<DesktopApplicationSettings> {
    return this.invoke('application-settings')
  }

  setLanguage(
    preference: AppLanguagePreference,
  ): Promise<DesktopApplicationSettings> {
    return this.invoke('set-language', { preference })
  }

  detectDefaultMediaBrowser(): Promise<DesktopDefaultMediaBrowser> {
    return this.invoke('detect-default-media-browser')
  }

  openExternalUrl(url: string): Promise<{ opened: true }> {
    return this.invoke('open-external-url', { url })
  }

  setBackgroundJobState(
    state: DesktopBackgroundJobState,
  ): Promise<DesktopBackgroundJobState> {
    return this.invoke('set-background-job-state', state)
  }

  onOpenSettings(listener: () => void): () => void {
    return this.bridge.onOpenSettings(listener)
  }

  onLanguageChanged(
    listener: (settings: DesktopApplicationSettings) => void,
  ): () => void {
    return this.bridge.onLanguageChanged(listener)
  }

  setRuntimeMode(
    mode: 'local' | 'cloud',
    projectId?: string,
  ): Promise<void> {
    return this.invoke('set-runtime-mode', { mode, projectId })
  }

  configureCloudRuntime(
    apiRoot: string,
    projectId?: string,
  ): Promise<{ configured: true }> {
    return this.invoke('configure-cloud-runtime', { apiRoot, projectId })
  }

  configureHuggingFaceOAuth(
    clientId: string,
  ): Promise<{ configured: true }> {
    return this.invoke('configure-huggingface-oauth', { clientId })
  }

  setTranscriptionProvider(input: {
    providerId: TranscriptionProviderId
    selectionMode?: TranscriptionSelectionMode
    variant?: TranscriptionModelVariant
    licenseAccepted?: boolean
  }): Promise<DesktopRuntimeStatus['transcription']> {
    return this.invoke('set-transcription-provider', input)
  }

  cancelTranscriptionSetup(): Promise<{ cancelled: boolean }> {
    return this.invoke('cancel-transcription-setup')
  }

  openTranscriptionAuthorization(): Promise<void> {
    return this.invoke('open-transcription-authorization')
  }

  importTranscriptionWeights(
    variant: TranscriptionModelVariant,
    licenseAccepted: boolean,
  ): Promise<{ importedAt: string } | null> {
    return this.invoke('import-transcription-weights', {
      variant,
      licenseAccepted,
    })
  }

  openTranscriptionProviderPage(
    variant?: TranscriptionModelVariant,
  ): Promise<void> {
    return this.invoke('open-transcription-provider-page', { variant })
  }

  localProjects(): Promise<DesktopRecentProject[]> {
    return this.invoke('local-projects')
  }

  removeProject(id: string): Promise<void> {
    return this.invoke('remove-project', { id })
  }

  onTranscriptionProgress(
    listener: (progress: TranscriptionSetupProgress) => void,
  ): () => void {
    return this.bridge.onTranscriptionProgress(listener)
  }

  modelPackages(): Promise<InstalledModelPackage[]> {
    return this.invoke('model-packages')
  }

  installModelPackage(): Promise<InstalledModelPackage | null> {
    return this.invoke('install-model-package')
  }

  downloadModelPackage(manifestUrl: string): Promise<InstalledModelPackage> {
    return this.invoke('download-model-package', { manifestUrl })
  }

  activateModelPackage(id: string, version: string): Promise<void> {
    return this.invoke('activate-model-package', { id, version })
  }

  rollbackModelPackage(): Promise<void> {
    return this.invoke('rollback-model-package')
  }

  removeModelPackage(id: string, version: string): Promise<void> {
    return this.invoke('remove-model-package', { id, version })
  }

  exportProject(id: string): Promise<{ exported: true } | null> {
    return this.invoke('export-project', { id })
  }

  async importProject(): Promise<StudioProject | null> {
    const project = await this.invoke<unknown>('import-project')
    return project === null ? null : migrateStudioProject(project)
  }

  createBackup(): Promise<{ name: string } | null> {
    return this.invoke('create-backup')
  }

  restoreBackup(): Promise<{ restored: true } | null> {
    return this.invoke('restore-backup')
  }

  clearCache(): Promise<{ removedBytes: number }> {
    return this.invoke('clear-cache')
  }

  openFullDiskAccessSettings(): Promise<void> {
    return this.invoke('open-full-disk-access-settings')
  }

  storageCleanupStatus(): Promise<StorageCleanupStatus> {
    return this.invoke('storage-cleanup-status')
  }

  setStorageCleanupConfig(
    config: StorageCleanupConfig,
  ): Promise<StorageCleanupStatus> {
    return this.invoke('set-storage-cleanup-config', config)
  }

  runStorageCleanup(): Promise<StorageCleanupStatus> {
    return this.invoke('run-storage-cleanup')
  }

  exportDiagnostics(): Promise<{ exported: true } | null> {
    return this.invoke('export-diagnostics')
  }

  private invoke<T>(
    method: Parameters<DesktopBridge['invoke']>[0]['method'],
    payload?: unknown,
  ): Promise<T> {
    return this.bridge.invoke({ method, payload }) as Promise<T>
  }
}

export function desktopStudioApplication(): DesktopStudioApplication | null {
  return window.gtsDesktop
    ? new DesktopStudioApplication(window.gtsDesktop)
    : null
}
