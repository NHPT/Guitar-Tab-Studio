import { contextBridge, ipcRenderer } from 'electron'
import type {
  DesktopApplicationSettings,
  DesktopBridge,
  DesktopInvokeRequest,
  DesktopStudioMethod,
  TranscriptionSetupProgress,
} from '@gts/core'

const allowedMethods = new Set<DesktopStudioMethod>([
  'capabilities',
  'platforms',
  'demo-project',
  'project',
  'project-revisions',
  'save-project-revision',
  'restore-project-revision',
  'job',
  'cancel-job',
  'submit-link',
  'submit-upload',
  'submit-score',
  'select-local-file',
  'import-media-key-database',
  'local-projects',
  'remove-project',
  'recent-projects',
  'runtime-status',
  'set-runtime-mode',
  'configure-cloud-runtime',
  'configure-huggingface-oauth',
  'set-transcription-provider',
  'cancel-transcription-setup',
  'open-transcription-authorization',
  'import-transcription-weights',
  'open-transcription-provider-page',
  'model-packages',
  'install-model-package',
  'download-model-package',
  'activate-model-package',
  'rollback-model-package',
  'remove-model-package',
  'export-project',
  'import-project',
  'create-backup',
  'restore-backup',
  'clear-cache',
  'open-full-disk-access-settings',
  'storage-cleanup-status',
  'set-storage-cleanup-config',
  'run-storage-cleanup',
  'export-diagnostics',
  'application-settings',
  'set-language',
  'detect-default-media-browser',
  'open-external-url',
  'set-background-job-state',
])

const bridge: DesktopBridge = {
  invoke(request: DesktopInvokeRequest): Promise<unknown> {
    if (!allowedMethods.has(request.method)) {
      return Promise.reject(new Error('Desktop operation is not allowed'))
    }
    return ipcRenderer.invoke('studio:invoke', request)
  },
  onOpenMedia(listener: () => void): () => void {
    const handler = () => listener()
    ipcRenderer.on('studio:open-media', handler)
    return () => ipcRenderer.removeListener('studio:open-media', handler)
  },
  onOpenProject(listener: (projectId: string) => void): () => void {
    const handler = (_event: unknown, projectId: string) => listener(projectId)
    ipcRenderer.on('studio:open-project', handler)
    return () => ipcRenderer.removeListener('studio:open-project', handler)
  },
  onOpenSettings(listener: () => void): () => void {
    const handler = () => listener()
    ipcRenderer.on('studio:open-settings', handler)
    return () => ipcRenderer.removeListener('studio:open-settings', handler)
  },
  onLanguageChanged(
    listener: (settings: DesktopApplicationSettings) => void,
  ): () => void {
    const handler = (
      _event: unknown,
      settings: DesktopApplicationSettings,
    ) => listener(settings)
    ipcRenderer.on('studio:language-changed', handler)
    return () =>
      ipcRenderer.removeListener('studio:language-changed', handler)
  },
  onTranscriptionProgress(
    listener: (progress: TranscriptionSetupProgress) => void,
  ): () => void {
    const handler = (
      _event: unknown,
      progress: TranscriptionSetupProgress,
    ) => listener(progress)
    ipcRenderer.on('studio:transcription-progress', handler)
    return () =>
      ipcRenderer.removeListener('studio:transcription-progress', handler)
  },
}

contextBridge.exposeInMainWorld('gtsDesktop', Object.freeze(bridge))
