import { spawn, type ChildProcess } from 'node:child_process'
import { createServer } from 'node:net'
import { StringDecoder } from 'node:string_decoder'
import {
  migrateStudioProject,
  type AnalysisJob,
  type AuthorizedMediaConsent,
  type CapabilityState,
  type PlatformDescriptor,
  type ProjectRevision,
  type ProjectRevisionKind,
  type StorageCleanupConfig,
  type StorageCleanupStatus,
  type StudioProject,
} from '@gts/core'

export interface DesktopUpload {
  name: string
  mediaType: string
  bytes: Uint8Array
}

export interface DesktopScoreUpload {
  name: string
  pages: DesktopUpload[]
}

export type DesktopMediaKeyProvider = 'kuwo' | 'kugou'

export interface LocalApiServiceOptions {
  apiEntry: string
  workingDirectory: string
  dataRoot: string
  homePath?: string
  workerCommand?: string
  ffmpegBinary?: string
  separationModelRepository?: string
  tabModel?: string
  techniqueModel?: string
  techniqueGate?: string
  transcriptionProvider?:
    | 'basic-pitch'
    | 'muscriptor'
    | 'guitar-specialist'
    | 'multitrack-research'
  transcriptionModel?: string
  transcriptionVariant?: 'small' | 'medium' | 'large'
  tabModelMode?: 'onset-position-research'
  positionModel?: string
  verificationModel?: string
  modelCacheRoot?: string
}

export interface StudioApiTransport {
  assetUrl(path: string): string
  storageAssetUrl(url: string): string
  capabilities(): Promise<CapabilityState>
  platforms(): Promise<PlatformDescriptor[]>
  demoProject(): Promise<StudioProject>
  project(id: string): Promise<StudioProject>
  removeProject(id: string): Promise<{ removed: boolean }>
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
  job(id: string): Promise<AnalysisJob>
  cancelJob(id: string): Promise<AnalysisJob>
  submitLink(
    url: string,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob>
  submitUpload(
    upload: DesktopUpload,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob>
  submitScore(upload: DesktopScoreUpload): Promise<StudioProject>
}

function availablePort(): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        server.close()
        reject(new Error('Unable to allocate a local API port'))
        return
      }
      const { port } = address
      server.close((error) => (error ? reject(error) : resolvePort(port)))
    })
  })
}

export class LocalApiService implements StudioApiTransport {
  private child?: ChildProcess
  private root?: string

  constructor(private readonly options: LocalApiServiceOptions) {}

  assetUrl(path: string): string {
    if (!this.root) throw new Error('Local API is not running')
    return new URL(path, `${this.root}/`).toString()
  }

  storageAssetUrl(url: string): string {
    if (!this.root) throw new Error('Local API is not running')
    try {
      const parsed = new URL(url)
      const root = new URL(this.root)
      return parsed.origin === root.origin
        ? `${parsed.pathname}${parsed.search}`
        : url
    } catch {
      return url
    }
  }

  async start(): Promise<void> {
    if (this.child && this.root) return
    const port = await availablePort()
    this.root = `http://127.0.0.1:${port}`
    const child = spawn(process.execPath, [this.options.apiEntry], {
      cwd: this.options.workingDirectory,
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: '1',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8',
        PORT: String(port),
        GTS_DATA_ROOT: this.options.dataRoot,
        GTS_DESKTOP_LOCAL: '1',
        ...(this.options.homePath
          ? { GTS_DESKTOP_HOME: this.options.homePath }
          : {}),
        ...(this.options.workerCommand
          ? { GTS_WORKER_COMMAND: this.options.workerCommand }
          : {}),
        ...(this.options.ffmpegBinary
          ? { FFMPEG_BINARY: this.options.ffmpegBinary }
          : {}),
        ...(this.options.separationModelRepository
          ? {
              DEMUCS_MODEL_REPOSITORY:
                this.options.separationModelRepository,
            }
          : {}),
        ...(this.options.tabModel
          ? { STRINGTRACE_TAB_MODEL: this.options.tabModel }
          : {}),
        ...(this.options.tabModelMode
          ? { STRINGTRACE_TAB_MODEL_MODE: this.options.tabModelMode }
          : {}),
        ...(this.options.techniqueModel
          ? { STRINGTRACE_TECHNIQUE_MODEL: this.options.techniqueModel }
          : {}),
        ...(this.options.techniqueGate
          ? { STRINGTRACE_TECHNIQUE_GATE: this.options.techniqueGate }
          : {}),
        ...(this.options.transcriptionProvider
          ? {
              GTS_TRANSCRIPTION_PROVIDER:
                this.options.transcriptionProvider,
            }
          : {}),
        ...(this.options.transcriptionModel
          ? { GTS_TRANSCRIPTION_MODEL: this.options.transcriptionModel }
          : {}),
        ...(this.options.transcriptionVariant
          ? { GTS_TRANSCRIPTION_VARIANT: this.options.transcriptionVariant }
          : {}),
        ...(this.options.positionModel
          ? { NOISE2FRET_MODEL: this.options.positionModel }
          : {}),
        ...(this.options.verificationModel
          ? { MR_MT3_MODEL: this.options.verificationModel }
          : {}),
        ...(this.options.modelCacheRoot
          ? { HF_HUB_CACHE: this.options.modelCacheRoot }
          : {}),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    this.child = child
    const stdoutDecoder = new StringDecoder('utf8')
    const stderrDecoder = new StringDecoder('utf8')
    child.stdout?.on('data', (chunk: Buffer) => {
      process.stdout.write(`[local-api] ${stdoutDecoder.write(chunk)}`)
    })
    child.stderr?.on('data', (chunk: Buffer) => {
      process.stderr.write(`[local-api] ${stderrDecoder.write(chunk)}`)
    })
    child.once('close', () => {
      const finalStdout = stdoutDecoder.end()
      const finalStderr = stderrDecoder.end()
      if (finalStdout) process.stdout.write(`[local-api] ${finalStdout}`)
      if (finalStderr) process.stderr.write(`[local-api] ${finalStderr}`)
    })
    await this.waitUntilReady()
  }

  async stop(): Promise<void> {
    const child = this.child
    this.child = undefined
    this.root = undefined
    if (!child || child.exitCode !== null) return
    child.kill('SIGTERM')
    await new Promise<void>((resolveStop) => {
      const timeout = setTimeout(() => {
        child.kill('SIGKILL')
        resolveStop()
      }, 3000)
      child.once('exit', () => {
        clearTimeout(timeout)
        resolveStop()
      })
    })
  }

  async capabilities(): Promise<CapabilityState> {
    const response = await this.request<{ capabilities: CapabilityState }>(
      '/health',
    )
    return response.capabilities
  }

  async platforms(): Promise<PlatformDescriptor[]> {
    const response = await this.request<{ platforms: PlatformDescriptor[] }>(
      '/api/platforms',
    )
    return response.platforms
  }

  async demoProject(): Promise<StudioProject> {
    return migrateStudioProject(await this.request('/api/projects/demo'))
  }

  async project(id: string): Promise<StudioProject> {
    return migrateStudioProject(
      await this.request(`/api/projects/${encodeURIComponent(id)}`),
    )
  }

  removeProject(id: string): Promise<{ removed: boolean }> {
    return this.request(`/api/projects/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    })
  }

  listProjectRevisions(projectId: string): Promise<ProjectRevision[]> {
    return this.request<{ revisions: ProjectRevision[] }>(
      `/api/projects/${encodeURIComponent(projectId)}/revisions`,
    ).then((response) => response.revisions)
  }

  saveProjectRevision(
    project: StudioProject,
    label: string,
    kind: ProjectRevisionKind = 'manual',
    expectedParentRevisionId?: string | null,
  ): Promise<ProjectRevision> {
    return this.request(
      `/api/projects/${encodeURIComponent(project.id)}/revisions`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project,
          label,
          kind,
          expectedParentRevisionId,
        }),
      },
    )
  }

  restoreProjectRevision(
    projectId: string,
    revisionId: string,
  ): Promise<ProjectRevision> {
    return this.request(
      `/api/projects/${encodeURIComponent(projectId)}/revisions/${encodeURIComponent(revisionId)}/restore`,
      { method: 'POST' },
    )
  }

  job(id: string): Promise<AnalysisJob> {
    return this.request(`/api/jobs/${encodeURIComponent(id)}`)
  }

  cancelJob(id: string): Promise<AnalysisJob> {
    return this.request(`/api/jobs/${encodeURIComponent(id)}/cancel`, {
      method: 'POST',
    })
  }

  submitLink(
    url: string,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob> {
    return this.request('/api/import/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        authorizedProcessingConsent: authorizedMedia?.accepted,
        authorizationPolicyVersion: authorizedMedia?.policyVersion,
      }),
    })
  }

  submitUpload(
    upload: DesktopUpload,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob> {
    const body = new FormData()
    const arrayBuffer = new ArrayBuffer(upload.bytes.byteLength)
    new Uint8Array(arrayBuffer).set(upload.bytes)
    body.append(
      'media',
      new Blob([arrayBuffer], {
        type: upload.mediaType || 'application/octet-stream',
      }),
      upload.name,
    )
    if (authorizedMedia?.accepted) {
      body.append('authorizedProcessingConsent', 'true')
      body.append(
        'authorizationPolicyVersion',
        String(authorizedMedia.policyVersion),
      )
    }
    return this.request('/api/import/upload', { method: 'POST', body })
  }

  submitScore(upload: DesktopScoreUpload): Promise<StudioProject> {
    const body = new FormData()
    for (const page of upload.pages) {
      const arrayBuffer = new ArrayBuffer(page.bytes.byteLength)
      new Uint8Array(arrayBuffer).set(page.bytes)
      body.append(
        'score',
        new Blob([arrayBuffer], {
          type: page.mediaType || 'application/octet-stream',
        }),
        page.name,
      )
    }
    body.append('originalName', upload.name)
    return this.request('/api/import/score', { method: 'POST', body }).then(
      migrateStudioProject,
    )
  }

  importMediaKeyDatabase(
    provider: DesktopMediaKeyProvider,
    upload: DesktopUpload,
  ): Promise<{ provider: DesktopMediaKeyProvider; count: number }> {
    const body = new FormData()
    const arrayBuffer = new ArrayBuffer(upload.bytes.byteLength)
    new Uint8Array(arrayBuffer).set(upload.bytes)
    body.append(
      'database',
      new Blob([arrayBuffer], {
        type: upload.mediaType || 'application/octet-stream',
      }),
      upload.name,
    )
    return this.request(
      `/api/runtime/authorized-media-keys/${provider}`,
      { method: 'POST', body },
    )
  }

  storageCleanupStatus(): Promise<StorageCleanupStatus> {
    return this.request('/api/runtime/storage-cleanup')
  }

  updateStorageCleanup(
    config: StorageCleanupConfig,
  ): Promise<StorageCleanupStatus> {
    return this.request('/api/runtime/storage-cleanup', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
  }

  runStorageCleanup(): Promise<StorageCleanupStatus> {
    return this.request('/api/runtime/storage-cleanup/run', {
      method: 'POST',
    })
  }

  private async waitUntilReady(): Promise<void> {
    const deadline = Date.now() + 15_000
    let lastError: unknown
    while (Date.now() < deadline) {
      if (this.child?.exitCode !== null) {
        throw new Error(`Local API exited with code ${this.child?.exitCode}`)
      }
      try {
        await this.request('/health')
        return
      } catch (error) {
        lastError = error
        await new Promise((resolveWait) => setTimeout(resolveWait, 100))
      }
    }
    throw lastError instanceof Error
      ? lastError
      : new Error('Local API did not become ready')
  }

  private async request<T = unknown>(
    path: string,
    init?: RequestInit,
  ): Promise<T> {
    if (!this.root) throw new Error('Local API is not running')
    let response: Response
    try {
      response = await fetch(`${this.root}${path}`, init)
    } catch {
      throw new Error(
        '本地分析服务未响应。请完全退出应用后重新打开；若仍失败，请导出诊断信息。',
      )
    }
    const body = (await response.json()) as T & { error?: string }
    if (!response.ok) {
      throw new Error(body.error ?? `Local API request failed (${response.status})`)
    }
    return body
  }
}
