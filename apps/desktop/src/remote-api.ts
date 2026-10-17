import {
  migrateStudioProject,
  type AnalysisJob,
  type AuthorizedMediaConsent,
  type CapabilityState,
  type PlatformDescriptor,
  type ProjectRevision,
  type ProjectRevisionKind,
  type StudioProject,
} from '@gts/core'
import type {
  DesktopScoreUpload,
  DesktopUpload,
  StudioApiTransport,
} from './local-api.js'

export class RemoteApiService implements StudioApiTransport {
  private readonly root: string

  constructor(root: string) {
    const parsed = new URL(root)
    if (parsed.protocol !== 'https:') {
      throw new Error('Cloud API must use HTTPS')
    }
    this.root = parsed.toString().replace(/\/$/u, '')
  }

  assetUrl(path: string): string {
    return new URL(path, `${this.root}/`).toString()
  }

  storageAssetUrl(url: string): string {
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

  async capabilities(): Promise<CapabilityState> {
    const response = await this.request<{ capabilities: CapabilityState }>(
      '/health',
      undefined,
      5_000,
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
    return this.request(
      '/api/import/upload',
      { method: 'POST', body },
      10 * 60_000,
    )
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
    return this.request(
      '/api/import/score',
      { method: 'POST', body },
      2 * 60_000,
    ).then(migrateStudioProject)
  }

  private async request<T = unknown>(
    path: string,
    init?: RequestInit,
    timeoutMs = 30_000,
  ): Promise<T> {
    const response = await fetch(`${this.root}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    })
    const body = (await response.json()) as T & { error?: string }
    if (!response.ok) {
      throw new Error(body.error ?? `Cloud API request failed (${response.status})`)
    }
    return body
  }
}
