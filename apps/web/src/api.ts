import type {
  AnalysisJob,
  AdminAuditEvent,
  AdminDashboard,
  AdminSession,
  AdminUser,
  AuthorizedMediaConsent,
  BeatBackup,
  BeatGoldStatus,
  BeatAdjudicationSummary,
  BeatEvaluationReport,
  BeatGoldRelease,
  BeatReviewComparison,
  BeatReviewDocument,
  BeatReviewDraft,
  BeatReviewPackage,
  BeatReviewSummary,
  BeatStructureDocument,
  BeatStructureDraft,
  BeatStructureSummary,
  BeatTrainingDataset,
  BeatTrainingJob,
  BeatTrainingStatus,
  CapabilityState,
  MediaBrowserSession,
  CommunityAccount,
  CommunityAnswer,
  CommunityContribution,
  CommunityDashboard,
  CommunityTask,
  DatasetRelease,
  ExperimentRun,
  InferenceDeploymentStatus,
  ModelPromotion,
  PlatformDescriptor,
  ProjectRevision,
  ProjectRevisionKind,
  ReviewDocument,
  ReviewDraft,
  ReviewLayerStatus,
  ReviewSummary,
  StudioProject,
  TrainingRecipe,
} from './types'
import {
  migrateStudioProject,
  type StudioApplication,
} from '@gts/core'
import { prepareScorePages } from './scoreImport'

const API_ROOT =
  import.meta.env.VITE_API_ROOT ??
  (import.meta.env.DEV ? 'http://127.0.0.1:8787' : '')

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    credentials: 'include',
    ...init,
  })
  const body = (await response.json()) as T & { error?: string }

  if (!response.ok) {
    throw new Error(body.error ?? `请求失败 (${response.status})`)
  }

  return body
}

export class HttpStudioApplication implements StudioApplication<File> {
  async getCapabilities(): Promise<CapabilityState> {
    const response = await jsonRequest<{ capabilities: CapabilityState }>(
      '/health',
    )
    return response.capabilities
  }

  async listPlatforms(): Promise<PlatformDescriptor[]> {
    const response = await jsonRequest<{ platforms: PlatformDescriptor[] }>(
      '/api/platforms',
    )
    return response.platforms
  }

  async getDemoProject(): Promise<StudioProject> {
    return migrateStudioProject(
      await jsonRequest<unknown>('/api/projects/demo'),
    )
  }

  async getProject(id: string): Promise<StudioProject> {
    return migrateStudioProject(
      await jsonRequest<unknown>(`/api/projects/${id}`),
    )
  }

  listProjectRevisions(projectId: string): Promise<ProjectRevision[]> {
    return jsonRequest<{ revisions: ProjectRevision[] }>(
      `/api/projects/${encodeURIComponent(projectId)}/revisions`,
    ).then((response) => response.revisions)
  }

  saveProjectRevision(
    project: StudioProject,
    label: string,
    kind: ProjectRevisionKind = 'manual',
    expectedParentRevisionId?: string | null,
  ): Promise<ProjectRevision> {
    return jsonRequest(`/api/projects/${encodeURIComponent(project.id)}/revisions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project,
        label,
        kind,
        expectedParentRevisionId,
      }),
    })
  }

  restoreProjectRevision(
    projectId: string,
    revisionId: string,
  ): Promise<ProjectRevision> {
    return jsonRequest(
      `/api/projects/${encodeURIComponent(projectId)}/revisions/${encodeURIComponent(revisionId)}/restore`,
      { method: 'POST' },
    )
  }

  getJob(id: string): Promise<AnalysisJob> {
    return jsonRequest<AnalysisJob>(`/api/jobs/${id}`)
  }

  cancelJob(id: string): Promise<AnalysisJob> {
    return jsonRequest<AnalysisJob>(`/api/jobs/${id}/cancel`, {
      method: 'POST',
    })
  }

  submitLink(
    url: string,
    authorizedMedia?: AuthorizedMediaConsent,
    mediaSession?: MediaBrowserSession,
  ): Promise<AnalysisJob> {
    return jsonRequest<AnalysisJob>('/api/import/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        authorizedProcessingConsent: authorizedMedia?.accepted,
        authorizationPolicyVersion: authorizedMedia?.policyVersion,
        mediaSession,
      }),
    })
  }

  submitUpload(
    file: File,
    authorizedMedia?: AuthorizedMediaConsent,
  ): Promise<AnalysisJob> {
    const body = new FormData()
    body.append('originalName', file.name)
    body.append('media', file)
    if (authorizedMedia?.accepted) {
      body.append('authorizedProcessingConsent', 'true')
      body.append(
        'authorizationPolicyVersion',
        String(authorizedMedia.policyVersion),
      )
    }
    return jsonRequest<AnalysisJob>('/api/import/upload', {
      method: 'POST',
      body,
    })
  }

  async submitScore(file: File): Promise<StudioProject> {
    const body = new FormData()
    const pages = await prepareScorePages(file)
    for (const page of pages) body.append('score', page)
    body.append('originalName', file.name)
    return migrateStudioProject(
      await jsonRequest<unknown>('/api/import/score', {
        method: 'POST',
        body,
      }),
    )
  }
}

export const studioApplication = new HttpStudioApplication()

export const fetchHealth = () => studioApplication.getCapabilities()
export const fetchPlatforms = () => studioApplication.listPlatforms()
export const fetchDemoProject = () => studioApplication.getDemoProject()
export const fetchProject = (id: string) => studioApplication.getProject(id)
export const fetchJob = (id: string) => studioApplication.getJob(id)
export const cancelJob = (id: string) => studioApplication.cancelJob(id)
export const submitLink = (
  url: string,
  authorizedMedia?: AuthorizedMediaConsent,
  mediaSession?: MediaBrowserSession,
) => studioApplication.submitLink(url, authorizedMedia, mediaSession)
export const submitUpload = (
  file: File,
  authorizedMedia?: AuthorizedMediaConsent,
) => studioApplication.submitUpload(file, authorizedMedia)
export const submitScore = (file: File) =>
  studioApplication.submitScore(file)

function reviewApiPath(path: string, queue?: string): string {
  if (!queue) return path
  return `${path}?${new URLSearchParams({ queue }).toString()}`
}

export function fetchReviews(queue?: string): Promise<ReviewSummary[]> {
  return jsonRequest<{ reviews: ReviewSummary[] }>(
    reviewApiPath('/api/reviews', queue),
  ).then(
    (response) => response.reviews,
  )
}

export function fetchReview(
  name: string,
  queue?: string,
): Promise<ReviewDocument> {
  return jsonRequest<ReviewDocument>(
    reviewApiPath(`/api/reviews/${encodeURIComponent(name)}`, queue),
  )
}

export function saveReview(
  name: string,
  draft: ReviewDraft,
  queue?: string,
): Promise<ReviewDocument> {
  return jsonRequest<ReviewDocument>(
    reviewApiPath(`/api/reviews/${encodeURIComponent(name)}`, queue),
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function approveReview(
  name: string,
  draft: ReviewDraft,
  reviewerAlias: string,
  queue?: string,
): Promise<ReviewDocument> {
  return jsonRequest<ReviewDocument>(
    reviewApiPath(
      `/api/reviews/${encodeURIComponent(name)}/approve`,
      queue,
    ),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...draft, reviewerAlias }),
    },
  )
}

export function reopenReview(
  name: string,
  reason: string,
  queue?: string,
): Promise<ReviewDocument> {
  return jsonRequest<ReviewDocument>(
    reviewApiPath(
      `/api/reviews/${encodeURIComponent(name)}/reopen`,
      queue,
    ),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    },
  )
}

function beatReviewApiPath(path: string, reviewer: string): string {
  return `${path}?${new URLSearchParams({ reviewer }).toString()}`
}

export function fetchBeatGoldStatus(): Promise<BeatGoldStatus> {
  return jsonRequest('/api/admin/beat-gold')
}

export function fetchBeatReviews(
  reviewer: string,
): Promise<BeatReviewSummary[]> {
  return jsonRequest<{ reviews: BeatReviewSummary[] }>(
    beatReviewApiPath('/api/admin/beat-gold/reviews', reviewer),
  ).then((response) => response.reviews)
}

export function fetchBeatReview(
  clipId: string,
  reviewer: string,
): Promise<BeatReviewDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/reviews/${encodeURIComponent(clipId)}`,
      reviewer,
    ),
  )
}

export function saveBeatReview(
  clipId: string,
  reviewer: string,
  draft: BeatReviewDraft,
): Promise<BeatReviewDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/reviews/${encodeURIComponent(clipId)}`,
      reviewer,
    ),
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function submitBeatReview(
  clipId: string,
  reviewer: string,
  draft: BeatReviewDraft,
): Promise<BeatReviewDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/reviews/${encodeURIComponent(clipId)}/submit`,
      reviewer,
    ),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function fetchBeatStructureReviews(
  reviewer: string,
): Promise<BeatStructureSummary[]> {
  return jsonRequest<{ reviews: BeatStructureSummary[] }>(
    beatReviewApiPath('/api/admin/beat-gold/structure-reviews', reviewer),
  ).then((response) => response.reviews)
}

export function fetchBeatStructureReview(
  recordingId: string,
  reviewer: string,
): Promise<BeatStructureDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/structure-reviews/${encodeURIComponent(recordingId)}`,
      reviewer,
    ),
  )
}

export function saveBeatStructureReview(
  recordingId: string,
  reviewer: string,
  draft: BeatStructureDraft,
): Promise<BeatStructureDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/structure-reviews/${encodeURIComponent(recordingId)}`,
      reviewer,
    ),
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function submitBeatStructureReview(
  recordingId: string,
  reviewer: string,
  draft: BeatStructureDraft,
): Promise<BeatStructureDocument> {
  return jsonRequest(
    beatReviewApiPath(
      `/api/admin/beat-gold/structure-reviews/${encodeURIComponent(recordingId)}/submit`,
      reviewer,
    ),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function apiAssetUrl(path: string): string {
  return `${API_ROOT}${path}`
}

export function fetchAdminSession(): Promise<AdminSession> {
  return jsonRequest('/api/admin/auth/session')
}

export function bootstrapAdmin(input: {
  username: string
  password: string
  displayName: string
}): Promise<{ user: AdminUser }> {
  return jsonRequest('/api/admin/auth/bootstrap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function loginAdmin(input: {
  username: string
  password: string
}): Promise<{ user: AdminUser }> {
  return jsonRequest('/api/admin/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export async function logoutAdmin(): Promise<void> {
  const response = await fetch(`${API_ROOT}/api/admin/auth/logout`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!response.ok) {
    const body = (await response.json()) as { error?: string }
    throw new Error(body.error ?? `请求失败 (${response.status})`)
  }
}

export function fetchAdminDashboard(): Promise<AdminDashboard> {
  return jsonRequest('/api/admin/dashboard')
}

export function fetchAdminUsers(): Promise<AdminUser[]> {
  return jsonRequest<{ users: AdminUser[] }>('/api/admin/users').then(
    (response) => response.users,
  )
}

export function createAdminUser(input: {
  username: string
  password: string
  displayName: string
  role: AdminUser['role']
  reviewerAlias?: string
}): Promise<AdminUser> {
  return jsonRequest('/api/admin/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function updateAdminUser(
  userId: string,
  input: Partial<
    Pick<
      AdminUser,
      'displayName' | 'role' | 'reviewerAlias' | 'active'
    >
  > & { password?: string },
): Promise<AdminUser> {
  return jsonRequest(`/api/admin/users/${encodeURIComponent(userId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function fetchAdminAudit(): Promise<AdminAuditEvent[]> {
  return jsonRequest<{ events: AdminAuditEvent[] }>(
    '/api/admin/audit?limit=200',
  ).then((response) => response.events)
}

function adjudicationPath(path: string): string {
  return `${path}?${new URLSearchParams({
    left: 'reviewer-a',
    right: 'reviewer-b',
  }).toString()}`
}

export function fetchBeatAdjudicationQueue(): Promise<
  BeatAdjudicationSummary[]
> {
  return jsonRequest<{ items: BeatAdjudicationSummary[] }>(
    adjudicationPath('/api/admin/beat-gold/adjudication'),
  ).then((response) => response.items)
}

export function fetchBeatComparison(
  clipId: string,
): Promise<BeatReviewComparison> {
  return jsonRequest(
    adjudicationPath(
      `/api/admin/beat-gold/adjudication/${encodeURIComponent(clipId)}`,
    ),
  )
}

export function adjudicateBeat(
  clipId: string,
  draft: BeatReviewDraft,
): Promise<BeatReviewPackage> {
  return jsonRequest(
    adjudicationPath(
      `/api/admin/beat-gold/adjudication/${encodeURIComponent(clipId)}`,
    ),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    },
  )
}

export function fetchBeatEvaluation(): Promise<{
  current: BeatEvaluationReport
  runs: BeatEvaluationReport[]
}> {
  return jsonRequest('/api/admin/beat-gold/evaluation')
}

export function saveBeatEvaluation(
  modelVersion: string,
): Promise<BeatEvaluationReport> {
  return jsonRequest('/api/admin/beat-gold/evaluation/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelVersion }),
  })
}

export function fetchBeatGoldReleases(): Promise<BeatGoldRelease[]> {
  return jsonRequest<{ releases: BeatGoldRelease[] }>(
    '/api/admin/beat-gold/releases',
  ).then((response) => response.releases)
}

export function createBeatGoldRelease(
  version: string,
): Promise<BeatGoldRelease> {
  return jsonRequest('/api/admin/beat-gold/releases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version }),
  })
}

export function beatGoldReleaseUrl(version: string): string {
  return apiAssetUrl(
    `/api/admin/beat-gold/releases/${encodeURIComponent(version)}/download`,
  )
}

export function fetchBeatTrainingStatus(): Promise<BeatTrainingStatus> {
  return jsonRequest('/api/admin/beat-training')
}

export function createBeatTrainingDataset(
  version: string,
): Promise<BeatTrainingDataset> {
  return jsonRequest('/api/admin/beat-training/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version }),
  })
}

export function createBeatTrainingJob(input: {
  datasetVersion: string
  codeRevision: string
}): Promise<BeatTrainingJob> {
  return jsonRequest('/api/admin/beat-training/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function fetchReviewLayer(
  layerId: string,
): Promise<ReviewLayerStatus> {
  return jsonRequest(
    `/api/admin/review-layers/${encodeURIComponent(layerId)}`,
  )
}

export function fetchBeatBackups(): Promise<BeatBackup[]> {
  return jsonRequest<{ backups: BeatBackup[] }>(
    '/api/admin/beat-backups',
  ).then((response) => response.backups)
}

export function createBeatBackup(): Promise<BeatBackup> {
  return jsonRequest('/api/admin/beat-backups', { method: 'POST' })
}

export function verifyBeatBackup(backupId: string): Promise<BeatBackup> {
  return jsonRequest(
    `/api/admin/beat-backups/${encodeURIComponent(backupId)}/verify`,
    { method: 'POST' },
  )
}

export function restoreBeatBackup(backupId: string): Promise<BeatBackup> {
  return jsonRequest(
    `/api/admin/beat-backups/${encodeURIComponent(backupId)}/restore`,
    { method: 'POST' },
  )
}

function communityRequest<T>(
  path: string,
  token?: string,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return jsonRequest<T>(path, { ...init, headers })
}

export function registerCommunityAccount(
  alias: string,
  bootstrapKey?: string,
): Promise<{ account: CommunityAccount; token: string }> {
  let deviceId = window.localStorage.getItem('gts-community-device')
  if (!deviceId) {
    deviceId = crypto.randomUUID()
    window.localStorage.setItem('gts-community-device', deviceId)
  }
  return communityRequest('/api/community/accounts', undefined, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Community-Device': deviceId,
    },
    body: JSON.stringify({ alias, bootstrapKey }),
  })
}

export function fetchCommunityAccount(token: string): Promise<CommunityAccount> {
  return communityRequest('/api/community/me', token)
}

export function fetchCommunityDashboard(): Promise<CommunityDashboard> {
  return communityRequest('/api/community/dashboard')
}

export function fetchCommunityTasks(
  token: string,
  all = false,
): Promise<CommunityTask[]> {
  return communityRequest<{ tasks: CommunityTask[] }>(
    all ? '/api/community/tasks/all' : '/api/community/tasks',
    token,
  ).then((response) => response.tasks)
}

export function submitCommunityTask(
  token: string,
  taskId: string,
  answer: CommunityAnswer,
  durationMs: number,
): Promise<{
  submissionId: string
  taskStatus: CommunityTask['status']
  consensus?: CommunityTask['consensus']
  calibration?: { correct: boolean }
}> {
  return communityRequest(
    `/api/community/tasks/${encodeURIComponent(taskId)}/submissions`,
    token,
    {
      method: 'POST',
      body: JSON.stringify({ answer, confidence: 1, durationMs }),
    },
  )
}

export function resolveCommunityTask(
  token: string,
  taskId: string,
  answer: CommunityAnswer,
): Promise<CommunityTask> {
  return communityRequest(
    `/api/community/tasks/${encodeURIComponent(taskId)}/resolve`,
    token,
    {
      method: 'POST',
      body: JSON.stringify({ answer }),
    },
  )
}

export function fetchCommunityContributions(
  token: string,
): Promise<CommunityContribution[]> {
  return communityRequest<{ contributions: CommunityContribution[] }>(
    '/api/community/contributions',
    token,
  ).then((response) => response.contributions)
}

export function withdrawCommunityContribution(
  token: string,
  submissionId: string,
): Promise<void> {
  return communityRequest(
    `/api/community/submissions/${encodeURIComponent(submissionId)}/withdraw`,
    token,
    { method: 'POST' },
  )
}

export function importCommunityReviews(
  token: string,
  queue: 'residual-onset' | 'residual-onset-v2',
): Promise<{ addedTasks: number }> {
  return communityRequest('/api/community/tasks/import-reviews', token, {
    method: 'POST',
    body: JSON.stringify({ queue }),
  })
}

export function fetchDatasetReleases(): Promise<DatasetRelease[]> {
  return communityRequest<{ releases: DatasetRelease[] }>(
    '/api/community/releases',
  ).then((response) => response.releases)
}

export function createDatasetRelease(
  token: string,
  name: string,
): Promise<DatasetRelease> {
  return communityRequest('/api/community/releases', token, {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
}

export function fetchTrainingRecipes(): Promise<{
  recipes: TrainingRecipe[]
  builtinDatasets: Array<{ id: string; name: string }>
}> {
  return communityRequest('/api/community/recipes')
}

export function fetchExperiments(token: string): Promise<ExperimentRun[]> {
  return communityRequest<{ experiments: ExperimentRun[] }>(
    '/api/community/experiments',
    token,
  ).then((response) => response.experiments)
}

export function createExperiment(
  token: string,
  recipeId: string,
  datasetReleaseId: string,
  parameters: Record<string, number>,
): Promise<ExperimentRun> {
  return communityRequest('/api/community/experiments', token, {
    method: 'POST',
    body: JSON.stringify({ recipeId, datasetReleaseId, parameters }),
  })
}

export function fetchPromotions(token: string): Promise<ModelPromotion[]> {
  return communityRequest<{ promotions: ModelPromotion[] }>(
    '/api/community/promotions',
    token,
  ).then((response) => response.promotions)
}

export function updateExperiment(
  token: string,
  experimentId: string,
  input: {
    status: 'running' | 'completed' | 'failed'
    metrics?: Record<string, number>
    artifactSha256?: string
    failureReason?: string
  },
): Promise<ExperimentRun> {
  return communityRequest(
    `/api/community/experiments/${encodeURIComponent(experimentId)}`,
    token,
    {
      method: 'PATCH',
      body: JSON.stringify(input),
    },
  )
}

export function retryExperiment(
  token: string,
  experimentId: string,
): Promise<ExperimentRun> {
  return communityRequest(
    `/api/community/experiments/${encodeURIComponent(experimentId)}/retry`,
    token,
    { method: 'POST' },
  )
}

export function reproduceExperiment(
  token: string,
  experimentId: string,
): Promise<ExperimentRun> {
  return communityRequest(
    `/api/community/experiments/${encodeURIComponent(experimentId)}/reproduce`,
    token,
    { method: 'POST' },
  )
}

export function createPromotion(
  token: string,
  experimentId: string,
): Promise<ModelPromotion> {
  return communityRequest('/api/community/promotions', token, {
    method: 'POST',
    body: JSON.stringify({ experimentId }),
  })
}

export function recordPromotionCheck(
  token: string,
  promotionId: string,
  input: {
    check:
      | 'lineage'
      | 'reproduction'
      | 'public-validation'
      | 'sealed-evaluation'
      | 'robustness'
      | 'shadow'
      | 'canary'
    passed: boolean
    summary: string
  },
): Promise<ModelPromotion> {
  return communityRequest(
    `/api/community/promotions/${encodeURIComponent(promotionId)}/checks`,
    token,
    {
      method: 'POST',
      body: JSON.stringify(input),
    },
  )
}

export function advancePromotion(
  token: string,
  promotionId: string,
  target: 'shadow' | 'canary' | 'champion',
): Promise<ModelPromotion> {
  return communityRequest(
    `/api/community/promotions/${encodeURIComponent(promotionId)}/advance`,
    token,
    {
      method: 'POST',
      body: JSON.stringify({ target }),
    },
  )
}

export function rollbackChampion(
  token: string,
  reason: string,
): Promise<ModelPromotion> {
  return communityRequest('/api/community/promotions/rollback', token, {
    method: 'POST',
    body: JSON.stringify({ reason }),
  })
}

export function fetchInferenceDeployment(
  token: string,
): Promise<InferenceDeploymentStatus> {
  return communityRequest('/api/community/deployment', token)
}

export function updateInferenceDeployment(
  token: string,
  config: InferenceDeploymentStatus['config'],
): Promise<InferenceDeploymentStatus> {
  return communityRequest('/api/community/deployment', token, {
    method: 'PUT',
    body: JSON.stringify(config),
  })
}
