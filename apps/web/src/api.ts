import type {
  AnalysisJob,
  CapabilityState,
  CommunityAccount,
  CommunityAnswer,
  CommunityContribution,
  CommunityDashboard,
  CommunityTask,
  DatasetRelease,
  ExperimentRun,
  ModelPromotion,
  PlatformDescriptor,
  ReviewDocument,
  ReviewDraft,
  ReviewSummary,
  StudioProject,
  TrainingRecipe,
} from './types'

const API_ROOT =
  import.meta.env.VITE_API_ROOT ??
  (import.meta.env.DEV ? 'http://127.0.0.1:8787' : '')

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init)
  const body = (await response.json()) as T & { error?: string }

  if (!response.ok) {
    throw new Error(body.error ?? `请求失败 (${response.status})`)
  }

  return body
}

export async function fetchHealth(): Promise<CapabilityState> {
  const response = await jsonRequest<{ capabilities: CapabilityState }>('/health')
  return response.capabilities
}

export async function fetchPlatforms(): Promise<PlatformDescriptor[]> {
  const response = await jsonRequest<{ platforms: PlatformDescriptor[] }>('/api/platforms')
  return response.platforms
}

export function fetchDemoProject(): Promise<StudioProject> {
  return jsonRequest<StudioProject>('/api/projects/demo')
}

export function fetchProject(id: string): Promise<StudioProject> {
  return jsonRequest<StudioProject>(`/api/projects/${id}`)
}

export function fetchJob(id: string): Promise<AnalysisJob> {
  return jsonRequest<AnalysisJob>(`/api/jobs/${id}`)
}

export function submitLink(url: string): Promise<AnalysisJob> {
  return jsonRequest<AnalysisJob>('/api/import/link', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
}

export function submitUpload(file: File): Promise<AnalysisJob> {
  const body = new FormData()
  body.append('media', file)
  return jsonRequest<AnalysisJob>('/api/import/upload', {
    method: 'POST',
    body,
  })
}

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

export function apiAssetUrl(path: string): string {
  return `${API_ROOT}${path}`
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
