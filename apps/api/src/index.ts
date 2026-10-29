import { createHash, randomUUID, timingSafeEqual } from 'node:crypto'
import {
  closeSync,
  createReadStream,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  realpathSync,
  renameSync,
  statSync,
  unlinkSync,
} from 'node:fs'
import { relative, resolve } from 'node:path'
import cors from 'cors'
import express from 'express'
import multer from 'multer'
import { createDeploymentAlertSink } from './alerts.js'
import {
  importAuthorizedMediaKeyDatabase,
  type SeparatedKeyProvider,
} from './authorized-media-keys.js'
import { builtinDemoMediaPaths } from './builtin-demo.js'
import {
  CommunityRequestError,
  createCommunityStore,
} from './community.js'
import {
  authorizedMediaProviderForContent,
  normalizeUploadedFilename,
  requiresAuthorizedMediaDecode,
  scoreImageContentValidationError,
  scoreUploadPrefilterError,
  uploadContentValidationError,
  uploadPrefilterError,
} from './media.js'
import { PipelineInferenceProvider } from './inference.js'
import { getCapabilities, platforms, sourceFromLink } from './pipeline.js'
import { importScoreImages } from './score-import.js'
import { unlinkIfExists } from './managed-files.js'
import {
  analysisRequiredBytes,
  hasAnalysisCapacity,
  runtimeDataRoot,
} from './runtime.js'
import {
  defaultReviewRoot,
  getReview,
  listReviews,
  ReviewRequestError,
} from './reviews.js'
import { StorageCleanupService } from './storage-cleanup.js'
import {
  cancelJob,
  createJob,
  getDemoProject,
  getJob,
  getProject,
  listJobs,
  listProjectRevisions,
  ProjectRevisionConflictError,
  removeProject,
  restoreProjectRevision,
  saveProjectRevision,
  shutdownJobs,
  storageCleanupProtection,
} from './store.js'

const port = Number(process.env.PORT ?? 8787)
const dataRoot = runtimeDataRoot()
const uploadRoot = resolve(dataRoot, 'uploads')
const trainingArtifactRoot = resolve(dataRoot, 'community', 'artifacts')
const trainingArtifactIncomingRoot = resolve(trainingArtifactRoot, '.incoming')
const mediaBrowserIds = [
  'brave',
  'chrome',
  'chromium',
  'edge',
  'firefox',
  'opera',
  'safari',
  'vivaldi',
] as const
const mediaSessionRoot = process.env.GTS_MEDIA_SESSION_ROOT
mkdirSync(uploadRoot, { recursive: true })
mkdirSync(trainingArtifactIncomingRoot, { recursive: true })

function desktopSessionFile(
  value: unknown,
  suffix: string,
  maximumBytes: number,
): string | undefined {
  if (value === undefined) return undefined
  if (
    process.env.GTS_DESKTOP_LOCAL !== '1' ||
    !mediaSessionRoot ||
    typeof value !== 'string' ||
    !value.endsWith(suffix)
  ) {
    throw new Error('无效的一次性登录授权')
  }
  const root = realpathSync(mediaSessionRoot)
  const path = realpathSync(value)
  const relativePath = relative(root, path)
  const stats = statSync(path)
  if (
    !relativePath ||
    relativePath.startsWith('..') ||
    stats.size <= 0 ||
    stats.size > maximumBytes
  ) {
    throw new Error('一次性登录授权文件无效')
  }
  return path
}

const deploymentAlertUrl = process.env.GTS_DEPLOYMENT_ALERT_URL
const deploymentAlertSecret = process.env.GTS_DEPLOYMENT_ALERT_SECRET
if (Boolean(deploymentAlertUrl) !== Boolean(deploymentAlertSecret)) {
  throw new Error(
    'GTS_DEPLOYMENT_ALERT_URL and GTS_DEPLOYMENT_ALERT_SECRET must be configured together',
  )
}
const deploymentAlertSink =
  deploymentAlertUrl && deploymentAlertSecret
    ? createDeploymentAlertSink({
        url: deploymentAlertUrl,
        secret: deploymentAlertSecret,
      })
    : undefined

function sha256Buffer(value: Buffer): string {
  return createHash('sha256').update(value).digest('hex')
}

function builtinDatasetManifestSha256(): Record<string, string> {
  const manifests = {
    'builtin:guitarset-v1': resolve(
      dataRoot,
      'training',
      'guitarset',
      'manifest.jsonl',
    ),
    'builtin:synthetic-smoke-v1': resolve(
      dataRoot,
      'training',
      'synthetic-smoke',
      'manifest.jsonl',
    ),
  }
  return Object.fromEntries(
    Object.entries(manifests)
      .filter(([, path]) => existsSync(path))
      .map(([id, path]) => [id, sha256Buffer(readFileSync(path))]),
  )
}

function sha256File(path: string): Promise<string> {
  return new Promise((resolveHash, rejectHash) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('error', rejectHash)
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

function trainingRunnerCredentials(value: string | undefined): Record<string, string> {
  if (!value) return {}
  try {
    const parsed = JSON.parse(value) as unknown
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('expected an object')
    }
    return Object.fromEntries(
      Object.entries(parsed).map(([runnerId, secret]) => {
        if (typeof secret !== 'string') throw new Error('secret must be a string')
        return [runnerId, secret]
      }),
    )
  } catch {
    throw new Error('GTS_TRAINING_RUNNER_CREDENTIALS 必须是 JSON 字符串映射')
  }
}

function promotionGateCredentials(value: string | undefined): Record<string, string> {
  if (!value) return {}
  try {
    const parsed = JSON.parse(value) as unknown
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('expected an object')
    }
    return Object.fromEntries(
      Object.entries(parsed).map(([gateId, secret]) => {
        if (typeof secret !== 'string') throw new Error('secret must be a string')
        return [gateId, secret]
      }),
    )
  } catch {
    throw new Error('GTS_PROMOTION_GATE_CREDENTIALS 必须是 JSON 字符串映射')
  }
}

const community = createCommunityStore(resolve(dataRoot, 'community'), {
  bootstrapKey:
    process.env.GTS_COMMUNITY_BOOTSTRAP_KEY ??
    (process.env.NODE_ENV === 'production' ? randomUUID() : undefined),
  runnerCredentials: trainingRunnerCredentials(
    process.env.GTS_TRAINING_RUNNER_CREDENTIALS,
  ),
  gateCredentials: promotionGateCredentials(
    process.env.GTS_PROMOTION_GATE_CREDENTIALS,
  ),
  sealedGateCredentials: promotionGateCredentials(
    process.env.GTS_SEALED_GATE_CREDENTIALS,
  ),
  runnerLeaseSeconds: Number(process.env.GTS_TRAINING_LEASE_SECONDS ?? 300),
  builtinDatasetManifestSha256: builtinDatasetManifestSha256(),
  onOperationalAlert: deploymentAlertSink,
})
const storageCleanup = new StorageCleanupService({
  dataRoot,
  protection: storageCleanupProtection,
})

const upload = multer({
  dest: uploadRoot,
  limits: {
    fileSize: 500 * 1024 * 1024,
  },
  fileFilter: (_request, file, callback) => {
    const validationError = uploadPrefilterError(file.originalname, file.mimetype)
    if (!validationError) {
      callback(null, true)
      return
    }
    callback(new Error(validationError))
  },
})

const scoreUpload = multer({
  dest: uploadRoot,
  limits: {
    files: 24,
    fileSize: 24 * 1024 * 1024,
  },
  fileFilter: (_request, file, callback) => {
    const validationError = scoreUploadPrefilterError(
      file.originalname,
      file.mimetype,
    )
    if (!validationError) {
      callback(null, true)
      return
    }
    callback(new Error(validationError))
  },
})

const mediaKeyUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    files: 1,
    fileSize: 128 * 1024 * 1024,
  },
})

const trainingArtifactUpload = multer({
  dest: trainingArtifactIncomingRoot,
  limits: {
    files: 1,
    fileSize: 512 * 1024 * 1024,
  },
})

const app = express()
app.disable('x-powered-by')
const allowedOrigins = new Set([
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  ...(process.env.GTS_DESKTOP_LOCAL === '1' ? ['null'] : []),
])
const localDevelopmentOrigin =
  /^http:\/\/(?:localhost|127\.0\.0\.1):[1-9][0-9]{0,4}$/u
app.use(
  cors({
    credentials: true,
    origin(origin, callback) {
      if (
        !origin ||
        allowedOrigins.has(origin) ||
        (process.env.NODE_ENV !== 'production' &&
          localDevelopmentOrigin.test(origin))
      ) {
        callback(null, true)
        return
      }
      callback(new Error('Origin is not allowed'))
    },
  }),
)
app.use(express.json({ limit: '8mb' }))

if (process.env.GTS_DESKTOP_API_BUILD !== '1') {
  const { registerAdminApi } = await import('./admin-api.js')
  registerAdminApi(app, dataRoot)
}

function reviewQueueConfig(value: unknown): {
  root: string
  mediaPrefix: string
} {
  if (value === undefined || value === 'reference') {
    return {
      root: defaultReviewRoot(),
      mediaPrefix: '/media/training/reviews',
    }
  }
  if (value === 'residual-onset' || value === 'residual-onset-v2') {
    return {
      root: resolve(defaultReviewRoot(), value),
      mediaPrefix: `/media/training/reviews/${value}`,
    }
  }
  throw new ReviewRequestError('审核队列无效')
}

function communityAccount(request: express.Request) {
  return community.authenticate(request.headers.authorization)
}

function runtimeAdminAuthorized(request: express.Request): boolean {
  if (process.env.GTS_DESKTOP_LOCAL === '1') return true
  const expected = process.env.GTS_RUNTIME_ADMIN_TOKEN
  const provided = request.headers.authorization?.replace(/^Bearer\s+/iu, '')
  if (!expected || !provided) return false
  const expectedBytes = Buffer.from(expected)
  const providedBytes = Buffer.from(provided)
  return (
    expectedBytes.length === providedBytes.length &&
    timingSafeEqual(expectedBytes, providedBytes)
  )
}

function requireRuntimeAdmin(
  request: express.Request,
  response: express.Response,
): boolean {
  if (runtimeAdminAuthorized(request)) return true
  response.status(403).json({ error: '需要运行时管理员权限' })
  return false
}

function trainingRunner(request: express.Request) {
  return community.authenticateRunner(request.headers.authorization)
}

function promotionGate(request: express.Request) {
  return community.authenticatePromotionGate(request.headers.authorization)
}

function sealedPromotionGate(request: express.Request) {
  return community.authenticateSealedPromotionGate(
    request.headers.authorization,
  )
}

function inferenceExecution(jobId: string) {
  return {
    deployment: community.inferenceDeployment(jobId),
    onInferenceObservation: (
      observation: Parameters<
        typeof community.recordInferenceObservation
      >[0],
    ) => {
      community.recordInferenceObservation(observation)
    },
  }
}

const inferenceProvider = new PipelineInferenceProvider(inferenceExecution)

function sendCommunityArtifact(
  objectKey: string,
  response: express.Response,
): void {
  const communityRoot = resolve(dataRoot, 'community')
  const artifactPath = resolve(communityRoot, objectKey)
  const relativePath = relative(communityRoot, artifactPath)
  if (relativePath.startsWith('..') || relativePath === '') {
    throw new CommunityRequestError('实验制品路径无效', 500)
  }
  response.type('application/octet-stream').sendFile(artifactPath)
}

app.use(
  '/media/jobs',
  express.static(resolve(dataRoot, 'jobs'), { dotfiles: 'deny', index: false }),
)

app.get('/media/builtin/:filename', (request, response) => {
  const path = builtinDemoMediaPaths.get(request.params.filename)
  if (!path) {
    response.status(404).json({ error: '内置音频不存在' })
    return
  }
  const contentType = request.params.filename.endsWith('.flac')
    ? 'audio/flac'
    : request.params.filename.endsWith('.m4a')
      ? 'audio/mp4'
      : request.params.filename.endsWith('.wav')
        ? 'audio/wav'
        : 'audio/mpeg'
  response
    .type(contentType)
    .set('Cache-Control', 'public, max-age=86400')
    .sendFile(path)
})

app.get('/health', (_request, response) => {
  response.json({
    status: 'ok',
    service: 'guitar-tab-studio-api',
    capabilities: getCapabilities(),
  })
})

app.get('/api/platforms', (_request, response) => {
  response.json({ platforms })
})

app.get('/api/runtime/storage-cleanup', (request, response) => {
  if (!requireRuntimeAdmin(request, response)) return
  response.json(storageCleanup.status())
})

app.put('/api/runtime/storage-cleanup', (request, response) => {
  if (!requireRuntimeAdmin(request, response)) return
  response.json(storageCleanup.updateConfig(request.body))
})

app.post('/api/runtime/storage-cleanup/run', (request, response) => {
  if (!requireRuntimeAdmin(request, response)) return
  storageCleanup.run('manual', { force: true })
  response.json(storageCleanup.status())
})

app.post(
  '/api/runtime/authorized-media-keys/:provider',
  mediaKeyUpload.single('database'),
  async (request, response) => {
    if (!requireRuntimeAdmin(request, response)) return
    if (process.env.GTS_DESKTOP_LOCAL !== '1') {
      response.status(403).json({
        error: '密钥库导入仅允许在桌面本地运行时使用',
      })
      return
    }
    const provider = String(request.params.provider)
    if (provider !== 'kuwo' && provider !== 'kugou') {
      response.status(400).json({ error: '不支持的密钥库类型' })
      return
    }
    if (!request.file) {
      response.status(400).json({ error: '请选择密钥库文件' })
      return
    }
    try {
      const count = await importAuthorizedMediaKeyDatabase(
        provider as SeparatedKeyProvider,
        request.file.buffer,
      )
      response.status(201).json({ provider, count })
    } catch (error) {
      response.status(400).json({
        error:
          error instanceof Error ? error.message : '密钥库解析失败',
      })
    }
  },
)

app.get('/api/projects/demo', (_request, response) => {
  response.json(getDemoProject())
})

app.get('/api/projects/:id', (request, response) => {
  const project = getProject(request.params.id)
  if (!project) {
    response.status(404).json({ error: '项目不存在' })
    return
  }
  response.json(project)
})

app.delete('/api/projects/:id', (request, response) => {
  response.json({ removed: removeProject(request.params.id) })
})

app.get('/api/projects/:id/revisions', (request, response) => {
  response.json({ revisions: listProjectRevisions(request.params.id) })
})

app.post('/api/projects/:id/revisions', (request, response) => {
  try {
    if (request.body?.project?.id !== request.params.id) {
      response.status(400).json({ error: '项目标识不一致' })
      return
    }
    response.status(201).json(
      saveProjectRevision(
        request.body.project,
        request.body.label,
        request.body.kind,
        request.body.expectedParentRevisionId,
      ),
    )
  } catch (error) {
    response.status(
      error instanceof ProjectRevisionConflictError ? 409 : 400,
    ).json({
      error: error instanceof Error ? error.message : '保存修订失败',
    })
  }
})

app.post(
  '/api/projects/:id/revisions/:revisionId/restore',
  (request, response) => {
    try {
      const revision = restoreProjectRevision(
        request.params.id,
        request.params.revisionId,
      )
      if (!revision) {
        response.status(404).json({ error: '项目修订不存在' })
        return
      }
      response.status(201).json(revision)
    } catch (error) {
      response.status(400).json({
        error: error instanceof Error ? error.message : '恢复修订失败',
      })
    }
  },
)

app.get('/api/jobs', (_request, response) => {
  response.json({ jobs: listJobs() })
})

app.get('/api/jobs/:id', (request, response) => {
  const job = getJob(request.params.id)
  if (!job) {
    response.status(404).json({ error: '任务不存在' })
    return
  }
  response.json(job)
})

app.post('/api/jobs/:id/cancel', (request, response) => {
  const job = cancelJob(request.params.id)
  if (!job) {
    response.status(404).json({ error: '任务不存在' })
    return
  }
  response.json(job)
})

app.post('/api/community/accounts', (request, response) => {
  const credential = community.registerAccount({
    alias: request.body?.alias,
    bootstrapKey: request.body?.bootstrapKey,
    riskHint: String(
      request.headers['x-community-device'] ??
        request.ip ??
        request.socket.remoteAddress ??
        'unknown',
    ),
  })
  response.status(201).json(credential)
})

app.get('/api/community/me', (request, response) => {
  const account = communityAccount(request)
  response.json(community.getAccount(account))
})

app.get('/api/community/dashboard', (_request, response) => {
  response.json(community.dashboard())
})

app.get('/api/community/tasks', (request, response) => {
  const account = communityAccount(request)
  response.json({ tasks: community.listTasks(account) })
})

app.get('/api/community/tasks/all', (request, response) => {
  const account = communityAccount(request)
  response.json({ tasks: community.listAllTasks(account) })
})

app.post('/api/community/tasks/import-reviews', (request, response) => {
  const account = communityAccount(request)
  const queueName =
    typeof request.body?.queue === 'string' ? request.body.queue : 'reference'
  const queue = reviewQueueConfig(queueName)
  const requestedNames: string[] = Array.isArray(request.body?.names)
    ? request.body.names.filter(
        (value: unknown): value is string => typeof value === 'string',
      )
    : listReviews(queue.root).map((review) => review.name)
  const imported: Array<{ sourceId: string; addedTasks: number }> =
    requestedNames.map((name: string) =>
      community.importReview(
        account,
        getReview(name, queue.root, queue.mediaPrefix),
        queueName,
      ),
    )
  response.status(201).json({
    imported,
    addedTasks: imported.reduce((total, item) => total + item.addedTasks, 0),
  })
})

app.post('/api/community/tasks/:id/submissions', (request, response) => {
  const account = communityAccount(request)
  response.status(201).json(
    community.submit(account, request.params.id, {
      answer: request.body?.answer,
      confidence: request.body?.confidence,
      durationMs: request.body?.durationMs,
    }),
  )
})

app.post('/api/community/tasks/:id/resolve', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.resolveTask(account, request.params.id, request.body?.answer),
  )
})

app.get('/api/community/contributions', (request, response) => {
  const account = communityAccount(request)
  response.json({ contributions: community.listContributions(account) })
})

app.post('/api/community/submissions/:id/withdraw', (request, response) => {
  const account = communityAccount(request)
  community.withdrawSubmission(account, request.params.id)
  response.status(204).end()
})

app.post('/api/community/sources/:id/withdraw', (request, response) => {
  const account = communityAccount(request)
  community.withdrawSource(account, request.params.id, request.body?.reason)
  response.status(204).end()
})

app.get('/api/community/releases', (_request, response) => {
  response.json({ releases: community.listReleases() })
})

app.get('/api/community/releases/:id/manifest', (request, response) => {
  const account = communityAccount(request)
  response.json(community.releaseManifest(account, request.params.id))
})

app.post('/api/community/releases', (request, response) => {
  const account = communityAccount(request)
  response.status(201).json(
    community.createRelease(account, {
      name: request.body?.name,
      taskIds: request.body?.taskIds,
    }),
  )
})

app.get('/api/community/recipes', (_request, response) => {
  response.json({
    recipes: community.listRecipes(),
    builtinDatasets: community.listBuiltinDatasets(),
  })
})

app.get('/api/community/experiments', (request, response) => {
  const account = communityAccount(request)
  response.json({ experiments: community.listExperiments(account) })
})

app.post('/api/community/experiments', (request, response) => {
  const account = communityAccount(request)
  response.status(201).json(
    community.createExperiment(account, {
      recipeId: request.body?.recipeId,
      datasetReleaseId: request.body?.datasetReleaseId,
      parameters: request.body?.parameters,
    }),
  )
})

app.patch('/api/community/experiments/:id', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.recordExperiment(account, request.params.id, {
      status: request.body?.status,
      metrics: request.body?.metrics,
      artifactSha256: request.body?.artifactSha256,
      failureReason: request.body?.failureReason,
    }),
  )
})

app.post('/api/community/experiments/:id/retry', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.retryExperiment(account, String(request.params.id)),
  )
})

app.post('/api/community/experiments/:id/reproduce', (request, response) => {
  const account = communityAccount(request)
  response.status(201).json(
    community.reproduceExperiment(account, String(request.params.id)),
  )
})

app.get('/api/community/experiments/:id/artifact', (request, response) => {
  const account = communityAccount(request)
  const objectKey = community.experimentArtifactObjectKey(
    account,
    String(request.params.id),
  )
  sendCommunityArtifact(objectKey, response)
})

app.post('/api/community/runner/jobs/claim', (request, response) => {
  const runner = trainingRunner(request)
  const claim = community.claimExperiment(runner)
  if (!claim) {
    response.status(204).end()
    return
  }
  response.json(claim)
})

app.post(
  '/api/community/runner/jobs/:id/heartbeat',
  (request, response) => {
    const runner = trainingRunner(request)
    response.json(
      community.heartbeatExperiment(
        runner,
        request.params.id,
        request.body?.leaseToken,
      ),
    )
  },
)

app.get('/api/community/runner/jobs/:id/dataset-manifest', (request, response) => {
  const runner = trainingRunner(request)
  response.json(
    community.runnerDatasetManifest(
      runner,
      String(request.params.id),
      request.headers['x-gts-lease-token'],
    ),
  )
})

app.post(
  '/api/community/runner/jobs/:id/artifact',
  (request, _response, next) => {
    try {
      trainingRunner(request)
      next()
    } catch (error) {
      next(error)
    }
  },
  trainingArtifactUpload.single('artifact'),
  async (request, response, next) => {
    const incomingPath = request.file?.path
    let finalPath = ''
    try {
      const runner = trainingRunner(request)
      const leaseToken = request.headers['x-gts-lease-token']
      const experimentId = String(request.params.id)
      const policy = community.getRunnerArtifactUpload(
        runner,
        experimentId,
        leaseToken,
      )
      if (!request.file) {
        throw new CommunityRequestError('必须上传训练制品')
      }
      const sizeBytes = statSync(request.file.path).size
      if (sizeBytes <= 0 || sizeBytes > policy.maxBytes) {
        throw new CommunityRequestError('训练制品大小无效')
      }
      const artifactSha256 = await sha256File(request.file.path)
      const artifactDirectory = resolve(
        trainingArtifactRoot,
        policy.experimentId,
        String(policy.attempt),
      )
      mkdirSync(artifactDirectory, { recursive: true })
      finalPath = resolve(artifactDirectory, `${artifactSha256}.bin`)
      renameSync(request.file.path, finalPath)
      response.status(201).json(
        community.registerRunnerArtifact(
          runner,
          experimentId,
          leaseToken,
          {
            sha256: artifactSha256,
            sizeBytes,
            mediaType: policy.expectedMediaType,
            objectKey: `artifacts/${policy.experimentId}/${policy.attempt}/${artifactSha256}.bin`,
          },
        ),
      )
    } catch (error) {
      for (const path of [incomingPath, finalPath]) {
        if (!path) continue
        try {
          unlinkSync(path)
        } catch {
          // The upload may already have been moved or removed.
        }
      }
      next(error)
    }
  },
)

app.post('/api/community/runner/jobs/:id/result', (request, response) => {
  const runner = trainingRunner(request)
  response.json(
    community.recordRunnerResult(runner, request.params.id, {
      leaseToken: request.body?.leaseToken,
      status: request.body?.status,
      metrics: request.body?.metrics,
      artifactSha256: request.body?.artifactSha256,
      executionEvidence: request.body?.executionEvidence,
      failureReason: request.body?.failureReason,
    }),
  )
})

app.get('/api/community/promotions', (request, response) => {
  const account = communityAccount(request)
  response.json({ promotions: community.listPromotions(account) })
})

app.get('/api/community/promotions/:id/model-card', (request, response) => {
  response.json(community.modelCard(String(request.params.id)))
})

app.post('/api/community/promotions', (request, response) => {
  const account = communityAccount(request)
  response.status(201).json(
    community.createPromotion(account, request.body?.experimentId),
  )
})

app.post('/api/community/promotions/:id/checks', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.recordPromotionCheck(account, request.params.id, {
      check: request.body?.check,
      passed: request.body?.passed,
      summary: request.body?.summary,
    }),
  )
})

app.post('/api/community/promotions/:id/advance', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.advancePromotion(account, request.params.id, request.body?.target),
  )
})

app.post('/api/community/promotions/rollback', (request, response) => {
  const account = communityAccount(request)
  response.json(community.rollbackChampion(account, request.body?.reason))
})

app.get('/api/community/deployment', (request, response) => {
  const account = communityAccount(request)
  response.json(community.inferenceDeploymentStatus(account))
})

app.put('/api/community/deployment', (request, response) => {
  const account = communityAccount(request)
  response.json(
    community.configureInferenceDeployment(account, {
      shadowSamplePercent: request.body?.shadowSamplePercent,
      canaryTrafficPercent: request.body?.canaryTrafficPercent,
      errorBudgetPercent: request.body?.errorBudgetPercent,
      minimumObservations: request.body?.minimumObservations,
    }),
  )
})

app.put('/api/community/governance-mode', (request, response) => {
  const account = communityAccount(request)
  response.json({ mode: community.setGovernanceMode(account, request.body?.mode) })
})

app.get('/api/community/audit', (request, response) => {
  const account = communityAccount(request)
  response.json({ events: community.auditEvents(account) })
})

app.get('/api/community/gates/promotions', (request, response) => {
  const agent = promotionGate(request)
  response.json({ jobs: community.listPromotionGateJobs(agent) })
})

app.post(
  '/api/community/gates/promotions/:id/checks',
  (request, response) => {
    const agent = promotionGate(request)
    response.json(
      community.recordAutomatedPromotionCheck(
        agent,
        String(request.params.id),
        {
          check: request.body?.check,
          passed: request.body?.passed,
          summary: request.body?.summary,
          evidenceSha256: request.body?.evidenceSha256,
        },
      ),
    )
  },
)

app.get(
  '/api/community/gates/experiments/:id/artifact',
  (request, response) => {
    const agent = promotionGate(request)
    const objectKey = community.promotionGateArtifactObjectKey(
      agent,
      String(request.params.id),
    )
    sendCommunityArtifact(objectKey, response)
  },
)

app.post(
  '/api/community/sealed-gates/promotions/:id/checks',
  (request, response) => {
    const agent = sealedPromotionGate(request)
    response.json(
      community.recordAutomatedPromotionCheck(
        agent,
        String(request.params.id),
        {
          check: 'sealed-evaluation',
          passed: request.body?.passed,
          summary: request.body?.summary,
          evidenceSha256: request.body?.evidenceSha256,
        },
      ),
    )
  },
)

app.get('/api/community/sealed-gates/promotions', (request, response) => {
  const agent = sealedPromotionGate(request)
  response.json({ jobs: community.listSealedPromotionGateJobs(agent) })
})

app.get(
  '/api/community/sealed-gates/experiments/:id/artifact',
  (request, response) => {
    const agent = sealedPromotionGate(request)
    const objectKey = community.promotionGateArtifactObjectKey(
      agent,
      String(request.params.id),
    )
    sendCommunityArtifact(objectKey, response)
  },
)

app.post('/api/import/link', (request, response) => {
  const url = typeof request.body?.url === 'string' ? request.body.url.trim() : ''
  if (!url) {
    response.status(400).json({ error: '缺少链接' })
    return
  }

  if (!hasAnalysisCapacity(0)) {
    storageCleanup.run('capacity', {
      requiredBytes: analysisRequiredBytes(0),
    })
  }
  if (!hasAnalysisCapacity(0)) {
    response.status(507).json({
      error: '可用磁盘空间不足，自动清理后仍无法安全开始分析',
    })
    return
  }

  let cookieFile: string | undefined
  try {
    const source = sourceFromLink(url)
    const requestedBrowser = request.body?.mediaSession?.browser
    cookieFile = desktopSessionFile(
      request.body?.mediaSession?.cookieFile,
      '.cookies.txt',
      2 * 1024 * 1024,
    )
    if (requestedBrowser !== undefined) {
      if (process.env.GTS_DESKTOP_LOCAL !== '1') {
        throw new Error('浏览器登录态仅可用于本机媒体获取')
      }
      if (
        typeof requestedBrowser !== 'string' ||
        !mediaBrowserIds.some((browser) => browser === requestedBrowser)
      ) {
        throw new Error('不支持该浏览器登录态')
      }
      Object.defineProperty(source, 'mediaSession', {
        value: {
          browser: requestedBrowser as (typeof mediaBrowserIds)[number],
          ...(cookieFile ? { cookieFile } : {}),
        },
        enumerable: false,
        configurable: false,
        writable: false,
      })
    }
    const authorizedProcessing =
      request.body?.authorizedProcessingConsent === true &&
      request.body?.authorizationPolicyVersion === 2
    const protectedProvider =
      source.kind === 'qqmusic' || source.kind === 'netease'
        ? source.kind
        : null
    if (authorizedProcessing && protectedProvider) {
      source.authorizedMedia = {
        provider: protectedProvider,
        purpose: 'analysis-and-score-generation',
        decodedOutput: 'ephemeral',
        policyVersion: 2,
      }
    }
    const job = createJob(source, undefined, inferenceProvider, {
      onSettled: () => {
        if (cookieFile) unlinkIfExists(cookieFile)
        storageCleanup.run('job-settled')
      },
    })
    response.status(202).json(job)
  } catch (error) {
    if (cookieFile) unlinkIfExists(cookieFile)
    response.status(400).json({
      error: error instanceof Error ? error.message : '链接无效',
    })
  }
})

app.post('/api/import/upload', upload.single('media'), (request, response) => {
  if (!request.file) {
    response.status(400).json({ error: '请选择音频或视频文件' })
    return
  }

  const header = Buffer.alloc(16)
  const descriptor = openSync(request.file.path, 'r')
  const bytesRead = readSync(descriptor, header, 0, header.length, 0)
  closeSync(descriptor)
  const originalName = normalizeUploadedFilename(
    request.body?.originalName,
    request.file.originalname,
  )
  const validationError = uploadContentValidationError(
    originalName,
    request.file.mimetype,
    header.subarray(0, bytesRead),
  )
  const requiresAuthorizedDecode = requiresAuthorizedMediaDecode(
    originalName,
    header.subarray(0, bytesRead),
  )
  const authorizedProcessing =
    request.body?.authorizedProcessingConsent === 'true' &&
    request.body?.authorizationPolicyVersion === '2'
  const protectedProvider =
    authorizedMediaProviderForContent(
      originalName,
      header.subarray(0, bytesRead),
    ) ?? 'unknown'
  if (requiresAuthorizedDecode && !authorizedProcessing) {
    unlinkIfExists(request.file.path)
    response.status(409).json({
      error: '该文件需要确认版权与处理声明后才能临时解码',
      code: 'authorized_media_consent_required',
      provider: protectedProvider,
    })
    return
  }
  if (validationError && !requiresAuthorizedDecode) {
    unlinkIfExists(request.file.path)
    response.status(400).json({ error: validationError })
    return
  }
  if (!hasAnalysisCapacity(request.file.size)) {
    storageCleanup.run('capacity', {
      requiredBytes: analysisRequiredBytes(request.file.size),
    })
  }
  if (!hasAnalysisCapacity(request.file.size)) {
    unlinkIfExists(request.file.path)
    response.status(507).json({
      error: '可用磁盘空间不足，自动清理后仍无法安全完成分轨与转录',
    })
    return
  }

  let qqMusicCredentialFile: string | undefined
  try {
    qqMusicCredentialFile = desktopSessionFile(
      request.body?.qqMusicCredentialFile,
      '.qqmusic.json',
      64 * 1024,
    )
    if (qqMusicCredentialFile && protectedProvider !== 'qqmusic') {
      throw new Error('QQ 音乐授权只能用于 QQ 音乐文件')
    }
  } catch (error) {
    unlinkIfExists(request.file.path)
    if (qqMusicCredentialFile) unlinkIfExists(qqMusicCredentialFile)
    response.status(400).json({
      error: error instanceof Error ? error.message : '一次性登录授权无效',
    })
    return
  }
  const source = {
    kind: 'upload' as const,
    label: '本地上传',
    filename: originalName,
    authorizedMedia: requiresAuthorizedDecode
      ? {
          provider: protectedProvider,
          purpose: 'analysis-and-score-generation' as const,
          decodedOutput: 'ephemeral' as const,
          policyVersion: 2 as const,
        }
      : undefined,
  }
  if (qqMusicCredentialFile) {
    Object.defineProperty(source, 'qqMusicCredentialFile', {
      value: qqMusicCredentialFile,
      enumerable: false,
      configurable: false,
      writable: false,
    })
  }
  const job = createJob(
    source,
    request.file.path,
    inferenceProvider,
    {
      removeInputOnSettled: true,
      onSettled: () => {
        if (qqMusicCredentialFile) unlinkIfExists(qqMusicCredentialFile)
        storageCleanup.run('job-settled')
      },
    },
  )
  response.status(202).json(job)
})

app.post(
  '/api/import/score',
  scoreUpload.array('score', 24),
  async (request, response) => {
    const files = Array.isArray(request.files)
      ? request.files as Express.Multer.File[]
      : []
    if (files.length === 0) {
      response.status(400).json({ error: '请选择曲谱图片或 PDF' })
      return
    }
    try {
      for (const file of files) {
        const header = Buffer.alloc(16)
        const descriptor = openSync(file.path, 'r')
        const bytesRead = readSync(descriptor, header, 0, header.length, 0)
        closeSync(descriptor)
        const validationError = scoreImageContentValidationError(
          file.originalname,
          file.mimetype,
          header.subarray(0, bytesRead),
        )
        if (validationError) throw new Error(validationError)
      }
      const originalName =
        typeof request.body?.originalName === 'string' &&
        request.body.originalName.trim()
          ? request.body.originalName.trim().slice(0, 255)
          : files[0].originalname
      const project = await importScoreImages(
        files.map((file) => file.path),
        originalName,
      )
      const revision = saveProjectRevision(project, '图片曲谱识别', 'import')
      response.status(201).json(revision.project)
    } catch (error) {
      response.status(400).json({
        error: error instanceof Error ? error.message : '曲谱识别失败',
      })
    } finally {
      for (const file of files) {
        try {
          unlinkSync(file.path)
        } catch {
          // The periodic cleanup will remove an upload that is still locked.
        }
      }
    }
  },
)

app.use(
  (
    error: Error,
    _request: express.Request,
    response: express.Response,
    _next: express.NextFunction,
  ) => {
    const candidateStatus =
      'statusCode' in error ? error.statusCode : undefined
    const statusCode =
      typeof candidateStatus === 'number' &&
      Number.isInteger(candidateStatus) &&
      candidateStatus >= 400 &&
      candidateStatus <= 599
        ? candidateStatus
        : 400
    response.status(statusCode).json({ error: error.message || '请求处理失败' })
  },
)

const server = app.listen(port, '127.0.0.1', () => {
  console.log(`Guitar Tab Studio API listening on http://127.0.0.1:${port}`)
})
const startupCleanupTimer = setTimeout(
  () => storageCleanup.run('startup'),
  1000,
)
startupCleanupTimer.unref()
const cleanupInterval = setInterval(
  () => storageCleanup.run('interval'),
  30 * 60 * 1000,
)
cleanupInterval.unref()

function shutdown(): void {
  clearTimeout(startupCleanupTimer)
  clearInterval(cleanupInterval)
  shutdownJobs()
  server.close(() => process.exit(0))
  setTimeout(() => process.exit(1), 5000).unref()
}

process.once('SIGTERM', shutdown)
process.once('SIGINT', shutdown)
