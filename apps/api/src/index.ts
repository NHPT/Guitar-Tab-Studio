import { createHash, randomUUID } from 'node:crypto'
import {
  closeSync,
  createReadStream,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  renameSync,
  statSync,
  unlinkSync,
} from 'node:fs'
import { relative, resolve } from 'node:path'
import cors from 'cors'
import express from 'express'
import multer from 'multer'
import {
  CommunityRequestError,
  createCommunityStore,
} from './community.js'
import {
  uploadContentValidationError,
  uploadPrefilterError,
} from './media.js'
import { getCapabilities, platforms, sourceFromLink } from './pipeline.js'
import {
  approveReview,
  defaultReviewRoot,
  getReview,
  listReviews,
  reopenReview,
  ReviewRequestError,
  saveReviewDraft,
} from './reviews.js'
import { createJob, getDemoProject, getJob, getProject, listJobs } from './store.js'

const port = Number(process.env.PORT ?? 8787)
const dataRoot = resolve(process.cwd(), 'data')
const uploadRoot = resolve(dataRoot, 'uploads')
const trainingArtifactRoot = resolve(dataRoot, 'community', 'artifacts')
const trainingArtifactIncomingRoot = resolve(trainingArtifactRoot, '.incoming')
mkdirSync(uploadRoot, { recursive: true })
mkdirSync(trainingArtifactIncomingRoot, { recursive: true })

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

const trainingArtifactUpload = multer({
  dest: trainingArtifactIncomingRoot,
  limits: {
    files: 1,
    fileSize: 512 * 1024 * 1024,
  },
})

const app = express()
app.disable('x-powered-by')
app.use(cors({ origin: ['http://localhost:5173', 'http://127.0.0.1:5173'] }))
app.use(express.json({ limit: '1mb' }))

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

function serveReviewAudio(
  queueName: unknown,
  filename: string,
  response: express.Response,
  next: express.NextFunction,
): void {
  try {
    if (!/^(?:real-reference|residual-onset)-\d{2}\.wav$/.test(filename)) {
      throw new ReviewRequestError('试听音频不存在', 404)
    }
    const queue = reviewQueueConfig(queueName)
    const reviewName = filename.slice(0, -4)
    const document = getReview(reviewName, queue.root, queue.mediaPrefix)
    if (document.review.review_audio !== filename) {
      throw new ReviewRequestError('试听音频不存在', 404)
    }
    response.sendFile(resolve(queue.root, filename))
  } catch (error) {
    next(error)
  }
}

app.use(
  '/media/jobs',
  express.static(resolve(dataRoot, 'jobs'), { dotfiles: 'deny', index: false }),
)

app.get('/media/training/reviews/:filename', (request, response, next) => {
  serveReviewAudio('reference', request.params.filename, response, next)
})

app.get(
  '/media/training/reviews/:queue/:filename',
  (request, response, next) => {
    serveReviewAudio(
      request.params.queue,
      request.params.filename,
      response,
      next,
    )
  },
)

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

app.get('/api/reviews', (request, response) => {
  const queue = reviewQueueConfig(request.query.queue)
  response.json({ reviews: listReviews(queue.root) })
})

app.get('/api/reviews/:name', (request, response) => {
  const queue = reviewQueueConfig(request.query.queue)
  response.json(
    getReview(request.params.name, queue.root, queue.mediaPrefix),
  )
})

app.put('/api/reviews/:name', (request, response) => {
  const queue = reviewQueueConfig(request.query.queue)
  response.json(
    saveReviewDraft(
      request.params.name,
      request.body,
      queue.root,
      queue.mediaPrefix,
    ),
  )
})

app.post('/api/reviews/:name/approve', (request, response) => {
  const queue = reviewQueueConfig(request.query.queue)
  response.json(
    approveReview(
      request.params.name,
      request.body,
      queue.root,
      queue.mediaPrefix,
    ),
  )
})

app.post('/api/reviews/:name/reopen', (request, response) => {
  const queue = reviewQueueConfig(request.query.queue)
  response.json(
    reopenReview(
      request.params.name,
      request.body?.reason,
      queue.root,
      queue.mediaPrefix,
    ),
  )
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

  try {
    const source = sourceFromLink(url)
    const job = createJob(source, undefined, inferenceExecution)
    response.status(202).json(job)
  } catch (error) {
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
  const validationError = uploadContentValidationError(
    request.file.originalname,
    request.file.mimetype,
    header.subarray(0, bytesRead),
  )
  if (validationError) {
    unlinkSync(request.file.path)
    response.status(400).json({ error: validationError })
    return
  }

  const job = createJob(
    {
      kind: 'upload',
      label: '本地上传',
      filename: request.file.originalname,
    },
    request.file.path,
    inferenceExecution,
  )
  response.status(202).json(job)
})

app.use(
  (
    error: Error,
    _request: express.Request,
    response: express.Response,
    _next: express.NextFunction,
  ) => {
    const statusCode =
      error instanceof ReviewRequestError || error instanceof CommunityRequestError
        ? error.statusCode
        : 400
    response.status(statusCode).json({ error: error.message || '请求处理失败' })
  },
)

app.listen(port, '127.0.0.1', () => {
  console.log(`Guitar Tab Studio API listening on http://127.0.0.1:${port}`)
})
