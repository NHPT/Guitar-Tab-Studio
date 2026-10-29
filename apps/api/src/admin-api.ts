import { timingSafeEqual } from 'node:crypto'
import { resolve } from 'node:path'
import type express from 'express'
import {
  AdminAuthError,
  AdminAuthStore,
  type AdminPermission,
  type AdminUser,
  adminHasPermission,
  adminSessionCookie,
  clearAdminSessionCookie,
} from './admin-auth.js'
import {
  createBeatBackup,
  listBeatBackups,
  restoreBeatBackup,
  verifyBeatBackup,
} from './beat-backups.js'
import {
  evaluateBeatGold,
  listBeatEvaluationRuns,
  saveBeatEvaluationRun,
} from './beat-evaluation.js'
import {
  beatGoldReleaseArchivePath,
  createBeatGoldRelease,
  listBeatGoldReleases,
} from './beat-gold-releases.js'
import {
  adjudicateBeatReview,
  beatGoldStatus,
  beatReviewAudioPath,
  compareBeatReviews,
  defaultBeatGoldRoot,
  getBeatReview,
  listBeatAdjudication,
  listBeatGold,
  listBeatReviews,
  saveBeatReviewDraft,
  submitBeatReview,
} from './beat-reviews.js'
import {
  beatStructureAudioPath,
  defaultBeatSourceRoot,
  getBeatStructureReview,
  listBeatStructureReviews,
  saveBeatStructureDraft,
  submitBeatStructureReview,
} from './beat-structure-reviews.js'
import {
  beatTrainingStatus,
  createBeatTrainingDataset,
  createBeatTrainingJob,
} from './beat-training.js'
import {
  approveReview,
  defaultReviewRoot,
  getReview,
  listReviews,
  reopenReview,
  ReviewRequestError,
  saveReviewDraft,
} from './reviews.js'
import {
  reviewLayerStatus,
  reviewLayerStatuses,
} from './review-layers.js'

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

export function registerAdminApi(
  app: express.Express,
  dataRoot: string,
): void {
  const beatGoldRoot = defaultBeatGoldRoot()
  const beatSourceRoot = defaultBeatSourceRoot(beatGoldRoot)
  const adminAuth = new AdminAuthStore(resolve(dataRoot, 'admin'))

  function requireAdmin(
    request: express.Request,
    response: express.Response,
    permission: AdminPermission,
  ): AdminUser | null {
    const user = adminAuth.authenticate(request.headers.cookie)
    if (!user) {
      response.status(401).json({ error: '请先登录管理后台' })
      return null
    }
    if (!adminHasPermission(user, permission)) {
      response.status(403).json({ error: '当前账号没有执行此操作的权限' })
      return null
    }
    return user
  }

  function requestedReviewer(
    user: AdminUser,
    value: unknown,
  ): string {
    const requested =
      typeof value === 'string' && value.trim()
        ? value.trim()
        : user.reviewerAlias ?? 'reviewer-a'
    if (
      user.role === 'reviewer' &&
      (!user.reviewerAlias || requested !== user.reviewerAlias)
    ) {
      throw new AdminAuthError('复核员只能访问已绑定的审核队列', 403)
    }
    return requested
  }

  function secureAdminCookie(request: express.Request): boolean {
    return (
      process.env.NODE_ENV === 'production' ||
      request.secure ||
      request.headers['x-forwarded-proto'] === 'https'
    )
  }

  function adminBootstrapAllowed(request: express.Request): boolean {
    if (process.env.NODE_ENV !== 'production') return true
    const expected = process.env.GTS_ADMIN_BOOTSTRAP_TOKEN
    const provided = request.headers['x-admin-bootstrap-token']
    if (!expected || typeof provided !== 'string') return false
    const expectedBytes = Buffer.from(expected)
    const providedBytes = Buffer.from(provided)
    return (
      expectedBytes.length === providedBytes.length &&
      timingSafeEqual(expectedBytes, providedBytes)
    )
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

  app.get(
    '/api/admin/beat-gold/media/:reviewer/:filename',
    (request, response, next) => {
      try {
        const user = requireAdmin(request, response, 'reviews:read')
        if (!user) return
        const reviewer = requestedReviewer(user, request.params.reviewer)
        const path = beatReviewAudioPath(
          reviewer,
          request.params.filename,
          beatGoldRoot,
        )
        response
          .type('audio/wav')
          .set('Cache-Control', 'private, no-store')
          .sendFile(path, { dotfiles: 'allow' })
      } catch (error) {
        next(error)
      }
    },
  )

  app.get(
    '/api/admin/beat-gold/structure-media/:recordingId',
    (request, response, next) => {
      try {
        if (!requireAdmin(request, response, 'reviews:read')) return
        const path = beatStructureAudioPath(
          request.params.recordingId,
          beatGoldRoot,
          beatSourceRoot,
        )
        response
          .type('audio/ogg')
          .set('Cache-Control', 'private, no-store')
          .sendFile(path, { dotfiles: 'allow' })
      } catch (error) {
        next(error)
      }
    },
  )

  app.get('/api/admin/auth/session', (request, response) => {
    response.json({
      configured: adminAuth.configured(),
      authenticated: Boolean(adminAuth.authenticate(request.headers.cookie)),
      user: adminAuth.authenticate(request.headers.cookie),
    })
  })

  app.post('/api/admin/auth/bootstrap', (request, response) => {
    if (!adminBootstrapAllowed(request)) {
      response.status(403).json({ error: '当前环境不允许直接初始化管理后台' })
      return
    }
    const session = adminAuth.bootstrap(request.body ?? {})
    response
      .set(
        'Set-Cookie',
        adminSessionCookie(session.token, secureAdminCookie(request)),
      )
      .status(201)
      .json({ user: session.user })
  })

  app.post('/api/admin/auth/login', (request, response) => {
    const session = adminAuth.login(request.body ?? {})
    response
      .set(
        'Set-Cookie',
        adminSessionCookie(session.token, secureAdminCookie(request)),
      )
      .json({ user: session.user })
  })

  app.post('/api/admin/auth/logout', (request, response) => {
    adminAuth.logout(request.headers.cookie)
    response
      .set(
        'Set-Cookie',
        clearAdminSessionCookie(secureAdminCookie(request)),
      )
      .status(204)
      .end()
  })

  app.get('/api/admin/users', (request, response) => {
    const user = requireAdmin(request, response, 'users:manage')
    if (!user) return
    response.json({ users: adminAuth.listUsers(user) })
  })

  app.post('/api/admin/users', (request, response) => {
    const user = requireAdmin(request, response, 'users:manage')
    if (!user) return
    response.status(201).json(adminAuth.createUser(user, request.body ?? {}))
  })

  app.put('/api/admin/users/:userId', (request, response) => {
    const user = requireAdmin(request, response, 'users:manage')
    if (!user) return
    response.json(
      adminAuth.updateUser(user, request.params.userId, request.body ?? {}),
    )
  })

  app.get('/api/admin/audit', (request, response) => {
    const user = requireAdmin(request, response, 'users:manage')
    if (!user) return
    response.json({ events: adminAuth.listAudit(user, request.query.limit) })
  })

  app.get('/api/admin/dashboard', (request, response) => {
    if (!requireAdmin(request, response, 'admin:read')) return
    const status = beatGoldStatus(beatGoldRoot)
    const reviewer = status.reviewers[0]
    const reviews = reviewer
      ? listBeatReviews(reviewer, beatGoldRoot)
      : []
    const counts = {
      draft: reviews.filter((review) => review.status === 'draft').length,
      submitted: reviews.filter((review) => review.status === 'submitted').length,
      adjudicated: listBeatGold(beatGoldRoot).length,
      checksComplete: reviews.filter((review) =>
        Object.values(review.checks).every(Boolean),
      ).length,
    }
    response.json({
      generatedAt: new Date().toISOString(),
      corpus: status.manifest,
      reviewer: reviewer ?? null,
      reviews: {
        total: reviews.length,
        ...counts,
      },
      layers: reviewLayerStatuses(beatGoldRoot).map((layer) => ({
        id: layer.id,
        label: layer.label,
        status: layer.status === 'active' ? 'ready' : 'blocked',
        dependsOn: layer.dependsOn,
      })),
      gates: {
        beatF1: 0.95,
        medianErrorMs: 20,
        p95ErrorMs: 50,
        downbeatF1: 0.95,
      },
    })
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

  app.get('/api/admin/beat-gold', (request, response) => {
    if (!requireAdmin(request, response, 'reviews:read')) return
    response.json(beatGoldStatus(beatGoldRoot))
  })

  app.get('/api/admin/review-layers', (request, response) => {
    if (!requireAdmin(request, response, 'admin:read')) return
    response.json({ layers: reviewLayerStatuses(beatGoldRoot) })
  })

  app.get('/api/admin/review-layers/:layerId', (request, response) => {
    if (!requireAdmin(request, response, 'admin:read')) return
    const layer = reviewLayerStatus(request.params.layerId, beatGoldRoot)
    if (!layer) {
      response.status(404).json({ error: '审核层不存在' })
      return
    }
    response.json(layer)
  })

  app.get('/api/admin/beat-gold/adjudication', (request, response) => {
    if (!requireAdmin(request, response, 'reviews:read')) return
    response.json({
      items: listBeatAdjudication(
        request.query.left ?? 'reviewer-a',
        request.query.right ?? 'reviewer-b',
        beatGoldRoot,
      ),
    })
  })

  app.get(
    '/api/admin/beat-gold/adjudication/:clipId',
    (request, response) => {
      if (!requireAdmin(request, response, 'reviews:read')) return
      response.json(
        compareBeatReviews(
          request.params.clipId,
          request.query.left ?? 'reviewer-a',
          request.query.right ?? 'reviewer-b',
          beatGoldRoot,
        ),
      )
    },
  )

  app.post(
    '/api/admin/beat-gold/adjudication/:clipId',
    (request, response) => {
      const user = requireAdmin(request, response, 'adjudication:write')
      if (!user) return
      response.status(201).json(
        adjudicateBeatReview(
          request.params.clipId,
          request.query.left ?? 'reviewer-a',
          request.query.right ?? 'reviewer-b',
          request.body,
          {
            adminUserId: user.id,
            adminUsername: user.username,
          },
          beatGoldRoot,
        ),
      )
    },
  )

  app.get('/api/admin/beat-training', (request, response) => {
    if (!requireAdmin(request, response, 'models:write')) return
    response.json(beatTrainingStatus(beatGoldRoot))
  })

  app.post('/api/admin/beat-training/datasets', (request, response) => {
    if (!requireAdmin(request, response, 'models:write')) return
    response.status(201).json(
      createBeatTrainingDataset(request.body?.version, beatGoldRoot),
    )
  })

  app.post('/api/admin/beat-training/jobs', (request, response) => {
    if (!requireAdmin(request, response, 'models:write')) return
    response.status(201).json(
      createBeatTrainingJob(request.body ?? {}, beatGoldRoot),
    )
  })

  app.get('/api/admin/beat-backups', (request, response) => {
    if (!requireAdmin(request, response, 'users:manage')) return
    response.json({ backups: listBeatBackups(beatGoldRoot) })
  })

  app.post('/api/admin/beat-backups', (request, response) => {
    if (!requireAdmin(request, response, 'users:manage')) return
    response.status(201).json(createBeatBackup(beatGoldRoot))
  })

  app.post('/api/admin/beat-backups/:backupId/verify', (request, response) => {
    if (!requireAdmin(request, response, 'users:manage')) return
    response.json(verifyBeatBackup(request.params.backupId, beatGoldRoot))
  })

  app.post('/api/admin/beat-backups/:backupId/restore', (request, response) => {
    if (!requireAdmin(request, response, 'users:manage')) return
    response.json(restoreBeatBackup(request.params.backupId, beatGoldRoot))
  })

  app.get('/api/admin/beat-gold/evaluation', (request, response) => {
    if (!requireAdmin(request, response, 'admin:read')) return
    response.json({
      current: evaluateBeatGold(
        typeof request.query.modelVersion === 'string'
          ? request.query.modelVersion
          : 'current-candidate',
        beatGoldRoot,
      ),
      runs: listBeatEvaluationRuns(beatGoldRoot),
    })
  })

  app.post('/api/admin/beat-gold/evaluation/runs', (request, response) => {
    if (!requireAdmin(request, response, 'models:write')) return
    response.status(201).json(
      saveBeatEvaluationRun(
        request.body?.modelVersion,
        beatGoldRoot,
      ),
    )
  })

  app.get('/api/admin/beat-gold/releases', (request, response) => {
    if (!requireAdmin(request, response, 'admin:read')) return
    response.json({ releases: listBeatGoldReleases(beatGoldRoot) })
  })

  app.post('/api/admin/beat-gold/releases', (request, response) => {
    if (!requireAdmin(request, response, 'models:write')) return
    response.status(201).json(
      createBeatGoldRelease(request.body?.version, beatGoldRoot),
    )
  })

  app.get(
    '/api/admin/beat-gold/releases/:version/download',
    (request, response) => {
      if (!requireAdmin(request, response, 'admin:read')) return
      const path = beatGoldReleaseArchivePath(
        request.params.version,
        beatGoldRoot,
      )
      response
        .type('application/gzip')
        .set(
          'Content-Disposition',
          `attachment; filename="${request.params.version}.tar.gz"`,
        )
        .sendFile(path)
    },
  )

  app.get('/api/admin/beat-gold/reviews', (request, response) => {
    const user = requireAdmin(request, response, 'reviews:read')
    if (!user) return
    const reviewer = requestedReviewer(user, request.query.reviewer)
    response.json({
      reviews: listBeatReviews(reviewer, beatGoldRoot),
    })
  })

  app.get('/api/admin/beat-gold/structure-reviews', (request, response) => {
    const user = requireAdmin(request, response, 'reviews:read')
    if (!user) return
    const reviewer = requestedReviewer(user, request.query.reviewer)
    response.json({
      reviews: listBeatStructureReviews(reviewer, beatGoldRoot),
    })
  })

  app.get(
    '/api/admin/beat-gold/structure-reviews/:recordingId',
    (request, response) => {
      const user = requireAdmin(request, response, 'reviews:read')
      if (!user) return
      const reviewer = requestedReviewer(user, request.query.reviewer)
      response.json(
        getBeatStructureReview(
          request.params.recordingId,
          reviewer,
          beatGoldRoot,
          beatSourceRoot,
        ),
      )
    },
  )

  app.put(
    '/api/admin/beat-gold/structure-reviews/:recordingId',
    (request, response) => {
      const user = requireAdmin(request, response, 'reviews:write')
      if (!user) return
      const reviewer = requestedReviewer(user, request.query.reviewer)
      response.json(
        saveBeatStructureDraft(
          request.params.recordingId,
          reviewer,
          request.body,
          beatGoldRoot,
          beatSourceRoot,
        ),
      )
    },
  )

  app.post(
    '/api/admin/beat-gold/structure-reviews/:recordingId/submit',
    (request, response) => {
      const user = requireAdmin(request, response, 'reviews:write')
      if (!user) return
      const reviewer = requestedReviewer(user, request.query.reviewer)
      response.json(
        submitBeatStructureReview(
          request.params.recordingId,
          reviewer,
          request.body,
          beatGoldRoot,
          beatSourceRoot,
          {
            adminUserId: user.id,
            adminUsername: user.username,
          },
        ),
      )
    },
  )

  app.get('/api/admin/beat-gold/reviews/:clipId', (request, response) => {
    const user = requireAdmin(request, response, 'reviews:read')
    if (!user) return
    const reviewer = requestedReviewer(user, request.query.reviewer)
    response.json(
      getBeatReview(
        request.params.clipId,
        reviewer,
        beatGoldRoot,
      ),
    )
  })

  app.put('/api/admin/beat-gold/reviews/:clipId', (request, response) => {
    const user = requireAdmin(request, response, 'reviews:write')
    if (!user) return
    const reviewer = requestedReviewer(user, request.query.reviewer)
    response.json(
      saveBeatReviewDraft(
        request.params.clipId,
        reviewer,
        request.body,
        beatGoldRoot,
      ),
    )
  })

  app.post('/api/admin/beat-gold/reviews/:clipId/submit', (request, response) => {
    const user = requireAdmin(request, response, 'reviews:write')
    if (!user) return
    const reviewer = requestedReviewer(user, request.query.reviewer)
    response.json(
      submitBeatReview(
        request.params.clipId,
        reviewer,
        request.body,
        beatGoldRoot,
        {
          adminUserId: user.id,
          adminUsername: user.username,
        },
      ),
    )
  })
}
