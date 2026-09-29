import assert from 'node:assert/strict'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import {
  CommunityRequestError,
  createCommunityStore,
  type CommunityStore,
  type CommunityStoreOptions,
} from './community.js'
import type { ReviewDocument, ReviewPackage } from './reviews.js'

function reviewDocument(
  status: ReviewPackage['status'],
  name: string,
): ReviewDocument {
  const review: ReviewPackage = {
    schema_version: 1,
    status,
    review_kind: 'residual-onset-training',
    display_title: `Community ${name}`,
    display_subtitle: 'CC BY training excerpt',
    review_id: `${name}-review`,
    project_id: `${name}-project`,
    project_sha256: 'a'.repeat(64),
    source_audio_sha256: 'b'.repeat(64),
    review_audio: `${name}.wav`,
    review_audio_sha256: 'c'.repeat(64),
    training_audio: `${name}.training.wav`,
    training_audio_sha256: 'd'.repeat(64),
    source_kind: 'residual-onset-training',
    measure_numbers: [1],
    source_start: 0,
    source_end: 4,
    duration: 4,
    capo: 0,
    bpm: 120,
    beats: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    time_signature: [4, 4],
    events: [0.5, 1.5, 2.5].map((onset, index) => ({
      onset,
      offset: onset + 0.3,
      string: index + 1,
      fret: index + 3,
      technique: 'pick',
      confidence: 1,
    })),
    checks: {
      timing: status === 'approved',
      string_fret: false,
      completeness: status === 'approved',
      technique: false,
    },
    approval:
      status === 'approved'
        ? {
            reviewer_alias: 'owner',
            reviewed_at: '2026-09-29T00:00:00.000Z',
            content_sha256: 'e'.repeat(64),
          }
        : null,
    training_provenance: {
      source_manifest: 'fixture.jsonl',
      source_manifest_sha256: 'f'.repeat(64),
      source_track_id: `${name}-recording`,
      review_source_track_id: `${name}-work`,
      source_group_id: `${name}-performer`,
      source_split: 'train',
      source_dataset: 'Fixture',
      source_license: 'CC-BY-4.0',
      audio_condition: 'clean',
      training_source_audio_sha256: '1'.repeat(64),
      audio_alignment: 'sample-aligned-no-offset',
      review_audio_processing:
        'rms-normalized--20-dbfs-soft-limited--1-dbfs',
      review_audio_gain_db: 0,
    },
  }
  return {
    name,
    audioUrl: `/media/training/reviews/${name}.wav`,
    review,
  }
}

const runnerCredentials = {
  'runner-a': 'runner-a-secret-credential-0001',
  'runner-b': 'runner-b-secret-credential-0002',
}
const gateCredentials = {
  'gate-a': 'gate-a-secret-credential-000001',
}
const sealedGateCredentials = {
  'sealed-a': 'sealed-a-secret-credential-0001',
}

function withStore(
  run: (store: CommunityStore) => void,
  options: CommunityStoreOptions = {},
): void {
  const root = resolve(process.cwd(), 'data', `.test-community-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    run(
      createCommunityStore(root, {
        bootstrapKey: 'bootstrap-secret',
        now: () => new Date('2026-09-29T00:00:00.000Z'),
        runnerCredentials,
        gateCredentials,
        sealedGateCredentials,
        builtinDatasetManifestSha256: {
          'builtin:guitarset-v1': '8'.repeat(64),
          'builtin:synthetic-smoke-v1': '9'.repeat(64),
        },
        ...options,
      }),
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

function register(
  store: CommunityStore,
  alias: string,
  riskHint: string,
  bootstrapKey?: string,
) {
  const credential = store.registerAccount({ alias, riskHint, bootstrapKey })
  const account = store.authenticate(`Bearer ${credential.token}`)
  return { account, credential }
}

test('bootstraps one owner and authenticates opaque credentials', () => {
  withStore((store) => {
    assert.throws(
      () => register(store, 'owner', 'device-owner', 'wrong'),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 403,
    )
    const { account, credential } = register(
      store,
      'owner',
      'device-owner',
      'bootstrap-secret',
    )
    assert.ok(credential.account.roles.includes('owner'))
    assert.equal(store.getAccount(account).governanceMode, 'single-maintainer')
    assert.throws(
      () => store.authenticate('Bearer invalid.secret'),
      /凭据无效/,
    )
    const contributor = register(store, 'player-1', 'device-1')
    assert.deepEqual(contributor.credential.account.roles, [
      'contributor',
      'experimenter',
    ])
  })
})

test('uses calibration before cross-account consensus and freezes a release', () => {
  withStore((store) => {
    const owner = register(
      store,
      'owner',
      'device-owner',
      'bootstrap-secret',
    ).account
    const left = register(store, 'left-player', 'device-left').account
    const right = register(store, 'right-player', 'device-right').account

    store.importReview(owner, reviewDocument('approved', 'approved-clip'), 'seed')
    const calibrationTasks = store
      .listAllTasks(owner)
      .filter((task) => task.type === 'event-presence')
    assert.equal(calibrationTasks.length, 3)
    let firstSubmissionId = ''
    for (const task of calibrationTasks) {
      const leftResult = store.submit(left, task.id, {
        answer: { choice: 'yes' },
      })
      firstSubmissionId ||= leftResult.submissionId
      assert.equal(leftResult.calibration?.correct, true)
      assert.equal(
        store.submit(right, task.id, { answer: { choice: 'yes' } }).calibration
          ?.correct,
        true,
      )
    }
    assert.equal(
      store.getAccount(left).reputation['event-presence']?.qualified,
      true,
    )
    store.withdrawSubmission(left, firstSubmissionId)
    assert.equal(
      store.getAccount(left).reputation['event-presence']?.qualified,
      false,
    )
    assert.ok(
      store.listTasks(left).some((task) => task.id === calibrationTasks[0].id),
    )
    store.submit(left, calibrationTasks[0].id, { answer: { choice: 'yes' } })

    const imported = store.importReview(
      owner,
      reviewDocument('draft', 'community-clip'),
      'residual-onset-v2',
    )
    const target = store
      .listAllTasks(owner)
      .find(
        (task) =>
          task.sourceId === imported.sourceId &&
          task.type === 'event-presence',
      )
    assert.ok(target)
    store.submit(left, target.id, {
      answer: { choice: 'yes' },
      confidence: 0.9,
      durationMs: 3200,
    })
    const result = store.submit(right, target.id, {
      answer: { choice: 'yes' },
      confidence: 0.8,
      durationMs: 4100,
    })
    assert.equal(result.taskStatus, 'consensus')
    assert.equal(result.consensus?.answer.choice, 'yes')

    const release = store.createRelease(owner, {
      name: 'residual-community',
      taskIds: [target.id],
    })
    assert.equal(release.status, 'active')
    assert.equal(release.verification, 'owner_verified')
    assert.match(release.manifestSha256, /^[0-9a-f]{64}$/)
    const manifest = store.releaseManifest(owner, release.id)
    assert.equal(manifest.records.length, 1)
    assert.equal(manifest.records[0].source.reviewQueue, 'residual-onset-v2')
    assert.equal(manifest.release.manifestSha256, release.manifestSha256)

    const queued = store.createExperiment(left, {
      recipeId: 'community-release-audit-v1',
      datasetReleaseId: release.id,
      parameters: {},
    })
    assert.equal(queued.status, 'queued')
    const runner = store.authenticateRunner(
      `Runner runner-a.${runnerCredentials['runner-a']}`,
    )
    const claim = store.claimExperiment(runner)
    assert.equal(claim?.job.experimentId, queued.id)
    store.withdrawSource(owner, imported.sourceId, 'rights withdrawn')
    assert.equal(store.listReleases()[0].status, 'blocked')
    const cancelled = store.listExperiments(owner)[0]
    assert.equal(cancelled.status, 'cancelled')
    assert.equal(cancelled.lease, undefined)
    assert.throws(
      () =>
        store.recordRunnerResult(runner, queued.id, {
          leaseToken: claim?.leaseToken,
          status: 'failed',
        }),
      /没有有效执行租约/,
    )
  })
})

test('claims experiments exclusively and reclaims an expired lease', () => {
  let now = new Date('2026-09-29T00:00:00.000Z')
  withStore(
    (store) => {
      const owner = register(
        store,
        'owner',
        'device-owner',
        'bootstrap-secret',
      ).account
      const experiment = store.createExperiment(owner, {
        recipeId: 'onset-position-smoke-v1',
        datasetReleaseId: 'builtin:synthetic-smoke-v1',
        parameters: { epochs: 2 },
      })
      assert.throws(
        () => store.authenticateRunner('Runner runner-a.invalid'),
        (error: unknown) =>
          error instanceof CommunityRequestError && error.statusCode === 401,
      )
      const runnerA = store.authenticateRunner(
        `Runner runner-a.${runnerCredentials['runner-a']}`,
      )
      const runnerB = store.authenticateRunner(
        `Runner runner-b.${runnerCredentials['runner-b']}`,
      )
      const firstClaim = store.claimExperiment(runnerA)
      assert.ok(firstClaim)
      assert.equal(firstClaim.job.experimentId, experiment.id)
      assert.equal(firstClaim.job.recipe.id, 'onset-position-smoke-v1')
      assert.equal(firstClaim.job.recipe.specVersion, 1)
      assert.equal(firstClaim.job.recipe.resourceClass, 'cpu-small')
      assert.equal(firstClaim.job.dataset.kind, 'builtin')
      assert.equal(firstClaim.job.dataset.manifestSha256, '9'.repeat(64))
      assert.equal('submittedBy' in firstClaim.job, false)
      assert.equal(store.claimExperiment(runnerB), null)

      const visible = store.listExperiments(owner)[0]
      assert.equal(visible.lease?.runnerId, 'runner-a')
      assert.equal('tokenHash' in (visible.lease ?? {}), false)

      now = new Date('2026-09-29T00:00:31.000Z')
      const secondClaim = store.claimExperiment(runnerB)
      assert.equal(secondClaim?.job.experimentId, experiment.id)
      assert.equal(
        store.listExperiments(owner)[0].lease?.attempt,
        2,
      )
      assert.notEqual(secondClaim?.leaseToken, firstClaim.leaseToken)
      assert.throws(
        () =>
          store.heartbeatExperiment(
            runnerA,
            experiment.id,
            firstClaim.leaseToken,
          ),
        (error: unknown) =>
          error instanceof CommunityRequestError && error.statusCode === 403,
      )
    },
    {
      now: () => now,
      runnerLeaseSeconds: 30,
    },
  )
})

test('accepts results only from the lease owner with valid metrics and artifact', () => {
  withStore((store) => {
    const owner = register(
      store,
      'owner',
      'device-owner',
      'bootstrap-secret',
    ).account
    const experiment = store.createExperiment(owner, {
      recipeId: 'onset-position-smoke-v1',
      datasetReleaseId: 'builtin:synthetic-smoke-v1',
      parameters: { epochs: 2 },
    })
    const runnerA = store.authenticateRunner(
      `Runner runner-a.${runnerCredentials['runner-a']}`,
    )
    const runnerB = store.authenticateRunner(
      `Runner runner-b.${runnerCredentials['runner-b']}`,
    )
    const claim = store.claimExperiment(runnerA)
    assert.ok(claim)
    assert.throws(
      () =>
        store.recordRunnerResult(runnerB, experiment.id, {
          leaseToken: claim.leaseToken,
          status: 'completed',
          metrics: { tablatureF1: 0.8 },
          artifactSha256: 'a'.repeat(64),
        }),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 403,
    )
    assert.throws(
      () =>
        store.recordRunnerResult(runnerA, experiment.id, {
          leaseToken: claim.leaseToken,
          status: 'completed',
          metrics: { 'invalid metric': 0.8 },
          artifactSha256: 'a'.repeat(64),
        }),
      /指标名称无效/,
    )
    assert.throws(
      () =>
        store.recordRunnerResult(runnerA, experiment.id, {
          leaseToken: claim.leaseToken,
          status: 'completed',
          metrics: { tablatureF1: 0.8 },
          artifactSha256: 'a'.repeat(64),
          executionEvidence: {
            isolation: 'sandbox-exec',
            durationMs: 1200,
            stdoutSha256: 'b'.repeat(64),
            stderrSha256: 'c'.repeat(64),
            reportSha256: 'd'.repeat(64),
          },
        }),
      /尚未由当前租约上传并校验/,
    )
    const artifactSha256 = 'a'.repeat(64)
    store.registerRunnerArtifact(
      runnerA,
      experiment.id,
      claim.leaseToken,
      {
        sha256: artifactSha256,
        sizeBytes: 1024,
        mediaType: 'application/octet-stream',
        objectKey: `artifacts/${experiment.id}/1/${artifactSha256}.bin`,
      },
    )
    assert.throws(
      () =>
        store.recordRunnerResult(runnerA, experiment.id, {
          leaseToken: claim.leaseToken,
          status: 'completed',
          metrics: { tablatureF1: 0.8 },
          artifactSha256: 'invalid',
        }),
      /SHA-256 无效/,
    )
    const heartbeat = store.heartbeatExperiment(
      runnerA,
      experiment.id,
      claim.leaseToken,
    )
    assert.equal(heartbeat.status, 'running')
    const completed = store.recordRunnerResult(runnerA, experiment.id, {
      leaseToken: claim.leaseToken,
      status: 'completed',
      metrics: { tablatureF1: 0.8, repeatedRecall: 0.86 },
      artifactSha256,
      executionEvidence: {
        isolation: 'sandbox-exec',
        durationMs: 1200,
        stdoutSha256: 'b'.repeat(64),
        stderrSha256: 'c'.repeat(64),
        reportSha256: 'd'.repeat(64),
      },
    })
    assert.equal(completed.status, 'completed')
    assert.equal(completed.lastRunnerId, 'runner-a')
    assert.equal(completed.lease, undefined)
    assert.equal(completed.metrics?.repeatedRecall, 0.86)
    assert.equal(completed.artifactSizeBytes, 1024)
    assert.equal(completed.executionEvidence?.isolation, 'sandbox-exec')
  })
})

test('enforces active quotas and allows a bounded failed experiment retry', () => {
  withStore((store) => {
    const owner = register(
      store,
      'owner',
      'device-owner',
      'bootstrap-secret',
    ).account
    const first = store.createExperiment(owner, {
      recipeId: 'onset-position-smoke-v1',
      datasetReleaseId: 'builtin:synthetic-smoke-v1',
      parameters: { epochs: 1 },
    })
    store.createExperiment(owner, {
      recipeId: 'onset-position-smoke-v1',
      datasetReleaseId: 'builtin:synthetic-smoke-v1',
      parameters: { epochs: 1 },
    })
    assert.throws(
      () =>
        store.createExperiment(owner, {
          recipeId: 'onset-position-smoke-v1',
          datasetReleaseId: 'builtin:synthetic-smoke-v1',
          parameters: { epochs: 1 },
        }),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 429,
    )
    const runner = store.authenticateRunner(
      `Runner runner-a.${runnerCredentials['runner-a']}`,
    )
    const claim = store.claimExperiment(runner)
    assert.equal(claim?.job.experimentId, first.id)
    store.recordRunnerResult(runner, first.id, {
      leaseToken: claim?.leaseToken,
      status: 'failed',
      failureReason: 'transient runner failure',
    })
    const retried = store.retryExperiment(owner, first.id)
    assert.equal(retried.status, 'queued')
    assert.equal(retried.attemptCount, 1)
    assert.equal(retried.failureReason, undefined)
    assert.throws(
      () => store.retryExperiment(owner, first.id),
      /只有失败实验可以重试/,
    )
  })
})

test('enforces recipe bounds and the single-maintainer promotion sequence', () => {
  withStore((store) => {
    const owner = register(
      store,
      'owner',
      'device-owner',
      'bootstrap-secret',
    ).account
    const contributor = register(store, 'trainer', 'device-trainer').account
    assert.throws(
      () =>
        store.createExperiment(contributor, {
          recipeId: 'candidate-activity-temporal-v1',
          datasetReleaseId: 'builtin:guitarset-v1',
        }),
      /尚未部署执行规范/,
    )
    assert.throws(
      () =>
        store.createExperiment(contributor, {
          recipeId: 'onset-position-smoke-v1',
          datasetReleaseId: 'builtin:guitarset-v1',
        }),
      /不兼容/,
    )
    assert.throws(
      () =>
        store.createExperiment(contributor, {
          recipeId: 'onset-position-guitarset-v1',
          datasetReleaseId: 'builtin:guitarset-v1',
          parameters: { epochs: 500 },
        }),
      /超出允许范围/,
    )

    const experiment = store.createExperiment(contributor, {
      recipeId: 'onset-position-guitarset-v1',
      datasetReleaseId: 'builtin:guitarset-v1',
      parameters: { epochs: 2, learningRate: 0.00003 },
    })
    assert.equal(experiment.status, 'queued')
    assert.equal(experiment.codeRevision, 'workspace')
    assert.equal(experiment.runtimeImage, 'local-development')
    assert.match(experiment.datasetManifestSha256, /^[0-9a-f]{64}$/)
    assert.throws(
      () =>
        store.recordExperiment(contributor, experiment.id, {
          status: 'completed',
          metrics: { tablatureF1: 0.8 },
          artifactSha256: 'a'.repeat(64),
        }),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 403,
    )
    const runner = store.authenticateRunner(
      `Runner runner-a.${runnerCredentials['runner-a']}`,
    )
    const claim = store.claimExperiment(runner)
    assert.ok(claim)
    const artifactSha256 = 'a'.repeat(64)
    store.registerRunnerArtifact(
      runner,
      experiment.id,
      claim.leaseToken,
      {
        sha256: artifactSha256,
        sizeBytes: 1024,
        mediaType: 'application/octet-stream',
        objectKey: `artifacts/${experiment.id}/1/${artifactSha256}.bin`,
      },
    )
    store.recordRunnerResult(runner, experiment.id, {
      leaseToken: claim.leaseToken,
      status: 'completed',
      metrics: {
        tablatureF1: 0.8,
        repeatedRecall: 0.86,
      },
      artifactSha256,
      executionEvidence: {
        isolation: 'sandbox-exec',
        durationMs: 1200,
        stdoutSha256: 'b'.repeat(64),
        stderrSha256: 'c'.repeat(64),
        reportSha256: 'd'.repeat(64),
      },
    })
    const promotion = store.createPromotion(owner, experiment.id)
    assert.equal(promotion.checks.lineage?.passed, true)
    const modelCard = store.modelCard(promotion.id)
    assert.equal(modelCard.model.version, promotion.modelVersion)
    assert.equal(modelCard.artifact.sha256, artifactSha256)
    assert.ok(modelCard.limitations.some((item) => item.includes('密封评测')))
    const reproduction = store.reproduceExperiment(owner, experiment.id)
    assert.equal(reproduction.status, 'queued')
    assert.equal(reproduction.lineageSha256, experiment.lineageSha256)
    assert.equal(
      store.reproduceExperiment(owner, experiment.id).id,
      reproduction.id,
    )
    assert.throws(
      () =>
        store.recordPromotionCheck(owner, promotion.id, {
          check: 'sealed-evaluation',
          passed: true,
          summary: 'manual bypass',
        }),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 403,
    )
    const gate = store.authenticatePromotionGate(
      `Gate gate-a.${gateCredentials['gate-a']}`,
    )
    assert.throws(
      () => store.authenticatePromotionGate('Gate gate-a.invalid'),
      (error: unknown) =>
        error instanceof CommunityRequestError && error.statusCode === 401,
    )
    const gateJobs = store.listPromotionGateJobs(gate)
    assert.equal(gateJobs.length, 1)
    assert.equal(gateJobs[0].promotion.id, promotion.id)
    assert.equal(gateJobs[0].experiment.id, experiment.id)
    assert.throws(
      () =>
        store.recordAutomatedPromotionCheck(gate, promotion.id, {
          check: 'sealed-evaluation',
          passed: true,
          summary: 'public gate bypass',
          evidenceSha256: 'e'.repeat(64),
        }),
      /自动门禁检查项无效/,
    )
    for (const check of [
      'reproduction',
      'public-validation',
      'robustness',
    ] as const) {
      store.recordAutomatedPromotionCheck(gate, promotion.id, {
        check,
        passed: true,
        summary: `${check} passed`,
        evidenceSha256: 'e'.repeat(64),
      })
    }
    const sealedGate = store.authenticateSealedPromotionGate(
      `SealedGate sealed-a.${sealedGateCredentials['sealed-a']}`,
    )
    assert.equal(store.listSealedPromotionGateJobs(sealedGate).length, 1)
    store.recordAutomatedPromotionCheck(sealedGate, promotion.id, {
      check: 'sealed-evaluation',
      passed: true,
      summary: 'sealed-evaluation passed',
      evidenceSha256: 'f'.repeat(64),
    })
    assert.equal(store.listSealedPromotionGateJobs(sealedGate).length, 0)
    assert.equal(
      store.advancePromotion(owner, promotion.id, 'shadow').status,
      'shadow',
    )
    store.recordPromotionCheck(owner, promotion.id, {
      check: 'shadow',
      passed: true,
      summary: 'shadow passed',
    })
    assert.equal(
      store.advancePromotion(owner, promotion.id, 'canary').status,
      'canary',
    )
    store.recordPromotionCheck(owner, promotion.id, {
      check: 'canary',
      passed: true,
      summary: 'canary passed',
    })
    assert.equal(
      store.advancePromotion(owner, promotion.id, 'champion').status,
      'champion',
    )
    assert.equal(store.dashboard().counts.completedExperiments, 1)
    assert.ok(store.auditEvents(owner).length > 0)
  })
})
