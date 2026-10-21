import { expect, test, type Page } from '@playwright/test'

const owner = {
  id: 'owner-1',
  username: 'owner',
  displayName: 'Owner',
  role: 'owner',
  active: true,
  createdAt: '2026-10-21T00:00:00.000Z',
}

async function mockSession(page: Page) {
  await page.route('**/api/admin/auth/session', (route) =>
    route.fulfill({
      json: { configured: true, authenticated: true, user: owner },
    }),
  )
}

function review(alias: string, offset: number) {
  return {
    schema_version: 2,
    review_kind: 'beat-grid',
    status: 'submitted',
    corpus_version: 'beat-pilot-v2-test',
    clip_id: 'fixture-01',
    group_id: 'fixture-recording',
    annotator_alias: alias,
    audio: 'fixture-01.wav',
    audio_source_start: 6,
    audio_source_end: 14,
    audio_duration: 8,
    evaluation_start: 2,
    evaluation_end: 6,
    evaluation_duration: 4,
    time_signature: [4, 4],
    pickup_beats: 0,
    beats: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5].map(
      (at, index) => {
        const absolute = index + 3
        const beat = (absolute % 4) + 1
        return {
          at: at + offset,
          beat,
          measure: 7 + Math.floor(absolute / 4),
          downbeat: beat === 1,
        }
      },
    ),
    tags: ['train'],
    checks: {
      beat_times: true,
      downbeats: true,
      meter: true,
      completeness: true,
    },
    verification_status: 'independent_annotation',
    benchmark_eligible: false,
  }
}

test('creates a reviewer account bound to reviewer-b', async ({ page }) => {
  await mockSession(page)
  const users = [owner]
  await page.route('**/api/admin/users**', async (route) => {
    const request = route.request()
    if (request.method() === 'POST') {
      const input = request.postDataJSON() as Record<string, string>
      const user = {
        id: 'reviewer-user-1',
        username: input.username,
        displayName: input.displayName,
        role: input.role,
        reviewerAlias: input.reviewerAlias,
        active: true,
        createdAt: '2026-10-21T01:00:00.000Z',
      }
      users.push(user)
      await route.fulfill({ status: 201, json: user })
      return
    }
    await route.fulfill({ json: { users } })
  })
  await page.route('**/api/admin/audit**', (route) =>
    route.fulfill({ json: { events: [] } }),
  )

  await page.goto('/admin/users')
  await expect(
    page.getByRole('heading', { name: '用户与审计' }),
  ).toBeVisible()
  await page.getByLabel('账号').fill('reviewer-b-user')
  await page.getByLabel('显示名称').fill('Reviewer B')
  await page.getByLabel('初始密码').fill('reviewer-password-123')
  await page.getByRole('button', { name: '创建账号' }).click()
  await expect(page.getByText('Reviewer B')).toBeVisible()
  await expect(
    page.getByRole('code').filter({ hasText: 'reviewer-b' }),
  ).toBeVisible()
})

test('shows independent beat differences in the adjudication workspace', async ({
  page,
}) => {
  await mockSession(page)
  const left = review('reviewer-a', 0)
  const right = review('reviewer-b', 0.01)
  await page.route('**/api/admin/beat-gold/adjudication?**', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            clipId: 'fixture-01',
            recordingId: 'fixture',
            split: 'train',
            leftStatus: 'submitted',
            rightStatus: 'submitted',
            ready: true,
            adjudicated: false,
            independentAccounts: true,
            beatF1: 1,
            medianErrorMs: 10,
            disagreements: 0,
          },
        ],
      },
    }),
  )
  await page.route(
    '**/api/admin/beat-gold/adjudication/fixture-01?**',
    (route) =>
      route.fulfill({
        json: {
          clipId: 'fixture-01',
          reviewers: ['reviewer-a', 'reviewer-b'],
          independentAccounts: true,
          meterAgreement: true,
          pickupAgreement: true,
          beatMetrics: {
            true_positive: 4,
            false_positive: 0,
            false_negative: 0,
            precision: 1,
            recall: 1,
            f1: 1,
            mean_error_ms: 10,
            median_error_ms: 10,
            p95_error_ms: 10,
            tolerance_ms: 70,
          },
          downbeatMetrics: {
            true_positive: 1,
            false_positive: 0,
            false_negative: 0,
            precision: 1,
            recall: 1,
            f1: 1,
            mean_error_ms: 10,
            median_error_ms: 10,
            p95_error_ms: 10,
            tolerance_ms: 70,
          },
          differences: left.beats.slice(2, 6).map((marker, index) => ({
            left: marker,
            right: right.beats[index + 2],
            deltaMs: 10,
            phaseAgreement: true,
          })),
          left: {
            review: left,
            audioUrl:
              '/api/admin/beat-gold/media/reviewer-a/fixture-01.wav',
          },
          right: {
            review: right,
            audioUrl:
              '/api/admin/beat-gold/media/reviewer-b/fixture-01.wav',
          },
        },
      }),
  )
  await page.route('**/api/admin/beat-gold/media/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'audio/wav',
      body: Buffer.from('RIFF'),
    }),
  )

  await page.goto('/admin/adjudication')
  await expect(
    page.getByRole('heading', { name: 'fixture-01' }),
  ).toBeVisible()
  await expect(page.getByText('1.000', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '采用 B' }).click()
  await expect(page.getByLabel('裁决拍点时间')).toHaveValue('2.510')
})

test('renders gate, gold, training, backup and downstream layer pages', async ({
  page,
}) => {
  await mockSession(page)
  const metrics = {
    true_positive: 0,
    false_positive: 0,
    false_negative: 0,
    precision: 0,
    recall: 0,
    f1: 0,
    mean_error_ms: null,
    median_error_ms: null,
    p95_error_ms: null,
    tolerance_ms: 70,
  }
  await page.route('**/api/admin/beat-gold/evaluation', (route) =>
    route.fulfill({
      json: {
        current: {
          schemaVersion: 1,
          generatedAt: '2026-10-21T00:00:00.000Z',
          modelVersion: 'current-candidate',
          corpusVersion: 'pilot-v2',
          corpusSha256: 'a'.repeat(64),
          coverage: {
            adjudicated: 0,
            total: 48,
            complete: false,
            splitCounts: { train: 0, validation: 0, test: 0 },
          },
          aggregate: {
            beatMetrics: metrics,
            downbeatMetrics: metrics,
            meterAccuracy: 0,
            pickupAccuracy: 0,
          },
          gate: {
            passed: false,
            checks: {
              coverage: false,
              beatF1: false,
              medianError: false,
              p95Error: false,
              downbeatF1: false,
              meter: false,
              pickup: false,
            },
            thresholds: {
              beatF1: 0.95,
              medianErrorMs: 20,
              p95ErrorMs: 50,
              downbeatF1: 0.95,
            },
          },
          items: [],
          reportSha256: 'b'.repeat(64),
        },
        runs: [],
      },
    }),
  )
  await page.route('**/api/admin/dashboard', (route) =>
    route.fulfill({
      json: {
        generatedAt: '2026-10-21T00:00:00.000Z',
        corpus: {
          corpusVersion: 'pilot-v2',
          corpusSha256: 'a'.repeat(64),
          immutable: true,
          evaluationDuration: 24,
          context: { preRoll: 4, postRoll: 4 },
          recordings: 10,
          clips: 48,
          splitCounts: { train: 24, validation: 10, test: 14 },
          verificationStatus: 'draft',
          benchmarkEligible: false,
        },
        reviewer: 'reviewer-a',
        reviews: {
          total: 48,
          draft: 48,
          submitted: 0,
          adjudicated: 0,
          checksComplete: 0,
        },
        layers: [],
        gates: {
          beatF1: 0.95,
          medianErrorMs: 20,
          p95ErrorMs: 50,
          downbeatF1: 0.95,
        },
      },
    }),
  )
  await page.route('**/api/admin/beat-gold/releases', (route) =>
    route.fulfill({ json: { releases: [] } }),
  )
  await page.route('**/api/admin/beat-training', (route) =>
    route.fulfill({
      json: {
        minimumTrainClips: 100,
        adjudicatedTrainClips: 0,
        datasets: [],
        jobs: [],
      },
    }),
  )
  await page.route('**/api/admin/beat-backups', (route) =>
    route.fulfill({ json: { backups: [] } }),
  )
  await page.route('**/api/admin/review-layers/onset', (route) =>
    route.fulfill({
      json: {
        id: 'onset',
        label: '起音 / 拨弦',
        order: 2,
        schemaVersion: 1,
        reviewKind: 'onset-picking',
        dependsOn: 'beat',
        fields: [{ name: 'at', type: 'seconds', required: true }],
        checks: ['timing'],
        status: 'blocked',
        blocker: '等待节拍层完成',
        reviewed: 0,
        total: 48,
      },
    }),
  )

  for (const [path, heading] of [
    ['/admin/evaluation', '评测与门禁'],
    ['/admin/gold', '金标准版本'],
    ['/admin/models', '模型与实验'],
    ['/admin/system', '系统与备份'],
    ['/admin/reviews/onset', '起音 / 拨弦'],
  ] as const) {
    await page.goto(path)
    await expect(page.getByRole('heading', { name: heading })).toBeVisible()
  }
})
