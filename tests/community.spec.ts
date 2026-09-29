import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

const account = {
  id: 'account-owner',
  alias: 'owner',
  roles: [
    'contributor',
    'experimenter',
    'owner',
    'data-publisher',
    'model-maintainer',
    'governance-admin',
  ],
  governanceMode: 'single-maintainer',
  reputation: {},
  createdAt: '2026-09-29T00:00:00.000Z',
}

const dashboard = {
  governanceMode: 'single-maintainer',
  counts: {
    contributors: 1,
    openTasks: 1,
    consensusTasks: 0,
    escalatedTasks: 0,
    submissions: 0,
    activeReleases: 0,
    queuedExperiments: 0,
    completedExperiments: 0,
    promotionCandidates: 0,
  },
  quality: {
    agreementRate: 0,
    calibrationAccuracy: 0,
    rightsCoverage: 1,
  },
  phases: [
    { id: 'C0', label: '治理与契约', status: 'ready', detail: '账号与审计可用' },
    { id: 'C1', label: '校准与金题', status: 'active', detail: '校准任务可用' },
    { id: 'C2', label: '社区复核', status: 'active', detail: '微任务可用' },
    { id: 'C3', label: '数据发布', status: 'waiting', detail: '等待共识数据' },
    { id: 'C4', label: '受控训练', status: 'ready', detail: '配方目录可用' },
    { id: 'C5', label: '模型晋级', status: 'waiting', detail: '等待模型候选' },
    { id: 'C6', label: '长期运营', status: 'active', detail: '指标持续观测' },
  ],
  maintenance: [{ severity: 'info', message: '当前没有需要立即处理的治理事件' }],
}

const task = {
  id: 'task-1',
  schemaVersion: 1,
  sourceId: 'source-1',
  type: 'event-presence',
  difficulty: 'entry',
  status: 'open',
  question: '提示位置是否能听到一次新的触弦起音？',
  context: { start: 0.2, end: 0.9, cueAt: 0.5 },
  requiredResponses: 2,
  maxResponses: 3,
  snapshotSha256: 'a'.repeat(64),
  createdAt: '2026-09-29T00:00:00.000Z',
  source: {
    title: 'AG-PT-set · 起音片段',
    subtitle: '可听源音频 · 0.2-0.9 秒',
    audioUrl: '/media/training/reviews/residual-onset-v2/residual-onset-01.wav',
    dataset: 'AG-PT-set',
    license: 'CC-BY-4.0',
  },
  submissionCount: 0,
}

const experiment = {
  id: 'experiment-running',
  recipeId: 'onset-position-smoke-v1',
  datasetReleaseId: 'builtin:synthetic-smoke-v1',
  submittedBy: 'account-owner',
  parameters: { epochs: 2 },
  status: 'running',
  createdAt: '2026-09-29T00:00:00.000Z',
  updatedAt: '2026-09-29T00:01:00.000Z',
  codeRevision: 'workspace',
  runtimeImage: 'training-runtime@sha256:test',
  datasetManifestSha256: 'b'.repeat(64),
  lineageSha256: 'c'.repeat(64),
  attemptCount: 1,
  lastRunnerId: 'gpu-runner-01',
  lease: {
    runnerId: 'gpu-runner-01',
    claimedAt: '2026-09-29T00:00:30.000Z',
    heartbeatAt: '2026-09-29T00:01:00.000Z',
    expiresAt: '2026-09-29T00:06:00.000Z',
    attempt: 1,
  },
}

const failedExperiment = {
  ...experiment,
  id: 'failed-experiment',
  status: 'failed',
  attemptCount: 1,
  lastRunnerId: undefined,
  lease: undefined,
  failureReason: 'transient runner failure',
}

async function mockCommunity(page: Page): Promise<void> {
  let submitted = false
  await page.addInitScript(() => {
    window.localStorage.setItem('gts-community-token', 'account-owner.secret')
  })
  await page.route('**/api/community/me', (route) => route.fulfill({ json: account }))
  await page.route('**/api/community/dashboard', (route) =>
    route.fulfill({ json: dashboard }),
  )
  await page.route('**/api/community/tasks', (route) =>
    route.fulfill({ json: { tasks: submitted ? [] : [task] } }),
  )
  await page.route('**/api/community/contributions', (route) =>
    route.fulfill({ json: { contributions: [] } }),
  )
  await page.route('**/api/community/releases', (route) =>
    route.fulfill({ json: { releases: [] } }),
  )
  await page.route('**/api/community/recipes', (route) =>
    route.fulfill({
      json: {
        recipes: [
          {
            id: 'dataset-audit-v1',
            specVersion: 1,
            name: '数据完整性审计',
            modelFamily: 'dataset-audit',
            executor: 'builtin-audit',
            availability: 'ready',
            compatibleDatasets: ['*'],
            promotionEligible: false,
            resourceClass: 'control-plane',
            timeoutSeconds: 30,
            description: '验证数据清单',
            parameters: {},
          },
        ],
        builtinDatasets: [
          { id: 'builtin:guitarset-v1', name: 'GuitarSet v1' },
        ],
      },
    }),
  )
  await page.route('**/api/community/experiments', (route) =>
    route.fulfill({ json: { experiments: [experiment, failedExperiment] } }),
  )
  await page.route('**/api/community/promotions', (route) =>
    route.fulfill({ json: { promotions: [] } }),
  )
  await page.route('**/api/community/tasks/task-1/submissions', async (route) => {
    submitted = true
    await route.fulfill({
      status: 201,
      json: {
        submissionId: 'submission-1',
        taskStatus: 'open',
      },
    })
  })
}

test('shows the community registration entry without an account', async ({ page }) => {
  await page.goto('/community')
  await expect(
    page.getByRole('heading', { name: '创建贡献者身份' }),
  ).toBeVisible()
  await expect(page.getByLabel('贡献者代号')).toBeVisible()
  await expect(page.getByRole('button', { name: '进入协作台' })).toBeDisabled()
  await page.getByRole('button', { name: '已有密钥' }).click()
  await expect(page.getByRole('heading', { name: '使用访问密钥' })).toBeVisible()
  await expect(
    page.getByRole('textbox', { name: '访问密钥' }),
  ).toBeVisible()
})

test('supports the contributor task and owner operations', async ({
  page,
}, testInfo) => {
  await mockCommunity(page)
  await page.goto('/community')

  await expect(page.getByText('单人维护模式')).toBeVisible()
  await expect(
    page.getByRole('heading', {
      name: '提示位置是否能听到一次新的触弦起音？',
    }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: '是', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '否', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '无法判断' })).toBeVisible()
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-community-task.png`,
    fullPage: true,
  })

  await page.getByRole('button', { name: '是', exact: true }).click()
  await expect(page.getByRole('heading', { name: '当前任务已处理完' })).toBeVisible()

  await page.getByRole('button', { name: '数据与训练' }).click()
  await expect(page.getByRole('heading', { name: '任务来源与数据版本' })).toBeVisible()
  await expect(page.getByRole('button', { name: '迁移 v2 队列' })).toBeVisible()
  await expect(page.getByRole('button', { name: '提交实验' })).toBeVisible()
  await expect(page.getByText('gpu-runner-01')).toBeVisible()
  await expect(
    page.locator('.community-runner-state').filter({
      hasText: 'gpu-runner-01',
    }).getByText('第 1 次', { exact: false }),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: '重试实验 failed-e' }),
  ).toBeVisible()
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-community-data.png`,
    fullPage: true,
  })

  await page.getByRole('button', { name: '运行状态' }).click()
  await expect(page.getByRole('heading', { name: '阶段状态' })).toBeVisible()
  await expect(page.locator('.community-phase-list article')).toHaveCount(7)

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(1)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-community.png`,
    fullPage: true,
  })
})

test('does not expose training audio or checkpoints through media routes', async ({
  request,
}) => {
  const trainingAudio = await request.get(
    'http://127.0.0.1:8787/media/training/reviews/residual-onset-v2/residual-onset-01.training.wav',
  )
  expect(trainingAudio.status()).toBe(404)

  const trainingData = await request.get(
    'http://127.0.0.1:8787/media/training/checkpoints/guitarset-onset-position-confidence-v5.pt',
  )
  expect(trainingData.status()).toBe(404)
})
