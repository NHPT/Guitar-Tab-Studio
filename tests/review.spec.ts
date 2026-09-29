import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

async function mockReviewApi(page: Page): Promise<void> {
  const checks = {
    timing: false,
    string_fret: false,
    completeness: false,
    technique: false,
  }
  const review = {
    schema_version: 1,
    status: 'draft',
    review_id: 'project-1-m1-2',
    project_id: 'project-1',
    project_sha256: 'a'.repeat(64),
    source_audio_sha256: 'b'.repeat(64),
    review_audio: 'real-reference-01.wav',
    review_audio_sha256: 'c'.repeat(64),
    source_kind: 'upload',
    measure_numbers: [1, 2],
    source_start: 0,
    source_end: 3.9938,
    duration: 3.9938,
    capo: 0,
    bpm: 120,
    beats: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    time_signature: [4, 4],
    events: [
      {
        onset: 0.0328,
        offset: 0.3346,
        string: 4,
        fret: 1,
        technique: 'unknown',
        confidence: 1,
      },
      {
        onset: 0.04,
        offset: 0.4,
        string: 3,
        fret: 2,
        technique: 'unknown',
        confidence: 1,
      },
      {
        onset: 0.5,
        offset: 0.8,
        string: 1,
        fret: 3,
        technique: 'pick',
        confidence: 1,
      },
    ],
    excluded_ranges: [],
    checks,
    approval: null,
  }
  await page.route('**/api/reviews', (route) =>
    route.fulfill({
      json: {
        reviews: [
          {
            name: 'real-reference-01',
            status: 'draft',
            projectTitle: 'Independent review song',
            projectArtist: 'Review artist',
            eventCount: 3,
            duration: review.duration,
            measureNumbers: review.measure_numbers,
            capo: 0,
            checks,
          },
        ],
      },
    }),
  )
  await page.route('**/api/reviews/real-reference-01', (route) =>
    route.fulfill({
      json: {
        name: 'real-reference-01',
        audioUrl: '/media/training/reviews/real-reference-01.wav',
        review,
      },
    }),
  )
}

test('loads all review packages and supports an explicit approval flow', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop interaction check')
  await mockReviewApi(page)
  await page.goto('/review')

  await expect(
    page.getByRole('heading', { name: 'Independent review song' }),
  ).toBeVisible()
  await expect(page.getByText('第 1-2 小节 · Review artist')).toBeVisible()
  await expect(page.locator('.review-package')).toHaveCount(1)
  await expect(page.locator('.review-event-table tbody tr')).toHaveCount(3)
  await expect(page.getByText('3 个事件')).toBeVisible()

  const audio = page.locator('audio')
  await expect(audio).toHaveAttribute(
    'src',
    /\/media\/training\/reviews\/real-reference-01\.wav$/,
  )
  await expect
    .poll(() => audio.evaluate((element: HTMLAudioElement) => element.readyState))
    .toBeGreaterThan(0)

  const progress = page.getByLabel('精确播放进度')
  await expect(progress).toHaveAttribute('step', '0.0001')
  await progress.fill('0.0328')
  await expect(page.locator('.review-precise-time output')).toHaveText('0.0328s')
  await expect(page.locator('.review-time-axis')).toContainText('0.0000s')
  await expect(page.locator('.review-time-axis')).toContainText('3.9938s')
  await expect(page.getByLabel('速度').locator('option')).toHaveCount(5)
  await expect(page.getByLabel('速度').locator('option[value="0.25"]')).toHaveText(
    '0.25x',
  )

  await expect(page.locator('.review-segment-guides button')).toHaveCount(2)
  await expect(page.locator('.review-segment-nav')).toContainText('和弦 1/2')
  await page.getByRole('button', { name: '下一个检测音组' }).click()
  await expect(page.locator('.review-precise-time output')).toHaveText('0.5000s')
  await expect(page.locator('.review-segment-nav')).toContainText('单音 2/2')
  await page.getByRole('button', { name: '上一个检测音组' }).click()
  await expect(page.locator('.review-segment-nav')).toContainText('和弦 1/2')

  await page.getByLabel('排除区间起点').fill('0.9000')
  await page.getByLabel('排除区间终点').fill('1.1000')
  await page.getByLabel('排除原因').fill('弱音无法确认')
  await page.getByRole('button', { name: '添加区间' }).click()
  await expect(page.locator('.review-excluded-range-layer > span')).toHaveCount(1)
  await expect(page.getByText('0.2000s 不参与评测')).toBeVisible()

  await page.getByRole('button', { name: '从头播放' }).click()
  const restartedAt = await audio.evaluate(
    (element: HTMLAudioElement) => element.currentTime,
  )
  expect(restartedAt).toBeLessThan(1)

  const fret = page.getByLabel('绝对品位')
  await expect(fret).toHaveValue('1')
  await fret.fill('2')
  await expect(page.getByText('未保存')).toBeVisible()

  await page.getByRole('checkbox', { name: /时间边界/ }).check()
  await page.getByRole('checkbox', { name: /弦与品位/ }).check()
  await page.getByRole('checkbox', { name: /事件完整性/ }).check()
  await page.getByLabel('审核人代号').fill('reviewer-e2e')

  const approve = page.getByRole('button', { name: '批准', exact: true })
  await expect(approve).toBeEnabled()
  await approve.click()
  await expect(
    page.getByRole('dialog', { name: '批准并锁定此审核包？' }),
  ).toBeVisible()
  await expect(page.getByRole('dialog')).toContainText('1 段 / 0.2000s')
  await expect(page.getByRole('button', { name: '确认批准' })).toBeVisible()
  await page.getByRole('button', { name: '取消' }).click()
})

test('loads the isolated residual-onset training queue', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop queue routing check')
  const checks = {
    timing: false,
    string_fret: false,
    completeness: false,
    technique: false,
  }
  const review = {
    schema_version: 1,
    status: 'draft',
    review_kind: 'residual-onset-training',
    display_title: 'P1 · 掌根制音 · 直录',
    display_subtitle: '可听源音频 · 0.0-12.0 秒',
    review_id: 'residual-onset-p1-palm-mute-01',
    project_id: 'guitar-techs-p1-palm-mute',
    project_sha256: 'a'.repeat(64),
    source_audio_sha256: 'b'.repeat(64),
    review_audio: 'residual-onset-01.wav',
    review_audio_sha256: 'c'.repeat(64),
    training_audio: 'residual-onset-01.training.wav',
    training_audio_sha256: 'd'.repeat(64),
    source_kind: 'residual-onset-training',
    measure_numbers: [1],
    source_start: 0,
    source_end: 12,
    duration: 12,
    capo: 0,
    bpm: 60,
    beats: Array.from({ length: 12 }, (_value, index) => index),
    time_signature: [4, 4],
    events: [
      {
        onset: 0.25,
        offset: 0.5,
        string: 2,
        fret: 3,
        technique: 'palm-mute',
        confidence: 1,
      },
    ],
    excluded_ranges: [],
    checks,
    approval: null,
    training_provenance: {
      source_manifest: 'technique-mixture-demucs-v1/manifest-combined.jsonl',
      source_manifest_sha256: 'e'.repeat(64),
      source_track_id:
        'guitar-techs-P1-palm-mute-directinput-mixture-demucs-guitar',
      review_source_track_id: 'guitar-techs-P1-palm-mute-directinput',
      source_group_id: 'guitar-techs-P1-palm-mute',
      source_split: 'train',
      source_dataset: 'Guitar-TECHS',
      source_license: 'CC-BY-4.0',
      audio_condition: 'procedural-mixture-demucs-guitar-stem',
      training_source_audio_sha256: 'f'.repeat(64),
      audio_alignment: 'sample-aligned-no-offset',
      review_audio_processing:
        'rms-normalized--20-dbfs-soft-limited--1-dbfs',
      review_audio_gain_db: 24.5,
    },
  }
  await page.route('**/api/reviews?queue=residual-onset', (route) =>
    route.fulfill({
      json: {
        reviews: [
          {
            name: 'residual-onset-01',
            status: 'draft',
            projectTitle: review.display_title,
            projectArtist: review.display_subtitle,
            reviewKind: review.review_kind,
            eventCount: 1,
            duration: review.duration,
            measureNumbers: review.measure_numbers,
            capo: 0,
            checks,
          },
        ],
      },
    }),
  )
  await page.route(
    '**/api/reviews/residual-onset-01?queue=residual-onset',
    (route) =>
      route.fulfill({
        json: {
          name: 'residual-onset-01',
          audioUrl:
            '/media/training/reviews/residual-onset/residual-onset-01.wav',
          review,
        },
      }),
  )

  await page.goto('/review?queue=residual-onset')

  await expect(page.getByText('Guitar Tab Review')).toBeVisible()
  await expect(page.getByText('训练标注片段')).toBeVisible()
  await expect(
    page.getByRole('heading', { name: review.display_title }),
  ).toBeVisible()
  await expect(page.getByText(review.display_subtitle)).toHaveCount(2)
  await expect(page.locator('.review-package')).toHaveCount(1)
  await expect(
    page.getByRole('checkbox', { name: /弦与品位/ }),
  ).not.toBeChecked()
  await page.getByRole('checkbox', { name: /时间边界/ }).check()
  await page.getByRole('checkbox', { name: /事件完整性/ }).check()
  await page.getByLabel('审核人代号').fill('reviewer-e2e')
  await expect(
    page.getByRole('button', { name: '批准', exact: true }),
  ).toBeEnabled()
})

test('loads the cross-performer residual-onset v2 queue', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop data routing check')

  const response = await page.request.get(
    'http://localhost:8787/api/reviews?queue=residual-onset-v2',
  )
  expect(response.ok()).toBeTruthy()
  const payload = await response.json()
  expect(payload.reviews).toHaveLength(24)
  expect(
    payload.reviews.every(
      (review: { status: string }) => review.status === 'draft',
    ),
  ).toBeTruthy()
  expect(
    new Set(
      payload.reviews.map(
        (review: { projectTitle: string }) =>
          review.projectTitle.split(' · ')[1],
      ),
    ).size,
  ).toBe(4)

  await page.goto('/review?queue=residual-onset-v2')

  await expect(page.getByText('Guitar Tab Review')).toBeVisible()
  await expect(page.locator('.review-package')).toHaveCount(24)
  await expect(
    page.getByRole('heading', {
      name: 'AG-PT-set · P0 · 掌根制音',
    }),
  ).toBeVisible()
  await expect(page.locator('audio')).toHaveAttribute(
    'src',
    /reviews\/residual-onset-v2\/residual-onset-01\.wav$/,
  )
})

test('deletes the selected event with the keyboard', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop keyboard check')
  await mockReviewApi(page)
  await page.goto('/review')
  const row = page.locator('.review-event-table tbody tr').first()
  await row.click()
  await row.press('Delete')
  await expect(page.locator('.review-event-table tbody tr')).toHaveCount(2)
  await expect(page.getByRole('button', { name: '保存草稿' })).toBeEnabled()
  await page.getByRole('button', { name: '撤回上一步' }).click()
  await expect(page.locator('.review-event-table tbody tr')).toHaveCount(3)
  await expect(page.getByRole('button', { name: '保存草稿' })).toBeDisabled()

  await page.locator('.review-event-table tbody tr').first().click()
  await page.keyboard.press('Backspace')
  await expect(page.locator('.review-event-table tbody tr')).toHaveCount(2)
  await page.keyboard.press('Control+z')
  await expect(page.locator('.review-event-table tbody tr')).toHaveCount(3)
})

test('keeps the review workspace within the mobile viewport', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile', 'mobile-only layout check')
  await mockReviewApi(page)
  await page.goto('/review')
  await expect(
    page.getByRole('heading', { name: 'Independent review song' }),
  ).toBeVisible()

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(1)
  await expect(page.locator('.review-package-rail')).toBeVisible()
  await expect(page.getByLabel('绝对品位')).toBeVisible()
})
