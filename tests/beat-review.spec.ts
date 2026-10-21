import assert from 'node:assert/strict'
import { expect, test } from '@playwright/test'

function wavTone(durationSeconds = 4): Buffer {
  const sampleRate = 8_000
  const samples = sampleRate * durationSeconds
  const dataBytes = samples * 2
  const buffer = Buffer.alloc(44 + dataBytes)
  buffer.write('RIFF', 0)
  buffer.writeUInt32LE(36 + dataBytes, 4)
  buffer.write('WAVEfmt ', 8)
  buffer.writeUInt32LE(16, 16)
  buffer.writeUInt16LE(1, 20)
  buffer.writeUInt16LE(1, 22)
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * 2, 28)
  buffer.writeUInt16LE(2, 32)
  buffer.writeUInt16LE(16, 34)
  buffer.write('data', 36)
  buffer.writeUInt32LE(dataBytes, 40)
  for (let index = 0; index < samples; index += 1) {
    const value = Math.round(
      Math.sin((2 * Math.PI * 220 * index) / sampleRate) * 12_000,
    )
    buffer.writeInt16LE(value, 44 + index * 2)
  }
  return buffer
}

function markers(
  times: number[],
  signature: [number, number],
) {
  return times.map((at, index) => {
    const absolute = index + 3
    const beat = absolute % signature[0] + 1
    return {
      at,
      beat,
      measure: 7 + Math.floor(absolute / signature[0]),
      downbeat: beat === 1,
    }
  })
}

test('edits, saves and submits an independent beat review', async ({ page }) => {
  let savedDraft: Record<string, unknown> | undefined
  let submittedDraft: Record<string, unknown> | undefined
  const checks = {
    beat_times: false,
    downbeats: false,
    meter: false,
    completeness: false,
  }
  const review = {
    schema_version: 2,
    review_kind: 'beat-grid',
    status: 'draft',
    corpus_version: 'beat-pilot-v2-test',
    clip_id: 'fixture-01',
    group_id: 'fixture-recording',
    annotator_alias: 'reviewer-a',
    audio: 'fixture-01.wav',
    audio_source_start: 6,
    audio_source_end: 14,
    audio_duration: 8,
    evaluation_start: 2,
    evaluation_end: 6,
    evaluation_duration: 4,
    time_signature: [4, 4] as [number, number],
    pickup_beats: 0,
    beats: markers([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], [4, 4]),
    tags: ['train', 'acoustic'],
    checks,
    verification_status: 'draft',
    benchmark_eligible: false,
  }
  const summary = () => ({
    clipId: 'fixture-01',
    recordingId: 'fixture',
    split: 'train',
    status: review.status,
    duration: 4,
    sourceStart: 8,
    audioDuration: 8,
    evaluationStart: 2,
    evaluationEnd: 6,
    beatCount: 4,
    timeSignature: review.time_signature,
    pickupBeats: review.pickup_beats,
    checks: review.checks,
    tags: review.tags,
  })
  const document = () => ({
    review,
    audioUrl: '/api/admin/beat-gold/media/reviewer-a/fixture-01.wav',
  })

  await page.route('**/api/admin/auth/session', (route) =>
    route.fulfill({
      json: {
        configured: true,
        authenticated: true,
        user: {
          id: 'owner-1',
          username: 'owner',
          displayName: 'Owner',
          role: 'owner',
          active: true,
          createdAt: '2026-10-21T00:00:00.000Z',
        },
      },
    }),
  )
  await page.route('**/api/admin/beat-gold/media/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'audio/wav',
      body: wavTone(),
    }),
  )
  await page.route('**/api/admin/beat-gold**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/admin/beat-gold') {
      await route.fulfill({
        json: {
          enabled: true,
          reviewers: ['reviewer-a'],
          manifest: {
            corpusVersion: 'beat-pilot-v2-test',
            corpusSha256: 'a'.repeat(64),
            immutable: true,
            evaluationDuration: 4,
            context: { preRoll: 2, postRoll: 2 },
            recordings: 1,
            clips: 1,
            splitCounts: { train: 1, validation: 0, test: 0 },
            verificationStatus: 'draft',
            benchmarkEligible: false,
          },
        },
      })
      return
    }
    if (path === '/api/admin/beat-gold/reviews') {
      await route.fulfill({ json: { reviews: [summary()] } })
      return
    }
    if (
      path === '/api/admin/beat-gold/reviews/fixture-01' &&
      request.method() === 'GET'
    ) {
      await route.fulfill({ json: document() })
      return
    }
    if (
      path === '/api/admin/beat-gold/reviews/fixture-01' &&
      request.method() === 'PUT'
    ) {
      savedDraft = request.postDataJSON() as Record<string, unknown>
      Object.assign(review, {
        beats: savedDraft.beats,
        time_signature: savedDraft.time_signature,
        pickup_beats: savedDraft.pickup_beats,
        checks: savedDraft.checks,
      })
      await route.fulfill({ json: document() })
      return
    }
    if (
      path === '/api/admin/beat-gold/reviews/fixture-01/submit' &&
      request.method() === 'POST'
    ) {
      submittedDraft = request.postDataJSON() as Record<string, unknown>
      Object.assign(review, {
        status: 'submitted',
        checks: submittedDraft.checks,
        verification_status: 'independent_annotation',
      })
      await route.fulfill({ json: document() })
      return
    }
    await route.fulfill({ status: 404, json: { error: 'not found' } })
  })

  await page.goto('/admin/beat-review')
  await expect(page).toHaveURL(/\/admin\/reviews\/beat/u)
  await expect(
    page.getByRole('heading', { name: 'fixture-01' }),
  ).toBeVisible()
  await expect(page.getByText('不计分', { exact: true })).toHaveCount(2)
  await expect(page.locator('.beat-review-timeline > button')).toHaveCount(7)
  await expect(
    page.locator('.beat-review-timeline > button span'),
  ).toHaveText(['4', '1', '2', '3', '4', '1', '2'])
  await expect(
    page
      .locator('.beat-review-timeline > button.is-context.is-downbeat span')
      .first(),
  ).toHaveText('1')

  const secondMarker = page.locator('.beat-review-timeline > button').nth(2)
  const markerBounds = await secondMarker.boundingBox()
  assert(markerBounds)
  await page.mouse.move(
    markerBounds.x + markerBounds.width / 2,
    markerBounds.y + markerBounds.height / 2,
  )
  await page.mouse.down()
  await page.mouse.move(
    markerBounds.x + markerBounds.width / 2 + 8,
    markerBounds.y + markerBounds.height / 2,
  )
  await page.mouse.move(
    markerBounds.x + markerBounds.width / 2 + 24,
    markerBounds.y + markerBounds.height / 2,
  )
  await page.mouse.up()
  const timeInput = page.getByRole('spinbutton', { name: '拍点时间' })
  expect(Number(await timeInput.inputValue())).toBeGreaterThan(2.6)
  await timeInput.fill('')
  await timeInput.pressSequentially('2.625')
  await timeInput.blur()
  await expect(timeInput).toHaveValue('2.625')
  for (const label of [
    '拍点时间',
    '强拍位置',
    '拍号与弱起',
    '片段完整性',
  ]) {
    await page.getByRole('checkbox', { name: label }).check()
  }
  await page.getByRole('button', { name: '保存草稿' }).click()
  await expect(page.getByText('草稿已保存')).toBeVisible()
  expect(
    (
      savedDraft?.beats as Array<{ at: number; beat: number; measure: number }>
    )[2],
  ).toEqual({ at: 2.625, beat: 2, measure: 8, downbeat: false })

  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '提交独立审核' }).click()
  await expect(page.locator('.beat-review-status > strong')).toHaveText(
    '已提交',
  )
  expect(submittedDraft?.checks).toEqual({
    beat_times: true,
    downbeats: true,
    meter: true,
    completeness: true,
  })
})

test('bootstraps the admin owner and opens the operational dashboard', async ({
  page,
}) => {
  await page.route('**/api/admin/auth/session', (route) =>
    route.fulfill({
      json: { configured: false, authenticated: false, user: null },
    }),
  )
  await page.route('**/api/admin/auth/bootstrap', (route) =>
    route.fulfill({
      status: 201,
      json: {
        user: {
          id: 'owner-1',
          username: 'admin',
          displayName: 'Gold Owner',
          role: 'owner',
          active: true,
          createdAt: '2026-10-21T00:00:00.000Z',
        },
      },
    }),
  )
  await page.route('**/api/admin/dashboard', (route) =>
    route.fulfill({
      json: {
        generatedAt: '2026-10-21T08:00:00.000Z',
        corpus: {
          corpusVersion: 'beat-pilot-v2-test',
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
        layers: [
          { id: 'beat', label: '节拍', status: 'ready' },
          {
            id: 'onset',
            label: '起音 / 拨弦',
            status: 'blocked',
            dependsOn: 'beat',
          },
          {
            id: 'pitch',
            label: '音高',
            status: 'blocked',
            dependsOn: 'onset',
          },
          {
            id: 'string-fret',
            label: '弦品',
            status: 'blocked',
            dependsOn: 'pitch',
          },
          {
            id: 'chord',
            label: '和弦',
            status: 'blocked',
            dependsOn: 'string-fret',
          },
          {
            id: 'technique',
            label: '技法',
            status: 'blocked',
            dependsOn: 'chord',
          },
        ],
        gates: {
          beatF1: 0.95,
          medianErrorMs: 20,
          p95ErrorMs: 50,
          downbeatF1: 0.95,
        },
      },
    }),
  )

  await page.goto('/admin/login')
  await expect(
    page.getByRole('heading', { name: '创建所有者账号' }),
  ).toBeVisible()
  await page.getByLabel('显示名称').fill('Gold Owner')
  await page.getByLabel('密码').fill('correct-horse-battery')
  await page.getByRole('button', { name: '初始化并进入' }).click()

  await expect(page).toHaveURL(/\/admin$/u)
  await expect(page.getByRole('heading', { name: '仪表盘' })).toBeVisible()
  await expect(page.getByText('48', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('起音 / 拨弦').first()).toBeVisible()
  await expect(page.getByText('等待依赖').first()).toBeVisible()
})
