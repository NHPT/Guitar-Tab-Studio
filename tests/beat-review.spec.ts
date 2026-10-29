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
    if (path === '/api/admin/beat-gold/structure-reviews') {
      await route.fulfill({
        json: {
          reviews: [
            {
              recordingId: 'fixture',
              split: 'train',
              status: 'submitted',
              duration: 8,
              timeSignature: [4, 4],
              pickupBeats: 0,
              phaseStatus: 'confirmed',
              meterChangeCount: 0,
              checks: {
                meter: true,
                downbeat_phase: true,
                pickup: true,
                meter_changes: true,
              },
            },
          ],
        },
      })
      return
    }
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

  await page.goto('/admin/beat-review?stage=clips')
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
      .locator('.beat-review-timeline > button:not(.is-downbeat) span')
      .first(),
  ).toHaveCSS('opacity', '1')
  await expect(
    page
      .locator('.beat-review-timeline > button.is-context.is-downbeat span')
      .first(),
  ).toHaveText('1')

  const timeInput = page.getByRole('spinbutton', { name: '拍点时间' })
  if (!test.info().project.use.isMobile) {
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
    expect(Number(await timeInput.inputValue())).toBeGreaterThan(2.6)
  }
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

test('reviews full-song structure before enabling clip timing', async ({
  page,
}) => {
  let savedDraft: Record<string, unknown> | undefined
  let submittedDraft: Record<string, unknown> | undefined
  const review = {
    schema_version: 1,
    review_kind: 'beat-structure',
    status: 'draft',
    corpus_version: 'beat-pilot-v2-test',
    recording_id: 'fixture-song',
    annotator_alias: 'reviewer-a',
    audio_duration: 8,
    time_signature: [4, 4] as [number, number],
    pickup_beats: 0,
    phase_status: 'confirmed' as const,
    first_downbeat_at: 0.5,
    meter_changes: [] as Array<{
      at: number
      time_signature: [number, number]
    }>,
    checks: {
      meter: false,
      downbeat_phase: false,
      pickup: false,
      meter_changes: false,
    },
    submission: null,
  }
  const summary = () => ({
    recordingId: 'fixture-song',
    split: 'train',
    status: review.status,
    duration: review.audio_duration,
    timeSignature: review.time_signature,
    pickupBeats: review.pickup_beats,
    phaseStatus: review.phase_status,
    meterChangeCount: review.meter_changes.length,
    checks: review.checks,
  })
  const document = () => ({
    review,
    audioUrl: '/api/admin/beat-gold/structure-media/fixture-song',
    candidateBeats: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    candidateBpm: 60,
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
  await page.route('**/api/admin/beat-gold/structure-media/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'audio/wav',
      body: wavTone(8),
    }),
  )
  await page.route('**/api/admin/beat-gold/structure-reviews**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/admin/beat-gold/structure-reviews') {
      await route.fulfill({
        json: {
          reviews: [
            summary(),
            { ...summary(), recordingId: 'other-song', status: 'draft' },
          ],
        },
      })
      return
    }
    if (
      path === '/api/admin/beat-gold/structure-reviews/fixture-song' &&
      request.method() === 'GET'
    ) {
      await route.fulfill({ json: document() })
      return
    }
    if (
      path === '/api/admin/beat-gold/structure-reviews/fixture-song' &&
      request.method() === 'PUT'
    ) {
      savedDraft = request.postDataJSON() as Record<string, unknown>
      Object.assign(review, savedDraft)
      await route.fulfill({ json: document() })
      return
    }
    if (
      path ===
        '/api/admin/beat-gold/structure-reviews/fixture-song/submit' &&
      request.method() === 'POST'
    ) {
      submittedDraft = request.postDataJSON() as Record<string, unknown>
      Object.assign(review, submittedDraft, { status: 'submitted' })
      await route.fulfill({ json: document() })
      return
    }
    await route.fulfill({ status: 404, json: { error: 'not found' } })
  })

  await page.goto('/admin/reviews/beat?stage=clips')
  await expect(
    page.getByRole('button', { name: /全曲结构/u }),
  ).toHaveClass(/is-active/u)
  await expect(
    page.getByRole('button', { name: /片段时间/u }),
  ).toBeDisabled()
  await expect(
    page.getByRole('heading', { name: 'fixture-song' }),
  ).toBeVisible()
  await expect(page.getByText('自动分析结果')).toBeVisible()
  await expect(page.getByRole('combobox', { name: '时间窗' })).toHaveValue('4')
  await expect(
    page.getByRole('button', { name: '分析正确，确认并提交' }),
  ).toBeVisible()
  for (const label of ['开头', '中段', '结尾']) {
    await expect(
      page.getByRole('button', { name: new RegExp(`试听${label}`, 'u') }),
    ).toBeVisible()
  }
  page.once('dialog', (dialog) => dialog.dismiss())
  await page
    .getByRole('button', { name: '分析正确，确认并提交' })
    .click()
  assert.equal(submittedDraft, undefined)
  await page.getByRole('button', { name: '需要纠正' }).click()
  await page.getByRole('button', { name: '无法确认' }).click()
  await expect(
    page.getByRole('button', { name: '分析正确，确认并提交' }),
  ).toBeDisabled()
  await page.getByRole('button', { name: '可确认' }).click()
  await page.getByRole('button', { name: /0:02.50.*第 3 拍/u }).click()
  await page
    .getByRole('button', { name: '当前拍设为首个强拍' })
    .click()
  await page.getByRole('button', { name: '保存草稿' }).click()
  await expect(page.getByText('结构草稿已保存')).toBeVisible()
  assert(savedDraft)
  assert.equal(savedDraft.first_downbeat_at, 2.5)

  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '提交纠正结果' }).click()
  await expect(page.getByText('全曲结构已提交')).toBeVisible()
  assert(submittedDraft)
  assert.deepEqual(submittedDraft.checks, {
    meter: true,
    downbeat_phase: true,
    pickup: true,
    meter_changes: true,
  })
  await expect(
    page.getByRole('button', { name: /片段时间/u }),
  ).toBeEnabled()
})

for (const phase of ['confirmed', 'uncertain'] as const) {
  test(`uses a core queue for confirmed songs and preserves ${phase} phase`, async ({
    page,
  }) => {
    let savedDraft: Record<string, unknown> | undefined
    const checked = {
      meter: true,
      downbeat_phase: true,
      pickup: true,
      meter_changes: true,
    }
    const structure = {
      schema_version: 1,
      review_kind: 'beat-structure',
      status: 'submitted',
      corpus_version: 'beat-pilot-v2-test',
      recording_id: 'fixture',
      annotator_alias: 'reviewer-a',
      audio_duration: 60,
      time_signature: [4, 4],
      pickup_beats: 2,
      phase_status: phase,
      first_downbeat_at: phase === 'confirmed' ? 2.5 : null,
      meter_changes: [],
      checks: checked,
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
      audio_source_start: 0,
      audio_source_end: 8,
      audio_duration: 8,
      evaluation_start: 2,
      evaluation_end: 6,
      evaluation_duration: 4,
      time_signature: [4, 4],
      pickup_beats: 0,
      beats: markers([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], [4, 4]),
      tags: ['train', 'acoustic'],
      checks: {
        beat_times: false,
        downbeats: false,
        meter: false,
        completeness: false,
      },
    }
    const summaries = ['fixture', 'locked'].flatMap((recordingId) =>
      Array.from({ length: 5 }, (_, index) => ({
        clipId: `${recordingId}-${String(index + 1).padStart(2, '0')}`,
        recordingId,
        split: 'train',
        status: 'draft',
        duration: 4,
        sourceStart: index * 8 + 2,
        audioDuration: 8,
        evaluationStart: 2,
        evaluationEnd: 6,
        beatCount: 4,
        timeSignature: [4, 4],
        pickupBeats: 0,
        checks: review.checks,
        tags: review.tags,
      })),
    )
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
          },
        },
      }),
    )
    await page.route('**/api/admin/beat-gold**', async (route) => {
      const request = route.request()
      const path = new URL(request.url()).pathname
      if (path.includes('/media/')) {
        await route.fulfill({ contentType: 'audio/wav', body: wavTone(8) })
      } else if (path.endsWith('/structure-reviews')) {
        const summary = {
          recordingId: 'fixture',
          split: 'train',
          status: 'submitted',
          duration: 60,
          timeSignature: [4, 4],
          pickupBeats: 2,
          phaseStatus: phase,
          meterChangeCount: 0,
          checks: checked,
        }
        await route.fulfill({
          json: {
            reviews: [
              summary,
              { ...summary, recordingId: 'locked', status: 'draft' },
            ],
          },
        })
      } else if (path.endsWith('/structure-reviews/fixture')) {
        await route.fulfill({
          json: {
            review: structure,
            audioUrl: '',
            candidateBpm: 60,
            candidateBeats: Array.from({ length: 60 }, (_, index) => index + 0.5),
          },
        })
      } else if (path === '/api/admin/beat-gold') {
        await route.fulfill({
          json: {
            enabled: true,
            reviewers: ['reviewer-a'],
            manifest: { corpusVersion: 'beat-pilot-v2-test' },
          },
        })
      } else if (path.endsWith('/reviews')) {
        await route.fulfill({ json: { reviews: summaries } })
      } else if (path.endsWith('/reviews/fixture-01')) {
        if (request.method() === 'PUT') {
          savedDraft = request.postDataJSON() as Record<string, unknown>
          Object.assign(review, savedDraft)
        }
        await route.fulfill({ json: document() })
      } else {
        await route.fulfill({ status: 404, json: { error: 'not found' } })
      }
    })

    await page.goto('/admin/reviews/beat?stage=clips')
    await expect(page.getByRole('heading', { name: 'fixture-01' })).toBeVisible()
    await expect(page.getByRole('button', { name: /片段时间/u })).toBeEnabled()
    await expect(page.locator('.beat-review-list > button')).toHaveCount(
      phase === 'confirmed' ? 2 : 5,
    )
    await expect(page.locator('.beat-review-list')).not.toContainText('locked')
    await page.getByRole('combobox', { name: '片段队列' }).selectOption('all')
    await expect(page.locator('.beat-review-list > button')).toHaveCount(5)
    await page.getByRole('combobox', { name: '片段队列' }).selectOption('core')
    const apply = page.getByRole('button', { name: '应用全曲结构' })
    if (phase === 'uncertain') {
      await expect(apply).toBeDisabled()
      await expect(page.getByText('4/4 · 强拍相位待裁决')).toBeVisible()
      return
    }
    await expect(apply).toBeEnabled()
    page.once('dialog', (dialog) => dialog.accept())
    await apply.click()
    await expect(
      page.getByRole('button', { name: '第 1 小节第 1 拍，2.500 秒' }),
    ).toBeVisible()
    page.once('dialog', (dialog) => dialog.dismiss())
    await page
      .getByRole('navigation', { name: '节拍复核阶段' })
      .getByRole('button', { name: /全曲结构/u })
      .click()
    await expect(page.getByRole('heading', { name: 'fixture-01' })).toBeVisible()
    await page.getByRole('button', { name: '保存草稿' }).click()
    await expect(page.getByText('草稿已保存', { exact: true })).toBeVisible()
    assert(savedDraft)
    assert.deepEqual(
      (savedDraft.beats as Array<{ at: number }>).map((marker) => marker.at),
      [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    )
    assert.equal(savedDraft.pickup_beats, 2)
    assert.deepEqual(savedDraft.checks, review.checks)
  })
}

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
