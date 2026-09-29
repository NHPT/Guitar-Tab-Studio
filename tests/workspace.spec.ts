import { expect, test } from '@playwright/test'

test('loads a playable project without layout overflow', async ({ page }, testInfo) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '午夜练习段落' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '吉他谱' })).toBeVisible()
  await expect(page.locator('.score-measure')).toHaveCount(8)
  await expect(page.locator('.chord-diagram')).toHaveCount(8)
  await expect(page.locator('.tab-notation')).toHaveCount(8)
  await expect(page.locator('.tab-string-line')).toHaveCount(48)
  await expect(page.locator('.rhythm-stem')).toHaveCount(32)
  await expect(page.locator('.track-strip')).toHaveCount(3)
  await expect(page.locator('.full-score')).not.toHaveClass(/is-expanded/)
  await expect(page.locator('.inspector')).toHaveCount(0)
  const visualizerBounds = await page.locator('.waveform').boundingBox()
  expect(visualizerBounds?.width ?? 0).toBeGreaterThan(300)
  expect(visualizerBounds?.height ?? 0).toBeGreaterThanOrEqual(80)

  const viewportOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(viewportOverflow).toBeLessThanOrEqual(1)

  const loopToggle = page.getByRole('button', { name: '练习循环' })
  const loopStart = page.getByLabel('循环起始小节')
  const loopEnd = page.getByLabel('循环结束小节')
  await expect(loopToggle).toHaveAttribute('aria-pressed', 'false')
  await expect(loopStart).toBeDisabled()
  await expect(loopEnd).toBeDisabled()
  await expect(loopStart).toHaveValue('1')
  await expect(loopEnd).toHaveValue('8')
  await expect(loopEnd.locator('option:checked')).toHaveText('第 8 小节')
  await loopToggle.click()
  await expect(loopToggle).toHaveAttribute('aria-pressed', 'true')
  await loopStart.selectOption('3')
  await loopEnd.selectOption('2')
  await expect(loopStart).toHaveValue('2')
  await expect(loopEnd).toHaveValue('2')
  await loopEnd.selectOption('4')
  await expect(loopStart).toHaveValue('2')
  await expect(loopEnd).toHaveValue('4')
  await loopToggle.click()

  const play = page.getByRole('button', { name: '播放', exact: true })
  await play.click()
  await expect(page.getByRole('button', { name: '暂停' })).toBeVisible()
  await page.waitForTimeout(800)
  await expect(page.locator('.time-readout strong')).not.toHaveText('0:00')
  await expect(page.locator('.waveform')).toHaveClass(/is-playing/)

  await page.locator('.waveform').evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    element.dispatchEvent(
      new MouseEvent('click', {
        bubbles: true,
        clientX: bounds.left + bounds.width * 0.8,
        clientY: bounds.top + bounds.height / 2,
      }),
    )
  })
  await expect
    .poll(() =>
      page.locator('.score-track').evaluate(
        (track) => new DOMMatrix(getComputedStyle(track).transform).m41,
      ),
    )
    .toBeLessThan(0)

  const firstMute = page.getByRole('button', { name: '吉他静音' })
  await firstMute.click()
  await expect(page.getByRole('button', { name: '吉他取消静音' })).toHaveClass(/is-active/)

  await page.screenshot({
    path: `test-results/${testInfo.project.name}-workspace.png`,
    fullPage: testInfo.project.name === 'desktop',
  })

  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.full-score')).toHaveClass(/is-expanded/)
  await expect(page.locator('.score-measure')).toHaveCount(8)
})

test('opens local chord and note inspectors only after selecting notation', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.inspector')).toHaveCount(0)

  await page.getByRole('button', { name: /校对第 1 小节/ }).click()
  await expect(page.getByText('和弦校对')).toBeVisible()
  await expect(page.getByLabel('实际和弦（原调）')).toHaveValue('Em')
  await page.getByRole('button', { name: '关闭校谱检查器' }).click()
  await expect(page.locator('.inspector')).toHaveCount(0)

  await page.locator('.tab-glyph').first().click()
  await expect(page.getByText('单音校对')).toBeVisible()
  await expect(page.getByText('弦（1 为最细弦）')).toBeVisible()
  await expect(page.getByText('技法置信度')).toBeVisible()
  await expect(page.getByText('技法候选')).toBeVisible()
})

test('opens the Chinese link import workflow', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  await expect(page.getByRole('dialog', { name: '导入歌曲' })).toBeVisible()
  await page
    .getByRole('dialog', { name: '导入歌曲' })
    .getByRole('button', { name: '在线链接' })
    .click()
  await page
    .getByPlaceholder('粘贴哔哩哔哩、抖音或网易云音乐公开链接')
    .fill('分享视频 https://www.bilibili.com/video/BV1test 来自哔哩哔哩')
  await expect(page.locator('.platform-list .is-detected')).toContainText('哔哩哔哩')
  await expect(page.getByText('已识别：哔哩哔哩')).toBeVisible()
  await expect(page.getByRole('button', { name: '开始分析' })).toBeEnabled()
})

test('shows the numeric percentage while an import job is running', async ({ page }) => {
  await page.route('**/api/import/upload', async (route) => {
    await route.fulfill({
      json: {
        id: 'progress-test',
        status: 'queued',
        progress: 0,
        stageLabel: '等待处理',
      },
      status: 202,
    })
  })
  await page.route('**/api/jobs/progress-test', async (route) => {
    await route.fulfill({
      json: {
        id: 'progress-test',
        status: 'separating',
        progress: 35,
        stageLabel: 'Worker 正在分离音轨',
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  await page.locator('input[type="file"]').setInputFiles({
    name: 'progress-test.mp3',
    mimeType: 'audio/mpeg',
    buffer: Buffer.from('ID3-progress-test'),
  })

  const progress = page.getByRole('progressbar', { name: '歌曲分析进度' })
  await expect(progress).toHaveAttribute('aria-valuenow', '35')
  await expect(page.locator('.job-progress-value')).toHaveText('35%')
})

test('switches score parts without fabricating notation', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '贝斯', exact: true }).click()
  await expect(page.getByText('贝斯谱尚未生成')).toBeVisible()
  await page.getByRole('button', { name: '鼓', exact: true }).click()
  await expect(page.getByText('鼓谱尚未生成')).toBeVisible()
  await page.getByRole('button', { name: '吉他', exact: true }).click()
  await expect(page.locator('.score-measure')).toHaveCount(8)
})

test('does not advertise unsupported QQ Music online import', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴哔哩哔哩、抖音或网易云音乐公开链接')
    .fill('https://y.qq.com/n/ryqq/songDetail/example')
  await expect(page.getByRole('button', { name: '开始分析' })).toBeDisabled()
  await expect(page.getByText('尚未识别链接来源')).toBeVisible()
  await expect(page.locator('.platform-list span')).toHaveCount(3)
  await expect(page.locator('.platform-list')).not.toContainText('QQ 音乐')
})

test('supports recommended capo, manual position and disabling capo', async ({ page }) => {
  await page.goto('/')
  const capo = page.getByLabel('变调夹位置')
  await expect(capo).toHaveValue('0')
  await capo.selectOption('2')
  await expect(page.locator('.chord-diagram figcaption').first()).toHaveText('Dm')
  await expect(page.getByText('原调 Em')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /4弦0品/ }).first()).toBeVisible()

  await page.getByRole('checkbox').uncheck()
  await expect(capo).toBeDisabled()
  await expect(page.locator('.chord-diagram figcaption').first()).toHaveText('Em')
})

test('renders chord strums as arrows without x fret markers', async ({
  page,
}) => {
  await page.goto('/')
  const firstMeasure = page.locator('.score-measure').first()
  await expect(firstMeasure.locator('.strum-mark')).toHaveCount(2)
  await expect(firstMeasure.locator('.strum-muted')).toHaveCount(0)
  await expect(firstMeasure.locator('.strum-mark text')).toHaveCount(0)
  await expect(firstMeasure.locator('.chord-string-number')).toHaveText(['6', '1'])
})

test('shows advanced guitar technique notation and allows rhythm correction', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '午夜练习段落' })).toBeVisible()
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.technique-label').first()).toBeAttached()
  const techniqueText = await page.locator('.technique-label').allTextContents()
  expect(techniqueText).toEqual(
    expect.arrayContaining(['H', 'P', 'S', 'Harm.', 'P.M.', 'SLAP', 'TR']),
  )
  await expect(page.getByText('ARP.', { exact: true }).first()).toBeAttached()
  await expect(page.getByText('RASG.', { exact: true }).first()).toBeAttached()

  await page.getByRole('button', { name: /第 2 小节琶音/ }).click()
  await expect(page.getByText('节奏技法校对')).toBeVisible()
  await expect(page.getByLabel('节奏技法')).toHaveValue('arpeggio')
})

test('maps a relational technique to its source note on the TAB timeline', async ({
  page,
}) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json()
  const beat = project.tab.measures[0].beats[0]
  const source = beat.notes[0]
  const target = beat.notes[1]
  source.string = 3
  source.fret = 2
  source.at = 0.1
  source.technique = 'pick'
  target.string = 3
  target.fret = 4
  target.at = 0.32
  target.technique = 'hammer-on'
  target.techniqueConfidence = 0.91
  target.techniqueSource = 'transition-heuristic-v2'
  target.techniqueEvidence = [
    'same-string',
    'connected-notes',
    'weak-second-onset',
    'ascending-fret',
  ]
  target.techniqueCandidates = [
    {
      technique: 'hammer-on',
      confidence: 0.91,
      evidence: target.techniqueEvidence,
    },
  ]
  target.relatedNoteId = source.id

  await page.route('**/api/projects/demo', async (route) => {
    await route.fulfill({ json: project })
  })
  await page.goto('/')

  const firstMeasure = page.locator('.score-measure').first()
  await expect(firstMeasure.locator('.technique-connection')).toHaveCount(1)
  await expect(
    firstMeasure.locator('.technique-connection .technique-label'),
  ).toHaveText('H')
  const noteBounds = await firstMeasure.locator('.tab-glyph').evaluateAll((notes) =>
    notes.slice(0, 2).map((note) => note.getBoundingClientRect().x),
  )
  expect(noteBounds[1]).toBeGreaterThan(noteBounds[0])

  await firstMeasure.locator('.tab-glyph').nth(1).click()
  const techniqueConfidence = page
    .locator('.confidence')
    .filter({ hasText: '技法置信度' })
  await expect(techniqueConfidence).toBeVisible()
  await expect(techniqueConfidence.getByText('91%')).toBeVisible()
  await expect(page.getByText('同弦 · 音符衔接 · 后音弱起音 · 品位上行')).toBeVisible()
})

test('continues a legato relation across a measure boundary', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json()
  const sourceMeasure = project.tab.measures[0]
  const targetMeasure = project.tab.measures[1]
  const source = sourceMeasure.beats[0].notes[0]
  const target = targetMeasure.beats[0].notes[0]

  sourceMeasure.beats.forEach((beat: { notes: unknown[] }) => {
    beat.notes = []
  })
  targetMeasure.beats.forEach((beat: { notes: unknown[] }) => {
    beat.notes = []
  })
  source.string = 3
  source.fret = 2
  source.at = sourceMeasure.start + sourceMeasure.duration - 0.05
  source.technique = 'pick'
  target.string = 3
  target.fret = 7
  target.at = targetMeasure.start + 0.05
  target.technique = 'slide'
  target.techniqueConfidence = 0.9
  target.positionSource = 'legato-optimizer-v1'
  target.relatedNoteId = source.id
  sourceMeasure.beats[3].notes = [source]
  targetMeasure.beats[0].notes = [target]

  await page.route('**/api/projects/demo', async (route) => {
    await route.fulfill({ json: project })
  })
  await page.goto('/')

  const secondMeasure = page.locator('.score-measure').nth(1)
  const continuation = secondMeasure.locator('.technique-connection.is-continuation')
  await expect(continuation).toHaveCount(1)
  await expect(continuation.locator('.technique-label')).toHaveText('S')

  await secondMeasure.locator('.tab-glyph').click()
  await expect(page.getByText('连奏把位优化')).toBeVisible()
})

test('renders natural and artificial harmonic positions on the TAB', async ({
  page,
}) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json()
  const beat = project.tab.measures[0].beats[0]
  const natural = beat.notes[0]
  const artificial = beat.notes[1]
  natural.string = 6
  natural.fret = 7
  natural.at = 0.1
  natural.technique = 'harmonic'
  natural.harmonicType = 'natural'
  natural.positionConfidence = 0.92
  natural.positionSource = 'harmonic-optimizer-v1'
  artificial.string = 4
  artificial.fret = 4
  artificial.at = 0.32
  artificial.technique = 'harmonic'
  artificial.harmonicType = 'artificial'
  artificial.harmonicTouchFret = 16
  artificial.positionConfidence = 0.68
  artificial.positionSource = 'harmonic-optimizer-v1'

  await page.route('**/api/projects/demo', async (route) => {
    await route.fulfill({ json: project })
  })
  await page.goto('/')

  const firstMeasure = page.locator('.score-measure').first()
  await expect(firstMeasure.locator('.tab-glyph text')).toContainText([
    '<7>',
    '4<16>',
  ])

  await firstMeasure.locator('.tab-glyph').nth(1).click()
  await expect(page.getByText('指板位置置信度')).toBeVisible()
  await expect(page.getByText('人工泛音触弦品')).toBeVisible()
  await expect(page.getByLabel('人工泛音触弦品')).toHaveValue('16')
  await expect(page.getByText('泛音映射')).toBeVisible()
})

test('offers seven persistent real-time visualizer modes', async ({ page }) => {
  await page.goto('/')
  const picker = page.getByLabel('播放可视化模式')
  const canvas = page.locator('.visualizer-canvas')
  await expect(picker.locator('option')).toHaveCount(7)
  await expect(picker.locator('option[value="dots"]')).toHaveText('鼓点弹球')
  for (const mode of ['wave', 'glow', 'ribbon', 'ecg', 'spectrum', 'dots', 'reactor']) {
    await picker.selectOption(mode)
    await expect(page.locator('.waveform')).toHaveClass(new RegExp(`visualizer-${mode}`))
    await expect(canvas).toHaveAttribute('data-motion-scope', 'current-only')
    await expect(canvas).toHaveAttribute('data-future-visible', 'false')
    await expect(canvas).toHaveAttribute('data-reactor-rows', mode === 'reactor' ? '6' : '0')
    await expect(page.locator('.waveform-cursor')).toHaveCount(mode === 'ecg' ? 1 : 0)
    const bounds = await page.locator('.waveform').boundingBox()
    expect(bounds?.width ?? 0).toBeGreaterThan(300)
    expect(bounds?.height ?? 0).toBeGreaterThanOrEqual(80)
  }
  await picker.selectOption('ecg')
  await expect(page.locator('.waveform')).toHaveClass(/visualizer-ecg/)
  await page.locator('.waveform').evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    element.dispatchEvent(
      new MouseEvent('click', {
        bubbles: true,
        clientX: bounds.left + bounds.width * 0.5,
        clientY: bounds.top + bounds.height / 2,
      }),
    )
  })
  await expect
    .poll(() =>
      canvas.evaluate((element) => {
        const target = element as HTMLCanvasElement
        const context = target.getContext('2d')!
        return context
          .getImageData(0, 0, Math.floor(target.width * 0.45), target.height)
          .data.reduce((total, value) => total + value, 0)
      }),
    )
    .toBeGreaterThan(0)
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(canvas).toHaveAttribute(
    'data-signal-source',
    /^(audio|overview-fallback)$/,
  )
  const firstFrame = Number(await canvas.getAttribute('data-frame'))
  const futureChecksum = async () =>
    canvas.evaluate((element) => {
      const target = element as HTMLCanvasElement
      const context = target.getContext('2d')!
      const start = Math.floor(target.width * 0.72)
      const width = Math.floor(target.width * 0.16)
      return context
        .getImageData(start, 0, width, target.height)
        .data.reduce((total, value) => total + value, 0)
    })
  const firstFuture = await futureChecksum()
  expect(firstFuture).toBe(0)
  await page.waitForTimeout(220)
  const secondFrame = Number(await canvas.getAttribute('data-frame'))
  const secondFuture = await futureChecksum()
  expect(secondFrame).toBeGreaterThan(firstFrame)
  expect(secondFuture).toBe(firstFuture)
  await page.reload()
  await expect(picker).toHaveValue('ecg')
})

test('moves the score smoothly beneath a fixed center playhead', async ({ page }) => {
  await page.goto('/?project=a1028519-3b28-4160-bbdb-d0f94871b2ee')
  await expect(page.getByText('通用基线', { exact: true })).toBeVisible({
    timeout: 10_000,
  })
  await expect(page.locator('.score-playhead')).toBeVisible()
  await page.locator('.waveform').evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    element.dispatchEvent(
      new MouseEvent('click', {
        bubbles: true,
        clientX: bounds.left + bounds.width * 0.004,
        clientY: bounds.top + bounds.height / 2,
      }),
    )
  })
  await page.getByRole('button', { name: '播放', exact: true }).click()
  const motionSamples = await page.locator('.score-track').evaluate(async (track) => {
    const activeMeasure = Number(
      track
        .querySelector('.score-measure.is-active')
        ?.getAttribute('data-measure') ?? 1,
    )
    const marker =
      track.querySelector<HTMLElement>(
        `[data-measure="${activeMeasure + 1}"]`,
      ) ?? track.querySelector<HTMLElement>('.score-measure')
    const values: Array<{ time: number; x: number; measure: string | null }> = []
    for (let index = 0; index < 40; index += 1) {
      await new Promise(requestAnimationFrame)
      values.push({
        time: performance.now(),
        x: marker?.getBoundingClientRect().left ?? 0,
        measure:
          document
            .querySelector('.score-measure.is-active')
            ?.getAttribute('data-measure') ?? null,
      })
    }
    return values
  })
  const motionDeltas = motionSamples
    .slice(1)
    .map((sample, index) => ({
      velocity:
        (sample.x - motionSamples[index].x) /
        ((sample.time - motionSamples[index].time) / 1000),
      changedMeasure: sample.measure !== motionSamples[index].measure,
    }))
  const stableVelocities = motionDeltas
    .filter((sample) => !sample.changedMeasure)
    .map((sample) => sample.velocity)
  motionDeltas.forEach((sample, index) => {
    if (!sample.changedMeasure || index === 0 || index + 1 >= motionDeltas.length) {
      return
    }
    const neighboringVelocity = Math.max(
      Math.abs(motionDeltas[index - 1].velocity),
      Math.abs(motionDeltas[index + 1].velocity),
    )
    expect(Math.abs(sample.velocity)).toBeLessThanOrEqual(
      neighboringVelocity * 1.25,
    )
  })
  expect(stableVelocities.length).toBeGreaterThan(30)
  expect(
    stableVelocities.filter((velocity) => Math.abs(velocity) < 6).length,
  ).toBeLessThanOrEqual(1)
  expect(stableVelocities.filter((velocity) => velocity > 0)).toHaveLength(0)

  const alignment = await page.evaluate(() => {
    const score = document.querySelector('.full-score')!.getBoundingClientRect()
    const playhead = document.querySelector('.score-playhead')!.getBoundingClientRect()
    return Math.abs(playhead.left - (score.left + score.width / 2))
  })
  expect(alignment).toBeLessThanOrEqual(1)
})

test('keeps the complete long score available past measure seven', async ({
  page,
}) => {
  const projectId = 'a1028519-3b28-4160-bbdb-d0f94871b2ee'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()

  await page.goto(`/?project=${projectId}`)
  await expect(page.locator('.score-measure')).toHaveCount(
    project.tab.measures.length,
  )
  await expect(
    page.locator(
      `.score-measure[data-measure="${project.tab.measures.length}"]`,
    ),
  ).toHaveCount(1)

  await page.getByRole('button', { name: '练习循环' }).click()
  await page.getByLabel('循环起始小节').selectOption('7')
  await page.getByLabel('循环结束小节').selectOption('9')
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(
    page.locator('.score-measure[data-measure="7"]'),
  ).toHaveClass(/is-active/)
  await expect(
    page.locator('.score-measure[data-measure="8"]'),
  ).toHaveClass(/is-active/, { timeout: 5_000 })
  const renderedEighthMeasure = await page
    .locator('.score-measure[data-measure="8"]')
    .evaluate((measure) => {
      const score = document.querySelector('.full-score')!.getBoundingClientRect()
      const bounds = measure.getBoundingClientRect()
      return {
        visibleWidth:
          Math.min(bounds.right, score.right) -
          Math.max(bounds.left, score.left),
        glyphs: measure.querySelectorAll('.tab-glyph').length,
      }
    })
  expect(renderedEighthMeasure.visibleWidth).toBeGreaterThan(0)
  expect(renderedEighthMeasure.glyphs).toBeGreaterThan(0)
})

test('shows only detected stems for a real inference project', async ({ page }) => {
  const projectId = '1abfa729-5c72-4784-a5c1-ebd18b6301a0'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()
  expect(project.techniqueModel.version).toBe('heuristic-v3')
  expect(project.tab.measures[0].beats[0].notes[0].relatedNoteId).toBeUndefined()
  expect(
    project.tab.measures[0].beats[0].notes.map(
      (note: { string: number; fret: number }) => [note.string, note.fret],
    ),
  ).toEqual([
    [6, 0],
    [4, 2],
    [5, 2],
  ])

  await page.goto(`/?project=${projectId}`)
  await expect(page.getByText('1 条有效音轨')).toBeVisible()
  await expect(page.locator('.track-strip')).toHaveCount(1)
  await expect(page.getByText('贝斯', { exact: true }).last()).toBeVisible()
  await page.locator('.tab-glyph').first().click()
  const positionConfidence = page
    .locator('.confidence')
    .filter({ hasText: '指板位置置信度' })
  await expect(positionConfidence).toContainText('94%')
  await expect(page.getByText('和弦把位优化')).toBeVisible()
  await expect(page.locator('.technique-candidates')).toContainText('拨弦')
})

test('renders every real Worker v8 legato relation in the complex song', async ({
  page,
}) => {
  const projectId = 'a1028519-3b28-4160-bbdb-d0f94871b2ee'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()
  type RelationNote = {
    id: string
    relatedNoteId?: string
    technique: string
    string: number
    positionSource?: string
  }
  const notes = project.tab.measures.flatMap(
    (measure: { number: number; beats: Array<{ notes: RelationNote[] }> }) =>
      measure.beats.flatMap((beat) =>
        beat.notes.map((note) => ({ ...note, measure: measure.number })),
      ),
  )
  const noteById = new Map(notes.map((note) => [note.id, note]))
  const relations = notes.filter((note) => note.relatedNoteId)
  const crossMeasureRelations = relations.filter(
    (note) => noteById.get(note.relatedNoteId!)?.measure !== note.measure,
  )

  expect(project.techniqueModel.version).toBe('heuristic-v4')
  expect(notes.filter((note) => note.technique === 'hammer-on').length).toBeGreaterThan(0)
  expect(notes.filter((note) => note.technique === 'pull-off').length).toBeGreaterThan(0)
  expect(notes.filter((note) => note.technique === 'slide').length).toBeGreaterThan(0)
  expect(
    notes.filter((note) => note.positionSource === 'legato-optimizer-v1').length,
  ).toBeGreaterThan(0)
  expect(relations.every((note) => noteById.has(note.relatedNoteId!))).toBe(true)
  expect(
    relations.every(
      (note) => noteById.get(note.relatedNoteId!)?.string === note.string,
    ),
  ).toBe(true)

  await page.goto(`/?project=${projectId}`)
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.technique-connection')).toHaveCount(relations.length)
  await expect(
    page.locator('.technique-connection.is-continuation'),
  ).toHaveCount(crossMeasureRelations.length)
  await expect(
    page.locator('.technique-connection .technique-label', { hasText: 'H' }),
  ).not.toHaveCount(0)
  await expect(
    page.locator('.technique-connection .technique-label', { hasText: 'P' }),
  ).not.toHaveCount(0)
  await expect(
    page.locator('.technique-connection .technique-label', { hasText: 'S' }),
  ).not.toHaveCount(0)
})

test('renders the first real measure with aligned 4/4 rhythm and legato', async ({
  page,
}) => {
  const projectId = 'a1028519-3b28-4160-bbdb-d0f94871b2ee'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()
  const firstMeasure = project.tab.measures[0]

  expect(project.bpm).toBe(123.05)
  expect(firstMeasure.start).toBeCloseTo(0.6037, 3)
  expect(firstMeasure.chord).toBe('A#m9')
  expect(firstMeasure.beats).toHaveLength(4)
  expect(firstMeasure.beats.map((beat: { style: string }) => beat.style)).toEqual([
    'pick',
    'pick',
    'pick',
    'pick',
  ])

  await page.goto(`/?project=${projectId}`)
  await expect(page.getByText('通用基线', { exact: true })).toBeVisible({
    timeout: 10_000,
  })
  const loopEnd = page.getByLabel('循环结束小节')
  await expect(loopEnd).toHaveValue(String(project.tab.measures.length))
  await expect(loopEnd.locator('option:checked')).toHaveText(
    `第 ${project.tab.measures.length} 小节`,
  )
  const loopEndWidth = await loopEnd.evaluate(
    (select) => select.parentElement?.getBoundingClientRect().width ?? 0,
  )
  expect(loopEndWidth).toBeGreaterThanOrEqual(110)
  const renderedMeasureWidths = await page
    .locator('.score-measure')
    .evaluateAll((measures) =>
      measures.map((measure) => {
        const notation = measure.querySelector<SVGSVGElement>('.tab-notation')!
        return {
          number: Number(measure.getAttribute('data-measure')),
          width: notation.getBoundingClientRect().width,
          viewBoxWidth: notation.viewBox.baseVal.width,
          timelineStart: Number(notation.dataset.timelineStart),
          timelineEnd: Number(notation.dataset.timelineEnd),
        }
      }),
    )
  const timelineSpeeds = renderedMeasureWidths.map((rendered) => {
    const { number, width, viewBoxWidth, timelineStart, timelineEnd } = rendered
    const measure = project.tab.measures[number - 1]
    return (
      (width * (timelineEnd - timelineStart)) /
      viewBoxWidth /
      measure.duration
    )
  })
  expect(Math.max(...timelineSpeeds) - Math.min(...timelineSpeeds)).toBeLessThan(1)
  const measure = page.locator('.score-measure').first()
  await expect(measure).toHaveClass(/is-active/)
  await expect(measure.locator('.arpeggio-mark')).toHaveCount(0)
  await expect(measure.locator('.rhythm-beat-slot')).toHaveCount(4)
  await expect(measure.locator('.rhythm-rest')).toHaveCount(0)
  await expect(measure.locator('.rhythm-tuplet')).toHaveCount(0)
  await expect
    .poll(() =>
      measure.locator('.rhythm-beat-slot').evaluateAll((slots) =>
        slots.map((slot) => slot.getAttribute('data-event-count')),
      ),
    )
    .toEqual(['5', '4', '5', '4'])
  await expect(measure.locator('.time-signature text')).toHaveText(['4', '4'])
  await expect(measure.locator('.chord-diagram figcaption')).toHaveText('Am9')
  expect(await measure.locator('.tab-glyph text').allTextContents()).toEqual([
    '0',
    '2',
    '0',
    '0',
    '1',
    '1',
    '0',
    '2',
    '0',
    '0',
    '2',
    '0',
    '0',
    '1',
    '12',
    '2',
    '12',
    '2',
  ])
  await expect(measure.locator('.technique-connection')).toHaveCount(2)
  await expect(
    measure.locator('.technique-connection .technique-label'),
  ).toHaveText(['H', 'H'])

  const firstNote = firstMeasure.beats
    .flatMap((beat: { notes: Array<{ at: number; notationAt?: number }> }) => beat.notes)
    .sort(
      (
        left: { at: number; notationAt?: number },
        right: { at: number; notationAt?: number },
      ) => (left.notationAt ?? left.at) - (right.notationAt ?? right.at),
    )[0]
  await page.locator('.waveform').evaluate(
    (element, seekRatio) => {
      const bounds = element.getBoundingClientRect()
      element.dispatchEvent(
        new MouseEvent('click', {
          bubbles: true,
          clientX: bounds.left + bounds.width * seekRatio,
          clientY: bounds.top + bounds.height / 2,
        }),
      )
    },
    firstNote.at / project.duration,
  )
  await expect
    .poll(async () =>
      page.evaluate(() => {
        const playhead = document
          .querySelector('.score-playhead')!
          .getBoundingClientRect()
        const firstGlyph = document
          .querySelector('.score-measure:first-child .tab-glyph')!
          .getBoundingClientRect()
        return Math.abs(
          firstGlyph.left +
            firstGlyph.width / 2 -
            (playhead.left + playhead.width / 2),
        )
      }),
    )
    .toBeLessThanOrEqual(2)

  await measure.locator('.tab-glyph').first().click()
  await expect(page.getByText('来源谱验证', { exact: true })).toHaveCount(2)
})

test('renders a real detected natural harmonic at the correct touch fret', async ({
  page,
}) => {
  await page.goto('/?project=c2442ea5-5a8c-4e64-94b4-509b20fb3b8c')
  await expect(page.locator('.chord-diagram figcaption').first()).toHaveText('N.C.')
  const harmonic = page.getByRole('button', { name: '6弦7品' })
  await expect(harmonic).toContainText('<7>')
  await harmonic.click()
  await expect(page.getByLabel('泛音类型')).toHaveValue('natural')
  await expect(
    page.locator('.confidence').filter({ hasText: '技法置信度' }),
  ).toContainText('94%')
  await expect(
    page.locator('.confidence').filter({ hasText: '指板位置置信度' }),
  ).toContainText('92%')
  await expect(page.locator('.technique-candidates')).toContainText('泛音倍频簇')
})

test('persists one of ten theme presets', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '选择界面配色' }).click()
  await expect(page.getByRole('menuitemradio')).toHaveCount(10)
  await page.getByRole('menuitemradio', { name: '夜谱' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'night')
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'night')
})

test('supports select-all in link input and rejects encrypted local formats', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  const linkInput = page.getByPlaceholder(
    '粘贴哔哩哔哩、抖音或网易云音乐公开链接',
  )
  await linkInput.fill('https://music.163.com/song?id=123')
  await linkInput.press('Control+A')
  await expect
    .poll(() =>
      linkInput.evaluate(
        (input) =>
          (input as HTMLInputElement).selectionEnd! - (input as HTMLInputElement).selectionStart!,
      ),
    )
    .toBe('https://music.163.com/song?id=123'.length)

  await dialog.getByRole('button', { name: '本地上传' }).click()
  await dialog.locator('input[type="file"]').setInputFiles({
    name: 'protected-song.ncm',
    mimeType: 'audio/mpeg',
    buffer: Buffer.from('encrypted'),
  })
  await expect(page.getByText(/专有加密容器/)).toBeVisible()
})
