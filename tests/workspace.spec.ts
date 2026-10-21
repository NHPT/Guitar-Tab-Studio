import { randomUUID } from 'node:crypto'
import { expect, test } from '@playwright/test'

test('serves the requested Internationale project and generated stems', async ({
  page,
}) => {
  const projectResponse = await page.request.get(
    'http://127.0.0.1:8787/api/projects/demo',
  )
  expect(projectResponse.ok()).toBe(true)
  const project = await projectResponse.json()
  const guitars = project.tracks.filter(
    (track: { role: string }) => track.role === 'guitar',
  )
  expect(guitars.map((track: { url: string }) => track.url)).toEqual([
    '/media/builtin/internationale-guitar.m4a',
  ])
  expect(guitars.map((track: { name: string }) => track.name)).toEqual(['吉他'])
  expect(project.title).toBe('国际歌')
  expect(project.artist).toBe('唐朝')
  expect(project.duration).toBeCloseTo(266.73333, 3)
  expect(project.tab.measures).toHaveLength(110)
  expect(project.tab.measures[0].section).toBe('弱起')
  expect(project.tab.measures[0].beats).toHaveLength(1)
  expect(project.tab.measures[0].start).toBe(0.3317)
  expect(project.tab.measures[1].section).toBe('前奏')
  expect(project.tab.measures[1].start).toBe(0.9433)
  expect(
    project.tab.measures.flatMap(
      (measure: { beats: Array<{ notes: unknown[] }> }) =>
        measure.beats.flatMap((beat) => beat.notes),
    ),
  ).toHaveLength(1462)
  expect(project.source).toEqual({
    kind: 'upload',
    label: '唐朝 · 国际歌',
    filename: 'internationale.ogg',
  })
  const vocals = project.tracks.find(
    (track: { role: string }) => track.role === 'vocals',
  )
  expect(vocals.url).toBe(
    '/media/builtin/internationale-vocals.m4a',
  )
  expect(vocals.available).toBe(true)
  expect(vocals.url).not.toBe(guitars[0].url)
  expect(project.tracks.map((track: { name: string }) => track.name)).toEqual([
    '人声',
    '吉他',
    '贝斯',
    '鼓组',
    '键盘',
    '其他',
  ])

  const audioResponse = await page.request.get(
    `http://127.0.0.1:8787${guitars[0].url}`,
    { headers: { Range: 'bytes=0-1023' } },
  )
  expect(audioResponse.status()).toBe(206)
  expect(audioResponse.headers()['content-type']).toContain('audio/mp4')
  expect(audioResponse.headers()['content-range']).toMatch(
    /^bytes 0-1023\/\d+$/u,
  )
  expect((await audioResponse.body()).byteLength).toBe(1024)
})

test('loads a playable project without layout overflow', async ({ page }, testInfo) => {
  await page.goto('/')
  await expect(
    page.getByRole('heading', {
      name: '国际歌',
    }),
  ).toBeVisible()
  await expect(page.getByRole('heading', { name: '吉他谱' })).toBeVisible()
  await expect(page.locator('.score-measure')).toHaveCount(110)
  await expect(page.locator('.chord-diagram')).toHaveCount(110)
  await expect(page.locator('.tab-notation')).toHaveCount(110)
  await expect(page.locator('.tab-string-line')).toHaveCount(660)
  await expect(page.locator('.rhythm-beat-slot')).toHaveCount(437)
  expect(await page.locator('.rhythm-stem').count()).toBeGreaterThan(150)
  await expect(page.locator('.track-strip')).toHaveCount(6)
  await expect(page.locator('.track-title strong')).toHaveText([
    '人声', '吉他', '贝斯', '鼓组', '键盘', '其他',
  ])
  await expect(page.locator('.time-readout')).toContainText('/ 4:26')
  const downbeatsClearBarlines = await page
    .locator('.score-measure')
    .evaluateAll((measures) => {
      const withNotes = measures.slice(1).filter((measure) =>
        measure.querySelector('.tab-glyph'),
      )
      return withNotes.length > 0 && withNotes.every((measure) => {
        const barline = measure.querySelector<SVGLineElement>('.tab-bar-line')
        const note = measure.querySelector<SVGGElement>('.tab-glyph')
        if (!barline || !note) return false
        return (
          note.getBoundingClientRect().left -
            barline.getBoundingClientRect().left >
          5
        )
      })
    })
  expect(downbeatsClearBarlines).toBe(true)
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
  await expect(loopEnd).toHaveValue('110')
  await expect(loopEnd.locator('option:checked')).toHaveText('第 110 小节')
  const countInBars = page.getByLabel('预备小节数')
  await expect(countInBars).toHaveValue('2')
  await countInBars.selectOption('0')
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
  await expect(
    page.getByRole('button', { name: '吉他取消静音' }),
  ).toHaveClass(/is-active/)

  await page.screenshot({
    path: `test-results/${testInfo.project.name}-workspace.png`,
    fullPage: testInfo.project.name === 'desktop',
  })

  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.full-score')).toHaveClass(/is-expanded/)
  await expect(page.locator('.score-measure')).toHaveCount(110)
})

test('switches the workspace between Chinese and English', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '设置' }).click()
  const settings = page.getByRole('dialog', { name: '应用设置' })
  await expect(settings).toBeVisible()
  await expect(settings.getByText('在线媒体登录态')).toHaveCount(0)
  await settings.getByRole('button', { name: 'English' }).click()
  await expect(page.getByText('AI guitar tabs and stem separation')).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Import song' }),
  ).toBeVisible()
  const englishSettings = page.getByRole('dialog', {
    name: 'Application settings',
  })
  await expect(
    englishSettings.getByRole('button', { name: 'English' }),
  ).toHaveAttribute('aria-pressed', 'true')
})

test('counts in before playback and filters the rhythm library', async ({
  page,
}, testInfo) => {
  await page.goto('/')
  const countInBars = page.getByLabel('预备小节数')
  const countInKit = page.getByLabel('预备节拍音色')
  const countInVolume = page.getByLabel('预备节拍音量')
  const rhythmToggle = page.getByRole('button', { name: '节奏伴奏' })
  const rhythmSelect = page.getByLabel('节奏型')
  const rhythmKit = page.getByLabel('鼓组音色')
  const rhythmVolume = page.getByLabel('节奏伴奏音量')
  const speed = page.getByLabel('播放速度')

  await expect(countInKit).toHaveValue('click')
  await expect(countInVolume).toHaveValue('0.8')
  await countInKit.selectOption('wood')
  await countInVolume.fill('0.35')
  await expect(rhythmSelect.locator('option')).toHaveCount(46)
  await expect(
    rhythmSelect.locator('option[value="house"]'),
  ).toHaveAttribute('disabled', '')
  await expect(
    rhythmSelect.locator('option[value="waltz-classic"]'),
  ).toHaveAttribute('disabled', '')
  await speed.fill('1.25')
  await expect(
    rhythmSelect.locator('option[value="rock-double-time"]'),
  ).not.toHaveAttribute('disabled', '')
  await expect(
    rhythmSelect.locator('option[value="house"]'),
  ).not.toHaveAttribute('disabled', '')
  await expect(
    rhythmSelect.locator('option[value="funk-pocket"]'),
  ).toHaveAttribute('disabled', '')
  await expect(
    page.getByText(/4\/4 · 121–124 BPM · 32\/46 可用/),
  ).toBeVisible()

  await rhythmSelect.selectOption('rock-double-time')
  await expect(rhythmToggle).toHaveAttribute('aria-pressed', 'false')
  await expect(rhythmKit).toBeDisabled()
  await expect(rhythmVolume).toBeDisabled()
  await rhythmToggle.click()
  await expect(rhythmToggle).toHaveAttribute('aria-pressed', 'true')
  await expect(rhythmKit).toBeEnabled()
  await expect(rhythmVolume).toBeEnabled()
  await rhythmKit.selectOption('electronic')
  await rhythmVolume.fill('0.45')

  await countInBars.selectOption('1')
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(page.locator('.count-in-status')).toContainText('预备 1/1 小节')
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-count-in.png`,
  })
  await expect(page.getByRole('button', { name: '暂停' })).toBeVisible({
    timeout: 5_000,
  })

  await page.getByRole('button', { name: '暂停' }).click()
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(page.locator('.count-in-status')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '暂停' })).toBeVisible()
  await page.getByRole('button', { name: '暂停' }).click()
  await page.getByRole('button', { name: '回到开始' }).click()
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(page.getByRole('button', { name: '取消预备' })).toBeVisible()
  await page.getByRole('button', { name: '取消预备' }).click()
  await page.reload()
  await expect(countInBars).toHaveValue('1')
  await expect(countInKit).toHaveValue('wood')
  await expect(countInVolume).toHaveValue('0.35')
  await expect(rhythmToggle).toHaveAttribute('aria-pressed', 'true')
  await expect(rhythmKit).toHaveValue('electronic')
  await expect(rhythmVolume).toHaveValue('0.45')
  await expect(rhythmSelect).toHaveValue('click-quarter')
  await speed.fill('1.25')
  await expect(rhythmSelect).toHaveValue('rock-double-time')

  await page.getByRole('button', { name: '播放', exact: true }).click()
  await page.getByRole('button', { name: '取消预备' }).click()
  await expect(page.getByRole('button', { name: '播放', exact: true })).toBeVisible()
  await expect(page.locator('.count-in-status')).toHaveCount(0)
})

test('runs the metronome and drum machine without starting song playback', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '节拍器 / 鼓机' })).toBeHidden()
  await page.getByRole('button', { name: '节拍器 / 鼓机' }).click()
  const practiceStart = page.getByRole('button', {
    name: '启动节拍器 / 鼓机',
  })
  const songPlay = page.getByRole('button', { name: '播放', exact: true })
  await expect(page.getByRole('heading', { name: '节拍器 / 鼓机' })).toBeVisible()
  await page
    .getByRole('spinbutton', { name: '练习速度', exact: true })
    .fill('112')
  await page.getByLabel('节拍细分').selectOption('2')
  await practiceStart.click()
  await expect(
    page.getByRole('button', { name: '停止节拍器 / 鼓机' }),
  ).toHaveAttribute('aria-pressed', 'true')
  await expect(songPlay).toBeHidden()
  await expect
    .poll(() => page.locator('.metronome-pulse-strip i.is-current').count())
    .toBe(1)

  await page.getByRole('button', { name: '鼓机', exact: true }).click()
  const drumPatterns = page.getByLabel('独立鼓机节奏型')
  await expect(drumPatterns).toBeVisible()
  for (const label of [
    '体育场摇滚加花',
    '渐进摇滚轮鼓',
    '流行过门加花',
  ]) {
    await expect(drumPatterns.locator('option', { hasText: label })).toHaveCount(
      1,
    )
  }
  await drumPatterns.selectOption('custom-drums')
  await expect(drumPatterns).toHaveValue('custom-drums')
  const customKick = page.getByRole('button', {
    name: '底鼓，第 2 步',
  })
  await expect(customKick).toHaveAttribute('aria-pressed', 'false')
  await customKick.click()
  await expect(customKick).toHaveAttribute('aria-pressed', 'true')
  await expect
    .poll(() =>
      page.evaluate(() => {
        const patterns = JSON.parse(
          window.localStorage.getItem('gts-custom-rhythms-v1') ?? '{}',
        ) as Record<string, Array<{ step: number; voice: string }>>
        return patterns['4/4']?.some(
          (hit) => hit.step === 1 && hit.voice === 'kick',
        )
      }),
    )
    .toBe(true)
  await drumPatterns.selectOption('rock-driving')
  await drumPatterns.selectOption('custom-drums')
  await expect(customKick).toHaveAttribute('aria-pressed', 'true')
  const practiceMeter = page.getByLabel('练习拍号')
  await practiceMeter.selectOption('6/8')
  await drumPatterns.selectOption('custom-drums')
  await expect(
    page.getByRole('button', { name: '底鼓，第 12 步' }),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: '底鼓，第 13 步' }),
  ).toHaveCount(0)
  await practiceMeter.selectOption('4/4')
  await drumPatterns.selectOption('custom-drums')
  await expect(customKick).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.practice-step-sequencer > div')).toHaveCount(4)
  const kickMute = page.getByRole('button', { name: '底鼓静音' })
  await kickMute.click()
  await expect(
    page.getByRole('button', { name: '底鼓取消静音' }),
  ).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: '停止节拍器 / 鼓机' }).click()
  await expect(practiceStart).toHaveAttribute('aria-pressed', 'false')
  await page.getByRole('button', { name: '返回分析' }).click()
  await expect(songPlay).toBeVisible()
})

test('opens the tuner and live transcription as separate tools', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: '调音器' }).click()
  await expect(page.getByRole('heading', { name: '调音器' })).toBeVisible()
  await expect(
    page.getByRole('button', { name: '开始调音' }),
  ).toBeVisible()
  await expect(page.locator('.tuner-strings button > span')).toHaveText([
    'AUTO',
    'E2',
    'A2',
    'D3',
    'G3',
    'B3',
    'E4',
  ])
  const lowE = page.getByRole('button', { name: '6弦 E2' })
  await lowE.click()
  await expect(lowE).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('heading', { name: '吉他谱' })).toBeHidden()

  await page.getByRole('button', { name: '实时记谱' }).click()
  await expect(
    page.getByRole('heading', { name: '实时记谱' }),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: '开始实时记谱' }),
  ).toBeVisible()
  await expect(page.locator('.live-tab > div')).toHaveCount(6)
  await expect(page.locator('.live-tab > div').first().locator('b')).toHaveCount(
    16,
  )
  await expect(page.getByRole('heading', { name: '调音器' })).toBeHidden()

  await page.getByRole('button', { name: '返回分析' }).click()
  await expect(page.getByRole('heading', { name: '吉他谱' })).toBeVisible()
})

test('stabilizes a harmonic-rich low E as the correct guitar string', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const mediaDevices = navigator.mediaDevices ?? {}
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: mediaDevices,
    })
    Object.defineProperty(mediaDevices, 'getUserMedia', {
      configurable: true,
      value: async () => {
        const context = new AudioContext()
        const destination = context.createMediaStreamDestination()
        const master = context.createGain()
        master.gain.value = 0.32
        master.connect(destination)
        const nodes: AudioNode[] = [master, destination]
        for (const [multiple, gainValue] of [
          [1, 0.03],
          [2, 0.75],
          [3, 0.02],
          [4, 0.15],
        ]) {
          const oscillator = context.createOscillator()
          const gain = context.createGain()
          oscillator.frequency.value = 82.4069 * multiple
          gain.gain.value = gainValue
          oscillator.connect(gain).connect(master)
          oscillator.start()
          nodes.push(oscillator, gain)
        }
        await context.resume()
        Object.assign(window, {
          __tunerFixture: { context, destination, nodes },
        })
        return destination.stream
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '调音器' }).click()
  await page.getByRole('button', { name: '开始调音' }).click()
  await expect(page.locator('.tuner-note strong')).toHaveText('E2')
  await expect(page.locator('.tuner-readout > b')).toHaveText('音准准确')
  await expect(page.locator('.tuner-meter-labels strong')).toContainText(
    '0.0 cent',
  )
  await page.getByRole('button', { name: '停止调音' }).click()
})

test('starts the song at its first detected measure after count-in', async ({
  page,
}) => {
  const projectId = 'a1028519-3b28-4160-bbdb-d0f94871b2ee'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()
  const firstMeasureStart = project.tab.measures[0].start

  await page.goto(`/?project=${projectId}`)
  await expect(page.getByText('通用基线', { exact: true })).toBeVisible()
  await expect(page.getByText(/4\/4 · 117–129 BPM · \d+\/46 可用/)).toBeVisible()
  await page.getByLabel('播放可视化模式').selectOption('ecg')
  await page.getByLabel('预备小节数').selectOption('1')
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(page.getByRole('button', { name: '取消预备' })).toBeVisible()
  await expect(page.getByText('预备 1/1 小节 · 123 BPM')).toBeVisible()

  const cursorPercent = await page
    .locator('.waveform-cursor')
    .evaluate((cursor) => Number.parseFloat((cursor as HTMLElement).style.left))
  expect(cursorPercent).toBeCloseTo(
    (firstMeasureStart / project.duration) * 100,
    1,
  )

  await page.getByRole('button', { name: '取消预备' }).click()
})

test('uses the project tempo for count-in when the first measure is irregular', async ({
  page,
}) => {
  const projectId = '6aa34f79-343c-48b9-b8d8-76029f6e3f8c'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json()
  const firstMeasure = project.tab.measures[0]
  const firstMeasureBpm =
    60 /
    (firstMeasure.beats.reduce(
      (total: number, beat: { duration: number }) => total + beat.duration,
      0,
    ) /
      firstMeasure.beats.length)
  expect(project.bpm).toBeCloseTo(117.45, 2)
  expect(firstMeasureBpm).toBeGreaterThan(124)

  await page.goto(`/?project=${projectId}`)
  await expect(page.getByRole('heading', { name: '程艾影' })).toBeVisible()
  await page.getByLabel('预备小节数').selectOption('1')
  await page.getByRole('button', { name: '节奏伴奏' }).click()
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect(page.getByText('预备 1/1 小节 · 117 BPM')).toBeVisible()
  await expect(page.getByRole('button', { name: '暂停' })).toBeVisible({
    timeout: 6_000,
  })
  await page.getByRole('button', { name: '暂停' }).click()
})

test('scales count-in tempo across projects and playback speeds', async ({
  page,
}) => {
  test.setTimeout(120_000)
  const projectIds = [
    'demo',
    'a1028519-3b28-4160-bbdb-d0f94871b2ee',
    '6aa34f79-343c-48b9-b8d8-76029f6e3f8c',
    '1abfa729-5c72-4784-a5c1-ebd18b6301a0',
  ]
  let availabilityChangedWithSpeed = false

  for (const projectId of projectIds) {
    const response = await page.request.get(
      `http://127.0.0.1:8787/api/projects/${projectId}`,
    )
    const project = await response.json()
    const url = projectId === 'demo' ? '/' : `/?project=${projectId}`
    await page.goto(url)
    await expect(page.getByRole('heading', { name: project.title })).toBeVisible()
    await page.getByLabel('预备小节数').selectOption('1')

    let previousCompatibleCount: number | undefined
    for (const speed of [1, 1.25]) {
      await page.getByLabel('播放速度').fill(String(speed))
      const compatibleCount = await page
        .getByLabel('节奏型')
        .locator('option:not(:disabled)')
        .count()
      await expect(page.locator('.rhythm-availability')).toContainText(
        `${compatibleCount}/46 可用`,
      )
      if (
        previousCompatibleCount !== undefined &&
        previousCompatibleCount !== compatibleCount
      ) {
        availabilityChangedWithSpeed = true
      }
      previousCompatibleCount = compatibleCount
      await page.getByRole('button', { name: '播放', exact: true }).click()
      await expect(
        page.getByText(
          `预备 1/1 小节 · ${Math.round(project.bpm * speed)} BPM`,
        ),
      ).toBeVisible()
      await page.getByRole('button', { name: '取消预备' }).click()
    }
  }
  expect(availabilityChangedWithSpeed).toBe(true)
})

test('opens local chord and note inspectors only after selecting notation', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.inspector')).toHaveCount(0)

  await page.getByRole('button', { name: /校对第 2 小节/ }).click()
  await expect(page.getByText('和弦校对')).toBeVisible()
  await expect(page.getByLabel('实际和弦（原调）')).toHaveValue('D7')
  await page.getByRole('button', { name: '关闭校谱检查器' }).click()
  await expect(page.locator('.inspector')).toHaveCount(0)

  await page.locator('.tab-glyph').first().click()
  await expect(page.getByText('单音校对')).toBeVisible()
  await expect(page.getByText('弦（1 为最细弦）')).toBeVisible()
  await expect(page.getByText('技法置信度')).toBeVisible()
  await expect(page.getByRole('option', { name: '待确认', exact: true })).toBeAttached()
})

test('recognizes a YouTube link in the import workflow', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  await expect(page.getByRole('dialog', { name: '导入歌曲' })).toBeVisible()
  await page
    .getByRole('dialog', { name: '导入歌曲' })
    .getByRole('button', { name: '在线链接' })
    .click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill('https://www.youtube.com/watch?v=jNQXAC9IVRw')
  await expect(page.locator('.platform-list .is-detected')).toContainText('YouTube')
  await expect(page.getByText('已识别：YouTube')).toBeVisible()
  await expect(page.getByRole('button', { name: '开始分析' })).toBeEnabled()
})

test('always opens the local source in the local upload tab', async ({ page }) => {
  await page.goto('/')
  const sources = page.locator('.library-rail .rail-section').filter({
    hasText: '来源',
  })
  await sources.getByRole('button', { name: '在线链接' }).click()
  let dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await expect(
    dialog.getByRole('button', { name: '在线链接' }),
  ).toHaveClass(/is-active/u)
  await dialog.getByRole('button', { name: '关闭窗口' }).click()

  await sources.getByRole('button', { name: '本地音视频' }).click()
  dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await expect(
    dialog.getByRole('button', { name: '本地上传' }),
  ).toHaveClass(/is-active/u)
  await expect(
    dialog.getByRole('button', { name: '选择音频或视频' }),
  ).toBeVisible()
})

test('tries public media anonymously before any browser session', async ({
  page,
}) => {
  let submittedBody: Record<string, unknown> | undefined
  const url = 'https://www.youtube.com/watch?v=jNQXAC9IVRw'
  await page.route('**/api/import/link', async (route) => {
    submittedBody = route.request().postDataJSON()
    await route.fulfill({
      status: 202,
      json: {
        id: 'anonymous-media-job',
        source: { kind: 'youtube', label: 'YouTube', url },
        status: 'queued',
        progress: 0,
        stageLabel: '等待处理',
      },
    })
  })
  await page.route('**/api/jobs/anonymous-media-job', async (route) => {
    await route.fulfill({
      json: {
        id: 'anonymous-media-job',
        source: { kind: 'youtube', label: 'YouTube', url },
        status: 'failed',
        progress: 3,
        stageLabel: '处理失败',
        error: 'YouTube 要求登录会话或人机验证',
        errorCode: 'media-auth-required',
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill(url)
  await dialog.getByRole('button', { name: '开始分析' }).click()
  await expect(dialog.locator('.dialog-error')).toContainText('要求登录')
  expect(submittedBody?.mediaSession).toBeUndefined()
})

test('asks before retrying with the system default browser session', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const mediaSubmissions: Array<Record<string, unknown>> = []
    let submissionCount = 0
    const bridge = {
      invoke: async ({
        method,
        payload,
      }: {
        method: string
        payload?: Record<string, unknown>
      }) => {
        if (method === 'capabilities') {
          return (
            await (
              await fetch('http://127.0.0.1:8787/health')
            ).json()
          ).capabilities
        }
        if (method === 'platforms') {
          return (
            await (
              await fetch('http://127.0.0.1:8787/api/platforms')
            ).json()
          ).platforms
        }
        if (method === 'demo-project') {
          return (
            await fetch('http://127.0.0.1:8787/api/projects/demo')
          ).json()
        }
        if (method === 'project-revisions' || method === 'local-projects') {
          return []
        }
        if (method === 'application-settings') {
          return {
            language: 'zh-CN',
            languagePreference: 'system',
            systemLanguage: 'zh-CN',
          }
        }
        if (method === 'set-background-job-state') return payload
        if (method === 'detect-default-media-browser') {
          return {
            supported: true,
            browser: { id: 'chrome', label: 'Google Chrome' },
          }
        }
        if (method === 'open-external-url') return { opened: true }
        if (method === 'submit-link') {
          submissionCount += 1
          mediaSubmissions.push(payload ?? {})
          return {
            id: `browser-session-${submissionCount}`,
            source: {
              kind: 'youtube',
              label: 'YouTube',
              url: payload?.url,
            },
            status: 'queued',
            progress: 0,
            stageLabel: '等待处理',
          }
        }
        if (method === 'job') {
          const id = String(payload?.id)
          return {
            id,
            source: {
              kind: 'youtube',
              label: 'YouTube',
              url: 'https://www.youtube.com/watch?v=jNQXAC9IVRw',
            },
            status: id.endsWith('-1') ? 'failed' : 'cancelled',
            progress: 3,
            stageLabel: id.endsWith('-1') ? '处理失败' : '任务已取消',
            error: id.endsWith('-1')
              ? 'YouTube 要求登录会话或人机验证'
              : undefined,
            errorCode: id.endsWith('-1')
              ? 'media-auth-required'
              : undefined,
          }
        }
        throw new Error(`Unexpected desktop method: ${method}`)
      },
      onOpenMedia: () => () => undefined,
      onOpenProject: () => () => undefined,
      onOpenSettings: () => () => undefined,
      onLanguageChanged: () => () => undefined,
      onTranscriptionProgress: () => () => undefined,
    }
    Object.assign(window, {
      gtsDesktop: bridge,
      __mediaSubmissions: mediaSubmissions,
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill('https://www.youtube.com/watch?v=jNQXAC9IVRw')
  await dialog.getByRole('button', { name: '开始分析' }).click()
  await expect(
    dialog.getByRole('button', {
      name: '使用默认浏览器登录状态重试',
    }),
  ).toBeVisible()
  expect(
    await page.evaluate(
      () =>
        (window as Window & {
          __mediaSubmissions: Array<Record<string, unknown>>
        }).__mediaSubmissions[0].mediaSession,
    ),
  ).toBeUndefined()

  await dialog
    .getByRole('button', { name: '使用默认浏览器登录状态重试' })
    .click()
  await expect(
    dialog.getByText('仅为本次重试读取 Google Chrome'),
  ).toBeVisible()
  await dialog.getByRole('button', { name: '确认并重试' }).click()
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as Window & {
            __mediaSubmissions: Array<Record<string, unknown>>
          }).__mediaSubmissions.length,
      ),
    )
    .toBe(2)
  expect(
    await page.evaluate(
      () =>
        (window as Window & {
          __mediaSubmissions: Array<Record<string, unknown>>
        }).__mediaSubmissions[1].mediaSession,
    ),
  ).toEqual({ browser: 'chrome' })
})

test('shows the numeric percentage while an import job is running', async ({ page }) => {
  let cancelled = false
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
        status: cancelled ? 'cancelled' : 'transcribing',
        progress: 65,
        stageLabel: cancelled ? '任务已取消' : '正在识别吉他音符',
      },
    })
  })
  await page.route('**/api/jobs/progress-test/cancel', async (route) => {
    cancelled = true
    await route.fulfill({
      json: {
        id: 'progress-test',
        status: 'cancelled',
        progress: 65,
        stageLabel: '任务已取消',
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
  await expect(progress).toHaveAttribute('aria-valuenow', '65')
  await expect(page.locator('.job-progress-value')).toHaveText('65%')
  await page
    .getByRole('dialog', { name: '导入歌曲' })
    .getByRole('button', { name: '关闭窗口' })
    .click()
  await expect(page.getByRole('dialog', { name: '导入歌曲' })).toBeHidden()
  await expect(
    page.getByRole('button', { name: '打开分析进度' }),
  ).toContainText('65%')
  await page.getByRole('button', { name: '打开分析进度' }).click()
  await page.getByRole('button', { name: '取消任务' }).click()
  await expect(page.getByText('任务已取消')).toBeVisible()
})

test('keeps analysis errors selectable for standard copy commands', async ({
  page,
}) => {
  await page.route('**/api/import/upload', async (route) => {
    await route.fulfill({
      status: 507,
      json: {
        error:
          '可用磁盘空间不足，无法完成音轨分离。请释放更多磁盘空间后重试。',
      },
    })
  })
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.locator('input[type="file"]').setInputFiles({
    name: 'storage-test.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.concat([
      Buffer.from('RIFF'),
      Buffer.alloc(4),
      Buffer.from('WAVE'),
    ]),
  })
  const error = dialog.locator('.dialog-error')
  await expect(error).toContainText('可用磁盘空间不足')
  expect(
    await error.evaluate((element) => getComputedStyle(element).userSelect),
  ).toBe('text')
  await error.evaluate((element) => {
    const selection = window.getSelection()
    const range = document.createRange()
    range.selectNodeContents(element)
    selection?.removeAllRanges()
    selection?.addRange(range)
  })
  expect(await page.evaluate(() => window.getSelection()?.toString())).toContain(
    '可用磁盘空间不足',
  )
  await expect(dialog.getByRole('button', { name: '复制错误信息' })).toHaveCount(
    0,
  )
})

test('switches score parts without fabricating notation', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '贝斯', exact: true }).click()
  await expect(page.getByText('贝斯谱尚未生成')).toBeVisible()
  await expect(page.getByRole('button', { name: '鼓', exact: true })).toHaveCount(
    0,
  )
  await expect(page.locator('.track-title strong')).toContainText(['鼓组'])
  await page.getByRole('button', { name: '吉他', exact: true }).click()
  await expect(page.locator('.score-measure')).toHaveCount(110)
})

test('recognizes QQ Music as an online import source', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill('https://y.qq.com/n/ryqq_v2/songDetail/004Ti8rT003TaZ')
  await expect(page.getByRole('button', { name: '开始分析' })).toBeEnabled()
  await expect(page.getByText('已识别：QQ 音乐')).toBeVisible()
  await expect(page.locator('.platform-list span')).toHaveCount(12)
  await expect(page.locator('.platform-list')).toContainText('QQ 音乐')
})

test('supports recommended capo, manual position and disabling capo', async ({ page }) => {
  await page.goto('/')
  const capo = page.getByLabel('变调夹位置')
  const firstFullMeasureChord = page.locator('.chord-diagram figcaption').nth(1)
  await expect(capo).toHaveValue('0')
  await expect(firstFullMeasureChord).toHaveText('D7')
  await capo.selectOption('4')
  await expect(firstFullMeasureChord).toHaveText('A#7')

  await page.getByRole('checkbox').uncheck()
  await expect(capo).toBeDisabled()
  await expect(firstFullMeasureChord).toHaveText('D7')
})

test('creates a pitch-preserving easy layer and saves an immutable revision', async ({
  page,
}, testInfo) => {
  const response = await page.request.get(
    'http://127.0.0.1:8787/api/projects/demo',
  )
  const project = await response.json()
  project.id = `e2e-easy-${testInfo.project.name}-${randomUUID()}`
  await page.route('**/api/projects/demo', (route) =>
    route.fulfill({ json: project }),
  )
  await page.goto('/')
  const easy = page.getByRole('button', { name: '易弹版' })
  await easy.click()
  await expect(easy).toHaveClass(/is-active/)
  await expect(page.getByText('易弹版已生成，保存修订后持久化')).toBeVisible()

  await page.getByRole('button', { name: '保存修订' }).click()
  await expect(page.getByText('修订已保存')).toBeVisible()
  await expect(page.getByLabel('恢复项目修订')).toBeEnabled()
})

test('renders the generated Internationale opening TAB positions', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('checkbox').uncheck()
  const firstMeasure = page.locator('.score-measure').first()
  await expect(firstMeasure.locator('.strum-mark')).toHaveCount(0)
  await expect(firstMeasure.locator('.strum-muted')).toHaveCount(0)
  await expect(firstMeasure.getByText('弱起')).toBeVisible()
  await expect(firstMeasure.locator('.tab-glyph')).toHaveCount(4)
  await expect(firstMeasure.locator('.rhythm-event')).toHaveCount(2)
  await expect(firstMeasure.locator('.rhythm-event').nth(0)).toHaveAttribute(
    'data-duration-beats',
    '0.5',
  )
  await expect(firstMeasure.locator('.rhythm-event').nth(1)).toHaveAttribute(
    'data-duration-beats',
    '0.5',
  )
  await expect(firstMeasure.getByRole('button', { name: '2弦0品' })).toHaveCount(2)
  await expect(firstMeasure.getByRole('button', { name: '1弦7品' })).toHaveCount(2)
  const firstFullMeasure = page.locator('.score-measure').nth(1)
  await expect(firstFullMeasure.getByText('前奏')).toBeVisible()
  await expect(firstFullMeasure.locator('.tab-glyph')).toHaveCount(18)
  await expect(firstFullMeasure.getByRole('button', { name: '1弦10品' })).toHaveCount(3)
  await expect(firstFullMeasure.getByRole('button', { name: '4弦9品' })).toHaveCount(3)
  await expect(firstFullMeasure.getByRole('button', { name: '3弦7品' })).toHaveCount(4)
  await expect(firstFullMeasure.getByRole('button', { name: '1弦7品' })).toHaveCount(2)
})

test('shows the complete generated TAB and playable positions', async ({
  page,
}) => {
  await page.goto('/')
  await expect(
    page.getByRole('heading', {
      name: '国际歌',
    }),
  ).toBeVisible()
  await page.getByRole('checkbox').uncheck()
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.score-measure')).toHaveCount(110)
  await expect(page.locator('.tab-glyph')).toHaveCount(1462)
  await expect(page.locator('.score-review-status')).toContainText('尚未通过独立验证')

  await page.locator('.score-measure').first().getByRole('button', {
    name: '2弦0品',
  }).first().click()
  await expect(page.getByText('单音校对')).toBeVisible()
  await expect(page.locator('.inspector')).toBeVisible()
})

test('fits more score measures on wider windows and keeps playback controls visible', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  const measuresInFirstRow = () =>
    page.locator('.score-measure').evaluateAll((measures) => {
      const firstTop = measures[0]?.getBoundingClientRect().top ?? 0
      return measures.filter(
        (measure) =>
          Math.abs(measure.getBoundingClientRect().top - firstTop) < 2,
      ).length
    })
  await page.setViewportSize({ width: 1360, height: 1000 })
  await page.goto('/')
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect.poll(measuresInFirstRow).toBeGreaterThanOrEqual(4)
  const standardColumns = await measuresInFirstRow()
  await page.setViewportSize({ width: 1920, height: 1080 })
  await expect.poll(measuresInFirstRow).toBeGreaterThan(standardColumns)
  expect(await measuresInFirstRow()).toBeGreaterThanOrEqual(6)

  await page.locator('.score-measure').nth(20).scrollIntoViewIfNeeded()
  const transport = await page.locator('.transport-section').boundingBox()
  expect(transport).not.toBeNull()
  expect(transport!.y).toBeLessThanOrEqual(1)
  await expect(
    page.getByRole('button', { name: '播放', exact: true }),
  ).toBeVisible()
})

test('maps a relational technique to its source note on the TAB timeline', async ({
  page,
}) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json()
  const beat = project.tab.measures[0].beats[0]
  const source = { ...beat.notes[0] }
  const target = {
    ...source,
    id: `${source.id}-target`,
  }
  beat.style = 'pick'
  delete beat.direction
  beat.notes = [source, target]
  source.string = 3
  source.fret = 2
  source.at = beat.at + 0.1
  source.notationAt = source.at
  source.technique = 'pick'
  target.string = 3
  target.fret = 4
  target.at = beat.at + 0.32
  target.notationAt = target.at
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
  await page.getByRole('checkbox').uncheck()

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
  const completeMeasures = project.tab.measures.filter(
    (measure: { beats: unknown[] }) =>
      measure.beats.length === project.tab.timeSignature[0],
  )
  const sourceMeasure = completeMeasures[0]
  const targetMeasure = completeMeasures[1]
  const targetMeasureIndex = project.tab.measures.indexOf(targetMeasure)
  const source = { ...sourceMeasure.beats[0].notes[0] }
  const target = {
    ...source,
    id: `${source.id}-target`,
  }

  sourceMeasure.beats.forEach((beat: { notes: unknown[] }) => {
    beat.notes = []
  })
  targetMeasure.beats.forEach((beat: { notes: unknown[] }) => {
    beat.notes = []
  })
  source.string = 3
  source.fret = 2
  source.at = sourceMeasure.start + sourceMeasure.duration - 0.05
  source.notationAt = source.at
  source.technique = 'pick'
  target.string = 3
  target.fret = 7
  target.at = targetMeasure.start + 0.05
  target.notationAt = target.at
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

  const targetMeasureElement = page
    .locator('.score-measure')
    .nth(targetMeasureIndex)
  const continuation = targetMeasureElement.locator(
    '.technique-connection.is-continuation',
  )
  await expect(continuation).toHaveCount(1)
  await expect(continuation.locator('.technique-label')).toHaveText('S')

  await targetMeasureElement.locator('.tab-glyph').click()
  await expect(page.getByText('连奏把位优化')).toBeVisible()
})

test('renders natural and artificial harmonic positions on the TAB', async ({
  page,
}) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json()
  const beat = project.tab.measures[0].beats[0]
  const natural = { ...beat.notes[0] }
  const artificial = {
    ...natural,
    id: `${natural.id}-artificial`,
  }
  beat.style = 'pick'
  delete beat.direction
  beat.notes = [natural, artificial]
  natural.string = 6
  natural.fret = 7
  natural.at = 0.1
  natural.notationAt = natural.at
  natural.technique = 'harmonic'
  natural.harmonicType = 'natural'
  natural.positionConfidence = 0.92
  natural.positionSource = 'harmonic-optimizer-v1'
  artificial.string = 4
  artificial.fret = 4
  artificial.at = 0.32
  artificial.notationAt = artificial.at
  artificial.technique = 'harmonic'
  artificial.harmonicType = 'artificial'
  artificial.harmonicTouchFret = 16
  artificial.positionConfidence = 0.68
  artificial.positionSource = 'harmonic-optimizer-v1'

  await page.route('**/api/projects/demo', async (route) => {
    await route.fulfill({ json: project })
  })
  await page.goto('/')
  await page.getByRole('checkbox').uncheck()

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
  await page.getByLabel('预备小节数').selectOption('0')
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
  await page.getByLabel('预备小节数').selectOption('0')
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

test('plays every repaired project stem in sync without media or score rewind', async ({
  page,
}) => {
  test.setTimeout(60_000)
  await page.addInitScript(() => {
    const trackedWindow = window as typeof window & {
      __gtsTrackedAudio: HTMLAudioElement[]
    }
    trackedWindow.__gtsTrackedAudio = []
    const NativeAudio = window.Audio
    window.Audio = new Proxy(NativeAudio, {
      construct(Target, argumentsList) {
        const audio = Reflect.construct(
          Target,
          argumentsList,
        ) as HTMLAudioElement
        trackedWindow.__gtsTrackedAudio.push(audio)
        return audio
      },
    })
  })
  for (const projectId of [
    '6f579959-aebf-4d01-96c0-80de5a1eb8b7',
    '228e3288-429f-49c6-924e-fda060e366be',
  ]) {
    const project = await (
      await page.request.get(
        `http://127.0.0.1:8787/api/projects/${projectId}`,
      )
    ).json()
    const expectedSources = project.tracks
      .filter(
        (track: { available: boolean; url: string | null }) =>
          track.available && track.url,
      )
      .map((track: { url: string }) => track.url)
    await page.goto(`/?project=${projectId}`)
    await page.getByLabel('预备小节数').selectOption('0')
    await expect
      .poll(() =>
        page.evaluate(() => {
          const trackedWindow = window as typeof window & {
            __gtsTrackedAudio: HTMLAudioElement[]
          }
          return trackedWindow.__gtsTrackedAudio.length
        }),
      )
      .toBe(expectedSources.length)
    await expect
      .poll(() =>
        page.evaluate(() => {
          const trackedWindow = window as typeof window & {
            __gtsTrackedAudio: HTMLAudioElement[]
          }
          return Math.min(
            ...trackedWindow.__gtsTrackedAudio.map((audio) => audio.readyState),
          )
        }), { timeout: 20_000 },
      )
      .toBeGreaterThanOrEqual(1)

    await page.getByRole('button', { name: '播放', exact: true }).click()
    const result = await page.evaluate(async () => {
      const trackedWindow = window as typeof window & {
        __gtsTrackedAudio: HTMLAudioElement[]
      }
      const audios = trackedWindow.__gtsTrackedAudio
      let previousTimes = audios.map((audio) => audio.currentTime)
      let previousScoreX = new DOMMatrix(
        getComputedStyle(document.querySelector('.score-track')!).transform,
      ).m41
      let mediaRegressions = 0
      let scoreRegressions = 0
      let maximumSkew = 0
      for (let index = 0; index < 50; index += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 50))
        const times = audios.map((audio) => audio.currentTime)
        mediaRegressions += times.filter(
          (time, audioIndex) => time + 0.03 < previousTimes[audioIndex],
        ).length
        maximumSkew = Math.max(
          maximumSkew,
          Math.max(...times) - Math.min(...times),
        )
        const scoreX = new DOMMatrix(
          getComputedStyle(document.querySelector('.score-track')!).transform,
        ).m41
        if (scoreX > previousScoreX + 2) scoreRegressions += 1
        previousTimes = times
        previousScoreX = scoreX
      }
      return {
        mediaRegressions,
        scoreRegressions,
        maximumSkew,
        paused: audios.map((audio) => audio.paused),
        sources: audios.map((audio) => audio.currentSrc),
      }
    })
    for (const expectedSource of expectedSources) {
      expect(
        result.sources.filter((source) =>
          source.endsWith(expectedSource),
        ),
      ).toHaveLength(1)
    }
    expect(result.paused).toEqual(
      Array.from({ length: expectedSources.length }, () => false),
    )
    expect(result.mediaRegressions).toBe(0)
    expect(result.scoreRegressions).toBe(0)
    expect(result.maximumSkew).toBeLessThanOrEqual(0.04)
  }
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
  await page.getByLabel('预备小节数').selectOption('0')
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

test('withholds ungated Worker heuristic relations in the complex song', async ({
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
  const relations = notes.filter((note) => note.relatedNoteId)

  expect(project.techniqueModel.version).toBe('heuristic-v4')
  expect(notes.filter((note) => note.technique === 'hammer-on')).toHaveLength(0)
  expect(notes.filter((note) => note.technique === 'pull-off')).toHaveLength(0)
  expect(notes.filter((note) => note.technique === 'slide')).toHaveLength(0)
  expect(
    notes.filter((note) => note.positionSource === 'legato-optimizer-v1').length,
  ).toBeGreaterThan(0)
  expect(relations).toHaveLength(0)

  await page.goto(`/?project=${projectId}`)
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.technique-connection')).toHaveCount(0)
})

test('renders the first real measure with aligned 4/4 rhythm and gated techniques', async ({
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
  await expect(measure.locator('.technique-connection')).toHaveCount(0)

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

test('defaults to amber and persists one of eighteen theme presets', async ({
  page,
}) => {
  await page.goto('/')
  await page.evaluate(() => window.localStorage.removeItem('gts-theme'))
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'amber')
  await page.getByRole('button', { name: '选择界面配色' }).click()
  await expect(page.getByRole('menuitemradio')).toHaveCount(18)
  await page.getByRole('menuitemradio', { name: '夜谱' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'night')
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'night')
})

test('requires explicit authorization before protected file or link processing', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  const linkInput = page.getByPlaceholder(
    '粘贴 YouTube 或其他受支持的公开音视频链接',
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
  await expect(
    dialog.getByRole('heading', { name: '版权与处理确认' }),
  ).toBeVisible()
  await expect(dialog.getByText(/网易云音乐 加密文件/)).toBeVisible()
  await expect(dialog.getByText(/版权、有效许可或其他合法处理权/)).toBeVisible()
  await expect(dialog.getByText(/工具不核验或授予权利/)).toBeVisible()
  await expect(dialog.getByText(/点击“确认并继续”即表示/)).toBeVisible()
  await expect(dialog.getByText('API 服务端离线解码器已就绪')).toBeVisible()
  const confirm = dialog.getByRole('button', { name: '确认并继续' })
  await expect(confirm).toBeEnabled()
  await expect(dialog.getByRole('checkbox')).toHaveCount(0)
  await dialog.getByRole('button', { name: '取消', exact: true }).click()
  await expect(dialog.getByText('选择音频或视频')).toBeVisible()

  await page.locator('.app-shell').evaluate((element) => {
    const transfer = new DataTransfer()
    transfer.items.add(
      new File(['encrypted'], 'protected-song.kgma', {
        type: 'application/octet-stream',
      }),
    )
    element.dispatchEvent(
      new DragEvent('drop', {
        bubbles: true,
        cancelable: true,
        dataTransfer: transfer,
      }),
    )
  })
  await expect(
    dialog.getByRole('heading', { name: '版权与处理确认' }),
  ).toBeVisible()
  await expect(dialog.getByText(/酷狗音乐 加密文件/)).toBeVisible()
  await dialog.getByRole('button', { name: '取消', exact: true }).click()

  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill('https://music.163.com/song?id=123')
  await dialog.getByRole('button', { name: '开始分析' }).click()
  await expect(
    dialog.getByRole('heading', { name: '版权与处理确认' }),
  ).toBeVisible()
  await expect(dialog.getByText(/网易云音乐 可能受保护/)).toBeVisible()
})

test('submits versioned consent to the authorized media decoder', async ({
  page,
}) => {
  let submittedBody: Record<string, unknown> | undefined
  await page.route('**/api/import/link', async (route) => {
    submittedBody = route.request().postDataJSON()
    await route.fulfill({
      status: 202,
      json: {
        id: 'authorized-link',
        status: 'validating',
        progress: 1,
        stageLabel: '通过厂商授权服务准备临时媒体',
      },
    })
  })
  await page.route('**/api/jobs/authorized-link', async (route) => {
    await route.fulfill({
      json: {
        id: 'authorized-link',
        status: 'cancelled',
        progress: 1,
        stageLabel: '任务已取消',
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  await dialog.getByRole('button', { name: '在线链接' }).click()
  await page
    .getByPlaceholder('粘贴 YouTube 或其他受支持的公开音视频链接')
    .fill('https://music.163.com/song?id=123')
  await dialog.getByRole('button', { name: '开始分析' }).click()

  await expect(dialog.getByText('API 服务端离线解码器已就绪')).toBeVisible()
  const confirm = dialog.getByRole('button', { name: '确认并继续' })
  await expect(confirm).toBeEnabled()
  await confirm.click()

  await expect.poll(() => submittedBody).toEqual({
    url: 'https://music.163.com/song?id=123',
    authorizedProcessingConsent: true,
    authorizationPolicyVersion: 2,
  })
  await expect(dialog.getByText('任务已取消')).toBeVisible()
})
