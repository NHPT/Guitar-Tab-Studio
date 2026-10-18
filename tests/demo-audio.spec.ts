import { expect, test } from '@playwright/test'

type AudioProbe = Window & {
  demoSources: Array<{ audio: HTMLMediaElement; gain?: GainNode }>
}

test('plays and mixes the generated Country Roads stems independently', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const probe = window as unknown as AudioProbe
    probe.demoSources = []
    const createSource = AudioContext.prototype.createMediaElementSource
    AudioContext.prototype.createMediaElementSource = function (audio) {
      const node = createSource.call(this, audio)
      const record: AudioProbe['demoSources'][number] = { audio }
      probe.demoSources.push(record)
      const connect = node.connect.bind(node)
      node.connect = ((destination: AudioNode, output?: number, input?: number) => {
        if (destination instanceof GainNode) record.gain = destination
        return connect(destination, output ?? 0, input ?? 0)
      }) as typeof node.connect
      return node
    }
  })
  await page.goto('/')
  await expect(page.locator('.track-strip')).toHaveCount(7)
  await expect(page.getByLabel('人声音量')).toBeVisible()
  await expect(page.getByLabel('吉他 1音量')).toBeVisible()
  await expect(page.getByLabel('吉他 2音量')).toBeVisible()
  await page.getByLabel('预备小节数').selectOption('0')
  await page.getByRole('button', { name: '播放', exact: true }).click()

  const mediaState = () => page.evaluate(() =>
    (window as unknown as AudioProbe).demoSources
      .filter(({ audio }) =>
        audio.getAttribute('src')?.includes('take-me-home-country-roads-'),
      )
      .map(({ audio, gain }) => ({
        role:
          /take-me-home-country-roads-(vocals|guitar-1|guitar-2|bass|drums|piano|other)\.m4a/u
            .exec(audio.src)?.[1] ?? 'unknown',
        time: audio.currentTime, duration: audio.duration,
        paused: audio.paused, gain: gain?.gain.value,
      })),
  )
  await expect.poll(async () => (await mediaState()).filter(
    (track) => !track.paused && track.time > 0.5,
  ).length).toBe(7)
  const state = await mediaState()
  expect(state).toHaveLength(7)
  expect(Math.max(...state.map(track => track.time)) -
    Math.min(...state.map(track => track.time))).toBeLessThan(0.075)
  for (const track of state) expect(track.duration).toBeCloseTo(194.147, 0)
  for (const track of state) {
    expect(track.gain).toBeGreaterThan(0.7)
    expect(track.gain).toBeCloseTo(state[0].gain!, 5)
  }

  await page.getByRole('button', { name: '吉他 1独奏', exact: true }).click()
  await expect.poll(async () =>
    (await mediaState()).filter(t => t.role !== 'guitar-1').every(t => t.gain === 0),
  ).toBe(true)
  expect((await mediaState()).find(t => t.role === 'guitar-1')?.gain).toBeGreaterThan(0.7)
  await page.getByRole('button', { name: '吉他 1取消独奏', exact: true }).click()
  await page.getByRole('button', { name: '吉他 1静音', exact: true }).click()
  await expect.poll(async () => (await mediaState()).find(t => t.role === 'guitar-1')?.gain).toBe(0)
  expect((await mediaState()).find(t => t.role === 'vocals')?.gain).toBeGreaterThan(0.7)

  // All decoders must stay on the same media time after seeking.
  const waveform = page.locator('.waveform')
  await waveform.scrollIntoViewIfNeeded()
  const bounds = await waveform.boundingBox()
  await waveform.click({ position: { x: bounds!.width * 0.55, y: 20 } })
  await expect.poll(async () => Math.min(...(await mediaState()).map(t => t.time))).toBeGreaterThan(100)
  const sought = await mediaState()
  expect(Math.abs(sought[0].time - sought[1].time)).toBeLessThan(0.075)
  await page.getByRole('button', { name: '暂停', exact: true }).click()
  await expect.poll(async () => (await mediaState()).every(t => t.paused)).toBe(true)
})
