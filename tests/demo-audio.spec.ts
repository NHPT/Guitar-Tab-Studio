import { expect, test } from '@playwright/test'

type AudioProbe = Window & {
  demoSources: Array<{ audio: HTMLMediaElement; gain?: GainNode }>
}

test('plays the authored guitar and guide vocal independently', async ({ page }) => {
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
  await expect(page.locator('.track-strip')).toHaveCount(2)
  await expect(page.getByLabel('人声音量')).toBeVisible()
  await expect(page.getByLabel('吉他音量')).toBeVisible()
  await page.getByLabel('预备小节数').selectOption('0')
  await page.getByRole('button', { name: '播放', exact: true }).click()

  const mediaState = () => page.evaluate(() =>
    (window as unknown as AudioProbe).demoSources
      .filter(({ audio }) => audio.getAttribute('src')?.includes('wayfaring-stranger-'))
      .map(({ audio, gain }) => ({
        role: audio.src.includes('guide-vocals') ? 'vocals' : 'guitar',
        time: audio.currentTime, duration: audio.duration,
        paused: audio.paused, gain: gain?.gain.value,
      })),
  )
  await expect.poll(async () => (await mediaState()).filter(
    (track) => !track.paused && track.time > 0.5,
  ).length).toBe(2)
  const state = await mediaState()
  expect(Math.max(...state.map(track => track.time)) -
    Math.min(...state.map(track => track.time))).toBeLessThan(0.05)
  for (const track of state) expect(track.duration).toBeCloseTo(174.545, 0)
  expect(state[0].gain).toBeCloseTo(state[1].gain!, 5)

  await page.getByRole('button', { name: '吉他独奏', exact: true }).click()
  await expect.poll(async () => (await mediaState()).find(t => t.role === 'vocals')?.gain).toBe(0)
  expect((await mediaState()).find(t => t.role === 'guitar')?.gain).toBeGreaterThan(0.8)
  await page.getByRole('button', { name: '吉他取消独奏', exact: true }).click()
  await page.getByRole('button', { name: '吉他静音', exact: true }).click()
  await expect.poll(async () => (await mediaState()).find(t => t.role === 'guitar')?.gain).toBe(0)
  expect((await mediaState()).find(t => t.role === 'vocals')?.gain).toBeGreaterThan(0.8)

  // Seek into the vocal section; both decoders must use the same media time.
  const waveform = page.locator('.waveform')
  await waveform.scrollIntoViewIfNeeded()
  const bounds = await waveform.boundingBox()
  await waveform.click({ position: { x: bounds!.width * 0.55, y: 20 } })
  await expect.poll(async () => Math.min(...(await mediaState()).map(t => t.time))).toBeGreaterThan(90)
  const sought = await mediaState()
  expect(Math.abs(sought[0].time - sought[1].time)).toBeLessThan(0.05)
  await page.getByRole('button', { name: '暂停', exact: true }).click()
  await expect.poll(async () => (await mediaState()).every(t => t.paused)).toBe(true)
})
