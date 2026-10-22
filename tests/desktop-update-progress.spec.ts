import { expect, test } from '@playwright/test'

test('shows desktop update download progress and dismisses it when ready', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')

  await page.addInitScript(() => {
    type UpdateProgress = {
      phase: 'downloading' | 'ready' | 'failed'
      version: string
      percent: number
      transferredBytes: number
      totalBytes: number
      bytesPerSecond: number
    }

    let updateListener: ((progress: UpdateProgress) => void) | undefined
    const demoProject = {
      schemaVersion: 1,
      id: 'update-progress-demo',
      title: '更新测试项目',
      artist: 'Local',
      source: {
        kind: 'upload',
        label: 'Local recording',
        filename: 'fixture.wav',
      },
      bpm: 120,
      key: 'E minor',
      duration: 2,
      createdAt: '2026-10-21T00:00:00.000Z',
      analysisMode: 'demo',
      tracks: [],
      tab: {
        tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
        capo: 0,
        timeSignature: [4, 4],
        measures: [
          {
            number: 1,
            start: 0,
            duration: 2,
            chord: 'Em',
            beats: [0, 0.5, 1, 1.5].map((at, index) => ({
              id: `beat-${index}`,
              at,
              duration: 0.5,
              notes: [],
            })),
          },
        ],
      },
      warnings: [],
    }
    const bridge = {
      async invoke({
        method,
        payload,
      }: {
        method: string
        payload?: unknown
      }) {
        if (method === 'capabilities') {
          return {
            ffmpeg: true,
            ffprobe: true,
            ytDlp: true,
            worker: true,
            mode: 'inference',
          }
        }
        if (method === 'platforms') return []
        if (method === 'demo-project') return demoProject
        if (
          method === 'project-revisions' ||
          method === 'local-projects' ||
          method === 'model-packages'
        ) {
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
        throw new Error(`Unexpected desktop method: ${method}`)
      },
      onOpenMedia: () => () => undefined,
      onOpenProject: () => () => undefined,
      onOpenSettings: () => () => undefined,
      onLanguageChanged: () => () => undefined,
      onTranscriptionProgress: () => () => undefined,
      onUpdateProgress(listener: (progress: UpdateProgress) => void) {
        updateListener = listener
        return () => {
          updateListener = undefined
        }
      },
    }

    Object.assign(window, {
      gtsDesktop: bridge,
      __emitUpdateProgress: (progress: UpdateProgress) =>
        updateListener?.(progress),
    })
  })

  await page.goto('/')
  await expect(
    page.getByRole('heading', { name: '更新测试项目' }),
  ).toBeVisible()
  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: {
          phase: 'downloading'
          version: string
          percent: number
          transferredBytes: number
          totalBytes: number
          bytesPerSecond: number
        }) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'downloading',
      version: '0.1.0-preview.13',
      percent: 37.4,
      transferredBytes: 225 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 10 * 1024 ** 2,
    })
  })

  const status = page.getByRole('status', { name: '应用更新下载进度' })
  await expect(status).toBeVisible()
  await expect(status.getByText('正在下载应用更新')).toBeVisible()
  await expect(status.getByText('Guitar Tab Studio 0.1.0-preview.13')).toBeVisible()
  await expect(status.getByText('37%')).toBeVisible()
  await expect(status.getByText('225.0 MB / 600.0 MB')).toBeVisible()
  await expect(status.getByText('10.0 MB/s')).toBeVisible()
  await expect(status.getByRole('progressbar')).toHaveAttribute(
    'aria-valuenow',
    '37',
  )

  const bounds = await status.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.y).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(1440)
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(1000)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-update-progress.png`,
  })

  await page.setViewportSize({ width: 900, height: 640 })
  await expect(status).toBeVisible()
  const compactBounds = await status.boundingBox()
  expect(compactBounds).not.toBeNull()
  expect(compactBounds!.x).toBeGreaterThanOrEqual(0)
  expect(compactBounds!.y).toBeGreaterThanOrEqual(0)
  expect(compactBounds!.x + compactBounds!.width).toBeLessThanOrEqual(900)
  expect(compactBounds!.y + compactBounds!.height).toBeLessThanOrEqual(640)
  expect(
    await status.evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    ),
  ).toBe(true)

  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: {
          phase: 'ready'
          version: string
          percent: number
          transferredBytes: number
          totalBytes: number
          bytesPerSecond: number
        }) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'ready',
      version: '0.1.0-preview.13',
      percent: 100,
      transferredBytes: 600 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 0,
    })
  })
  await expect(status).toHaveCount(0)
})
