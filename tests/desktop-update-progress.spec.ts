import { expect, test } from '@playwright/test'

test('renders the complete desktop update flow and controls', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')

  await page.addInitScript(() => {
    type UpdateProgress = {
      phase:
        | 'available'
        | 'downloading'
        | 'paused'
        | 'ready'
        | 'failed'
        | 'dismissed'
      version: string
      percent: number
      transferredBytes: number
      totalBytes: number
      bytesPerSecond: number
      error?: string
    }

    let updateListener: ((progress: UpdateProgress) => void) | undefined
    let latestUpdate: UpdateProgress | null = null
    const updateInvocations: string[] = []
    const emitUpdate = (progress: UpdateProgress) => {
      latestUpdate = progress
      updateListener?.(progress)
    }
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
        if (method === 'update-status') return latestUpdate
        if (method.startsWith('update-')) {
          updateInvocations.push(method)
          if (!latestUpdate) return null
          if (method === 'update-install') return { installing: true }
          const phase = {
            'update-download': 'downloading',
            'update-pause': 'paused',
            'update-resume': 'downloading',
            'update-cancel': 'dismissed',
            'update-dismiss': 'dismissed',
          }[method] as UpdateProgress['phase']
          emitUpdate({
            ...latestUpdate,
            phase,
            bytesPerSecond:
              phase === 'downloading'
                ? latestUpdate.bytesPerSecond
                : 0,
          })
          return latestUpdate
        }
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
      __emitUpdateProgress: emitUpdate,
      __updateInvocations: updateInvocations,
    })
  })

  await page.goto('/')
  await expect(
    page.getByRole('heading', { name: '更新测试项目' }),
  ).toBeVisible()

  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: unknown) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'available',
      version: '0.1.0-preview.15',
      percent: 0,
      transferredBytes: 0,
      totalBytes: 0,
      bytesPerSecond: 0,
    })
  })

  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await expect(
    dialog.getByRole('heading', { name: '发现新版本' }),
  ).toBeVisible()
  await expect(dialog.getByText('0.1.0-preview.15')).toBeVisible()
  await page.waitForTimeout(250)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-update-available.png`,
  })
  await dialog.getByRole('button', { name: '下载更新' }).click()

  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: unknown) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'downloading',
      version: '0.1.0-preview.15',
      percent: 37.4,
      transferredBytes: 225 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 10 * 1024 ** 2,
    })
  })

  const status = page.getByRole('status', { name: '应用更新下载进度' })
  await expect(status).toBeVisible()
  await expect(status.getByText('正在下载应用更新')).toBeVisible()
  await expect(status.getByText('37%')).toBeVisible()
  await expect(status.getByText('225.0 MB / 600.0 MB')).toBeVisible()
  await expect(status.getByText('10.0 MB/s')).toBeVisible()
  await expect(status.getByRole('progressbar')).toHaveAttribute(
    'aria-valuenow',
    '37',
  )

  await status.getByRole('button', { name: '暂停下载' }).click()
  await expect(status.getByText('更新下载已暂停')).toBeVisible()
  await status.getByRole('button', { name: '继续下载' }).click()
  await expect(status.getByText('正在下载应用更新')).toBeVisible()

  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: unknown) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'failed',
      version: '0.1.0-preview.15',
      percent: 37.4,
      transferredBytes: 225 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 0,
      error: '网络连接已中断',
    })
  })
  await expect(status.getByText('更新下载失败')).toBeVisible()
  await expect(status.getByText('网络连接已中断')).toBeVisible()
  await status.getByRole('button', { name: '重试下载' }).click()
  await expect(status.getByText('正在下载应用更新')).toBeVisible()

  await page.setViewportSize({ width: 390, height: 844 })
  const bounds = await status.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.y).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(844)
  expect(
    await status.evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    ),
  ).toBe(true)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-update-downloading.png`,
  })

  await status.getByRole('button', { name: '取消下载' }).click()
  await expect(status).toHaveCount(0)

  await page.evaluate(() => {
    const emit = (
      window as typeof window & {
        __emitUpdateProgress: (progress: unknown) => void
      }
    ).__emitUpdateProgress
    emit({
      phase: 'ready',
      version: '0.1.0-preview.15',
      percent: 100,
      transferredBytes: 600 * 1024 ** 2,
      totalBytes: 600 * 1024 ** 2,
      bytesPerSecond: 0,
    })
  })
  await expect(dialog).toBeVisible()
  await expect(
    dialog.getByRole('heading', { name: '更新已准备好' }),
  ).toBeVisible()
  await expect(
    dialog.getByRole('button', { name: '重启并安装' }),
  ).toBeVisible()
  await expect(
    dialog.getByRole('button', { name: '退出时安装' }),
  ).toBeVisible()
  expect(
    await dialog.evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    ),
  ).toBe(true)
  await page.waitForTimeout(250)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-update-ready.png`,
  })

  expect(
    await page.evaluate(
      () =>
        (
          window as typeof window & {
            __updateInvocations: string[]
          }
        ).__updateInvocations,
    ),
  ).toEqual([
    'update-download',
    'update-pause',
    'update-resume',
    'update-resume',
    'update-cancel',
  ])
})
