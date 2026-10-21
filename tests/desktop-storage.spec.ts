import { expect, test } from '@playwright/test'

test('lets a desktop administrator configure and run automatic cleanup', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  await page.addInitScript(() => {
    const gibibyte = 1024 ** 3
    let config = {
      enabled: true,
      triggerFreeBytes: 5 * gibibyte,
      targetFreeBytes: 8 * gibibyte,
      retentionHours: 24,
    }
    let lastRun:
      | {
          id: string
          reason: 'manual'
          startedAt: string
          completedAt: string
          outcome: 'completed'
          availableBeforeBytes: number
          availableAfterBytes: number
          removedBytes: number
          removedItems: number
          targetReached: boolean
        }
      | undefined

    const cleanupStatus = () => ({
      config,
      availableBytes: 6.25 * gibibyte,
      reclaimableBytes: 1.5 * gibibyte,
      reclaimableItems: 4,
      running: false,
      ...(lastRun ? { lastRun, history: [lastRun] } : { history: [] }),
    })
    const demoProject = {
      schemaVersion: 1,
      id: 'desktop-cleanup-demo',
      title: '桌面清理测试',
      artist: 'Local',
      source: {
        kind: 'upload',
        label: '本地文件',
        filename: 'fixture.wav',
      },
      bpm: 120,
      key: 'E minor',
      duration: 2,
      createdAt: '2026-10-10T00:00:00.000Z',
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

    Object.defineProperty(window, 'gtsDesktop', {
      configurable: false,
      value: {
        async invoke(request: { method: string; payload?: unknown }) {
          switch (request.method) {
            case 'capabilities':
              return {
                ffmpeg: true,
                ffprobe: true,
                ytDlp: true,
                worker: true,
                mode: 'inference',
              }
            case 'platforms':
              return []
            case 'demo-project':
              return demoProject
            case 'project-revisions':
            case 'recent-projects':
            case 'local-projects':
            case 'model-packages':
              return []
            case 'application-settings':
              return {
                language: 'zh-CN',
                languagePreference: 'system',
                systemLanguage: 'zh-CN',
              }
            case 'set-background-job-state':
              return request.payload
            case 'runtime-status':
              return {
                mode: 'desktop-local',
                providerMode: 'local',
                cloudAvailable: false,
                cloudReady: false,
                uploadsAudio: false,
                apiReady: true,
                storageReady: true,
                workerReady: true,
                storageScope: 'application-data',
                estimatedSeconds: { minimum: 60, maximum: 600 },
                resourceNote: '使用本机 CPU/GPU 与已安装模型',
                transcription: {
                  providers: [
                    {
                      id: 'basic-pitch',
                      label: 'Basic Pitch',
                      description: 'Built in',
                      licenseClass: 'commercial-compatible',
                      licenseName: 'Apache-2.0',
                      bundled: true,
                      variants: [],
                    },
                  ],
                  preference: {
                    providerId: 'basic-pitch',
                    selectionMode: 'auto',
                  },
                  recommendation: {
                    variant: 'small',
                    reason: 'Test hardware',
                  },
                  device: {
                    platform: 'darwin',
                    arch: 'arm64',
                    totalMemoryBytes: 8 * gibibyte,
                    logicalCpuCount: 4,
                    accelerator: 'cpu',
                  },
                  variantWeights: {
                    small: 'download-required',
                    medium: 'download-required',
                    large: 'download-required',
                  },
                  pipeline: [],
                  runtimeAvailable: true,
                  weightSource: 'built-in',
                  ready: true,
                  oauthConfigured: false,
                },
              }
            case 'storage-cleanup-status':
              return cleanupStatus()
            case 'set-storage-cleanup-config':
              config = request.payload as typeof config
              return cleanupStatus()
            case 'run-storage-cleanup': {
              const timestamp = new Date().toISOString()
              lastRun = {
                id: 'cleanup-test',
                reason: 'manual',
                startedAt: timestamp,
                completedAt: timestamp,
                outcome: 'completed',
                availableBeforeBytes: 6.25 * gibibyte,
                availableAfterBytes: 6.75 * gibibyte,
                removedBytes: 0.5 * gibibyte,
                removedItems: 2,
                targetReached: true,
              }
              return cleanupStatus()
            }
            default:
              throw new Error(`Unexpected desktop method: ${request.method}`)
          }
        },
        onOpenMedia() {
          return () => undefined
        },
        onOpenProject() {
          return () => undefined
        },
        onOpenSettings() {
          return () => undefined
        },
        onLanguageChanged() {
          return () => undefined
        },
        onTranscriptionProgress() {
          return () => undefined
        },
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '桌面运行时' }).click()
  const dialog = page.getByRole('dialog', { name: '桌面运行时' })
  await expect(dialog.getByText('自动清理', { exact: true })).toBeVisible()
  await expect(dialog.getByText('可用 6.3 GB · 可回收 1.5 GB')).toBeVisible()
  await expect(
    dialog
      .locator('.runtime-status-local > div')
      .filter({ hasText: '本地 API' })
      .locator('dd'),
  ).toHaveClass(/is-ready/u)
  await expect(
    dialog
      .locator('.runtime-status-local > div')
      .filter({ hasText: '推理 Worker' })
      .locator('dd'),
  ).toHaveClass(/is-ready/u)

  await dialog.locator('.runtime-cleanup-toggle').click()
  await dialog.getByLabel('启动阈值').fill('6')
  await dialog.getByLabel('清理目标').fill('9')
  await dialog.getByLabel('保留时间').fill('48')
  await dialog.getByRole('button', { name: '保存' }).click()
  await expect(dialog.getByText('自动清理配置已保存')).toBeVisible()
  await expect(dialog.locator('.runtime-cleanup-toggle input')).not.toBeChecked()

  await dialog.getByRole('button', { name: '立即清理' }).click()
  await expect(dialog.getByText('已释放 512 MB')).toBeVisible()
  await expect(dialog.getByText(/最近运行 .*释放 512 MB/)).toBeVisible()
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-storage-cleanup.png`,
  })
})
