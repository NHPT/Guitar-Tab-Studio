import { expect, test } from '@playwright/test'
import type { DesktopTranscriptionStatus } from '@gts/core'

test('recovers from a closed transcription authorization page', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  await page.addInitScript(() => {
    const demoProject = {
      schemaVersion: 1,
      id: 'transcription-auth-demo',
      title: 'Authorization recovery',
      artist: 'Local',
      source: {
        kind: 'upload',
        label: 'Local recording',
        filename: 'fixture.wav',
      },
      bpm: 120,
      key: 'E minor',
      duration: 2,
      createdAt: '2026-10-16T00:00:00.000Z',
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
    const transcription: DesktopTranscriptionStatus = {
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
        {
          id: 'muscriptor',
          label: 'High accuracy',
          description: 'Multi-model',
          licenseClass: 'noncommercial',
          licenseName: 'CC BY-NC 4.0 weights',
          bundled: false,
          variants: [
            {
              id: 'medium',
              label: 'Medium',
              parameterCount: '307M',
              weightSizeBytes: 1_320_000_000,
              minimumMemoryBytes: 16 * 1024 ** 3,
            },
          ],
        },
        {
          id: 'guitar-specialist',
          label: '吉他专用',
          description: 'StringTrace',
          licenseClass: 'noncommercial',
          licenseName: 'Research model package',
          bundled: false,
          variants: [],
        },
        {
          id: 'multitrack-research',
          label: '多轨研究',
          description: 'YourMT3+',
          licenseClass: 'noncommercial',
          licenseName: 'Research-only dependency set',
          bundled: false,
          variants: [],
        },
      ],
      preference: {
        providerId: 'basic-pitch',
        selectionMode: 'auto',
      },
      recommendation: {
        variant: 'medium',
        reason: 'Test hardware',
      },
      device: {
        platform: 'darwin',
        arch: 'arm64',
        totalMemoryBytes: 32 * 1024 ** 3,
        logicalCpuCount: 8,
        accelerator: 'mps',
      },
      variantWeights: {
        small: 'download-required',
        medium: 'download-required',
        large: 'download-required',
      },
      pipeline: [
        {
          id: 'muscriptor',
          label: 'MuScriptor',
          role: 'primary-events',
          selected: true,
          runtimeAvailable: true,
          ready: false,
          weightSizeBytes: 1_320_000_000,
        },
        {
          id: 'basic-pitch',
          label: 'Basic Pitch',
          role: 'event-verification',
          selected: true,
          runtimeAvailable: true,
          ready: true,
        },
        {
          id: 'mr-mt3',
          label: 'MR-MT3',
          role: 'event-verification',
          selected: true,
          runtimeAvailable: true,
          ready: false,
          weightSizeBytes: 183_672_643,
        },
        {
          id: 'noise2fret',
          label: 'Noise2Fret',
          role: 'tablature-position',
          selected: true,
          runtimeAvailable: true,
          ready: false,
          weightSizeBytes: 111_317_004,
        },
        {
          id: 'stringtrace',
          label: 'StringTrace',
          role: 'tablature-position',
          selected: false,
          runtimeAvailable: true,
          ready: true,
        },
        {
          id: 'playability-decoder',
          label: 'Playability decoder',
          role: 'tablature-position',
          selected: false,
          runtimeAvailable: true,
          ready: true,
        },
        {
          id: 'yourmt3',
          label: 'YourMT3+',
          role: 'primary-events',
          selected: false,
          runtimeAvailable: true,
          ready: false,
          weightSizeBytes: 561_544_628,
        },
      ],
      runtimeAvailable: true,
      weightSource: 'built-in',
      ready: true,
      oauthConfigured: false,
    }
    let progressListener:
      | ((progress: {
          phase: string
          progress: number
          stageLabel: string
          authorizationUrl?: string
          userCode?: string
        }) => void)
      | undefined
    let rejectPreparation: ((error: Error) => void) | undefined
    let cloudApiRoot = ''
    let runtimeMode: 'local' | 'cloud' = 'local'
    let oauthConfigured = false
    let highAccuracyAttempt = 0

    Object.defineProperty(window, 'gtsDesktop', {
      configurable: false,
      value: {
        async invoke(request: {
          method: string
          payload?: {
            apiRoot?: string
            licenseAccepted?: boolean
            mode?: 'local' | 'cloud'
            providerId?: string
            selectionMode?: 'auto' | 'manual'
            variant?: 'small' | 'medium' | 'large'
          }
        }) {
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
            case 'project-revisions':
            case 'local-projects':
            case 'model-packages':
              return []
            case 'demo-project':
              return demoProject
            case 'runtime-status': {
              const providerMode =
                runtimeMode === 'cloud' && cloudApiRoot ? 'cloud' : 'local'
              return {
                mode: 'desktop-local',
                providerMode,
                cloudAvailable: Boolean(cloudApiRoot),
                cloudReady: Boolean(cloudApiRoot),
                ...(cloudApiRoot ? { cloudApiRoot } : {}),
                uploadsAudio: providerMode === 'cloud',
                apiReady: true,
                storageReady: true,
                workerReady: true,
                storageScope: 'application-data',
                estimatedSeconds:
                  providerMode === 'cloud'
                    ? { minimum: 20, maximum: 180 }
                    : { minimum: 60, maximum: 600 },
                resourceNote:
                  providerMode === 'cloud'
                    ? 'Using cloud inference'
                    : 'Using local models',
                transcription: {
                  ...transcription,
                  oauthConfigured,
                  ...(oauthConfigured
                    ? { oauthClientId: 'desktop-oauth-client' }
                    : {}),
                },
              }
            }
            case 'configure-cloud-runtime':
              cloudApiRoot = String(request.payload?.apiRoot ?? '')
              runtimeMode = 'cloud'
              return { configured: true }
            case 'set-runtime-mode': {
              runtimeMode = request.payload?.mode ?? 'local'
              return { mode: runtimeMode }
            }
            case 'configure-huggingface-oauth':
              oauthConfigured = true
              return { configured: true }
            case 'storage-cleanup-status':
              return null
            case 'set-transcription-provider':
              if (request.payload?.providerId === 'basic-pitch') {
                return transcription
              }
              highAccuracyAttempt += 1
              if (highAccuracyAttempt === 2) {
                throw new Error('模型校验失败：SHA-256 不匹配')
              }
              if (highAccuracyAttempt >= 3) {
                transcription.preference = {
                  providerId: 'muscriptor',
                  selectionMode: request.payload?.selectionMode ?? 'auto',
                  ...(request.payload?.variant
                    ? { variant: request.payload.variant }
                    : {}),
                  licenseAcceptedAt: '2026-10-17T00:00:00.000Z',
                  licenseAcceptedVersion: 'CC-BY-NC-4.0',
                }
                transcription.variantWeights.medium = 'managed-cache'
                transcription.weightSource = 'managed-cache'
                transcription.pipeline.forEach((component) => {
                  if (
                    component.id === 'muscriptor' ||
                    component.id === 'mr-mt3' ||
                    component.id === 'noise2fret'
                  ) {
                    component.ready = true
                  }
                })
                return transcription
              }
              queueMicrotask(() =>
                progressListener?.({
                  phase: 'authorizing',
                  progress: 3,
                  stageLabel: '等待 Hugging Face 登录完成',
                  authorizationUrl:
                    'https://huggingface.co/device?user_code=ABCD-EFGH',
                  userCode: 'ABCD-EFGH',
                }),
              )
              return new Promise((_, reject) => {
                rejectPreparation = reject
              })
            case 'open-transcription-authorization':
              Object.assign(window, { transcriptionAuthorizationReopened: true })
              return { opened: true }
            case 'cancel-transcription-setup':
              rejectPreparation?.(
                new Error(
                  '已取消高精度模型准备，可以重新点击“下载并启用”',
                ),
              )
              rejectPreparation = undefined
              return { cancelled: true }
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
        onTranscriptionProgress(listener: typeof progressListener) {
          progressListener = listener
          return () => {
            progressListener = undefined
          }
        },
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '桌面运行时' }).click()
  const dialog = page.getByRole('dialog', { name: '桌面运行时' })
  await expect(dialog.getByLabel('转录方案').locator('option')).toHaveCount(4)
  const localButton = dialog.getByRole('button', { name: '本地' })
  const cloudButton = dialog.getByRole('button', { name: '云端 · 配置' })
  await expect(localButton).toHaveClass(/is-active/u)
  await expect(cloudButton).toBeEnabled()
  await cloudButton.click()
  await expect(cloudButton).toHaveClass(/is-active/u)
  await expect(localButton).not.toHaveClass(/is-active/u)
  await expect(dialog.getByLabel('云端运行配置')).toBeVisible()
  await expect(dialog.getByLabel('转录方案')).toBeHidden()
  await dialog
    .getByLabel('云端 API 地址')
    .fill('https://cloud.example.test')
  await dialog.getByRole('button', { name: '保存并切换' }).click()
  const configuredCloudButton = dialog.getByRole('button', { name: '云端' })
  await expect(configuredCloudButton).toHaveClass(/is-active/u)
  await expect(dialog.getByText('当前项目正在使用')).toBeVisible()

  await localButton.click()
  await expect(localButton).toHaveClass(/is-active/u)
  await expect(configuredCloudButton).not.toHaveClass(/is-active/u)
  await expect(dialog.getByLabel('云端运行配置')).toHaveCount(0)
  await expect(dialog.getByLabel('转录方案')).toBeVisible()

  await dialog.getByLabel('转录方案').selectOption('muscriptor')
  await expect(
    dialog.getByRole('link', { name: '创建 OAuth 应用' }),
  ).toHaveAttribute(
    'href',
    'https://huggingface.co/settings/applications/new',
  )
  await expect(
    dialog.getByText('Public · 不创建 Client Secret'),
  ).toBeVisible()
  await expect(
    dialog.getByText('http://127.0.0.1/oauth/callback'),
  ).toBeVisible()
  await dialog
    .getByLabel('Hugging Face OAuth Client ID')
    .fill('desktop-oauth-client')
  await dialog.getByRole('button', { name: '保存 OAuth 配置' }).click()
  await expect(
    dialog.getByLabel('Hugging Face OAuth Client ID'),
  ).toHaveCount(0)
  await dialog.getByText(/我确认 MuScriptor/u).click()
  await dialog.getByRole('button', { name: '下载并启用' }).click()

  await expect(
    dialog
      .locator('.runtime-model-progress')
      .getByText('等待 Hugging Face 登录完成'),
  ).toBeVisible()
  const reopen = dialog.getByRole('button', { name: '重新打开登录页' })
  const cancel = dialog.getByRole('button', { name: '取消等待' })
  await expect(reopen).toBeEnabled()
  await expect(cancel).toBeEnabled()

  await reopen.click()
  expect(
    await page.evaluate(
      () =>
        (window as Window & {
          transcriptionAuthorizationReopened?: boolean
        }).transcriptionAuthorizationReopened,
    ),
  ).toBe(true)

  await cancel.click()
  await expect(
    dialog.getByText(/已取消高精度模型准备/u),
  ).toBeVisible()
  await expect(
    dialog.getByRole('button', { name: '下载并启用' }),
  ).toBeEnabled()

  await dialog.getByRole('button', { name: '下载并启用' }).click()
  await expect(dialog.getByRole('alert')).toContainText('启用失败')
  await expect(dialog.getByRole('alert')).toContainText(
    '模型校验失败：SHA-256 不匹配',
  )
  await expect(
    dialog.getByRole('button', { name: '下载并启用' }),
  ).toBeEnabled()

  await dialog.getByRole('button', { name: '下载并启用' }).click()
  await expect(dialog.getByText('已启用高精度 · MEDIUM')).toBeVisible()
  await expect(
    dialog.getByRole('button', { name: '高精度已启用' }),
  ).toBeDisabled()

  await dialog.getByLabel('模型档位').selectOption('medium')
  await expect(dialog.getByText('已启用高精度 · MEDIUM')).toBeVisible()
  await expect(
    dialog.getByRole('button', { name: '高精度已启用' }),
  ).toBeDisabled()

  await dialog.getByLabel('模型档位').selectOption('large')
  await expect(
    dialog.getByRole('button', { name: '下载并启用' }),
  ).toBeEnabled()

  await dialog.getByLabel('转录方案').selectOption('guitar-specialist')
  await expect(
    dialog.getByRole('button', { name: '启用吉他专用' }),
  ).toBeEnabled()
  await dialog.getByLabel('转录方案').selectOption('multitrack-research')
  await expect(
    dialog.getByRole('button', { name: '下载并启用' }),
  ).toBeEnabled()
})
