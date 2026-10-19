import { createHash, randomUUID } from 'node:crypto'
import {
  chmodSync,
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  statfsSync,
} from 'node:fs'
import { writeFile } from 'node:fs/promises'
import { cpus, totalmem } from 'node:os'
import { basename, dirname, resolve } from 'node:path'
import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { StringDecoder } from 'node:string_decoder'
import {
  filterTranscriptionProviders,
  recommendTranscriptionVariant,
  transcriptionProvider,
  type DesktopTranscriptionStatus,
  type TranscriptionSetupProgress,
  type TranscriptionAccelerator,
  type TranscriptionDeviceProfile,
  type TranscriptionModelVariant,
  type TranscriptionProviderId,
  type TranscriptionProviderPreference,
  type TranscriptionWeightSource,
} from '@gts/core'
import type { DesktopStateStore } from './state.js'

interface WorkerHardwareProbe {
  accelerator?: string
  acceleratorName?: string
  acceleratorMemoryBytes?: number
  providers?: Record<string, boolean>
}

interface ImportedWeightManifest {
  schemaVersion: 1
  providerId: 'muscriptor'
  variant: TranscriptionModelVariant
  filename: string
  sizeBytes: number
  sha256: string
  importedAt: string
}

const modelConfigs: Record<
  TranscriptionModelVariant,
  { dim: number; num_heads: number; num_layers: number; card: number }
> = {
  small: { dim: 768, num_heads: 12, num_layers: 14, card: 1393 },
  medium: { dim: 1024, num_heads: 16, num_layers: 24, card: 1395 },
  large: { dim: 1536, num_heads: 24, num_layers: 48, card: 1395 },
}

const providerIds = new Set<TranscriptionProviderId>([
  'basic-pitch',
  'muscriptor',
  'guitar-specialist',
  'multitrack-research',
])
const variants = new Set<TranscriptionModelVariant>([
  'small',
  'medium',
  'large',
])
const noise2FretArchiveBytes = 111_317_004
const noise2FretModelBytes = 60_039_447
const noise2FretModelSha256 =
  '1f454f75e9e6f44d80286e2c13188264b10d09c08fa8bdd791b1c809217bab95'
const mrMt3ModelBytes = 183_672_643
const mrMt3ModelSha256 =
  'b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f'
const yourMt3ModelBytes = 561_544_628
const yourMt3ModelSha256 =
  'ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25'

function shouldUseMrMt3(
  variant: TranscriptionModelVariant,
  accelerator: TranscriptionAccelerator,
): boolean {
  return variant !== 'small' && accelerator === 'cuda'
}

function distributionProviders(): TranscriptionProviderId[] {
  const configured = process.env.GTS_TRANSCRIPTION_PROVIDER_ALLOWLIST
  if (configured) {
    const selected = configured
      .split(',')
      .map((value) => value.trim())
      .filter((value): value is TranscriptionProviderId =>
        providerIds.has(value as TranscriptionProviderId),
      )
    return selected.includes('basic-pitch')
      ? selected
      : ['basic-pitch', ...selected]
  }
  return process.env.GTS_DISTRIBUTION_CHANNEL === 'commercial' ||
    process.env.GTS_COMMERCIAL_BUILD === '1'
    ? ['basic-pitch']
    : process.env.GTS_ENABLE_RESEARCH_MODELS === '1'
      ? [
        'basic-pitch',
        'muscriptor',
        'guitar-specialist',
        'multitrack-research',
      ]
      : ['basic-pitch']
}

function parseAccelerator(value: unknown): TranscriptionAccelerator {
  return value === 'mps' || value === 'cuda' || value === 'cpu'
    ? value
    : 'unknown'
}

function safeProbe(workerCommand?: string): WorkerHardwareProbe {
  if (!workerCommand || !existsSync(workerCommand)) return {}
  const result = spawnSync(workerCommand, ['--hardware-probe'], {
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
    },
    timeout: 30_000,
    maxBuffer: 1024 * 1024,
  })
  if (result.status !== 0) return {}
  try {
    return JSON.parse(result.stdout.trim()) as WorkerHardwareProbe
  } catch {
    return {}
  }
}

export function detectTranscriptionDevice(
  workerCommand?: string,
): {
  device: TranscriptionDeviceProfile
  providerAvailability: Record<TranscriptionProviderId, boolean>
  mrMt3Available: boolean
} {
  const probe = safeProbe(workerCommand)
  const fallbackAccelerator: TranscriptionAccelerator =
    process.platform === 'darwin' && process.arch === 'arm64'
      ? 'mps'
      : 'cpu'
  return {
    device: {
      platform: process.platform as 'darwin' | 'win32' | 'linux',
      arch: process.arch as 'arm64' | 'x64',
      totalMemoryBytes: totalmem(),
      logicalCpuCount: cpus().length,
      accelerator:
        parseAccelerator(probe.accelerator) === 'unknown'
          ? fallbackAccelerator
          : parseAccelerator(probe.accelerator),
      ...(typeof probe.acceleratorName === 'string'
        ? { acceleratorName: probe.acceleratorName }
        : {}),
      ...(typeof probe.acceleratorMemoryBytes === 'number'
        ? { acceleratorMemoryBytes: probe.acceleratorMemoryBytes }
        : {}),
    },
    providerAvailability: {
      'basic-pitch': true,
      muscriptor: probe.providers?.muscriptor === true,
      'guitar-specialist':
        probe.providers?.['guitar-specialist'] === true,
      'multitrack-research':
        probe.providers?.['multitrack-research'] === true,
    },
    mrMt3Available: probe.providers?.['mr-mt3'] === true,
  }
}

export class DesktopTranscriptionManager {
  private workerProbe:
    | ReturnType<typeof detectTranscriptionDevice>
    | undefined
  private probedWorkerCommand?: string
  private activePreparation?: {
    child: ChildProcess
    cancelled: boolean
  }
  private activeAuthorization?: AbortController
  private readonly allowedProviderIds = distributionProviders()

  constructor(
    private readonly root: string,
    private readonly cacheRoot: string,
    private readonly state: DesktopStateStore,
    private readonly workerCommand: () => string | undefined,
    private readonly onProgress: (
      progress: TranscriptionSetupProgress,
    ) => void = () => undefined,
    private readonly guitarModelPath: () => string | undefined = () =>
      undefined,
    private readonly authorizeHuggingFace?: (
      signal: AbortSignal,
    ) => Promise<string>,
    private readonly oauthClientId: () => string | undefined = () =>
      undefined,
    private readonly networkEnvironment: () => Promise<NodeJS.ProcessEnv> =
      async () => ({}),
  ) {
    mkdirSync(root, { recursive: true })
    mkdirSync(cacheRoot, { recursive: true })
  }

  status(): DesktopTranscriptionStatus {
    const probe = this.probe()
    const hardwareRecommendation = recommendTranscriptionVariant(
      probe.device,
    )
    const recommendation =
      shouldUseMrMt3(
        hardwareRecommendation.variant,
        probe.device.accelerator,
      ) &&
      !probe.mrMt3Available
        ? {
            variant: 'small' as const,
            reason:
              '当前 Worker 未包含 MR-MT3 复核组件，自动使用完整可用的 Small 组合',
          }
        : hardwareRecommendation
    const preference = this.effectivePreference()
    const resolvedVariant =
      preference.providerId === 'muscriptor'
        ? preference.selectionMode === 'manual' && preference.variant
          ? preference.variant
          : recommendation.variant
        : undefined
    const pipelineVariant =
      resolvedVariant ?? recommendation.variant
    const verificationSelected =
      shouldUseMrMt3(
        pipelineVariant,
        probe.device.accelerator,
      ) && probe.mrMt3Available
    const weightSource =
      preference.providerId === 'guitar-specialist'
        ? this.guitarModelPath()
          ? 'managed-cache'
          : 'download-required'
        : preference.providerId === 'multitrack-research'
          ? this.yourMt3WeightPath()
            ? 'managed-cache'
            : 'download-required'
          : this.weightSource(preference.providerId, resolvedVariant)
    const runtimeAvailable =
      probe.providerAvailability[preference.providerId] &&
      (preference.providerId !== 'guitar-specialist' ||
        Boolean(this.guitarModelPath()))
    const positionModelReady = Boolean(this.noise2FretWeightPath())
    const verificationModelReady = Boolean(this.mrMt3WeightPath())
    const oauthClientId = this.oauthClientId()
    return {
      providers: filterTranscriptionProviders(
        this.allowedProviderIds,
        process.env.GTS_DISTRIBUTION_CHANNEL === 'commercial' ||
          process.env.GTS_COMMERCIAL_BUILD === '1',
      ),
      preference,
      ...(resolvedVariant ? { resolvedVariant } : {}),
      recommendation,
      device: probe.device,
      variantWeights: {
        small: this.weightSource('muscriptor', 'small'),
        medium: this.weightSource('muscriptor', 'medium'),
        large: this.weightSource('muscriptor', 'large'),
      },
      pipeline: [
        {
          id: 'muscriptor',
          label: 'MuScriptor',
          role: 'primary-events',
          selected: preference.providerId === 'muscriptor',
          runtimeAvailable: probe.providerAvailability.muscriptor,
          ...(pipelineVariant
            ? {
                weightSizeBytes: transcriptionProvider(
                  'muscriptor',
                ).variants.find((item) => item.id === pipelineVariant)
                  ?.weightSizeBytes,
              }
            : {}),
          ready:
            probe.providerAvailability.muscriptor &&
            this.weightSource('muscriptor', pipelineVariant) !==
              'download-required',
        },
        {
          id: 'basic-pitch',
          label: 'Basic Pitch',
          role: 'event-verification',
          selected:
            preference.providerId === 'basic-pitch' ||
            preference.providerId === 'muscriptor' ||
            preference.providerId === 'multitrack-research',
          runtimeAvailable: true,
          ready: true,
        },
        {
          id: 'mr-mt3',
          label: 'MR-MT3',
          role: 'event-verification',
          selected:
            preference.providerId === 'muscriptor' &&
            verificationSelected,
          runtimeAvailable: probe.mrMt3Available,
          ready: probe.mrMt3Available && verificationModelReady,
          weightSizeBytes: mrMt3ModelBytes,
        },
        {
          id: 'noise2fret',
          label: 'Noise2Fret',
          role: 'tablature-position',
          selected:
            preference.providerId === 'muscriptor' ||
            preference.providerId === 'multitrack-research',
          runtimeAvailable: true,
          ready: positionModelReady,
          weightSizeBytes: noise2FretArchiveBytes,
        },
        {
          id: 'stringtrace',
          label: 'StringTrace',
          role: 'tablature-position',
          selected: preference.providerId === 'guitar-specialist',
          runtimeAvailable:
            probe.providerAvailability['guitar-specialist'],
          ready: Boolean(this.guitarModelPath()),
        },
        {
          id: 'playability-decoder',
          label: '可演奏性指法解码',
          role: 'tablature-position',
          selected: preference.providerId === 'guitar-specialist',
          runtimeAvailable: true,
          ready: true,
        },
        {
          id: 'yourmt3',
          label: 'YourMT3+',
          role: 'primary-events',
          selected: preference.providerId === 'multitrack-research',
          runtimeAvailable:
            probe.providerAvailability['multitrack-research'],
          ready: Boolean(this.yourMt3WeightPath()),
          weightSizeBytes: yourMt3ModelBytes,
        },
      ],
      runtimeAvailable,
      weightSource,
      ready:
        preference.providerId === 'basic-pitch' ||
        (preference.providerId === 'muscriptor'
          ? runtimeAvailable &&
            Boolean(preference.licenseAcceptedAt) &&
            preference.licenseAcceptedVersion === 'CC-BY-NC-4.0' &&
            weightSource !== 'download-required' &&
            positionModelReady &&
            (!verificationSelected ||
              (probe.mrMt3Available && verificationModelReady))
          : runtimeAvailable &&
            weightSource !== 'download-required' &&
            (preference.providerId !== 'multitrack-research' ||
              positionModelReady)),
      oauthConfigured: Boolean(oauthClientId),
      ...(oauthClientId ? { oauthClientId } : {}),
    }
  }

  async prepareAndSetPreference(input: {
    providerId: TranscriptionProviderId
    selectionMode?: 'auto' | 'manual'
    variant?: TranscriptionModelVariant
    licenseAccepted?: boolean
  }): Promise<TranscriptionProviderPreference> {
    if (input.providerId === 'basic-pitch') {
      return this.setPreference(input)
    }
    if (!this.allowedProviderIds.includes(input.providerId)) {
      throw new Error('当前发行版本不允许使用该转录模型')
    }
    if (input.providerId === 'guitar-specialist') {
      if (!this.guitarModelPath()) {
        throw new Error('吉他专用方案需要先安装包含转录模型的研究模型包')
      }
      return this.setPreference({
        providerId: input.providerId,
        selectionMode: 'auto',
      })
    }
    if (input.providerId === 'multitrack-research') {
      if (!this.probe().providerAvailability[input.providerId]) {
        throw new Error('当前 Worker 未包含 YourMT3+ 运行组件')
      }
      if (!this.yourMt3WeightPath() || !this.noise2FretWeightPath()) {
        await this.prepareResearchModels()
      }
      return this.setPreference({
        providerId: input.providerId,
        selectionMode: 'auto',
      })
    }
    const previous = this.state.transcriptionPreference()
    const licenseAccepted =
      input.licenseAccepted === true ||
      (previous.providerId === 'muscriptor' &&
        previous.licenseAcceptedAt !== undefined &&
        previous.licenseAcceptedVersion === 'CC-BY-NC-4.0')
    if (!licenseAccepted) {
      throw new Error('启用 MuScriptor 前必须确认非商用权重许可')
    }
    if (input.selectionMode === 'manual' && !input.variant) {
      throw new Error('手动模式必须选择模型档位')
    }
    if (input.variant && !variants.has(input.variant)) {
      throw new Error('未知的模型档位')
    }
    const status = this.status()
    if (!status.providers.some((provider) => provider.id === input.providerId)) {
      throw new Error('当前发行版本不允许使用该转录模型')
    }
    if (!this.probe().providerAvailability[input.providerId]) {
      throw new Error('当前 Worker 未包含 MuScriptor 运行组件')
    }
    const variant =
      input.selectionMode === 'manual' && input.variant
        ? input.variant
        : status.recommendation.variant
    if (!variants.has(variant)) throw new Error('未知的模型档位')
    const probe = this.probe()
    const verificationSelected =
      shouldUseMrMt3(
        variant,
        probe.device.accelerator,
      ) && probe.mrMt3Available
    if (
      this.weightSource(input.providerId, variant) === 'download-required' ||
      !this.noise2FretWeightPath() ||
      (verificationSelected && !this.mrMt3WeightPath())
    ) {
      await this.prepareHighAccuracyModels(variant)
    }
    return this.setPreference(input)
  }

  cancelPreparation(): boolean {
    const authorization = this.activeAuthorization
    const active = this.activePreparation
    if (!authorization && !active) return false
    authorization?.abort()
    if (active) {
      active.cancelled = true
    }
    if (active?.child.exitCode === null) {
      active.child.kill('SIGTERM')
      const forceStop = setTimeout(() => {
        if (active.child.exitCode === null) active.child.kill('SIGKILL')
      }, 1500)
      forceStop.unref()
    }
    return true
  }

  setPreference(input: {
    providerId: TranscriptionProviderId
    selectionMode?: 'auto' | 'manual'
    variant?: TranscriptionModelVariant
    licenseAccepted?: boolean
  }): TranscriptionProviderPreference {
    if (!this.allowedProviderIds.includes(input.providerId)) {
      throw new Error('当前发行版本不允许使用该转录模型')
    }
    if (input.variant && !variants.has(input.variant)) {
      throw new Error('未知的模型档位')
    }
    if (input.providerId === 'basic-pitch') {
      const preference: TranscriptionProviderPreference = {
        providerId: 'basic-pitch',
        selectionMode: 'auto',
      }
      this.state.setTranscriptionPreference(preference)
      return preference
    }
    if (
      input.providerId === 'guitar-specialist' ||
      input.providerId === 'multitrack-research'
    ) {
      const preference: TranscriptionProviderPreference = {
        providerId: input.providerId,
        selectionMode: 'auto',
      }
      this.state.setTranscriptionPreference(preference)
      return preference
    }
    const previous = this.state.transcriptionPreference()
    const acceptedAt =
      previous.providerId === 'muscriptor' &&
      previous.licenseAcceptedVersion === 'CC-BY-NC-4.0'
        ? previous.licenseAcceptedAt
        : undefined
    if (!acceptedAt && input.licenseAccepted !== true) {
      throw new Error('启用 MuScriptor 前必须确认非商用权重许可')
    }
    const selectionMode = input.selectionMode ?? 'auto'
    if (selectionMode === 'manual' && !input.variant) {
      throw new Error('手动模式必须选择模型档位')
    }
    const preference: TranscriptionProviderPreference = {
      providerId: 'muscriptor',
      selectionMode,
      ...(selectionMode === 'manual' && input.variant
        ? { variant: input.variant }
        : {}),
      licenseAcceptedAt: acceptedAt ?? new Date().toISOString(),
      licenseAcceptedVersion: 'CC-BY-NC-4.0',
    }
    this.state.setTranscriptionPreference(preference)
    return preference
  }

  async importWeights(
    source: string,
    variant: TranscriptionModelVariant,
    licenseAccepted = false,
  ): Promise<ImportedWeightManifest> {
    if (!this.allowedProviderIds.includes('muscriptor')) {
      throw new Error('当前发行版本不允许导入 MuScriptor 权重')
    }
    const existingPreference = this.state.transcriptionPreference()
    if (
      !licenseAccepted &&
      !(
        existingPreference.providerId === 'muscriptor' &&
        existingPreference.licenseAcceptedAt &&
        existingPreference.licenseAcceptedVersion === 'CC-BY-NC-4.0'
      )
    ) {
      throw new Error('导入 MuScriptor 权重前必须确认非商用权重许可')
    }
    if (!variants.has(variant)) throw new Error('未知的模型档位')
    const sourceStats = statSync(source)
    if (!sourceStats.isFile() || !source.toLowerCase().endsWith('.safetensors')) {
      throw new Error('请选择 MuScriptor 的 .safetensors 权重文件')
    }
    const fileSystem = statfsSync(this.root)
    const availableBytes = fileSystem.bavail * fileSystem.bsize
    if (availableBytes < sourceStats.size + 512 * 1024 * 1024) {
      throw new Error('磁盘空间不足，无法导入所选模型权重')
    }

    const destination = resolve(this.root, 'muscriptor', variant)
    const staging = resolve(
      this.root,
      `.import-muscriptor-${variant}-${randomUUID()}`,
    )
    const stagedWeights = resolve(staging, 'model.safetensors')
    mkdirSync(staging, { recursive: true })
    const hash = createHash('sha256')
    try {
      await pipeline(
        createReadStream(source),
        new Transform({
          transform(chunk, _encoding, callback) {
            hash.update(chunk)
            callback(null, chunk)
          },
        }),
        createWriteStream(stagedWeights, { mode: 0o600 }),
      )
      const manifest: ImportedWeightManifest = {
        schemaVersion: 1,
        providerId: 'muscriptor',
        variant,
        filename: basename(source),
        sizeBytes: sourceStats.size,
        sha256: hash.digest('hex'),
        importedAt: new Date().toISOString(),
      }
      await Promise.all([
        writeFile(
          resolve(staging, 'config.json'),
          `${JSON.stringify(modelConfigs[variant], null, 2)}\n`,
          { encoding: 'utf8', mode: 0o600 },
        ),
        writeFile(
          resolve(staging, 'import.json'),
          `${JSON.stringify(manifest, null, 2)}\n`,
          { encoding: 'utf8', mode: 0o600 },
        ),
      ])
      const previous = `${destination}.previous`
      rmSync(previous, { recursive: true, force: true })
      mkdirSync(resolve(this.root, 'muscriptor'), { recursive: true })
      if (existsSync(destination)) renameSync(destination, previous)
      try {
        renameSync(staging, destination)
        rmSync(previous, { recursive: true, force: true })
      } catch (error) {
        if (existsSync(previous) && !existsSync(destination)) {
          renameSync(previous, destination)
        }
        throw error
      }
      chmodSync(resolve(destination, 'model.safetensors'), 0o600)
      return manifest
    } catch (error) {
      rmSync(staging, { recursive: true, force: true })
      throw error
    }
  }

  workerConfiguration(): {
    provider: TranscriptionProviderId
    model?: string
    variant?: TranscriptionModelVariant
    tabModel?: string
    tabModelMode?: 'onset-position-research'
    positionModel?: string
    verificationModel?: string
    cacheRoot: string
  } {
    const status = this.status()
    if (
      status.preference.providerId === 'basic-pitch'
    ) {
      return {
        provider: 'basic-pitch',
        cacheRoot: this.cacheRoot,
      }
    }
    if (status.preference.providerId === 'guitar-specialist') {
      return {
        provider: 'guitar-specialist',
        tabModel: this.guitarModelPath(),
        tabModelMode: 'onset-position-research',
        cacheRoot: this.cacheRoot,
      }
    }
    if (status.preference.providerId === 'multitrack-research') {
      return {
        provider: 'multitrack-research',
        model: this.yourMt3WeightPath(),
        positionModel: this.noise2FretWeightPath(),
        cacheRoot: this.cacheRoot,
      }
    }
    if (!status.resolvedVariant) {
      return {
        provider: 'basic-pitch',
        cacheRoot: this.cacheRoot,
      }
    }
    const verificationSelected =
      status.pipeline.find((component) => component.id === 'mr-mt3')
        ?.selected === true
    return {
      provider: 'muscriptor',
      model:
        this.managedWeightPath(status.resolvedVariant) ??
        status.resolvedVariant,
      variant: status.resolvedVariant,
      positionModel: this.noise2FretWeightPath(),
      ...(verificationSelected
        ? { verificationModel: this.mrMt3WeightPath() }
        : {}),
      cacheRoot: this.cacheRoot,
    }
  }

  private effectivePreference(): TranscriptionProviderPreference {
    const preference = this.state.transcriptionPreference()
    return this.allowedProviderIds.includes(preference.providerId)
      ? preference
      : { providerId: 'basic-pitch', selectionMode: 'auto' }
  }

  private probe(): ReturnType<typeof detectTranscriptionDevice> {
    const command = this.workerCommand()
    if (!this.workerProbe || command !== this.probedWorkerCommand) {
      this.probedWorkerCommand = command
      this.workerProbe = detectTranscriptionDevice(command)
    }
    return this.workerProbe
  }

  private weightSource(
    providerId: TranscriptionProviderId,
    variant?: TranscriptionModelVariant,
  ): TranscriptionWeightSource {
    if (providerId === 'basic-pitch') return 'built-in'
    if (!variant) return 'download-required'
    if (this.managedWeightPath(variant)) return 'local-import'
    return this.cachedWeightExists(variant)
      ? 'managed-cache'
      : 'download-required'
  }

  private managedWeightPath(
    variant: TranscriptionModelVariant,
  ): string | undefined {
    const path = resolve(
      this.root,
      'muscriptor',
      variant,
      'model.safetensors',
    )
    return existsSync(path) && statSync(path).isFile() ? path : undefined
  }

  private cachedWeightExists(variant: TranscriptionModelVariant): boolean {
    const snapshots = resolve(
      this.cacheRoot,
      `models--MuScriptor--muscriptor-${variant}`,
      'snapshots',
    )
    if (!existsSync(snapshots)) return false
    return readdirSync(snapshots).some((snapshot) =>
      existsSync(resolve(snapshots, snapshot, 'model.safetensors')),
    )
  }

  private async prepareHighAccuracyModels(
    variant: TranscriptionModelVariant,
  ): Promise<void> {
    if (this.activePreparation || this.activeAuthorization) {
      throw new Error('研究模型正在准备，请等待完成或先取消当前任务')
    }
    const command = this.workerCommand()
    if (!command || !existsSync(command)) {
      throw new Error('本地推理 Worker 不可用，无法准备研究模型')
    }
    const descriptor = transcriptionProvider('muscriptor').variants.find(
      (item) => item.id === variant,
    )
    if (!descriptor) throw new Error('未知的 MuScriptor 模型档位')
    this.onProgress({
      phase: 'checking',
      progress: 1,
      stageLabel: '正在检查研究模型和登录状态',
    })
    const disk = statfsSync(this.cacheRoot)
    const availableBytes = disk.bavail * disk.bsize
    const probe = this.probe()
    const verificationSelected =
      shouldUseMrMt3(
        variant,
        probe.device.accelerator,
      ) && probe.mrMt3Available
    const requiredDownloadBytes =
      (this.weightSource('muscriptor', variant) === 'download-required'
        ? descriptor.weightSizeBytes
        : 0) +
      (this.noise2FretWeightPath() ? 0 : noise2FretArchiveBytes) +
      (!verificationSelected || this.mrMt3WeightPath()
        ? 0
        : mrMt3ModelBytes)
    if (
      availableBytes <
      requiredDownloadBytes * 1.25 +
        512 * 1024 * 1024
    ) {
      throw new Error('磁盘空间不足，无法下载所选研究模型')
    }

    const authorization = new AbortController()
    this.activeAuthorization = authorization
    let accessToken: string | undefined
    try {
      if (
        this.weightSource('muscriptor', variant) === 'download-required'
      ) {
        if (this.authorizeHuggingFace) {
          accessToken = await this.authorizeHuggingFace(
            authorization.signal,
          )
        }
      }
    } catch (error) {
      if (authorization.signal.aborted) {
        throw new Error(
          '已取消研究模型准备，可以重新点击“下载并启用”',
        )
      }
      throw error
    } finally {
      if (this.activeAuthorization === authorization) {
        this.activeAuthorization = undefined
      }
    }

    await this.runPreparationWorker(
      [
        '--prepare-transcription-provider',
        '--transcription-provider',
        'muscriptor',
        '--transcription-model',
        this.managedWeightPath(variant) ?? variant,
        '--transcription-variant',
        variant,
      ],
      accessToken,
      variant,
      '研究模型',
    )
  }

  private async prepareResearchModels(): Promise<void> {
    if (this.activePreparation || this.activeAuthorization) {
      throw new Error('研究模型正在准备，请等待完成或先取消当前任务')
    }
    this.onProgress({
      phase: 'checking',
      progress: 1,
      stageLabel: '正在检查 YourMT3+ 与吉他弦品模型',
    })
    const disk = statfsSync(this.cacheRoot)
    const requiredDownloadBytes =
      (this.yourMt3WeightPath() ? 0 : yourMt3ModelBytes) +
      (this.noise2FretWeightPath() ? 0 : noise2FretArchiveBytes)
    if (
      disk.bavail * disk.bsize <
      requiredDownloadBytes * 1.25 + 512 * 1024 * 1024
    ) {
      throw new Error('磁盘空间不足，无法下载多轨研究模型')
    }
    await this.runPreparationWorker(
      [
        '--prepare-transcription-provider',
        '--transcription-provider',
        'multitrack-research',
      ],
      undefined,
      undefined,
      '多轨研究模型',
    )
  }

  private async runPreparationWorker(
    argumentsList: string[],
    accessToken?: string,
    muscriptorVariant?: TranscriptionModelVariant,
    operationLabel = '模型',
  ): Promise<void> {
    const command = this.workerCommand()
    if (!command || !existsSync(command)) {
      throw new Error('本地推理 Worker 不可用，无法准备模型')
    }
    const networkEnvironment = await this.networkEnvironment()
    await new Promise<void>((resolvePromise, rejectPromise) => {
      const child = spawn(
        command,
        argumentsList,
        {
          env: {
            ...process.env,
            ...networkEnvironment,
            PYTHONUTF8: '1',
            PYTHONIOENCODING: 'utf-8',
            HF_HOME: dirname(this.cacheRoot),
            HF_HUB_CACHE: this.cacheRoot,
            GTS_MODEL_PROVIDER_ROOT: this.root,
            ...(accessToken ? { HF_TOKEN: accessToken } : {}),
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      )
      const activePreparation = {
        child,
        cancelled: false,
      }
      this.activePreparation = activePreparation
      let stderr = ''
      let stdout = ''
      const stdoutDecoder = new StringDecoder('utf8')
      const stderrDecoder = new StringDecoder('utf8')
      const handleLine = (line: string) => {
        if (!line.trim()) return
        try {
          const event = JSON.parse(line) as Record<string, unknown>
          const progress =
            typeof event.progress === 'number' ? event.progress : 0
          const stageLabel =
            typeof event.stageLabel === 'string'
              ? event.stageLabel
              : '正在准备研究模型'
          if (
            event.type === 'gts-auth-required' ||
            event.type === 'gts-license-required'
          ) {
            this.onProgress({
              phase: 'authorizing',
              progress,
              stageLabel,
              ...(typeof event.authorizationUrl === 'string'
                ? { authorizationUrl: event.authorizationUrl }
                : {}),
              ...(typeof event.userCode === 'string'
                ? { userCode: event.userCode }
                : {}),
            })
          } else if (event.type === 'gts-model-progress') {
            this.onProgress({
              phase: /验证/u.test(stageLabel)
                ? 'verifying'
                : 'downloading',
              progress,
              stageLabel,
            })
          } else if (event.type === 'gts-model-ready') {
            this.onProgress({
              phase: 'ready',
              progress: 100,
              stageLabel,
            })
          }
        } catch {
          // Non-JSON worker output is diagnostic only.
        }
      }
      child.stdout.on('data', (chunk: Buffer) => {
        stdout += stdoutDecoder.write(chunk)
        const lines = stdout.split(/\r?\n/u)
        stdout = lines.pop() ?? ''
        for (const line of lines) handleLine(line)
      })
      const timeout = setTimeout(() => {
        child.kill('SIGTERM')
        rejectPromise(
          new Error(`${operationLabel}下载超时，请检查网络后重试`),
        )
      }, 2 * 60 * 60 * 1000)
      child.stderr.on('data', (chunk: Buffer) => {
        stderr = `${stderr}${stderrDecoder.write(chunk)}`.slice(-8000)
      })
      child.on('error', (error) => {
        clearTimeout(timeout)
        if (this.activePreparation === activePreparation) {
          this.activePreparation = undefined
        }
        rejectPromise(error)
      })
      child.on('close', (code) => {
        clearTimeout(timeout)
        if (this.activePreparation === activePreparation) {
          this.activePreparation = undefined
        }
        stdout += stdoutDecoder.end()
        stderr = `${stderr}${stderrDecoder.end()}`.slice(-8000)
        handleLine(stdout)
        if (activePreparation.cancelled) {
          rejectPromise(
            new Error(
              `已取消${operationLabel}准备，可以重新点击“下载并启用”`,
            ),
          )
          return
        }
        if (
          code === 0 &&
          (muscriptorVariant
            ? this.weightSource('muscriptor', muscriptorVariant) !==
                'download-required' &&
              this.noise2FretWeightPath() &&
              (!shouldUseMrMt3(
                muscriptorVariant,
                this.probe().device.accelerator,
              ) ||
                !this.probe().mrMt3Available ||
                this.mrMt3WeightPath())
            : Boolean(
                this.yourMt3WeightPath() &&
                  this.noise2FretWeightPath(),
              ))
        ) {
          resolvePromise()
          return
        }
        if (/授权未完成|401|403|gated/iu.test(stderr)) {
          rejectPromise(
            new Error(
              'Hugging Face 授权未完成。请先接受所选权重许可并完成本机登录，或使用“导入权重”。',
            ),
          )
          return
        }
        if (/登录已取消|access.denied/iu.test(stderr)) {
          rejectPromise(
            new Error('Hugging Face 登录已取消，可以重新点击“下载并启用”'),
          )
          return
        }
        if (/登录等待超时|授权确认超时|device code expired/iu.test(stderr)) {
          rejectPromise(
            new Error('Hugging Face 授权等待超时，可以重新点击“下载并启用”'),
          )
          return
        }
        rejectPromise(
          new Error(
            /space|磁盘/iu.test(stderr)
              ? `磁盘空间不足，无法完成${operationLabel}下载`
              : `${operationLabel}下载或校验失败，请检查网络后重试`,
          ),
        )
      })
    })
  }

  private yourMt3WeightPath(): string | undefined {
    const path = resolve(
      this.root,
      'yourmt3',
      '1.0.0',
      'last.ckpt',
    )
    const manifestPath = resolve(dirname(path), 'manifest.json')
    if (
      !existsSync(path) ||
      !statSync(path).isFile() ||
      statSync(path).size !== yourMt3ModelBytes ||
      !existsSync(manifestPath)
    ) {
      return undefined
    }
    try {
      const manifest = JSON.parse(
        readFileSync(manifestPath, 'utf8'),
      ) as Record<string, unknown>
      return manifest.sha256 === yourMt3ModelSha256 &&
        manifest.sizeBytes === yourMt3ModelBytes
        ? path
        : undefined
    } catch {
      return undefined
    }
  }

  private noise2FretWeightPath(): string | undefined {
    const root = resolve(
      this.root,
      'noise2fret',
      '1.0.0',
    )
    const path = resolve(root, 'model.pth')
    const manifestPath = resolve(root, 'manifest.json')
    if (
      !existsSync(path) ||
      !statSync(path).isFile() ||
      statSync(path).size !== noise2FretModelBytes ||
      !existsSync(manifestPath)
    ) {
      return undefined
    }
    try {
      const manifest = JSON.parse(
        readFileSync(manifestPath, 'utf8'),
      ) as Record<string, unknown>
      return manifest.sha256 === noise2FretModelSha256 &&
        manifest.sizeBytes === noise2FretModelBytes
        ? path
        : undefined
    } catch {
      return undefined
    }
  }

  private mrMt3WeightPath(): string | undefined {
    const root = resolve(this.root, 'mr-mt3', '1.0.0')
    const path = resolve(root, 'model.pth')
    const manifestPath = resolve(root, 'manifest.json')
    if (
      !existsSync(path) ||
      !statSync(path).isFile() ||
      statSync(path).size !== mrMt3ModelBytes ||
      !existsSync(manifestPath)
    ) {
      return undefined
    }
    try {
      const manifest = JSON.parse(
        readFileSync(manifestPath, 'utf8'),
      ) as Record<string, unknown>
      return manifest.sha256 === mrMt3ModelSha256 &&
        manifest.sizeBytes === mrMt3ModelBytes
        ? path
        : undefined
    } catch {
      return undefined
    }
  }
}
