export type TranscriptionProviderId =
  | 'basic-pitch'
  | 'muscriptor'
  | 'guitar-specialist'
  | 'multitrack-research'

export type TranscriptionModelVariant = 'small' | 'medium' | 'large'

export type TranscriptionSelectionMode = 'auto' | 'manual'

export type TranscriptionLicenseClass =
  | 'commercial-compatible'
  | 'noncommercial'

export type TranscriptionAccelerator = 'cpu' | 'mps' | 'cuda' | 'unknown'

export interface TranscriptionVariantDescriptor {
  id: TranscriptionModelVariant
  label: string
  parameterCount: string
  weightSizeBytes: number
  minimumMemoryBytes: number
}

export interface TranscriptionProviderDescriptor {
  id: TranscriptionProviderId
  label: string
  description: string
  licenseClass: TranscriptionLicenseClass
  licenseName: string
  licenseUrl?: string
  modelPageUrl?: string
  bundled: boolean
  variants: TranscriptionVariantDescriptor[]
}

export interface TranscriptionDeviceProfile {
  platform: 'darwin' | 'win32' | 'linux'
  arch: 'arm64' | 'x64'
  totalMemoryBytes: number
  logicalCpuCount: number
  accelerator: TranscriptionAccelerator
  acceleratorName?: string
  acceleratorMemoryBytes?: number
}

export interface TranscriptionRecommendation {
  variant: TranscriptionModelVariant
  reason: string
}

export interface TranscriptionProviderPreference {
  providerId: TranscriptionProviderId
  selectionMode: TranscriptionSelectionMode
  variant?: TranscriptionModelVariant
  licenseAcceptedAt?: string
  licenseAcceptedVersion?: 'CC-BY-NC-4.0'
}

export type TranscriptionWeightSource =
  | 'built-in'
  | 'managed-cache'
  | 'local-import'
  | 'download-required'

export interface TranscriptionPipelineComponent {
  id:
    | 'muscriptor'
    | 'mr-mt3'
    | 'basic-pitch'
    | 'noise2fret'
    | 'stringtrace'
    | 'playability-decoder'
    | 'yourmt3'
  label: string
  role: 'primary-events' | 'event-verification' | 'tablature-position'
  selected: boolean
  runtimeAvailable: boolean
  ready: boolean
  weightSizeBytes?: number
}

export interface DesktopTranscriptionStatus {
  providers: TranscriptionProviderDescriptor[]
  preference: TranscriptionProviderPreference
  resolvedVariant?: TranscriptionModelVariant
  recommendation: TranscriptionRecommendation
  device: TranscriptionDeviceProfile
  variantWeights: Record<
    TranscriptionModelVariant,
    TranscriptionWeightSource
  >
  pipeline: TranscriptionPipelineComponent[]
  runtimeAvailable: boolean
  weightSource: TranscriptionWeightSource
  ready: boolean
  oauthConfigured: boolean
  oauthClientId?: string
}

const GIB = 1024 ** 3

export const transcriptionProviders: readonly TranscriptionProviderDescriptor[] =
  [
    {
      id: 'basic-pitch',
      label: 'Basic Pitch',
      description: '内置基础转录，启动快且无需额外权重。',
      licenseClass: 'commercial-compatible',
      licenseName: 'Apache-2.0',
      bundled: true,
      variants: [],
    },
    {
      id: 'muscriptor',
      label: 'MuScriptor 研究融合',
      description:
        '通用音符转录研究链路，不代表已通过吉他 TAB 准确率门禁。',
      licenseClass: 'noncommercial',
      licenseName: 'CC BY-NC 4.0 weights',
      licenseUrl: 'https://creativecommons.org/licenses/by-nc/4.0/',
      modelPageUrl: 'https://huggingface.co/MuScriptor',
      bundled: false,
      variants: [
        {
          id: 'small',
          label: 'Small',
          parameterCount: '103M',
          weightSizeBytes: Math.round(0.41 * GIB),
          minimumMemoryBytes: 8 * GIB,
        },
        {
          id: 'medium',
          label: 'Medium',
          parameterCount: '307M',
          weightSizeBytes: Math.round(1.23 * GIB),
          minimumMemoryBytes: 16 * GIB,
        },
        {
          id: 'large',
          label: 'Large',
          parameterCount: '1.4B',
          weightSizeBytes: Math.round(5.47 * GIB),
          minimumMemoryBytes: 32 * GIB,
        },
      ],
    },
    {
      id: 'guitar-specialist',
      label: '吉他专用研究',
      description:
        'StringTrace 起音与弦品联合模型，结合可演奏性约束生成吉他 TAB。',
      licenseClass: 'noncommercial',
      licenseName: 'Research model package',
      bundled: false,
      variants: [],
    },
    {
      id: 'multitrack-research',
      label: '多轨研究',
      description:
        'YourMT3+ 多乐器转录，结合 Basic Pitch 与 Noise2Fret 生成吉他声部。',
      licenseClass: 'noncommercial',
      licenseName: 'Research-only dependency set',
      modelPageUrl: 'https://huggingface.co/spaces/mimbres/YourMT3',
      bundled: false,
      variants: [],
    },
  ] as const

export function transcriptionProvider(
  id: TranscriptionProviderId,
): TranscriptionProviderDescriptor {
  const provider = transcriptionProviders.find((item) => item.id === id)
  if (!provider) throw new Error(`Unknown transcription provider: ${id}`)
  return provider
}

export function filterTranscriptionProviders(
  allowedIds: readonly TranscriptionProviderId[],
  commercialDistribution = false,
): TranscriptionProviderDescriptor[] {
  const allowlist = new Set(allowedIds)
  return transcriptionProviders.filter(
    (provider) =>
      allowlist.has(provider.id) &&
      (!commercialDistribution ||
        provider.licenseClass === 'commercial-compatible'),
  )
}

export function recommendTranscriptionVariant(
  device: TranscriptionDeviceProfile,
): TranscriptionRecommendation {
  const memoryGiB = device.totalMemoryBytes / GIB
  const acceleratorMemoryGiB =
    (device.acceleratorMemoryBytes ?? 0) / GIB

  if (
    memoryGiB >= 32 &&
    ((device.accelerator === 'mps' && device.arch === 'arm64') ||
      (device.accelerator === 'cuda' && acceleratorMemoryGiB >= 12))
  ) {
    return {
      variant: 'large',
      reason:
        device.accelerator === 'mps'
          ? 'Apple 芯片与至少 32 GB 统一内存可运行 Large'
          : 'CUDA 显存至少 12 GB 且系统内存充足，可运行 Large',
    }
  }

  if (
    (device.accelerator === 'mps' && memoryGiB >= 24) ||
    (device.accelerator === 'cuda' &&
      memoryGiB >= 16 &&
      acceleratorMemoryGiB >= 6)
  ) {
    return {
      variant: 'medium',
      reason:
        device.accelerator === 'mps'
          ? 'Apple 芯片与至少 24 GB 统一内存可兼顾 Medium 的精度和速度'
          : 'CUDA 显存至少 6 GB 且系统内存充足，可运行 Medium',
    }
  }

  if (memoryGiB >= 24 && device.logicalCpuCount >= 8) {
    return {
      variant: 'medium',
      reason: '未检测到可用 GPU，但 CPU 与系统内存可承担 Medium',
    }
  }

  return {
    variant: 'small',
    reason:
      memoryGiB < 24 && device.accelerator === 'mps'
        ? '统一内存低于 24 GB，Small 可减少多模型换页并显著缩短等待'
        : memoryGiB < 16
          ? '当前可用内存更适合 Small'
          : '未检测到适合中大型权重的硬件加速，优先保证稳定性',
  }
}
