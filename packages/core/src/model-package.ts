import { ANALYSIS_SCHEMA_VERSION } from './analysis.js'
import {
  CoreValidationError,
  arrayAt,
  booleanAt,
  enumAt,
  numberAt,
  optionalStringAt,
  recordAt,
  stringAt,
} from './validation.js'

export const MODEL_PACKAGE_SCHEMA_VERSION = 1 as const

export type ModelPackageFileRole =
  | 'worker'
  | 'ffmpeg'
  | 'separation-model'
  | 'transcription-model'
  | 'technique-model'
  | 'technique-gate'
  | 'license'

export interface ModelPackageManifest {
  schemaVersion: typeof MODEL_PACKAGE_SCHEMA_VERSION
  id: string
  version: string
  releaseChannel?: 'research' | 'production'
  createdAt: string
  application: {
    minimumVersion: string
    maximumVersion?: string
  }
  runtime: {
    platform: 'darwin' | 'win32' | 'linux'
    arch: 'arm64' | 'x64'
    analysisSchemaVersion: typeof ANALYSIS_SCHEMA_VERSION
  }
  files: Array<{
    path: string
    role: ModelPackageFileRole
    sizeBytes: number
    sha256: string
    executable: boolean
  }>
}

export interface InstalledModelPackage {
  manifest: ModelPackageManifest
  active: boolean
}

const identifierPattern = /^[a-z0-9][a-z0-9._-]{0,63}$/u
const versionPattern = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u
const sha256Pattern = /^[0-9a-f]{64}$/u

function semanticVersion(value: string): [number, number, number] {
  const match = value.match(/^(\d+)\.(\d+)\.(\d+)/u)
  if (!match) throw new CoreValidationError('version', 'expected semantic version')
  return [Number(match[1]), Number(match[2]), Number(match[3])]
}

function compareVersion(left: string, right: string): number {
  const leftParts = semanticVersion(left)
  const rightParts = semanticVersion(right)
  for (let index = 0; index < leftParts.length; index += 1) {
    const delta = leftParts[index] - rightParts[index]
    if (delta !== 0) return delta
  }
  return 0
}

export function isApplicationVersionCompatible(
  manifest: ModelPackageManifest,
  applicationVersion: string,
): boolean {
  return (
    compareVersion(applicationVersion, manifest.application.minimumVersion) >=
      0 &&
    (!manifest.application.maximumVersion ||
      compareVersion(applicationVersion, manifest.application.maximumVersion) <=
        0)
  )
}

export function parseModelPackageManifest(
  value: unknown,
): ModelPackageManifest {
  const manifest = recordAt(value, 'manifest')
  const schemaVersion = numberAt(
    manifest.schemaVersion,
    'manifest.schemaVersion',
  )
  if (schemaVersion !== MODEL_PACKAGE_SCHEMA_VERSION) {
    throw new CoreValidationError(
      'manifest.schemaVersion',
      `unsupported version ${schemaVersion}`,
    )
  }
  const id = stringAt(manifest.id, 'manifest.id')
  const version = stringAt(manifest.version, 'manifest.version')
  if (!identifierPattern.test(id)) {
    throw new CoreValidationError('manifest.id', 'invalid package identifier')
  }
  if (!versionPattern.test(version)) {
    throw new CoreValidationError('manifest.version', 'invalid semantic version')
  }
  const application = recordAt(manifest.application, 'manifest.application')
  const minimumVersion = stringAt(
    application.minimumVersion,
    'manifest.application.minimumVersion',
  )
  const maximumVersion = optionalStringAt(
    application.maximumVersion,
    'manifest.application.maximumVersion',
  )
  semanticVersion(minimumVersion)
  if (maximumVersion) semanticVersion(maximumVersion)
  const runtime = recordAt(manifest.runtime, 'manifest.runtime')
  const analysisSchemaVersion = numberAt(
    runtime.analysisSchemaVersion,
    'manifest.runtime.analysisSchemaVersion',
  )
  if (analysisSchemaVersion !== ANALYSIS_SCHEMA_VERSION) {
    throw new CoreValidationError(
      'manifest.runtime.analysisSchemaVersion',
      `expected ${ANALYSIS_SCHEMA_VERSION}`,
    )
  }
  const files = arrayAt(manifest.files, 'manifest.files').map((value, index) => {
    const path = `manifest.files[${index}]`
    const file = recordAt(value, path)
    const relativePath = stringAt(file.path, `${path}.path`)
    if (
      relativePath.startsWith('/') ||
      relativePath.startsWith('\\') ||
      relativePath.split(/[\\/]/u).includes('..')
    ) {
      throw new CoreValidationError(`${path}.path`, 'path must stay in package')
    }
    const sizeBytes = numberAt(file.sizeBytes, `${path}.sizeBytes`)
    if (!Number.isInteger(sizeBytes) || sizeBytes < 0) {
      throw new CoreValidationError(
        `${path}.sizeBytes`,
        'expected non-negative integer',
      )
    }
    const sha256 = stringAt(file.sha256, `${path}.sha256`)
    if (!sha256Pattern.test(sha256)) {
      throw new CoreValidationError(`${path}.sha256`, 'invalid SHA-256')
    }
    return {
      path: relativePath,
      role: enumAt(
        file.role,
        [
          'worker',
          'ffmpeg',
          'separation-model',
          'transcription-model',
          'technique-model',
          'technique-gate',
          'license',
        ] as const,
        `${path}.role`,
      ),
      sizeBytes,
      sha256,
      executable: booleanAt(file.executable, `${path}.executable`),
    }
  })
  if (files.length === 0) {
    throw new CoreValidationError('manifest.files', 'package cannot be empty')
  }
  const paths = new Set(files.map((file) => file.path))
  if (paths.size !== files.length) {
    throw new CoreValidationError('manifest.files', 'duplicate file path')
  }
  return {
    schemaVersion: MODEL_PACKAGE_SCHEMA_VERSION,
    id,
    version,
    releaseChannel:
      manifest.releaseChannel === undefined
        ? 'research'
        : enumAt(
            manifest.releaseChannel,
            ['research', 'production'] as const,
            'manifest.releaseChannel',
          ),
    createdAt: stringAt(manifest.createdAt, 'manifest.createdAt'),
    application: { minimumVersion, maximumVersion },
    runtime: {
      platform: enumAt(
        runtime.platform,
        ['darwin', 'win32', 'linux'] as const,
        'manifest.runtime.platform',
      ),
      arch: enumAt(
        runtime.arch,
        ['arm64', 'x64'] as const,
        'manifest.runtime.arch',
      ),
      analysisSchemaVersion: ANALYSIS_SCHEMA_VERSION,
    },
    files,
  }
}
