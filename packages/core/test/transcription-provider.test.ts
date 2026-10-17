import assert from 'node:assert/strict'
import test from 'node:test'
import {
  filterTranscriptionProviders,
  recommendTranscriptionVariant,
  type TranscriptionDeviceProfile,
} from '../src/index.js'

function device(
  overrides: Partial<TranscriptionDeviceProfile>,
): TranscriptionDeviceProfile {
  return {
    platform: 'linux',
    arch: 'x64',
    totalMemoryBytes: 16 * 1024 ** 3,
    logicalCpuCount: 8,
    accelerator: 'cpu',
    ...overrides,
  }
}

test('recommends Small for memory-constrained or CPU-only devices', () => {
  assert.equal(
    recommendTranscriptionVariant(
      device({
        totalMemoryBytes: 12 * 1024 ** 3,
        logicalCpuCount: 12,
      }),
    ).variant,
    'small',
  )
  assert.equal(
    recommendTranscriptionVariant(
      device({
        totalMemoryBytes: 16 * 1024 ** 3,
        logicalCpuCount: 4,
      }),
    ).variant,
    'small',
  )
})

test('keeps 16 GB unified-memory devices on Small', () => {
  assert.equal(
    recommendTranscriptionVariant(
      device({
        platform: 'darwin',
        arch: 'arm64',
        accelerator: 'mps',
        totalMemoryBytes: 16 * 1024 ** 3,
      }),
    ).variant,
    'small',
  )
})

test('recommends Medium when accelerator memory or CPU capacity is sufficient', () => {
  assert.equal(
    recommendTranscriptionVariant(
      device({
        platform: 'darwin',
        arch: 'arm64',
        accelerator: 'mps',
        totalMemoryBytes: 24 * 1024 ** 3,
      }),
    ).variant,
    'medium',
  )
  assert.equal(
    recommendTranscriptionVariant(
      device({
        totalMemoryBytes: 24 * 1024 ** 3,
        logicalCpuCount: 12,
      }),
    ).variant,
    'medium',
  )
})

test('recommends Large only when accelerator memory is sufficient', () => {
  assert.equal(
    recommendTranscriptionVariant(
      device({
        accelerator: 'cuda',
        acceleratorMemoryBytes: 16 * 1024 ** 3,
        totalMemoryBytes: 32 * 1024 ** 3,
      }),
    ).variant,
    'large',
  )
  assert.equal(
    recommendTranscriptionVariant(
      device({
        accelerator: 'cuda',
        acceleratorMemoryBytes: 8 * 1024 ** 3,
        totalMemoryBytes: 32 * 1024 ** 3,
      }),
    ).variant,
    'medium',
  )
})

test('commercial policy excludes noncommercial providers even if allowlisted', () => {
  assert.deepEqual(
    filterTranscriptionProviders(
      [
        'basic-pitch',
        'muscriptor',
        'guitar-specialist',
        'multitrack-research',
      ],
      true,
    ).map((provider) => provider.id),
    ['basic-pitch'],
  )
  assert.deepEqual(
    filterTranscriptionProviders(
      [
        'basic-pitch',
        'muscriptor',
        'guitar-specialist',
        'multitrack-research',
      ],
      false,
    ).map((provider) => provider.id),
    [
      'basic-pitch',
      'muscriptor',
      'guitar-specialist',
      'multitrack-research',
    ],
  )
})
