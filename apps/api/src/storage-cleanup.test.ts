import assert from 'node:assert/strict'
import {
  existsSync,
  mkdirSync,
  rmSync,
  truncateSync,
  utimesSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import type { AnalysisJob } from '@gts/core'
import {
  DEFAULT_STORAGE_CLEANUP_CONFIG,
  StorageCleanupService,
} from './storage-cleanup.js'

const GIBIBYTE = 1024 ** 3
const NOW = new Date('2026-10-10T12:00:00.000Z')

function job(id: string, status: AnalysisJob['status']): AnalysisJob {
  return {
    id,
    source: { kind: 'upload', label: 'fixture', filename: 'fixture.wav' },
    status,
    progress: 0,
    stageLabel: 'fixture',
    createdAt: '2026-10-08T00:00:00.000Z',
    updatedAt: '2026-10-08T00:00:00.000Z',
  }
}

function fixtureRoot(): string {
  const root = resolve(
    process.cwd(),
    'data',
    `.test-storage-cleanup-${randomUUID()}`,
  )
  mkdirSync(resolve(root, 'jobs'), { recursive: true })
  mkdirSync(resolve(root, 'uploads'), { recursive: true })
  return root
}

function touchOld(path: string): void {
  const date = new Date('2026-10-08T00:00:00.000Z')
  utimesSync(path, date, date)
}

test('enables automatic cleanup by default and persists administrator settings', () => {
  const root = fixtureRoot()
  try {
    const service = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [],
        activeJobIds: new Set(),
        activeUploadPaths: new Set(),
        referencedJobIds: new Set(),
      }),
      availableBytes: () => 12 * GIBIBYTE,
      now: () => NOW,
    })
    assert.deepEqual(service.status().config, DEFAULT_STORAGE_CLEANUP_CONFIG)
    service.updateConfig({
      enabled: false,
      triggerFreeBytes: 2 * GIBIBYTE,
      targetFreeBytes: 4 * GIBIBYTE,
      retentionHours: 48,
    })

    const reopened = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [],
        activeJobIds: new Set(),
        activeUploadPaths: new Set(),
        referencedJobIds: new Set(),
      }),
      availableBytes: () => 12 * GIBIBYTE,
      now: () => NOW,
    })
    assert.deepEqual(reopened.status().config, {
      enabled: false,
      triggerFreeBytes: 2 * GIBIBYTE,
      targetFreeBytes: 4 * GIBIBYTE,
      retentionHours: 48,
    })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('protects active inputs and project media while removing only managed artifacts', () => {
  const root = fixtureRoot()
  const protectedRoot = resolve(root, 'jobs/protected')
  const failedRoot = resolve(root, 'jobs/failed')
  const activeRoot = resolve(root, 'jobs/active')
  const orphanRoot = resolve(root, 'jobs/orphan')
  const activeUpload = resolve(root, 'uploads/active-upload')
  const staleUpload = resolve(root, 'uploads/stale-upload')
  const uploadPlaceholder = resolve(root, 'uploads/.gitkeep')
  mkdirSync(resolve(protectedRoot, 'media'), { recursive: true })
  mkdirSync(failedRoot, { recursive: true })
  mkdirSync(activeRoot, { recursive: true })
  mkdirSync(orphanRoot, { recursive: true })
  writeFileSync(resolve(protectedRoot, 'media/guitar.m4a'), 'playable')
  writeFileSync(resolve(protectedRoot, 'normalized.wav'), 'temporary')
  writeFileSync(resolve(protectedRoot, 'analysis.json'), '{}')
  writeFileSync(resolve(failedRoot, 'normalized.wav'), 'failed')
  writeFileSync(resolve(activeRoot, 'normalized.wav'), 'active')
  writeFileSync(resolve(orphanRoot, 'source.m4a'), 'orphan')
  writeFileSync(activeUpload, 'active upload')
  writeFileSync(staleUpload, 'stale upload')
  writeFileSync(uploadPlaceholder, '\n')

  try {
    const service = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [
          job('protected', 'completed'),
          job('failed', 'failed'),
          job('active', 'separating'),
        ],
        activeJobIds: new Set(['active']),
        activeUploadPaths: new Set([activeUpload]),
        referencedJobIds: new Set(['protected']),
      }),
      availableBytes: () => GIBIBYTE,
      now: () => NOW,
    })
    const run = service.run('manual', { force: true })

    assert.equal(run.outcome, 'completed')
    assert.equal(existsSync(resolve(protectedRoot, 'media/guitar.m4a')), true)
    assert.equal(existsSync(resolve(protectedRoot, 'normalized.wav')), false)
    assert.equal(existsSync(resolve(protectedRoot, 'analysis.json')), false)
    assert.equal(existsSync(failedRoot), false)
    assert.equal(existsSync(orphanRoot), false)
    assert.equal(existsSync(activeRoot), true)
    assert.equal(existsSync(activeUpload), true)
    assert.equal(existsSync(staleUpload), false)
    assert.equal(existsSync(uploadPlaceholder), true)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('honors the disabled mode for automatic runs but still allows manual cleanup', () => {
  const root = fixtureRoot()
  const failedRoot = resolve(root, 'jobs/failed')
  mkdirSync(failedRoot, { recursive: true })
  writeFileSync(resolve(failedRoot, 'partial.wav'), 'partial')
  try {
    const service = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [job('failed', 'failed')],
        activeJobIds: new Set(),
        activeUploadPaths: new Set(),
        referencedJobIds: new Set(),
      }),
      availableBytes: () => 0,
      now: () => NOW,
    })
    service.updateConfig({
      ...DEFAULT_STORAGE_CLEANUP_CONFIG,
      enabled: false,
    })
    const automatic = service.run('interval')
    assert.equal(automatic.outcome, 'skipped')
    assert.equal(existsSync(failedRoot), true)

    const manual = service.run('manual', { force: true })
    assert.equal(manual.outcome, 'completed')
    assert.equal(existsSync(failedRoot), false)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('removes oldest candidates only until the configured free-space target', () => {
  const root = fixtureRoot()
  const orphanRoots = ['oldest', 'middle', 'newest'].map((name) =>
    resolve(root, 'jobs', name),
  )
  for (const [index, path] of orphanRoots.entries()) {
    mkdirSync(path, { recursive: true })
    const file = resolve(path, 'partial.wav')
    writeFileSync(file, '')
    truncateSync(file, GIBIBYTE)
    const date = new Date(NOW.getTime() - (72 - index) * 60 * 60 * 1000)
    utimesSync(file, date, date)
    utimesSync(path, date, date)
  }
  try {
    const service = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [],
        activeJobIds: new Set(),
        activeUploadPaths: new Set(),
        referencedJobIds: new Set(),
      }),
      availableBytes: () =>
        orphanRoots.filter((path) => !existsSync(path)).length * GIBIBYTE,
      now: () => NOW,
    })
    service.updateConfig({
      enabled: true,
      triggerFreeBytes: GIBIBYTE,
      targetFreeBytes: 2 * GIBIBYTE,
      retentionHours: 24,
    })
    const run = service.run('capacity', { requiredBytes: GIBIBYTE })

    assert.equal(run.removedItems, 2)
    assert.equal(run.removedBytes, 2 * GIBIBYTE)
    assert.equal(run.targetReached, true)
    assert.equal(existsSync(orphanRoots[0]), false)
    assert.equal(existsSync(orphanRoots[1]), false)
    assert.equal(existsSync(orphanRoots[2]), true)
    assert.equal(service.status().history.length, 1)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('expires stale uploads during normal runs without touching fresh uploads', () => {
  const root = fixtureRoot()
  const stale = resolve(root, 'uploads/stale')
  const fresh = resolve(root, 'uploads/fresh')
  writeFileSync(stale, 'stale')
  writeFileSync(fresh, 'fresh')
  touchOld(stale)
  try {
    const service = new StorageCleanupService({
      dataRoot: root,
      protection: () => ({
        jobs: [],
        activeJobIds: new Set(),
        activeUploadPaths: new Set(),
        referencedJobIds: new Set(),
      }),
      availableBytes: () => 12 * GIBIBYTE,
      now: () => NOW,
    })
    const run = service.run('interval')
    assert.equal(run.removedItems, 1)
    assert.equal(existsSync(stale), false)
    assert.equal(existsSync(fresh), true)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
