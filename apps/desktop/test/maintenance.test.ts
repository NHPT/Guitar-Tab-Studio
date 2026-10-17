import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import { migrateStudioProject } from '@gts/core'
import { DesktopMaintenance } from '../src/maintenance.js'
import { DesktopStateStore } from '../src/state.js'

function projectFixture(title: string) {
  return migrateStudioProject({
    id: 'project-backup',
    title,
    artist: 'Local',
    source: { kind: 'upload', label: 'fixture', filename: 'fixture.wav' },
    bpm: 100,
    key: 'C major',
    duration: 1,
    createdAt: '2026-09-29T00:00:00.000Z',
    analysisMode: 'inference',
    tracks: [],
    tab: {
      tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
      capo: 0,
      measures: [],
    },
    warnings: [],
  })
}

test('creates a verified backup, restores it and clears only cache data', async () => {
  const root = resolve(
    process.cwd(),
    'data',
    `.test-maintenance-${randomUUID()}`,
  )
  const runtime = resolve(root, 'runtime')
  const cache = resolve(root, 'cache')
  const backups = resolve(root, 'backups')
  mkdirSync(resolve(runtime, 'jobs'), { recursive: true })
  mkdirSync(cache, { recursive: true })
  writeFileSync(resolve(runtime, 'jobs/result.bin'), 'result')
  writeFileSync(resolve(cache, 'temporary.bin'), 'cache')
  const state = new DesktopStateStore(resolve(root, 'state/desktop.sqlite'))
  try {
    state.saveProject(projectFixture('Before backup'))
    const maintenance = new DesktopMaintenance(
      '0.1.0',
      state,
      runtime,
      cache,
    )
    const backupName = await maintenance.createBackup(backups)
    const backupRoot = resolve(backups, backupName)
    assert.equal(existsSync(resolve(backupRoot, 'backup-manifest.json')), true)

    state.saveProject(projectFixture('After backup'))
    rmSync(resolve(runtime, 'jobs/result.bin'))
    await maintenance.restoreBackup(backupRoot)
    assert.equal(state.getProject('project-backup')?.title, 'Before backup')
    assert.equal(
      readFileSync(resolve(runtime, 'jobs/result.bin'), 'utf8'),
      'result',
    )

    const removedBytes = maintenance.clearCache()
    assert.ok(removedBytes >= Buffer.byteLength('cache'))
    assert.equal(existsSync(cache), false)
    assert.equal(existsSync(resolve(runtime, 'uploads')), false)
    assert.equal(existsSync(resolve(runtime, 'jobs/result.bin')), true)
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})
