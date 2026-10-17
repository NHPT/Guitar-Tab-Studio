import assert from 'node:assert/strict'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import { migrateStudioProject, type AnalysisJob } from '@gts/core'
import { DesktopStateStore } from '../src/state.js'

function projectFixture() {
  return migrateStudioProject({
    id: 'project-1',
    title: 'Desktop fixture',
    artist: 'Local',
    source: {
      kind: 'upload',
      label: 'fixture',
      filename: 'fixture.wav',
    },
    bpm: 120,
    key: 'E minor',
    duration: 2,
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

test('persists projects, recoverable jobs, recents and window state in SQLite', () => {
  const root = resolve(process.cwd(), 'data', `.test-desktop-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  const path = resolve(root, 'desktop.sqlite')
  try {
    const project = projectFixture()
    const queued: AnalysisJob = {
      id: 'job-1',
      source: project.source,
      status: 'queued',
      progress: 0,
      stageLabel: '等待处理',
      createdAt: '2026-09-29T00:00:00.000Z',
      updatedAt: '2026-09-29T00:00:00.000Z',
    }
    const store = new DesktopStateStore(path)
    store.saveProject(project)
    store.saveProject({
      ...project,
      id: 'project-2',
      title: 'Unopened local project',
    })
    store.saveRevision({
      schemaVersion: 1,
      id: 'revision-1',
      projectId: project.id,
      createdAt: '2026-09-29T00:00:01.000Z',
      label: 'Initial',
      kind: 'inference',
      project,
    })
    store.touchRecentProject(project.id)
    store.saveJob(queued)
    store.setRuntimeMode('cloud')
    store.setRuntimeMode('local', project.id)
    store.setCloudApiRoot('https://cloud.example.test')
    store.setHuggingFaceOAuthClientId('desktop-oauth-client')
    store.setTranscriptionPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'small',
      licenseAcceptedAt: '2026-10-14T00:00:00.000Z',
      licenseAcceptedVersion: 'CC-BY-NC-4.0',
    })
    store.saveWindowState({
      x: 100,
      y: 120,
      width: 1280,
      height: 800,
      maximized: false,
    })
    store.close()

    const reopened = new DesktopStateStore(path)
    assert.equal(reopened.getProject(project.id)?.title, 'Desktop fixture')
    assert.equal(reopened.getRevision('revision-1')?.label, 'Initial')
    assert.equal(reopened.listRevisions(project.id).length, 1)
    assert.throws(
      () =>
        reopened.saveRevision({
          schemaVersion: 1,
          id: 'revision-1',
          projectId: project.id,
          createdAt: '2026-09-29T00:00:02.000Z',
          label: 'Overwrite',
          kind: 'manual',
          project,
        }),
      /UNIQUE constraint failed/,
    )
    assert.deepEqual(reopened.recentProjectIds(), [project.id])
    assert.deepEqual(reopened.recentProjects(), [
      {
        id: project.id,
        title: project.title,
        sourceLabel: project.source.label,
        openedAt: reopened.recentProjects()[0].openedAt,
      },
    ])
    assert.deepEqual(
      reopened.localProjects().map((item) => item.id).sort(),
      ['project-1', 'project-2'],
    )
    assert.equal(reopened.listRecoverableJobs()[0]?.id, queued.id)
    assert.equal(reopened.runtimeMode(), 'cloud')
    assert.equal(reopened.runtimeMode(project.id), 'local')
    assert.equal(
      reopened.cloudApiRoot(),
      'https://cloud.example.test',
    )
    assert.equal(
      reopened.huggingFaceOAuthClientId(),
      'desktop-oauth-client',
    )
    assert.deepEqual(reopened.transcriptionPreference(), {
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'small',
      licenseAcceptedAt: '2026-10-14T00:00:00.000Z',
      licenseAcceptedVersion: 'CC-BY-NC-4.0',
    })
    assert.deepEqual(reopened.diagnostics(), {
      projects: 2,
      revisions: 1,
      jobs: { queued: 1 },
    })
    assert.deepEqual(reopened.getWindowState(), {
      x: 100,
      y: 120,
      width: 1280,
      height: 800,
      maximized: false,
    })
    reopened.removeProject(project.id)
    assert.equal(reopened.getProject(project.id), undefined)
    assert.equal(reopened.getRevision('revision-1'), undefined)
    assert.deepEqual(
      reopened.localProjects().map((item) => item.id),
      ['project-2'],
    )
    reopened.close()
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
