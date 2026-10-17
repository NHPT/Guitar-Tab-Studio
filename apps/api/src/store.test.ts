import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import { createStudioProject } from './pipeline.js'
import {
  JsonProjectRepository,
  JsonRevisionRepository,
  referencedJobIdsInStateDirectories,
  recoverInterruptedJob,
} from './store.js'

test('loads legacy project JSON through the shared schema migrator', () => {
  const root = resolve(process.cwd(), 'data', `.test-projects-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const current = createStudioProject({
      kind: 'upload',
      label: 'fixture',
      filename: 'fixture.wav',
    })
    const { schemaVersion: _schemaVersion, ...legacy } = current
    writeFileSync(
      resolve(root, `${legacy.id}.json`),
      JSON.stringify(legacy),
      'utf8',
    )

    const repository = new JsonProjectRepository(root)
    const restored = repository.getProject(legacy.id)
    assert.equal(restored?.schemaVersion, 1)
    assert.equal(restored?.title, legacy.title)

    if (!restored) assert.fail('project was not restored')
    repository.saveProject(restored)
    const persisted = JSON.parse(
      readFileSync(resolve(root, `${legacy.id}.json`), 'utf8'),
    ) as { schemaVersion?: number }
    assert.equal(persisted.schemaVersion, 1)

    repository.removeProject(legacy.id)
    assert.equal(repository.getProject(legacy.id), undefined)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('marks unfinished jobs as interrupted after a process restart', () => {
  const project = createStudioProject({
    kind: 'upload',
    label: 'fixture',
    filename: 'fixture.wav',
  })
  const recovered = recoverInterruptedJob(
    {
      id: 'job-interrupted',
      source: project.source,
      status: 'separating',
      progress: 35,
      stageLabel: 'Worker 正在分离音轨',
      createdAt: '2026-09-29T00:00:00.000Z',
      updatedAt: '2026-09-29T00:01:00.000Z',
    },
    '2026-09-29T00:02:00.000Z',
  )
  assert.equal(recovered.status, 'failed')
  assert.equal(recovered.stageLabel, '任务已中断')
  assert.equal(recovered.updatedAt, '2026-09-29T00:02:00.000Z')
})

test('stores revision snapshots immutably and filters them by project', () => {
  const root = resolve(process.cwd(), 'data', `.test-revisions-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const project = createStudioProject({
      kind: 'upload',
      label: 'fixture',
      filename: 'fixture.wav',
    })
    const repository = new JsonRevisionRepository(root)
    const revision = {
      schemaVersion: 1 as const,
      id: 'revision-1',
      projectId: project.id,
      createdAt: '2026-09-29T00:01:00.000Z',
      label: 'Initial',
      kind: 'inference' as const,
      project,
    }
    repository.saveRevision(revision)
    assert.equal(repository.getRevision(revision.id)?.label, 'Initial')
    const stored = repository.listRevisions(project.id)
    assert.equal(stored.length, 1)
    assert.equal(stored[0].id, revision.id)
    assert.equal(stored[0].project.id, project.id)
    assert.throws(() => repository.saveRevision(revision), /EEXIST/)
    assert.deepEqual(repository.listRevisions('other-project'), [])
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('protects media referenced by legacy or partially written project state', () => {
  const root = resolve(process.cwd(), 'data', `.test-state-media-${randomUUID()}`)
  const projects = resolve(root, 'projects')
  const revisions = resolve(root, 'revisions')
  mkdirSync(projects, { recursive: true })
  mkdirSync(revisions, { recursive: true })
  try {
    writeFileSync(
      resolve(projects, 'legacy.json'),
      JSON.stringify({
        tracks: [
          {
            url: '/media/jobs/legacy-job/media/guitar.m4a',
            unsupportedLegacyField: true,
          },
        ],
      }),
      'utf8',
    )
    writeFileSync(
      resolve(revisions, 'partial.json'),
      '{"project":{"tracks":[{"url":"/media/jobs/partial-job/media/bass.m4a"}]',
      'utf8',
    )

    assert.deepEqual(
      [...referencedJobIdsInStateDirectories([projects, revisions])].sort(),
      ['legacy-job', 'partial-job'],
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
