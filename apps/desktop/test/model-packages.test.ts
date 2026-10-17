import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import {
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import type { ModelPackageManifest } from '@gts/core'
import { ModelPackageManager } from '../src/model-packages.js'

function fixturePackage(
  root: string,
  id: string,
  version: string,
): ModelPackageManifest {
  const content = Buffer.from(`model-${id}-${version}`)
  const manifest: ModelPackageManifest = {
    schemaVersion: 1,
    id,
    version,
    createdAt: '2026-09-29T00:00:00.000Z',
    application: { minimumVersion: '0.1.0' },
    runtime: {
      platform: process.platform as 'darwin' | 'win32' | 'linux',
      arch: process.arch as 'arm64' | 'x64',
      analysisSchemaVersion: 9,
    },
    files: [
      {
        path: 'models/tab.bin',
        role: 'transcription-model',
        sizeBytes: content.length,
        sha256: createHash('sha256').update(content).digest('hex'),
        executable: false,
      },
    ],
  }
  mkdirSync(resolve(root, 'models'), { recursive: true })
  writeFileSync(resolve(root, 'models/tab.bin'), content)
  writeFileSync(
    resolve(root, 'manifest.json'),
    JSON.stringify(manifest),
    'utf8',
  )
  return manifest
}

test('installs, activates and rolls back verified model packages', async () => {
  const root = resolve(process.cwd(), 'data', `.test-models-${randomUUID()}`)
  const sourceA = resolve(root, 'source-a')
  const sourceB = resolve(root, 'source-b')
  const installed = resolve(root, 'installed')
  mkdirSync(sourceA, { recursive: true })
  mkdirSync(sourceB, { recursive: true })
  try {
    fixturePackage(sourceA, 'gts-tab', '1.0.0')
    fixturePackage(sourceB, 'gts-tab', '1.1.0')
    const manager = new ModelPackageManager(installed, '0.1.0')
    await manager.installFromDirectory(sourceA)
    await manager.installFromDirectory(sourceB)
    await manager.activate('gts-tab', '1.0.0')
    await manager.activate('gts-tab', '1.1.0')
    assert.equal(manager.activeManifest()?.version, '1.1.0')
    assert.equal(manager.activeManifest()?.releaseChannel, 'research')
    assert.equal(
      manager.resolveActiveFile('transcription-model'),
      resolve(installed, 'gts-tab', '1.1.0', 'models', 'tab.bin'),
    )
    assert.throws(() => manager.remove('gts-tab', '1.1.0'), /active/)
    await manager.rollback()
    assert.equal(manager.activeManifest()?.version, '1.0.0')
    writeFileSync(
      resolve(installed, 'gts-tab/1.0.0/models/tab.bin'),
      'tampered',
    )
    await assert.rejects(manager.verifyActive(), /file is invalid|hash mismatch/)
    await manager.recoverActive()
    assert.equal(manager.activeManifest()?.version, '1.1.0')
    assert.match(manager.recoveryWarning() ?? '', /已回滚/)
    manager.remove('gts-tab', '1.0.0')
    assert.equal(manager.listInstalled().length, 1)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects a model package with a mismatched file hash', async () => {
  const root = resolve(process.cwd(), 'data', `.test-models-${randomUUID()}`)
  const source = resolve(root, 'source')
  mkdirSync(source, { recursive: true })
  try {
    const manifest = fixturePackage(source, 'gts-invalid', '1.0.0')
    manifest.files[0].sha256 = '0'.repeat(64)
    writeFileSync(
      resolve(source, 'manifest.json'),
      JSON.stringify(manifest),
      'utf8',
    )
    const manager = new ModelPackageManager(
      resolve(root, 'installed'),
      '0.1.0',
    )
    await assert.rejects(
      manager.installFromDirectory(source),
      /hash mismatch/,
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('resumes an HTTPS package download and verifies its digest', async () => {
  const root = resolve(process.cwd(), 'data', `.test-models-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const destination = resolve(root, 'package.bin')
    writeFileSync(`${destination}.part`, 'hello ')
    const expected = createHash('sha256').update('hello world').digest('hex')
    const manager = new ModelPackageManager(
      resolve(root, 'installed'),
      '0.1.0',
    )
    const requests: RequestInit[] = []
    await manager.download(
      'https://models.example/package.bin',
      destination,
      expected,
      async (_input, init) => {
        requests.push(init ?? {})
        return new Response('world', { status: 206 })
      },
    )
    assert.equal(
      new Headers(requests[0].headers).get('Range'),
      'bytes=6-',
    )
    assert.equal(readFileSync(destination, 'utf8'), 'hello world')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('stops a model download that exceeds its declared size', async () => {
  const root = resolve(process.cwd(), 'data', `.test-models-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const destination = resolve(root, 'oversized.bin')
    const manager = new ModelPackageManager(
      resolve(root, 'installed'),
      '0.1.0',
    )
    await assert.rejects(
      manager.download(
        'https://models.example/oversized.bin',
        destination,
        createHash('sha256').update('too large').digest('hex'),
        async () => new Response('too large'),
        3,
      ),
      /exceeds its declared size/,
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('installs all files from a same-origin HTTPS manifest', async () => {
  const root = resolve(process.cwd(), 'data', `.test-models-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const content = Buffer.from('downloaded-model')
    const manifest: ModelPackageManifest = {
      schemaVersion: 1,
      id: 'gts-download',
      version: '1.0.0',
      createdAt: '2026-09-29T00:00:00.000Z',
      application: { minimumVersion: '0.1.0' },
      runtime: {
        platform: process.platform as 'darwin' | 'win32' | 'linux',
        arch: process.arch as 'arm64' | 'x64',
        analysisSchemaVersion: 9,
      },
      files: [
        {
          path: 'model.bin',
          role: 'transcription-model',
          sizeBytes: content.length,
          sha256: createHash('sha256').update(content).digest('hex'),
          executable: false,
        },
      ],
    }
    const manager = new ModelPackageManager(
      resolve(root, 'installed'),
      '0.1.0',
    )
    const installed = await manager.installFromUrl(
      'https://models.example/releases/manifest.json',
      async (input) => {
        const url = String(input)
        return url.endsWith('/manifest.json')
          ? Response.json(manifest)
          : new Response(content)
      },
    )
    assert.equal(installed.manifest.id, manifest.id)
    assert.equal(manager.listInstalled().length, 1)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
