import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import { DesktopMediaRepository } from '../src/media-repository.js'

test('stores validated media behind an opaque identifier', () => {
  const root = resolve(process.cwd(), 'data', `.test-media-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  try {
    const repository = new DesktopMediaRepository(root)
    const bytes = new Uint8Array(16)
    bytes.set(Buffer.from('RIFF'), 0)
    bytes.set(Buffer.from('WAVE'), 8)
    const media = repository.putMedia({
      displayName: 'fixture.wav',
      mediaType: 'audio/wav',
      bytes,
    })
    assert.match(media.id, /^[0-9a-f-]{36}$/u)
    assert.equal(repository.getMedia(media.id)?.displayName, 'fixture.wav')
    assert.ok(repository.contentPathFor(media.id)?.endsWith('.media'))
    assert.equal(repository.getMedia('../outside'), undefined)
    repository.removeMedia(media.id)
    assert.equal(repository.getMedia(media.id), undefined)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
