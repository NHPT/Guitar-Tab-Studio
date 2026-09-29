import assert from 'node:assert/strict'
import test from 'node:test'
import {
  uploadContentValidationError,
  uploadPrefilterError,
} from './media.js'

test('accepts standard audio and video containers', () => {
  assert.equal(uploadPrefilterError('song.flac', 'application/octet-stream'), null)
  assert.equal(uploadPrefilterError('practice.M4A', 'audio/mp4'), null)
  assert.equal(uploadPrefilterError('camera.webm', 'video/webm'), null)
  assert.equal(uploadPrefilterError('archive.ape', 'application/octet-stream'), null)
})

test('inspects proprietary extensions and rejects encrypted containers', () => {
  for (const filename of ['song.qmcflac', 'song.mflac', 'song.mgg', 'song.ncm', 'song.kgm']) {
    assert.equal(uploadPrefilterError(filename, 'application/octet-stream'), null)
    assert.match(
      uploadContentValidationError(
        filename,
        'application/octet-stream',
        Buffer.from('encrypted-container'),
      ) ?? '',
      /专有加密容器/,
    )
  }
})

test('accepts a proprietary extension only when the content is standard audio', () => {
  assert.equal(
    uploadContentValidationError(
      'renamed.mflac',
      'application/octet-stream',
      Buffer.from('fLaC-standard-audio'),
    ),
    null,
  )
  assert.equal(
    uploadContentValidationError(
      'renamed.mgg',
      'application/octet-stream',
      Buffer.from('OggS-standard-audio'),
    ),
    null,
  )
})

test('rejects unrelated file formats', () => {
  assert.match(uploadPrefilterError('notes.pdf', 'application/pdf') ?? '', /仅支持/)
})
