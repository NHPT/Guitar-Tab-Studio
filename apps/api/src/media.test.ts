import assert from 'node:assert/strict'
import test from 'node:test'
import {
  authorizedMediaProvider,
  authorizedMediaProviderForContent,
  requiresAuthorizedMediaDecode,
  scoreImageContentValidationError,
  scoreUploadPrefilterError,
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
  for (const filename of [
    'song.qmcflac',
    'song.qmcogg',
    'song.tkm',
    'song.tm6',
    'song.bkcmp3',
    'song.mflac',
    'song.mflac0',
    'song.mgg',
    'song.mgg0',
    'song.mgg1',
    'song.ncm',
    'song.kgg',
    'song.kgm',
    'song.kgma',
    'song.kwm',
  ]) {
    assert.equal(uploadPrefilterError(filename, 'application/octet-stream'), null)
    assert.equal(
      requiresAuthorizedMediaDecode(
        filename,
        Buffer.from('encrypted-container'),
      ),
      true,
    )
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

test('maps protected containers to the authorized provider adapter', () => {
  assert.equal(authorizedMediaProvider('song.qmcflac'), 'qqmusic')
  assert.equal(authorizedMediaProvider('song.ncm'), 'netease')
  assert.equal(authorizedMediaProvider('song.bkcflac'), 'moo')
  assert.equal(authorizedMediaProvider('song.kwm'), 'kuwo')
  assert.equal(authorizedMediaProvider('song.kgg'), 'kugou')
  assert.equal(authorizedMediaProvider('song.kgm'), 'kugou')
  assert.equal(authorizedMediaProvider('song.kgma'), 'kugou')
  assert.equal(authorizedMediaProvider('song.flac'), null)
})

test('uses container headers to distinguish shared encrypted extensions', () => {
  assert.equal(
    authorizedMediaProviderForContent(
      'song.mflac0',
      Buffer.from('yeelion-kuwo-tme'),
    ),
    'kuwo',
  )
  assert.equal(
    authorizedMediaProviderForContent(
      'song.mgg1',
      Buffer.from('7cd532eb86027f4ba8afa68e0fff9914', 'hex'),
    ),
    'kugou',
  )
  assert.equal(
    authorizedMediaProviderForContent(
      'song.mflac',
      Buffer.from('encrypted-qmc'),
    ),
    'qqmusic',
  )
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
    requiresAuthorizedMediaDecode(
      'renamed.mflac',
      Buffer.from('fLaC-standard-audio'),
    ),
    false,
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

test('accepts score images and requires PDFs to be rasterized', () => {
  assert.equal(scoreUploadPrefilterError('score.png', 'image/png'), null)
  assert.equal(scoreUploadPrefilterError('score.pdf', 'application/pdf'), null)
  assert.equal(
    scoreImageContentValidationError(
      'score.png',
      'image/png',
      Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    ),
    null,
  )
  assert.match(
    scoreImageContentValidationError(
      'score.pdf',
      'application/pdf',
      new TextEncoder().encode('%PDF-1.7'),
    ) ?? '',
    /客户端转换/,
  )
  assert.match(
    scoreUploadPrefilterError('score.txt', 'text/plain') ?? '',
    /仅支持/,
  )
})
