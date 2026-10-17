import assert from 'node:assert/strict'
import { createCipheriv } from 'node:crypto'
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import test from 'node:test'
import { QMC2, decryptQMC1, ready } from '@clamber_l/crypto/inline'
import {
  decodeAuthorizedMediaLocally,
  localAuthorizedMediaDecoderAvailable,
} from './authorized-media.js'

function uint32le(value: number): Buffer {
  const result = Buffer.alloc(4)
  result.writeUInt32LE(value)
  return result
}

function ncmFixture(audio: Uint8Array): Buffer {
  const contentKey = Buffer.from([
    0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f,
  ])
  const cipher = createCipheriv(
    'aes-128-ecb',
    Buffer.from('687a4852416d736f356b496e62617857', 'hex'),
    null,
  )
  const encryptedKey = Buffer.concat([
    cipher.update(
      Buffer.concat([Buffer.from('neteasecloudmusic'), contentKey]),
    ),
    cipher.final(),
  ]).map((value) => value ^ 0x64)
  const box = Uint8Array.from({ length: 256 }, (_, index) => index)
  let keyOffset = 0
  for (let index = 0; index < 256; index += 1) {
    keyOffset =
      (box[index] +
        keyOffset +
        contentKey[index % contentKey.byteLength]) &
      0xff
    ;[box[index], box[keyOffset]] = [box[keyOffset], box[index]]
  }
  const keyStream = box.map((_, index, values) => {
    const first = values[(index + 1) & 0xff]
    const second = values[((index + 1) + first) & 0xff]
    return values[(first + second) & 0xff]
  })
  const encryptedAudio = Buffer.from(
    audio.map((value, index) => value ^ keyStream[index & 0xff]),
  )
  return Buffer.concat([
    Buffer.from('4354454e4644414d', 'hex'),
    Buffer.alloc(2),
    uint32le(encryptedKey.byteLength),
    encryptedKey,
    uint32le(0),
    Buffer.alloc(13),
    encryptedAudio,
  ])
}

function musicExFooter(
  mediaName: string,
  songMid = '003w2xz20QlUZt',
): Buffer {
  const footer = Buffer.alloc(192)
  footer.write(songMid, 12, 'utf16le')
  footer.write(mediaName, 72, 'utf16le')
  footer.writeUInt32LE(0xc0, 176)
  footer.writeUInt32LE(1, 180)
  footer.write('musicex\0', 184, 'ascii')
  return footer
}

test('decodes QMC1 entirely offline into an ephemeral standard file', async () => {
  const root = resolve(process.cwd(), '.test-authorized-media')
  const inputPath = resolve(root, 'input.qmcflac')
  const outputPath = resolve(root, 'output.media')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  const standardAudio = Buffer.concat([
    Buffer.from('fLaC'),
    Buffer.from('offline-qmc-audio'),
  ])
  const protectedAudio = Uint8Array.from(standardAudio)
  await ready
  decryptQMC1(protectedAudio, 0)
  await writeFile(inputPath, protectedAudio)
  try {
    assert.equal(localAuthorizedMediaDecoderAvailable(), true)
    await decodeAuthorizedMediaLocally({
      provider: 'qqmusic',
      inputPath,
      outputPath,
      originalName: 'song.qmcflac',
    })
    assert.deepEqual(await readFile(outputPath), standardAudio)
    if (process.platform !== 'win32') {
      assert.equal((await stat(outputPath)).mode & 0o777, 0o600)
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('decodes NCM entirely offline without returning a decoded download', async () => {
  const root = resolve(process.cwd(), '.test-authorized-media-ncm')
  const inputPath = resolve(root, 'input.ncm')
  const outputPath = resolve(root, 'output.media')
  const standardAudio = Buffer.concat([
    Buffer.from('fLaC'),
    Buffer.from('offline-ncm-audio'),
  ])
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  await writeFile(inputPath, ncmFixture(standardAudio))
  try {
    await decodeAuthorizedMediaLocally({
      provider: 'netease',
      inputPath,
      outputPath,
      originalName: 'song.ncm',
    })
    assert.deepEqual(await readFile(outputPath), standardAudio)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('decodes a keyless MusicEx QMC2 file after official authorization', async () => {
  const root = resolve(process.cwd(), '.test-authorized-media-qmc2-ekey')
  const inputPath = resolve(root, 'input.mflac')
  const outputPath = resolve(root, 'output.media')
  const mediaName = 'F0M0003w2xz20QlUZt.mflac'
  const songMid = '002atvb82SRmG2'
  const ekey =
    'eXc3eFdPeU6+3f7GVeF35bMpIEIQj5JWOWt7G+jsR68Hx3BUFBavkTQ8dpPdP0XBIwPe+OfdsnTGVQqPyg3GCtQSrkgA0mwSQdr4DPzKLkEZFX+Cf1V6ChyipOuC6KT37eAxWMdV1UHf9/OCvydr1dc6SWK1ijRUcP6IAHQhiB+mZLay7XXrSPo32WjdBkn9c9sa2SLtI48atj5kfZ4oOq6QGeld2JA3Z+3wwCe6uTHthKaEHY8ufDYodEe3qqrjYpzkdx55pCtxCQa1JiNqFmJigWm4m3CDzhuJ7YqnjbD+mXxLi7BP1+z4L6nccE2h+DGHVqpGjR9+4LBpe4WHB4DrAzVp2qQRRQJxeHd1v88='
  const standardAudio = Buffer.concat([
    Buffer.from('fLaC'),
    Buffer.from('officially-authorized-qmc2-audio'),
  ])
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  await ready
  const protectedAudio = Buffer.from(standardAudio)
  const encoder = new QMC2(ekey)
  try {
    encoder.decrypt(protectedAudio, 0)
  } finally {
    encoder.free()
  }
  await writeFile(
    inputPath,
    Buffer.concat([protectedAudio, musicExFooter(mediaName, songMid)]),
  )
  const controller = new AbortController()
  let requestedMedia:
    | { filename: string; songMid: string }
    | undefined
  let requestedSignal: AbortSignal | undefined
  try {
    await decodeAuthorizedMediaLocally({
      provider: 'qqmusic',
      inputPath,
      outputPath,
      originalName: 'song.mflac',
      signal: controller.signal,
      resolveQmcEkey: async (value, signal) => {
        requestedMedia = value
        requestedSignal = signal
        return ekey
      },
    })
    assert.deepEqual(requestedMedia, {
      filename: mediaName,
      songMid,
    })
    assert.equal(requestedSignal, controller.signal)
    assert.deepEqual(await readFile(outputPath), standardAudio)
    if (process.platform !== 'win32') {
      assert.equal((await stat(outputPath)).mode & 0o777, 0o600)
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('removes a partial local decode after a malformed container fails', async () => {
  const root = resolve(process.cwd(), '.test-authorized-media-invalid')
  const inputPath = resolve(root, 'input.ncm')
  const outputPath = resolve(root, 'output.media')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  await writeFile(inputPath, 'not-an-ncm-container')
  try {
    await assert.rejects(
      decodeAuthorizedMediaLocally({
        provider: 'netease',
        inputPath,
        outputPath,
        originalName: 'broken.ncm',
      }),
      /NCM 文件头无效/u,
    )
    await assert.rejects(stat(outputPath), { code: 'ENOENT' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('explains QMC2 files whose local footer has no embedded key', async () => {
  const root = resolve(process.cwd(), '.test-authorized-media-qmc2-no-key')
  const inputPath = resolve(root, 'input.mgg')
  const outputPath = resolve(root, 'output.media')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  await writeFile(inputPath, Buffer.alloc(8192, 0x5a))
  try {
    await assert.rejects(
      decodeAuthorizedMediaLocally({
        provider: 'qqmusic',
        inputPath,
        outputPath,
        originalName: 'song.mgg',
      }),
      /未识别到完整的 QMC2 文件尾部.*下载可能尚未完成.*尾部可能已损坏/u,
    )
    await assert.rejects(stat(outputPath), { code: 'ENOENT' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
