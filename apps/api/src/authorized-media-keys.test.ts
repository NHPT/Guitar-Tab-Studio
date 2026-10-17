import assert from 'node:assert/strict'
import { mkdir, readFile, rm } from 'node:fs/promises'
import { resolve } from 'node:path'
import test from 'node:test'
import { KuGou } from '@clamber_l/crypto/inline'
import {
  clearAuthorizedMediaKeys,
  importAuthorizedMediaKeyDatabase,
  resolveKugouV5Ekey,
  resolveKuwoV2Ekey,
} from './authorized-media-keys.js'

function variableInteger(value: number): Buffer {
  const bytes: number[] = []
  let remaining = value
  do {
    let byte = remaining & 0x7f
    remaining >>>= 7
    if (remaining > 0) byte |= 0x80
    bytes.push(byte)
  } while (remaining > 0)
  return Buffer.from(bytes)
}

function encodedString(value: string): Buffer {
  const bytes = Buffer.from(value, 'utf8')
  return Buffer.concat([variableInteger(bytes.byteLength), bytes])
}

function stringEntry(key: string, value: string): Buffer {
  const encodedValue = encodedString(value)
  return Buffer.concat([
    encodedString(key),
    variableInteger(encodedValue.byteLength),
    encodedValue,
  ])
}

function mmkvFixture(entries: Array<[string, string]>): Buffer {
  const payload = Buffer.concat([
    variableInteger(0),
    ...entries.map(([key, value]) => stringEntry(key, value)),
  ])
  const length = Buffer.alloc(4)
  length.writeUInt32LE(payload.byteLength)
  return Buffer.concat([length, payload])
}

test('imports Kuwo Android MMKV keys by resource and quality id', async () => {
  clearAuthorizedMediaKeys()
  const ekey = 'kuwo-fixture-ekey-123456'
  const count = await importAuthorizedMediaKeyDatabase(
    'kuwo',
    mmkvFixture([
      ['unrelated', 'ignored'],
      ['sec_ekey#123456-p4000', ekey],
    ]),
  )
  assert.equal(count, 1)
  assert.equal(resolveKuwoV2Ekey(123456, 4000), ekey)
  clearAuthorizedMediaKeys()
})

test('imports Kugou Android MMKV keys by audio hash', async () => {
  clearAuthorizedMediaKeys()
  const audioHash = 'AC89EC47A70B76F307CB39A0D74BCCB0'
  const ekey = 'kugou-fixture-ekey-123456'
  const count = await importAuthorizedMediaKeyDatabase(
    'kugou',
    mmkvFixture([
      ['unrelated', 'ignored'],
      [audioHash, ekey],
    ]),
  )
  assert.equal(count, 1)
  assert.equal(resolveKugouV5Ekey(audioHash), ekey)
  clearAuthorizedMediaKeys()
})

test('imports Kugou desktop encrypted SQLite key databases', async (context) => {
  let DatabaseSync: typeof import('node:sqlite').DatabaseSync
  try {
    ;({ DatabaseSync } = await import('node:sqlite'))
  } catch {
    context.skip('node:sqlite is unavailable')
    return
  }
  const root = resolve(process.cwd(), '.test-kugou-key-database')
  const path = resolve(root, 'kugou.db')
  const audioHash = 'AC89EC47A70B76F307CB39A0D74BCCB0'
  const ekey = 'kugou-desktop-ekey-123456'
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  const database = new DatabaseSync(path)
  try {
    database.exec(`
      CREATE TABLE ShareFileItems (
        EncryptionKeyId TEXT,
        EncryptionKey TEXT
      );
      CREATE TABLE DownloadItem (
        EnHash TEXT,
        EnKey TEXT
      );
    `)
    database
      .prepare(
        'INSERT INTO ShareFileItems (EncryptionKeyId, EncryptionKey) VALUES (?, ?)',
      )
      .run(audioHash, ekey)
  } finally {
    database.close()
  }
  try {
    const encrypted = Uint8Array.from(await readFile(path))
    KuGou.decryptDatabase(encrypted)
    clearAuthorizedMediaKeys()
    const count = await importAuthorizedMediaKeyDatabase(
      'kugou',
      encrypted,
    )
    assert.equal(count, 1)
    assert.equal(resolveKugouV5Ekey(audioHash), ekey)
  } finally {
    clearAuthorizedMediaKeys()
    await rm(root, { recursive: true, force: true })
  }
})

test('rejects malformed separated-key databases', async () => {
  clearAuthorizedMediaKeys()
  await assert.rejects(
    importAuthorizedMediaKeyDatabase(
      'kuwo',
      Buffer.from('not-an-mmkv-database'),
    ),
    /MMKV/u,
  )
})
