import { randomUUID } from 'node:crypto'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { KuGou } from '@clamber_l/crypto/inline'

const MAX_KEY_DATABASE_BYTES = 128 * 1024 * 1024
const MAX_MMKV_STRING_BYTES = 64 * 1024

export type SeparatedKeyProvider = 'kuwo' | 'kugou'

const kuwoKeys = new Map<string, string>()
const kugouKeys = new Map<string, string>()

class MmkvParser {
  private offset = 4
  private readonly length: number
  private readonly decoder = new TextDecoder('utf-8', { fatal: true })

  constructor(private readonly buffer: Uint8Array) {
    if (buffer.byteLength < 5) throw new Error('MMKV 文件过短')
    const payloadLength = Buffer.from(
      buffer.buffer,
      buffer.byteOffset,
      4,
    ).readUInt32LE(0)
    this.length = 4 + payloadLength
    if (payloadLength <= 0 || this.length > buffer.byteLength) {
      throw new Error('MMKV 负载长度无效')
    }
    this.readInteger()
  }

  get eof(): boolean {
    return this.offset >= this.length
  }

  private readByte(): number {
    if (this.offset >= this.length) throw new Error('MMKV 数据提前结束')
    return this.buffer[this.offset++]
  }

  private readInteger(): number {
    let value = 0n
    let shift = 0n
    for (let count = 0; count < 10; count += 1) {
      const byte = this.readByte()
      value |= BigInt(byte & 0x7f) << shift
      if ((byte & 0x80) === 0) {
        if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
          throw new Error('MMKV 整数超过安全范围')
        }
        return Number(value)
      }
      shift += 7n
    }
    throw new Error('MMKV 变长整数无效')
  }

  private readBytes(length: number): Uint8Array {
    if (
      length < 0 ||
      length > MAX_MMKV_STRING_BYTES ||
      this.offset + length > this.length
    ) {
      throw new Error('MMKV 字段长度无效')
    }
    const value = this.buffer.subarray(this.offset, this.offset + length)
    this.offset += length
    return value
  }

  readString(): string {
    const length = this.readInteger()
    return this.decoder.decode(this.readBytes(length)).normalize()
  }

  readStringValue(): string | undefined {
    const containerLength = this.readInteger()
    if (containerLength === 0) return undefined
    if (
      containerLength < 0 ||
      this.offset + containerLength > this.length
    ) {
      throw new Error('MMKV 容器长度无效')
    }
    const expectedEnd = this.offset + containerLength
    const value = this.readString()
    if (this.offset !== expectedEnd) {
      throw new Error('MMKV 字符串容器结构无效')
    }
    return value
  }

  skipContainer(): void {
    const containerLength = this.readInteger()
    if (
      containerLength < 0 ||
      this.offset + containerLength > this.length
    ) {
      throw new Error('MMKV 容器长度无效')
    }
    this.offset += containerLength
  }
}

function validEkey(value: string | undefined): value is string {
  return Boolean(
    value &&
      value.length >= 8 &&
      value.length <= 16 * 1024 &&
      !/[\u0000-\u001f\u007f\s]/u.test(value),
  )
}

function parseKuwoAndroidMmkv(
  buffer: Uint8Array,
): Array<{ key: string; ekey: string }> {
  const parser = new MmkvParser(buffer)
  const result: Array<{ key: string; ekey: string }> = []
  while (!parser.eof) {
    const name = parser.readString()
    const match = /^sec_ekey#(\d+)-(\w+)$/u.exec(name)
    if (!match) {
      parser.skipContainer()
      continue
    }
    const ekey = parser.readStringValue()
    if (validEkey(ekey)) {
      result.push({
        key: `${match[1]}-${match[2].replaceAll(/\D/gu, '')}`,
        ekey,
      })
    }
  }
  return result
}

function parseKuwoIosMmkv(
  buffer: Uint8Array,
): Array<{ key: string; ekey: string }> {
  const parser = new MmkvParser(buffer)
  const result: Array<{ key: string; ekey: string }> = []
  while (!parser.eof) {
    const name = parser.readString()
    const match = /^(\d+)_(\d+)$/u.exec(name)
    if (!match) {
      parser.skipContainer()
      continue
    }
    const ekey = parser.readStringValue()
    if (validEkey(ekey)) {
      result.push({ key: `${match[1]}-${match[2]}`, ekey })
    }
  }
  return result
}

function parseKugouAndroidMmkv(
  buffer: Uint8Array,
): Array<{ key: string; ekey: string }> {
  const parser = new MmkvParser(buffer)
  const result: Array<{ key: string; ekey: string }> = []
  while (!parser.eof) {
    const audioHash = parser.readString()
    if (!/^[a-f0-9]{32}$/iu.test(audioHash)) {
      parser.skipContainer()
      continue
    }
    const ekey = parser.readStringValue()
    if (validEkey(ekey)) {
      result.push({ key: audioHash.toLowerCase(), ekey })
    }
  }
  return result
}

async function parseKugouDesktopDatabase(
  source: Uint8Array,
): Promise<Array<{ key: string; ekey: string }>> {
  const buffer = Uint8Array.from(source)
  KuGou.decryptDatabase(buffer)
  if (
    Buffer.from(buffer.subarray(0, 16)).toString('ascii') !==
    'SQLite format 3\0'
  ) {
    throw new Error('酷狗密钥数据库解密后不是有效 SQLite 文件')
  }
  let database:
    | {
        prepare(sql: string): {
          all(...values: unknown[]): Array<Record<string, unknown>>
          get(...values: unknown[]): Record<string, unknown> | undefined
        }
        close(): void
      }
    | undefined
  const temporaryRoot = resolve(
    process.env.GTS_DATA_ROOT ?? process.cwd(),
    'media-key-import',
  )
  const temporaryPath = resolve(
    temporaryRoot,
    `${randomUUID()}.sqlite`,
  )
  try {
    const { DatabaseSync } = await import('node:sqlite')
    await mkdir(temporaryRoot, { recursive: true, mode: 0o700 })
    await writeFile(temporaryPath, buffer, { flag: 'wx', mode: 0o600 })
    database = new DatabaseSync(temporaryPath, {
      readOnly: true,
    })
    const tableExists = (name: string) =>
      Boolean(
        database!
          .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
          )
          .get(name),
      )
    const sources: string[] = []
    if (tableExists('ShareFileItems')) {
      sources.push(
        'SELECT EncryptionKeyId AS H, EncryptionKey AS K FROM ShareFileItems',
      )
    }
    if (tableExists('DownloadItem')) {
      sources.push('SELECT EnHash AS H, EnKey AS K FROM DownloadItem')
    }
    if (sources.length === 0) {
      throw new Error(
        '酷狗密钥数据库缺少 ShareFileItems 或 DownloadItem 表',
      )
    }
    const rows = database
      .prepare(`
        SELECT H AS audioHash, K AS ekey FROM (${sources.join(
          ' UNION ALL ',
        )})
        WHERE H IS NOT NULL AND H != '' AND K IS NOT NULL AND K != ''
        GROUP BY H
      `)
      .all()
    return rows.flatMap((row) => {
      const key = String(row.audioHash ?? '').toLowerCase()
      const ekey = String(row.ekey ?? '')
      return /^[a-f0-9]{32}$/u.test(key) && validEkey(ekey)
        ? [{ key, ekey }]
        : []
    })
  } catch (error) {
    if (
      error instanceof Error &&
      error.message.includes('node:sqlite')
    ) {
      throw new Error('当前桌面运行时不支持读取酷狗 SQLite 密钥库')
    }
    throw error
  } finally {
    database?.close()
    await rm(temporaryPath, { force: true }).catch(() => undefined)
  }
}

function firstSuccessfulParse(
  parsers: Array<() => Array<{ key: string; ekey: string }>>,
): Array<{ key: string; ekey: string }> {
  let lastError: unknown
  for (const parse of parsers) {
    try {
      const entries = parse()
      if (entries.length > 0) return entries
    } catch (error) {
      lastError = error
    }
  }
  if (lastError instanceof Error) throw lastError
  return []
}

export async function importAuthorizedMediaKeyDatabase(
  provider: SeparatedKeyProvider,
  bytes: Uint8Array,
): Promise<number> {
  if (bytes.byteLength <= 0 || bytes.byteLength > MAX_KEY_DATABASE_BYTES) {
    throw new Error('密钥库为空或超过 128 MB 安全限制')
  }

  let entries: Array<{ key: string; ekey: string }>
  let target: Map<string, string>
  if (provider === 'kuwo') {
    entries = firstSuccessfulParse([
      () => parseKuwoAndroidMmkv(bytes),
      () => parseKuwoIosMmkv(bytes),
    ])
    target = kuwoKeys
  } else {
    try {
      entries = parseKugouAndroidMmkv(bytes)
    } catch {
      entries = []
    }
    if (entries.length === 0) {
      entries = await parseKugouDesktopDatabase(bytes)
    }
    target = kugouKeys
  }

  if (entries.length === 0) {
    throw new Error(
      provider === 'kuwo'
        ? '密钥库中没有可识别的 KWM v2 密钥'
        : '密钥库中没有可识别的 KGM v5 密钥',
    )
  }
  target.clear()
  for (const { key, ekey } of entries) target.set(key, ekey)
  return target.size
}

export function resolveKuwoV2Ekey(
  resourceId: number,
  qualityId: number,
): string | undefined {
  return kuwoKeys.get(`${resourceId}-${qualityId}`)
}

export function resolveKugouV5Ekey(audioHash: string): string | undefined {
  return kugouKeys.get(audioHash.toLowerCase())
}

export function clearAuthorizedMediaKeys(): void {
  kuwoKeys.clear()
  kugouKeys.clear()
}
