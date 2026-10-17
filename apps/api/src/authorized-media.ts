import { createDecipheriv } from 'node:crypto'
import { open, rm, type FileHandle } from 'node:fs/promises'
import type { AuthorizedMediaProvider } from '@gts/core'
import {
  KWMDecipher,
  KuGou,
  KuGouHeader,
  KuwoHeader,
  QMC2,
  QMCFooter,
  decryptQMC1,
  ready,
} from '@clamber_l/crypto/inline'
import {
  parseQQMusicMusicExFooter,
  resolveQQMusicEkeyForDesktop,
  type QQMusicMediaIdentity,
} from './qqmusic-ekey.js'
import {
  resolveKugouV5Ekey,
  resolveKuwoV2Ekey,
} from './authorized-media-keys.js'

const MAX_DECODED_BYTES = 2 * 1024 * 1024 * 1024
const BLOCK_BYTES = 1024 * 1024
const MAX_HEADER_BYTES = 64 * 1024 * 1024
const NCM_CORE_KEY = Buffer.from(
  '687a4852416d736f356b496e62617857',
  'hex',
)
const NCM_KEY_PREFIX = Buffer.from('neteasecloudmusic')
const KGM_MAGIC = Buffer.from(
  '7cd532eb86027f4ba8afa68e0fff9914',
  'hex',
)
const VPR_MAGIC = Buffer.from(
  '0528bc96e9e45a4391aabdd07af53631',
  'hex',
)
const QMC2_EXTENSIONS = new Set([
  '.mflac',
  '.mflac0',
  '.mgg',
  '.mgg0',
  '.mgg1',
])

export interface AuthorizedMediaDecodeRequest {
  provider: AuthorizedMediaProvider
  outputPath: string
  originalName: string
  inputPath: string
  resolveQmcEkey?: (
    media: QQMusicMediaIdentity,
    signal?: AbortSignal,
  ) => Promise<string>
  signal?: AbortSignal
}

function fileExtension(filename: string): string {
  const basename = filename.split(/[\\/]/u).at(-1) ?? ''
  const dot = basename.lastIndexOf('.')
  return dot <= 0 ? '' : basename.slice(dot).toLowerCase()
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error('任务已取消')
}

function isKwmHeader(header: Uint8Array): boolean {
  const magic = Buffer.from(
    header.buffer,
    header.byteOffset,
    Math.min(16, header.byteLength),
  ).toString('binary')
  return (
    magic === 'yeelion-kuwo-tme' ||
    magic === 'yeelion-kuwo\0\0\0\0'
  )
}

function isKgmHeader(header: Uint8Array): boolean {
  if (header.byteLength < 16) return false
  const magic = Buffer.from(
    header.buffer,
    header.byteOffset,
    16,
  )
  return magic.equals(KGM_MAGIC) || magic.equals(VPR_MAGIC)
}

async function readSlice(
  handle: FileHandle,
  position: number,
  length: number,
): Promise<Uint8Array> {
  const buffer = Buffer.allocUnsafe(length)
  const { bytesRead } = await handle.read(buffer, 0, length, position)
  return new Uint8Array(
    buffer.buffer,
    buffer.byteOffset,
    bytesRead,
  )
}

async function readUint32le(
  handle: FileHandle,
  position: number,
): Promise<number> {
  const bytes = await readSlice(handle, position, 4)
  if (bytes.byteLength !== 4) throw new Error('受保护媒体文件头不完整')
  return Buffer.from(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  ).readUInt32LE(0)
}

async function writeAll(
  handle: FileHandle,
  buffer: Uint8Array,
  position: number,
): Promise<void> {
  let written = 0
  while (written < buffer.byteLength) {
    const result = await handle.write(
      buffer,
      written,
      buffer.byteLength - written,
      position + written,
    )
    if (result.bytesWritten <= 0) {
      throw new Error('本地解码输出写入失败')
    }
    written += result.bytesWritten
  }
}

async function decodeRange(
  input: FileHandle,
  output: FileHandle,
  start: number,
  end: number,
  decrypt: (buffer: Uint8Array, offset: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const decodedBytes = end - start
  if (decodedBytes <= 0) throw new Error('受保护媒体不包含音频数据')
  if (decodedBytes > MAX_DECODED_BYTES) {
    throw new Error('本地解码结果超过 2 GB 安全限制')
  }

  let inputPosition = start
  let outputPosition = 0
  while (inputPosition < end) {
    throwIfAborted(signal)
    const requested = Math.min(BLOCK_BYTES, end - inputPosition)
    const chunk = await readSlice(input, inputPosition, requested)
    if (chunk.byteLength === 0) {
      throw new Error('受保护媒体读取提前结束')
    }
    decrypt(chunk, inputPosition - start)
    await writeAll(output, chunk, outputPosition)
    inputPosition += chunk.byteLength
    outputPosition += chunk.byteLength
  }
}

async function decodeNcm(
  input: FileHandle,
  output: FileHandle,
  fileSize: number,
  signal?: AbortSignal,
): Promise<void> {
  const prefix = await readSlice(input, 0, 14)
  if (
    prefix.byteLength !== 14 ||
    Buffer.from(prefix.subarray(0, 8)).toString('ascii') !== 'CTENFDAM'
  ) {
    throw new Error('NCM 文件头无效')
  }
  const encryptedKeyLength = Buffer.from(
    prefix.buffer,
    prefix.byteOffset + 10,
    4,
  ).readUInt32LE(0)
  if (
    encryptedKeyLength === 0 ||
    encryptedKeyLength > MAX_HEADER_BYTES ||
    14 + encryptedKeyLength > fileSize
  ) {
    throw new Error('NCM 密钥段长度无效')
  }
  const encryptedKey = await readSlice(input, 14, encryptedKeyLength)
  const unmaskedKey = Buffer.from(
    encryptedKey.map((value) => value ^ 0x64),
  )
  const decipher = createDecipheriv('aes-128-ecb', NCM_CORE_KEY, null)
  const decryptedKey = Buffer.concat([
    decipher.update(unmaskedKey),
    decipher.final(),
  ])
  if (!decryptedKey.subarray(0, NCM_KEY_PREFIX.length).equals(NCM_KEY_PREFIX)) {
    throw new Error('NCM 内容密钥无效')
  }
  const contentKey = decryptedKey.subarray(NCM_KEY_PREFIX.length)
  if (contentKey.byteLength === 0) throw new Error('NCM 内容密钥为空')

  const box = Uint8Array.from({ length: 256 }, (_, index) => index)
  let lastByte = 0
  let keyOffset = 0
  for (let index = 0; index < 256; index += 1) {
    const swap = box[index]
    const next =
      (swap + lastByte + contentKey[keyOffset]) & 0xff
    keyOffset = (keyOffset + 1) % contentKey.byteLength
    box[index] = box[next]
    box[next] = swap
    lastByte = next
  }

  const metadataLengthOffset = 14 + encryptedKeyLength
  const metadataLength = await readUint32le(input, metadataLengthOffset)
  const postMetadataOffset =
    metadataLengthOffset + 4 + metadataLength
  if (
    metadataLength > MAX_HEADER_BYTES ||
    postMetadataOffset + 13 > fileSize
  ) {
    throw new Error('NCM 元数据段长度无效')
  }
  const coverFrameLength = await readUint32le(
    input,
    postMetadataOffset + 5,
  )
  const audioOffset =
    postMetadataOffset + 13 + coverFrameLength
  if (
    coverFrameLength > MAX_HEADER_BYTES ||
    audioOffset >= fileSize
  ) {
    throw new Error('NCM 封面段长度无效')
  }
  await decodeRange(
    input,
    output,
    audioOffset,
    fileSize,
    (buffer, offset) => {
      for (let index = 0; index < buffer.byteLength; index += 1) {
        const position = (offset + index + 1) & 0xff
        buffer[index] ^=
          box[
            (box[position] +
              box[(box[position] + position) & 0xff]) &
              0xff
          ]
      }
    },
    signal,
  )
}

async function decodeQmc(
  input: FileHandle,
  output: FileHandle,
  fileSize: number,
  extension: string,
  resolveQmcEkey:
    | ((
        media: QQMusicMediaIdentity,
        signal?: AbortSignal,
      ) => Promise<string>)
    | undefined,
  signal?: AbortSignal,
): Promise<void> {
  const footerBytes = Math.min(fileSize, 4096)
  const footerBuffer = await readSlice(
    input,
    fileSize - footerBytes,
    footerBytes,
  )
  const footer = QMCFooter.parse(footerBuffer)
  const footerDetected = Boolean(footer)
  let embeddedEkey: string | undefined
  let musicExMedia: QQMusicMediaIdentity | undefined
  let footerSize = 0
  try {
    if (footer) {
      embeddedEkey = footer.ekey
      footerSize = footer.size
      if (!embeddedEkey && footer.mediaName) {
        const start = footerBuffer.byteLength - footerSize
        if (start < 0) {
          throw new Error('QMC2 MusicEx 页脚超过读取窗口')
        }
        musicExMedia = parseQQMusicMusicExFooter(
          footerBuffer.subarray(start),
        )
      }
    }
  } finally {
    footer?.free()
  }

  if (embeddedEkey || musicExMedia) {
    const ekey =
      embeddedEkey ??
      (resolveQmcEkey
        ? await resolveQmcEkey(musicExMedia!, signal)
        : await resolveQQMusicEkeyForDesktop(musicExMedia!, { signal }))
    const decoder = new QMC2(ekey)
    try {
      await decodeRange(
        input,
        output,
        0,
        fileSize - footerSize,
        (buffer, offset) => decoder.decrypt(buffer, offset),
        signal,
      )
    } finally {
      decoder.free()
    }
    return
  }

  if (QMC2_EXTENSIONS.has(extension)) {
    throw new Error(
      footerDetected
        ? '已识别 QMC2 文件尾部，但其中既没有内嵌内容密钥，也没有可用于官方授权验证的 MusicEx 文件名。'
        : '未识别到完整的 QMC2 文件尾部。下载可能尚未完成、文件尾部可能已损坏，或该页脚格式暂不受支持；请在原客户端确认下载完整后重试。',
    )
  }
  await decodeRange(
    input,
    output,
    0,
    fileSize,
    decryptQMC1,
    signal,
  )
}

async function decodeKwm(
  input: FileHandle,
  output: FileHandle,
  fileSize: number,
  signal?: AbortSignal,
): Promise<void> {
  const dataOffset = 0x400
  if (fileSize <= dataOffset) throw new Error('KWM 文件长度无效')
  const headerBytes = await readSlice(input, 0, dataOffset)
  const header = KuwoHeader.parse(headerBytes)
  let decoder: KWMDecipher | undefined
  try {
    try {
      decoder = new KWMDecipher(header)
    } catch (error) {
      const ekey = resolveKuwoV2Ekey(
        header.resourceId,
        header.qualityId,
      )
      if (!ekey) {
        throw new Error(
          `该 KWM v2 文件需要分离密钥，但尚未导入匹配的酷我密钥库（资源 ${header.resourceId}，音质 ${header.qualityId}）。`,
        )
      }
      decoder = new KWMDecipher(header, ekey)
    }
    await decodeRange(
      input,
      output,
      dataOffset,
      fileSize,
      (buffer, offset) => decoder!.decrypt(buffer, offset),
      signal,
    )
  } finally {
    decoder?.free()
    header.free()
  }
}

async function decodeKgm(
  input: FileHandle,
  output: FileHandle,
  fileSize: number,
  signal?: AbortSignal,
): Promise<void> {
  const headerBytes = await readSlice(input, 0, Math.min(fileSize, 0x400))
  const header = new KuGouHeader(headerBytes)
  let decoder: KuGou | undefined
  try {
    if (header.version === 5) {
      const ekey = resolveKugouV5Ekey(header.audioHash)
      if (!ekey) {
        throw new Error(
          `该 KGM v5 文件需要分离密钥，但尚未导入匹配的酷狗密钥库（音频哈希 ${header.audioHash.slice(0, 8)}…）。`,
        )
      }
      decoder = KuGou.fromHeaderV5(header, ekey)
    } else {
      decoder = KuGou.from_header(headerBytes)
    }
    const dataOffset = header.offsetToData
    await decodeRange(
      input,
      output,
      dataOffset,
      fileSize,
      (buffer, offset) => decoder!.decrypt(buffer, offset),
      signal,
    )
  } finally {
    decoder?.free()
    header.free()
  }
}

export function localAuthorizedMediaDecoderAvailable(): boolean {
  return typeof WebAssembly !== 'undefined'
}

export async function decodeAuthorizedMediaLocally(
  request: AuthorizedMediaDecodeRequest,
): Promise<void> {
  throwIfAborted(request.signal)
  await ready
  const input = await open(request.inputPath, 'r')
  let output: FileHandle | undefined
  try {
    const { size } = await input.stat()
    if (size <= 0) throw new Error('受保护媒体为空文件')
    output = await open(request.outputPath, 'wx', 0o600)
    const extension = fileExtension(request.originalName)
    const containerHeader = await readSlice(
      input,
      0,
      Math.min(size, 0x400),
    )
    if (
      extension === '.ncm' ||
      Buffer.from(containerHeader.subarray(0, 8)).toString('ascii') ===
        'CTENFDAM'
    ) {
      await decodeNcm(input, output, size, request.signal)
    } else if (
      extension === '.kwm' ||
      isKwmHeader(containerHeader)
    ) {
      await decodeKwm(input, output, size, request.signal)
    } else if (
      extension === '.kgg' ||
      extension === '.kgm' ||
      extension === '.kgma' ||
      extension === '.vpr' ||
      isKgmHeader(containerHeader)
    ) {
      await decodeKgm(input, output, size, request.signal)
    } else {
      await decodeQmc(
        input,
        output,
        size,
        extension,
        request.resolveQmcEkey,
        request.signal,
      )
    }
  } catch (error) {
    await output?.close().catch(() => undefined)
    output = undefined
    await rm(request.outputPath, { force: true }).catch(() => undefined)
    throw error
  } finally {
    await output?.close().catch(() => undefined)
    await input.close()
  }
}
