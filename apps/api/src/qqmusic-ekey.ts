import { spawn } from 'node:child_process'
import { lstat, readFile, readdir } from 'node:fs/promises'
import {
  extname,
  isAbsolute,
  resolve as resolvePath,
} from 'node:path'
import plist from 'plist'

const QQMUSIC_EKEY_ENDPOINT =
  'https://u.y.qq.com/cgi-bin/musicu.fcg'
const MAX_PLIST_BYTES = 2 * 1024 * 1024
const MAX_RESPONSE_BYTES = 1024 * 1024
const REQUEST_TIMEOUT_MS = 10_000
const MAX_SERIALIZED_VALUE_BYTES = 128 * 1024
const MAX_PLUTIL_ERROR_BYTES = 32 * 1024
const PLUTIL_TIMEOUT_MS = 10_000
const MAX_WINDOWS_CREDENTIAL_BYTES = 2 * 1024 * 1024
const MAX_WINDOWS_CREDENTIAL_FILES = 128
const MAX_WINDOWS_CREDENTIAL_DEPTH = 5
const WINDOWS_REGISTRY_TIMEOUT_MS = 5_000
const WINDOWS_CREDENTIAL_EXTENSIONS = new Set([
  '.cfg',
  '.config',
  '.dat',
  '.ini',
  '.json',
  '.ldb',
  '.log',
  '.txt',
  '.xml',
])
const WINDOWS_SKIPPED_DIRECTORIES = new Set([
  'cache',
  'download',
  'downloads',
  'image',
  'images',
  'music',
  'song',
  'songs',
  'temp',
  'tmp',
])

const musicIdKeys = new Set([
  'musicid',
  'strmusicid',
  'uin',
  'qq',
  'qqmusicuin',
  'nuserid',
  'struseraccount',
])
const musicKeyKeys = new Set([
  'musickey',
  'authst',
  'strauthst',
  'qmkeyst',
  'qqmusickey',
])
const loginTypeKeys = new Set(['logintype', 'tmelogintype'])

export interface QQMusicCredential {
  musicId: string
  musicKey: string
  loginType: 1 | 2 | 3
}

export interface QQMusicMediaIdentity {
  filename: string
  songMid: string
}

export interface QQMusicEkeyRequestOptions {
  credential: QQMusicCredential
  fetchImpl?: typeof fetch
  media: QQMusicMediaIdentity
  platform?: '20' | '27'
  signal?: AbortSignal
}

export interface QQMusicDesktopEkeyOptions {
  env?: NodeJS.ProcessEnv
  fetchImpl?: typeof fetch
  loadCredential?: (
    homePath: string,
  ) => Promise<QQMusicCredential | undefined>
  platform?: NodeJS.Platform
  signal?: AbortSignal
}

function normalizedKey(value: string): string {
  return value.toLowerCase().replaceAll(/[^a-z0-9]/gu, '')
}

function scalarText(value: unknown): string | undefined {
  if (typeof value === 'string') return value.trim()
  if (
    typeof value === 'number' &&
    Number.isSafeInteger(value) &&
    value > 0
  ) {
    return String(value)
  }
  return undefined
}

function validMusicId(value: unknown): string | undefined {
  const text = scalarText(value)
  return text && /^\d{4,24}$/u.test(text) ? text : undefined
}

function validMusicKey(value: unknown): string | undefined {
  const text = scalarText(value)
  if (
    !text ||
    text.length < 8 ||
    text.length > 4096 ||
    /[\u0000-\u001f\u007f\s]/u.test(text)
  ) {
    return undefined
  }
  return text
}

function directCredential(
  value: Record<string, unknown>,
): QQMusicCredential | undefined {
  let musicId: string | undefined
  let musicKey: string | undefined
  let loginType: 1 | 2 | 3 | undefined
  for (const [key, fieldValue] of Object.entries(value)) {
    const normalized = normalizedKey(key)
    if (!musicId && musicIdKeys.has(normalized)) {
      musicId = validMusicId(fieldValue)
    }
    if (!musicKey && musicKeyKeys.has(normalized)) {
      musicKey = validMusicKey(fieldValue)
    }
    if (loginTypeKeys.has(normalized)) {
      const candidate = Number(scalarText(fieldValue))
      if (candidate === 1 || candidate === 2 || candidate === 3) {
        loginType = candidate
      }
    }
  }
  if (!musicId || !musicKey) return undefined
  return {
    musicId,
    musicKey,
    loginType: loginType ?? (musicKey.startsWith('W_X') ? 1 : 2),
  }
}

function keyValueCredential(value: string): QQMusicCredential | undefined {
  const record: Record<string, string> = {}
  for (const part of value.split(/[;&\n\r]/u)) {
    const separator = part.indexOf('=')
    if (separator <= 0) continue
    const key = part.slice(0, separator).trim()
    const rawValue = part.slice(separator + 1).trim()
    try {
      record[key] = decodeURIComponent(rawValue)
    } catch {
      record[key] = rawValue
    }
  }
  return directCredential(record)
}

function serializedValues(value: string): unknown[] {
  const trimmed = value.trim()
  if (
    trimmed.length === 0 ||
    Buffer.byteLength(trimmed, 'utf8') > MAX_SERIALIZED_VALUE_BYTES
  ) {
    return []
  }

  const parsed: unknown[] = []
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      parsed.push(JSON.parse(trimmed))
    } catch {
      // Continue with the base64 form used by plist data values.
    }
  }
  if (
    trimmed.length >= 16 &&
    /^[a-z0-9+/_=-]+$/iu.test(trimmed)
  ) {
    try {
      const decoded = Buffer.from(
        trimmed.replaceAll('-', '+').replaceAll('_', '/'),
        'base64',
      ).toString('utf8')
      if (
        decoded.length > 0 &&
        !decoded.includes('\u0000') &&
        Buffer.byteLength(decoded, 'utf8') <= MAX_SERIALIZED_VALUE_BYTES
      ) {
        parsed.push(decoded)
      }
    } catch {
      // Ignore malformed serialized values.
    }
  }
  return parsed
}

export function parseQQMusicCredential(
  value: unknown,
): QQMusicCredential | undefined {
  const seen = new Set<unknown>()

  function visit(candidate: unknown, depth: number): QQMusicCredential | undefined {
    if (depth > 8 || candidate === null || candidate === undefined) {
      return undefined
    }
    if (typeof candidate === 'string') {
      const direct = keyValueCredential(candidate)
      if (direct) return direct
      for (const nested of serializedValues(candidate)) {
        const found = visit(nested, depth + 1)
        if (found) return found
      }
      return undefined
    }
    if (typeof candidate !== 'object' || seen.has(candidate)) {
      return undefined
    }
    seen.add(candidate)
    if (Array.isArray(candidate)) {
      for (const nested of candidate) {
        const found = visit(nested, depth + 1)
        if (found) return found
      }
      return undefined
    }

    const record = candidate as Record<string, unknown>
    const direct = directCredential(record)
    if (direct) return direct
    const entries = Object.entries(record).sort(([left], [right]) => {
      const leftPriority = normalizedKey(left).includes('autologinuserinfo')
        ? 0
        : 1
      const rightPriority = normalizedKey(right).includes('autologinuserinfo')
        ? 0
        : 1
      return leftPriority - rightPriority
    })
    for (const [, nested] of entries) {
      const found = visit(nested, depth + 1)
      if (found) return found
    }
    return undefined
  }

  return visit(value, 0)
}

function archivedScalar(
  value: unknown,
  objects: unknown[],
  depth = 0,
): unknown {
  if (depth > 8) return undefined
  const record = objectValue(value)
  if (!record) return value
  const uid = scalarText(record['CF$UID'])
  if (uid && /^\d+$/u.test(uid)) {
    return archivedScalar(objects[Number(uid)], objects, depth + 1)
  }
  return value
}

export function parseQQMusicCredentialArchive(
  value: unknown,
): QQMusicCredential | undefined {
  const root = objectValue(value)
  const objects = Array.isArray(root?.$objects) ? root.$objects : undefined
  if (!objects) return parseQQMusicCredential(value)

  for (const candidate of objects) {
    const record = objectValue(candidate)
    if (!record || !('strAuthst' in record)) continue
    const musicKey = validMusicKey(
      archivedScalar(record.strAuthst, objects),
    )
    const musicId = validMusicId(
      archivedScalar(
        record.nUserId ?? record.strUserAccount,
        objects,
      ),
    )
    const rawLoginType = Number(
      scalarText(archivedScalar(record.loginType, objects)) ?? '3',
    )
    const loginType =
      rawLoginType === 1 || rawLoginType === 2 || rawLoginType === 3
        ? rawLoginType
        : 3
    if (musicId && musicKey) {
      return { musicId, musicKey, loginType }
    }
  }
  return undefined
}

export function qqMusicMacCredentialPaths(homePath: string): string[] {
  if (!isAbsolute(homePath)) return []
  return [
    resolvePath(
      homePath,
      'Library',
      'Containers',
      'com.tencent.QQMusicMac',
      'Data',
      'Library',
      'Preferences',
      'com.tencent.QQMusicMac.plist',
    ),
    resolvePath(
      homePath,
      'Library',
      'Preferences',
      'com.tencent.QQMusicMac.plist',
    ),
  ]
}

export function qqMusicWindowsCredentialRoots(
  homePath: string,
): string[] {
  if (!isAbsolute(homePath)) return []
  return [
    resolvePath(
      homePath,
      'AppData',
      'Roaming',
      'Tencent',
      'QQMusic',
    ),
    resolvePath(
      homePath,
      'AppData',
      'Local',
      'Tencent',
      'QQMusic',
    ),
  ]
}

class PlutilCommandError extends Error {
  constructor(
    readonly exitCode: number | null,
    readonly details: string,
  ) {
    super('plutil failed')
  }
}

function runPlutil(
  argumentsList: string[],
  input?: Uint8Array,
): Promise<string> {
  return new Promise((resolveValue, reject) => {
    const child = spawn('/usr/bin/plutil', argumentsList, {
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let stdoutBytes = 0
    let stderrBytes = 0
    let forcedError: Error | undefined
    const timer = setTimeout(() => {
      forcedError = new Error('plutil timed out')
      child.kill('SIGKILL')
    }, PLUTIL_TIMEOUT_MS)

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.byteLength
      if (stdoutBytes > MAX_PLIST_BYTES) {
        forcedError = new Error('plist exceeds the 2 MB safety limit')
        child.kill('SIGKILL')
        return
      }
      stdout.push(chunk)
    })
    child.stderr.on('data', (chunk: Buffer) => {
      if (stderrBytes >= MAX_PLUTIL_ERROR_BYTES) return
      const remaining = MAX_PLUTIL_ERROR_BYTES - stderrBytes
      const bounded = chunk.subarray(0, remaining)
      stderrBytes += bounded.byteLength
      stderr.push(bounded)
    })
    child.once('error', (error) => {
      clearTimeout(timer)
      reject(error)
    })
    child.once('close', (code) => {
      clearTimeout(timer)
      if (forcedError) {
        reject(forcedError)
      } else if (code !== 0) {
        reject(
          new PlutilCommandError(
            code,
            Buffer.concat(stderr).toString('utf8'),
          ),
        )
      } else {
        resolveValue(Buffer.concat(stdout).toString('utf8'))
      }
    })
    child.stdin.end(input)
  })
}

function plistFailureKind(
  error: unknown,
): 'missing' | 'permission' | 'invalid' {
  const details =
    error instanceof PlutilCommandError
      ? error.details.toLowerCase()
      : error instanceof Error
        ? error.message.toLowerCase()
        : ''
  if (
    details.includes('operation not permitted') ||
    details.includes('permission denied') ||
    details.includes('not authorized') ||
    (error instanceof Error &&
      (error as NodeJS.ErrnoException).code === 'EACCES')
  ) {
    return 'permission'
  }
  if (
    details.includes('no such file') ||
    details.includes('does not exist') ||
    details.includes('could not open file')
  ) {
    return 'missing'
  }
  return 'invalid'
}

async function readPlist(path: string): Promise<unknown> {
  const xml = await runPlutil(['-convert', 'xml1', '-o', '-', path])
  return plist.parse(xml)
}

async function readArchivedPlist(data: Uint8Array): Promise<unknown> {
  const xml = await runPlutil(
    ['-convert', 'xml1', '-o', '-', '-'],
    data,
  )
  return plist.parse(xml)
}

function autoLoginArchive(value: unknown): Uint8Array | undefined {
  const outer = objectValue(value)
  const archive = outer?.AutoLoginUserInfo
  if (archive instanceof Uint8Array) return archive
  if (
    typeof archive === 'string' &&
    archive.length > 0 &&
    /^[a-z0-9+/=_-]+$/iu.test(archive)
  ) {
    return Buffer.from(
      archive.replaceAll('-', '+').replaceAll('_', '/'),
      'base64',
    )
  }
  return undefined
}

export async function loadQQMusicMacCredential(
  homePath: string,
): Promise<QQMusicCredential | undefined> {
  let permissionDenied = false
  let loginFileFound = false
  let unsupportedFormat = false
  for (const path of qqMusicMacCredentialPaths(homePath)) {
    let outer: unknown
    try {
      outer = await readPlist(path)
      loginFileFound = true
    } catch (error) {
      const kind = plistFailureKind(error)
      if (kind === 'permission') permissionDenied = true
      if (kind === 'invalid') unsupportedFormat = true
      continue
    }
    const direct = parseQQMusicCredential(outer)
    if (direct) return direct
    const archive = autoLoginArchive(outer)
    if (!archive) continue
    try {
      const credential = parseQQMusicCredentialArchive(
        await readArchivedPlist(archive),
      )
      if (credential) return credential
      unsupportedFormat = true
    } catch (error) {
      if (plistFailureKind(error) === 'permission') {
        permissionDenied = true
      } else {
        unsupportedFormat = true
      }
    }
  }
  if (permissionDenied) {
    throw new Error(
      'macOS 已阻止读取 QQ 音乐客户端登录信息。请在“系统设置 → 隐私与安全性 → 完全磁盘访问权限”中添加并启用 Guitar Tab Studio，然后彻底退出并重新打开本应用。',
    )
  }
  if (unsupportedFormat) {
    throw new Error(
      '已找到 QQ 音乐客户端配置，但无法解析当前登录信息。请完全退出 QQ 音乐，重新登录下载该文件的账号后再试。',
    )
  }
  if (loginFileFound) {
    throw new Error(
      'QQ 音乐客户端配置中没有当前登录信息。请完全退出 QQ 音乐，重新登录下载该文件的账号后再试。',
    )
  }
  return undefined
}

function windowsCredentialText(
  value: string,
): QQMusicCredential | undefined {
  const text = value.replaceAll('\u0000', '').trim()
  if (!text) return undefined
  const direct = parseQQMusicCredential(text)
  if (direct) return direct

  try {
    const parsed = parseQQMusicCredential(JSON.parse(text))
    if (parsed) return parsed
  } catch {
    // Continue with registry and XML-shaped text.
  }

  const record: Record<string, string> = {}
  for (const line of text.split(/\r?\n/u)) {
    const registry = /^\s*([a-z0-9_.-]+)\s+REG_[A-Z0-9_]+\s+(.+?)\s*$/iu.exec(
      line,
    )
    if (registry) record[registry[1]] = registry[2]
  }
  for (const match of text.matchAll(
    /<([a-z][a-z0-9_.-]*)[^>]*>\s*([^<]+?)\s*<\/\1>/giu,
  )) {
    record[match[1]] = match[2]
  }
  for (const match of text.matchAll(
    /["']?([a-z][a-z0-9_.-]*)["']?\s*[:=]\s*["']([^"'<>;\r\n]+)["']/giu,
  )) {
    record[match[1]] = match[2]
  }
  return parseQQMusicCredential(record)
}

function decodedWindowsCredentialTexts(data: Uint8Array): string[] {
  const buffer = Buffer.from(data)
  const decoded = new Set<string>()
  if (buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xfe) {
    decoded.add(buffer.subarray(2).toString('utf16le'))
  } else if (
    buffer.length >= 2 &&
    buffer[0] === 0xfe &&
    buffer[1] === 0xff
  ) {
    const swapped = Buffer.from(buffer.subarray(2))
    if (swapped.length % 2 !== 0) {
      decoded.add(swapped.subarray(0, swapped.length - 1).swap16().toString('utf16le'))
    } else {
      decoded.add(swapped.swap16().toString('utf16le'))
    }
  }

  const sampleBytes = Math.min(buffer.length, 4096)
  let oddNulls = 0
  for (let index = 1; index < sampleBytes; index += 2) {
    if (buffer[index] === 0) oddNulls += 1
  }
  if (sampleBytes >= 8 && oddNulls >= Math.floor(sampleBytes / 8)) {
    decoded.add(
      buffer
        .subarray(0, buffer.length - (buffer.length % 2))
        .toString('utf16le'),
    )
  }
  decoded.add(buffer.toString('utf8'))
  return [...decoded]
}

function credentialFromWindowsData(
  data: Uint8Array,
): QQMusicCredential | undefined {
  for (const text of decodedWindowsCredentialTexts(data)) {
    const credential = windowsCredentialText(text)
    if (credential) return credential
  }
  return undefined
}

function windowsCredentialFilePriority(name: string): number {
  const normalized = name.toLowerCase()
  if (/login|auth|account|cookie|credential/u.test(normalized)) return 0
  if (/user|profile|session/u.test(normalized)) return 1
  if (/config|setting|preference/u.test(normalized)) return 2
  return 3
}

function supportedWindowsCredentialFile(name: string): boolean {
  return (
    WINDOWS_CREDENTIAL_EXTENSIONS.has(extname(name).toLowerCase()) ||
    /login|auth|account|cookie|credential|profile|session|setting|user/iu.test(
      name,
    )
  )
}

async function windowsCredentialFiles(homePath: string): Promise<{
  files: string[]
  permissionDenied: boolean
}> {
  const queue = qqMusicWindowsCredentialRoots(homePath).map((path) => ({
    depth: 0,
    path,
  }))
  const files: string[] = []
  let permissionDenied = false

  while (queue.length > 0 && files.length < MAX_WINDOWS_CREDENTIAL_FILES) {
    const directory = queue.shift()!
    let entries
    try {
      const stats = await lstat(directory.path)
      if (!stats.isDirectory() || stats.isSymbolicLink()) continue
      entries = await readdir(directory.path, { withFileTypes: true })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EACCES') {
        permissionDenied = true
      }
      continue
    }

    entries.sort(
      (left, right) =>
        windowsCredentialFilePriority(left.name) -
          windowsCredentialFilePriority(right.name) ||
        left.name.localeCompare(right.name),
    )
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue
      const path = resolvePath(directory.path, entry.name)
      if (entry.isDirectory()) {
        if (
          directory.depth < MAX_WINDOWS_CREDENTIAL_DEPTH &&
          !WINDOWS_SKIPPED_DIRECTORIES.has(entry.name.toLowerCase())
        ) {
          queue.push({ depth: directory.depth + 1, path })
        }
        continue
      }
      if (
        entry.isFile() &&
        supportedWindowsCredentialFile(entry.name)
      ) {
        files.push(path)
        if (files.length >= MAX_WINDOWS_CREDENTIAL_FILES) break
      }
    }
  }

  return { files, permissionDenied }
}

function readWindowsRegistryCredential(): Promise<
  QQMusicCredential | undefined
> {
  if (process.platform !== 'win32') return Promise.resolve(undefined)
  return new Promise((resolveValue) => {
    const child = spawn(
      'reg.exe',
      [
        'query',
        'HKCU\\Software\\Tencent\\QQMusic',
        '/s',
      ],
      { stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true },
    )
    const chunks: Buffer[] = []
    let bytes = 0
    let failed = false
    const timer = setTimeout(() => {
      failed = true
      child.kill()
    }, WINDOWS_REGISTRY_TIMEOUT_MS)
    child.stdout.on('data', (chunk: Buffer) => {
      bytes += chunk.byteLength
      if (bytes > MAX_WINDOWS_CREDENTIAL_BYTES) {
        failed = true
        child.kill()
        return
      }
      chunks.push(chunk)
    })
    child.once('error', () => {
      clearTimeout(timer)
      resolveValue(undefined)
    })
    child.once('close', (code) => {
      clearTimeout(timer)
      resolveValue(
        failed || code !== 0
          ? undefined
          : credentialFromWindowsData(Buffer.concat(chunks)),
      )
    })
  })
}

export async function loadQQMusicWindowsCredential(
  homePath: string,
): Promise<QQMusicCredential | undefined> {
  if (!isAbsolute(homePath)) return undefined
  const registryCredential = await readWindowsRegistryCredential()
  if (registryCredential) return registryCredential

  const candidates = await windowsCredentialFiles(homePath)
  let readableFileFound = false
  let unsupportedFormat = false
  for (const path of candidates.files) {
    try {
      const stats = await lstat(path)
      if (
        !stats.isFile() ||
        stats.isSymbolicLink() ||
        stats.size <= 0 ||
        stats.size > MAX_WINDOWS_CREDENTIAL_BYTES
      ) {
        continue
      }
      const data = await readFile(path)
      readableFileFound = true
      const credential = credentialFromWindowsData(data)
      if (credential) return credential
      unsupportedFormat = true
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EACCES') {
        candidates.permissionDenied = true
      }
    }
  }
  if (candidates.permissionDenied) {
    throw new Error(
      'Windows 已阻止读取 QQ 音乐客户端登录信息。请退出 QQ 音乐与本应用，重新打开后再试；若启用了安全软件的受控文件夹访问，请允许 Guitar Tab Studio 读取当前用户的 QQ 音乐配置。',
    )
  }
  if (readableFileFound && unsupportedFormat) {
    throw new Error(
      '已找到 QQ 音乐客户端配置，但无法解析当前登录信息。请将 QQ 音乐更新到最新版，完全退出后重新登录下载该文件的账号，再打开本应用重试。',
    )
  }
  return undefined
}

export function parseQQMusicMediaName(
  mediaName: string,
  songMid?: string,
): QQMusicMediaIdentity {
  const filename = mediaName.split(/[\\/]/u).at(-1)?.trim() ?? ''
  if (
    filename.length === 0 ||
    filename.length > 160 ||
    /[\u0000-\u001f\u007f]/u.test(filename)
  ) {
    throw new Error('QMC2 MusicEx 媒体文件名无效')
  }
  const match = /^(?:AIM0|D0M4|DTM3|F0M0|O4M0|O6M0|O8M0|O8M1|Q0M0|Q0M1|Q0M3|TLM1|V0M0)([a-z0-9]{8,64})\.(?:mflac|mgg|mmp4|mnac)$/iu.exec(
    filename,
  )
  if (!match) {
    throw new Error('QMC2 MusicEx 文件名不包含可识别的媒体 MID')
  }
  let resolvedSongMid = songMid
  if (!resolvedSongMid) {
    resolvedSongMid = match[1]
    if (
      resolvedSongMid.length % 2 === 0 &&
      resolvedSongMid.slice(0, resolvedSongMid.length / 2) ===
        resolvedSongMid.slice(resolvedSongMid.length / 2)
    ) {
      resolvedSongMid = resolvedSongMid.slice(
        0,
        resolvedSongMid.length / 2,
      )
    }
  }
  if (!/^00[a-z0-9]{12}$/iu.test(resolvedSongMid)) {
    throw new Error('QMC2 MusicEx 页脚中的歌曲 MID 无效')
  }
  return { filename, songMid: resolvedSongMid }
}

function utf16leField(
  buffer: Uint8Array,
  offset: number,
  length: number,
): string {
  const end = offset + length
  if (offset < 0 || end > buffer.byteLength || length % 2 !== 0) {
    throw new Error('QMC2 MusicEx 页脚字段越界')
  }
  const values: number[] = []
  for (let index = offset; index < end; index += 2) {
    const value = buffer[index] | (buffer[index + 1] << 8)
    if (value === 0) break
    values.push(value)
  }
  return String.fromCharCode(...values)
}

export function parseQQMusicMusicExFooter(
  footer: Uint8Array,
): QQMusicMediaIdentity {
  if (
    footer.byteLength < 192 ||
    Buffer.from(footer.subarray(footer.byteLength - 8)).toString('ascii') !==
      'musicex\0'
  ) {
    throw new Error('QMC2 MusicEx 页脚无效')
  }
  const songMid = utf16leField(footer, 0x0c, 60)
  const filename = utf16leField(
    footer,
    0x48,
    Math.min(100, footer.byteLength - 0x48 - 16),
  )
  return parseQQMusicMediaName(filename, songMid)
}

export function buildQQMusicEkeyRequest(
  media: QQMusicMediaIdentity,
  credential: QQMusicCredential,
  guid = '10000',
  platform: '20' | '27' = '20',
): Record<string, unknown> {
  return {
    comm: {
      authst: credential.musicKey,
      ct: '19',
      cv: '1859',
      uin: credential.musicId,
      tmeLoginType: String(credential.loginType),
    },
    req_1: {
      module: 'music.vkey.GetEVkey',
      method: 'CgiGetEVkey',
      param: {
        filename: [media.filename],
        guid,
        songmid: [media.songMid],
        songtype: [1],
        uin: credential.musicId,
        loginflag: 1,
        platform,
        ctx: 1,
      },
    },
  }
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function officialAuthorizationError(result: number): Error {
  if (result === 101404) {
    return new Error(
      'QQ 音乐官方接口未接受文件中的歌曲标识。请确认文件下载完整，并使用下载该文件的同一账号重新下载后重试。',
    )
  }
  if (result === 104003) {
    return new Error(
      'QQ 音乐官方接口未授权当前账号访问该曲目或音质。请确认该账号已购买、已开通相应会员且文件由同一账号下载。',
    )
  }
  if (result === 104013) {
    return new Error(
      'QQ 音乐官方接口提示播放设备数量受限，请在 QQ 音乐中释放其他设备后重试。',
    )
  }
  if (result === 104004) {
    return new Error(
      'QQ 音乐官方接口未能为该文件签发内容密钥，请确认下载完整并重新下载后重试。',
    )
  }
  if (result === 405) {
    return new Error(
      'QQ 音乐客户端登录凭据已过期。请完全退出 QQ 音乐，重新登录下载该文件的账号后再试。',
    )
  }
  return new Error(`QQ 音乐官方授权验证失败（代码 ${result}）`)
}

async function boundedResponseText(response: Response): Promise<string> {
  if (!response.body) return ''
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_RESPONSE_BYTES) {
        throw new Error('QQ 音乐官方授权响应超过安全限制')
      }
      chunks.push(value)
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined)
    throw error
  } finally {
    reader.releaseLock()
  }
  return Buffer.concat(chunks, total).toString('utf8')
}

export async function fetchQQMusicEkey(
  options: QQMusicEkeyRequestOptions,
): Promise<string> {
  if (options.signal?.aborted) throw new Error('任务已取消')
  const fetchImpl = options.fetchImpl ?? fetch
  const controller = new AbortController()
  let timedOut = false
  const timeout = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, REQUEST_TIMEOUT_MS)
  const abort = () => controller.abort(options.signal?.reason)
  options.signal?.addEventListener('abort', abort, { once: true })

  try {
    const response = await fetchImpl(QQMUSIC_EKEY_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent':
          options.platform === '27'
            ? 'QQMusic/20 GuitarTabStudio (Windows)'
            : 'QQMusic/20 GuitarTabStudio (macOS)',
      },
      body: JSON.stringify(
        buildQQMusicEkeyRequest(
          options.media,
          options.credential,
          '10000',
          options.platform,
        ),
      ),
      redirect: 'error',
      signal: controller.signal,
    })
    if (!response.ok) {
      throw new Error(
        `QQ 音乐官方授权接口暂时不可用（HTTP ${response.status}）`,
      )
    }
    const declaredLength = Number(response.headers.get('content-length'))
    if (
      Number.isFinite(declaredLength) &&
      declaredLength > MAX_RESPONSE_BYTES
    ) {
      throw new Error('QQ 音乐官方授权响应超过安全限制')
    }
    const text = await boundedResponseText(response)
    let payload: unknown
    try {
      payload = JSON.parse(text)
    } catch {
      throw new Error('QQ 音乐官方授权响应格式无效')
    }
    const root = objectValue(payload)
    const requestResult = objectValue(root?.req_1)
    const data = objectValue(requestResult?.data)
    const entries = Array.isArray(data?.midurlinfo)
      ? data.midurlinfo
      : []
    const entry = objectValue(entries[0])
    const ekey = validMusicKey(entry?.ekey)
    if (ekey && /^[a-z0-9+/=_-]+$/iu.test(ekey)) {
      return ekey
    }
    const result = Number(
      entry?.result ??
        entry?.errcode ??
        data?.retcode ??
        requestResult?.code,
    )
    if (Number.isFinite(result) && result !== 0) {
      throw officialAuthorizationError(result)
    }
    throw new Error(
      'QQ 音乐官方接口未返回内容密钥。当前登录凭据可能已过期，或该账号无权访问对应曲目和音质。',
    )
  } catch (error) {
    if (options.signal?.aborted) throw new Error('任务已取消')
    if (timedOut) throw new Error('QQ 音乐官方授权验证超时')
    if (error instanceof Error) throw error
    throw new Error('QQ 音乐官方授权验证失败')
  } finally {
    clearTimeout(timeout)
    options.signal?.removeEventListener('abort', abort)
  }
}

export async function resolveQQMusicEkeyForDesktop(
  media: QQMusicMediaIdentity,
  options: QQMusicDesktopEkeyOptions = {},
): Promise<string> {
  const env = options.env ?? process.env
  if (env.GTS_DESKTOP_LOCAL !== '1') {
    throw new Error(
      '该 QMC2 文件没有内嵌密钥。浏览器端无法读取本机 QQ 音乐登录态，请改用 macOS 或 Windows 桌面版处理。',
    )
  }
  const platform = options.platform ?? process.platform
  if (platform !== 'darwin' && platform !== 'win32') {
    throw new Error(
      '当前桌面系统暂不支持自动读取 QQ 音乐登录态；无内嵌密钥的 QMC2 文件目前仅支持 macOS 和 Windows 桌面版。',
    )
  }
  const homePath = env.GTS_DESKTOP_HOME
  if (!homePath || !isAbsolute(homePath)) {
    throw new Error('桌面运行时未提供可用的用户目录')
  }
  const loadCredential =
    options.loadCredential ??
    (platform === 'win32'
      ? loadQQMusicWindowsCredential
      : loadQQMusicMacCredential)
  const credential = await loadCredential(homePath)
  if (!credential) {
    throw new Error(
      '未读取到有效的 QQ 音乐登录态。请先在 QQ 音乐桌面端登录下载该文件的账号，然后重试。',
    )
  }
  return fetchQQMusicEkey({
    credential,
    fetchImpl: options.fetchImpl,
    media,
    platform: platform === 'win32' ? '27' : '20',
    signal: options.signal,
  })
}
