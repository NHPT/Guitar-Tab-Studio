import { createHash } from 'node:crypto'
import {
  copyFile,
  mkdir,
  open,
  readdir,
  rename,
  rm,
  stat,
} from 'node:fs/promises'
import { createReadStream } from 'node:fs'
import { basename, dirname, extname, resolve } from 'node:path'
import {
  CancellationError,
  type CancellationToken,
  type DownloadOptions,
} from 'builder-util-runtime'

export type UpdateFetch = (
  input: string | URL,
  init?: RequestInit,
) => Promise<Response>

interface ContentRange {
  start?: number
  end?: number
  total?: number
}

interface HashExpectation {
  algorithm: 'sha256' | 'sha512'
  encoding: 'base64' | 'hex'
  value: string
}

class ChecksumMismatchError extends Error {}

const maximumAttempts = 3
const progressIntervalMs = 200

export function parseContentRange(value: string | null): ContentRange | null {
  if (!value) return null
  const satisfied = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/iu.exec(value.trim())
  if (satisfied) {
    return {
      start: Number(satisfied[1]),
      end: Number(satisfied[2]),
      ...(satisfied[3] === '*' ? {} : { total: Number(satisfied[3]) }),
    }
  }
  const unsatisfied = /^bytes\s+\*\/(\d+)$/iu.exec(value.trim())
  return unsatisfied ? { total: Number(unsatisfied[1]) } : null
}

export function resumablePartName(
  url: URL,
  options: Pick<DownloadOptions, 'sha2' | 'sha512'>,
): string {
  const identity = `${url.toString()}\n${options.sha512 ?? options.sha2 ?? ''}`
  const key = createHash('sha256').update(identity).digest('hex')
  const suffix = extname(basename(url.pathname)).slice(0, 12)
  return `${key}${suffix}.part`
}

function expectedHash(
  options: Pick<DownloadOptions, 'sha2' | 'sha512'>,
): HashExpectation | null {
  if (options.sha512) {
    const hexadecimal =
      options.sha512.length === 128 &&
      !options.sha512.includes('+') &&
      !options.sha512.includes('Z') &&
      !options.sha512.includes('=')
    return {
      algorithm: 'sha512',
      encoding: hexadecimal ? 'hex' : 'base64',
      value: options.sha512,
    }
  }
  return options.sha2
    ? {
        algorithm: 'sha256',
        encoding: 'hex',
        value: options.sha2,
      }
    : null
}

async function verifyDownload(
  path: string,
  expectation: HashExpectation | null,
  token?: CancellationToken,
): Promise<void> {
  if (!expectation) return
  const hash = createHash(expectation.algorithm)
  for await (const chunk of createReadStream(path)) {
    if (token?.cancelled) throw new CancellationError()
    hash.update(chunk)
  }
  const actual = hash.digest(expectation.encoding)
  if (actual !== expectation.value) {
    throw new ChecksumMismatchError(
      `${expectation.algorithm} checksum mismatch, expected ${expectation.value}, got ${actual}`,
    )
  }
}

async function fileSize(path: string): Promise<number> {
  try {
    return (await stat(path)).size
  } catch (error) {
    if (
      error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT'
    ) {
      return 0
    }
    throw error
  }
}

function fetchHeaders(
  options: DownloadOptions,
  offset: number,
): Headers {
  const headers = new Headers()
  for (const [name, value] of Object.entries(options.headers ?? {})) {
    if (value === undefined) continue
    if (Array.isArray(value)) {
      for (const item of value) headers.append(name, String(item))
    } else {
      headers.set(name, String(value))
    }
  }
  headers.set('accept-encoding', 'identity')
  if (offset > 0) headers.set('range', `bytes=${offset}-`)
  return headers
}

function cancelled(token: CancellationToken): never {
  if (token.cancelled) throw new CancellationError()
  throw new Error('Update download was aborted')
}

async function waitBeforeRetry(
  attempt: number,
  token: CancellationToken,
): Promise<void> {
  await new Promise<void>((resolveWait, reject) => {
    const finish = () => {
      token.removeListener('cancel', cancel)
      resolveWait()
    }
    const timeout = setTimeout(finish, attempt * 600)
    const cancel = () => {
      clearTimeout(timeout)
      token.removeListener('cancel', cancel)
      reject(new CancellationError())
    }
    token.once('cancel', cancel)
    timeout.unref()
    void Promise.resolve().then(() => {
      if (token.cancelled) cancel()
    })
  })
}

async function moveCompletedFile(source: string, destination: string): Promise<void> {
  await mkdir(dirname(destination), { recursive: true })
  await rm(destination, { force: true })
  try {
    await rename(source, destination)
  } catch (error) {
    if (
      !error ||
      typeof error !== 'object' ||
      !('code' in error) ||
      error.code !== 'EXDEV'
    ) {
      throw error
    }
    await copyFile(source, destination)
    await rm(source, { force: true })
  }
}

export class ResumableUpdateDownloader {
  private readonly activeParts = new Set<string>()

  constructor(
    private readonly partialRoot: string,
    private readonly fetchUpdate: UpdateFetch,
  ) {}

  async download(
    url: URL,
    destination: string,
    options: DownloadOptions,
  ): Promise<string> {
    await mkdir(this.partialRoot, { recursive: true })
    const partPath = resolve(
      this.partialRoot,
      resumablePartName(url, options),
    )
    this.activeParts.add(partPath)
    try {
      for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
        try {
          await this.downloadAttempt(url, partPath, options)
          await verifyDownload(
            partPath,
            expectedHash(options),
            options.cancellationToken,
          )
          await moveCompletedFile(partPath, destination)
          return destination
        } catch (error) {
          if (
            options.cancellationToken.cancelled ||
            error instanceof CancellationError
          ) {
            throw new CancellationError()
          }
          if (error instanceof ChecksumMismatchError) {
            await rm(partPath, { force: true })
          }
          if (attempt === maximumAttempts) throw error
          await waitBeforeRetry(attempt, options.cancellationToken)
        }
      }
      throw new Error('Update download failed')
    } finally {
      this.activeParts.delete(partPath)
    }
  }

  async clearPartials(): Promise<void> {
    await mkdir(this.partialRoot, { recursive: true })
    const entries = await readdir(this.partialRoot, { withFileTypes: true })
    await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith('.part'))
        .map((entry) =>
          rm(resolve(this.partialRoot, entry.name), { force: true }),
        ),
    )
  }

  private async downloadAttempt(
    url: URL,
    partPath: string,
    options: DownloadOptions,
  ): Promise<void> {
    if (options.cancellationToken.cancelled) throw new CancellationError()
    let offset = await fileSize(partPath)
    const controller = new AbortController()
    const abort = () => controller.abort()
    options.cancellationToken.once('cancel', abort)

    try {
      let response = await this.fetchUpdate(url, {
        method: 'GET',
        headers: fetchHeaders(options, offset),
        redirect: 'follow',
        cache: 'no-store',
        signal: controller.signal,
      })
      let contentRange = parseContentRange(
        response.headers.get('content-range'),
      )

      if (response.status === 416 && offset > 0) {
        const total = contentRange?.total
        if (total === offset) {
          await verifyDownload(
            partPath,
            expectedHash(options),
            options.cancellationToken,
          )
          options.onProgress?.({
            percent: 100,
            delta: 0,
            transferred: offset,
            total,
            bytesPerSecond: 0,
          })
          return
        }
        await response.body?.cancel()
        await rm(partPath, { force: true })
        offset = 0
        response = await this.fetchUpdate(url, {
          method: 'GET',
          headers: fetchHeaders(options, 0),
          redirect: 'follow',
          cache: 'no-store',
          signal: controller.signal,
        })
        contentRange = parseContentRange(
          response.headers.get('content-range'),
        )
      }

      if (!response.ok) {
        await response.body?.cancel()
        throw new Error(
          `Cannot download update, status ${response.status}: ${response.statusText}`,
        )
      }

      const rangeAccepted =
        offset > 0 &&
        response.status === 206 &&
        contentRange?.start === offset
      if (offset > 0 && !rangeAccepted) offset = 0
      if (
        response.status === 206 &&
        contentRange?.start !== undefined &&
        contentRange.start !== offset
      ) {
        await response.body?.cancel()
        await rm(partPath, { force: true })
        throw new Error('Update server returned an invalid byte range')
      }

      const contentLength = Number(response.headers.get('content-length'))
      const total =
        contentRange?.total ??
        (Number.isFinite(contentLength) && contentLength >= 0
          ? offset + contentLength
          : 0)
      const body = response.body
      if (!body) throw new Error('Update server returned an empty response')

      const file = await open(partPath, offset > 0 ? 'a' : 'w')
      const reader = body.getReader()
      let transferred = offset
      let sampledBytes = transferred
      let sampledAt = Date.now()
      let lastProgressAt = 0

      options.onProgress?.({
        percent: total > 0 ? (transferred / total) * 100 : 0,
        delta: 0,
        transferred,
        total,
        bytesPerSecond: 0,
      })

      try {
        while (true) {
          if (options.cancellationToken.cancelled) {
            controller.abort()
            cancelled(options.cancellationToken)
          }
          const { done, value } = await reader.read()
          if (done) break
          await file.write(value)
          transferred += value.byteLength
          const now = Date.now()
          if (now - lastProgressAt >= progressIntervalMs) {
            const elapsed = Math.max(1, now - sampledAt)
            const bytesPerSecond =
              ((transferred - sampledBytes) * 1000) / elapsed
            options.onProgress?.({
              percent: total > 0 ? (transferred / total) * 100 : 0,
              delta: transferred - sampledBytes,
              transferred,
              total,
              bytesPerSecond,
            })
            sampledBytes = transferred
            sampledAt = now
            lastProgressAt = now
          }
        }
      } finally {
        reader.releaseLock()
        await file.close()
      }

      if (total > 0 && transferred !== total) {
        throw new Error(
          `Update download ended early at ${transferred} of ${total} bytes`,
        )
      }
      options.onProgress?.({
        percent: 100,
        delta: transferred - sampledBytes,
        transferred,
        total: total || transferred,
        bytesPerSecond: 0,
      })
    } catch (error) {
      if (
        options.cancellationToken.cancelled ||
        (error instanceof Error && error.name === 'AbortError')
      ) {
        throw new CancellationError()
      }
      throw error
    } finally {
      options.cancellationToken.removeListener('cancel', abort)
    }
  }
}
