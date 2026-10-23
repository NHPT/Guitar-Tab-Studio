import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { once } from 'node:events'
import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises'
import { createServer, type Server } from 'node:http'
import { resolve } from 'node:path'
import test from 'node:test'
import {
  CancellationError,
  CancellationToken,
  type DownloadOptions,
} from 'builder-util-runtime'
import {
  parseContentRange,
  ResumableUpdateDownloader,
  resumablePartName,
} from '../src/resumable-update-download.js'

const fixture = Buffer.alloc(512 * 1024, 0x47)

async function testRoot(): Promise<string> {
  const parent = resolve(process.cwd(), 'data')
  await mkdir(parent, { recursive: true })
  return mkdtemp(resolve(parent, 'resumable-update-test-'))
}

async function listen(server: Server): Promise<URL> {
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (!address || typeof address === 'string') {
    throw new Error('Test server did not expose a TCP address')
  }
  return new URL(`http://127.0.0.1:${address.port}/update.zip`)
}

function downloadOptions(token = new CancellationToken()): DownloadOptions {
  return {
    cancellationToken: token,
    sha512: createHash('sha512').update(fixture).digest('base64'),
  }
}

test('parses satisfied and unsatisfied HTTP content ranges', () => {
  assert.deepEqual(parseContentRange('bytes 128-255/512'), {
    start: 128,
    end: 255,
    total: 512,
  })
  assert.deepEqual(parseContentRange('bytes */512'), { total: 512 })
  assert.equal(parseContentRange('invalid'), null)
})

test('resumes an existing partial update with an HTTP range request', async () => {
  const root = await testRoot()
  let requestedRange = ''
  const server = createServer((request, response) => {
    requestedRange = String(request.headers.range ?? '')
    const start = Number(/^bytes=(\d+)-$/u.exec(requestedRange)?.[1] ?? 0)
    response.writeHead(start > 0 ? 206 : 200, {
      'content-length': fixture.length - start,
      ...(start > 0
        ? {
            'content-range': `bytes ${start}-${fixture.length - 1}/${fixture.length}`,
          }
        : {}),
    })
    response.end(fixture.subarray(start))
  })
  const url = await listen(server)

  try {
    const partialRoot = resolve(root, 'parts')
    const destination = resolve(root, 'complete.zip')
    const options = downloadOptions()
    await mkdir(partialRoot, { recursive: true })
    await writeFile(
      resolve(partialRoot, resumablePartName(url, options)),
      fixture.subarray(0, 128 * 1024),
    )
    const downloader = new ResumableUpdateDownloader(partialRoot, fetch)

    await downloader.download(url, destination, options)

    assert.equal(requestedRange, `bytes=${128 * 1024}-`)
    assert.deepEqual(await readFile(destination), fixture)
  } finally {
    server.close()
    await rm(root, { recursive: true, force: true })
  }
})

test('keeps partial bytes on pause, resumes them, and deletes them on cancel', async () => {
  const root = await testRoot()
  const ranges: string[] = []
  const server = createServer((request, response) => {
    const range = String(request.headers.range ?? '')
    ranges.push(range)
    const start = Number(/^bytes=(\d+)-$/u.exec(range)?.[1] ?? 0)
    response.writeHead(start > 0 ? 206 : 200, {
      'content-length': fixture.length - start,
      ...(start > 0
        ? {
            'content-range': `bytes ${start}-${fixture.length - 1}/${fixture.length}`,
          }
        : {}),
    })
    let offset = start
    const interval = setInterval(() => {
      if (offset >= fixture.length || response.destroyed) {
        clearInterval(interval)
        if (!response.destroyed) response.end()
        return
      }
      const next = Math.min(offset + 16 * 1024, fixture.length)
      response.write(fixture.subarray(offset, next))
      offset = next
    }, 5)
    interval.unref()
  })
  const url = await listen(server)

  try {
    const partialRoot = resolve(root, 'parts')
    const destination = resolve(root, 'complete.zip')
    const downloader = new ResumableUpdateDownloader(partialRoot, fetch)
    const pauseToken = new CancellationToken()
    const pausedDownload = downloader.download(
      url,
      destination,
      downloadOptions(pauseToken),
    )
    await new Promise((resolveWait) => setTimeout(resolveWait, 35))
    pauseToken.cancel()
    await assert.rejects(pausedDownload, CancellationError)

    const partialFiles = await readdir(partialRoot)
    assert.equal(partialFiles.length, 1)
    const partial = await readFile(resolve(partialRoot, partialFiles[0]))
    assert.ok(partial.length > 0)
    assert.ok(partial.length < fixture.length)

    await downloader.download(url, destination, downloadOptions())
    assert.match(ranges.at(-1) ?? '', /^bytes=\d+-$/u)
    assert.deepEqual(await readFile(destination), fixture)

    await writeFile(
      resolve(partialRoot, 'cancelled.part'),
      fixture.subarray(0, 32),
    )
    await downloader.clearPartials()
    assert.deepEqual(await readdir(partialRoot), [])
  } finally {
    server.close()
    await rm(root, { recursive: true, force: true })
  }
})
