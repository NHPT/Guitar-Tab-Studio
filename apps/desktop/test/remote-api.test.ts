import assert from 'node:assert/strict'
import test from 'node:test'
import { RemoteApiService } from '../src/remote-api.js'

test('requires HTTPS for cloud inference', () => {
  assert.throws(
    () => new RemoteApiService('http://example.test'),
    /must use HTTPS/,
  )
})

test('reads cloud capability state from the configured HTTPS origin', async () => {
  const originalFetch = globalThis.fetch
  let requestedUrl = ''
  globalThis.fetch = (async (input: string | URL | Request) => {
    requestedUrl = String(input)
    return new Response(
      JSON.stringify({
        capabilities: {
          ffmpeg: true,
          ffprobe: true,
          ytDlp: true,
          worker: true,
          mode: 'inference',
        },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    )
  }) as typeof fetch
  try {
    const service = new RemoteApiService('https://cloud.example.test/')
    const capabilities = await service.capabilities()
    assert.equal(capabilities.worker, true)
    assert.equal(requestedUrl, 'https://cloud.example.test/health')
  } finally {
    globalThis.fetch = originalFetch
  }
})
