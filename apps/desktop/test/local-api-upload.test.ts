import assert from 'node:assert/strict'
import test from 'node:test'
import { LocalApiService } from '../src/local-api.js'

test('sends the original UTF-8 filename outside multipart file headers', async () => {
  const originalFetch = globalThis.fetch
  let submittedName: FormDataEntryValue | null = null
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = init?.body
    assert.ok(body instanceof FormData)
    submittedName = body.get('originalName')
    return new Response(
      JSON.stringify({
        id: 'fixture-job',
        status: 'queued',
        progress: 0,
        stageLabel: '等待处理',
      }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      },
    )
  }) as typeof fetch

  try {
    const service = new LocalApiService({
      apiEntry: 'unused',
      workingDirectory: process.cwd(),
      dataRoot: process.cwd(),
    })
    Object.assign(service, { root: 'http://127.0.0.1:8787' })
    await service.submitUpload({
      name: '薛之谦-我好像在哪见过你.mgg',
      mediaType: 'application/octet-stream',
      bytes: new Uint8Array([1, 2, 3]),
    })
    assert.equal(submittedName, '薛之谦-我好像在哪见过你.mgg')
  } finally {
    globalThis.fetch = originalFetch
  }
})
