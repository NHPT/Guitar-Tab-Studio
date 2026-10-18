import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import { get } from 'node:http'
import { rmSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import { HuggingFaceOAuth } from '../src/huggingface-oauth.js'

test('completes authorization code with PKCE through a loopback callback', async () => {
  const tokenPath = resolve(
    process.cwd(),
    'data',
    `.test-hf-oauth-${randomUUID()}`,
    'token.bin',
  )
  const originalFetch = globalThis.fetch
  let exchangeCount = 0
  globalThis.fetch = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ) => {
    assert.equal(String(input), 'https://huggingface.co/oauth/token')
    const body = init?.body as URLSearchParams
    assert.equal(body.get('grant_type'), 'authorization_code')
    assert.equal(body.get('client_id'), 'desktop-client')
    assert.equal(body.get('code'), 'authorization-code')
    const verifier = body.get('code_verifier')
    assert.ok(verifier)
    const challenge = createHash('sha256')
      .update(verifier)
      .digest('base64url')
    assert.equal(challenge, expectedChallenge)
    exchangeCount += 1
    return new Response(
      JSON.stringify({
        access_token: 'hf_oauth_access',
        refresh_token: 'hf_oauth_refresh',
        expires_in: 3600,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    )
  }) as typeof fetch
  let expectedChallenge = ''
  try {
    const oauth = new HuggingFaceOAuth(
      tokenPath,
      () => 'desktop-client',
      (progress) => {
        if (!progress.authorizationUrl) return
        const authorization = new URL(progress.authorizationUrl)
        assert.equal(authorization.searchParams.get('response_type'), 'code')
        assert.equal(
          authorization.searchParams.get('code_challenge_method'),
          'S256',
        )
        assert.match(
          authorization.searchParams.get('scope') ?? '',
          /gated-repos/u,
        )
        expectedChallenge =
          authorization.searchParams.get('code_challenge') ?? ''
        const callback = new URL(
          authorization.searchParams.get('redirect_uri')!,
        )
        assert.equal(callback.protocol, 'http:')
        assert.equal(callback.hostname, '127.0.0.1')
        assert.equal(callback.pathname, '/oauth/callback')
        assert.notEqual(callback.port, '')
        callback.searchParams.set(
          'state',
          authorization.searchParams.get('state')!,
        )
        callback.searchParams.set('code', 'authorization-code')
        queueMicrotask(() => {
          get(callback, (response) => response.resume())
        })
      },
    )

    assert.equal(await oauth.accessToken(), 'hf_oauth_access')
    assert.equal(await oauth.accessToken(), 'hf_oauth_access')
    assert.equal(exchangeCount, 1)
  } finally {
    globalThis.fetch = originalFetch
    rmSync(resolve(tokenPath, '..'), { recursive: true, force: true })
  }
})

test('cancels a pending PKCE callback wait', async () => {
  const controller = new AbortController()
  const oauth = new HuggingFaceOAuth(
    resolve(
      process.cwd(),
      'data',
      `.test-hf-oauth-${randomUUID()}`,
      'token.bin',
    ),
    () => 'desktop-client',
    () => controller.abort(),
  )

  await assert.rejects(
    oauth.accessToken(controller.signal),
    /已取消 Hugging Face 浏览器授权/,
  )
})

test('reports a system-network failure during OAuth token exchange', async () => {
  const root = resolve(
    process.cwd(),
    'data',
    `.test-hf-oauth-${randomUUID()}`,
  )
  const request = (async () => {
    const cause = Object.assign(new Error('DNS lookup failed'), {
      code: 'ENOTFOUND',
    })
    throw new TypeError('fetch failed', { cause })
  }) as typeof fetch
  try {
    const oauth = new HuggingFaceOAuth(
      resolve(root, 'token.bin'),
      () => 'desktop-client',
      (progress) => {
        if (!progress.authorizationUrl) return
        const authorization = new URL(progress.authorizationUrl)
        const callback = new URL(
          authorization.searchParams.get('redirect_uri')!,
        )
        callback.searchParams.set(
          'state',
          authorization.searchParams.get('state')!,
        )
        callback.searchParams.set('code', 'authorization-code')
        queueMicrotask(() => {
          get(callback, (response) => response.resume())
        })
      },
      undefined,
      request,
    )

    await assert.rejects(
      oauth.accessToken(),
      /无法连接 Hugging Face OAuth 服务（ENOTFOUND）.*系统代理/u,
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
