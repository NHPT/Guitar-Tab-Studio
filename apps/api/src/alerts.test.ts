import assert from 'node:assert/strict'
import { createHmac } from 'node:crypto'
import test from 'node:test'
import { createDeploymentAlertSink } from './alerts.js'

const alert = {
  schemaVersion: 1 as const,
  type: 'deployment-auto-rollback' as const,
  severity: 'critical' as const,
  occurredAt: '2026-09-29T00:00:00.000Z',
  promotionId: 'promotion-1',
  modelVersion: 'v1',
  mode: 'canary' as const,
  errorRatePercent: 12.5,
  observations: 20,
}

test('requires a credential-free HTTPS deployment alert target', () => {
  assert.throws(
    () =>
      createDeploymentAlertSink({
        url: 'http://alerts.example.test/hook',
        secret: 'a'.repeat(24),
      }),
    /requires HTTPS/,
  )
  assert.throws(
    () =>
      createDeploymentAlertSink({
        url: 'https://user:pass@alerts.example.test/hook',
        secret: 'a'.repeat(24),
      }),
    /requires HTTPS/,
  )
})

test('signs deployment alerts before delivery', async () => {
  let body = ''
  let signature = ''
  const sink = createDeploymentAlertSink({
    url: 'https://alerts.example.test/hook',
    secret: 'alert-signing-secret-000001',
    fetchImpl: (async (_input, init) => {
      body = String(init?.body)
      signature = new Headers(init?.headers).get('X-GTS-Signature') ?? ''
      return new Response(null, { status: 204 })
    }) as typeof fetch,
  })
  await sink(alert)
  assert.deepEqual(JSON.parse(body), alert)
  assert.equal(
    signature,
    `sha256=${createHmac('sha256', 'alert-signing-secret-000001')
      .update(body, 'utf8')
      .digest('hex')}`,
  )
})
