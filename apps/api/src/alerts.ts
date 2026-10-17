import { createHmac } from 'node:crypto'
import type { OperationalAlert } from './community.js'

export interface DeploymentAlertWebhookOptions {
  url: string
  secret: string
  fetchImpl?: typeof fetch
}

export function createDeploymentAlertSink(
  options: DeploymentAlertWebhookOptions,
): (alert: OperationalAlert) => Promise<void> {
  const target = new URL(options.url)
  if (
    target.protocol !== 'https:' ||
    target.username ||
    target.password ||
    options.secret.length < 24
  ) {
    throw new Error(
      'Deployment alert webhook requires HTTPS, no URL credentials, and a 24-character secret',
    )
  }
  const fetchImpl = options.fetchImpl ?? fetch
  return async (alert) => {
    const body = JSON.stringify(alert)
    const signature = createHmac('sha256', options.secret)
      .update(body, 'utf8')
      .digest('hex')
    const response = await fetchImpl(target, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-GTS-Signature': `sha256=${signature}`,
      },
      body,
      signal: AbortSignal.timeout(10_000),
    })
    if (!response.ok) {
      throw new Error(`Deployment alert delivery failed (${response.status})`)
    }
  }
}
