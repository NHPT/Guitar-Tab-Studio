import { createHash, randomBytes } from 'node:crypto'
import { createServer, type Server } from 'node:http'
import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { dirname } from 'node:path'
import type { TranscriptionSetupProgress } from '@gts/core'

interface OAuthToken {
  accessToken: string
  refreshToken?: string
  expiresAt?: number
}

interface OAuthTokenResponse {
  access_token?: string
  refresh_token?: string
  expires_in?: number
  error?: string
  error_description?: string
}

export interface OAuthTokenEncryption {
  available(): boolean
  encrypt(value: string): Buffer
  decrypt(value: Buffer): string
}

const oauthEndpoint = 'https://huggingface.co/oauth/authorize'
const tokenEndpoint = 'https://huggingface.co/oauth/token'
const callbackPath = '/oauth/callback'
const authorizationTimeoutMs = 10 * 60_000

function base64Url(value: Buffer): string {
  return value.toString('base64url')
}

function completionPage(success: boolean, message: string): string {
  const title = success ? '授权完成' : '授权未完成'
  const color = success ? '#087443' : '#a33a32'
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${title}</title>
  <style>
    body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f7f7f8;color:#171717;font:14px/1.6 system-ui,-apple-system,sans-serif}
    main{width:min(420px,calc(100vw - 40px));padding:24px;border:1px solid rgba(23,23,23,.12);border-radius:12px;background:#fff}
    h1{margin:0 0 8px;font-size:20px;color:${color}}p{margin:0;color:#52525b}
  </style>
</head>
<body>
  <main><h1>${title}</h1><p>${message}</p></main>
  ${success ? '<script>window.setTimeout(() => window.close(), 350)</script>' : ''}
</body>
</html>`
}

export class HuggingFaceOAuth {
  private memoryToken?: OAuthToken

  constructor(
    private readonly tokenPath: string,
    private readonly clientId: () => string | undefined,
    private readonly onProgress: (
      progress: TranscriptionSetupProgress,
    ) => void,
    private readonly encryption?: OAuthTokenEncryption,
  ) {}

  configured(): boolean {
    return Boolean(this.clientId()?.trim())
  }

  resetToken(): void {
    this.clearToken()
  }

  async accessToken(signal?: AbortSignal): Promise<string> {
    const existing = this.readToken()
    if (
      existing &&
      (!existing.expiresAt || existing.expiresAt > Date.now() + 60_000)
    ) {
      return existing.accessToken
    }
    if (existing?.refreshToken) {
      try {
        const refreshed = await this.exchange({
          grant_type: 'refresh_token',
          refresh_token: existing.refreshToken,
          client_id: this.requiredClientId(),
        })
        return this.saveToken(refreshed).accessToken
      } catch {
        this.clearToken()
      }
    }
    return this.authorize(signal)
  }

  private async authorize(signal?: AbortSignal): Promise<string> {
    const clientId = this.requiredClientId()
    const verifier = base64Url(randomBytes(48))
    const challenge = base64Url(
      createHash('sha256').update(verifier).digest(),
    )
    const expectedState = base64Url(randomBytes(24))
    const { server, port } = await this.listen()
    const redirectUri = `http://127.0.0.1:${port}${callbackPath}`
    const authorizationUrl = new URL(oauthEndpoint)
    authorizationUrl.searchParams.set('client_id', clientId)
    authorizationUrl.searchParams.set('redirect_uri', redirectUri)
    authorizationUrl.searchParams.set('response_type', 'code')
    authorizationUrl.searchParams.set(
      'scope',
      'openid profile gated-repos',
    )
    authorizationUrl.searchParams.set('state', expectedState)
    authorizationUrl.searchParams.set('code_challenge', challenge)
    authorizationUrl.searchParams.set('code_challenge_method', 'S256')

    this.onProgress({
      phase: 'authorizing',
      progress: 3,
      stageLabel: '等待 Hugging Face 浏览器授权',
      authorizationUrl: authorizationUrl.toString(),
    })

    try {
      const code = await this.waitForCallback(
        server,
        expectedState,
        signal,
      )
      this.onProgress({
        phase: 'checking',
        progress: 5,
        stageLabel: '授权成功，正在验证模型访问权限',
      })
      const response = await this.exchange({
        grant_type: 'authorization_code',
        client_id: clientId,
        code,
        redirect_uri: redirectUri,
        code_verifier: verifier,
      })
      return this.saveToken(response).accessToken
    } finally {
      server.close()
    }
  }

  private listen(): Promise<{ server: Server; port: number }> {
    return new Promise((resolveListen, rejectListen) => {
      const server = createServer()
      server.once('error', rejectListen)
      server.listen(0, '127.0.0.1', () => {
        const address = server.address()
        if (!address || typeof address === 'string') {
          server.close()
          rejectListen(new Error('无法启动 Hugging Face 本机授权回调'))
          return
        }
        resolveListen({ server, port: address.port })
      })
    })
  }

  private waitForCallback(
    server: Server,
    expectedState: string,
    signal?: AbortSignal,
  ): Promise<string> {
    return new Promise((resolveCode, rejectCode) => {
      let settled = false
      const finish = (error?: Error, code?: string) => {
        if (settled) return
        settled = true
        clearTimeout(timeout)
        signal?.removeEventListener('abort', cancel)
        if (error) rejectCode(error)
        else resolveCode(code!)
      }
      const cancel = () =>
        finish(new Error('已取消 Hugging Face 浏览器授权'))
      const timeout = setTimeout(
        () =>
          finish(
            new Error(
              'Hugging Face 浏览器授权等待超时，可以重新点击“下载并启用”',
            ),
          ),
        authorizationTimeoutMs,
      )
      timeout.unref()
      signal?.addEventListener('abort', cancel, { once: true })
      server.on('request', (request, response) => {
        const requestUrl = new URL(
          request.url ?? '/',
          'http://127.0.0.1',
        )
        if (requestUrl.pathname !== callbackPath) {
          response.writeHead(404).end()
          return
        }
        const error = requestUrl.searchParams.get('error')
        const state = requestUrl.searchParams.get('state')
        const code = requestUrl.searchParams.get('code')
        if (error) {
          response.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' })
          response.end(completionPage(false, '可以返回桌面应用重新尝试。'))
          finish(new Error('Hugging Face 授权未完成，可以重新尝试'))
          return
        }
        if (state !== expectedState || !code) {
          response.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' })
          response.end(completionPage(false, '授权响应无效，请返回应用重试。'))
          finish(new Error('Hugging Face 授权响应校验失败'))
          return
        }
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
        response.end(
          completionPage(true, '正在返回 Guitar Tab Studio 并继续下载。'),
        )
        finish(undefined, code)
      })
      if (signal?.aborted) cancel()
    })
  }

  private async exchange(
    parameters: Record<string, string>,
  ): Promise<OAuthTokenResponse> {
    const response = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(parameters),
      signal: AbortSignal.timeout(30_000),
    })
    const result = (await response.json()) as OAuthTokenResponse
    if (!response.ok || !result.access_token) {
      throw new Error(
        result.error_description ??
          result.error ??
          `Hugging Face OAuth 失败（${response.status}）`,
      )
    }
    return result
  }

  private saveToken(response: OAuthTokenResponse): OAuthToken {
    const token: OAuthToken = {
      accessToken: response.access_token!,
      ...(response.refresh_token
        ? { refreshToken: response.refresh_token }
        : {}),
      ...(response.expires_in
        ? { expiresAt: Date.now() + response.expires_in * 1000 }
        : {}),
    }
    this.memoryToken = token
    if (this.encryption?.available()) {
      mkdirSync(dirname(this.tokenPath), { recursive: true })
      writeFileSync(
        this.tokenPath,
        this.encryption.encrypt(JSON.stringify(token)),
        { mode: 0o600 },
      )
      chmodSync(this.tokenPath, 0o600)
    }
    return token
  }

  private readToken(): OAuthToken | undefined {
    if (this.memoryToken) return this.memoryToken
    if (
      !this.encryption?.available() ||
      !existsSync(this.tokenPath)
    ) {
      return undefined
    }
    try {
      const token = JSON.parse(
        this.encryption.decrypt(readFileSync(this.tokenPath)),
      ) as OAuthToken
      if (!token.accessToken) return undefined
      this.memoryToken = token
      return token
    } catch {
      this.clearToken()
      return undefined
    }
  }

  private clearToken(): void {
    this.memoryToken = undefined
    rmSync(this.tokenPath, { force: true })
  }

  private requiredClientId(): string {
    const clientId = this.clientId()?.trim()
    if (!clientId) {
      throw new Error(
        '尚未配置 Hugging Face OAuth Client ID，可先配置或使用“导入权重”',
      )
    }
    return clientId
  }
}
