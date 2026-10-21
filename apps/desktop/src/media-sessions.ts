import { randomUUID } from 'node:crypto'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import {
  BrowserWindow,
  session,
  type Cookie,
  type Session,
} from 'electron'
import type {
  DesktopMediaSessionAuthorization,
  DesktopQQMusicAuthorization,
  MediaBrowserDescriptor,
} from '@gts/core'
import {
  netscapeCookieFile,
  qqMusicCredentialFromCookies,
} from './media-session-data.js'

const authorizationTimeoutMs = 10 * 60 * 1000
const sessionLifetimeMs = 15 * 60 * 1000

interface TemporarySession {
  expiresAt: number
  path: string
}

interface LoginTarget {
  acceptedDomains: string[]
  authenticated: (cookies: Cookie[]) => boolean
  startUrl: string
  title: string
}

function belongsToDomain(hostname: string, domains: string[]): boolean {
  const host = hostname.toLowerCase()
  return domains.some(
    (domain) => host === domain || host.endsWith(`.${domain}`),
  )
}

function mediaLoginTarget(url: string): LoginTarget {
  const hostname = new URL(url).hostname.toLowerCase()
  if (
    hostname === 'b23.tv' ||
    hostname === 'bili2233.cn' ||
    belongsToDomain(hostname, ['bilibili.com'])
  ) {
    return {
      acceptedDomains: ['bilibili.com'],
      authenticated: (cookies) => {
        const names = new Set(cookies.map((cookie) => cookie.name))
        return names.has('SESSDATA') && names.has('DedeUserID')
      },
      startUrl: 'https://passport.bilibili.com/login',
      title: '登录哔哩哔哩',
    }
  }
  throw new Error('该平台暂不支持应用内隔离登录')
}

const qqMusicLoginTarget: LoginTarget = {
  acceptedDomains: ['qq.com'],
  authenticated: (cookies) =>
    qqMusicCredentialFromCookies(cookies) !== undefined,
  startUrl: 'https://y.qq.com/',
  title: '登录 QQ 音乐',
}

export class DesktopMediaSessions {
  private readonly cookieSessions = new Map<string, TemporarySession>()
  private readonly credentialSessions = new Map<string, TemporarySession>()
  private readonly windows = new Set<BrowserWindow>()

  constructor(
    private readonly root: string,
    private readonly windowProvider: () => BrowserWindow | undefined,
  ) {}

  async initialize(): Promise<void> {
    await rm(this.root, { recursive: true, force: true })
    await mkdir(this.root, { recursive: true, mode: 0o700 })
  }

  async authorizeMedia(
    url: string,
    browser: MediaBrowserDescriptor,
  ): Promise<DesktopMediaSessionAuthorization> {
    const target = mediaLoginTarget(url)
    const cookies = await this.authorize(target)
    const token = randomUUID()
    const path = resolve(this.root, `${token}.cookies.txt`)
    await writeFile(
      path,
      netscapeCookieFile(cookies, target.acceptedDomains),
      { encoding: 'utf8', mode: 0o600 },
    )
    this.cookieSessions.set(token, {
      expiresAt: Date.now() + sessionLifetimeMs,
      path,
    })
    return { browser, sessionToken: token }
  }

  async authorizeQQMusic(): Promise<DesktopQQMusicAuthorization> {
    const cookies = await this.authorize(qqMusicLoginTarget)
    const credential = qqMusicCredentialFromCookies(cookies)
    if (!credential) throw new Error('QQ 音乐登录完成，但未返回有效授权信息')
    const token = randomUUID()
    const path = resolve(this.root, `${token}.qqmusic.json`)
    await writeFile(path, `${JSON.stringify(credential)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    this.credentialSessions.set(token, {
      expiresAt: Date.now() + sessionLifetimeMs,
      path,
    })
    return { credentialSession: token }
  }

  consumeCookieSession(token: string): string | undefined {
    return this.consume(this.cookieSessions, token)
  }

  consumeCredentialSession(token: string): string | undefined {
    return this.consume(this.credentialSessions, token)
  }

  async stop(): Promise<void> {
    for (const window of this.windows) {
      if (!window.isDestroyed()) window.destroy()
    }
    this.windows.clear()
    this.cookieSessions.clear()
    this.credentialSessions.clear()
    await rm(this.root, { recursive: true, force: true })
  }

  private consume(
    sessions: Map<string, TemporarySession>,
    token: string,
  ): string | undefined {
    const entry = sessions.get(token)
    sessions.delete(token)
    if (!entry || entry.expiresAt < Date.now()) {
      if (entry) void rm(entry.path, { force: true })
      return undefined
    }
    return entry.path
  }

  private authorize(target: LoginTarget): Promise<Cookie[]> {
    const partition = `gts-media-login-${randomUUID()}`
    const loginSession = session.fromPartition(partition, { cache: false })
    const parent = this.windowProvider()
    const window = new BrowserWindow({
      width: 1060,
      height: 760,
      minWidth: 760,
      minHeight: 560,
      show: false,
      title: target.title,
      ...(parent && !parent.isDestroyed() ? { parent } : {}),
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        session: loginSession,
      },
    })
    this.windows.add(window)
    window.once('ready-to-show', () => window.show())
    window.webContents.setWindowOpenHandler(({ url }) => {
      if (this.allowedLoginUrl(url, target)) void window.loadURL(url)
      return { action: 'deny' }
    })
    window.webContents.on('will-navigate', (event, url) => {
      if (!this.allowedLoginUrl(url, target)) event.preventDefault()
    })

    return new Promise((resolveCookies, reject) => {
      let settled = false
      let checking = false
      const finish = (
        error?: Error,
        cookies?: Cookie[],
      ) => {
        if (settled) return
        settled = true
        clearInterval(interval)
        clearTimeout(timeout)
        this.windows.delete(window)
        if (!window.isDestroyed()) window.destroy()
        void this.clearLoginSession(loginSession)
        if (error) reject(error)
        else resolveCookies(cookies ?? [])
      }
      const inspect = async () => {
        if (settled || checking) return
        checking = true
        try {
          const cookies = await loginSession.cookies.get({})
          if (target.authenticated(cookies)) finish(undefined, cookies)
        } catch {
          // Keep the login window usable and retry until timeout.
        } finally {
          checking = false
        }
      }
      const interval = setInterval(() => void inspect(), 750)
      const timeout = setTimeout(
        () => finish(new Error('登录授权等待超时，请重新尝试')),
        authorizationTimeoutMs,
      )
      window.once('closed', () => {
        if (!settled) finish(new Error('登录授权已取消'))
      })
      void loginSession.clearStorageData().then(() =>
        window.loadURL(target.startUrl),
      ).catch((error: unknown) =>
        finish(
          error instanceof Error
            ? error
            : new Error('无法打开登录授权窗口'),
        ),
      )
    })
  }

  private allowedLoginUrl(url: string, target: LoginTarget): boolean {
    try {
      const parsed = new URL(url)
      return (
        parsed.protocol === 'https:' &&
        belongsToDomain(parsed.hostname, target.acceptedDomains)
      )
    } catch {
      return false
    }
  }

  private async clearLoginSession(loginSession: Session): Promise<void> {
    await loginSession.clearStorageData().catch(() => undefined)
    await loginSession.clearCache().catch(() => undefined)
  }
}
