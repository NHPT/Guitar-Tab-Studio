import {
  createHash,
  randomBytes,
  randomUUID,
  scryptSync,
  timingSafeEqual,
} from 'node:crypto'
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'

const USERNAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$/
const SESSION_COOKIE = 'gts_admin_session'
const SESSION_SECONDS = 8 * 60 * 60
const MAX_LOGIN_FAILURES = 5
const LOGIN_WINDOW_MS = 15 * 60 * 1000
const REVIEWER_ALIAS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

export type AdminRole =
  | 'owner'
  | 'reviewer'
  | 'adjudicator'
  | 'model-maintainer'
  | 'viewer'

export type AdminPermission =
  | 'admin:read'
  | 'reviews:read'
  | 'reviews:write'
  | 'adjudication:write'
  | 'models:write'
  | 'users:manage'

export interface AdminUser {
  id: string
  username: string
  displayName: string
  role: AdminRole
  active: boolean
  reviewerAlias?: string
  createdAt: string
  lastLoginAt?: string
}

export interface AdminAuditEvent {
  at: string
  action: string
  actorId: string | null
  actorUsername: string | null
  [key: string]: unknown
}

interface StoredAdminUser extends AdminUser {
  password: {
    algorithm: 'scrypt-v1'
    salt: string
    hash: string
  }
}

interface AdminState {
  schemaVersion: 1
  users: StoredAdminUser[]
}

interface Session {
  userId: string
  expiresAt: number
}

interface LoginFailure {
  attempts: number[]
}

const ROLE_PERMISSIONS: Record<AdminRole, ReadonlySet<AdminPermission>> = {
  owner: new Set([
    'admin:read',
    'reviews:read',
    'reviews:write',
    'adjudication:write',
    'models:write',
    'users:manage',
  ]),
  reviewer: new Set(['admin:read', 'reviews:read', 'reviews:write']),
  adjudicator: new Set([
    'admin:read',
    'reviews:read',
    'adjudication:write',
  ]),
  'model-maintainer': new Set([
    'admin:read',
    'reviews:read',
    'models:write',
  ]),
  viewer: new Set(['admin:read', 'reviews:read']),
}

export class AdminAuthError extends Error {
  constructor(
    message: string,
    readonly statusCode = 400,
  ) {
    super(message)
  }
}

function normalizeUsername(value: unknown): string {
  const username = typeof value === 'string' ? value.trim() : ''
  if (!USERNAME_PATTERN.test(username)) {
    throw new AdminAuthError('账号需为 3-32 位字母、数字、点、横线或下划线')
  }
  return username
}

function normalizeDisplayName(value: unknown, fallback: string): string {
  const displayName = typeof value === 'string' ? value.trim() : ''
  const normalized = displayName || fallback
  if (normalized.length > 48) {
    throw new AdminAuthError('显示名称不能超过 48 个字符')
  }
  return normalized
}

function normalizePassword(value: unknown): string {
  if (
    typeof value !== 'string' ||
    value.length < 12 ||
    value.length > 128
  ) {
    throw new AdminAuthError('密码长度需为 12-128 个字符')
  }
  return value
}

function normalizeRole(value: unknown): AdminRole {
  if (
    value === 'owner' ||
    value === 'reviewer' ||
    value === 'adjudicator' ||
    value === 'model-maintainer' ||
    value === 'viewer'
  ) {
    return value
  }
  throw new AdminAuthError('账号角色无效')
}

function normalizeReviewerAlias(
  value: unknown,
  role: AdminRole,
): string | undefined {
  const alias = typeof value === 'string' ? value.trim() : ''
  if (!alias) {
    if (role === 'reviewer') {
      throw new AdminAuthError('复核员必须绑定审核者标识')
    }
    return undefined
  }
  if (!REVIEWER_ALIAS_PATTERN.test(alias)) {
    throw new AdminAuthError('审核者标识无效')
  }
  return alias
}

function passwordDigest(password: string, salt: Buffer): Buffer {
  return scryptSync(password, salt, 32, {
    N: 16_384,
    r: 8,
    p: 1,
    maxmem: 32 * 1024 * 1024,
  })
}

function hashPassword(password: string): StoredAdminUser['password'] {
  const salt = randomBytes(16)
  return {
    algorithm: 'scrypt-v1',
    salt: salt.toString('base64url'),
    hash: passwordDigest(password, salt).toString('base64url'),
  }
}

function passwordMatches(
  password: string,
  stored: StoredAdminUser['password'],
): boolean {
  if (stored.algorithm !== 'scrypt-v1') return false
  const expected = Buffer.from(stored.hash, 'base64url')
  const actual = passwordDigest(
    password,
    Buffer.from(stored.salt, 'base64url'),
  )
  return (
    expected.length === actual.length &&
    timingSafeEqual(expected, actual)
  )
}

function sessionHash(token: string): string {
  return createHash('sha256').update(token).digest('hex')
}

function parseCookie(cookieHeader: string | undefined): Map<string, string> {
  const cookies = new Map<string, string>()
  for (const part of cookieHeader?.split(';') ?? []) {
    const separator = part.indexOf('=')
    if (separator < 0) continue
    const name = part.slice(0, separator).trim()
    const value = part.slice(separator + 1).trim()
    if (name) cookies.set(name, value)
  }
  return cookies
}

function publicUser(user: StoredAdminUser): AdminUser {
  const { password: _password, ...safe } = user
  return safe
}

export function adminSessionCookie(
  token: string,
  secure: boolean,
): string {
  return [
    `${SESSION_COOKIE}=${token}`,
    'HttpOnly',
    'Path=/',
    'SameSite=Lax',
    `Max-Age=${SESSION_SECONDS}`,
    secure ? 'Secure' : '',
  ]
    .filter(Boolean)
    .join('; ')
}

export function clearAdminSessionCookie(secure: boolean): string {
  return [
    `${SESSION_COOKIE}=`,
    'HttpOnly',
    'Path=/',
    'SameSite=Lax',
    'Max-Age=0',
    secure ? 'Secure' : '',
  ]
    .filter(Boolean)
    .join('; ')
}

export function adminHasPermission(
  user: AdminUser,
  permission: AdminPermission,
): boolean {
  return user.active && ROLE_PERMISSIONS[user.role].has(permission)
}

export class AdminAuthStore {
  private readonly statePath: string
  private readonly auditPath: string
  private readonly sessions = new Map<string, Session>()
  private readonly loginFailures = new Map<string, LoginFailure>()
  private state: AdminState

  constructor(
    root: string,
    private readonly now: () => Date = () => new Date(),
  ) {
    mkdirSync(root, { recursive: true })
    this.statePath = resolve(root, 'state.json')
    this.auditPath = resolve(root, 'audit.jsonl')
    this.state = this.readState()
  }

  private readState(): AdminState {
    if (!existsSync(this.statePath)) {
      return { schemaVersion: 1, users: [] }
    }
    let value: unknown
    try {
      value = JSON.parse(readFileSync(this.statePath, 'utf8'))
    } catch {
      throw new AdminAuthError('管理后台账号数据无法读取', 500)
    }
    if (
      !value ||
      typeof value !== 'object' ||
      Array.isArray(value) ||
      (value as AdminState).schemaVersion !== 1 ||
      !Array.isArray((value as AdminState).users)
    ) {
      throw new AdminAuthError('管理后台账号数据格式无效', 500)
    }
    return value as AdminState
  }

  private writeState(): void {
    const temporary = `${this.statePath}.${process.pid}.${Date.now()}.tmp`
    writeFileSync(temporary, `${JSON.stringify(this.state, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    renameSync(temporary, this.statePath)
  }

  private audit(
    action: string,
    user: AdminUser | null,
    details: Record<string, unknown> = {},
  ): void {
    appendFileSync(
      this.auditPath,
      `${JSON.stringify({
        at: this.now().toISOString(),
        action,
        actorId: user?.id ?? null,
        actorUsername: user?.username ?? null,
        ...details,
      })}\n`,
      { encoding: 'utf8', mode: 0o600 },
    )
  }

  configured(): boolean {
    return this.state.users.some((user) => user.active)
  }

  bootstrap(input: {
    username?: unknown
    password?: unknown
    displayName?: unknown
  }): { user: AdminUser; token: string } {
    if (this.state.users.length > 0) {
      throw new AdminAuthError('管理后台已完成初始化', 409)
    }
    const username = normalizeUsername(input.username)
    const password = normalizePassword(input.password)
    const now = this.now().toISOString()
    const stored: StoredAdminUser = {
      id: randomUUID(),
      username,
      displayName: normalizeDisplayName(input.displayName, username),
      role: 'owner',
      active: true,
      createdAt: now,
      lastLoginAt: now,
      password: hashPassword(password),
    }
    this.state.users.push(stored)
    this.writeState()
    const user = publicUser(stored)
    this.audit('admin.bootstrap', user, { role: user.role })
    return { user, token: this.createSession(user.id) }
  }

  login(input: {
    username?: unknown
    password?: unknown
  }): { user: AdminUser; token: string } {
    const username = normalizeUsername(input.username)
    const password = normalizePassword(input.password)
    const nowMs = this.now().getTime()
    const key = username.toLowerCase()
    const recent = (this.loginFailures.get(key)?.attempts ?? []).filter(
      (at) => nowMs - at < LOGIN_WINDOW_MS,
    )
    if (recent.length >= MAX_LOGIN_FAILURES) {
      throw new AdminAuthError('登录尝试过多，请稍后再试', 429)
    }
    const stored = this.state.users.find(
      (candidate) => candidate.username.toLowerCase() === key,
    )
    if (!stored || !stored.active || !passwordMatches(password, stored.password)) {
      this.loginFailures.set(key, { attempts: [...recent, nowMs] })
      this.audit('admin.login.failed', stored ? publicUser(stored) : null, {
        username,
      })
      throw new AdminAuthError('账号或密码错误', 401)
    }
    this.loginFailures.delete(key)
    stored.lastLoginAt = this.now().toISOString()
    this.writeState()
    const user = publicUser(stored)
    this.audit('admin.login.succeeded', user)
    return { user, token: this.createSession(user.id) }
  }

  private createSession(userId: string): string {
    const token = randomBytes(32).toString('base64url')
    this.sessions.set(sessionHash(token), {
      userId,
      expiresAt: this.now().getTime() + SESSION_SECONDS * 1000,
    })
    return token
  }

  authenticate(cookieHeader: string | undefined): AdminUser | null {
    const token = parseCookie(cookieHeader).get(SESSION_COOKIE)
    if (!token) return null
    const key = sessionHash(token)
    const session = this.sessions.get(key)
    if (!session) return null
    if (session.expiresAt <= this.now().getTime()) {
      this.sessions.delete(key)
      return null
    }
    const stored = this.state.users.find(
      (candidate) => candidate.id === session.userId && candidate.active,
    )
    return stored ? publicUser(stored) : null
  }

  logout(cookieHeader: string | undefined): void {
    const token = parseCookie(cookieHeader).get(SESSION_COOKIE)
    if (!token) return
    const user = this.authenticate(cookieHeader)
    this.sessions.delete(sessionHash(token))
    this.audit('admin.logout', user)
  }

  listUsers(actor: AdminUser): AdminUser[] {
    if (!adminHasPermission(actor, 'users:manage')) {
      throw new AdminAuthError('需要账号管理权限', 403)
    }
    return this.state.users.map(publicUser)
  }

  createUser(
    actor: AdminUser,
    input: {
      username?: unknown
      password?: unknown
      displayName?: unknown
      role?: unknown
      reviewerAlias?: unknown
    },
  ): AdminUser {
    if (!adminHasPermission(actor, 'users:manage')) {
      throw new AdminAuthError('需要账号管理权限', 403)
    }
    const username = normalizeUsername(input.username)
    const key = username.toLowerCase()
    if (
      this.state.users.some(
        (candidate) => candidate.username.toLowerCase() === key,
      )
    ) {
      throw new AdminAuthError('账号已存在', 409)
    }
    const role = normalizeRole(input.role)
    const reviewerAlias = normalizeReviewerAlias(
      input.reviewerAlias,
      role,
    )
    if (
      reviewerAlias &&
      this.state.users.some(
        (candidate) =>
          candidate.active &&
          candidate.reviewerAlias === reviewerAlias,
      )
    ) {
      throw new AdminAuthError('审核者标识已绑定其他账号', 409)
    }
    const stored: StoredAdminUser = {
      id: randomUUID(),
      username,
      displayName: normalizeDisplayName(input.displayName, username),
      role,
      active: true,
      reviewerAlias,
      createdAt: this.now().toISOString(),
      password: hashPassword(normalizePassword(input.password)),
    }
    this.state.users.push(stored)
    this.writeState()
    const user = publicUser(stored)
    this.audit('admin.user.created', actor, {
      targetUserId: user.id,
      targetUsername: user.username,
      role: user.role,
      reviewerAlias: user.reviewerAlias ?? null,
    })
    return user
  }

  updateUser(
    actor: AdminUser,
    userId: string,
    input: {
      displayName?: unknown
      role?: unknown
      reviewerAlias?: unknown
      active?: unknown
      password?: unknown
    },
  ): AdminUser {
    if (!adminHasPermission(actor, 'users:manage')) {
      throw new AdminAuthError('需要账号管理权限', 403)
    }
    const stored = this.state.users.find((candidate) => candidate.id === userId)
    if (!stored) throw new AdminAuthError('账号不存在', 404)
    const role =
      input.role === undefined ? stored.role : normalizeRole(input.role)
    const reviewerAlias =
      input.reviewerAlias === undefined
        ? stored.reviewerAlias
        : normalizeReviewerAlias(input.reviewerAlias, role)
    const active =
      input.active === undefined
        ? stored.active
        : input.active === true
    if (
      reviewerAlias &&
      this.state.users.some(
        (candidate) =>
          candidate.id !== userId &&
          candidate.active &&
          candidate.reviewerAlias === reviewerAlias,
      )
    ) {
      throw new AdminAuthError('审核者标识已绑定其他账号', 409)
    }
    const remainingOwners = this.state.users.filter(
      (candidate) =>
        candidate.id !== userId &&
        candidate.active &&
        candidate.role === 'owner',
    )
    if (
      stored.role === 'owner' &&
      stored.active &&
      (!active || role !== 'owner') &&
      remainingOwners.length === 0
    ) {
      throw new AdminAuthError('不能停用或降级最后一个所有者', 409)
    }
    stored.displayName =
      input.displayName === undefined
        ? stored.displayName
        : normalizeDisplayName(input.displayName, stored.username)
    stored.role = role
    stored.reviewerAlias = reviewerAlias
    stored.active = active
    if (input.password !== undefined) {
      stored.password = hashPassword(normalizePassword(input.password))
    }
    this.writeState()
    const user = publicUser(stored)
    this.audit('admin.user.updated', actor, {
      targetUserId: user.id,
      targetUsername: user.username,
      role: user.role,
      reviewerAlias: user.reviewerAlias ?? null,
      active: user.active,
      passwordChanged: input.password !== undefined,
    })
    return user
  }

  listAudit(actor: AdminUser, limitValue: unknown = 100): AdminAuditEvent[] {
    if (!adminHasPermission(actor, 'users:manage')) {
      throw new AdminAuthError('需要账号管理权限', 403)
    }
    const limit = Math.max(
      1,
      Math.min(
        500,
        Number.isFinite(Number(limitValue)) ? Number(limitValue) : 100,
      ),
    )
    if (!existsSync(this.auditPath)) return []
    return readFileSync(this.auditPath, 'utf8')
      .split('\n')
      .filter(Boolean)
      .slice(-limit)
      .reverse()
      .flatMap((line) => {
        try {
          return [JSON.parse(line) as AdminAuditEvent]
        } catch {
          return []
        }
      })
  }
}
