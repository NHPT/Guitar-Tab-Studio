import assert from 'node:assert/strict'
import { mkdir, rm } from 'node:fs/promises'
import { resolve } from 'node:path'
import test from 'node:test'
import {
  AdminAuthError,
  AdminAuthStore,
  adminHasPermission,
  adminSessionCookie,
} from './admin-auth.js'

test('bootstraps one owner and authenticates an HttpOnly session', async () => {
  const root = resolve(process.cwd(), '.test-admin-auth')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  try {
    const store = new AdminAuthStore(root)
    assert.equal(store.configured(), false)
    const session = store.bootstrap({
      username: 'admin',
      password: 'correct-horse-battery',
      displayName: 'Owner',
    })
    assert.equal(store.configured(), true)
    assert.equal(session.user.role, 'owner')
    assert.equal(adminHasPermission(session.user, 'users:manage'), true)
    const cookie = adminSessionCookie(session.token, false)
    assert.match(cookie, /HttpOnly/u)
    assert.match(cookie, /SameSite=Lax/u)
    assert.equal(store.authenticate(cookie)?.username, 'admin')
    assert.throws(
      () =>
        store.bootstrap({
          username: 'second',
          password: 'another-long-password',
        }),
      (error: unknown) =>
        error instanceof AdminAuthError && error.statusCode === 409,
    )
    store.logout(cookie)
    assert.equal(store.authenticate(cookie), null)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('persists users but keeps sessions process-local', async () => {
  const root = resolve(process.cwd(), '.test-admin-auth-persist')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  try {
    const first = new AdminAuthStore(root)
    const bootstrap = first.bootstrap({
      username: 'review-owner',
      password: 'review-password-123',
    })
    const cookie = adminSessionCookie(bootstrap.token, false)
    assert.equal(first.authenticate(cookie)?.role, 'owner')

    const restarted = new AdminAuthStore(root)
    assert.equal(restarted.authenticate(cookie), null)
    const login = restarted.login({
      username: 'review-owner',
      password: 'review-password-123',
    })
    assert.equal(login.user.username, 'review-owner')
    assert.equal(
      restarted.authenticate(adminSessionCookie(login.token, false))?.id,
      login.user.id,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('binds one account to each independent reviewer alias', async () => {
  const root = resolve(process.cwd(), '.test-admin-auth-reviewers')
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  try {
    const store = new AdminAuthStore(root)
    const owner = store.bootstrap({
      username: 'owner',
      password: 'owner-password-123',
    }).user
    const reviewer = store.createUser(owner, {
      username: 'reviewer-b-user',
      password: 'reviewer-password-123',
      displayName: 'Reviewer B',
      role: 'reviewer',
      reviewerAlias: 'reviewer-b',
    })
    assert.equal(reviewer.reviewerAlias, 'reviewer-b')
    assert.throws(
      () =>
        store.createUser(owner, {
          username: 'duplicate-reviewer',
          password: 'reviewer-password-456',
          role: 'reviewer',
          reviewerAlias: 'reviewer-b',
        }),
      (error: unknown) =>
        error instanceof AdminAuthError && error.statusCode === 409,
    )
    assert.throws(
      () =>
        store.updateUser(owner, owner.id, {
          role: 'viewer',
        }),
      /最后一个所有者/u,
    )
    assert.equal(store.listUsers(owner).length, 2)
    assert.ok(
      store
        .listAudit(owner)
        .some((event) => event.action === 'admin.user.created'),
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
