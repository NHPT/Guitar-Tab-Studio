import assert from 'node:assert/strict'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import { unlinkIfExists } from './managed-files.js'

test('treats repeated managed upload deletion as idempotent', () => {
  const root = resolve(
    process.cwd(),
    'data',
    `.test-managed-files-${randomUUID()}`,
  )
  const path = resolve(root, 'upload')
  mkdirSync(root, { recursive: true })
  writeFileSync(path, 'temporary upload')
  try {
    assert.equal(unlinkIfExists(path), true)
    assert.equal(unlinkIfExists(path), false)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
