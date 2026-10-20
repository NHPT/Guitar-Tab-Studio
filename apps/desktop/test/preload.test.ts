import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'

test('allows every implemented desktop file import operation', () => {
  const source = readFileSync(
    resolve(process.cwd(), 'src', 'preload.cts'),
    'utf8',
  )

  for (const method of [
    'submit-score',
    'import-media-key-database',
    'open-full-disk-access-settings',
    'detect-default-media-browser',
    'open-external-url',
  ]) {
    assert.match(source, new RegExp(`'${method}'`, 'u'))
  }
})
