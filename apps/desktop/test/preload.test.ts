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
    'authorize-media-session',
    'authorize-qqmusic',
    'open-external-url',
  ]) {
    assert.match(source, new RegExp(`'${method}'`, 'u'))
  }
})

test('exposes update progress as a removable read-only event listener', () => {
  const source = readFileSync(
    resolve(process.cwd(), 'src', 'preload.cts'),
    'utf8',
  )

  assert.match(source, /onUpdateProgress/u)
  assert.match(source, /ipcRenderer\.on\('studio:update-progress'/u)
  assert.match(
    source,
    /ipcRenderer\.removeListener\('studio:update-progress'/u,
  )
  for (const method of [
    'update-status',
    'update-download',
    'update-pause',
    'update-resume',
    'update-cancel',
    'update-dismiss',
    'update-install',
  ]) {
    assert.match(source, new RegExp(`'${method}'`, 'u'))
  }
})
