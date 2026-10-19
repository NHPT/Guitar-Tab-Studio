import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import { desktopDataRoot } from '../src/data-root.js'

test('keeps settings and model weights in one stable upgrade directory', () => {
  const appData = resolve(
    process.cwd(),
    'data',
    `.test-app-data-${randomUUID()}`,
  )
  const firstVersionRoot = desktopDataRoot(appData)
  const settings = resolve(firstVersionRoot, 'state', 'desktop.sqlite')
  const weights = resolve(
    firstVersionRoot,
    'model-cache',
    'huggingface',
    'hub',
    'model.safetensors',
  )
  try {
    mkdirSync(resolve(settings, '..'), { recursive: true })
    mkdirSync(resolve(weights, '..'), { recursive: true })
    writeFileSync(settings, 'settings')
    writeFileSync(weights, 'weights')

    const upgradedVersionRoot = desktopDataRoot(appData)
    assert.equal(upgradedVersionRoot, firstVersionRoot)
    assert.equal(existsSync(settings), true)
    assert.equal(existsSync(weights), true)
  } finally {
    rmSync(appData, { recursive: true, force: true })
  }
})

test('allows isolated data roots for local previews and smoke tests', () => {
  const configured = resolve(
    process.cwd(),
    'data',
    `.test-configured-root-${randomUUID()}`,
  )
  assert.equal(desktopDataRoot('/unused-app-data', configured), configured)
})
