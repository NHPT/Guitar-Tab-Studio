import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeUpdateProgress } from '../src/update-progress.js'

test('normalizes updater progress from transferred and total bytes', () => {
  assert.deepEqual(
    normalizeUpdateProgress('0.1.0-preview.12', {
      percent: 12,
      transferred: 300,
      total: 1_000,
      bytesPerSecond: 50,
    }),
    {
      phase: 'downloading',
      version: '0.1.0-preview.12',
      percent: 30,
      transferredBytes: 300,
      totalBytes: 1_000,
      bytesPerSecond: 50,
    },
  )
})

test('clamps invalid updater metrics to safe display values', () => {
  assert.deepEqual(
    normalizeUpdateProgress('0.1.0-preview.12', {
      percent: Number.NaN,
      transferred: 2_000,
      total: 1_000,
      bytesPerSecond: -10,
    }),
    {
      phase: 'downloading',
      version: '0.1.0-preview.12',
      percent: 100,
      transferredBytes: 1_000,
      totalBytes: 1_000,
      bytesPerSecond: 0,
    },
  )
})
