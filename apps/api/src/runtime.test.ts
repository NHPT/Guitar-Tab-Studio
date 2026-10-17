import assert from 'node:assert/strict'
import test from 'node:test'
import { hasAnalysisCapacity } from './runtime.js'

test('reserves working space before accepting an analysis upload', () => {
  const gibibyte = 1024 * 1024 * 1024
  assert.equal(hasAnalysisCapacity(10 * 1024 * 1024, gibibyte), false)
  assert.equal(hasAnalysisCapacity(10 * 1024 * 1024, 2 * gibibyte), true)
  assert.equal(hasAnalysisCapacity(300 * 1024 * 1024, gibibyte), false)
  assert.equal(hasAnalysisCapacity(300 * 1024 * 1024, 2 * gibibyte), true)
})
