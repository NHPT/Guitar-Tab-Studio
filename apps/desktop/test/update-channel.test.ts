import assert from 'node:assert/strict'
import test from 'node:test'
import { acceptsPrereleaseUpdates } from '../src/update-channel.js'

test('keeps preview builds on the prerelease update channel', () => {
  assert.equal(acceptsPrereleaseUpdates('0.1.0-preview.5'), true)
  assert.equal(acceptsPrereleaseUpdates('1.0.0-beta.2'), true)
  assert.equal(acceptsPrereleaseUpdates('1.0.0'), false)
})
