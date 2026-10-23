import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'

test('reuses isolated login sessions until application shutdown', () => {
  const source = readFileSync(
    resolve(process.cwd(), 'src', 'media-sessions.ts'),
    'utf8',
  )

  assert.match(source, /gts-media-login-\$\{target\.id\}/u)
  assert.match(
    source,
    /if \(target\.authenticated\(existingCookies\)\) return existingCookies/u,
  )
  assert.match(
    source,
    /loginSessions\.map\(\(loginSession\) =>\s*this\.clearLoginSession/u,
  )
  assert.doesNotMatch(
    source,
    /const partition = `gts-media-login-\$\{randomUUID\(\)\}`/u,
  )
})
