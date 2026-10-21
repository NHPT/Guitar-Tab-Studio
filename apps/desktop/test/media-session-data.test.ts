import assert from 'node:assert/strict'
import test from 'node:test'
import {
  netscapeCookieFile,
  qqMusicCredentialFromCookies,
  type SessionCookie,
} from '../src/media-session-data.js'

function cookie(
  name: string,
  value: string,
  domain = '.qq.com',
): SessionCookie {
  return {
    domain,
    httpOnly: true,
    name,
    path: '/',
    secure: true,
    value,
  }
}

test('exports only accepted domains in Netscape cookie format', () => {
  const output = netscapeCookieFile(
    [
      cookie('SESSDATA', 'session-value', '.bilibili.com'),
      cookie('tracker', 'ignored', '.example.com'),
    ],
    ['bilibili.com'],
  )

  assert.match(output, /#HttpOnly_\.bilibili\.com\tTRUE\t\/\tTRUE\t0\tSESSDATA\tsession-value/u)
  assert.doesNotMatch(output, /tracker|example\.com/u)
})

test('extracts QQ Music web credentials for QQ and WeChat login', () => {
  assert.deepEqual(
    qqMusicCredentialFromCookies([
      cookie('uin', 'o0012345678'),
      cookie('qm_keyst', 'Q_H_L_web-session-key'),
      cookie('tmeLoginType', '3'),
    ]),
    {
      musicId: '12345678',
      musicKey: 'Q_H_L_web-session-key',
      loginType: 3,
    },
  )
  assert.deepEqual(
    qqMusicCredentialFromCookies([
      cookie('wxuin', '87654321'),
      cookie('qm_keyst', 'W_X_web-session-key'),
    ]),
    {
      musicId: '87654321',
      musicKey: 'W_X_web-session-key',
      loginType: 1,
    },
  )
})
