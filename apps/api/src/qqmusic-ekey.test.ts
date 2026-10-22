import assert from 'node:assert/strict'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import test from 'node:test'
import plist from 'plist'
import {
  buildQQMusicEkeyRequest,
  fetchQQMusicEkey,
  loadQQMusicMacCredential,
  loadQQMusicWindowsCredential,
  parseQQMusicCredential,
  parseQQMusicCredentialArchive,
  parseQQMusicMediaName,
  resolveQQMusicEkeyForDesktop,
} from './qqmusic-ekey.js'

const credential = {
  musicId: '12345678',
  musicKey: 'Q_H_L_fixture-session-key',
  loginType: 3 as const,
}
const mediaName = 'F0M0003w2xz20QlUZt.mflac'
const media = {
  filename: mediaName,
  songMid: '002atvb82SRmG2',
}
const fixtureEkey = 'fixtureEkeyValue0123456789+/='

test('extracts QQ Music credentials from nested plist JSON and base64 data', () => {
  const encoded = Buffer.from(
    JSON.stringify({
      str_musicid: credential.musicId,
      musickey: credential.musicKey,
      loginType: credential.loginType,
    }),
  ).toString('base64')
  assert.deepEqual(
    parseQQMusicCredential({
      Unrelated: 'value',
      AutoLoginUserInfo: encoded,
    }),
    credential,
  )
  assert.deepEqual(
    parseQQMusicCredential(
      `uin=${credential.musicId}; qm_keyst=${credential.musicKey}`,
    ),
    { ...credential, loginType: 2 },
  )
})

test('resolves the NSKeyedArchiver structure used by QQ Music for macOS', () => {
  assert.deepEqual(
    parseQQMusicCredentialArchive({
      $objects: [
        '$null',
        {
          strAuthst: { 'CF$UID': 2 },
          nUserId: { 'CF$UID': 3 },
          loginType: { 'CF$UID': 4 },
        },
        credential.musicKey,
        Number(credential.musicId),
        credential.loginType,
      ],
    }),
    credential,
  )
})

test(
  'loads the nested AutoLoginUserInfo plist used by the macOS client',
  { skip: process.platform !== 'darwin' },
  async () => {
    const homePath = resolve(process.cwd(), '.test-qqmusic-home')
    const loginPath = resolve(
      homePath,
      'Library/Containers/com.tencent.QQMusicMac/Data/Library/Preferences/com.tencent.QQMusicMac.plist',
    )
    const archive = plist.build({
      $objects: [
        '$null',
        {
          strAuthst: { 'CF$UID': 2 },
          nUserId: { 'CF$UID': 3 },
          loginType: { 'CF$UID': 4 },
        },
        credential.musicKey,
        Number(credential.musicId),
        credential.loginType,
      ],
    })
    await rm(homePath, { recursive: true, force: true })
    await mkdir(dirname(loginPath), { recursive: true })
    await writeFile(
      loginPath,
      plist.build({
        AutoLoginUserInfo: Buffer.from(archive, 'utf8'),
      }),
    )
    try {
      assert.deepEqual(
        await loadQQMusicMacCredential(homePath),
        credential,
      )
    } finally {
      await rm(homePath, { recursive: true, force: true })
    }
  },
)

test('loads a UTF-16 Windows QQ Music login fixture from UserData', async () => {
  const homePath = resolve(process.cwd(), '.test-qqmusic-windows-home')
  const loginPath = resolve(
    homePath,
    'AppData/Roaming/Tencent/QQMusic/UserData/login.ini',
  )
  await rm(homePath, { recursive: true, force: true })
  await mkdir(dirname(loginPath), { recursive: true })
  await writeFile(
    loginPath,
    Buffer.from(
      `\ufeffmusicid=${credential.musicId}\r\nstrAuthst=${credential.musicKey}\r\nloginType=${credential.loginType}\r\n`,
      'utf16le',
    ),
  )
  try {
    assert.deepEqual(
      await loadQQMusicWindowsCredential(homePath),
      credential,
    )
  } finally {
    await rm(homePath, { recursive: true, force: true })
  }
})

test('loads current Windows QQ Music Qt and binary credential formats', async () => {
  const homePath = resolve(process.cwd(), '.test-qqmusic-windows-current-home')
  const configPath = resolve(
    homePath,
    'AppData/Roaming/Tencent/QQMusic/QQMusicServiceConfig.ini',
  )
  await rm(homePath, { recursive: true, force: true })
  await mkdir(dirname(configPath), { recursive: true })
  await writeFile(
    configPath,
    [
      `str_musicid=o00${credential.musicId}`,
      `qm_keyst=@ByteArray(${encodeURIComponent(credential.musicKey)})`,
      `tmeLoginType=${credential.loginType}`,
    ].join('\r\n'),
    'utf8',
  )
  try {
    assert.deepEqual(
      await loadQQMusicWindowsCredential(homePath),
      credential,
    )
    await writeFile(
      configPath,
      Buffer.concat([
        Buffer.from('str_musicid\0', 'utf8'),
        Buffer.from(`o00${credential.musicId}\0`, 'utf8'),
        Buffer.from('qm_keyst\0', 'utf8'),
        Buffer.from(`${credential.musicKey}\0`, 'utf8'),
        Buffer.from('tmeLoginType\0', 'utf8'),
        Buffer.from(String(credential.loginType), 'utf8'),
      ]),
    )
    assert.deepEqual(
      await loadQQMusicWindowsCredential(homePath),
      credential,
    )
  } finally {
    await rm(homePath, { recursive: true, force: true })
  }
})

test('falls back to the running Windows QQ Music session', async () => {
  const homePath = resolve(process.cwd(), '.test-qqmusic-windows-runtime-home')
  const configPath = resolve(
    homePath,
    'AppData/Roaming/Tencent/QQMusic/QQMusicServiceConfig.ini',
  )
  await rm(homePath, { recursive: true, force: true })
  await mkdir(dirname(configPath), { recursive: true })
  await writeFile(
    configPath,
    `[Account]\r\nUin=${credential.musicId}\r\n`,
    'utf8',
  )
  let runtimeReads = 0
  try {
    assert.deepEqual(
      await loadQQMusicWindowsCredential(homePath, async () => {
        runtimeReads += 1
        return credential
      }),
      credential,
    )
    assert.equal(runtimeReads, 1)
  } finally {
    await rm(homePath, { recursive: true, force: true })
  }
})

test('extracts the media MID from a MusicEx encrypted filename', () => {
  assert.deepEqual(parseQQMusicMediaName(mediaName), {
    filename: mediaName,
    songMid: '003w2xz20QlUZt',
  })
  assert.deepEqual(
    parseQQMusicMediaName(
      'F0M0003w2xz20QlUZt003w2xz20QlUZt.mflac',
    ),
    {
      filename: 'F0M0003w2xz20QlUZt003w2xz20QlUZt.mflac',
      songMid: '003w2xz20QlUZt',
    },
  )
  assert.throws(
    () => parseQQMusicMediaName('../song.mflac'),
    /不包含可识别的媒体 MID/u,
  )
})

test('builds the official GetEVkey request without third-party services', () => {
  const request = buildQQMusicEkeyRequest(
    media,
    credential,
    'A'.repeat(32),
  )
  const item = request.req_1 as Record<string, unknown>
  const param = item.param as Record<string, unknown>
  const comm = request.comm as Record<string, unknown>
  assert.equal(item.module, 'music.vkey.GetEVkey')
  assert.equal(item.method, 'CgiGetEVkey')
  assert.deepEqual(param.filename, [mediaName])
  assert.deepEqual(param.songmid, [media.songMid])
  assert.deepEqual(param.songtype, [1])
  assert.equal(param.loginflag, 1)
  assert.equal(param.platform, '20')
  assert.equal(param.ctx, 1)
  assert.equal(comm.authst, credential.musicKey)
  assert.equal(comm.uin, credential.musicId)
})

test('accepts only a successful ekey response from the official endpoint', async () => {
  const received: { url?: string; body?: Record<string, unknown> } = {}
  const result = await fetchQQMusicEkey({
    credential,
    media,
    fetchImpl: (async (input, init) => {
      received.url = String(input)
      received.body = JSON.parse(String(init?.body))
      return new Response(
        JSON.stringify({
          code: 0,
          req_1: {
            code: 0,
            data: {
              midurlinfo: [
                {
                  filename: mediaName,
                  songmid: media.songMid,
                  ekey: fixtureEkey,
                  result: 0,
                },
              ],
            },
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      )
    }) as typeof fetch,
  })
  assert.equal(result, fixtureEkey)
  assert.equal(
    received.url,
    'https://u.y.qq.com/cgi-bin/musicu.fcg',
  )
  assert.ok(received.body?.req_1)
})

test('reports account authorization failures without exposing credentials', async () => {
  await assert.rejects(
    fetchQQMusicEkey({
      credential,
      media,
      fetchImpl: (async () =>
        Response.json({
          req_1: {
            code: 101404,
            data: { midurlinfo: [] },
          },
        })) as typeof fetch,
    }),
    /未接受文件中的歌曲标识/u,
  )
  await assert.rejects(
    fetchQQMusicEkey({
      credential,
      media,
      fetchImpl: (async () =>
        new Response(
          JSON.stringify({
            req_1: {
              data: {
                midurlinfo: [{ result: 104003, ekey: '' }],
              },
            },
          }),
          { status: 200 },
        )) as typeof fetch,
    }),
    (error: unknown) => {
      assert.ok(error instanceof Error)
      assert.match(error.message, /未授权当前账号/u)
      assert.doesNotMatch(error.message, /fixture-session-key/u)
      return true
    },
  )
})

test('limits automatic credential access to supported desktop runtimes', async () => {
  await assert.rejects(
    resolveQQMusicEkeyForDesktop(media, {
      env: {},
      platform: 'darwin',
    }),
    /浏览器端无法读取本机 QQ 音乐登录态/u,
  )
  await assert.rejects(
    resolveQQMusicEkeyForDesktop(media, {
      env: {
        GTS_DESKTOP_LOCAL: '1',
        GTS_DESKTOP_HOME: '/fixture-home',
      },
      platform: 'linux',
    }),
    /仅支持 macOS 和 Windows 桌面版/u,
  )
})

test('uses the Windows QQ Music request profile for keyless QMC2', async () => {
  let requestBody: Record<string, unknown> | undefined
  let userAgent = ''
  const result = await resolveQQMusicEkeyForDesktop(media, {
    env: {
      GTS_DESKTOP_LOCAL: '1',
      GTS_DESKTOP_HOME: '/fixture-home',
    },
    platform: 'win32',
    loadCredential: async () => credential,
    fetchImpl: (async (_input, init) => {
      requestBody = JSON.parse(String(init?.body))
      userAgent = new Headers(init?.headers).get('User-Agent') ?? ''
      return Response.json({
        code: 0,
        req_1: {
          code: 0,
          data: {
            midurlinfo: [
              {
                filename: media.filename,
                songmid: media.songMid,
                ekey: fixtureEkey,
                result: 0,
              },
            ],
          },
        },
      })
    }) as typeof fetch,
  })

  const request = requestBody?.req_1 as Record<string, unknown>
  const param = request.param as Record<string, unknown>
  assert.equal(result, fixtureEkey)
  assert.equal(param.platform, '27')
  assert.match(userAgent, /Windows/u)
})
