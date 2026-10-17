import assert from 'node:assert/strict'
import test from 'node:test'
import {
  desktopMediaSources,
  localFileDefaultPath,
} from '../src/local-files.js'

const defaults = {
  platform: 'darwin' as const,
  homePath: '/home/tester',
  musicPath: '/home/tester/Music',
  documentsPath: '/home/tester/Documents',
  downloadsPath: '/home/tester/Downloads',
  appDataPath: '/home/tester/AppData',
  pathExists: () => true,
}

test('uses standard system directories for ordinary file selection', () => {
  assert.equal(
    localFileDefaultPath({
      ...defaults,
      kind: 'media',
      source: 'default',
    }),
    defaults.musicPath,
  )
  assert.equal(
    localFileDefaultPath({
      ...defaults,
      kind: 'score',
      source: 'default',
    }),
    defaults.documentsPath,
  )
})

test('opens the macOS QQ Music container download directory directly', () => {
  const path = localFileDefaultPath({
    ...defaults,
    kind: 'media',
    source: 'qqmusic',
  })
  assert.equal(
    path,
    '/home/tester/Library/Containers/com.tencent.QQMusicMac/Data/Library/Application Support/QQMusicMac/iQmc',
  )
})

test('provides provider candidates on macOS, Windows and Linux', () => {
  for (const platform of ['darwin', 'win32', 'linux'] as const) {
    for (const source of desktopMediaSources) {
      const path = localFileDefaultPath({
        ...defaults,
        kind: 'media',
        source,
        platform,
      })
      assert.notEqual(path, defaults.musicPath, `${platform}:${source}`)
    }
  }
})

test('falls back to the system music directory when defaults were customized', () => {
  for (const platform of ['darwin', 'win32', 'linux'] as const) {
    for (const source of desktopMediaSources) {
      assert.equal(
        localFileDefaultPath({
          ...defaults,
          kind: 'media',
          source,
          platform,
          pathExists: () => false,
        }),
        defaults.musicPath,
      )
    }
  }
})

test('rejects platform source shortcuts for score imports', () => {
  assert.throws(
    () =>
      localFileDefaultPath({
        ...defaults,
        kind: 'score',
        source: 'qqmusic',
      }),
    /音乐平台快捷入口只支持音视频导入/u,
  )
})
