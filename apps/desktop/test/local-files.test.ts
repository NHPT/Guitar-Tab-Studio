import assert from 'node:assert/strict'
import { resolve } from 'node:path'
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
    resolve(
      defaults.homePath,
      'Library',
      'Containers',
      'com.tencent.QQMusicMac',
      'Data',
      'Library',
      'Application Support',
      'QQMusicMac',
      'iQmc',
    ),
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

test('opens visible Windows download folders before application caches', () => {
  const windowsDefaults = {
    ...defaults,
    platform: 'win32' as const,
    appDataPath: resolve(defaults.homePath, 'AppData', 'Roaming'),
  }
  const expectedPaths = {
    qqmusic: windowsDefaults.musicPath,
    netease: resolve(windowsDefaults.musicPath, '网易云音乐'),
    moo: resolve(windowsDefaults.musicPath, 'MOO Music'),
    kuwo: resolve(windowsDefaults.homePath, '..', '..', 'KwDownload', 'song'),
    kugou: resolve(windowsDefaults.homePath, '..', '..', 'KuGou'),
  } as const

  for (const source of desktopMediaSources) {
    const cachePath =
      source === 'qqmusic'
        ? resolve(
            windowsDefaults.appDataPath,
            'Tencent',
            'QQMusic',
            'QQMusicCache',
          )
        : source === 'netease'
          ? resolve(
              windowsDefaults.homePath,
              'AppData',
              'Local',
              'NetEase',
              'CloudMusic',
              'Cache',
            )
          : source === 'moo'
            ? resolve(
                windowsDefaults.homePath,
                'AppData',
                'Local',
                'Tencent',
                'MooMusic',
              )
            : source === 'kugou'
              ? resolve(windowsDefaults.appDataPath, 'kugou8')
              : undefined
    const existing = new Set([
      expectedPaths[source],
      ...(cachePath ? [cachePath] : []),
    ])
    assert.equal(
      localFileDefaultPath({
        ...windowsDefaults,
        kind: 'media',
        source,
        pathExists: (path) => existing.has(path),
      }),
      expectedPaths[source],
      source,
    )
  }
})

test('uses the Windows QQ Music cache only as a fallback', () => {
  const appDataPath = resolve(defaults.homePath, 'AppData', 'Roaming')
  const cachePath = resolve(
    appDataPath,
    'Tencent',
    'QQMusic',
    'QQMusicCache',
  )
  assert.equal(
    localFileDefaultPath({
      ...defaults,
      platform: 'win32',
      appDataPath,
      kind: 'media',
      source: 'qqmusic',
      pathExists: (path) => path === cachePath,
    }),
    cachePath,
  )
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
