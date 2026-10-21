import { basename, resolve } from 'node:path'
import type { DesktopLocalFileSource } from '@gts/core'

interface LocalFileDefaultPathOptions {
  kind: 'media' | 'score'
  source: DesktopLocalFileSource
  platform: NodeJS.Platform
  homePath: string
  musicPath: string
  documentsPath: string
  downloadsPath: string
  appDataPath: string
  pathExists: (path: string) => boolean
}

export const desktopMediaSources = [
  'qqmusic',
  'netease',
  'moo',
  'kuwo',
  'kugou',
] as const satisfies ReadonlyArray<DesktopLocalFileSource>

export type DesktopMediaSource = (typeof desktopMediaSources)[number]

export const desktopMediaSourceLabels: Record<DesktopMediaSource, string> = {
  qqmusic: 'QQ 音乐',
  netease: '网易云音乐',
  moo: 'MOO 音乐',
  kuwo: '酷我音乐',
  kugou: '酷狗音乐',
}

const sourceDirectoryNames: Record<DesktopMediaSource, string[]> = {
  qqmusic: ['QQMusic', 'QQ 音乐'],
  netease: ['CloudMusic', '网易云音乐'],
  moo: ['MOO Music', 'MooMusic', 'MOO 音乐'],
  kuwo: ['KwDownload', '酷我音乐'],
  kugou: ['KuGou', 'KugouMusic', '酷狗音乐'],
}

function platformCandidates(
  source: DesktopMediaSource,
  platform: NodeJS.Platform,
  homePath: string,
  musicPath: string,
  documentsPath: string,
  downloadsPath: string,
  appDataPath: string,
): string[] {
  if (platform === 'darwin') {
    const containers = resolve(homePath, 'Library', 'Containers')
    const applicationSupport = resolve(homePath, 'Library', 'Application Support')
    const candidates: Record<DesktopMediaSource, string[]> = {
      qqmusic: [
        resolve(
          containers,
          'com.tencent.QQMusicMac',
          'Data',
          'Library',
          'Application Support',
          'QQMusicMac',
          'iQmc',
        ),
        resolve(applicationSupport, 'com.tencent.QQMusicMac'),
      ],
      netease: [
        resolve(
          containers,
          'com.netease.163music',
          'Data',
          'Documents',
          'storage',
        ),
        resolve(applicationSupport, 'com.netease.163music'),
      ],
      moo: [
        resolve(containers, 'com.tencent.MooMusic', 'Data', 'Documents'),
        resolve(containers, 'com.tencent.MOOMusic', 'Data', 'Documents'),
        resolve(applicationSupport, 'MooMusic'),
      ],
      kuwo: [
        resolve(musicPath, 'KwDownload', 'song'),
        resolve(homePath, 'KwDownload', 'song'),
      ],
      kugou: [
        resolve(containers, 'com.kugou.KGMusic', 'Data', 'Documents'),
        resolve(containers, 'com.kugou.mac.KGMusic', 'Data', 'Documents'),
        resolve(applicationSupport, 'KuGou'),
      ],
    }
    return candidates[source]
  }

  if (platform === 'win32') {
    const localAppData = resolve(homePath, 'AppData', 'Local')
    const windowsRoot = resolve(homePath, '..', '..')
    const publicMusic = resolve(
      windowsRoot,
      'Users',
      'Public',
      'Music',
    )
    const candidates: Record<DesktopMediaSource, string[]> = {
      qqmusic: [
        resolve(musicPath, 'QQMusic'),
        resolve(musicPath, 'QQ 音乐'),
        resolve(publicMusic, 'QQMusicData', 'Song'),
        resolve(publicMusic, 'QQMusic'),
        musicPath,
        resolve(downloadsPath, 'QQMusic'),
        resolve(documentsPath, 'Tencent', 'QQMusic'),
        resolve(appDataPath, 'Tencent', 'QQMusic', 'QQMusicCache'),
        resolve(localAppData, 'Tencent', 'QQMusic', 'Cache'),
        resolve(appDataPath, 'Tencent', 'QQMusic'),
      ],
      netease: [
        resolve(musicPath, '网易云音乐'),
        resolve(musicPath, 'CloudMusic'),
        resolve(documentsPath, 'CloudMusic'),
        resolve(downloadsPath, '网易云音乐'),
        resolve(downloadsPath, 'CloudMusic'),
        musicPath,
        resolve(localAppData, 'NetEase', 'CloudMusic', 'Cache'),
        resolve(appDataPath, 'NetEase', 'CloudMusic'),
      ],
      moo: [
        resolve(musicPath, 'MOO Music'),
        resolve(musicPath, 'MooMusic'),
        resolve(musicPath, 'MOO 音乐'),
        resolve(downloadsPath, 'MOO Music'),
        musicPath,
        resolve(localAppData, 'Tencent', 'MooMusic'),
        resolve(appDataPath, 'Tencent', 'MooMusic'),
      ],
      kuwo: [
        resolve(windowsRoot, 'KwDownload', 'song'),
        resolve(homePath, 'KwDownload', 'song'),
        resolve(musicPath, 'KwDownload', 'song'),
        resolve(musicPath, 'KwDownload'),
        resolve(downloadsPath, 'KwDownload'),
        musicPath,
      ],
      kugou: [
        resolve(windowsRoot, 'KuGou'),
        resolve(homePath, 'KuGou'),
        resolve(musicPath, 'KuGou'),
        resolve(musicPath, 'KugouMusic'),
        resolve(downloadsPath, 'KuGou'),
        musicPath,
        resolve(appDataPath, 'kugou8'),
        resolve(localAppData, 'KuGou'),
      ],
    }
    return candidates[source]
  }

  const wineHome = resolve(
    homePath,
    '.wine',
    'drive_c',
    'users',
    basename(homePath),
  )
  const candidates: Record<DesktopMediaSource, string[]> = {
    qqmusic: [
      resolve(
        wineHome,
        'AppData',
        'Local',
        'Tencent',
        'QQMusic',
        'Cache',
      ),
      resolve(homePath, '.local', 'share', 'qqmusic'),
    ],
    netease: [
      resolve(homePath, '.cache', 'netease-cloud-music'),
      resolve(homePath, '.config', 'netease-cloud-music'),
      resolve(
        wineHome,
        'AppData',
        'Local',
        'NetEase',
        'CloudMusic',
        'Cache',
      ),
    ],
    moo: [resolve(homePath, '.local', 'share', 'moomusic')],
    kuwo: [
      resolve(wineHome, 'KwDownload', 'song'),
      resolve(homePath, 'KwDownload', 'song'),
    ],
    kugou: [
      resolve(wineHome, 'KuGou'),
      resolve(homePath, '.local', 'share', 'kugou'),
    ],
  }
  return candidates[source]
}

export function localFileDefaultPath({
  kind,
  source,
  platform,
  homePath,
  musicPath,
  documentsPath,
  downloadsPath,
  appDataPath,
  pathExists,
}: LocalFileDefaultPathOptions): string {
  if (source === 'default') {
    return kind === 'media' ? musicPath : documentsPath
  }
  if (kind !== 'media') {
    throw new Error('音乐平台快捷入口只支持音视频导入')
  }
  const candidates = [
    ...platformCandidates(
      source,
      platform,
      homePath,
      musicPath,
      documentsPath,
      downloadsPath,
      appDataPath,
    ),
    ...sourceDirectoryNames[source].flatMap((directory) => [
      resolve(musicPath, directory),
      resolve(downloadsPath, directory),
      resolve(documentsPath, directory),
    ]),
  ]
  return [...new Set(candidates)].find(pathExists) ?? musicPath
}
