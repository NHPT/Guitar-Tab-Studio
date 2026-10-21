import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import {
  copyFileSync,
  existsSync,
  linkSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, relative, resolve } from 'node:path'
import {
  beatGoldStatus,
  BeatReviewRequestError,
  defaultBeatGoldRoot,
  listBeatGold,
} from './beat-reviews.js'

const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/

export interface BeatGoldRelease {
  schemaVersion: 1
  version: string
  createdAt: string
  corpusVersion: string
  corpusSha256: string
  clips: number
  splitCounts: Record<string, number>
  files: Array<{
    path: string
    size: number
    sha256: string
  }>
  archive: {
    filename: string
    size: number
    sha256: string
  }
}

function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function releasesRoot(root: string): string {
  return resolve(root, 'releases')
}

function safeVersion(value: unknown): string {
  const version = typeof value === 'string' ? value.trim() : ''
  if (!VERSION_PATTERN.test(version)) {
    throw new BeatReviewRequestError('金标准版本标识无效')
  }
  return version
}

function copyOrLink(source: string, target: string): void {
  mkdirSync(dirname(target), { recursive: true })
  try {
    linkSync(source, target)
  } catch {
    copyFileSync(source, target)
  }
}

function collectFiles(root: string, directory = root): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = resolve(directory, name)
    if (lstatSync(path).isSymbolicLink()) {
      throw new BeatReviewRequestError('发布数据包含符号链接', 500)
    }
    return lstatSync(path).isDirectory() ? collectFiles(root, path) : [path]
  })
}

export function createBeatGoldRelease(
  versionValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatGoldRelease {
  const version = safeVersion(versionValue)
  const status = beatGoldStatus(root)
  if (!status.manifest) {
    throw new BeatReviewRequestError('节拍 Pilot 尚未准备', 404)
  }
  const gold = listBeatGold(root)
  if (gold.length !== status.manifest.clips) {
    throw new BeatReviewRequestError(
      `金标准覆盖不足：${gold.length}/${status.manifest.clips}`,
      409,
    )
  }
  const rootDirectory = releasesRoot(root)
  const releaseDirectory = resolve(rootDirectory, version)
  const metadataPath = resolve(rootDirectory, `${version}.release.json`)
  const archivePath = resolve(rootDirectory, `${version}.tar.gz`)
  if (
    existsSync(releaseDirectory) ||
    existsSync(metadataPath) ||
    existsSync(archivePath)
  ) {
    throw new BeatReviewRequestError('该金标准版本已存在', 409)
  }
  mkdirSync(rootDirectory, { recursive: true })
  const temporary = resolve(
    rootDirectory,
    `.${version}.${process.pid}.${Date.now()}.tmp`,
  )
  mkdirSync(resolve(temporary, 'gold'), { recursive: true })
  try {
    copyFileSync(
      resolve(root, 'pilot-manifest.json'),
      resolve(temporary, 'pilot-manifest.json'),
    )
    for (const review of gold) {
      copyFileSync(
        resolve(root, 'gold', `${review.clip_id}.json`),
        resolve(temporary, 'gold', `${review.clip_id}.json`),
      )
      copyOrLink(
        resolve(root, 'gold', `${review.clip_id}.wav`),
        resolve(temporary, 'gold', `${review.clip_id}.wav`),
      )
    }
    const splitCounts = Object.fromEntries(
      ['train', 'validation', 'test'].map((split) => [
        split,
        gold.filter((review) => review.tags.includes(split)).length,
      ]),
    )
    const files = collectFiles(temporary).map((path) => ({
      path: relative(temporary, path),
      size: statSync(path).size,
      sha256: sha256File(path),
    }))
    const releaseManifest = {
      schemaVersion: 1 as const,
      version,
      createdAt: new Date().toISOString(),
      corpusVersion: status.manifest.corpusVersion,
      corpusSha256: status.manifest.corpusSha256,
      clips: gold.length,
      splitCounts,
      files,
    }
    writeFileSync(
      resolve(temporary, 'release-manifest.json'),
      `${JSON.stringify(releaseManifest, null, 2)}\n`,
      { encoding: 'utf8', mode: 0o600 },
    )
    renameSync(temporary, releaseDirectory)
    const archive = spawnSync(
      'tar',
      [
        '-czf',
        archivePath,
        '-C',
        rootDirectory,
        basename(releaseDirectory),
      ],
      { encoding: 'utf8' },
    )
    if (archive.status !== 0) {
      throw new BeatReviewRequestError(
        archive.stderr || '金标准归档失败',
        500,
      )
    }
    const metadata: BeatGoldRelease = {
      ...releaseManifest,
      archive: {
        filename: basename(archivePath),
        size: statSync(archivePath).size,
        sha256: sha256File(archivePath),
      },
    }
    writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    return metadata
  } catch (error) {
    rmSync(temporary, { recursive: true, force: true })
    if (existsSync(releaseDirectory) && !existsSync(metadataPath)) {
      rmSync(releaseDirectory, { recursive: true, force: true })
    }
    if (existsSync(archivePath) && !existsSync(metadataPath)) {
      rmSync(archivePath, { force: true })
    }
    throw error
  }
}

export function listBeatGoldReleases(
  root = defaultBeatGoldRoot(),
): BeatGoldRelease[] {
  const directory = releasesRoot(root)
  if (!existsSync(directory)) return []
  return readdirSync(directory)
    .filter((name) => name.endsWith('.release.json'))
    .map((name) => {
      const path = resolve(directory, name)
      if (lstatSync(path).isSymbolicLink()) {
        throw new BeatReviewRequestError('金标准版本路径无效', 500)
      }
      const release = JSON.parse(
        readFileSync(path, 'utf8'),
      ) as BeatGoldRelease
      if (
        release.schemaVersion !== 1 ||
        !VERSION_PATTERN.test(release.version) ||
        !SHA256_PATTERN.test(release.archive.sha256)
      ) {
        throw new BeatReviewRequestError('金标准版本元数据无效', 500)
      }
      return release
    })
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
}

export function beatGoldReleaseArchivePath(
  versionValue: unknown,
  root = defaultBeatGoldRoot(),
): string {
  const version = safeVersion(versionValue)
  const release = listBeatGoldReleases(root).find(
    (candidate) => candidate.version === version,
  )
  if (!release) throw new BeatReviewRequestError('金标准版本不存在', 404)
  const path = resolve(releasesRoot(root), release.archive.filename)
  if (
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    sha256File(path) !== release.archive.sha256
  ) {
    throw new BeatReviewRequestError('金标准归档不存在或哈希不匹配', 500)
  }
  return path
}
