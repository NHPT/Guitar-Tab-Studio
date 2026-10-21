import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, resolve } from 'node:path'
import {
  BeatReviewRequestError,
  defaultBeatGoldRoot,
} from './beat-reviews.js'

const BACKUP_PATTERN = /^beat-pilot-v2-\d{8}T\d{6}Z(?:-\d+)?$/u

export interface BeatBackup {
  schemaVersion: 1
  id: string
  createdAt: string
  corpusVersion: string
  corpusSha256: string
  archive: {
    filename: string
    size: number
    sha256: string
  }
  verified: boolean
  restored: boolean
}

function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function backupRoot(root: string): string {
  return resolve(root, '.backups')
}

function restoreRoot(root: string): string {
  return resolve(root, '.restores')
}

function metadataPath(root: string, id: string): string {
  return resolve(backupRoot(root), `${id}.json`)
}

function archivePath(root: string, id: string): string {
  return resolve(backupRoot(root), `${id}.tar.gz`)
}

function safeBackupId(value: unknown): string {
  const id = typeof value === 'string' ? value.trim() : ''
  if (!BACKUP_PATTERN.test(id)) {
    throw new BeatReviewRequestError('备份标识无效')
  }
  return id
}

function readManifest(root: string) {
  const path = resolve(root, 'pilot-manifest.json')
  if (!existsSync(path) || lstatSync(path).isSymbolicLink()) {
    throw new BeatReviewRequestError('Pilot manifest 不存在', 404)
  }
  return JSON.parse(readFileSync(path, 'utf8')) as {
    corpus_version: string
    corpus_sha256: string
  }
}

export function createBeatBackup(
  root = defaultBeatGoldRoot(),
): BeatBackup {
  const manifest = readManifest(root)
  const timestamp = new Date()
    .toISOString()
    .replace(/[-:]/gu, '')
    .replace(/\.\d{3}Z$/u, 'Z')
  const baseId = `beat-pilot-v2-${timestamp}`
  const directory = backupRoot(root)
  mkdirSync(directory, { recursive: true })
  let id = baseId
  let suffix = 1
  while (existsSync(archivePath(root, id))) {
    id = `${baseId}-${suffix}`
    suffix += 1
  }
  const archive = archivePath(root, id)
  const result = spawnSync(
    'tar',
    [
      '-czf',
      archive,
      '-C',
      dirname(root),
      '--exclude',
      `${basename(root)}/.backups`,
      '--exclude',
      `${basename(root)}/.restores`,
      basename(root),
    ],
    { encoding: 'utf8' },
  )
  if (result.status !== 0) {
    rmSync(archive, { force: true })
    throw new BeatReviewRequestError(result.stderr || 'Pilot 备份失败', 500)
  }
  const backup: BeatBackup = {
    schemaVersion: 1,
    id,
    createdAt: new Date().toISOString(),
    corpusVersion: manifest.corpus_version,
    corpusSha256: manifest.corpus_sha256,
    archive: {
      filename: basename(archive),
      size: statSync(archive).size,
      sha256: sha256File(archive),
    },
    verified: true,
    restored: false,
  }
  writeFileSync(metadataPath(root, id), `${JSON.stringify(backup, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  return backup
}

export function listBeatBackups(
  root = defaultBeatGoldRoot(),
): BeatBackup[] {
  const directory = backupRoot(root)
  if (!existsSync(directory)) return []
  return readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .flatMap((name) => {
      const path = resolve(directory, name)
      if (lstatSync(path).isSymbolicLink()) {
        throw new BeatReviewRequestError('备份元数据路径无效', 500)
      }
      const value = JSON.parse(readFileSync(path, 'utf8')) as BeatBackup
      return value.schemaVersion === 1 && BACKUP_PATTERN.test(value.id)
        ? [value]
        : []
    })
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
}

export function verifyBeatBackup(
  idValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatBackup {
  const id = safeBackupId(idValue)
  const backup = listBeatBackups(root).find((candidate) => candidate.id === id)
  if (!backup) throw new BeatReviewRequestError('备份不存在', 404)
  const archive = archivePath(root, id)
  const valid =
    existsSync(archive) &&
    !lstatSync(archive).isSymbolicLink() &&
    statSync(archive).size === backup.archive.size &&
    sha256File(archive) === backup.archive.sha256
  if (!valid) {
    throw new BeatReviewRequestError('备份归档哈希校验失败', 409)
  }
  return backup
}

export function restoreBeatBackup(
  idValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatBackup {
  const backup = verifyBeatBackup(idValue, root)
  const destination = resolve(restoreRoot(root), backup.id)
  if (existsSync(destination)) {
    throw new BeatReviewRequestError('该备份已经完成恢复演练', 409)
  }
  mkdirSync(destination, { recursive: true })
  const result = spawnSync(
    'tar',
    ['-xzf', archivePath(root, backup.id), '-C', destination],
    { encoding: 'utf8' },
  )
  if (result.status !== 0) {
    rmSync(destination, { recursive: true, force: true })
    throw new BeatReviewRequestError(result.stderr || '备份恢复失败', 500)
  }
  const restoredPilot = resolve(destination, basename(root))
  const manifest = readManifest(restoredPilot)
  if (
    manifest.corpus_version !== backup.corpusVersion ||
    manifest.corpus_sha256 !== backup.corpusSha256
  ) {
    rmSync(destination, { recursive: true, force: true })
    throw new BeatReviewRequestError('恢复后的语料摘要不匹配', 409)
  }
  backup.restored = true
  writeFileSync(
    metadataPath(root, backup.id),
    `${JSON.stringify(backup, null, 2)}\n`,
    { encoding: 'utf8', mode: 0o600 },
  )
  return backup
}
