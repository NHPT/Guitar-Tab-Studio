import { createHash, randomUUID } from 'node:crypto'
import {
  cpSync,
  createReadStream,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { rename } from 'node:fs/promises'
import { basename, relative, resolve } from 'node:path'
import { DesktopStateStore } from './state.js'

interface BackupManifest {
  schemaVersion: 1
  createdAt: string
  applicationVersion: string
  files: Array<{
    path: string
    sizeBytes: number
    sha256: string
  }>
}

function safeChild(root: string, path: string): string {
  const target = resolve(root, path)
  const relativePath = relative(root, target)
  if (
    relativePath === '' ||
    relativePath.startsWith('..') ||
    relativePath.startsWith('/')
  ) {
    throw new Error('Backup path escapes its root')
  }
  return target
}

function filesIn(root: string, prefix = ''): string[] {
  if (!existsSync(root)) return []
  const result: string[] = []
  for (const name of readdirSync(resolve(root, prefix))) {
    const relativePath = prefix ? `${prefix}/${name}` : name
    const path = safeChild(root, relativePath)
    const stats = lstatSync(path)
    if (stats.isSymbolicLink()) {
      throw new Error('Backup source cannot contain symbolic links')
    }
    if (stats.isDirectory()) result.push(...filesIn(root, relativePath))
    else if (stats.isFile()) result.push(relativePath)
  }
  return result
}

function sha256(path: string): Promise<string> {
  return new Promise((resolveHash, rejectHash) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('error', rejectHash)
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

function parseBackupManifest(value: unknown): BackupManifest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid desktop backup manifest')
  }
  const manifest = value as Partial<BackupManifest>
  if (
    manifest.schemaVersion !== 1 ||
    typeof manifest.createdAt !== 'string' ||
    typeof manifest.applicationVersion !== 'string' ||
    !Array.isArray(manifest.files)
  ) {
    throw new Error('Unsupported desktop backup manifest')
  }
  for (const file of manifest.files) {
    if (
      typeof file?.path !== 'string' ||
      typeof file.sizeBytes !== 'number' ||
      !/^[0-9a-f]{64}$/u.test(file.sha256)
    ) {
      throw new Error('Invalid desktop backup inventory')
    }
  }
  return manifest as BackupManifest
}

function directorySize(root: string): number {
  return filesIn(root).reduce(
    (total, file) => total + statSync(safeChild(root, file)).size,
    0,
  )
}

async function renameWithRetry(
  source: string,
  destination: string,
): Promise<void> {
  for (let attempt = 0; attempt < 12; attempt += 1) {
    try {
      await rename(source, destination)
      return
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      const retryable = code === 'EPERM' || code === 'EACCES' || code === 'EBUSY'
      if (!retryable || attempt === 11) throw error
      await new Promise((resolveDelay) =>
        setTimeout(resolveDelay, 50 * (attempt + 1)),
      )
    }
  }
}

export class DesktopMaintenance {
  constructor(
    private readonly applicationVersion: string,
    private readonly stateStore: DesktopStateStore,
    private readonly runtimeRoot: string,
    private readonly cacheRoot: string,
  ) {}

  async createBackup(destinationParent: string): Promise<string> {
    mkdirSync(destinationParent, { recursive: true })
    const stamp = new Date().toISOString().replaceAll(/[:.]/gu, '-')
    const destination = resolve(
      destinationParent,
      `guitar-tab-studio-backup-${stamp}`,
    )
    const staging = `${destination}.partial-${randomUUID()}`
    mkdirSync(staging, { recursive: true })
    try {
      await this.stateStore.backupTo(resolve(staging, 'desktop.sqlite'))
      if (existsSync(this.runtimeRoot)) {
        cpSync(this.runtimeRoot, resolve(staging, 'runtime'), {
          recursive: true,
          errorOnExist: true,
        })
      }
      const files = await Promise.all(
        filesIn(staging).map(async (path) => ({
          path,
          sizeBytes: statSync(safeChild(staging, path)).size,
          sha256: await sha256(safeChild(staging, path)),
        })),
      )
      const manifest: BackupManifest = {
        schemaVersion: 1,
        createdAt: new Date().toISOString(),
        applicationVersion: this.applicationVersion,
        files,
      }
      writeFileSync(
        resolve(staging, 'backup-manifest.json'),
        `${JSON.stringify(manifest, null, 2)}\n`,
        'utf8',
      )
      await renameWithRetry(staging, destination)
      return basename(destination)
    } catch (error) {
      rmSync(staging, { recursive: true, force: true })
      throw error
    }
  }

  async restoreBackup(source: string): Promise<void> {
    const manifest = parseBackupManifest(
      JSON.parse(readFileSync(resolve(source, 'backup-manifest.json'), 'utf8')),
    )
    const actualFiles = filesIn(source)
      .filter((path) => path !== 'backup-manifest.json')
      .sort()
    const declaredFiles = manifest.files.map((file) => file.path).sort()
    if (
      actualFiles.length !== declaredFiles.length ||
      actualFiles.some((path, index) => path !== declaredFiles[index])
    ) {
      throw new Error('Backup inventory does not match its manifest')
    }
    for (const file of manifest.files) {
      const path = safeChild(source, file.path)
      const stats = existsSync(path) ? lstatSync(path) : undefined
      if (
        !stats?.isFile() ||
        stats.isSymbolicLink() ||
        stats.size !== file.sizeBytes
      ) {
        throw new Error(`Backup file size mismatch: ${file.path}`)
      }
      if ((await sha256(path)) !== file.sha256) {
        throw new Error(`Backup file hash mismatch: ${file.path}`)
      }
    }
    const sourceRuntime = resolve(source, 'runtime')
    let staging: string | undefined
    if (existsSync(sourceRuntime)) {
      staging = `${this.runtimeRoot}.restore-${randomUUID()}`
      rmSync(staging, { recursive: true, force: true })
      cpSync(sourceRuntime, staging, { recursive: true, errorOnExist: true })
    }
    this.stateStore.restoreFrom(resolve(source, 'desktop.sqlite'))
    if (staging) {
      rmSync(this.runtimeRoot, { recursive: true, force: true })
      await renameWithRetry(staging, this.runtimeRoot)
    }
  }

  clearCache(): number {
    const paths = [
      this.cacheRoot,
      resolve(this.runtimeRoot, 'shadow'),
      resolve(this.runtimeRoot, 'uploads'),
    ]
    const removedBytes = paths.reduce(
      (total, path) => total + (existsSync(path) ? directorySize(path) : 0),
      0,
    )
    for (const path of paths) rmSync(path, { recursive: true, force: true })
    return removedBytes
  }
}
