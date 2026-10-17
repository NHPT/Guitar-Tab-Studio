import { createHash, randomUUID } from 'node:crypto'
import {
  chmodSync,
  copyFileSync,
  createReadStream,
  createWriteStream,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  statSync,
  statfsSync,
  writeFileSync,
} from 'node:fs'
import { Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { dirname, relative, resolve } from 'node:path'
import { Readable } from 'node:stream'
import {
  isApplicationVersionCompatible,
  parseModelPackageManifest,
  type InstalledModelPackage,
  type ModelPackageFileRole,
  type ModelPackageManifest,
} from '@gts/core'

interface ActiveModelState {
  active: {
    id: string
    version: string
  }
  previous?: {
    id: string
    version: string
  }
  activatedAt: string
}

function safeChild(root: string, path: string): string {
  const target = resolve(root, path)
  const relativePath = relative(root, target)
  if (
    relativePath === '' ||
    relativePath.startsWith('..') ||
    relativePath.startsWith('/')
  ) {
    throw new Error('Model package path escapes its root')
  }
  return target
}

function sha256File(path: string): Promise<string> {
  return new Promise((resolveHash, rejectHash) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('error', rejectHash)
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

export class ModelPackageManager {
  private readonly activeStatePath: string
  private recoveryMessage?: string

  constructor(
    private readonly root: string,
    private readonly applicationVersion: string,
    private readonly targetPlatform = process.platform,
    private readonly targetArch = process.arch,
  ) {
    mkdirSync(root, { recursive: true })
    this.activeStatePath = resolve(root, 'active.json')
  }

  async installFromDirectory(sourceRoot: string): Promise<InstalledModelPackage> {
    const verifiedSourceRoot = realpathSync(sourceRoot)
    const manifest = parseModelPackageManifest(
      JSON.parse(
        readFileSync(
          safeChild(verifiedSourceRoot, 'manifest.json'),
          'utf8',
        ),
      ),
    )
    this.assertCompatible(manifest)
    const requiredBytes = manifest.files.reduce(
      (total, file) => total + file.sizeBytes,
      0,
    )
    const disk = statfsSync(this.root)
    const availableBytes = disk.bavail * disk.bsize
    if (availableBytes < requiredBytes * 2 + 512 * 1024 * 1024) {
      throw new Error('Insufficient disk space for verified model installation')
    }

    const destination = this.packageRoot(manifest.id, manifest.version)
    if (existsSync(destination)) {
      throw new Error('Model package version is already installed')
    }
    const staging = resolve(
      this.root,
      `.install-${manifest.id}-${randomUUID()}`,
    )
    mkdirSync(staging, { recursive: true })
    try {
      for (const file of manifest.files) {
        const source = safeChild(verifiedSourceRoot, file.path)
        const target = safeChild(staging, file.path)
        const sourceStats = existsSync(source) ? lstatSync(source) : undefined
        if (
          !sourceStats?.isFile() ||
          sourceStats.isSymbolicLink() ||
          sourceStats.size !== file.sizeBytes
        ) {
          throw new Error(`Model package file size mismatch: ${file.path}`)
        }
        const actualSha256 = await sha256File(source)
        if (actualSha256 !== file.sha256) {
          throw new Error(`Model package file hash mismatch: ${file.path}`)
        }
        mkdirSync(dirname(target), { recursive: true })
        copyFileSync(source, target)
        if (file.executable) chmodSync(target, 0o755)
      }
      writeFileSync(
        resolve(staging, 'manifest.json'),
        `${JSON.stringify(manifest, null, 2)}\n`,
        'utf8',
      )
      mkdirSync(dirname(destination), { recursive: true })
      renameSync(staging, destination)
    } catch (error) {
      rmSync(staging, { recursive: true, force: true })
      throw error
    }
    return { manifest, active: this.isActive(manifest.id, manifest.version) }
  }

  async installFromUrl(
    manifestUrl: string,
    fetchImpl: typeof fetch = fetch,
  ): Promise<InstalledModelPackage> {
    const parsedUrl = new URL(manifestUrl)
    if (parsedUrl.protocol !== 'https:') {
      throw new Error('Model package manifests must use HTTPS')
    }
    const response = await fetchImpl(parsedUrl)
    if (!response.ok) {
      throw new Error(`Model package manifest download failed (${response.status})`)
    }
    const manifest = parseModelPackageManifest(await response.json())
    this.assertCompatible(manifest)
    const requiredBytes = manifest.files.reduce(
      (total, file) => total + file.sizeBytes,
      0,
    )
    const disk = statfsSync(this.root)
    if (disk.bavail * disk.bsize < requiredBytes * 2 + 512 * 1024 * 1024) {
      throw new Error('Insufficient disk space for model package download')
    }
    const downloadRoot = resolve(
      this.root,
      '.downloads',
      `${manifest.id}-${manifest.version}`,
    )
    mkdirSync(downloadRoot, { recursive: true })
    writeFileSync(
      resolve(downloadRoot, 'manifest.json'),
      `${JSON.stringify(manifest, null, 2)}\n`,
      'utf8',
    )
    const baseUrl = new URL('.', parsedUrl)
    for (const file of manifest.files) {
      const sourceUrl = new URL(file.path, baseUrl)
      if (
        sourceUrl.origin !== baseUrl.origin ||
        !sourceUrl.pathname.startsWith(baseUrl.pathname)
      ) {
        throw new Error(
          'Model package files must stay under the manifest origin and path',
        )
      }
      await this.download(
        sourceUrl.toString(),
        safeChild(downloadRoot, file.path),
        file.sha256,
        fetchImpl,
        file.sizeBytes,
      )
    }
    const installed = await this.installFromDirectory(downloadRoot)
    rmSync(downloadRoot, { recursive: true, force: true })
    return installed
  }

  listInstalled(): InstalledModelPackage[] {
    const active = this.readActiveState()?.active
    const packages: InstalledModelPackage[] = []
    for (const id of readdirSync(this.root)) {
      if (id.startsWith('.') || id === 'active.json') continue
      const idRoot = resolve(this.root, id)
      const idStats = lstatSync(idRoot)
      if (idStats.isSymbolicLink() || !idStats.isDirectory()) continue
      for (const version of readdirSync(idRoot)) {
        const versionRoot = resolve(idRoot, version)
        const versionStats = lstatSync(versionRoot)
        if (versionStats.isSymbolicLink() || !versionStats.isDirectory()) {
          continue
        }
        const manifestPath = resolve(versionRoot, 'manifest.json')
        if (!existsSync(manifestPath)) continue
        try {
          const manifest = parseModelPackageManifest(
            JSON.parse(readFileSync(manifestPath, 'utf8')),
          )
          packages.push({
            manifest,
            active:
              active?.id === manifest.id && active.version === manifest.version,
          })
        } catch {
          // Invalid packages are excluded and cannot be activated.
        }
      }
    }
    return packages.sort((left, right) =>
      right.manifest.createdAt.localeCompare(left.manifest.createdAt),
    )
  }

  async activate(id: string, version: string): Promise<ActiveModelState> {
    const manifest = await this.verifyInstalledPackage(id, version)
    this.assertCompatible(manifest)
    const current = this.readActiveState()?.active
    const state: ActiveModelState = {
      active: { id, version },
      ...(current && (current.id !== id || current.version !== version)
        ? { previous: current }
        : {}),
      activatedAt: new Date().toISOString(),
    }
    this.writeActiveState(state)
    return state
  }

  async rollback(): Promise<ActiveModelState> {
    const current = this.readActiveState()
    if (!current?.previous) throw new Error('No previous model package to restore')
    await this.verifyInstalledPackage(
      current.previous.id,
      current.previous.version,
    )
    const state: ActiveModelState = {
      active: current.previous,
      previous: current.active,
      activatedAt: new Date().toISOString(),
    }
    this.writeActiveState(state)
    return state
  }

  async verifyActive(): Promise<void> {
    const active = this.readActiveState()?.active
    if (!active) return
    await this.verifyInstalledPackage(active.id, active.version)
  }

  async recoverActive(): Promise<void> {
    let state: ActiveModelState | undefined
    try {
      state = this.readActiveState()
    } catch {
      rmSync(this.activeStatePath, { force: true })
      this.recoveryMessage = '活动模型状态损坏，已停用并恢复内置基线'
      return
    }
    if (!state) return
    try {
      await this.verifyInstalledPackage(state.active.id, state.active.version)
      this.recoveryMessage = undefined
      return
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : 'Model package validation failed'
      if (state.previous) {
        try {
          await this.verifyInstalledPackage(
            state.previous.id,
            state.previous.version,
          )
          this.writeActiveState({
            active: state.previous,
            activatedAt: new Date().toISOString(),
          })
          this.recoveryMessage = `活动模型损坏，已回滚：${reason}`
          return
        } catch {
          // Fall through to the built-in baseline.
        }
      }
      rmSync(this.activeStatePath, { force: true })
      this.recoveryMessage = `活动模型损坏，已停用：${reason}`
    }
  }

  recoveryWarning(): string | undefined {
    return this.recoveryMessage
  }

  activeManifest(): ModelPackageManifest | undefined {
    const active = this.readActiveState()?.active
    return active
      ? this.readInstalledManifest(active.id, active.version)
      : undefined
  }

  resolveActiveFile(role: ModelPackageFileRole): string | undefined {
    const manifest = this.activeManifest()
    const file = manifest?.files.find((candidate) => candidate.role === role)
    return manifest && file
      ? safeChild(this.packageRoot(manifest.id, manifest.version), file.path)
      : undefined
  }

  remove(id: string, version: string): void {
    if (this.isActive(id, version)) {
      throw new Error('Cannot remove the active model package')
    }
    rmSync(this.packageRoot(id, version), { recursive: true, force: true })
  }

  async download(
    url: string,
    destination: string,
    expectedSha256: string,
    fetchImpl: typeof fetch = fetch,
    expectedSize?: number,
  ): Promise<string> {
    const parsed = new URL(url)
    if (parsed.protocol !== 'https:') {
      throw new Error('Model packages must be downloaded over HTTPS')
    }
    const partial = `${destination}.part`
    mkdirSync(dirname(destination), { recursive: true })
    if (
      existsSync(destination) &&
      (await sha256File(destination)) === expectedSha256
    ) {
      return destination
    }
    rmSync(destination, { force: true })
    if (
      existsSync(partial) &&
      (await sha256File(partial)) === expectedSha256
    ) {
      renameSync(partial, destination)
      return destination
    }
    let offset = existsSync(partial) ? statSync(partial).size : 0
    if (expectedSize !== undefined && offset > expectedSize) {
      rmSync(partial, { force: true })
      offset = 0
    }
    let response = await fetchImpl(url, {
      headers: offset > 0 ? { Range: `bytes=${offset}-` } : undefined,
    })
    if (offset > 0 && response.status !== 206) {
      rmSync(partial, { force: true })
      offset = 0
      response = await fetchImpl(url)
    }
    if (!response.ok || !response.body) {
      throw new Error(`Model package download failed (${response.status})`)
    }
    const contentLength = Number(response.headers.get('content-length'))
    if (
      expectedSize !== undefined &&
      Number.isFinite(contentLength) &&
      offset + contentLength > expectedSize
    ) {
      rmSync(partial, { force: true })
      throw new Error('Downloaded model package exceeds its declared size')
    }
    let receivedBytes = offset
    const sizeLimit = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        receivedBytes += chunk.byteLength
        if (expectedSize !== undefined && receivedBytes > expectedSize) {
          callback(
            new Error('Downloaded model package exceeds its declared size'),
          )
          return
        }
        callback(null, chunk)
      },
    })
    try {
      await pipeline(
        Readable.fromWeb(
          response.body as Parameters<typeof Readable.fromWeb>[0],
        ),
        sizeLimit,
        createWriteStream(partial, { flags: offset > 0 ? 'a' : 'w' }),
      )
    } catch (error) {
      if (
        error instanceof Error &&
        error.message.includes('exceeds its declared size')
      ) {
        rmSync(partial, { force: true })
      }
      throw error
    }
    if (
      expectedSize !== undefined &&
      statSync(partial).size !== expectedSize
    ) {
      rmSync(partial, { force: true })
      throw new Error('Downloaded model package size mismatch')
    }
    const actualSha256 = await sha256File(partial)
    if (actualSha256 !== expectedSha256) {
      rmSync(partial, { force: true })
      throw new Error('Downloaded model package hash mismatch')
    }
    renameSync(partial, destination)
    return destination
  }

  private packageRoot(id: string, version: string): string {
    return safeChild(this.root, `${id}/${version}`)
  }

  private readInstalledManifest(
    id: string,
    version: string,
  ): ModelPackageManifest {
    const manifestPath = resolve(this.packageRoot(id, version), 'manifest.json')
    if (!existsSync(manifestPath)) throw new Error('Model package is not installed')
    return parseModelPackageManifest(
      JSON.parse(readFileSync(manifestPath, 'utf8')),
    )
  }

  private async verifyInstalledPackage(
    id: string,
    version: string,
  ): Promise<ModelPackageManifest> {
    const manifest = this.readInstalledManifest(id, version)
    const packageRoot = this.packageRoot(id, version)
    for (const file of manifest.files) {
      const path = safeChild(packageRoot, file.path)
      const stats = existsSync(path) ? lstatSync(path) : undefined
      if (
        !stats?.isFile() ||
        stats.isSymbolicLink() ||
        stats.size !== file.sizeBytes
      ) {
        throw new Error(`Installed model package file is invalid: ${file.path}`)
      }
      if ((await sha256File(path)) !== file.sha256) {
        throw new Error(
          `Installed model package file hash mismatch: ${file.path}`,
        )
      }
    }
    return manifest
  }

  private assertCompatible(manifest: ModelPackageManifest): void {
    if (
      manifest.runtime.platform !== this.targetPlatform ||
      manifest.runtime.arch !== this.targetArch
    ) {
      throw new Error('Model package is not compatible with this platform')
    }
    if (!isApplicationVersionCompatible(manifest, this.applicationVersion)) {
      throw new Error('Model package is not compatible with this application')
    }
  }

  private isActive(id: string, version: string): boolean {
    const active = this.readActiveState()?.active
    return active?.id === id && active.version === version
  }

  private readActiveState(): ActiveModelState | undefined {
    if (!existsSync(this.activeStatePath)) return undefined
    return JSON.parse(
      readFileSync(this.activeStatePath, 'utf8'),
    ) as ActiveModelState
  }

  private writeActiveState(state: ActiveModelState): void {
    const temporary = resolve(this.root, `.active-${randomUUID()}.tmp`)
    writeFileSync(temporary, `${JSON.stringify(state, null, 2)}\n`, 'utf8')
    renameSync(temporary, this.activeStatePath)
  }
}
