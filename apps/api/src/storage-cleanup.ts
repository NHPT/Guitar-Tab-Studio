import { randomUUID } from 'node:crypto'
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statfsSync,
  writeFileSync,
} from 'node:fs'
import { isAbsolute, relative, resolve } from 'node:path'
import type {
  AnalysisJob,
  StorageCleanupConfig,
  StorageCleanupReason,
  StorageCleanupRun,
  StorageCleanupStatus,
} from '@gts/core'

const GIBIBYTE = 1024 ** 3
const HOUR_MS = 60 * 60 * 1000
const TERMINAL_STATUSES = new Set([
  'completed',
  'degraded',
  'cancelled',
  'failed',
])

export const DEFAULT_STORAGE_CLEANUP_CONFIG: StorageCleanupConfig = {
  enabled: true,
  triggerFreeBytes: 5 * GIBIBYTE,
  targetFreeBytes: 8 * GIBIBYTE,
  retentionHours: 24,
}

interface CleanupProtection {
  jobs: AnalysisJob[]
  activeJobIds: Set<string>
  activeUploadPaths: Set<string>
  referencedJobIds: Set<string>
}

interface CleanupCandidate {
  paths: string[]
  bytes: number
  modifiedMs: number
}

interface PersistedCleanupState {
  schemaVersion: 1
  config: StorageCleanupConfig
  history: StorageCleanupRun[]
}

interface PathInventory {
  bytes: number
  modifiedMs: number
  safe: boolean
}

export interface StorageCleanupOptions {
  dataRoot: string
  protection: () => CleanupProtection
  availableBytes?: () => number
  now?: () => Date
}

function safeChild(root: string, name: string): string {
  const path = resolve(root, name)
  const relativePath = relative(root, path)
  if (
    !relativePath ||
    relativePath.startsWith('..') ||
    isAbsolute(relativePath)
  ) {
    throw new Error('Cleanup path escapes the managed data root')
  }
  return path
}

function inspectPath(path: string): PathInventory {
  try {
    if (!existsSync(path)) {
      return { bytes: 0, modifiedMs: 0, safe: true }
    }
    const stats = lstatSync(path)
    if (stats.isSymbolicLink()) {
      return { bytes: 0, modifiedMs: stats.mtimeMs, safe: false }
    }
    if (stats.isFile()) {
      return { bytes: stats.size, modifiedMs: stats.mtimeMs, safe: true }
    }
    if (!stats.isDirectory()) {
      return { bytes: 0, modifiedMs: stats.mtimeMs, safe: false }
    }
    let bytes = 0
    let modifiedMs = stats.mtimeMs
    for (const name of readdirSync(path)) {
      const child = inspectPath(safeChild(path, name))
      if (!child.safe) {
        return { bytes: 0, modifiedMs, safe: false }
      }
      bytes += child.bytes
      modifiedMs = Math.max(modifiedMs, child.modifiedMs)
    }
    return { bytes, modifiedMs, safe: true }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { bytes: 0, modifiedMs: 0, safe: true }
    }
    throw error
  }
}

function parsedConfig(
  value: unknown,
  fallback: StorageCleanupConfig,
): StorageCleanupConfig {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('自动清理配置无效')
  }
  const candidate = value as Partial<StorageCleanupConfig>
  const enabled =
    candidate.enabled === undefined ? fallback.enabled : candidate.enabled
  const triggerFreeBytes =
    candidate.triggerFreeBytes === undefined
      ? fallback.triggerFreeBytes
      : candidate.triggerFreeBytes
  const targetFreeBytes =
    candidate.targetFreeBytes === undefined
      ? fallback.targetFreeBytes
      : candidate.targetFreeBytes
  const retentionHours =
    candidate.retentionHours === undefined
      ? fallback.retentionHours
      : candidate.retentionHours
  if (
    typeof enabled !== 'boolean' ||
    !Number.isSafeInteger(triggerFreeBytes) ||
    triggerFreeBytes < 256 * 1024 ** 2 ||
    triggerFreeBytes > 4 * 1024 ** 4 ||
    !Number.isSafeInteger(targetFreeBytes) ||
    targetFreeBytes < triggerFreeBytes ||
    targetFreeBytes > 4 * 1024 ** 4 ||
    !Number.isFinite(retentionHours) ||
    retentionHours < 0 ||
    retentionHours > 24 * 365
  ) {
    throw new Error('自动清理阈值或保留时间无效')
  }
  return {
    enabled,
    triggerFreeBytes,
    targetFreeBytes,
    retentionHours,
  }
}

function readPersistedState(path: string): PersistedCleanupState | undefined {
  if (!existsSync(path)) return undefined
  try {
    const value = JSON.parse(
      readFileSync(path, 'utf8'),
    ) as Partial<PersistedCleanupState>
    if (value.schemaVersion !== 1 || !Array.isArray(value.history)) {
      return undefined
    }
    return {
      schemaVersion: 1,
      config: parsedConfig(value.config, DEFAULT_STORAGE_CLEANUP_CONFIG),
      history: value.history.slice(0, 20),
    }
  } catch {
    return undefined
  }
}

export class StorageCleanupService {
  private readonly dataRoot: string
  private readonly jobsRoot: string
  private readonly uploadsRoot: string
  private readonly statePath: string
  private readonly protection: () => CleanupProtection
  private readonly available: () => number
  private readonly now: () => Date
  private config: StorageCleanupConfig
  private history: StorageCleanupRun[]
  private running = false

  constructor(options: StorageCleanupOptions) {
    this.dataRoot = resolve(options.dataRoot)
    this.jobsRoot = resolve(this.dataRoot, 'jobs')
    this.uploadsRoot = resolve(this.dataRoot, 'uploads')
    this.statePath = resolve(this.dataRoot, 'state', 'storage-cleanup.json')
    this.protection = options.protection
    this.available =
      options.availableBytes ??
      (() => {
        const fileSystem = statfsSync(this.dataRoot)
        return fileSystem.bavail * fileSystem.bsize
      })
    this.now = options.now ?? (() => new Date())
    mkdirSync(this.jobsRoot, { recursive: true })
    mkdirSync(this.uploadsRoot, { recursive: true })
    const persisted = readPersistedState(this.statePath)
    this.config = persisted?.config ?? { ...DEFAULT_STORAGE_CLEANUP_CONFIG }
    this.history = persisted?.history ?? []
  }

  status(): StorageCleanupStatus {
    const candidates = this.candidates()
    return {
      config: { ...this.config },
      availableBytes: this.available(),
      reclaimableBytes: candidates.reduce(
        (total, candidate) => total + candidate.bytes,
        0,
      ),
      reclaimableItems: candidates.length,
      running: this.running,
      lastRun: this.history[0],
      history: [...this.history],
    }
  }

  updateConfig(value: unknown): StorageCleanupStatus {
    this.config = parsedConfig(value, this.config)
    this.persist()
    return this.status()
  }

  run(
    reason: StorageCleanupReason,
    options: { force?: boolean; requiredBytes?: number } = {},
  ): StorageCleanupRun {
    const started = this.now()
    const availableBeforeBytes = this.available()
    const requiredBytes = Math.max(0, options.requiredBytes ?? 0)
    const targetBytes = Math.max(this.config.targetFreeBytes, requiredBytes)
    if (this.running) {
      return this.record({
        id: randomUUID(),
        reason,
        startedAt: started.toISOString(),
        completedAt: this.now().toISOString(),
        outcome: 'skipped',
        availableBeforeBytes,
        availableAfterBytes: availableBeforeBytes,
        removedBytes: 0,
        removedItems: 0,
        targetReached: availableBeforeBytes >= targetBytes,
        error: '已有清理任务正在运行',
      })
    }
    if (!this.config.enabled && reason !== 'manual') {
      return this.record({
        id: randomUUID(),
        reason,
        startedAt: started.toISOString(),
        completedAt: this.now().toISOString(),
        outcome: 'skipped',
        availableBeforeBytes,
        availableAfterBytes: availableBeforeBytes,
        removedBytes: 0,
        removedItems: 0,
        targetReached: availableBeforeBytes >= requiredBytes,
      })
    }

    this.running = true
    let removedBytes = 0
    let removedItems = 0
    let failure: string | undefined
    try {
      const pressure =
        availableBeforeBytes < this.config.triggerFreeBytes ||
        availableBeforeBytes < requiredBytes
      const cutoff =
        started.getTime() - this.config.retentionHours * HOUR_MS
      const candidates = this.candidates()
        .filter(
          (candidate) =>
            options.force || pressure || candidate.modifiedMs <= cutoff,
        )
        .sort((left, right) => left.modifiedMs - right.modifiedMs)
      let availableAfterBytes = availableBeforeBytes
      for (const candidate of candidates) {
        if (pressure && !options.force && availableAfterBytes >= targetBytes) {
          break
        }
        try {
          for (const path of candidate.paths) {
            rmSync(path, { recursive: true, force: true })
          }
          removedBytes += candidate.bytes
          removedItems += 1
          availableAfterBytes = this.available()
        } catch (error) {
          failure ??=
            error instanceof Error ? error.message : '清理文件失败'
        }
      }
      availableAfterBytes = this.available()
      return this.record({
        id: randomUUID(),
        reason,
        startedAt: started.toISOString(),
        completedAt: this.now().toISOString(),
        outcome: failure ? 'failed' : 'completed',
        availableBeforeBytes,
        availableAfterBytes,
        removedBytes,
        removedItems,
        targetReached:
          pressure || requiredBytes > 0
            ? availableAfterBytes >= targetBytes
            : true,
        ...(failure ? { error: failure } : {}),
      })
    } catch (error) {
      return this.record({
        id: randomUUID(),
        reason,
        startedAt: started.toISOString(),
        completedAt: this.now().toISOString(),
        outcome: 'failed',
        availableBeforeBytes,
        availableAfterBytes: this.available(),
        removedBytes,
        removedItems,
        targetReached: false,
        error: error instanceof Error ? error.message : '自动清理失败',
      })
    } finally {
      this.running = false
    }
  }

  private candidates(): CleanupCandidate[] {
    const protection = this.protection()
    const jobs = new Map(protection.jobs.map((job) => [job.id, job]))
    const candidates: CleanupCandidate[] = []
    for (const name of readdirSync(this.jobsRoot)) {
      const path = safeChild(this.jobsRoot, name)
      let stats
      try {
        stats = lstatSync(path)
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue
        throw error
      }
      if (!stats.isDirectory() || stats.isSymbolicLink()) continue
      const job = jobs.get(name)
      if (
        protection.activeJobIds.has(name) ||
        (job && !TERMINAL_STATUSES.has(job.status))
      ) {
        continue
      }
      if (protection.referencedJobIds.has(name)) {
        const transientPaths = readdirSync(path)
          .filter(
            (entry) =>
              entry === 'analysis.json' ||
              entry === 'normalized.wav' ||
              entry === 'stems' ||
              entry.startsWith('source.'),
          )
          .map((entry) => safeChild(path, entry))
        const inventory = transientPaths.reduce<PathInventory>(
          (total, transientPath) => {
            const current = inspectPath(transientPath)
            return {
              bytes: total.bytes + current.bytes,
              modifiedMs: Math.max(total.modifiedMs, current.modifiedMs),
              safe: total.safe && current.safe,
            }
          },
          { bytes: 0, modifiedMs: stats.mtimeMs, safe: true },
        )
        if (transientPaths.length > 0 && inventory.safe) {
          candidates.push({
            paths: transientPaths,
            bytes: inventory.bytes,
            modifiedMs: inventory.modifiedMs,
          })
        }
        continue
      }
      const inventory = inspectPath(path)
      if (inventory.safe) {
        candidates.push({
          paths: [path],
          bytes: inventory.bytes,
          modifiedMs: inventory.modifiedMs,
        })
      }
    }

    for (const name of readdirSync(this.uploadsRoot)) {
      if (name.startsWith('.')) continue
      const path = safeChild(this.uploadsRoot, name)
      if (protection.activeUploadPaths.has(path)) continue
      const inventory = inspectPath(path)
      if (inventory.safe) {
        candidates.push({
          paths: [path],
          bytes: inventory.bytes,
          modifiedMs: inventory.modifiedMs,
        })
      }
    }
    return candidates
  }

  private record(run: StorageCleanupRun): StorageCleanupRun {
    this.history = [run, ...this.history].slice(0, 20)
    try {
      this.persist()
    } catch {
      // Cleanup must remain effective when the disk is too full to save history.
    }
    return run
  }

  private persist(): void {
    mkdirSync(resolve(this.dataRoot, 'state'), { recursive: true })
    const temporaryPath = `${this.statePath}.tmp-${process.pid}`
    const state: PersistedCleanupState = {
      schemaVersion: 1,
      config: this.config,
      history: this.history,
    }
    writeFileSync(temporaryPath, `${JSON.stringify(state, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    renameSync(temporaryPath, this.statePath)
  }
}
