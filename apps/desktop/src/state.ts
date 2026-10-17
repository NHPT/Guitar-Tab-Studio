import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { backup, DatabaseSync } from 'node:sqlite'
import {
  migrateStudioProject,
  parseProjectRevision,
  type AnalysisJob,
  type DesktopRecentProject,
  type ProjectRepository,
  type ProjectRevision,
  type RevisionRepository,
  type StudioProject,
  type TranscriptionModelVariant,
  type TranscriptionProviderId,
  type TranscriptionProviderPreference,
  type TranscriptionSelectionMode,
} from '@gts/core'

export interface WindowState {
  x?: number
  y?: number
  width: number
  height: number
  maximized: boolean
}

interface JsonRow {
  json: string
}

interface ProjectRow extends JsonRow {
  id: string
  updated_at: string
}

interface RecentRow {
  project_id: string
  opened_at: string
}

interface RevisionRow extends JsonRow {
  id: string
  project_id: string
  created_at: string
}

interface SettingRow extends JsonRow {
  key: string
}

const transcriptionProviderIds = new Set<TranscriptionProviderId>([
  'basic-pitch',
  'muscriptor',
  'guitar-specialist',
  'multitrack-research',
])
const transcriptionSelectionModes = new Set<TranscriptionSelectionMode>([
  'auto',
  'manual',
])
const transcriptionVariants = new Set<TranscriptionModelVariant>([
  'small',
  'medium',
  'large',
])

function tableExists(database: DatabaseSync, name: string): boolean {
  return Boolean(
    database
      .prepare(
        `SELECT 1 FROM sqlite_master
         WHERE type = 'table' AND name = ?`,
      )
      .get(name),
  )
}

function parseJob(value: unknown): AnalysisJob {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid persisted job')
  }
  const job = value as Partial<AnalysisJob>
  if (
    typeof job.id !== 'string' ||
    typeof job.status !== 'string' ||
    typeof job.progress !== 'number' ||
    typeof job.stageLabel !== 'string' ||
    typeof job.createdAt !== 'string' ||
    typeof job.updatedAt !== 'string' ||
    !job.source
  ) {
    throw new Error('Invalid persisted job')
  }
  return job as AnalysisJob
}

export class DesktopStateStore
  implements ProjectRepository, RevisionRepository
{
  private readonly database: DatabaseSync

  constructor(path: string) {
    mkdirSync(dirname(path), { recursive: true })
    this.database = new DatabaseSync(path)
    this.database.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA foreign_keys = ON;
      CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        json TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        json TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        json TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS recent_projects (
        project_id TEXT PRIMARY KEY,
        opened_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS runtime_assignments (
        entity_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL CHECK(mode IN ('local', 'cloud'))
      );
      CREATE TABLE IF NOT EXISTS project_revisions (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS project_revisions_project_created
        ON project_revisions(project_id, created_at DESC);
    `)
  }

  getProject(id: string): StudioProject | undefined {
    const row = this.database
      .prepare('SELECT json FROM projects WHERE id = ?')
      .get(id) as JsonRow | undefined
    if (!row) return undefined
    return migrateStudioProject(JSON.parse(row.json))
  }

  listProjects(): StudioProject[] {
    const rows = this.database
      .prepare('SELECT json FROM projects ORDER BY updated_at DESC')
      .all() as unknown as JsonRow[]
    return rows.map((row) => migrateStudioProject(JSON.parse(row.json)))
  }

  saveProject(project: StudioProject): void {
    const validated = migrateStudioProject(project)
    this.database
      .prepare(
        `INSERT INTO projects (id, json, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           json = excluded.json,
           updated_at = excluded.updated_at`,
      )
      .run(
        validated.id,
        JSON.stringify(validated),
        new Date().toISOString(),
      )
  }

  removeProject(id: string): void {
    this.database.prepare('DELETE FROM projects WHERE id = ?').run(id)
    this.database
      .prepare('DELETE FROM recent_projects WHERE project_id = ?')
      .run(id)
    this.database
      .prepare('DELETE FROM runtime_assignments WHERE entity_id = ?')
      .run(id)
  }

  getRevision(id: string): ProjectRevision | undefined {
    const row = this.database
      .prepare('SELECT json FROM project_revisions WHERE id = ?')
      .get(id) as JsonRow | undefined
    return row ? parseProjectRevision(JSON.parse(row.json)) : undefined
  }

  listRevisions(projectId: string): ProjectRevision[] {
    const rows = this.database
      .prepare(
        `SELECT json FROM project_revisions
         WHERE project_id = ?
         ORDER BY created_at DESC`,
      )
      .all(projectId) as unknown as JsonRow[]
    return rows.map((row) => parseProjectRevision(JSON.parse(row.json)))
  }

  saveRevision(revision: ProjectRevision): void {
    const validated = parseProjectRevision(revision)
    this.database
      .prepare(
        `INSERT INTO project_revisions (id, project_id, json, created_at)
         VALUES (?, ?, ?, ?)`,
      )
      .run(
        validated.id,
        validated.projectId,
        JSON.stringify(validated),
        validated.createdAt,
      )
  }

  saveJob(job: AnalysisJob): void {
    this.database
      .prepare(
        `INSERT INTO jobs (id, json, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           json = excluded.json,
           updated_at = excluded.updated_at`,
      )
      .run(job.id, JSON.stringify(job), job.updatedAt)
  }

  getJob(id: string): AnalysisJob | undefined {
    const row = this.database
      .prepare('SELECT json FROM jobs WHERE id = ?')
      .get(id) as JsonRow | undefined
    return row ? parseJob(JSON.parse(row.json)) : undefined
  }

  listRecoverableJobs(): AnalysisJob[] {
    const rows = this.database
      .prepare('SELECT json FROM jobs ORDER BY updated_at DESC')
      .all() as unknown as JsonRow[]
    return rows
      .map((row) => parseJob(JSON.parse(row.json)))
      .filter((job) => !['completed', 'failed'].includes(job.status))
  }

  touchRecentProject(id: string): void {
    this.database
      .prepare(
        `INSERT INTO recent_projects (project_id, opened_at)
         VALUES (?, ?)
         ON CONFLICT(project_id) DO UPDATE SET opened_at = excluded.opened_at`,
      )
      .run(id, new Date().toISOString())
  }

  recentProjectIds(limit = 10): string[] {
    const rows = this.database
      .prepare(
        'SELECT project_id FROM recent_projects ORDER BY opened_at DESC LIMIT ?',
      )
      .all(limit) as unknown as Array<{ project_id: string }>
    return rows.map((row) => row.project_id)
  }

  recentProjects(limit = 10): DesktopRecentProject[] {
    const rows = this.database
      .prepare(
        `SELECT recent_projects.project_id, recent_projects.opened_at,
                projects.json
         FROM recent_projects
         INNER JOIN projects ON projects.id = recent_projects.project_id
         ORDER BY recent_projects.opened_at DESC
         LIMIT ?`,
      )
      .all(limit) as unknown as Array<RecentRow & JsonRow>
    return rows.map((row) => {
      const project = migrateStudioProject(JSON.parse(row.json))
      return {
        id: project.id,
        title: project.title,
        sourceLabel: project.source.label,
        openedAt: row.opened_at,
      }
    })
  }

  localProjects(): DesktopRecentProject[] {
    const rows = this.database
      .prepare(
        `SELECT projects.id, projects.json,
                COALESCE(recent_projects.opened_at, projects.updated_at) AS opened_at
         FROM projects
         LEFT JOIN recent_projects
           ON recent_projects.project_id = projects.id
         ORDER BY opened_at DESC`,
      )
      .all() as unknown as Array<RecentRow & JsonRow>
    return rows.map((row) => {
      const project = migrateStudioProject(JSON.parse(row.json))
      return {
        id: project.id,
        title: project.title,
        sourceLabel: project.source.label,
        openedAt: row.opened_at,
      }
    })
  }

  getWindowState(): WindowState | undefined {
    const row = this.database
      .prepare("SELECT json FROM settings WHERE key = 'window-state'")
      .get() as JsonRow | undefined
    if (!row) return undefined
    const value = JSON.parse(row.json) as Partial<WindowState>
    if (
      typeof value.width !== 'number' ||
      typeof value.height !== 'number' ||
      typeof value.maximized !== 'boolean'
    ) {
      return undefined
    }
    return {
      x: typeof value.x === 'number' ? value.x : undefined,
      y: typeof value.y === 'number' ? value.y : undefined,
      width: Math.max(900, value.width),
      height: Math.max(640, value.height),
      maximized: value.maximized,
    }
  }

  saveWindowState(state: WindowState): void {
    this.database
      .prepare(
        `INSERT INTO settings (key, json)
         VALUES ('window-state', ?)
         ON CONFLICT(key) DO UPDATE SET json = excluded.json`,
      )
      .run(JSON.stringify(state))
  }

  runtimeMode(entityId?: string): 'local' | 'cloud' {
    if (entityId) {
      const assignment = this.database
        .prepare('SELECT mode FROM runtime_assignments WHERE entity_id = ?')
        .get(entityId) as { mode: 'local' | 'cloud' } | undefined
      if (assignment) return assignment.mode
    }
    const setting = this.database
      .prepare("SELECT json FROM settings WHERE key = 'default-runtime-mode'")
      .get() as JsonRow | undefined
    return setting?.json === '"cloud"' ? 'cloud' : 'local'
  }

  setRuntimeMode(mode: 'local' | 'cloud', entityId?: string): void {
    if (entityId) {
      this.database
        .prepare(
          `INSERT INTO runtime_assignments (entity_id, mode)
           VALUES (?, ?)
           ON CONFLICT(entity_id) DO UPDATE SET mode = excluded.mode`,
        )
        .run(entityId, mode)
      return
    }
    this.database
      .prepare(
        `INSERT INTO settings (key, json)
         VALUES ('default-runtime-mode', ?)
         ON CONFLICT(key) DO UPDATE SET json = excluded.json`,
      )
      .run(JSON.stringify(mode))
  }

  cloudApiRoot(): string | undefined {
    return this.stringSetting('cloud-api-root')
  }

  setCloudApiRoot(root?: string): void {
    this.setStringSetting('cloud-api-root', root)
  }

  huggingFaceOAuthClientId(): string | undefined {
    return this.stringSetting('huggingface-oauth-client-id')
  }

  setHuggingFaceOAuthClientId(clientId?: string): void {
    this.setStringSetting('huggingface-oauth-client-id', clientId)
  }

  transcriptionPreference(): TranscriptionProviderPreference {
    const row = this.database
      .prepare(
        "SELECT json FROM settings WHERE key = 'transcription-provider'",
      )
      .get() as JsonRow | undefined
    if (!row) {
      return {
        providerId: 'basic-pitch',
        selectionMode: 'auto',
      }
    }
    try {
      const value = JSON.parse(row.json) as Record<string, unknown>
      if (
        !transcriptionProviderIds.has(
          value.providerId as TranscriptionProviderId,
        ) ||
        !transcriptionSelectionModes.has(
          value.selectionMode as TranscriptionSelectionMode,
        )
      ) {
        throw new Error('Invalid transcription provider preference')
      }
      const variant =
        value.variant === undefined
          ? undefined
          : (value.variant as TranscriptionModelVariant)
      if (variant !== undefined && !transcriptionVariants.has(variant)) {
        throw new Error('Invalid transcription model variant')
      }
      return {
        providerId: value.providerId as TranscriptionProviderId,
        selectionMode: value.selectionMode as TranscriptionSelectionMode,
        ...(variant ? { variant } : {}),
        ...(typeof value.licenseAcceptedAt === 'string'
          ? { licenseAcceptedAt: value.licenseAcceptedAt }
          : {}),
        ...(value.licenseAcceptedVersion === 'CC-BY-NC-4.0'
          ? { licenseAcceptedVersion: value.licenseAcceptedVersion }
          : {}),
      }
    } catch {
      return {
        providerId: 'basic-pitch',
        selectionMode: 'auto',
      }
    }
  }

  setTranscriptionPreference(
    preference: TranscriptionProviderPreference,
  ): void {
    if (
      !transcriptionProviderIds.has(preference.providerId) ||
      !transcriptionSelectionModes.has(preference.selectionMode) ||
      (preference.variant !== undefined &&
        !transcriptionVariants.has(preference.variant))
    ) {
      throw new Error('Invalid transcription provider preference')
    }
    this.database
      .prepare(
        `INSERT INTO settings (key, json)
         VALUES ('transcription-provider', ?)
         ON CONFLICT(key) DO UPDATE SET json = excluded.json`,
      )
      .run(JSON.stringify(preference))
  }

  private stringSetting(key: string): string | undefined {
    const row = this.database
      .prepare('SELECT json FROM settings WHERE key = ?')
      .get(key) as JsonRow | undefined
    if (!row) return undefined
    try {
      const value = JSON.parse(row.json)
      return typeof value === 'string' && value.trim() ? value : undefined
    } catch {
      return undefined
    }
  }

  private setStringSetting(key: string, value?: string): void {
    const normalized = value?.trim()
    if (!normalized) {
      this.database.prepare('DELETE FROM settings WHERE key = ?').run(key)
      return
    }
    this.database
      .prepare(
        `INSERT INTO settings (key, json)
         VALUES (?, ?)
         ON CONFLICT(key) DO UPDATE SET json = excluded.json`,
      )
      .run(key, JSON.stringify(normalized))
  }

  diagnostics(): {
    projects: number
    revisions: number
    jobs: Record<string, number>
  } {
    const projectCount = this.database
      .prepare('SELECT COUNT(*) AS count FROM projects')
      .get() as { count: number }
    const revisionCount = this.database
      .prepare('SELECT COUNT(*) AS count FROM project_revisions')
      .get() as { count: number }
    const jobs: Record<string, number> = {}
    const rows = this.database
      .prepare('SELECT json FROM jobs')
      .all() as unknown as JsonRow[]
    for (const row of rows) {
      const status = parseJob(JSON.parse(row.json)).status
      jobs[status] = (jobs[status] ?? 0) + 1
    }
    return {
      projects: projectCount.count,
      revisions: revisionCount.count,
      jobs,
    }
  }

  async backupTo(path: string): Promise<void> {
    mkdirSync(dirname(path), { recursive: true })
    await backup(this.database, path)
  }

  restoreFrom(path: string): void {
    const source = new DatabaseSync(path, { readOnly: true })
    try {
      const projects = source
        .prepare('SELECT id, json, updated_at FROM projects')
        .all() as unknown as ProjectRow[]
      const jobs = source
        .prepare('SELECT id, json, updated_at FROM jobs')
        .all() as unknown as ProjectRow[]
      const settings = source
        .prepare('SELECT key, json FROM settings')
        .all() as unknown as SettingRow[]
      const recents = source
        .prepare('SELECT project_id, opened_at FROM recent_projects')
        .all() as unknown as RecentRow[]
      const assignments = tableExists(source, 'runtime_assignments')
        ? (source
            .prepare('SELECT entity_id, mode FROM runtime_assignments')
            .all() as unknown as Array<{
            entity_id: string
            mode: 'local' | 'cloud'
          }>)
        : []
      const revisions = tableExists(source, 'project_revisions')
        ? (source
            .prepare(
              `SELECT id, project_id, json, created_at
               FROM project_revisions`,
            )
            .all() as unknown as RevisionRow[])
        : []
      for (const project of projects) {
        migrateStudioProject(JSON.parse(project.json))
      }
      for (const job of jobs) parseJob(JSON.parse(job.json))
      for (const revision of revisions) {
        parseProjectRevision(JSON.parse(revision.json))
      }

      this.database.exec('BEGIN IMMEDIATE')
      try {
        this.database.exec(
          'DELETE FROM recent_projects; DELETE FROM project_revisions; DELETE FROM projects; DELETE FROM jobs; DELETE FROM settings;',
        )
        this.database.exec('DELETE FROM runtime_assignments')
        const projectInsert = this.database.prepare(
          'INSERT INTO projects (id, json, updated_at) VALUES (?, ?, ?)',
        )
        for (const project of projects) {
          projectInsert.run(project.id, project.json, project.updated_at)
        }
        const jobInsert = this.database.prepare(
          'INSERT INTO jobs (id, json, updated_at) VALUES (?, ?, ?)',
        )
        for (const job of jobs) {
          jobInsert.run(job.id, job.json, job.updated_at)
        }
        const settingInsert = this.database.prepare(
          'INSERT INTO settings (key, json) VALUES (?, ?)',
        )
        for (const setting of settings) {
          settingInsert.run(setting.key, setting.json)
        }
        const recentInsert = this.database.prepare(
          'INSERT INTO recent_projects (project_id, opened_at) VALUES (?, ?)',
        )
        for (const recent of recents) {
          recentInsert.run(recent.project_id, recent.opened_at)
        }
        const assignmentInsert = this.database.prepare(
          'INSERT INTO runtime_assignments (entity_id, mode) VALUES (?, ?)',
        )
        for (const assignment of assignments) {
          assignmentInsert.run(assignment.entity_id, assignment.mode)
        }
        const revisionInsert = this.database.prepare(
          `INSERT INTO project_revisions
           (id, project_id, json, created_at) VALUES (?, ?, ?, ?)`,
        )
        for (const revision of revisions) {
          revisionInsert.run(
            revision.id,
            revision.project_id,
            revision.json,
            revision.created_at,
          )
        }
        this.database.exec('COMMIT')
      } catch (error) {
        this.database.exec('ROLLBACK')
        throw error
      }
    } finally {
      source.close()
    }
  }

  close(): void {
    this.database.close()
  }
}
