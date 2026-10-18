import { randomUUID } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { isAbsolute, relative, resolve } from 'node:path'
import {
  migrateStudioProject,
  parseProjectRevision,
  type InferenceProvider,
  type ProjectRepository,
  type ProjectRevision,
  type ProjectRevisionKind,
  type RevisionRepository,
} from '@gts/core'
import { PipelineInferenceProvider } from './inference.js'
import { createStudioProject } from './pipeline.js'
import { runtimeDataRoot } from './runtime.js'
import type { AnalysisJob, SourceDescriptor, StudioProject } from './types.js'

const jobs = new Map<string, AnalysisJob>()
const activeJobControllers = new Map<string, AbortController>()
const activeJobInputs = new Map<string, string>()
const stateRoot = resolve(runtimeDataRoot(), 'state')
const projectRoot = resolve(stateRoot, 'projects')
const jobRoot = resolve(stateRoot, 'jobs')
const revisionRoot = resolve(stateRoot, 'revisions')
mkdirSync(projectRoot, { recursive: true })
mkdirSync(jobRoot, { recursive: true })
mkdirSync(revisionRoot, { recursive: true })

export class ProjectRevisionConflictError extends Error {
  constructor() {
    super('项目已有更新，请先载入最新修订再保存')
    this.name = 'ProjectRevisionConflictError'
  }
}

export class JsonProjectRepository implements ProjectRepository {
  private readonly projects = new Map<string, StudioProject>()

  constructor(private readonly directory: string) {
    mkdirSync(directory, { recursive: true })
    for (const filename of readdirSync(directory)) {
      if (!filename.endsWith('.json')) continue
      try {
        const project = migrateStudioProject(
          JSON.parse(readFileSync(resolve(directory, filename), 'utf8')),
        )
        this.projects.set(project.id, project)
      } catch {
        // Ignore partial or unsupported state files.
      }
    }
  }

  getProject(id: string): StudioProject | undefined {
    if (!this.projects.has(id)) {
      const path = resolve(this.directory, `${id}.json`)
      if (existsSync(path)) {
        const project = migrateStudioProject(
          JSON.parse(readFileSync(path, 'utf8')),
        )
        this.projects.set(project.id, project)
      }
    }
    return this.projects.get(id)
  }

  listProjects(): StudioProject[] {
    return [...this.projects.values()]
  }

  saveProject(project: StudioProject): void {
    this.projects.set(project.id, project)
    writeFileSync(
      resolve(this.directory, `${project.id}.json`),
      JSON.stringify(project),
      'utf8',
    )
  }

  removeProject(id: string): void {
    this.projects.delete(id)
    const path = resolve(this.directory, `${id}.json`)
    if (existsSync(path)) unlinkSync(path)
  }
}

export class JsonRevisionRepository implements RevisionRepository {
  constructor(private readonly directory: string) {
    mkdirSync(directory, { recursive: true })
  }

  getRevision(id: string): ProjectRevision | undefined {
    if (!/^[a-zA-Z0-9_-]{1,128}$/u.test(id)) return undefined
    const path = resolve(this.directory, `${id}.json`)
    if (!existsSync(path)) return undefined
    return parseProjectRevision(JSON.parse(readFileSync(path, 'utf8')))
  }

  listRevisions(projectId: string): ProjectRevision[] {
    return this.listAllRevisions()
      .filter((revision) => revision.projectId === projectId)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
  }

  listAllRevisions(): ProjectRevision[] {
    return readdirSync(this.directory)
      .filter((filename) => filename.endsWith('.json'))
      .flatMap((filename) => {
        try {
          return [
            parseProjectRevision(
              JSON.parse(
                readFileSync(resolve(this.directory, filename), 'utf8'),
              ),
            ),
          ]
        } catch {
          return []
        }
      })
  }

  saveRevision(revision: ProjectRevision): void {
    const validated = parseProjectRevision(revision)
    writeFileSync(
      resolve(this.directory, `${validated.id}.json`),
      `${JSON.stringify(validated)}\n`,
      { encoding: 'utf8', flag: 'wx' },
    )
  }
}

function persistJob(job: AnalysisJob): void {
  writeFileSync(resolve(jobRoot, `${job.id}.json`), JSON.stringify(job), 'utf8')
}

function restoreState<T extends { id: string }>(directory: string, target: Map<string, T>): void {
  for (const filename of readdirSync(directory)) {
    if (!filename.endsWith('.json')) continue
    try {
      const value = JSON.parse(readFileSync(resolve(directory, filename), 'utf8')) as T
      target.set(value.id, value)
    } catch {
      // Ignore partial state files; the underlying analysis artifacts remain available.
    }
  }
}

restoreState(jobRoot, jobs)
for (const job of jobs.values()) {
  const recovered = recoverInterruptedJob(job)
  if (recovered !== job) {
    jobs.set(recovered.id, recovered)
    Object.assign(job, recovered)
    persistJob(job)
  }
}
const projectRepository = new JsonProjectRepository(projectRoot)
const revisionRepository = new JsonRevisionRepository(revisionRoot)
const defaultInferenceProvider = new PipelineInferenceProvider()

const demo = createStudioProject(
  {
    kind: 'netease',
    label: 'John Denver · Take Me Home, Country Roads',
    url: 'https://music.163.com/#/song?id=1478037',
  },
)

export function getDemoProject(): StudioProject {
  return demo
}

export function getProject(id: string): StudioProject | undefined {
  return projectRepository.getProject(id)
}

export function removeProject(id: string): boolean {
  const project = projectRepository.getProject(id)
  if (!project) return false
  for (const revision of revisionRepository.listRevisions(id)) {
    const path = resolve(revisionRoot, `${revision.id}.json`)
    if (existsSync(path)) unlinkSync(path)
  }
  projectRepository.removeProject(id)
  return true
}

export function listProjectRevisions(projectId: string): ProjectRevision[] {
  return revisionRepository.listRevisions(projectId)
}

export function saveProjectRevision(
  projectValue: StudioProject,
  label: string,
  kind: ProjectRevisionKind = 'manual',
  expectedParentRevisionId?: string | null,
): ProjectRevision {
  const project = migrateStudioProject(projectValue)
  const normalizedLabel = label.trim().slice(0, 80)
  if (!normalizedLabel) throw new Error('Revision label is required')
  const previous = revisionRepository.listRevisions(project.id)[0]
  if (
    expectedParentRevisionId !== undefined &&
    (previous?.id ?? null) !== expectedParentRevisionId
  ) {
    throw new ProjectRevisionConflictError()
  }
  const revision: ProjectRevision = {
    schemaVersion: 1,
    id: randomUUID(),
    projectId: project.id,
    ...(previous ? { parentRevisionId: previous.id } : {}),
    createdAt: new Date().toISOString(),
    label: normalizedLabel,
    kind,
    project,
  }
  revisionRepository.saveRevision(revision)
  projectRepository.saveProject(project)
  return revision
}

export function restoreProjectRevision(
  projectId: string,
  revisionId: string,
): ProjectRevision | undefined {
  const target = revisionRepository.getRevision(revisionId)
  if (!target || target.projectId !== projectId) return undefined
  return saveProjectRevision(
    target.project,
    `恢复：${target.label}`.slice(0, 80),
    'restore',
  )
}

export function getJob(id: string): AnalysisJob | undefined {
  return jobs.get(id)
}

export function listJobs(): AnalysisJob[] {
  return [...jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt))
}

function referencedJobId(url: string | null): string | undefined {
  if (!url) return undefined
  let pathname = url
  try {
    pathname = new URL(url).pathname
  } catch {
    // Stored local media paths are relative URLs.
  }
  return pathname.match(/^\/media\/jobs\/([^/]+)\//u)?.[1]
}

function collectReferencedJobIds(
  value: unknown,
  target: Set<string>,
): void {
  if (typeof value === 'string') {
    const jobId = referencedJobId(value)
    if (jobId) target.add(jobId)
    return
  }
  if (Array.isArray(value)) {
    for (const item of value) collectReferencedJobIds(item, target)
    return
  }
  if (!value || typeof value !== 'object') return
  for (const item of Object.values(value)) {
    collectReferencedJobIds(item, target)
  }
}

export function referencedJobIdsInStateDirectories(
  directories: string[],
): Set<string> {
  const jobIds = new Set<string>()
  for (const directory of directories) {
    if (!existsSync(directory)) continue
    for (const filename of readdirSync(directory)) {
      if (!filename.endsWith('.json')) continue
      const document = readFileSync(resolve(directory, filename), 'utf8')
      try {
        collectReferencedJobIds(JSON.parse(document), jobIds)
      } catch {
        for (const match of document.matchAll(/\/media\/jobs\/([^/"\\]+)\//gu)) {
          jobIds.add(match[1])
        }
      }
    }
  }
  return jobIds
}

export function storageCleanupProtection(): {
  jobs: AnalysisJob[]
  activeJobIds: Set<string>
  activeUploadPaths: Set<string>
  referencedJobIds: Set<string>
} {
  return {
    jobs: [...jobs.values()],
    activeJobIds: new Set(activeJobControllers.keys()),
    activeUploadPaths: new Set(activeJobInputs.values()),
    referencedJobIds: referencedJobIdsInStateDirectories([
      projectRoot,
      revisionRoot,
    ]),
  }
}

export function recoverInterruptedJob(
  job: AnalysisJob,
  recoveredAt = new Date().toISOString(),
): AnalysisJob {
  if (['completed', 'degraded', 'failed', 'cancelled'].includes(job.status)) {
    return job
  }
  return {
    ...job,
    status: 'failed',
    stageLabel: '任务已中断',
    error: '服务重启前任务未完成，请重新提交',
    updatedAt: recoveredAt,
  }
}

export function createJob(
  source: SourceDescriptor,
  inputPath?: string,
  inferenceProvider: InferenceProvider<string> = defaultInferenceProvider,
  options: {
    removeInputOnSettled?: boolean
    onSettled?: () => void
  } = {},
): AnalysisJob {
  const now = new Date().toISOString()
  const job: AnalysisJob = {
    id: randomUUID(),
    source,
    status: 'queued',
    progress: 0,
    stageLabel: '等待处理',
    createdAt: now,
    updatedAt: now,
  }

  jobs.set(job.id, job)
  persistJob(job)
  const controller = new AbortController()
  activeJobControllers.set(job.id, controller)
  if (inputPath) activeJobInputs.set(job.id, resolve(inputPath))

  void inferenceProvider
    .analyze(
      { job, media: inputPath, signal: controller.signal },
      ({ job: updated }) => {
        if (jobs.get(updated.id)?.status === 'cancelled') return
        jobs.set(updated.id, { ...updated })
        persistJob(updated)
      },
    )
    .then((project) => {
      if (jobs.get(job.id)?.status === 'cancelled') return
      saveProjectRevision(project, '初始转录', 'inference')
    })
    .catch((error: unknown) => {
      if (jobs.get(job.id)?.status === 'cancelled') return
      job.status = 'failed'
      job.error = error instanceof Error ? error.message : '处理失败'
      job.stageLabel = '处理失败'
      job.updatedAt = new Date().toISOString()
      jobs.set(job.id, { ...job })
      persistJob(job)
    })
    .finally(() => {
      activeJobControllers.delete(job.id)
      activeJobInputs.delete(job.id)
      if (options.removeInputOnSettled && inputPath) {
        const uploadRoot = resolve(runtimeDataRoot(), 'uploads')
        const resolvedInput = resolve(inputPath)
        const relativeInput = relative(uploadRoot, resolvedInput)
        if (
          relativeInput &&
          !relativeInput.startsWith('..') &&
          !isAbsolute(relativeInput) &&
          existsSync(resolvedInput)
        ) {
          try {
            unlinkSync(resolvedInput)
          } catch {
            // The periodic cleanup will retry stale managed uploads.
          }
        }
      }
      try {
        options.onSettled?.()
      } catch {
        // Cleanup failures must not change the completed analysis result.
      }
    })

  return job
}

export function cancelJob(id: string): AnalysisJob | undefined {
  const job = jobs.get(id)
  if (!job) return undefined
  if (['completed', 'degraded', 'failed', 'cancelled'].includes(job.status)) {
    return job
  }
  activeJobControllers.get(id)?.abort()
  job.status = 'cancelled'
  job.progress = Math.min(job.progress, 99)
  job.stageLabel = '任务已取消'
  job.error = undefined
  job.updatedAt = new Date().toISOString()
  jobs.set(job.id, { ...job })
  persistJob(job)
  return job
}

export function shutdownJobs(): void {
  for (const controller of activeJobControllers.values()) controller.abort()
}
