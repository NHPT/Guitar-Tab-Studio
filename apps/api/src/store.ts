import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createStudioProject, runPipeline } from './pipeline.js'
import type { AnalysisJob, SourceDescriptor, StudioProject } from './types.js'

const jobs = new Map<string, AnalysisJob>()
const projects = new Map<string, StudioProject>()
const stateRoot = resolve(process.cwd(), 'data', 'state')
const projectRoot = resolve(stateRoot, 'projects')
const jobRoot = resolve(stateRoot, 'jobs')
mkdirSync(projectRoot, { recursive: true })
mkdirSync(jobRoot, { recursive: true })

function persistProject(project: StudioProject): void {
  writeFileSync(resolve(projectRoot, `${project.id}.json`), JSON.stringify(project), 'utf8')
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

restoreState(projectRoot, projects)
restoreState(jobRoot, jobs)

const demo = createStudioProject(
  {
    kind: 'upload',
    label: '内置演示',
    filename: 'midnight-practice.wav',
  },
  '午夜练习段落',
)

demo.artist = 'Demo Session'
projects.set(demo.id, demo)

export function getDemoProject(): StudioProject {
  return demo
}

export function getProject(id: string): StudioProject | undefined {
  if (!projects.has(id)) {
    const path = resolve(projectRoot, `${id}.json`)
    if (existsSync(path)) {
      const project = JSON.parse(readFileSync(path, 'utf8')) as StudioProject
      projects.set(id, project)
    }
  }
  return projects.get(id)
}

export function getJob(id: string): AnalysisJob | undefined {
  return jobs.get(id)
}

export function listJobs(): AnalysisJob[] {
  return [...jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt))
}

export function createJob(source: SourceDescriptor, inputPath?: string): AnalysisJob {
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

  void runPipeline(
    job,
    (updated) => {
      jobs.set(updated.id, { ...updated })
      persistJob(updated)
    },
    inputPath,
  )
    .then((project) => {
      projects.set(project.id, project)
      persistProject(project)
    })
    .catch((error: unknown) => {
      job.status = 'failed'
      job.error = error instanceof Error ? error.message : '处理失败'
      job.stageLabel = '处理失败'
      job.updatedAt = new Date().toISOString()
      jobs.set(job.id, { ...job })
      persistJob(job)
    })

  return job
}
