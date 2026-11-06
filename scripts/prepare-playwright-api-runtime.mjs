import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
} from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sourceRoot = resolve(repositoryRoot, 'apps/api/data')
const runtimeRoot = resolve(
  repositoryRoot,
  'apps/api/.cache/playwright-api-runtime',
)
const projectIds = [
  '1abfa729-5c72-4784-a5c1-ebd18b6301a0',
  '228e3288-429f-49c6-924e-fda060e366be',
  '6aa34f79-343c-48b9-b8d8-76029f6e3f8c',
  '6f579959-aebf-4d01-96c0-80de5a1eb8b7',
  'a1028519-3b28-4160-bbdb-d0f94871b2ee',
  'c2442ea5-5a8c-4e64-94b4-509b20fb3b8c',
]

rmSync(runtimeRoot, { force: true, recursive: true })

const runtimeProjectRoot = resolve(runtimeRoot, 'state/projects')
const runtimeJobRoot = resolve(runtimeRoot, 'jobs')
mkdirSync(runtimeProjectRoot, { recursive: true })
mkdirSync(runtimeJobRoot, { recursive: true })

const jobIds = new Set()
for (const projectId of projectIds) {
  const source = resolve(sourceRoot, 'state/projects', `${projectId}.json`)
  if (!existsSync(source)) {
    throw new Error(`Missing Playwright project fixture: ${projectId}`)
  }
  const project = JSON.parse(readFileSync(source, 'utf8'))
  if (project.id !== projectId) {
    throw new Error(`Playwright project fixture ID mismatch: ${projectId}`)
  }
  copyFileSync(source, resolve(runtimeProjectRoot, `${projectId}.json`))
  for (const track of project.tracks ?? []) {
    const match = track.url?.match(/^\/media\/jobs\/([^/]+)\//u)
    if (match) jobIds.add(match[1])
  }
}

for (const jobId of [...jobIds].sort()) {
  const source = resolve(sourceRoot, 'jobs', jobId)
  if (!existsSync(source)) {
    throw new Error(`Missing Playwright media fixture: ${jobId}`)
  }
  const target = resolve(runtimeJobRoot, jobId)
  symlinkSync(relative(dirname(target), source), target, 'dir')
}

console.log(
  `Prepared ${projectIds.length} Playwright projects and ${jobIds.size} media jobs.`,
)
