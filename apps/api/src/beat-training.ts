import { createHash, randomUUID } from 'node:crypto'
import {
  copyFileSync,
  existsSync,
  linkSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { dirname, resolve } from 'node:path'
import {
  BeatReviewRequestError,
  defaultBeatGoldRoot,
  listBeatGold,
  type BeatReviewPackage,
} from './beat-reviews.js'

const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const MINIMUM_TRAIN_CLIPS = 100

export interface BeatTrainingDataset {
  schemaVersion: 1
  version: string
  createdAt: string
  clips: number
  recordings: number
  corpusVersions: string[]
  manifestSha256: string
  readyForTraining: boolean
  minimumTrainClips: number
}

export interface BeatTrainingJob {
  id: string
  datasetVersion: string
  recipe: 'beat-tempo-map-v1'
  status: 'blocked' | 'queued'
  reason?: string
  createdAt: string
  codeRevision: string
}

interface BeatTrainingState {
  schemaVersion: 1
  jobs: BeatTrainingJob[]
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex')
}

function safeVersion(value: unknown, label: string): string {
  const version = typeof value === 'string' ? value.trim() : ''
  if (!VERSION_PATTERN.test(version)) {
    throw new BeatReviewRequestError(`${label}无效`)
  }
  return version
}

function trainingRoot(root: string): string {
  return resolve(root, 'training-datasets')
}

function datasetsRoot(root: string): string {
  return resolve(trainingRoot(root), 'datasets')
}

function statePath(root: string): string {
  return resolve(trainingRoot(root), 'state.json')
}

function copyOrLink(source: string, target: string): void {
  mkdirSync(dirname(target), { recursive: true })
  try {
    linkSync(source, target)
  } catch {
    copyFileSync(source, target)
  }
}

function datasetMetadataPath(root: string, version: string): string {
  return resolve(datasetsRoot(root), version, 'dataset.json')
}

function readState(root: string): BeatTrainingState {
  const path = statePath(root)
  if (!existsSync(path)) return { schemaVersion: 1, jobs: [] }
  return JSON.parse(readFileSync(path, 'utf8')) as BeatTrainingState
}

function writeState(root: string, state: BeatTrainingState): void {
  const path = statePath(root)
  mkdirSync(dirname(path), { recursive: true })
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(state, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  renameSync(temporary, path)
}

function trainingRecord(review: BeatReviewPackage) {
  const beats = review.beats
    .filter(
      (marker) =>
        marker.at >= review.evaluation_start &&
        marker.at < review.evaluation_end,
    )
    .map((marker) => ({
      ...marker,
      at: Math.round((marker.at - review.evaluation_start) * 1_000_000) / 1_000_000,
    }))
  return {
    schema_version: 1,
    task: 'beat-grid',
    track_id: review.clip_id,
    group_id: review.group_id,
    audio: `audio/${review.clip_id}.wav`,
    split: 'train',
    audio_window: {
      start: review.evaluation_start,
      end: review.evaluation_end,
      duration: review.evaluation_duration,
    },
    beats,
    time_signature: review.time_signature,
    pickup_beats: review.pickup_beats,
    provenance: {
      corpus_version: review.corpus_version,
      source_audio_sha256: review.source_audio_sha256,
      review_audio_sha256: review.audio_sha256,
      verification_status: review.verification_status,
      adjudication_sha256:
        typeof review.adjudication?.content_sha256 === 'string'
          ? review.adjudication.content_sha256
          : null,
    },
  }
}

export function createBeatTrainingDataset(
  versionValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatTrainingDataset {
  const version = safeVersion(versionValue, '训练数据版本')
  const directory = resolve(datasetsRoot(root), version)
  if (existsSync(directory)) {
    throw new BeatReviewRequestError('训练数据版本已存在', 409)
  }
  const gold = listBeatGold(root)
  const train = gold.filter((review) => review.tags.includes('train'))
  if (train.length === 0) {
    throw new BeatReviewRequestError('尚无可用的 train 金标准', 409)
  }
  const nonTrainGroups = new Set(
    gold
      .filter((review) => !review.tags.includes('train'))
      .map((review) => review.group_id),
  )
  if (train.some((review) => nonTrainGroups.has(review.group_id))) {
    throw new BeatReviewRequestError('检测到跨分区录音泄漏', 409)
  }
  const temporary = `${directory}.${process.pid}.${Date.now()}.tmp`
  mkdirSync(resolve(temporary, 'audio'), { recursive: true })
  try {
    const records = train.map(trainingRecord)
    for (const review of train) {
      copyOrLink(
        resolve(root, 'gold', `${review.clip_id}.wav`),
        resolve(temporary, 'audio', `${review.clip_id}.wav`),
      )
    }
    const manifest =
      records.map((record) => JSON.stringify(record)).join('\n') + '\n'
    writeFileSync(resolve(temporary, 'manifest.jsonl'), manifest, {
      encoding: 'utf8',
      mode: 0o600,
    })
    const metadata: BeatTrainingDataset = {
      schemaVersion: 1,
      version,
      createdAt: new Date().toISOString(),
      clips: train.length,
      recordings: new Set(train.map((review) => review.group_id)).size,
      corpusVersions: [...new Set(train.map((review) => review.corpus_version))],
      manifestSha256: sha256(manifest),
      readyForTraining: train.length >= MINIMUM_TRAIN_CLIPS,
      minimumTrainClips: MINIMUM_TRAIN_CLIPS,
    }
    writeFileSync(
      resolve(temporary, 'dataset.json'),
      `${JSON.stringify(metadata, null, 2)}\n`,
      { encoding: 'utf8', mode: 0o600 },
    )
    mkdirSync(dirname(directory), { recursive: true })
    renameSync(temporary, directory)
    return metadata
  } catch (error) {
    rmSync(temporary, { recursive: true, force: true })
    throw error
  }
}

export function listBeatTrainingDatasets(
  root = defaultBeatGoldRoot(),
): BeatTrainingDataset[] {
  const directory = datasetsRoot(root)
  if (!existsSync(directory)) return []
  return readdirSync(directory)
    .filter((version) => VERSION_PATTERN.test(version))
    .flatMap((version) => {
      const path = datasetMetadataPath(root, version)
      return existsSync(path)
        ? [JSON.parse(readFileSync(path, 'utf8')) as BeatTrainingDataset]
        : []
    })
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
}

export function createBeatTrainingJob(
  input: {
    datasetVersion?: unknown
    codeRevision?: unknown
  },
  root = defaultBeatGoldRoot(),
): BeatTrainingJob {
  const datasetVersion = safeVersion(
    input.datasetVersion,
    '训练数据版本',
  )
  const codeRevision = safeVersion(input.codeRevision, '代码版本')
  const dataset = listBeatTrainingDatasets(root).find(
    (candidate) => candidate.version === datasetVersion,
  )
  if (!dataset) throw new BeatReviewRequestError('训练数据版本不存在', 404)
  const job: BeatTrainingJob = {
    id: randomUUID(),
    datasetVersion,
    recipe: 'beat-tempo-map-v1',
    status: dataset.readyForTraining ? 'queued' : 'blocked',
    ...(dataset.readyForTraining
      ? {}
      : {
          reason: `train 金标准不足 ${dataset.minimumTrainClips} 段`,
        }),
    createdAt: new Date().toISOString(),
    codeRevision,
  }
  const state = readState(root)
  state.jobs.unshift(job)
  writeState(root, state)
  return job
}

export function beatTrainingStatus(root = defaultBeatGoldRoot()) {
  const datasets = listBeatTrainingDatasets(root)
  const state = readState(root)
  return {
    minimumTrainClips: MINIMUM_TRAIN_CLIPS,
    adjudicatedTrainClips: listBeatGold(root).filter((review) =>
      review.tags.includes('train'),
    ).length,
    datasets,
    jobs: state.jobs,
  }
}
