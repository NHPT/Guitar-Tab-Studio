import { createHash } from 'node:crypto'
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { dirname, isAbsolute, resolve } from 'node:path'
import {
  BeatReviewRequestError,
  defaultBeatGoldRoot,
  type BeatSubmissionIdentity,
} from './beat-reviews.js'

const SAFE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const REVIEWER_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const REQUIRED_CHECKS = [
  'meter',
  'downbeat_phase',
  'pickup',
  'meter_changes',
] as const

export interface BeatStructureChecks {
  meter: boolean
  downbeat_phase: boolean
  pickup: boolean
  meter_changes: boolean
}

export interface BeatMeterChange {
  at: number
  time_signature: [number, number]
}

export interface BeatStructureReviewPackage {
  schema_version: 1
  review_kind: 'beat-structure'
  status: 'draft' | 'submitted'
  corpus_version: string
  recording_id: string
  annotator_alias: string
  source_audio_sha256: string
  audio_duration: number
  time_signature: [number, number]
  pickup_beats: number
  phase_status: 'confirmed' | 'uncertain'
  first_downbeat_at: number | null
  meter_changes: BeatMeterChange[]
  checks: BeatStructureChecks
  candidate_source: {
    kind: 'worker-analysis'
    analysis_sha256: string
  }
  submission: {
    submitted_at: string
    content_sha256: string
    admin_user_id?: string
    admin_username?: string
  } | null
}

export interface BeatStructureSummary {
  recordingId: string
  split: 'train' | 'validation' | 'test'
  status: BeatStructureReviewPackage['status']
  duration: number
  timeSignature: [number, number]
  pickupBeats: number
  phaseStatus: BeatStructureReviewPackage['phase_status']
  meterChangeCount: number
  checks: BeatStructureChecks
}

export interface BeatStructureDocument {
  review: BeatStructureReviewPackage
  audioUrl: string
  candidateBeats: number[]
  candidateBpm: number
}

export interface BeatStructureDraftInput {
  time_signature?: unknown
  pickup_beats?: unknown
  phase_status?: unknown
  first_downbeat_at?: unknown
  meter_changes?: unknown
  checks?: unknown
}

interface RecordingCandidate {
  recordingId: string
  split: BeatStructureSummary['split']
  corpusVersion: string
  sourceAudioSha256: string
  audioDuration: number
  timeSignature: [number, number]
  pickupBeats: number
  beats: number[]
  bpm: number
  analysisSha256: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finiteNumber(value: unknown, label: string): number {
  const number = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(number)) {
    throw new BeatReviewRequestError(`${label}必须是有限数字`)
  }
  return number
}

function integer(value: unknown, label: string): number {
  const number = finiteNumber(value, label)
  if (!Number.isInteger(number)) {
    throw new BeatReviewRequestError(`${label}必须是整数`)
  }
  return number
}

function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function canonicalJson(value: unknown): string {
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'string'
  ) {
    return JSON.stringify(value)
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new BeatReviewRequestError('结构审核包含无效数字')
    }
    if (Object.is(value, -0) || value === 0) return '0'
    return value.toFixed(12).replace(/(?:\.0+|(\.\d+?)0+)$/u, '$1')
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`
  }
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(',')}}`
  }
  throw new BeatReviewRequestError('结构审核包含不支持的值')
}

export function beatStructureContentDigest(
  review: BeatStructureReviewPackage,
): string {
  const payload = Object.fromEntries(
    Object.entries(review).filter(
      ([key]) => !['status', 'submission'].includes(key),
    ),
  )
  return createHash('sha256')
    .update(canonicalJson(payload), 'utf8')
    .digest('hex')
}

export function defaultBeatSourceRoot(
  beatGoldRoot = defaultBeatGoldRoot(),
): string {
  const configured = process.env.GTS_BEAT_SOURCE_ROOT
  if (configured) {
    if (!isAbsolute(configured)) {
      throw new Error('GTS_BEAT_SOURCE_ROOT 必须是绝对路径')
    }
    return resolve(configured)
  }
  return resolve(beatGoldRoot, '../sources')
}

function normalizeReviewer(value: unknown): string {
  const reviewer = typeof value === 'string' ? value.trim() : ''
  if (!REVIEWER_PATTERN.test(reviewer)) {
    throw new BeatReviewRequestError('审核者标识无效')
  }
  return reviewer
}

function normalizeTimeSignature(value: unknown): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new BeatReviewRequestError('拍号无效')
  }
  const numerator = integer(value[0], '拍号分子')
  const denominator = integer(value[1], '拍号分母')
  if (
    numerator < 1 ||
    numerator > 16 ||
    ![2, 4, 8, 16].includes(denominator)
  ) {
    throw new BeatReviewRequestError('拍号超出支持范围')
  }
  return [numerator, denominator]
}

function normalizeChecks(value: unknown): BeatStructureChecks {
  if (
    !isRecord(value) ||
    REQUIRED_CHECKS.some((key) => typeof value[key] !== 'boolean')
  ) {
    throw new BeatReviewRequestError('结构审核检查项无效')
  }
  return {
    meter: value.meter as boolean,
    downbeat_phase: value.downbeat_phase as boolean,
    pickup: value.pickup as boolean,
    meter_changes: value.meter_changes as boolean,
  }
}

function readJson(path: string, label: string): Record<string, unknown> {
  if (
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    !lstatSync(path).isFile()
  ) {
    throw new BeatReviewRequestError(`${label}不存在`, 404)
  }
  try {
    const value: unknown = JSON.parse(readFileSync(path, 'utf8'))
    if (!isRecord(value)) throw new Error('invalid')
    return value
  } catch {
    throw new BeatReviewRequestError(`${label}无法读取`, 500)
  }
}

function recordingCandidates(root: string): RecordingCandidate[] {
  const manifest = readJson(resolve(root, 'pilot-manifest.json'), 'Pilot 清单')
  if (
    manifest.kind !== 'beat-gold-pilot-v2' ||
    !SAFE_ID_PATTERN.test(String(manifest.corpus_version ?? '')) ||
    !Array.isArray(manifest.items)
  ) {
    throw new BeatReviewRequestError('Pilot 清单格式无效', 500)
  }
  const recordings = new Map<
    string,
    BeatStructureSummary['split']
  >()
  for (const item of manifest.items) {
    if (
      !isRecord(item) ||
      !SAFE_ID_PATTERN.test(String(item.recording_id ?? '')) ||
      !['train', 'validation', 'test'].includes(String(item.split ?? ''))
    ) {
      throw new BeatReviewRequestError('Pilot 录音清单无效', 500)
    }
    const recordingId = String(item.recording_id)
    const split = item.split as BeatStructureSummary['split']
    const existing = recordings.get(recordingId)
    if (existing && existing !== split) {
      throw new BeatReviewRequestError('同一录音不能跨越数据分区', 500)
    }
    recordings.set(recordingId, split)
  }
  return [...recordings.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([recordingId, split]) => {
      const analysisPath = resolve(root, 'analysis', `${recordingId}.json`)
      if (dirname(analysisPath) !== resolve(root, 'analysis')) {
        throw new BeatReviewRequestError('候选分析路径无效', 500)
      }
      const analysis = readJson(analysisPath, '全曲候选分析')
      const beats = Array.isArray(analysis.beats)
        ? analysis.beats.map((beat, index) => {
            const at = finiteNumber(beat, `第 ${index + 1} 个候选拍点`)
            if (at < 0) {
              throw new BeatReviewRequestError('候选拍点无效', 500)
            }
            return at
          })
        : []
      if (beats.length < 2 || beats.length > 10_000) {
        throw new BeatReviewRequestError('全曲候选拍点数量无效', 500)
      }
      const sourceAudioSha256 = String(
        analysis.source_audio_sha256 ?? '',
      )
      if (!SHA256_PATTERN.test(sourceAudioSha256)) {
        throw new BeatReviewRequestError('全曲来源摘要无效', 500)
      }
      const timeSignature = normalizeTimeSignature(
        analysis.time_signature ?? [4, 4],
      )
      const pickupBeats = integer(
        analysis.pickup_beats ?? 0,
        '候选弱起拍数',
      )
      if (pickupBeats < 0 || pickupBeats >= timeSignature[0]) {
        throw new BeatReviewRequestError('候选弱起拍数无效', 500)
      }
      return {
        recordingId,
        split,
        corpusVersion: String(manifest.corpus_version),
        sourceAudioSha256,
        audioDuration: finiteNumber(analysis.duration, '全曲时长'),
        timeSignature,
        pickupBeats,
        beats,
        bpm: finiteNumber(analysis.bpm, '候选速度'),
        analysisSha256: sha256File(analysisPath),
      }
    })
}

function candidateFor(
  root: string,
  recordingId: string,
): RecordingCandidate {
  if (!SAFE_ID_PATTERN.test(recordingId)) {
    throw new BeatReviewRequestError('录音标识无效')
  }
  const candidate = recordingCandidates(root).find(
    (item) => item.recordingId === recordingId,
  )
  if (!candidate) throw new BeatReviewRequestError('全曲录音不存在', 404)
  return candidate
}

function reviewerExists(root: string, reviewer: string): void {
  const path = resolve(root, 'reviews', reviewer)
  if (
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    !lstatSync(path).isDirectory()
  ) {
    throw new BeatReviewRequestError('审核者工作区不存在', 404)
  }
}

function reviewDirectory(root: string, reviewer: string): string {
  reviewerExists(root, reviewer)
  const path = resolve(root, 'structure-reviews', reviewer)
  mkdirSync(path, { recursive: true, mode: 0o700 })
  if (lstatSync(path).isSymbolicLink()) {
    throw new BeatReviewRequestError('结构审核路径无效', 500)
  }
  return path
}

function reviewPath(
  root: string,
  reviewer: string,
  recordingId: string,
): string {
  candidateFor(root, recordingId)
  const directory = reviewDirectory(root, reviewer)
  const path = resolve(directory, `${recordingId}.json`)
  if (dirname(path) !== directory) {
    throw new BeatReviewRequestError('结构审核路径无效', 500)
  }
  return path
}

function matchesCandidateBeat(
  at: number,
  beats: readonly number[],
): boolean {
  return beats.some((beat) => Math.abs(beat - at) <= 0.001)
}

function normalizeMeterChanges(
  value: unknown,
  candidate: RecordingCandidate,
): BeatMeterChange[] {
  if (!Array.isArray(value) || value.length > 32) {
    throw new BeatReviewRequestError('变拍列表无效')
  }
  const changes = value.map((item, index) => {
    if (!isRecord(item)) {
      throw new BeatReviewRequestError(`第 ${index + 1} 个变拍点无效`)
    }
    const at = finiteNumber(item.at, `第 ${index + 1} 个变拍点`)
    if (
      at <= 0 ||
      at >= candidate.audioDuration ||
      !matchesCandidateBeat(at, candidate.beats)
    ) {
      throw new BeatReviewRequestError(
        `第 ${index + 1} 个变拍点必须落在候选拍点上`,
      )
    }
    return {
      at: Math.round(at * 1_000_000) / 1_000_000,
      time_signature: normalizeTimeSignature(item.time_signature),
    }
  })
  changes.sort((left, right) => left.at - right.at)
  for (let index = 1; index < changes.length; index += 1) {
    if (changes[index].at - changes[index - 1].at < 0.03) {
      throw new BeatReviewRequestError('相邻变拍点必须至少间隔 30ms')
    }
  }
  return changes
}

function validateReview(
  value: unknown,
  candidate: RecordingCandidate,
  reviewer: string,
): BeatStructureReviewPackage {
  if (
    !isRecord(value) ||
    value.schema_version !== 1 ||
    value.review_kind !== 'beat-structure' ||
    value.recording_id !== candidate.recordingId ||
    value.annotator_alias !== reviewer ||
    value.corpus_version !== candidate.corpusVersion ||
    value.source_audio_sha256 !== candidate.sourceAudioSha256 ||
    Math.abs(
      finiteNumber(value.audio_duration, '全曲时长') -
        candidate.audioDuration,
    ) > 0.00001
  ) {
    throw new BeatReviewRequestError('全曲结构审核文件格式无效', 500)
  }
  const timeSignature = normalizeTimeSignature(value.time_signature)
  const pickupBeats = integer(value.pickup_beats, '弱起拍数')
  if (pickupBeats < 0 || pickupBeats >= timeSignature[0]) {
    throw new BeatReviewRequestError('弱起拍数必须小于每小节拍数')
  }
  if (!['confirmed', 'uncertain'].includes(String(value.phase_status))) {
    throw new BeatReviewRequestError('强拍相位状态无效')
  }
  const phaseStatus =
    value.phase_status as BeatStructureReviewPackage['phase_status']
  const firstDownbeatAt =
    value.first_downbeat_at === null
      ? null
      : finiteNumber(value.first_downbeat_at, '首个强拍')
  if (
    (phaseStatus === 'confirmed' &&
      (firstDownbeatAt === null ||
        !matchesCandidateBeat(firstDownbeatAt, candidate.beats))) ||
    (phaseStatus === 'uncertain' && firstDownbeatAt !== null)
  ) {
    throw new BeatReviewRequestError('首个强拍必须选择候选拍点')
  }
  const status = String(value.status)
  if (!['draft', 'submitted'].includes(status)) {
    throw new BeatReviewRequestError('结构审核状态无效', 500)
  }
  const review: BeatStructureReviewPackage = {
    ...(value as unknown as BeatStructureReviewPackage),
    schema_version: 1,
    review_kind: 'beat-structure',
    status: status as BeatStructureReviewPackage['status'],
    audio_duration: candidate.audioDuration,
    time_signature: timeSignature,
    pickup_beats: pickupBeats,
    phase_status: phaseStatus,
    first_downbeat_at:
      firstDownbeatAt === null
        ? null
        : Math.round(firstDownbeatAt * 1_000_000) / 1_000_000,
    meter_changes: normalizeMeterChanges(value.meter_changes, candidate),
    checks: normalizeChecks(value.checks),
  }
  if (
    !isRecord(review.candidate_source) ||
    review.candidate_source.kind !== 'worker-analysis' ||
    review.candidate_source.analysis_sha256 !== candidate.analysisSha256
  ) {
    throw new BeatReviewRequestError('结构候选来源无效', 500)
  }
  if (review.status === 'submitted') {
    if (
      !isRecord(review.submission) ||
      !SHA256_PATTERN.test(String(review.submission.content_sha256 ?? '')) ||
      review.submission.content_sha256 !==
        beatStructureContentDigest(review)
    ) {
      throw new BeatReviewRequestError('已提交结构审核摘要无效', 500)
    }
  }
  return review
}

function initialReview(
  candidate: RecordingCandidate,
  reviewer: string,
): BeatStructureReviewPackage {
  return {
    schema_version: 1,
    review_kind: 'beat-structure',
    status: 'draft',
    corpus_version: candidate.corpusVersion,
    recording_id: candidate.recordingId,
    annotator_alias: reviewer,
    source_audio_sha256: candidate.sourceAudioSha256,
    audio_duration: candidate.audioDuration,
    time_signature: candidate.timeSignature,
    pickup_beats: candidate.pickupBeats,
    phase_status: 'confirmed',
    first_downbeat_at: candidate.beats[0],
    meter_changes: [],
    checks: {
      meter: false,
      downbeat_phase: false,
      pickup: false,
      meter_changes: false,
    },
    candidate_source: {
      kind: 'worker-analysis',
      analysis_sha256: candidate.analysisSha256,
    },
    submission: null,
  }
}

function writeReview(
  path: string,
  review: BeatStructureReviewPackage,
): void {
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(review, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  renameSync(temporary, path)
}

function readReview(
  root: string,
  reviewer: string,
  recordingId: string,
): BeatStructureReviewPackage {
  const candidate = candidateFor(root, recordingId)
  const path = reviewPath(root, reviewer, recordingId)
  if (!existsSync(path)) {
    writeReview(path, initialReview(candidate, reviewer))
  }
  const value = readJson(path, '全曲结构审核文件')
  return validateReview(value, candidate, reviewer)
}

function applyDraft(
  review: BeatStructureReviewPackage,
  candidate: RecordingCandidate,
  input: BeatStructureDraftInput,
): BeatStructureReviewPackage {
  if (review.status !== 'draft') {
    throw new BeatReviewRequestError('已提交结构审核不能直接修改', 409)
  }
  const phaseStatus = String(
    input.phase_status ?? review.phase_status,
  )
  const firstDownbeatAt =
    phaseStatus === 'uncertain'
      ? null
      : input.first_downbeat_at ?? review.first_downbeat_at
  return validateReview(
    {
      ...review,
      time_signature: input.time_signature ?? review.time_signature,
      pickup_beats: input.pickup_beats ?? review.pickup_beats,
      phase_status: phaseStatus,
      first_downbeat_at: firstDownbeatAt,
      meter_changes: input.meter_changes ?? review.meter_changes,
      checks: input.checks ?? review.checks,
      submission: null,
      status: 'draft',
    },
    candidate,
    review.annotator_alias,
  )
}

function sourceAudioPath(
  candidate: RecordingCandidate,
  sourceRoot: string,
): string {
  const path = resolve(sourceRoot, `${candidate.recordingId}.media`)
  if (
    dirname(path) !== sourceRoot ||
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    !lstatSync(path).isFile() ||
    sha256File(path) !== candidate.sourceAudioSha256
  ) {
    throw new BeatReviewRequestError('全曲音频不存在或哈希不匹配', 500)
  }
  return path
}

export function listBeatStructureReviews(
  reviewerValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatStructureSummary[] {
  const reviewer = normalizeReviewer(reviewerValue)
  reviewerExists(root, reviewer)
  return recordingCandidates(root).map((candidate) => {
    const review = readReview(root, reviewer, candidate.recordingId)
    return {
      recordingId: candidate.recordingId,
      split: candidate.split,
      status: review.status,
      duration: review.audio_duration,
      timeSignature: review.time_signature,
      pickupBeats: review.pickup_beats,
      phaseStatus: review.phase_status,
      meterChangeCount: review.meter_changes.length,
      checks: review.checks,
    }
  })
}

export function getBeatStructureReview(
  recordingId: string,
  reviewerValue: unknown,
  root = defaultBeatGoldRoot(),
  sourceRoot = defaultBeatSourceRoot(root),
): BeatStructureDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const candidate = candidateFor(root, recordingId)
  const review = readReview(root, reviewer, recordingId)
  sourceAudioPath(candidate, sourceRoot)
  return {
    review,
    audioUrl:
      `/api/admin/beat-gold/structure-media/` +
      encodeURIComponent(recordingId),
    candidateBeats: candidate.beats,
    candidateBpm: candidate.bpm,
  }
}

export function beatStructureAudioPath(
  recordingId: string,
  root = defaultBeatGoldRoot(),
  sourceRoot = defaultBeatSourceRoot(root),
): string {
  return sourceAudioPath(candidateFor(root, recordingId), sourceRoot)
}

export function saveBeatStructureDraft(
  recordingId: string,
  reviewerValue: unknown,
  input: BeatStructureDraftInput,
  root = defaultBeatGoldRoot(),
  sourceRoot = defaultBeatSourceRoot(root),
): BeatStructureDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const candidate = candidateFor(root, recordingId)
  const path = reviewPath(root, reviewer, recordingId)
  const review = applyDraft(
    readReview(root, reviewer, recordingId),
    candidate,
    input,
  )
  writeReview(path, review)
  return getBeatStructureReview(
    recordingId,
    reviewer,
    root,
    sourceRoot,
  )
}

export function submitBeatStructureReview(
  recordingId: string,
  reviewerValue: unknown,
  input: BeatStructureDraftInput,
  root = defaultBeatGoldRoot(),
  sourceRoot = defaultBeatSourceRoot(root),
  identity?: BeatSubmissionIdentity,
): BeatStructureDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const candidate = candidateFor(root, recordingId)
  const path = reviewPath(root, reviewer, recordingId)
  const review = applyDraft(
    readReview(root, reviewer, recordingId),
    candidate,
    input,
  )
  const missing = REQUIRED_CHECKS.filter((key) => !review.checks[key])
  if (missing.length > 0) {
    throw new BeatReviewRequestError(
      `提交前需完成检查：${missing.join(', ')}`,
    )
  }
  review.status = 'submitted'
  review.submission = {
    submitted_at: new Date().toISOString(),
    content_sha256: '',
    ...(identity
      ? {
          admin_user_id: identity.adminUserId,
          admin_username: identity.adminUsername,
        }
      : {}),
  }
  review.submission.content_sha256 = beatStructureContentDigest(review)
  validateReview(review, candidate, reviewer)
  writeReview(path, review)
  return getBeatStructureReview(
    recordingId,
    reviewer,
    root,
    sourceRoot,
  )
}

export function beatStructureReviewers(
  root = defaultBeatGoldRoot(),
): string[] {
  const reviewsRoot = resolve(root, 'reviews')
  if (!existsSync(reviewsRoot)) return []
  return readdirSync(reviewsRoot)
    .filter((name) => REVIEWER_PATTERN.test(name))
    .filter((name) => {
      const path = resolve(reviewsRoot, name)
      return (
        existsSync(path) &&
        !lstatSync(path).isSymbolicLink() &&
        lstatSync(path).isDirectory()
      )
    })
    .sort()
}
