import { createHash } from 'node:crypto'
import {
  copyFileSync,
  existsSync,
  linkSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { dirname, isAbsolute, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const SAFE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const REVIEWER_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const SPLITS = new Set(['train', 'validation', 'test'])
const REQUIRED_CHECKS = [
  'beat_times',
  'downbeats',
  'meter',
  'completeness',
] as const

export interface BeatReviewChecks {
  beat_times: boolean
  downbeats: boolean
  meter: boolean
  completeness: boolean
}

export interface BeatReviewMarker {
  at: number
  beat: number
  measure: number
  downbeat: boolean
}

export interface BeatReviewPackage {
  schema_version: 2
  review_kind: 'beat-grid'
  status: 'draft' | 'submitted' | 'adjudicated'
  corpus_version: string
  clip_id: string
  group_id: string
  annotator_alias: string
  audio: string
  source_audio_sha256: string
  audio_sha256: string
  audio_source_start: number
  audio_source_end: number
  audio_duration: number
  evaluation_start: number
  evaluation_end: number
  evaluation_duration: number
  time_signature: [number, number]
  pickup_beats: number
  beats: BeatReviewMarker[]
  tags: string[]
  checks: BeatReviewChecks
  candidate_source: Record<string, unknown>
  submission: {
    submitted_at: string
    content_sha256: string
    admin_user_id?: string
    admin_username?: string
  } | null
  adjudication: Record<string, unknown> | null
  verification_status:
    | 'draft'
    | 'independent_annotation'
    | 'adjudicated_gold'
  benchmark_eligible: boolean
}

export interface BeatSubmissionIdentity {
  adminUserId: string
  adminUsername: string
}

export interface BeatMetrics {
  true_positive: number
  false_positive: number
  false_negative: number
  precision: number
  recall: number
  f1: number
  mean_error_ms: number | null
  median_error_ms: number | null
  p95_error_ms: number | null
  tolerance_ms: number
}

export interface BeatMarkerDifference {
  left: BeatReviewMarker | null
  right: BeatReviewMarker | null
  deltaMs: number | null
  phaseAgreement: boolean
}

export interface BeatReviewComparison {
  clipId: string
  reviewers: [string, string]
  independentAccounts: boolean
  meterAgreement: boolean
  pickupAgreement: boolean
  beatMetrics: BeatMetrics
  downbeatMetrics: BeatMetrics
  differences: BeatMarkerDifference[]
  left: BeatReviewDocument
  right: BeatReviewDocument
}

export interface BeatAdjudicationSummary {
  clipId: string
  recordingId: string
  split: PilotManifestItem['split']
  leftStatus: BeatReviewPackage['status']
  rightStatus: BeatReviewPackage['status']
  ready: boolean
  adjudicated: boolean
  independentAccounts: boolean
  beatF1: number | null
  medianErrorMs: number | null
  disagreements: number | null
}

export interface BeatReviewSummary {
  clipId: string
  recordingId: string
  split: 'train' | 'validation' | 'test'
  status: BeatReviewPackage['status']
  duration: number
  sourceStart: number
  audioDuration: number
  evaluationStart: number
  evaluationEnd: number
  beatCount: number
  timeSignature: [number, number]
  pickupBeats: number
  checks: BeatReviewChecks
  tags: string[]
}

export interface BeatReviewDocument {
  review: BeatReviewPackage
  audioUrl: string
}

export interface BeatReviewDraftInput {
  beats?: unknown
  time_signature?: unknown
  pickup_beats?: unknown
  checks?: unknown
}

interface PilotManifestItem {
  clip_id: string
  recording_id: string
  split: 'train' | 'validation' | 'test'
  audio_source_start: number
  audio_source_end: number
  audio_duration: number
  evaluation_start: number
  evaluation_end: number
  evaluation_duration: number
  audio_sha256: string
}

interface PilotManifest {
  schema_version: 2
  kind: 'beat-gold-pilot-v2'
  corpus_version: string
  corpus_sha256: string
  immutable: true
  evaluation_duration: number
  context: {
    pre_roll: number
    post_roll: number
  }
  verification_status: string
  benchmark_eligible: boolean
  recordings: number
  clips: number
  split_counts: Record<string, number>
  items: PilotManifestItem[]
}

export class BeatReviewRequestError extends Error {
  constructor(
    message: string,
    readonly statusCode = 400,
  ) {
    super(message)
  }
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
      throw new BeatReviewRequestError('审核内容包含无效数字')
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
  throw new BeatReviewRequestError('审核内容包含不支持的值')
}

export function beatReviewContentDigest(
  review: BeatReviewPackage,
): string {
  const payload = Object.fromEntries(
    Object.entries(review).filter(
      ([key]) => !['status', 'submission', 'adjudication'].includes(key),
    ),
  )
  return createHash('sha256')
    .update(canonicalJson(payload), 'utf8')
    .digest('hex')
}

export function defaultBeatGoldRoot(): string {
  const configured = process.env.GTS_BEAT_GOLD_ROOT
  if (configured) {
    if (!isAbsolute(configured)) {
      throw new Error('GTS_BEAT_GOLD_ROOT 必须是绝对路径')
    }
    return resolve(configured)
  }
  return resolve(
    dirname(fileURLToPath(import.meta.url)),
    '../../../.cache/beat-gold/pilot-v2',
  )
}

function normalizeReviewer(value: unknown): string {
  const reviewer = typeof value === 'string' ? value.trim() : ''
  if (!REVIEWER_PATTERN.test(reviewer)) {
    throw new BeatReviewRequestError('审核者标识无效')
  }
  return reviewer
}

function readManifest(root: string): PilotManifest {
  const path = resolve(root, 'pilot-manifest.json')
  if (!existsSync(path)) {
    throw new BeatReviewRequestError('节拍 Pilot 尚未准备', 404)
  }
  if (lstatSync(path).isSymbolicLink()) {
    throw new BeatReviewRequestError('Pilot 清单路径无效')
  }
  let value: unknown
  try {
    value = JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    throw new BeatReviewRequestError('Pilot 清单无法读取', 500)
  }
  if (
    !isRecord(value) ||
    value.schema_version !== 2 ||
    value.kind !== 'beat-gold-pilot-v2' ||
    !SAFE_ID_PATTERN.test(String(value.corpus_version ?? '')) ||
    !SHA256_PATTERN.test(String(value.corpus_sha256 ?? '')) ||
    value.immutable !== true ||
    !isRecord(value.context) ||
    !Array.isArray(value.items)
  ) {
    throw new BeatReviewRequestError('Pilot 清单格式无效', 500)
  }
  const items = value.items.map((item, index): PilotManifestItem => {
    if (
      !isRecord(item) ||
      !SAFE_ID_PATTERN.test(String(item.clip_id ?? '')) ||
      !SAFE_ID_PATTERN.test(String(item.recording_id ?? '')) ||
      !SPLITS.has(String(item.split ?? '')) ||
      !SHA256_PATTERN.test(String(item.audio_sha256 ?? ''))
    ) {
      throw new BeatReviewRequestError(
        `Pilot 清单第 ${index + 1} 项无效`,
        500,
      )
    }
    return {
      clip_id: String(item.clip_id),
      recording_id: String(item.recording_id),
      split: item.split as PilotManifestItem['split'],
      audio_source_start: finiteNumber(
        item.audio_source_start,
        '音频来源起点',
      ),
      audio_source_end: finiteNumber(item.audio_source_end, '音频来源终点'),
      audio_duration: finiteNumber(item.audio_duration, '音频时长'),
      evaluation_start: finiteNumber(item.evaluation_start, '评测窗起点'),
      evaluation_end: finiteNumber(item.evaluation_end, '评测窗终点'),
      evaluation_duration: finiteNumber(
        item.evaluation_duration,
        '评测窗时长',
      ),
      audio_sha256: String(item.audio_sha256),
    }
  })
  return {
    schema_version: 2,
    kind: 'beat-gold-pilot-v2',
    corpus_version: String(value.corpus_version),
    corpus_sha256: String(value.corpus_sha256),
    immutable: true,
    evaluation_duration: finiteNumber(
      value.evaluation_duration,
      '评测窗时长',
    ),
    context: {
      pre_roll: finiteNumber(value.context.pre_roll, '前置上下文'),
      post_roll: finiteNumber(value.context.post_roll, '后置上下文'),
    },
    verification_status: String(value.verification_status ?? 'draft'),
    benchmark_eligible: value.benchmark_eligible === true,
    recordings: integer(value.recordings, '录音数量'),
    clips: integer(value.clips, '片段数量'),
    split_counts: isRecord(value.split_counts)
      ? Object.fromEntries(
          Object.entries(value.split_counts).map(([key, count]) => [
            key,
            integer(count, `${key} 分区数量`),
          ]),
        )
      : {},
    items,
  }
}

function reviewerRoot(root: string, reviewerValue: unknown): string {
  const reviewer = normalizeReviewer(reviewerValue)
  const path = resolve(root, 'reviews', reviewer)
  if (
    !existsSync(path) ||
    !lstatSync(path).isDirectory() ||
    lstatSync(path).isSymbolicLink()
  ) {
    throw new BeatReviewRequestError('审核者工作区不存在', 404)
  }
  return path
}

function manifestItem(
  root: string,
  clipId: string,
): PilotManifestItem {
  if (!SAFE_ID_PATTERN.test(clipId)) {
    throw new BeatReviewRequestError('片段标识无效')
  }
  const item = readManifest(root).items.find(
    (candidate) => candidate.clip_id === clipId,
  )
  if (!item) throw new BeatReviewRequestError('审核片段不存在', 404)
  return item
}

function reviewPath(
  root: string,
  reviewer: unknown,
  clipId: string,
): string {
  manifestItem(root, clipId)
  const directory = reviewerRoot(root, reviewer)
  const path = resolve(directory, `${clipId}.json`)
  if (
    dirname(path) !== directory ||
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink()
  ) {
    throw new BeatReviewRequestError('审核片段不存在', 404)
  }
  return path
}

function normalizeChecks(value: unknown): BeatReviewChecks {
  if (
    !isRecord(value) ||
    REQUIRED_CHECKS.some((key) => typeof value[key] !== 'boolean')
  ) {
    throw new BeatReviewRequestError('审核检查项无效')
  }
  return {
    beat_times: value.beat_times as boolean,
    downbeats: value.downbeats as boolean,
    meter: value.meter as boolean,
    completeness: value.completeness as boolean,
  }
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

function normalizeBeatMarkers(
  value: unknown,
  audioDuration: number,
  timeSignature: [number, number],
): BeatReviewMarker[] {
  if (!Array.isArray(value) || value.length > 1000) {
    throw new BeatReviewRequestError('拍点列表无效')
  }
  const numerator = timeSignature[0]
  const beats = value.map((item, index) => {
    if (!isRecord(item)) {
      throw new BeatReviewRequestError(`第 ${index + 1} 个拍点无效`)
    }
    const at = finiteNumber(item.at, `第 ${index + 1} 个拍点`)
    const beat = integer(item.beat, `第 ${index + 1} 个拍位`)
    const measure = integer(item.measure, `第 ${index + 1} 个小节号`)
    const downbeat = item.downbeat
    if (
      at < 0 ||
      at >= audioDuration ||
      beat < 1 ||
      beat > numerator ||
      measure < 0 ||
      typeof downbeat !== 'boolean' ||
      downbeat !== (beat === 1)
    ) {
      throw new BeatReviewRequestError(`第 ${index + 1} 个拍点超出片段`)
    }
    return {
      at: Math.round(at * 1_000_000) / 1_000_000,
      beat,
      measure,
      downbeat,
    }
  })
  for (let index = 1; index < beats.length; index += 1) {
    const previous = beats[index - 1]
    const current = beats[index]
    const expectedBeat = previous.beat % numerator + 1
    const expectedMeasure =
      previous.measure + (expectedBeat === 1 ? 1 : 0)
    if (current.at - previous.at < 0.03) {
      throw new BeatReviewRequestError('相邻拍点必须至少间隔 30ms')
    }
    if (
      current.beat !== expectedBeat ||
      current.measure !== expectedMeasure
    ) {
      throw new BeatReviewRequestError('拍点拍位与小节相位不连续')
    }
  }
  return beats
}

function validateReview(
  value: unknown,
  reviewer?: string,
): BeatReviewPackage {
  if (
    !isRecord(value) ||
    value.schema_version !== 2 ||
    value.review_kind !== 'beat-grid'
  ) {
    throw new BeatReviewRequestError('节拍审核文件格式无效', 500)
  }
  if (!SAFE_ID_PATTERN.test(String(value.corpus_version ?? ''))) {
    throw new BeatReviewRequestError('语料版本无效', 500)
  }
  for (const key of ['clip_id', 'group_id', 'annotator_alias'] as const) {
    if (!SAFE_ID_PATTERN.test(String(value[key] ?? ''))) {
      throw new BeatReviewRequestError(`${key} 无效`, 500)
    }
  }
  if (reviewer && value.annotator_alias !== reviewer) {
    throw new BeatReviewRequestError('审核者工作区与文件不匹配', 500)
  }
  if (
    typeof value.audio !== 'string' ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.wav$/.test(value.audio)
  ) {
    throw new BeatReviewRequestError('审核音频文件名无效', 500)
  }
  for (const key of ['source_audio_sha256', 'audio_sha256'] as const) {
    if (!SHA256_PATTERN.test(String(value[key] ?? ''))) {
      throw new BeatReviewRequestError(`${key} 无效`, 500)
    }
  }
  const audioDuration = finiteNumber(value.audio_duration, '音频时长')
  const audioSourceStart = finiteNumber(
    value.audio_source_start,
    '音频来源起点',
  )
  const audioSourceEnd = finiteNumber(
    value.audio_source_end,
    '音频来源终点',
  )
  const evaluationStart = finiteNumber(
    value.evaluation_start,
    '评测窗起点',
  )
  const evaluationEnd = finiteNumber(value.evaluation_end, '评测窗终点')
  const evaluationDuration = finiteNumber(
    value.evaluation_duration,
    '评测窗时长',
  )
  if (
    audioDuration <= 0 ||
    audioSourceStart < 0 ||
    Math.abs(audioSourceEnd - audioSourceStart - audioDuration) > 0.00001 ||
    evaluationStart < 0 ||
    evaluationDuration <= 0 ||
    evaluationEnd > audioDuration + 0.00001 ||
    Math.abs(evaluationEnd - evaluationStart - evaluationDuration) > 0.00001
  ) {
    throw new BeatReviewRequestError('片段边界无效', 500)
  }
  const timeSignature = normalizeTimeSignature(value.time_signature)
  const pickupBeats = integer(value.pickup_beats, '弱起拍数')
  if (pickupBeats < 0 || pickupBeats >= timeSignature[0]) {
    throw new BeatReviewRequestError('弱起拍数无效', 500)
  }
  if (!Array.isArray(value.beats)) {
    throw new BeatReviewRequestError('拍点列表无效', 500)
  }
  const beats = normalizeBeatMarkers(
    value.beats,
    audioDuration,
    timeSignature,
  )
  if (!Array.isArray(value.tags) || value.tags.some((tag) => typeof tag !== 'string')) {
    throw new BeatReviewRequestError('片段标签无效', 500)
  }
  const status = value.status
  if (!['draft', 'submitted', 'adjudicated'].includes(String(status))) {
    throw new BeatReviewRequestError('审核状态无效', 500)
  }
  const review: BeatReviewPackage = {
    ...(value as unknown as BeatReviewPackage),
    schema_version: 2,
    review_kind: 'beat-grid',
    status: status as BeatReviewPackage['status'],
    corpus_version: String(value.corpus_version),
    audio_source_start: audioSourceStart,
    audio_source_end: audioSourceEnd,
    audio_duration: audioDuration,
    evaluation_start: evaluationStart,
    evaluation_end: evaluationEnd,
    evaluation_duration: evaluationDuration,
    time_signature: timeSignature,
    pickup_beats: pickupBeats,
    beats,
    tags: [...new Set(value.tags as string[])].sort(),
    checks: normalizeChecks(value.checks),
  }
  if (review.status === 'submitted') {
    const submissionIdentityValid =
      !isRecord(review.submission) ||
      (review.submission.admin_user_id === undefined &&
        review.submission.admin_username === undefined) ||
      (SAFE_ID_PATTERN.test(String(review.submission.admin_user_id ?? '')) &&
        SAFE_ID_PATTERN.test(String(review.submission.admin_username ?? '')))
    if (
      review.verification_status !== 'independent_annotation' ||
      review.benchmark_eligible ||
      !isRecord(review.submission) ||
      !submissionIdentityValid ||
      !SHA256_PATTERN.test(String(review.submission.content_sha256 ?? '')) ||
      review.submission.content_sha256 !== beatReviewContentDigest(review)
    ) {
      throw new BeatReviewRequestError('已提交审核的内容摘要无效', 500)
    }
  }
  if (review.status === 'adjudicated') {
    if (
      review.verification_status !== 'adjudicated_gold' ||
      review.benchmark_eligible !== true ||
      !isRecord(review.adjudication) ||
      !SAFE_ID_PATTERN.test(
        String(review.adjudication.adjudicator_alias ?? ''),
      ) ||
      !SAFE_ID_PATTERN.test(
        String(review.adjudication.admin_user_id ?? ''),
      ) ||
      review.adjudication.content_sha256 !== beatReviewContentDigest(review)
    ) {
      throw new BeatReviewRequestError('裁决金标准的内容摘要无效', 500)
    }
  }
  return review
}

function readReview(
  root: string,
  reviewerValue: unknown,
  clipId: string,
): BeatReviewPackage {
  const reviewer = normalizeReviewer(reviewerValue)
  const path = reviewPath(root, reviewer, clipId)
  let value: unknown
  try {
    value = JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    throw new BeatReviewRequestError('审核文件无法读取', 500)
  }
  const review = validateReview(value, reviewer)
  const manifest = readManifest(root)
  const item = manifestItem(root, clipId)
  if (review.clip_id !== clipId) {
    throw new BeatReviewRequestError('审核文件与片段标识不匹配', 500)
  }
  const fixedNumbers: Array<
    [keyof PilotManifestItem, keyof BeatReviewPackage]
  > = [
    ['audio_source_start', 'audio_source_start'],
    ['audio_source_end', 'audio_source_end'],
    ['audio_duration', 'audio_duration'],
    ['evaluation_start', 'evaluation_start'],
    ['evaluation_end', 'evaluation_end'],
    ['evaluation_duration', 'evaluation_duration'],
  ]
  if (
    review.corpus_version !== manifest.corpus_version ||
    review.audio_sha256 !== item.audio_sha256 ||
    fixedNumbers.some(
      ([itemKey, reviewKey]) =>
        Math.abs(
          Number(item[itemKey]) - Number(review[reviewKey]),
        ) > 0.00001,
    )
  ) {
    throw new BeatReviewRequestError('审核文件与不可变语料清单不匹配', 500)
  }
  return review
}

function writeReview(
  root: string,
  reviewer: string,
  clipId: string,
  review: BeatReviewPackage,
): void {
  const path = reviewPath(root, reviewer, clipId)
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(review, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  renameSync(temporary, path)
}

function applyDraft(
  review: BeatReviewPackage,
  input: BeatReviewDraftInput,
): BeatReviewPackage {
  if (review.status !== 'draft') {
    throw new BeatReviewRequestError('已提交片段不能直接修改', 409)
  }
  const timeSignature = normalizeTimeSignature(
    input.time_signature ?? review.time_signature,
  )
  const pickupBeats = integer(
    input.pickup_beats ?? review.pickup_beats,
    '弱起拍数',
  )
  if (pickupBeats < 0 || pickupBeats >= timeSignature[0]) {
    throw new BeatReviewRequestError('弱起拍数必须小于每小节拍数')
  }
  const beats = normalizeBeatMarkers(
    input.beats ?? review.beats,
    review.audio_duration,
    timeSignature,
  )
  return validateReview({
    ...review,
    time_signature: timeSignature,
    pickup_beats: pickupBeats,
    beats,
    checks: normalizeChecks(input.checks ?? review.checks),
    submission: null,
    adjudication: null,
    verification_status: 'draft',
    benchmark_eligible: false,
  })
}

export function beatGoldStatus(root = defaultBeatGoldRoot()) {
  if (!existsSync(resolve(root, 'pilot-manifest.json'))) {
    return {
      enabled: false,
      reviewers: [] as string[],
      manifest: null,
    }
  }
  const manifest = readManifest(root)
  const reviewsRoot = resolve(root, 'reviews')
  const reviewers = existsSync(reviewsRoot)
    ? readdirSync(reviewsRoot)
        .filter((name) => REVIEWER_PATTERN.test(name))
        .filter((name) => {
          const path = resolve(reviewsRoot, name)
          return (
            existsSync(path) &&
            lstatSync(path).isDirectory() &&
            !lstatSync(path).isSymbolicLink()
          )
        })
        .sort()
    : []
  return {
    enabled: true,
    reviewers,
    manifest: {
      corpusVersion: manifest.corpus_version,
      corpusSha256: manifest.corpus_sha256,
      immutable: manifest.immutable,
      evaluationDuration: manifest.evaluation_duration,
      context: {
        preRoll: manifest.context.pre_roll,
        postRoll: manifest.context.post_roll,
      },
      recordings: manifest.recordings,
      clips: manifest.clips,
      splitCounts: manifest.split_counts,
      verificationStatus: manifest.verification_status,
      benchmarkEligible: manifest.benchmark_eligible,
    },
  }
}

export function listBeatReviews(
  reviewerValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatReviewSummary[] {
  const reviewer = normalizeReviewer(reviewerValue)
  reviewerRoot(root, reviewer)
  const manifest = readManifest(root)
  return manifest.items.map((item) => {
    const review = readReview(root, reviewer, item.clip_id)
    return {
      clipId: review.clip_id,
      recordingId: item.recording_id,
      split: item.split,
      status: review.status,
      duration: review.evaluation_duration,
      sourceStart:
        review.audio_source_start + review.evaluation_start,
      audioDuration: review.audio_duration,
      evaluationStart: review.evaluation_start,
      evaluationEnd: review.evaluation_end,
      beatCount: review.beats.filter(
        (marker) =>
          marker.at >= review.evaluation_start &&
          marker.at < review.evaluation_end,
      ).length,
      timeSignature: review.time_signature,
      pickupBeats: review.pickup_beats,
      checks: review.checks,
      tags: review.tags,
    }
  })
}

export function getBeatReview(
  clipId: string,
  reviewerValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatReviewDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const review = readReview(root, reviewer, clipId)
  const directory = reviewerRoot(root, reviewer)
  const audioPath = resolve(directory, review.audio)
  if (
    dirname(audioPath) !== directory ||
    !existsSync(audioPath) ||
    lstatSync(audioPath).isSymbolicLink() ||
    sha256File(audioPath) !== review.audio_sha256
  ) {
    throw new BeatReviewRequestError('审核音频不存在或哈希不匹配', 500)
  }
  return {
    review,
    audioUrl:
      `/api/admin/beat-gold/media/${encodeURIComponent(reviewer)}/` +
      encodeURIComponent(review.audio),
  }
}

export function beatReviewAudioPath(
  reviewerValue: unknown,
  filename: string,
  root = defaultBeatGoldRoot(),
): string {
  const reviewer = normalizeReviewer(reviewerValue)
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.wav$/.test(filename)) {
    throw new BeatReviewRequestError('审核音频不存在', 404)
  }
  const clipId = filename.slice(0, -4)
  const review = readReview(root, reviewer, clipId)
  if (review.audio !== filename) {
    throw new BeatReviewRequestError('审核音频不存在', 404)
  }
  const directory = reviewerRoot(root, reviewer)
  const path = resolve(directory, filename)
  if (
    dirname(path) !== directory ||
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink()
  ) {
    throw new BeatReviewRequestError('审核音频不存在', 404)
  }
  return path
}

export function saveBeatReviewDraft(
  clipId: string,
  reviewerValue: unknown,
  input: BeatReviewDraftInput,
  root = defaultBeatGoldRoot(),
): BeatReviewDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const review = applyDraft(readReview(root, reviewer, clipId), input)
  writeReview(root, reviewer, clipId, review)
  return getBeatReview(clipId, reviewer, root)
}

export function submitBeatReview(
  clipId: string,
  reviewerValue: unknown,
  input: BeatReviewDraftInput,
  root = defaultBeatGoldRoot(),
  identity?: BeatSubmissionIdentity,
): BeatReviewDocument {
  const reviewer = normalizeReviewer(reviewerValue)
  const review = applyDraft(readReview(root, reviewer, clipId), input)
  const missing = REQUIRED_CHECKS.filter((key) => !review.checks[key])
  if (missing.length > 0) {
    throw new BeatReviewRequestError(
      `提交前需完成检查：${missing.join(', ')}`,
    )
  }
  const evaluationBeats = review.beats.filter(
    (marker) =>
      marker.at >= review.evaluation_start &&
      marker.at < review.evaluation_end,
  )
  if (evaluationBeats.length < 2) {
    throw new BeatReviewRequestError('提交审核至少需要两个拍点')
  }
  review.status = 'submitted'
  review.verification_status = 'independent_annotation'
  review.benchmark_eligible = false
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
  review.submission.content_sha256 = beatReviewContentDigest(review)
  validateReview(review, reviewer)
  writeReview(root, reviewer, clipId, review)
  return getBeatReview(clipId, reviewer, root)
}

function evaluationMarkers(review: BeatReviewPackage): BeatReviewMarker[] {
  return review.beats.filter(
    (marker) =>
      marker.at >= review.evaluation_start &&
      marker.at < review.evaluation_end,
  )
}

function percentile(values: readonly number[], fraction: number): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((left, right) => left - right)
  const position = (sorted.length - 1) * fraction
  const lower = Math.floor(position)
  const upper = Math.min(sorted.length - 1, lower + 1)
  const weight = position - lower
  return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

export function evaluateBeatTimes(
  reference: readonly BeatReviewMarker[],
  predicted: readonly BeatReviewMarker[],
  tolerance = 0.07,
): BeatMetrics {
  const unmatched = new Set(reference.map((_, index) => index))
  const errors: number[] = []
  for (const prediction of [...predicted].sort((a, b) => a.at - b.at)) {
    const candidates = [...unmatched].filter(
      (index) =>
        Math.abs(reference[index].at - prediction.at) <= tolerance,
    )
    if (candidates.length === 0) continue
    const matched = candidates.reduce((best, current) =>
      Math.abs(reference[current].at - prediction.at) <
      Math.abs(reference[best].at - prediction.at)
        ? current
        : best,
    )
    unmatched.delete(matched)
    errors.push(Math.abs(reference[matched].at - prediction.at) * 1000)
  }
  const truePositive = errors.length
  const falsePositive = predicted.length - truePositive
  const falseNegative = reference.length - truePositive
  const precision = truePositive / Math.max(1, truePositive + falsePositive)
  const recall = truePositive / Math.max(1, truePositive + falseNegative)
  return {
    true_positive: truePositive,
    false_positive: falsePositive,
    false_negative: falseNegative,
    precision,
    recall,
    f1:
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0,
    mean_error_ms:
      errors.length > 0
        ? errors.reduce((sum, value) => sum + value, 0) / errors.length
        : null,
    median_error_ms: percentile(errors, 0.5),
    p95_error_ms: percentile(errors, 0.95),
    tolerance_ms: tolerance * 1000,
  }
}

function markerDifferences(
  left: readonly BeatReviewMarker[],
  right: readonly BeatReviewMarker[],
  tolerance = 0.07,
): BeatMarkerDifference[] {
  const unmatchedRight = new Set(right.map((_, index) => index))
  const differences: BeatMarkerDifference[] = []
  for (const leftMarker of left) {
    const candidates = [...unmatchedRight].filter(
      (index) => Math.abs(right[index].at - leftMarker.at) <= tolerance,
    )
    if (candidates.length === 0) {
      differences.push({
        left: leftMarker,
        right: null,
        deltaMs: null,
        phaseAgreement: false,
      })
      continue
    }
    const matched = candidates.reduce((best, current) =>
      Math.abs(right[current].at - leftMarker.at) <
      Math.abs(right[best].at - leftMarker.at)
        ? current
        : best,
    )
    unmatchedRight.delete(matched)
    const rightMarker = right[matched]
    differences.push({
      left: leftMarker,
      right: rightMarker,
      deltaMs: Math.round((rightMarker.at - leftMarker.at) * 1_000_000) / 1000,
      phaseAgreement:
        leftMarker.beat === rightMarker.beat &&
        leftMarker.measure === rightMarker.measure &&
        leftMarker.downbeat === rightMarker.downbeat,
    })
  }
  for (const index of unmatchedRight) {
    differences.push({
      left: null,
      right: right[index],
      deltaMs: null,
      phaseAgreement: false,
    })
  }
  return differences.sort(
    (a, b) =>
      (a.left?.at ?? a.right?.at ?? 0) -
      (b.left?.at ?? b.right?.at ?? 0),
  )
}

function assertSameReviewTarget(
  left: BeatReviewPackage,
  right: BeatReviewPackage,
): void {
  for (const key of [
    'corpus_version',
    'clip_id',
    'group_id',
    'source_audio_sha256',
    'audio_source_start',
    'audio_source_end',
    'audio_duration',
    'evaluation_start',
    'evaluation_end',
    'evaluation_duration',
  ] as const) {
    if (left[key] !== right[key]) {
      throw new BeatReviewRequestError(`独立审核的 ${key} 不一致`, 409)
    }
  }
}

export function compareBeatReviews(
  clipId: string,
  leftReviewerValue: unknown,
  rightReviewerValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatReviewComparison {
  const leftReviewer = normalizeReviewer(leftReviewerValue)
  const rightReviewer = normalizeReviewer(rightReviewerValue)
  if (leftReviewer === rightReviewer) {
    throw new BeatReviewRequestError('独立审核必须来自不同审核者')
  }
  const left = readReview(root, leftReviewer, clipId)
  const right = readReview(root, rightReviewer, clipId)
  assertSameReviewTarget(left, right)
  if (left.status !== 'submitted' || right.status !== 'submitted') {
    throw new BeatReviewRequestError('两个独立审核均提交后才能比较', 409)
  }
  const leftWindow = evaluationMarkers(left)
  const rightWindow = evaluationMarkers(right)
  const leftDownbeats = leftWindow.filter((marker) => marker.downbeat)
  const rightDownbeats = rightWindow.filter((marker) => marker.downbeat)
  const leftAccount = left.submission?.admin_user_id
  const rightAccount = right.submission?.admin_user_id
  return {
    clipId,
    reviewers: [leftReviewer, rightReviewer],
    independentAccounts:
      Boolean(leftAccount && rightAccount) && leftAccount !== rightAccount,
    meterAgreement:
      left.time_signature[0] === right.time_signature[0] &&
      left.time_signature[1] === right.time_signature[1],
    pickupAgreement: left.pickup_beats === right.pickup_beats,
    beatMetrics: evaluateBeatTimes(leftWindow, rightWindow),
    downbeatMetrics: evaluateBeatTimes(leftDownbeats, rightDownbeats),
    differences: markerDifferences(leftWindow, rightWindow),
    left: getBeatReview(clipId, leftReviewer, root),
    right: getBeatReview(clipId, rightReviewer, root),
  }
}

function goldReviewPath(root: string, clipId: string): string {
  manifestItem(root, clipId)
  const directory = resolve(root, 'gold')
  const path = resolve(directory, `${clipId}.json`)
  if (dirname(path) !== directory) {
    throw new BeatReviewRequestError('金标准路径无效', 500)
  }
  return path
}

function readGoldReview(
  root: string,
  clipId: string,
): BeatReviewPackage | null {
  const path = goldReviewPath(root, clipId)
  if (!existsSync(path)) return null
  if (lstatSync(path).isSymbolicLink()) {
    throw new BeatReviewRequestError('金标准路径无效', 500)
  }
  return validateReview(JSON.parse(readFileSync(path, 'utf8')))
}

export function listBeatAdjudication(
  leftReviewerValue: unknown,
  rightReviewerValue: unknown,
  root = defaultBeatGoldRoot(),
): BeatAdjudicationSummary[] {
  const leftReviewer = normalizeReviewer(leftReviewerValue)
  const rightReviewer = normalizeReviewer(rightReviewerValue)
  const manifest = readManifest(root)
  return manifest.items.map((item) => {
    const left = readReview(root, leftReviewer, item.clip_id)
    const right = readReview(root, rightReviewer, item.clip_id)
    const leftAccount = left.submission?.admin_user_id
    const rightAccount = right.submission?.admin_user_id
    const independentAccounts =
      Boolean(leftAccount && rightAccount) && leftAccount !== rightAccount
    const ready =
      left.status === 'submitted' &&
      right.status === 'submitted' &&
      independentAccounts
    let comparison: BeatReviewComparison | null = null
    if (ready) {
      comparison = compareBeatReviews(
        item.clip_id,
        leftReviewer,
        rightReviewer,
        root,
      )
    }
    return {
      clipId: item.clip_id,
      recordingId: item.recording_id,
      split: item.split,
      leftStatus: left.status,
      rightStatus: right.status,
      ready,
      adjudicated: Boolean(readGoldReview(root, item.clip_id)),
      independentAccounts,
      beatF1: comparison?.beatMetrics.f1 ?? null,
      medianErrorMs: comparison?.beatMetrics.median_error_ms ?? null,
      disagreements:
        comparison?.differences.filter(
          (difference) =>
            difference.left === null ||
            difference.right === null ||
            !difference.phaseAgreement ||
            Math.abs(difference.deltaMs ?? 0) > 20,
        ).length ?? null,
    }
  })
}

export function adjudicateBeatReview(
  clipId: string,
  leftReviewerValue: unknown,
  rightReviewerValue: unknown,
  input: BeatReviewDraftInput,
  identity: BeatSubmissionIdentity,
  root = defaultBeatGoldRoot(),
): BeatReviewPackage {
  const comparison = compareBeatReviews(
    clipId,
    leftReviewerValue,
    rightReviewerValue,
    root,
  )
  const left = comparison.left.review
  const right = comparison.right.review
  const sourceAccounts = new Set([
    left.submission?.admin_user_id,
    right.submission?.admin_user_id,
  ])
  if (
    !comparison.independentAccounts ||
    sourceAccounts.has(identity.adminUserId)
  ) {
    throw new BeatReviewRequestError(
      '裁决者必须独立于两名复核员',
      409,
    )
  }
  const path = goldReviewPath(root, clipId)
  if (existsSync(path)) {
    throw new BeatReviewRequestError('该片段已经完成裁决', 409)
  }
  const resolved = applyDraft(
    {
      ...left,
      status: 'draft',
      annotator_alias: `adjudicator-${identity.adminUsername}`,
      submission: null,
      adjudication: null,
      verification_status: 'draft',
      benchmark_eligible: false,
    },
    input,
  )
  if (!REQUIRED_CHECKS.every((key) => resolved.checks[key])) {
    throw new BeatReviewRequestError('裁决前需完成全部检查')
  }
  const directory = resolve(root, 'gold')
  mkdirSync(directory, { recursive: true })
  const outputAudio = resolve(directory, `${clipId}.wav`)
  const sourceAudio = beatReviewAudioPath(
    comparison.reviewers[0],
    left.audio,
    root,
  )
  try {
    linkSync(sourceAudio, outputAudio)
  } catch {
    copyFileSync(sourceAudio, outputAudio)
  }
  const final = {
    ...resolved,
    status: 'adjudicated' as const,
    audio: `${clipId}.wav`,
    audio_sha256: sha256File(outputAudio),
    submission: null,
    source_reviews: [left, right].map((review) => ({
      annotator_alias: review.annotator_alias,
      admin_user_id: review.submission?.admin_user_id,
      content_sha256: review.submission?.content_sha256,
    })),
    agreement: {
      beat_metrics: comparison.beatMetrics,
      downbeat_metrics: comparison.downbeatMetrics,
      meter_agreement: comparison.meterAgreement,
      pickup_agreement: comparison.pickupAgreement,
    },
    verification_status: 'adjudicated_gold' as const,
    benchmark_eligible: true,
    adjudication: null,
  } as BeatReviewPackage & Record<string, unknown>
  final.adjudication = {
    adjudicator_alias: final.annotator_alias,
    admin_user_id: identity.adminUserId,
    admin_username: identity.adminUsername,
    adjudicated_at: new Date().toISOString(),
    content_sha256: '',
  }
  ;(final.adjudication as Record<string, unknown>).content_sha256 =
    beatReviewContentDigest(final)
  validateReview(final)
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(final, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  renameSync(temporary, path)
  return final
}

export function listBeatGold(
  root = defaultBeatGoldRoot(),
): BeatReviewPackage[] {
  const manifest = readManifest(root)
  return manifest.items.flatMap((item) => {
    const review = readGoldReview(root, item.clip_id)
    return review ? [review] : []
  })
}
