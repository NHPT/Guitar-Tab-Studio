import { createHash, randomUUID } from 'node:crypto'
import {
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, resolve } from 'node:path'

const REVIEW_NAME_PATTERN = /^(?:real-reference|residual-onset)-\d{2}$/
const SAFE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const REVIEWER_ALIAS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const REQUIRED_CHECKS = ['timing', 'string_fret', 'completeness'] as const
const TECHNIQUES = new Set([
  'unknown',
  'pick',
  'strum-down',
  'strum-up',
  'hammer-on',
  'pull-off',
  'slide',
  'harmonic',
  'palm-mute',
  'dead-note',
  'slap',
  'body-tap',
  'tremolo',
  'bend',
  'pinch-harmonic',
  'vibrato',
])

export interface ReviewChecks {
  timing: boolean
  string_fret: boolean
  completeness: boolean
  technique: boolean
}

export interface ReviewEvent {
  onset: number
  offset: number
  string: number
  fret: number
  technique: string
  confidence: number
  candidate_provenance?: {
    project_note_id: string
    position_source: string
    technique_source: string
  }
}

export interface ReviewExcludedRange {
  start: number
  end: number
  reason: string
}

export interface ReviewApproval {
  reviewer_alias: string
  reviewed_at: string
  content_sha256: string
}

export interface ReviewApprovalHistoryEntry extends ReviewApproval {
  invalidated_at: string
  reason: string
}

export interface ReviewPackage {
  schema_version: number
  status: 'draft' | 'approved'
  review_kind?: 'reference' | 'residual-onset-training'
  display_title?: string
  display_subtitle?: string
  review_id: string
  project_id: string
  project_sha256: string
  source_audio_sha256: string
  review_audio: string
  review_audio_sha256: string
  training_audio?: string
  training_audio_sha256?: string
  source_kind: string
  measure_numbers: number[]
  source_start: number
  source_end: number
  duration: number
  capo: number
  bpm: number
  beats: number[]
  time_signature: [number, number]
  events: ReviewEvent[]
  excluded_ranges?: ReviewExcludedRange[]
  checks: ReviewChecks
  approval: ReviewApproval | null
  approval_history?: ReviewApprovalHistoryEntry[]
  training_provenance?: {
    source_manifest: string
    source_manifest_sha256: string
    source_track_id: string
    review_source_track_id: string
    source_group_id: string
    source_split: 'train'
    source_dataset: string
    source_license: string
    audio_condition: string
    training_source_audio_sha256: string
    audio_alignment: 'sample-aligned-no-offset'
    review_audio_processing: 'rms-normalized--20-dbfs-soft-limited--1-dbfs'
    review_audio_gain_db: number
  }
}

export interface ReviewSummary {
  name: string
  status: ReviewPackage['status']
  projectTitle?: string
  projectArtist?: string
  reviewKind?: ReviewPackage['review_kind']
  eventCount: number
  duration: number
  measureNumbers: number[]
  capo: number
  checks: ReviewChecks
}

export interface ReviewDocument {
  name: string
  audioUrl: string
  review: ReviewPackage
}

export interface ReviewDraftInput {
  events: unknown
  checks: unknown
  capo?: unknown
  excluded_ranges?: unknown
}

export class ReviewRequestError extends Error {
  constructor(
    message: string,
    readonly statusCode = 400,
  ) {
    super(message)
  }
}

function requiredChecksForReview(
  review: Pick<ReviewPackage, 'review_kind'>,
): ReadonlyArray<keyof ReviewChecks> {
  return review.review_kind === 'residual-onset-training'
    ? ['timing', 'completeness']
    : REQUIRED_CHECKS
}

export function defaultReviewRoot(): string {
  return resolve(process.cwd(), 'data', 'training', 'reviews')
}

function defaultProjectRoot(): string {
  return resolve(process.cwd(), 'data', 'state', 'projects')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finiteNumber(value: unknown, label: string): number {
  const number = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(number)) {
    throw new ReviewRequestError(`${label} 必须是有限数字`)
  }
  return number
}

function integer(value: unknown, label: string): number {
  const number = finiteNumber(value, label)
  if (!Number.isInteger(number)) {
    throw new ReviewRequestError(`${label} 必须是整数`)
  }
  return number
}

function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function reviewNames(root: string): string[] {
  if (!existsSync(root)) return []
  return readdirSync(root)
    .filter((filename) =>
      /^(?:real-reference|residual-onset)-\d{2}\.json$/.test(filename),
    )
    .map((filename) => filename.slice(0, -5))
    .sort()
}

function reviewPath(name: string, root: string): string {
  if (!REVIEW_NAME_PATTERN.test(name) || !reviewNames(root).includes(name)) {
    throw new ReviewRequestError('审核包不存在', 404)
  }
  const path = resolve(root, `${name}.json`)
  if (dirname(path) !== resolve(root) || lstatSync(path).isSymbolicLink()) {
    throw new ReviewRequestError('审核包路径无效', 400)
  }
  return path
}

function normalizeChecks(value: unknown): ReviewChecks {
  if (!isRecord(value)) {
    throw new ReviewRequestError('审核检查项无效')
  }
  const keys: Array<keyof ReviewChecks> = [
    'timing',
    'string_fret',
    'completeness',
    'technique',
  ]
  if (keys.some((key) => typeof value[key] !== 'boolean')) {
    throw new ReviewRequestError('审核检查项必须是布尔值')
  }
  return {
    timing: value.timing as boolean,
    string_fret: value.string_fret as boolean,
    completeness: value.completeness as boolean,
    technique: value.technique as boolean,
  }
}

function normalizeProvenance(
  value: unknown,
): ReviewEvent['candidate_provenance'] | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value)) {
    throw new ReviewRequestError('候选来源信息无效')
  }
  const fields = [
    'project_note_id',
    'position_source',
    'technique_source',
  ] as const
  if (
    fields.some(
      (field) => typeof value[field] !== 'string' || value[field].length > 128,
    )
  ) {
    throw new ReviewRequestError('候选来源信息无效')
  }
  return {
    project_note_id: value.project_note_id as string,
    position_source: value.position_source as string,
    technique_source: value.technique_source as string,
  }
}

function normalizeEvents(value: unknown, duration: number): ReviewEvent[] {
  if (!Array.isArray(value)) {
    throw new ReviewRequestError('音符事件必须是数组')
  }
  if (value.length > 5000) {
    throw new ReviewRequestError('单个审核包最多包含 5000 个事件')
  }
  const events = value.map((item, index): ReviewEvent => {
    if (!isRecord(item)) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件无效`)
    }
    const onset = finiteNumber(item.onset, `第 ${index + 1} 个事件的起点`)
    const offset = finiteNumber(item.offset, `第 ${index + 1} 个事件的终点`)
    const string = integer(item.string, `第 ${index + 1} 个事件的弦`)
    const fret = integer(item.fret, `第 ${index + 1} 个事件的品`)
    const confidence =
      item.confidence === undefined
        ? 1
        : finiteNumber(item.confidence, `第 ${index + 1} 个事件的置信度`)
    const technique =
      typeof item.technique === 'string' ? item.technique : 'unknown'
    if (!(onset >= 0 && onset < offset && offset <= duration + 0.05)) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件的时间范围无效`)
    }
    if (string < 1 || string > 6) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件的弦号必须在 1 到 6 之间`)
    }
    if (fret < 0 || fret > 24) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件的品位必须在 0 到 24 之间`)
    }
    if (confidence < 0 || confidence > 1) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件的置信度无效`)
    }
    if (!TECHNIQUES.has(technique)) {
      throw new ReviewRequestError(`第 ${index + 1} 个事件的技法无效`)
    }
    const candidateProvenance = normalizeProvenance(item.candidate_provenance)
    return {
      onset,
      offset,
      string,
      fret,
      technique,
      confidence,
      ...(candidateProvenance
        ? { candidate_provenance: candidateProvenance }
        : {}),
    }
  })
  events.sort((left, right) => left.onset - right.onset || left.string - right.string)
  return events
}

function normalizeExcludedRanges(
  value: unknown,
  duration: number,
): ReviewExcludedRange[] {
  if (!Array.isArray(value)) {
    throw new ReviewRequestError('排除区间必须是数组')
  }
  if (value.length > 100) {
    throw new ReviewRequestError('单个审核包最多包含 100 个排除区间')
  }
  const ranges = value.map((item, index): ReviewExcludedRange => {
    if (!isRecord(item)) {
      throw new ReviewRequestError(`第 ${index + 1} 个排除区间无效`)
    }
    const start = finiteNumber(item.start, `第 ${index + 1} 个排除区间的起点`)
    const end = finiteNumber(item.end, `第 ${index + 1} 个排除区间的终点`)
    const reason = typeof item.reason === 'string' ? item.reason.trim() : ''
    if (!(start >= 0 && start < end && end <= duration)) {
      throw new ReviewRequestError(`第 ${index + 1} 个排除区间的时间范围无效`)
    }
    if (reason.length === 0 || reason.length > 200) {
      throw new ReviewRequestError(
        `第 ${index + 1} 个排除区间的原因必须为 1 到 200 个字符`,
      )
    }
    return { start, end, reason }
  })
  ranges.sort((left, right) => left.start - right.start || left.end - right.end)
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index].start < ranges[index - 1].end) {
      throw new ReviewRequestError('排除区间不能重叠')
    }
  }
  return ranges
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value)
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new ReviewRequestError('审核内容包含无效数字')
    }
    return JSON.stringify(value)
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
  throw new ReviewRequestError('审核内容包含不支持的值')
}

export function reviewContentDigest(review: ReviewPackage): string {
  const payload = Object.fromEntries(
    Object.entries(review).filter(([key]) => key !== 'status' && key !== 'approval'),
  )
  return createHash('sha256').update(canonicalJson(payload), 'utf8').digest('hex')
}

function validateReview(review: unknown): asserts review is ReviewPackage {
  if (!isRecord(review) || review.schema_version !== 1) {
    throw new ReviewRequestError('不支持的审核包版本')
  }
  if (
    review.review_kind !== undefined &&
    review.review_kind !== 'reference' &&
    review.review_kind !== 'residual-onset-training'
  ) {
    throw new ReviewRequestError('审核包用途无效')
  }
  for (const key of ['display_title', 'display_subtitle'] as const) {
    if (
      review[key] !== undefined &&
      (typeof review[key] !== 'string' ||
        review[key].trim().length === 0 ||
        review[key].length > 200)
    ) {
      throw new ReviewRequestError(`${key} 无效`)
    }
  }
  for (const key of ['review_id', 'project_id'] as const) {
    if (!SAFE_ID_PATTERN.test(String(review[key] ?? ''))) {
      throw new ReviewRequestError(`${key} 无效`)
    }
  }
  for (const key of [
    'project_sha256',
    'source_audio_sha256',
    'review_audio_sha256',
  ] as const) {
    if (!SHA256_PATTERN.test(String(review[key] ?? ''))) {
      throw new ReviewRequestError(`${key} 无效`)
    }
  }
  if (
    typeof review.review_audio !== 'string' ||
    basename(review.review_audio) !== review.review_audio ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.wav$/.test(review.review_audio)
  ) {
    throw new ReviewRequestError('审核音频文件名无效')
  }
  if (review.review_kind === 'residual-onset-training') {
    if (
      typeof review.training_audio !== 'string' ||
      basename(review.training_audio) !== review.training_audio ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.wav$/.test(review.training_audio)
    ) {
      throw new ReviewRequestError('训练音频文件名无效')
    }
    if (review.training_audio === review.review_audio) {
      throw new ReviewRequestError('审核音频与训练音频必须使用不同文件')
    }
    if (!SHA256_PATTERN.test(String(review.training_audio_sha256 ?? ''))) {
      throw new ReviewRequestError('training_audio_sha256 无效')
    }
  }
  if (
    !Array.isArray(review.measure_numbers) ||
    review.measure_numbers.length === 0 ||
    review.measure_numbers.some((value) => !Number.isInteger(value))
  ) {
    throw new ReviewRequestError('小节编号无效')
  }
  const duration = finiteNumber(review.duration, '片段时长')
  const sourceStart = finiteNumber(review.source_start, '来源起点')
  const sourceEnd = finiteNumber(review.source_end, '来源终点')
  if (duration <= 0 || Math.abs(sourceEnd - sourceStart - duration) > 0.00001) {
    throw new ReviewRequestError('片段时间边界无效')
  }
  const capo = integer(review.capo, '变调夹')
  if (capo < 0 || capo > 12) {
    throw new ReviewRequestError('变调夹必须在 0 到 12 品之间')
  }
  if (finiteNumber(review.bpm, '速度') <= 0) {
    throw new ReviewRequestError('速度必须大于 0')
  }
  const beats = review.beats
  if (
    !Array.isArray(beats) ||
    beats.some(
      (beat, index) =>
        !Number.isFinite(beat) ||
        beat < 0 ||
        beat >= duration ||
        (index > 0 && beat < beats[index - 1]),
    )
  ) {
    throw new ReviewRequestError('节拍时间无效')
  }
  if (
    !Array.isArray(review.time_signature) ||
    review.time_signature.length !== 2 ||
    review.time_signature.some((value) => !Number.isInteger(value) || value <= 0)
  ) {
    throw new ReviewRequestError('拍号无效')
  }
  review.events = normalizeEvents(review.events, duration)
  if (review.excluded_ranges !== undefined) {
    review.excluded_ranges = normalizeExcludedRanges(
      review.excluded_ranges,
      duration,
    )
  }
  review.checks = normalizeChecks(review.checks)
  if (review.review_kind === 'residual-onset-training') {
    if (!isRecord(review.training_provenance)) {
      throw new ReviewRequestError('训练标注包缺少来源信息')
    }
    const provenance = review.training_provenance
    for (const key of [
      'source_manifest',
      'source_track_id',
      'review_source_track_id',
      'source_group_id',
      'source_dataset',
      'source_license',
      'audio_condition',
    ] as const) {
      if (
        typeof provenance[key] !== 'string' ||
        provenance[key].trim().length === 0 ||
        provenance[key].length > 256
      ) {
        throw new ReviewRequestError(`训练来源字段 ${key} 无效`)
      }
    }
    if (!SHA256_PATTERN.test(String(provenance.source_manifest_sha256 ?? ''))) {
      throw new ReviewRequestError('训练来源清单哈希无效')
    }
    if (
      !SHA256_PATTERN.test(
        String(provenance.training_source_audio_sha256 ?? ''),
      )
    ) {
      throw new ReviewRequestError('训练源音频哈希无效')
    }
    if (provenance.audio_alignment !== 'sample-aligned-no-offset') {
      throw new ReviewRequestError('审核音频与训练音频未声明采样对齐')
    }
    if (
      provenance.review_audio_processing !==
      'rms-normalized--20-dbfs-soft-limited--1-dbfs'
    ) {
      throw new ReviewRequestError('审核音频增益处理方式无效')
    }
    const reviewAudioGain = finiteNumber(
      provenance.review_audio_gain_db,
      '审核音频增益',
    )
    if (reviewAudioGain < -60 || reviewAudioGain > 60) {
      throw new ReviewRequestError('审核音频增益无效')
    }
    if (provenance.source_split !== 'train') {
      throw new ReviewRequestError('训练标注包只能来自 train split')
    }
  }
  if (review.status !== 'draft' && review.status !== 'approved') {
    throw new ReviewRequestError('审核状态无效')
  }
  if (review.status === 'approved') {
    const validatedReview = review as unknown as ReviewPackage
    const missingChecks = requiredChecksForReview(
      validatedReview,
    ).filter((key) => !validatedReview.checks[key])
    if (missingChecks.length > 0) {
      throw new ReviewRequestError(
        `已批准审核包缺少检查：${missingChecks.join(', ')}`,
      )
    }
    if (!isRecord(review.approval)) {
      throw new ReviewRequestError('已批准审核包缺少批准记录')
    }
    if (!REVIEWER_ALIAS_PATTERN.test(String(review.approval.reviewer_alias ?? ''))) {
      throw new ReviewRequestError('审核人代号无效')
    }
    if (
      typeof review.approval.content_sha256 !== 'string' ||
      review.approval.content_sha256 !==
        reviewContentDigest(review as unknown as ReviewPackage)
    ) {
      throw new ReviewRequestError('审核内容在批准后已发生变化')
    }
  }
}

function readReview(name: string, root: string): ReviewPackage {
  const path = reviewPath(name, root)
  let review: unknown
  try {
    review = JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    throw new ReviewRequestError('审核包 JSON 无法读取')
  }
  validateReview(review)
  const audioPath = resolve(root, review.review_audio)
  if (
    dirname(audioPath) !== resolve(root) ||
    !existsSync(audioPath) ||
    lstatSync(audioPath).isSymbolicLink() ||
    !lstatSync(audioPath).isFile()
  ) {
    throw new ReviewRequestError('审核音频不存在')
  }
  if (sha256File(audioPath) !== review.review_audio_sha256) {
    throw new ReviewRequestError('审核音频哈希不匹配')
  }
  if (review.review_kind === 'residual-onset-training') {
    const trainingAudioPath = resolve(root, review.training_audio as string)
    if (
      dirname(trainingAudioPath) !== resolve(root) ||
      !existsSync(trainingAudioPath) ||
      lstatSync(trainingAudioPath).isSymbolicLink() ||
      !lstatSync(trainingAudioPath).isFile()
    ) {
      throw new ReviewRequestError('训练音频不存在')
    }
    if (sha256File(trainingAudioPath) !== review.training_audio_sha256) {
      throw new ReviewRequestError('训练音频哈希不匹配')
    }
  }
  return review
}

function writeReview(name: string, review: ReviewPackage, root: string): void {
  const path = reviewPath(name, root)
  const temporary = resolve(root, `.${name}.${process.pid}.${randomUUID()}.tmp`)
  const mode = statSync(path).mode
  writeFileSync(temporary, `${JSON.stringify(review, null, 2)}\n`, {
    encoding: 'utf8',
    mode,
  })
  renameSync(temporary, path)
}

function projectMetadata(
  review: ReviewPackage,
  projectRoot: string,
): { projectTitle?: string; projectArtist?: string } {
  const displayTitle = review.display_title?.trim()
  const displaySubtitle = review.display_subtitle?.trim()
  if (displayTitle || displaySubtitle) {
    return {
      ...(displayTitle ? { projectTitle: displayTitle.slice(0, 200) } : {}),
      ...(displaySubtitle
        ? { projectArtist: displaySubtitle.slice(0, 200) }
        : {}),
    }
  }
  const path = resolve(projectRoot, `${review.project_id}.json`)
  if (
    dirname(path) !== resolve(projectRoot) ||
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    !lstatSync(path).isFile()
  ) {
    return {}
  }
  try {
    const project = JSON.parse(readFileSync(path, 'utf8')) as unknown
    if (!isRecord(project)) return {}
    const projectTitle =
      typeof project.title === 'string' ? project.title.trim().slice(0, 200) : ''
    const projectArtist =
      typeof project.artist === 'string'
        ? project.artist.trim().slice(0, 200)
        : ''
    return {
      ...(projectTitle ? { projectTitle } : {}),
      ...(projectArtist ? { projectArtist } : {}),
    }
  } catch {
    return {}
  }
}

export function listReviews(
  root = defaultReviewRoot(),
  projectRoot = defaultProjectRoot(),
): ReviewSummary[] {
  return reviewNames(root).map((name) => {
    const review = readReview(name, root)
    return {
      name,
      status: review.status,
      ...projectMetadata(review, projectRoot),
      reviewKind: review.review_kind,
      eventCount: review.events.length,
      duration: review.duration,
      measureNumbers: review.measure_numbers,
      capo: review.capo,
      checks: review.checks,
    }
  })
}

export function getReview(
  name: string,
  root = defaultReviewRoot(),
  mediaPrefix = '/media/training/reviews',
): ReviewDocument {
  const review = readReview(name, root)
  return {
    name,
    audioUrl: `${mediaPrefix}/${encodeURIComponent(review.review_audio)}`,
    review,
  }
}

function applyDraft(
  review: ReviewPackage,
  input: ReviewDraftInput,
): ReviewPackage {
  if (review.status === 'approved') {
    throw new ReviewRequestError('已批准的审核包不能再修改', 409)
  }
  const capo = input.capo === undefined ? review.capo : integer(input.capo, '变调夹')
  if (capo < 0 || capo > 12) {
    throw new ReviewRequestError('变调夹必须在 0 到 12 品之间')
  }
  return {
    ...review,
    status: 'draft',
    events: normalizeEvents(input.events, review.duration),
    excluded_ranges: normalizeExcludedRanges(
      input.excluded_ranges ?? review.excluded_ranges ?? [],
      review.duration,
    ),
    checks: normalizeChecks(input.checks),
    capo,
    approval: null,
  }
}

export function saveReviewDraft(
  name: string,
  input: ReviewDraftInput,
  root = defaultReviewRoot(),
  mediaPrefix = '/media/training/reviews',
): ReviewDocument {
  const review = applyDraft(readReview(name, root), input)
  validateReview(review)
  writeReview(name, review, root)
  return getReview(name, root, mediaPrefix)
}

export function approveReview(
  name: string,
  input: ReviewDraftInput & { reviewerAlias?: unknown },
  root = defaultReviewRoot(),
  mediaPrefix = '/media/training/reviews',
): ReviewDocument {
  const reviewerAlias =
    typeof input.reviewerAlias === 'string' ? input.reviewerAlias.trim() : ''
  if (!REVIEWER_ALIAS_PATTERN.test(reviewerAlias)) {
    throw new ReviewRequestError('审核人代号只能包含字母、数字、点、下划线和连字符')
  }
  const review = applyDraft(readReview(name, root), input)
  const missing = requiredChecksForReview(review).filter(
    (key) => !review.checks[key],
  )
  if (missing.length > 0) {
    throw new ReviewRequestError(`批准前必须完成检查：${missing.join(', ')}`)
  }
  review.status = 'approved'
  review.approval = {
    reviewer_alias: reviewerAlias,
    reviewed_at: new Date().toISOString(),
    content_sha256: reviewContentDigest(review),
  }
  validateReview(review)
  writeReview(name, review, root)
  return getReview(name, root, mediaPrefix)
}

export function reopenReview(
  name: string,
  reason: unknown,
  root = defaultReviewRoot(),
  mediaPrefix = '/media/training/reviews',
): ReviewDocument {
  const review = readReview(name, root)
  if (review.status !== 'approved' || !review.approval) {
    throw new ReviewRequestError('只有已批准的审核包可以重新打开', 409)
  }
  const normalizedReason =
    typeof reason === 'string' && reason.trim()
      ? reason.trim().slice(0, 200)
      : '人工修订'
  review.approval_history = [
    ...(review.approval_history ?? []),
    {
      ...review.approval,
      invalidated_at: new Date().toISOString(),
      reason: normalizedReason,
    },
  ]
  review.status = 'draft'
  review.checks = {
    ...review.checks,
    timing: false,
  }
  review.approval = null
  validateReview(review)
  writeReview(name, review, root)
  return getReview(name, root, mediaPrefix)
}
