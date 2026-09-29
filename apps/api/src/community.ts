import {
  createHash,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from 'node:crypto'
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { relative, resolve } from 'node:path'
import type { ReviewDocument, ReviewEvent, ReviewPackage } from './reviews.js'

const SAFE_ALIAS = /^[\p{L}\p{N}][\p{L}\p{N}._-]{1,31}$/u
const SAFE_RUNNER_ID = /^[a-z0-9][a-z0-9._-]{1,63}$/
const SAFE_METRIC_NAME = /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/
const SHA256 = /^[0-9a-f]{64}$/
const MAX_AUDIT_EVENTS = 500
const DEFAULT_RUNNER_LEASE_SECONDS = 300
const MAX_ACTIVE_EXPERIMENTS_PER_ACCOUNT = 2
const MAX_DAILY_EXPERIMENTS_PER_ACCOUNT = 10
const MAX_EXPERIMENT_ATTEMPTS = 3
const BUILTIN_DATASETS: ReadonlyMap<
  string,
  { manifest: string; name: string }
> = new Map([
  [
    'builtin:guitarset-v1',
    {
      manifest: 'data/training/guitarset/manifest.jsonl',
      name: 'GuitarSet v1',
    },
  ],
  [
    'builtin:synthetic-smoke-v1',
    {
      manifest: 'data/training/synthetic-smoke/manifest.jsonl',
      name: 'Synthetic smoke v1',
    },
  ],
] as const)

export type CommunityRole =
  | 'contributor'
  | 'experimenter'
  | 'owner'
  | 'data-publisher'
  | 'model-maintainer'
  | 'governance-admin'

export type CommunityTaskType =
  | 'audio-quality'
  | 'event-presence'
  | 'missing-event'
  | 'timing'
  | 'pitch'
  | 'string-fret'
  | 'technique'

export type CommunityAnswer = {
  choice?: 'yes' | 'no' | 'unsure'
  onset?: number
  offset?: number
  pitch?: number
  string?: number
  fret?: number
  technique?: string
}

interface ReputationCounter {
  submissions: number
  consensusMatches: number
  goldTotal: number
  goldCorrect: number
}

interface CommunityAccount {
  id: string
  alias: string
  tokenHash: string
  riskClusterHash: string
  roles: CommunityRole[]
  reputation: Partial<Record<CommunityTaskType, ReputationCounter>>
  createdAt: string
  disabledAt?: string
}

export interface CommunityAccountView {
  id: string
  alias: string
  roles: CommunityRole[]
  governanceMode: CommunityState['governanceMode']
  reputation: Partial<
    Record<
      CommunityTaskType,
      ReputationCounter & { goldAccuracy: number; qualified: boolean }
    >
  >
  createdAt: string
}

export interface CommunityCredential {
  account: CommunityAccountView
  token: string
}

interface CommunitySource {
  id: string
  reviewQueue: string
  reviewName: string
  title: string
  subtitle?: string
  audioUrl: string
  audioSha256: string
  trainingAudioSha256?: string
  dataset: string
  license: string
  rightsStatus: 'active' | 'withdrawn'
  performerId: string
  workId: string
  recordingId: string
  sessionId: string
  lineageId: string
  importedAt: string
  withdrawnAt?: string
  withdrawalReason?: string
}

export interface CommunityTask {
  id: string
  schemaVersion: 1
  sourceId: string
  type: CommunityTaskType
  difficulty: 'entry' | 'calibrated' | 'expert'
  status: 'open' | 'consensus' | 'escalated' | 'withdrawn'
  question: string
  context: {
    start: number
    end: number
    cueAt?: number
  }
  requiredResponses: number
  maxResponses: number
  snapshotSha256: string
  createdAt: string
  consensus?: {
    answer: CommunityAnswer
    method: 'community' | 'owner'
    supportingSubmissionIds: string[]
    resolvedAt: string
  }
  calibrationAnswer?: CommunityAnswer
}

export interface CommunityTaskView
  extends Omit<CommunityTask, 'calibrationAnswer'> {
  source: {
    title: string
    subtitle?: string
    audioUrl: string
    dataset: string
    license: string
  }
  submissionCount: number
}

interface CommunitySubmission {
  id: string
  taskId: string
  accountId: string
  riskClusterHash: string
  answer: CommunityAnswer
  confidence: number
  durationMs: number
  createdAt: string
  withdrawnAt?: string
  calibrationCorrect?: boolean
}

export interface CommunitySubmissionResult {
  submissionId: string
  taskStatus: CommunityTask['status']
  consensus?: CommunityTask['consensus']
  calibration?: {
    correct: boolean
  }
}

export interface CommunityContributionView {
  id: string
  taskId: string
  taskType: CommunityTaskType
  question: string
  sourceTitle: string
  answer: CommunityAnswer
  confidence: number
  createdAt: string
  withdrawnAt?: string
  calibrationCorrect?: boolean
}

export interface DatasetRelease {
  id: string
  name: string
  version: number
  status: 'active' | 'blocked'
  verification: 'owner_verified' | 'adjudicated_gold'
  taskIds: string[]
  sourceIds: string[]
  manifestSha256: string
  createdBy: string
  createdAt: string
  blockedAt?: string
  blockedReason?: string
}

export interface DatasetReleaseManifest {
  schemaVersion: 1
  release: {
    id: string
    name: string
    version: number
    verification: DatasetRelease['verification']
    manifestSha256: string
    createdAt: string
  }
  records: Array<{
    taskId: string
    type: CommunityTaskType
    answer: CommunityAnswer
    snapshotSha256: string
    source: {
      id: string
      dataset: string
      license: string
      performerId: string
      workId: string
      recordingId: string
      sessionId: string
      lineageId: string
      reviewQueue: string
      reviewName: string
    }
  }>
}

type RecipeParameterRule = {
  type: 'number' | 'integer'
  minimum: number
  maximum: number
  default: number
}

export interface TrainingRecipe {
  id: string
  specVersion: 1
  name: string
  modelFamily: string
  executor: 'builtin-audit' | 'external-worker'
  availability: 'ready' | 'planned'
  compatibleDatasets: string[]
  promotionEligible: boolean
  resourceClass: 'control-plane' | 'cpu-small' | 'gpu-standard'
  timeoutSeconds: number
  description: string
  parameters: Record<string, RecipeParameterRule>
}

export interface ExperimentExecutionEvidence {
  isolation: 'sandbox-exec' | 'podman'
  durationMs: number
  stdoutSha256: string
  stderrSha256: string
  reportSha256: string
}

export interface ExperimentRun {
  id: string
  recipeId: string
  datasetReleaseId: string
  submittedBy: string
  parameters: Record<string, number>
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  createdAt: string
  updatedAt: string
  codeRevision: string
  runtimeImage: string
  datasetManifestSha256: string
  lineageSha256: string
  attemptCount: number
  lastRunnerId?: string
  lease?: {
    runnerId: string
    claimedAt: string
    heartbeatAt: string
    expiresAt: string
    attempt: number
  }
  metrics?: Record<string, number>
  artifactSha256?: string
  artifactSizeBytes?: number
  artifactMediaType?: string
  executionEvidence?: ExperimentExecutionEvidence
  modelVersion?: string
  failureReason?: string
}

interface ExperimentRecord extends Omit<ExperimentRun, 'lease'> {
  lease?: NonNullable<ExperimentRun['lease']> & {
    tokenHash: string
  }
  uploadedArtifact?: {
    runnerId: string
    attempt: number
    sha256: string
    sizeBytes: number
    mediaType: string
    objectKey: string
    uploadedAt: string
  }
  artifactObjectKey?: string
}

export interface TrainingRunner {
  id: string
}

export interface TrainingRunnerClaim {
  schemaVersion: 1
  leaseToken: string
  heartbeatIntervalSeconds: number
  leaseExpiresAt: string
  job: {
    experimentId: string
    recipe: {
      id: string
      specVersion: 1
      modelFamily: string
      resourceClass: TrainingRecipe['resourceClass']
      timeoutSeconds: number
      promotionEligible: boolean
    }
    dataset: {
      id: string
      kind: 'builtin' | 'community-release'
      manifestSha256: string
    }
    parameters: Record<string, number>
    codeRevision: string
    runtimeImage: string
    lineageSha256: string
  }
}

export interface CommunityStoreOptions {
  bootstrapKey?: string
  now?: () => Date
  runnerCredentials?: Record<string, string>
  runnerLeaseSeconds?: number
  builtinDatasetManifestSha256?: Partial<Record<string, string>>
  gateCredentials?: Record<string, string>
  sealedGateCredentials?: Record<string, string>
}

export interface PromotionGateAgent {
  id: string
  scope: 'public' | 'sealed'
}

export type PromotionCheck =
  | 'lineage'
  | 'reproduction'
  | 'public-validation'
  | 'sealed-evaluation'
  | 'robustness'
  | 'shadow'
  | 'canary'

export interface ModelPromotion {
  id: string
  experimentId: string
  modelVersion: string
  submittedBy: string
  status:
    | 'candidate'
    | 'shadow'
    | 'canary'
    | 'champion'
    | 'rejected'
    | 'retired'
  checks: Partial<
    Record<
      PromotionCheck,
      {
        passed: boolean
        summary: string
        recordedAt: string
        recordedBy: string
        source?: 'system' | 'gate-service' | 'maintainer'
        evidenceSha256?: string
      }
    >
  >
  approvals: Array<{
    target: 'shadow' | 'canary' | 'champion'
    accountId: string
    approvedAt: string
  }>
  createdAt: string
  updatedAt: string
  previousChampionId?: string
}

export type InferenceDeploymentMode =
  | 'baseline'
  | 'shadow'
  | 'canary'
  | 'champion'

export interface InferenceDeploymentPolicy {
  mode: InferenceDeploymentMode
  promotionId?: string
  modelVersion?: string
  artifactPath?: string
  baselineModelVersion?: string
  baselineArtifactPath?: string
  baselineFallbackModelVersion?: string
  baselineFallbackArtifactPath?: string
}

export interface InferenceObservation {
  id: string
  promotionId: string
  jobId: string
  mode: Exclude<InferenceDeploymentMode, 'baseline'>
  success: boolean
  fallbackUsed: boolean
  durationMs: number
  baselineNoteCount?: number
  candidateNoteCount?: number
  noteCountDelta?: number
  error?: string
  recordedAt: string
}

export interface InferenceDeploymentStatus {
  config: {
    shadowSamplePercent: number
    canaryTrafficPercent: number
    errorBudgetPercent: number
    minimumObservations: number
  }
  active?: {
    promotionId: string
    modelVersion: string
    mode: 'shadow' | 'canary' | 'champion'
  }
  observations: {
    total: number
    successful: number
    fallbackCount: number
    errorRate: number
    averageDurationMs: number
    lastRecordedAt?: string
  }
  recent: InferenceObservation[]
}

export interface PromotionGateJob {
  promotion: ModelPromotion
  experiment: ExperimentRun
  reproductions: Array<{
    experimentId: string
    artifactSha256: string
    metrics: Record<string, number>
  }>
}

export interface ModelCard {
  schemaVersion: 1
  model: {
    version: string
    family: string
    status: ModelPromotion['status']
    createdAt: string
  }
  training: {
    recipeId: string
    datasetReleaseId: string
    datasetManifestSha256: string
    codeRevision: string
    runtimeImage: string
    parameters: Record<string, number>
    lineageSha256: string
  }
  artifact: {
    sha256: string
    sizeBytes: number
    mediaType: string
  }
  metrics: Record<string, number>
  checks: ModelPromotion['checks']
  limitations: string[]
}

interface AuditEvent {
  id: string
  at: string
  actorId: string
  action: string
  entityType: string
  entityId: string
  detailsSha256: string
}

interface CommunityState {
  schemaVersion: 1
  governanceMode: 'single-maintainer' | 'team'
  accounts: CommunityAccount[]
  sources: CommunitySource[]
  tasks: CommunityTask[]
  submissions: CommunitySubmission[]
  releases: DatasetRelease[]
  experiments: ExperimentRecord[]
  promotions: ModelPromotion[]
  championPromotionId?: string
  deploymentConfig: InferenceDeploymentStatus['config']
  inferenceObservations: InferenceObservation[]
}

export interface CommunityDashboard {
  governanceMode: CommunityState['governanceMode']
  counts: {
    contributors: number
    openTasks: number
    consensusTasks: number
    escalatedTasks: number
    submissions: number
    activeReleases: number
    queuedExperiments: number
    completedExperiments: number
    promotionCandidates: number
  }
  quality: {
    agreementRate: number
    calibrationAccuracy: number
    rightsCoverage: number
  }
  phases: Array<{
    id: 'C0' | 'C1' | 'C2' | 'C3' | 'C4' | 'C5' | 'C6'
    label: string
    status: 'ready' | 'active' | 'waiting'
    detail: string
  }>
  maintenance: Array<{
    severity: 'info' | 'warning'
    message: string
  }>
}

export class CommunityRequestError extends Error {
  constructor(
    message: string,
    readonly statusCode = 400,
  ) {
    super(message)
  }
}

const RECIPES: TrainingRecipe[] = [
  {
    id: 'dataset-audit-v1',
    specVersion: 1,
    name: '数据完整性审计',
    modelFamily: 'dataset-audit',
    executor: 'builtin-audit',
    availability: 'ready',
    compatibleDatasets: ['*'],
    promotionEligible: false,
    resourceClass: 'control-plane',
    timeoutSeconds: 30,
    description: '验证清单、来源、许可、切分和内容哈希，不训练模型。',
    parameters: {},
  },
  {
    id: 'onset-position-smoke-v1',
    specVersion: 1,
    name: '弦品网络隔离 Smoke',
    modelFamily: 'onset-position-net',
    executor: 'external-worker',
    availability: 'ready',
    compatibleDatasets: ['builtin:synthetic-smoke-v1'],
    promotionEligible: false,
    resourceClass: 'cpu-small',
    timeoutSeconds: 900,
    description: '使用非基准合成数据验证隔离执行、制品上传和结果回传闭环。',
    parameters: {
      epochs: { type: 'integer', minimum: 1, maximum: 3, default: 1 },
      learningRate: {
        type: 'number',
        minimum: 0.000001,
        maximum: 0.001,
        default: 0.0003,
      },
      seed: { type: 'integer', minimum: 1, maximum: 2147483647, default: 20260929 },
    },
  },
  {
    id: 'community-release-audit-v1',
    specVersion: 1,
    name: '社区快照执行审计',
    modelFamily: 'dataset-audit',
    executor: 'external-worker',
    availability: 'ready',
    compatibleDatasets: ['*'],
    promotionEligible: false,
    resourceClass: 'cpu-small',
    timeoutSeconds: 120,
    description: '在隔离执行器中复核冻结清单哈希并生成不可变审计制品。',
    parameters: {},
  },
  {
    id: 'onset-position-guitarset-v1',
    specVersion: 1,
    name: 'GuitarSet 弦品起音网络',
    modelFamily: 'onset-position-net',
    executor: 'external-worker',
    availability: 'ready',
    compatibleDatasets: ['builtin:guitarset-v1'],
    promotionEligible: true,
    resourceClass: 'gpu-standard',
    timeoutSeconds: 21600,
    description: '使用固定 GuitarSet 训练清单运行现有弦品起音网络训练入口。',
    parameters: {
      epochs: { type: 'integer', minimum: 1, maximum: 50, default: 12 },
      learningRate: {
        type: 'number',
        minimum: 0.000001,
        maximum: 0.001,
        default: 0.00003,
      },
      seed: { type: 'integer', minimum: 1, maximum: 2147483647, default: 20260929 },
    },
  },
  {
    id: 'candidate-activity-temporal-v1',
    specVersion: 1,
    name: '重复起音 Activity Head',
    modelFamily: 'candidate-activity-head',
    executor: 'external-worker',
    availability: 'planned',
    compatibleDatasets: ['builtin:guitarset-v1'],
    promotionEligible: true,
    resourceClass: 'gpu-standard',
    timeoutSeconds: 21600,
    description: '待可移植 GuitarSet 快照接入后训练现有 temporal activity head。',
    parameters: {
      epochs: { type: 'integer', minimum: 1, maximum: 50, default: 5 },
      learningRate: {
        type: 'number',
        minimum: 0.000001,
        maximum: 0.001,
        default: 0.00003,
      },
      seed: { type: 'integer', minimum: 1, maximum: 2147483647, default: 20260929 },
    },
  },
]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value)
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new CommunityRequestError('内容包含无效数字')
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
  throw new CommunityRequestError('内容包含不支持的值')
}

function digest(value: unknown): string {
  return createHash('sha256').update(canonicalJson(value), 'utf8').digest('hex')
}

function hashSecret(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex')
}

function safeEqual(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left)
  const rightBuffer = Buffer.from(right)
  return (
    leftBuffer.length === rightBuffer.length &&
    timingSafeEqual(leftBuffer, rightBuffer)
  )
}

function emptyState(): CommunityState {
  return {
    schemaVersion: 1,
    governanceMode: 'single-maintainer',
    accounts: [],
    sources: [],
    tasks: [],
    submissions: [],
    releases: [],
    experiments: [],
    promotions: [],
    deploymentConfig: {
      shadowSamplePercent: 10,
      canaryTrafficPercent: 5,
      errorBudgetPercent: 5,
      minimumObservations: 10,
    },
    inferenceObservations: [],
  }
}

function accountView(
  account: CommunityAccount,
  governanceMode: CommunityState['governanceMode'],
): CommunityAccountView {
  return {
    id: account.id,
    alias: account.alias,
    roles: [...account.roles],
    governanceMode,
    reputation: Object.fromEntries(
      Object.entries(account.reputation).map(([key, value]) => [
        key,
        {
          ...value,
          goldAccuracy: value.goldTotal
            ? value.goldCorrect / value.goldTotal
            : 0,
          qualified:
            account.roles.includes('owner') ||
            (value.goldTotal >= 3 && value.goldCorrect / value.goldTotal >= 0.8),
        },
      ]),
    ),
    createdAt: account.createdAt,
  }
}

function normalizeAnswer(
  task: Pick<CommunityTask, 'type' | 'context'>,
  value: unknown,
): CommunityAnswer {
  if (!isRecord(value)) {
    throw new CommunityRequestError('答案格式无效')
  }
  if (
    task.type === 'audio-quality' ||
    task.type === 'event-presence' ||
    task.type === 'missing-event'
  ) {
    if (value.choice !== 'yes' && value.choice !== 'no' && value.choice !== 'unsure') {
      throw new CommunityRequestError('请选择是、否或无法判断')
    }
    return { choice: value.choice }
  }
  if (task.type === 'timing') {
    const onset = Number(value.onset)
    const offset = Number(value.offset)
    if (
      !Number.isFinite(onset) ||
      !Number.isFinite(offset) ||
      onset < task.context.start ||
      onset >= offset ||
      offset > task.context.end
    ) {
      throw new CommunityRequestError('时间边界超出任务片段')
    }
    return { onset, offset }
  }
  if (task.type === 'pitch') {
    const pitch = Number(value.pitch)
    if (!Number.isInteger(pitch) || pitch < 0 || pitch > 127) {
      throw new CommunityRequestError('MIDI 音高必须在 0 到 127 之间')
    }
    return { pitch }
  }
  if (task.type === 'string-fret') {
    const string = Number(value.string)
    const fret = Number(value.fret)
    if (
      !Number.isInteger(string) ||
      string < 1 ||
      string > 6 ||
      !Number.isInteger(fret) ||
      fret < 0 ||
      fret > 24
    ) {
      throw new CommunityRequestError('弦号或品位无效')
    }
    return { string, fret }
  }
  const technique =
    typeof value.technique === 'string' ? value.technique.trim() : ''
  if (!/^[a-z][a-z-]{1,31}$/.test(technique)) {
    throw new CommunityRequestError('技法标签无效')
  }
  return { technique }
}

function answerKey(task: CommunityTask, answer: CommunityAnswer): string {
  if (answer.choice) return answer.choice
  if (task.type === 'timing') {
    return `${Math.round((answer.onset ?? 0) * 100) / 100}:${
      Math.round((answer.offset ?? 0) * 100) / 100
    }`
  }
  if (task.type === 'pitch') return String(answer.pitch)
  if (task.type === 'string-fret') return `${answer.string}:${answer.fret}`
  return String(answer.technique)
}

function requiredRole(account: CommunityAccount, role: CommunityRole): void {
  if (!account.roles.includes(role) && !account.roles.includes('owner')) {
    throw new CommunityRequestError('当前账号没有执行此操作的权限', 403)
  }
}

function finiteInput(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number,
  integer: boolean,
): number {
  const parsed = Number(value)
  if (
    !Number.isFinite(parsed) ||
    parsed < minimum ||
    parsed > maximum ||
    (integer && !Number.isInteger(parsed))
  ) {
    throw new CommunityRequestError(`${label} 超出允许范围`)
  }
  return parsed
}

function percentageBucket(value: string): number {
  const prefix = createHash('sha256').update(value, 'utf8').digest().readUInt32BE(0)
  return (prefix / 0x1_0000_0000) * 100
}

function statusLabelForDeployment(
  mode: Exclude<InferenceDeploymentMode, 'baseline'>,
): string {
  return {
    shadow: '影子',
    canary: '灰度',
    champion: '生产',
  }[mode]
}

function normalizeExperimentMetrics(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    throw new CommunityRequestError('完成实验必须提供指标')
  }
  const entries = Object.entries(value)
  if (entries.length === 0 || entries.length > 64) {
    throw new CommunityRequestError('实验指标数量必须在 1 到 64 之间')
  }
  return Object.fromEntries(
    entries.map(([key, metric]) => {
      if (!SAFE_METRIC_NAME.test(key)) {
        throw new CommunityRequestError(`实验指标名称无效：${key}`)
      }
      return [key, finiteInput(metric, key, -1_000_000, 1_000_000, false)]
    }),
  )
}

function normalizeExecutionEvidence(
  value: unknown,
): ExperimentExecutionEvidence {
  if (!isRecord(value)) {
    throw new CommunityRequestError('完成实验必须提供执行证据')
  }
  if (value.isolation !== 'sandbox-exec' && value.isolation !== 'podman') {
    throw new CommunityRequestError('执行隔离方式无效')
  }
  const durationMs = finiteInput(
    value.durationMs,
    'durationMs',
    1,
    86_400_000,
    true,
  )
  const hashes = [
    value.stdoutSha256,
    value.stderrSha256,
    value.reportSha256,
  ]
  if (
    hashes.some(
      (hash) => typeof hash !== 'string' || !SHA256.test(hash),
    )
  ) {
    throw new CommunityRequestError('执行证据 SHA-256 无效')
  }
  return {
    isolation: value.isolation,
    durationMs,
    stdoutSha256: value.stdoutSha256 as string,
    stderrSha256: value.stderrSha256 as string,
    reportSha256: value.reportSha256 as string,
  }
}

function sourceIdentity(review: ReviewPackage): {
  dataset: string
  license: string
  performerId: string
  workId: string
  recordingId: string
  sessionId: string
  lineageId: string
} {
  const provenance = review.training_provenance
  if (!provenance) {
    throw new CommunityRequestError('只有具备训练来源与许可证的审核包可以进入社区池')
  }
  if (!/^(CC0|CC-BY(?:-[A-Z]+)?-[0-9.]+)$/i.test(provenance.source_license)) {
    throw new CommunityRequestError('审核包许可证不允许进入社区训练池')
  }
  return {
    dataset: provenance.source_dataset,
    license: provenance.source_license,
    performerId: provenance.source_group_id,
    workId: provenance.review_source_track_id,
    recordingId: provenance.source_track_id,
    sessionId: review.project_id,
    lineageId: digest({
      source: review.source_audio_sha256,
      training: review.training_audio_sha256,
    }),
  }
}

function eventContext(event: ReviewEvent, duration: number): {
  start: number
  end: number
  cueAt: number
} {
  return {
    start: Math.max(0, event.onset - 0.35),
    end: Math.min(duration, Math.max(event.offset, event.onset + 0.35) + 0.15),
    cueAt: event.onset,
  }
}

export class CommunityStore {
  private readonly statePath: string
  private readonly auditPath: string
  private state: CommunityState

  constructor(
    private readonly root = resolve(process.cwd(), 'data', 'community'),
    private readonly options: CommunityStoreOptions = {},
  ) {
    for (const [runnerId, secret] of Object.entries(
      options.runnerCredentials ?? {},
    )) {
      if (
        !SAFE_RUNNER_ID.test(runnerId) ||
        typeof secret !== 'string' ||
        secret.length < 24
      ) {
        throw new CommunityRequestError('训练执行器凭据配置无效', 500)
      }
    }
    for (const [datasetId, manifestSha256] of Object.entries(
      options.builtinDatasetManifestSha256 ?? {},
    )) {
      if (
        !BUILTIN_DATASETS.has(datasetId) ||
        typeof manifestSha256 !== 'string' ||
        !SHA256.test(manifestSha256)
      ) {
        throw new CommunityRequestError('内置数据清单配置无效', 500)
      }
    }
    for (const credentials of [
      options.gateCredentials ?? {},
      options.sealedGateCredentials ?? {},
    ]) {
      for (const [gateId, secret] of Object.entries(credentials)) {
        if (
          !SAFE_RUNNER_ID.test(gateId) ||
          typeof secret !== 'string' ||
          secret.length < 24
        ) {
          throw new CommunityRequestError('模型门禁服务凭据配置无效', 500)
        }
      }
    }
    mkdirSync(root, { recursive: true })
    this.statePath = resolve(root, 'state.json')
    this.auditPath = resolve(root, 'audit.jsonl')
    this.state = this.readState()
  }

  private nowDate(): Date {
    return this.options.now?.() ?? new Date()
  }

  private now(): string {
    return this.nowDate().toISOString()
  }

  private runnerLeaseSeconds(): number {
    const configured = this.options.runnerLeaseSeconds
    return (
      typeof configured === 'number' &&
      Number.isInteger(configured) &&
      configured >= 30 &&
      configured <= 3600
    )
      ? configured
      : DEFAULT_RUNNER_LEASE_SECONDS
  }

  private readState(): CommunityState {
    if (!existsSync(this.statePath)) return emptyState()
    try {
      const value = JSON.parse(readFileSync(this.statePath, 'utf8')) as CommunityState
      if (value.schemaVersion !== 1) return emptyState()
      value.experiments = (value.experiments ?? []).map((experiment) => ({
        ...experiment,
        attemptCount:
          experiment.attemptCount ?? experiment.lease?.attempt ?? 0,
      }))
      value.deploymentConfig ??= emptyState().deploymentConfig
      value.inferenceObservations ??= []
      return value
    } catch {
      throw new CommunityRequestError('社区状态文件无法读取', 500)
    }
  }

  private persist(): void {
    const temporary = resolve(
      this.root,
      `.state.${process.pid}.${randomUUID()}.tmp`,
    )
    writeFileSync(temporary, `${JSON.stringify(this.state, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
    renameSync(temporary, this.statePath)
  }

  private audit(
    actorId: string,
    action: string,
    entityType: string,
    entityId: string,
    details: unknown,
  ): void {
    const event: AuditEvent = {
      id: randomUUID(),
      at: this.now(),
      actorId,
      action,
      entityType,
      entityId,
      detailsSha256: digest(details),
    }
    appendFileSync(this.auditPath, `${JSON.stringify(event)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    })
  }

  registerAccount(input: {
    alias: unknown
    riskHint: string
    bootstrapKey?: unknown
  }): CommunityCredential {
    const alias = typeof input.alias === 'string' ? input.alias.trim() : ''
    if (!SAFE_ALIAS.test(alias)) {
      throw new CommunityRequestError('代号需为 2 到 32 个字母、数字、点、下划线或连字符')
    }
    if (
      this.state.accounts.some(
        (account) => account.alias.toLowerCase() === alias.toLowerCase(),
      )
    ) {
      throw new CommunityRequestError('该代号已存在', 409)
    }
    const isOwner = this.state.accounts.length === 0
    if (
      isOwner &&
      this.options.bootstrapKey &&
      input.bootstrapKey !== this.options.bootstrapKey
    ) {
      throw new CommunityRequestError('首个所有者账号需要有效引导密钥', 403)
    }
    const id = randomUUID()
    const secret = randomBytes(32).toString('base64url')
    const roles: CommunityRole[] = isOwner
      ? [
          'contributor',
          'experimenter',
          'owner',
          'data-publisher',
          'model-maintainer',
          'governance-admin',
        ]
      : ['contributor', 'experimenter']
    const account: CommunityAccount = {
      id,
      alias,
      tokenHash: hashSecret(secret),
      riskClusterHash: hashSecret(`risk:${input.riskHint}`),
      roles,
      reputation: {},
      createdAt: this.now(),
    }
    this.state.accounts.push(account)
    this.persist()
    this.audit(id, 'account.created', 'account', id, { alias, roles })
    return {
      account: accountView(account, this.state.governanceMode),
      token: `${id}.${secret}`,
    }
  }

  authenticate(authorization: unknown): CommunityAccount {
    if (typeof authorization !== 'string' || !authorization.startsWith('Bearer ')) {
      throw new CommunityRequestError('需要社区账号凭据', 401)
    }
    const credential = authorization.slice(7)
    const separator = credential.indexOf('.')
    if (separator <= 0) {
      throw new CommunityRequestError('社区账号凭据无效', 401)
    }
    const id = credential.slice(0, separator)
    const secret = credential.slice(separator + 1)
    const account = this.state.accounts.find((candidate) => candidate.id === id)
    if (
      !account ||
      account.disabledAt ||
      !safeEqual(account.tokenHash, hashSecret(secret))
    ) {
      throw new CommunityRequestError('社区账号凭据无效', 401)
    }
    return account
  }

  authenticateRunner(authorization: unknown): TrainingRunner {
    if (
      typeof authorization !== 'string' ||
      !authorization.startsWith('Runner ')
    ) {
      throw new CommunityRequestError('需要训练执行器凭据', 401)
    }
    const credential = authorization.slice(7)
    const separator = credential.indexOf('.')
    if (separator <= 0) {
      throw new CommunityRequestError('训练执行器凭据无效', 401)
    }
    const id = credential.slice(0, separator)
    const secret = credential.slice(separator + 1)
    const expected = this.options.runnerCredentials?.[id]
    if (
      !expected ||
      !SAFE_RUNNER_ID.test(id) ||
      !safeEqual(hashSecret(expected), hashSecret(secret))
    ) {
      throw new CommunityRequestError('训练执行器凭据无效', 401)
    }
    return { id }
  }

  authenticateSealedPromotionGate(
    authorization: unknown,
  ): PromotionGateAgent {
    if (
      typeof authorization !== 'string' ||
      !authorization.startsWith('SealedGate ')
    ) {
      throw new CommunityRequestError('需要密封评测服务凭据', 401)
    }
    const credential = authorization.slice(11)
    const separator = credential.indexOf('.')
    if (separator <= 0) {
      throw new CommunityRequestError('密封评测服务凭据无效', 401)
    }
    const id = credential.slice(0, separator)
    const secret = credential.slice(separator + 1)
    const expected = this.options.sealedGateCredentials?.[id]
    if (
      !expected ||
      !SAFE_RUNNER_ID.test(id) ||
      !safeEqual(hashSecret(expected), hashSecret(secret))
    ) {
      throw new CommunityRequestError('密封评测服务凭据无效', 401)
    }
    return { id, scope: 'sealed' }
  }

  authenticatePromotionGate(authorization: unknown): PromotionGateAgent {
    if (
      typeof authorization !== 'string' ||
      !authorization.startsWith('Gate ')
    ) {
      throw new CommunityRequestError('需要模型门禁服务凭据', 401)
    }
    const credential = authorization.slice(5)
    const separator = credential.indexOf('.')
    if (separator <= 0) {
      throw new CommunityRequestError('模型门禁服务凭据无效', 401)
    }
    const id = credential.slice(0, separator)
    const secret = credential.slice(separator + 1)
    const expected = this.options.gateCredentials?.[id]
    if (
      !expected ||
      !SAFE_RUNNER_ID.test(id) ||
      !safeEqual(hashSecret(expected), hashSecret(secret))
    ) {
      throw new CommunityRequestError('模型门禁服务凭据无效', 401)
    }
    return { id, scope: 'public' }
  }

  getAccount(account: CommunityAccount): CommunityAccountView {
    return accountView(account, this.state.governanceMode)
  }

  listRecipes(): TrainingRecipe[] {
    return RECIPES.map((recipe) => ({
      ...recipe,
      compatibleDatasets: [...recipe.compatibleDatasets],
      parameters: { ...recipe.parameters },
    }))
  }

  listBuiltinDatasets(): Array<{ id: string; name: string }> {
    return [...BUILTIN_DATASETS.entries()]
      .filter(([id]) => Boolean(this.options.builtinDatasetManifestSha256?.[id]))
      .map(([id, dataset]) => ({ id, name: dataset.name }))
  }

  listTasks(account: CommunityAccount): CommunityTaskView[] {
    const submitted = new Set(
      this.state.submissions
        .filter(
          (submission) =>
            submission.accountId === account.id && !submission.withdrawnAt,
        )
        .map((submission) => submission.taskId),
    )
    return this.state.tasks
      .filter(
        (task) =>
          !submitted.has(task.id) &&
          (task.calibrationAnswer
            ? task.status !== 'withdrawn'
            : task.status === 'open' || task.status === 'escalated'),
      )
      .sort((left, right) => {
        const calibrationOrder =
          Number(Boolean(right.calibrationAnswer)) -
          Number(Boolean(left.calibrationAnswer))
        return calibrationOrder || left.createdAt.localeCompare(right.createdAt)
      })
      .map((task) => this.taskView(task))
  }

  listAllTasks(account: CommunityAccount): CommunityTaskView[] {
    requiredRole(account, 'data-publisher')
    return this.state.tasks.map((task) => this.taskView(task))
  }

  private taskView(task: CommunityTask): CommunityTaskView {
    const { calibrationAnswer: _calibrationAnswer, ...publicTask } = task
    const source = this.state.sources.find((candidate) => candidate.id === task.sourceId)
    if (!source) {
      throw new CommunityRequestError('社区任务来源不存在', 500)
    }
    return {
      ...publicTask,
      source: {
        title: source.title,
        subtitle: source.subtitle,
        audioUrl: source.audioUrl,
        dataset: source.dataset,
        license: source.license,
      },
      submissionCount: this.state.submissions.filter(
        (submission) => submission.taskId === task.id && !submission.withdrawnAt,
      ).length,
    }
  }

  importReview(
    account: CommunityAccount,
    document: ReviewDocument,
    queue: string,
  ): { sourceId: string; addedTasks: number } {
    requiredRole(account, 'data-publisher')
    const review = document.review
    const identity = sourceIdentity(review)
    const sourceId = digest({
      queue,
      name: document.name,
      audio: review.review_audio_sha256,
    }).slice(0, 24)
    let source = this.state.sources.find((candidate) => candidate.id === sourceId)
    if (!source) {
      source = {
        id: sourceId,
        reviewQueue: queue,
        reviewName: document.name,
        title: review.display_title ?? document.name,
        subtitle: review.display_subtitle,
        audioUrl: document.audioUrl,
        audioSha256: review.review_audio_sha256,
        trainingAudioSha256: review.training_audio_sha256,
        rightsStatus: 'active',
        importedAt: this.now(),
        ...identity,
      }
      this.state.sources.push(source)
    }

    const taskDefinitions: Array<
      Omit<CommunityTask, 'id' | 'snapshotSha256' | 'createdAt'>
    > = review.events.map((event, index) => ({
      schemaVersion: 1,
      sourceId,
      type: 'event-presence',
      difficulty: 'entry',
      status: 'open',
      question: '提示位置是否能听到一次新的触弦起音？',
      context: eventContext(event, review.duration),
      requiredResponses: 2,
      maxResponses: 3,
      ...(review.status === 'approved'
        ? { calibrationAnswer: { choice: 'yes' as const } }
        : {}),
    }))
    taskDefinitions.push({
      schemaVersion: 1,
      sourceId,
      type: 'missing-event',
      difficulty: 'entry',
      status: 'open',
      question: '这个片段是否还有未标出的清晰起音？',
      context: { start: 0, end: review.duration },
      requiredResponses: 2,
      maxResponses: 3,
      ...(review.status === 'approved'
        ? { calibrationAnswer: { choice: 'no' as const } }
        : {}),
    })

    let addedTasks = 0
    taskDefinitions.forEach((definition, index) => {
      const id = digest({
        sourceId,
        type: definition.type,
        index,
        context: definition.context,
      }).slice(0, 24)
      if (this.state.tasks.some((task) => task.id === id)) return
      const task: CommunityTask = {
        id,
        ...definition,
        snapshotSha256: digest({ source, definition }),
        createdAt: this.now(),
      }
      this.state.tasks.push(task)
      addedTasks += 1
    })
    this.persist()
    this.audit(account.id, 'review.imported', 'source', sourceId, {
      queue,
      reviewName: document.name,
      addedTasks,
    })
    return { sourceId, addedTasks }
  }

  submit(
    account: CommunityAccount,
    taskId: string,
    input: {
      answer: unknown
      confidence?: unknown
      durationMs?: unknown
    },
  ): CommunitySubmissionResult {
    const task = this.state.tasks.find((candidate) => candidate.id === taskId)
    if (!task || task.status === 'withdrawn') {
      throw new CommunityRequestError('社区任务不存在', 404)
    }
    if (
      this.state.submissions.some(
        (submission) =>
          submission.taskId === taskId &&
          submission.accountId === account.id &&
          !submission.withdrawnAt,
      )
    ) {
      throw new CommunityRequestError('该任务已经提交', 409)
    }
    const answer = normalizeAnswer(task, input.answer)
    const confidence = finiteInput(
      input.confidence ?? 1,
      '置信度',
      0,
      1,
      false,
    )
    const durationMs = finiteInput(
      input.durationMs ?? 0,
      '作答时长',
      0,
      60 * 60 * 1000,
      true,
    )
    const calibrationCorrect = task.calibrationAnswer
      ? answerKey(task, answer) === answerKey(task, task.calibrationAnswer)
      : undefined
    const submission: CommunitySubmission = {
      id: randomUUID(),
      taskId,
      accountId: account.id,
      riskClusterHash: account.riskClusterHash,
      answer,
      confidence,
      durationMs,
      createdAt: this.now(),
      ...(calibrationCorrect === undefined ? {} : { calibrationCorrect }),
    }
    this.state.submissions.push(submission)
    this.updateReputation(account, task.type, calibrationCorrect)
    if (!task.calibrationAnswer) this.recalculateConsensus(task)
    this.persist()
    this.audit(account.id, 'submission.created', 'submission', submission.id, {
      taskId,
      answer,
      calibration: calibrationCorrect !== undefined,
    })
    return {
      submissionId: submission.id,
      taskStatus: task.status,
      consensus: task.consensus,
      ...(calibrationCorrect === undefined
        ? {}
        : { calibration: { correct: calibrationCorrect } }),
    }
  }

  private updateReputation(
    account: CommunityAccount,
    type: CommunityTaskType,
    calibrationCorrect?: boolean,
  ): void {
    const current = account.reputation[type] ?? {
      submissions: 0,
      consensusMatches: 0,
      goldTotal: 0,
      goldCorrect: 0,
    }
    current.submissions += 1
    if (calibrationCorrect !== undefined) {
      current.goldTotal += 1
      if (calibrationCorrect) current.goldCorrect += 1
    }
    account.reputation[type] = current
  }

  private qualified(account: CommunityAccount, type: CommunityTaskType): boolean {
    if (account.roles.includes('owner')) return true
    const reputation = account.reputation[type]
    return Boolean(
      reputation &&
        reputation.goldTotal >= 3 &&
        reputation.goldCorrect / reputation.goldTotal >= 0.8,
    )
  }

  private recalculateConsensus(task: CommunityTask): void {
    const submissions = this.state.submissions.filter(
      (submission) => submission.taskId === task.id && !submission.withdrawnAt,
    )
    const byCluster = new Map<string, CommunitySubmission>()
    for (const submission of submissions) {
      const account = this.state.accounts.find(
        (candidate) => candidate.id === submission.accountId,
      )
      if (!account || !this.qualified(account, task.type)) continue
      if (!byCluster.has(submission.riskClusterHash)) {
        byCluster.set(submission.riskClusterHash, submission)
      }
    }
    const eligible = [...byCluster.values()].filter(
      (submission) => submission.answer.choice !== 'unsure',
    )
    const groups = new Map<string, CommunitySubmission[]>()
    for (const submission of eligible) {
      const key = answerKey(task, submission.answer)
      groups.set(key, [...(groups.get(key) ?? []), submission])
    }
    const winner = [...groups.values()].sort(
      (left, right) => right.length - left.length,
    )[0]
    if (winner && winner.length >= task.requiredResponses) {
      task.status = 'consensus'
      task.consensus = {
        answer: winner[0].answer,
        method: 'community',
        supportingSubmissionIds: winner.map((submission) => submission.id),
        resolvedAt: this.now(),
      }
      for (const submission of eligible) {
        const account = this.state.accounts.find(
          (candidate) => candidate.id === submission.accountId,
        )
        if (
          account &&
          answerKey(task, submission.answer) ===
            answerKey(task, task.consensus.answer)
        ) {
          const reputation = account.reputation[task.type]
          if (reputation) reputation.consensusMatches += 1
        }
      }
      return
    }
    task.consensus = undefined
    task.status =
      byCluster.size >= task.maxResponses ? 'escalated' : 'open'
  }

  listContributions(account: CommunityAccount): CommunityContributionView[] {
    return this.state.submissions
      .filter(
        (submission) =>
          submission.accountId === account.id ||
          account.roles.includes('governance-admin') ||
          account.roles.includes('owner'),
      )
      .map((submission) => {
        const task = this.state.tasks.find(
          (candidate) => candidate.id === submission.taskId,
        )
        const source = task
          ? this.state.sources.find((candidate) => candidate.id === task.sourceId)
          : undefined
        return {
          id: submission.id,
          taskId: submission.taskId,
          taskType: task?.type ?? 'event-presence',
          question: task?.question ?? '已删除任务',
          sourceTitle: source?.title ?? '已撤回来源',
          answer: submission.answer,
          confidence: submission.confidence,
          createdAt: submission.createdAt,
          withdrawnAt: submission.withdrawnAt,
          calibrationCorrect: submission.calibrationCorrect,
        }
      })
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
  }

  resolveTask(
    account: CommunityAccount,
    taskId: string,
    answerInput: unknown,
  ): CommunityTaskView {
    requiredRole(account, 'data-publisher')
    const task = this.state.tasks.find((candidate) => candidate.id === taskId)
    if (!task || task.status === 'withdrawn') {
      throw new CommunityRequestError('社区任务不存在', 404)
    }
    const answer = normalizeAnswer(task, answerInput)
    task.status = 'consensus'
    task.consensus = {
      answer,
      method: 'owner',
      supportingSubmissionIds: [],
      resolvedAt: this.now(),
    }
    this.persist()
    this.audit(account.id, 'task.resolved', 'task', task.id, { answer })
    return this.taskView(task)
  }

  withdrawSubmission(
    account: CommunityAccount,
    submissionId: string,
  ): void {
    const submission = this.state.submissions.find(
      (candidate) => candidate.id === submissionId,
    )
    if (!submission) throw new CommunityRequestError('贡献记录不存在', 404)
    if (
      submission.accountId !== account.id &&
      !account.roles.includes('governance-admin') &&
      !account.roles.includes('owner')
    ) {
      throw new CommunityRequestError('不能撤回其他用户的贡献', 403)
    }
    if (submission.withdrawnAt) return
    submission.withdrawnAt = this.now()
    const author = this.state.accounts.find(
      (candidate) => candidate.id === submission.accountId,
    )
    const task = this.state.tasks.find(
      (candidate) => candidate.id === submission.taskId,
    )
    if (task && author) {
      const reputation = author.reputation[task.type]
      if (reputation) {
        reputation.submissions = Math.max(0, reputation.submissions - 1)
        if (submission.calibrationCorrect !== undefined) {
          reputation.goldTotal = Math.max(0, reputation.goldTotal - 1)
          if (submission.calibrationCorrect) {
            reputation.goldCorrect = Math.max(0, reputation.goldCorrect - 1)
          }
        }
      }
    }
    if (task && !task.calibrationAnswer) this.recalculateConsensus(task)
    this.persist()
    this.audit(account.id, 'submission.withdrawn', 'submission', submission.id, {
      taskId: submission.taskId,
    })
  }

  withdrawSource(
    account: CommunityAccount,
    sourceId: string,
    reasonInput: unknown,
  ): void {
    requiredRole(account, 'governance-admin')
    const source = this.state.sources.find((candidate) => candidate.id === sourceId)
    if (!source) throw new CommunityRequestError('数据来源不存在', 404)
    const reason =
      typeof reasonInput === 'string' ? reasonInput.trim().slice(0, 200) : ''
    if (!reason) throw new CommunityRequestError('必须填写撤回原因')
    source.rightsStatus = 'withdrawn'
    source.withdrawnAt = this.now()
    source.withdrawalReason = reason
    const taskIds = new Set(
      this.state.tasks
        .filter((task) => task.sourceId === sourceId)
        .map((task) => {
          task.status = 'withdrawn'
          task.consensus = undefined
          return task.id
        }),
    )
    for (const release of this.state.releases) {
      if (
        release.status === 'active' &&
        release.taskIds.some((taskId) => taskIds.has(taskId))
      ) {
        release.status = 'blocked'
        release.blockedAt = this.now()
        release.blockedReason = `source-withdrawn:${sourceId}`
        for (const experiment of this.state.experiments) {
          if (
            experiment.datasetReleaseId === release.id &&
            (experiment.status === 'queued' || experiment.status === 'running')
          ) {
            experiment.status = 'cancelled'
            experiment.updatedAt = this.now()
            experiment.failureReason = '数据来源已撤回'
            experiment.lease = undefined
          }
        }
      }
    }
    this.persist()
    this.audit(account.id, 'source.withdrawn', 'source', sourceId, { reason })
  }

  listReleases(): DatasetRelease[] {
    return this.state.releases.map((release) => ({ ...release }))
  }

  releaseManifest(
    account: CommunityAccount,
    releaseId: string,
  ): DatasetReleaseManifest {
    requiredRole(account, 'model-maintainer')
    return this.buildReleaseManifest(releaseId)
  }

  private buildReleaseManifest(releaseId: string): DatasetReleaseManifest {
    const release = this.state.releases.find(
      (candidate) => candidate.id === releaseId,
    )
    if (!release) throw new CommunityRequestError('数据版本不存在', 404)
    if (release.status === 'blocked') {
      throw new CommunityRequestError('数据版本已被阻断', 409)
    }
    const records = release.taskIds.map((taskId) => {
      const task = this.state.tasks.find((candidate) => candidate.id === taskId)
      const source = task
        ? this.state.sources.find((candidate) => candidate.id === task.sourceId)
        : undefined
      if (!task?.consensus || !source || source.rightsStatus !== 'active') {
        throw new CommunityRequestError('数据版本内容不完整或来源已撤回', 409)
      }
      return {
        taskId: task.id,
        type: task.type,
        answer: task.consensus.answer,
        snapshotSha256: task.snapshotSha256,
        source: {
          id: source.id,
          dataset: source.dataset,
          license: source.license,
          performerId: source.performerId,
          workId: source.workId,
          recordingId: source.recordingId,
          sessionId: source.sessionId,
          lineageId: source.lineageId,
          reviewQueue: source.reviewQueue,
          reviewName: source.reviewName,
        },
      }
    })
    const compact = records
      .map((record) => ({
        taskId: record.taskId,
        sourceId: record.source.id,
        type: record.type,
        answer: record.answer,
        snapshotSha256: record.snapshotSha256,
      }))
      .sort((left, right) => left.taskId.localeCompare(right.taskId))
    if (digest(compact) !== release.manifestSha256) {
      throw new CommunityRequestError('数据版本清单哈希不匹配', 409)
    }
    return {
      schemaVersion: 1,
      release: {
        id: release.id,
        name: release.name,
        version: release.version,
        verification: release.verification,
        manifestSha256: release.manifestSha256,
        createdAt: release.createdAt,
      },
      records,
    }
  }

  runnerDatasetManifest(
    runner: TrainingRunner,
    experimentId: string,
    leaseToken: unknown,
  ): DatasetReleaseManifest {
    const experiment = this.requireActiveLease(
      runner,
      experimentId,
      leaseToken,
    )
    if (BUILTIN_DATASETS.has(experiment.datasetReleaseId)) {
      throw new CommunityRequestError('内置数据集不通过此接口分发', 409)
    }
    const manifest = this.buildReleaseManifest(experiment.datasetReleaseId)
    if (manifest.release.manifestSha256 !== experiment.datasetManifestSha256) {
      throw new CommunityRequestError('实验数据清单哈希不匹配', 409)
    }
    return manifest
  }

  createRelease(
    account: CommunityAccount,
    input: { name: unknown; taskIds?: unknown },
  ): DatasetRelease {
    requiredRole(account, 'data-publisher')
    const name = typeof input.name === 'string' ? input.name.trim() : ''
    if (!/^[\p{L}\p{N}][\p{L}\p{N} ._-]{1,63}$/u.test(name)) {
      throw new CommunityRequestError('数据版本名称无效')
    }
    const selectedIds = Array.isArray(input.taskIds)
      ? input.taskIds.filter((value): value is string => typeof value === 'string')
      : this.state.tasks
          .filter((task) => task.status === 'consensus')
          .map((task) => task.id)
    const selected = selectedIds.map((id) =>
      this.state.tasks.find((task) => task.id === id),
    )
    if (
      selected.length === 0 ||
      selected.some((task) => !task || task.status !== 'consensus' || !task.consensus)
    ) {
      throw new CommunityRequestError('数据版本只能包含已形成共识的任务')
    }
    const tasks = selected as CommunityTask[]
    const sourceIds = [...new Set(tasks.map((task) => task.sourceId))].sort()
    if (
      sourceIds.some(
        (id) =>
          this.state.sources.find((source) => source.id === id)?.rightsStatus !==
          'active',
      )
    ) {
      throw new CommunityRequestError('数据版本包含已撤回来源')
    }
    const version =
      Math.max(
        0,
        ...this.state.releases
          .filter((release) => release.name === name)
          .map((release) => release.version),
      ) + 1
    const manifest = tasks
      .map((task) => ({
        taskId: task.id,
        sourceId: task.sourceId,
        type: task.type,
        answer: task.consensus?.answer,
        snapshotSha256: task.snapshotSha256,
      }))
      .sort((left, right) => left.taskId.localeCompare(right.taskId))
    const release: DatasetRelease = {
      id: randomUUID(),
      name,
      version,
      status: 'active',
      verification: 'owner_verified',
      taskIds: manifest.map((entry) => entry.taskId),
      sourceIds,
      manifestSha256: digest(manifest),
      createdBy: account.id,
      createdAt: this.now(),
    }
    this.state.releases.push(release)
    this.persist()
    this.audit(account.id, 'dataset.released', 'dataset-release', release.id, {
      name,
      version,
      manifestSha256: release.manifestSha256,
    })
    return release
  }

  private experimentView(experiment: ExperimentRecord): ExperimentRun {
    const {
      lease,
      uploadedArtifact: _uploadedArtifact,
      artifactObjectKey: _artifactObjectKey,
      ...view
    } = experiment
    return {
      ...view,
      parameters: { ...experiment.parameters },
      attemptCount: experiment.attemptCount ?? 0,
      ...(experiment.metrics ? { metrics: { ...experiment.metrics } } : {}),
      ...(lease
        ? {
            lease: {
              runnerId: lease.runnerId,
              claimedAt: lease.claimedAt,
              heartbeatAt: lease.heartbeatAt,
              expiresAt: lease.expiresAt,
              attempt: lease.attempt,
            },
          }
        : {}),
    }
  }

  private requeueExpiredLeases(): void {
    const now = this.nowDate()
    const expired = this.state.experiments.filter(
      (experiment) =>
        experiment.status === 'running' &&
        experiment.lease &&
        new Date(experiment.lease.expiresAt).getTime() <= now.getTime(),
    )
    if (expired.length === 0) return
    for (const experiment of expired) {
      const previousLease = experiment.lease!
      experiment.status = 'queued'
      experiment.updatedAt = now.toISOString()
      experiment.lease = undefined
      experiment.uploadedArtifact = undefined
      this.audit(
        `runner:${previousLease.runnerId}`,
        'experiment.lease-expired',
        'experiment',
        experiment.id,
        {
          attempt: previousLease.attempt,
          expiredAt: previousLease.expiresAt,
        },
      )
    }
    this.persist()
  }

  private requireActiveLease(
    runner: TrainingRunner,
    experimentId: string,
    leaseToken: unknown,
  ): ExperimentRecord {
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (!experiment) throw new CommunityRequestError('实验不存在', 404)
    if (experiment.status !== 'running' || !experiment.lease) {
      throw new CommunityRequestError('实验没有有效执行租约', 409)
    }
    if (experiment.lease.runnerId !== runner.id) {
      throw new CommunityRequestError('实验租约不属于当前执行器', 403)
    }
    if (
      new Date(experiment.lease.expiresAt).getTime() <=
      this.nowDate().getTime()
    ) {
      this.requeueExpiredLeases()
      throw new CommunityRequestError('实验租约已过期并重新排队', 409)
    }
    if (
      typeof leaseToken !== 'string' ||
      !safeEqual(experiment.lease.tokenHash, hashSecret(leaseToken))
    ) {
      throw new CommunityRequestError('实验租约令牌无效', 403)
    }
    return experiment
  }

  getRunnerArtifactUpload(
    runner: TrainingRunner,
    experimentId: string,
    leaseToken: unknown,
  ): {
    experimentId: string
    attempt: number
    expectedMediaType: string
    maxBytes: number
  } {
    const experiment = this.requireActiveLease(
      runner,
      experimentId,
      leaseToken,
    )
    if (experiment.uploadedArtifact) {
      throw new CommunityRequestError('当前租约已上传训练制品', 409)
    }
    return {
      experimentId: experiment.id,
      attempt: experiment.lease!.attempt,
      expectedMediaType: 'application/octet-stream',
      maxBytes: 512 * 1024 * 1024,
    }
  }

  registerRunnerArtifact(
    runner: TrainingRunner,
    experimentId: string,
    leaseToken: unknown,
    artifact: {
      sha256: string
      sizeBytes: number
      mediaType: string
      objectKey: string
    },
  ): ExperimentRun {
    const experiment = this.requireActiveLease(
      runner,
      experimentId,
      leaseToken,
    )
    if (
      !SHA256.test(artifact.sha256) ||
      !Number.isInteger(artifact.sizeBytes) ||
      artifact.sizeBytes <= 0 ||
      artifact.sizeBytes > 512 * 1024 * 1024 ||
      artifact.mediaType !== 'application/octet-stream' ||
      !/^[a-z0-9][a-z0-9/._-]{1,255}$/.test(artifact.objectKey)
    ) {
      throw new CommunityRequestError('训练制品元数据无效')
    }
    experiment.uploadedArtifact = {
      runnerId: runner.id,
      attempt: experiment.lease!.attempt,
      ...artifact,
      uploadedAt: this.now(),
    }
    this.persist()
    this.audit(
      `runner:${runner.id}`,
      'experiment.artifact-uploaded',
      'experiment',
      experiment.id,
      {
        attempt: experiment.lease!.attempt,
        sha256: artifact.sha256,
        sizeBytes: artifact.sizeBytes,
        objectKey: artifact.objectKey,
      },
    )
    return this.experimentView(experiment)
  }

  claimExperiment(runner: TrainingRunner): TrainingRunnerClaim | null {
    this.requeueExpiredLeases()
    const experiment = this.state.experiments.find((candidate) => {
      if (candidate.status !== 'queued') return false
      const recipe = RECIPES.find((item) => item.id === candidate.recipeId)
      if (!recipe || recipe.executor !== 'external-worker') return false
      if (
        !recipe.compatibleDatasets.includes('*') &&
        !recipe.compatibleDatasets.includes(candidate.datasetReleaseId)
      ) {
        return false
      }
      if (BUILTIN_DATASETS.has(candidate.datasetReleaseId)) return true
      return this.state.releases.some(
        (release) =>
          release.id === candidate.datasetReleaseId &&
          release.status === 'active',
      )
    })
    if (!experiment) return null
    const recipe = RECIPES.find(
      (candidate) => candidate.id === experiment.recipeId,
    )!
    const leaseToken = randomBytes(32).toString('base64url')
    const claimedAt = this.nowDate()
    const leaseSeconds = this.runnerLeaseSeconds()
    const attempt = (experiment.attemptCount ?? 0) + 1
    const expiresAt = new Date(
      claimedAt.getTime() + leaseSeconds * 1000,
    ).toISOString()
    experiment.status = 'running'
    experiment.updatedAt = claimedAt.toISOString()
    experiment.attemptCount = attempt
    experiment.lastRunnerId = runner.id
    experiment.failureReason = undefined
    experiment.uploadedArtifact = undefined
    experiment.lease = {
      runnerId: runner.id,
      tokenHash: hashSecret(leaseToken),
      claimedAt: claimedAt.toISOString(),
      heartbeatAt: claimedAt.toISOString(),
      expiresAt,
      attempt,
    }
    this.persist()
    this.audit(
      `runner:${runner.id}`,
      'experiment.claimed',
      'experiment',
      experiment.id,
      { attempt, expiresAt },
    )
    return {
      schemaVersion: 1,
      leaseToken,
      heartbeatIntervalSeconds: Math.max(10, Math.floor(leaseSeconds / 3)),
      leaseExpiresAt: expiresAt,
      job: {
        experimentId: experiment.id,
        recipe: {
          id: recipe.id,
          specVersion: recipe.specVersion,
          modelFamily: recipe.modelFamily,
          resourceClass: recipe.resourceClass,
          timeoutSeconds: recipe.timeoutSeconds,
          promotionEligible: recipe.promotionEligible,
        },
        dataset: {
          id: experiment.datasetReleaseId,
          kind: BUILTIN_DATASETS.has(experiment.datasetReleaseId)
            ? 'builtin'
            : 'community-release',
          manifestSha256: experiment.datasetManifestSha256,
        },
        parameters: { ...experiment.parameters },
        codeRevision: experiment.codeRevision,
        runtimeImage: experiment.runtimeImage,
        lineageSha256: experiment.lineageSha256,
      },
    }
  }

  heartbeatExperiment(
    runner: TrainingRunner,
    experimentId: string,
    leaseToken: unknown,
  ): ExperimentRun {
    const experiment = this.requireActiveLease(
      runner,
      experimentId,
      leaseToken,
    )
    const heartbeatAt = this.nowDate()
    experiment.lease!.heartbeatAt = heartbeatAt.toISOString()
    experiment.lease!.expiresAt = new Date(
      heartbeatAt.getTime() + this.runnerLeaseSeconds() * 1000,
    ).toISOString()
    experiment.updatedAt = heartbeatAt.toISOString()
    this.persist()
    this.audit(
      `runner:${runner.id}`,
      'experiment.heartbeat',
      'experiment',
      experiment.id,
      {
        attempt: experiment.lease!.attempt,
        expiresAt: experiment.lease!.expiresAt,
      },
    )
    return this.experimentView(experiment)
  }

  recordRunnerResult(
    runner: TrainingRunner,
    experimentId: string,
    input: {
      leaseToken: unknown
      status: unknown
      metrics?: unknown
      artifactSha256?: unknown
      executionEvidence?: unknown
      failureReason?: unknown
    },
  ): ExperimentRun {
    const experiment = this.requireActiveLease(
      runner,
      experimentId,
      input.leaseToken,
    )
    if (input.status !== 'completed' && input.status !== 'failed') {
      throw new CommunityRequestError('执行器结果状态无效')
    }
    let metrics: Record<string, number> | undefined
    let artifactSha256: string | undefined
    let executionEvidence: ExperimentExecutionEvidence | undefined
    let failureReason: string | undefined
    if (input.status === 'completed') {
      metrics = normalizeExperimentMetrics(input.metrics)
      if (
        typeof input.artifactSha256 !== 'string' ||
        !SHA256.test(input.artifactSha256)
      ) {
        throw new CommunityRequestError('模型制品 SHA-256 无效')
      }
      artifactSha256 = input.artifactSha256
      if (
        !experiment.uploadedArtifact ||
        experiment.uploadedArtifact.runnerId !== runner.id ||
        experiment.uploadedArtifact.attempt !== experiment.lease!.attempt ||
        experiment.uploadedArtifact.sha256 !== artifactSha256
      ) {
        throw new CommunityRequestError(
          '模型制品尚未由当前租约上传并校验',
          409,
        )
      }
      executionEvidence = normalizeExecutionEvidence(input.executionEvidence)
    } else {
      const detail =
        typeof input.failureReason === 'string'
          ? input.failureReason.trim().slice(0, 300)
          : ''
      failureReason = detail || '执行器报告失败'
    }
    if (input.status === 'completed') {
      experiment.metrics = metrics
      experiment.artifactSha256 = artifactSha256
      experiment.artifactSizeBytes = experiment.uploadedArtifact!.sizeBytes
      experiment.artifactMediaType = experiment.uploadedArtifact!.mediaType
      experiment.artifactObjectKey = experiment.uploadedArtifact!.objectKey
      experiment.executionEvidence = executionEvidence
      experiment.modelVersion = `${experiment.recipeId}-${experiment.id.slice(0, 8)}`
      experiment.failureReason = undefined
    } else {
      experiment.failureReason = failureReason
    }
    const leaseAttempt = experiment.lease!.attempt
    experiment.status = input.status
    experiment.updatedAt = this.now()
    experiment.lease = undefined
    experiment.uploadedArtifact = undefined
    this.persist()
    this.audit(
      `runner:${runner.id}`,
      'experiment.result-recorded',
      'experiment',
      experiment.id,
      {
        status: experiment.status,
        attempt: leaseAttempt,
        ...(experiment.artifactSha256
          ? { artifactSha256: experiment.artifactSha256 }
          : {}),
      },
    )
    return this.experimentView(experiment)
  }

  listExperiments(account: CommunityAccount): ExperimentRun[] {
    this.requeueExpiredLeases()
    if (account.roles.includes('owner') || account.roles.includes('model-maintainer')) {
      return this.state.experiments.map((experiment) =>
        this.experimentView(experiment),
      )
    }
    return this.state.experiments
      .filter((experiment) => experiment.submittedBy === account.id)
      .map((experiment) => this.experimentView(experiment))
  }

  experimentArtifactObjectKey(
    account: CommunityAccount,
    experimentId: string,
  ): string {
    requiredRole(account, 'model-maintainer')
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (
      !experiment ||
      experiment.status !== 'completed' ||
      !experiment.artifactObjectKey
    ) {
      throw new CommunityRequestError('实验制品不存在', 404)
    }
    return experiment.artifactObjectKey
  }

  promotionGateArtifactObjectKey(
    _agent: PromotionGateAgent,
    experimentId: string,
  ): string {
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (
      !experiment ||
      experiment.status !== 'completed' ||
      !experiment.artifactObjectKey ||
      !this.state.promotions.some(
        (promotion) => promotion.experimentId === experiment.id,
      )
    ) {
      throw new CommunityRequestError('门禁实验制品不存在', 404)
    }
    return experiment.artifactObjectKey
  }

  createExperiment(
    account: CommunityAccount,
    input: {
      recipeId: unknown
      datasetReleaseId: unknown
      parameters?: unknown
    },
  ): ExperimentRun {
    requiredRole(account, 'experimenter')
    const recipe = RECIPES.find((candidate) => candidate.id === input.recipeId)
    if (!recipe) throw new CommunityRequestError('训练配方不存在', 404)
    const datasetReleaseId =
      typeof input.datasetReleaseId === 'string'
        ? input.datasetReleaseId
        : ''
    const release = this.state.releases.find(
      (candidate) => candidate.id === datasetReleaseId,
    )
    if (!BUILTIN_DATASETS.has(datasetReleaseId) && !release) {
      throw new CommunityRequestError('数据版本不存在', 404)
    }
    if (
      BUILTIN_DATASETS.has(datasetReleaseId) &&
      !this.options.builtinDatasetManifestSha256?.[datasetReleaseId]
    ) {
      throw new CommunityRequestError('内置训练数据未部署到当前环境', 409)
    }
    if (recipe.availability !== 'ready') {
      throw new CommunityRequestError('训练配方尚未部署执行规范', 409)
    }
    if (
      !recipe.compatibleDatasets.includes('*') &&
      !recipe.compatibleDatasets.includes(datasetReleaseId)
    ) {
      throw new CommunityRequestError('训练配方与所选数据版本不兼容', 409)
    }
    if (release?.status === 'blocked') {
      throw new CommunityRequestError('数据版本已因来源撤回而停用', 409)
    }
    if (recipe.executor === 'external-worker') {
      const activeCount = this.state.experiments.filter(
        (experiment) =>
          experiment.submittedBy === account.id &&
          (experiment.status === 'queued' || experiment.status === 'running'),
      ).length
      if (activeCount >= MAX_ACTIVE_EXPERIMENTS_PER_ACCOUNT) {
        throw new CommunityRequestError('当前账号的并发实验已达上限', 429)
      }
      const today = this.now().slice(0, 10)
      const dailyCount = this.state.experiments.filter(
        (experiment) =>
          experiment.submittedBy === account.id &&
          experiment.createdAt.slice(0, 10) === today,
      ).length
      if (dailyCount >= MAX_DAILY_EXPERIMENTS_PER_ACCOUNT) {
        throw new CommunityRequestError('当前账号的每日实验配额已用完', 429)
      }
    }
    const rawParameters = isRecord(input.parameters) ? input.parameters : {}
    const unexpected = Object.keys(rawParameters).filter(
      (key) => !(key in recipe.parameters),
    )
    if (unexpected.length > 0) {
      throw new CommunityRequestError(`配方不允许参数：${unexpected.join(', ')}`)
    }
    const parameters = Object.fromEntries(
      Object.entries(recipe.parameters).map(([key, rule]) => [
        key,
        finiteInput(
          rawParameters[key] ?? rule.default,
          key,
          rule.minimum,
          rule.maximum,
          rule.type === 'integer',
        ),
      ]),
    )
    const createdAt = this.now()
    const datasetManifestSha256 =
      release?.manifestSha256 ??
      this.options.builtinDatasetManifestSha256?.[datasetReleaseId] ??
      digest(datasetReleaseId)
    const codeRevision = process.env.GTS_CODE_REVISION ?? 'workspace'
    const runtimeImage = process.env.GTS_TRAINING_IMAGE ?? 'local-development'
    const experiment: ExperimentRecord = {
      id: randomUUID(),
      recipeId: recipe.id,
      datasetReleaseId,
      submittedBy: account.id,
      parameters,
      status: recipe.executor === 'builtin-audit' ? 'completed' : 'queued',
      createdAt,
      updatedAt: createdAt,
      codeRevision,
      runtimeImage,
      datasetManifestSha256,
      lineageSha256: digest({
        recipeId: recipe.id,
        datasetReleaseId,
        datasetManifestSha256,
        codeRevision,
        runtimeImage,
        parameters,
      }),
      attemptCount: 0,
      ...(recipe.executor === 'builtin-audit'
        ? {
            metrics: {
              manifestIntegrity: 1,
              rightsCoverage: 1,
              taskCount: release?.taskIds.length ?? 0,
            },
            artifactSha256: digest({
              kind: 'dataset-audit',
              release: release?.manifestSha256 ?? datasetReleaseId,
            }),
          }
        : {}),
    }
    this.state.experiments.push(experiment)
    this.persist()
    this.audit(account.id, 'experiment.submitted', 'experiment', experiment.id, {
      recipeId: recipe.id,
      datasetReleaseId,
      parameters,
    })
    return this.experimentView(experiment)
  }

  reproduceExperiment(
    account: CommunityAccount,
    experimentId: string,
  ): ExperimentRun {
    requiredRole(account, 'model-maintainer')
    const source = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (
      !source ||
      source.status !== 'completed' ||
      !source.artifactSha256 ||
      !source.executionEvidence
    ) {
      throw new CommunityRequestError('只有已完成实验可以发起复现', 409)
    }
    const existing = this.state.experiments.find(
      (candidate) =>
        candidate.id !== source.id &&
        candidate.lineageSha256 === source.lineageSha256 &&
        (candidate.status === 'queued' ||
          candidate.status === 'running' ||
          candidate.status === 'completed'),
    )
    if (existing) return this.experimentView(existing)
    const createdAt = this.now()
    const reproduction: ExperimentRecord = {
      id: randomUUID(),
      recipeId: source.recipeId,
      datasetReleaseId: source.datasetReleaseId,
      submittedBy: account.id,
      parameters: { ...source.parameters },
      status: 'queued',
      createdAt,
      updatedAt: createdAt,
      codeRevision: source.codeRevision,
      runtimeImage: source.runtimeImage,
      datasetManifestSha256: source.datasetManifestSha256,
      lineageSha256: source.lineageSha256,
      attemptCount: 0,
    }
    this.state.experiments.push(reproduction)
    this.persist()
    this.audit(
      account.id,
      'experiment.reproduction-requested',
      'experiment',
      reproduction.id,
      { sourceExperimentId: source.id, lineageSha256: source.lineageSha256 },
    )
    return this.experimentView(reproduction)
  }

  retryExperiment(
    account: CommunityAccount,
    experimentId: string,
  ): ExperimentRun {
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (!experiment) throw new CommunityRequestError('实验不存在', 404)
    if (
      experiment.submittedBy !== account.id &&
      !account.roles.includes('model-maintainer') &&
      !account.roles.includes('owner')
    ) {
      throw new CommunityRequestError('不能重试其他账号的实验', 403)
    }
    if (experiment.status !== 'failed') {
      throw new CommunityRequestError('只有失败实验可以重试', 409)
    }
    if (experiment.attemptCount >= MAX_EXPERIMENT_ATTEMPTS) {
      throw new CommunityRequestError('实验已达到最大尝试次数', 409)
    }
    const recipe = RECIPES.find(
      (candidate) => candidate.id === experiment.recipeId,
    )
    if (!recipe || recipe.availability !== 'ready') {
      throw new CommunityRequestError('实验配方当前不可执行', 409)
    }
    const release = this.state.releases.find(
      (candidate) => candidate.id === experiment.datasetReleaseId,
    )
    if (release?.status === 'blocked') {
      throw new CommunityRequestError('实验数据版本已被阻断', 409)
    }
    experiment.status = 'queued'
    experiment.updatedAt = this.now()
    experiment.failureReason = undefined
    experiment.lease = undefined
    experiment.uploadedArtifact = undefined
    experiment.metrics = undefined
    experiment.artifactSha256 = undefined
    experiment.artifactSizeBytes = undefined
    experiment.artifactMediaType = undefined
    experiment.artifactObjectKey = undefined
    experiment.executionEvidence = undefined
    experiment.modelVersion = undefined
    this.persist()
    this.audit(account.id, 'experiment.retried', 'experiment', experiment.id, {
      attemptCount: experiment.attemptCount,
    })
    return this.experimentView(experiment)
  }

  recordExperiment(
    account: CommunityAccount,
    experimentId: string,
    input: {
      status: unknown
      metrics?: unknown
      artifactSha256?: unknown
      failureReason?: unknown
    },
  ): ExperimentRun {
    requiredRole(account, 'model-maintainer')
    this.requeueExpiredLeases()
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (!experiment) throw new CommunityRequestError('实验不存在', 404)
    if (
      experiment.status === 'completed' ||
      experiment.status === 'failed' ||
      experiment.status === 'cancelled'
    ) {
      throw new CommunityRequestError('终态实验不能再次修改', 409)
    }
    if (experiment.lease) {
      throw new CommunityRequestError('实验由外部执行器租约持有', 409)
    }
    if (input.status !== 'failed') {
      throw new CommunityRequestError('外部实验只能由租约执行器登记运行结果', 409)
    }
    experiment.failureReason =
      typeof input.failureReason === 'string'
        ? input.failureReason.trim().slice(0, 300)
        : '维护者终止实验'
    experiment.status = 'failed'
    experiment.updatedAt = this.now()
    this.persist()
    this.audit(account.id, 'experiment.updated', 'experiment', experiment.id, {
      status: experiment.status,
      artifactSha256: experiment.artifactSha256,
    })
    return this.experimentView(experiment)
  }

  listPromotions(account: CommunityAccount): ModelPromotion[] {
    requiredRole(account, 'model-maintainer')
    return this.state.promotions.map((promotion) => ({ ...promotion }))
  }

  modelCard(promotionId: string): ModelCard {
    const promotion = this.state.promotions.find(
      (candidate) => candidate.id === promotionId,
    )
    if (!promotion) throw new CommunityRequestError('模型晋级记录不存在', 404)
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === promotion.experimentId,
    )
    const recipe = experiment
      ? RECIPES.find((candidate) => candidate.id === experiment.recipeId)
      : undefined
    if (
      !experiment ||
      !recipe ||
      !experiment.artifactSha256 ||
      !experiment.artifactSizeBytes ||
      !experiment.artifactMediaType
    ) {
      throw new CommunityRequestError('模型卡所需实验信息不完整', 409)
    }
    const limitations = [
      '候选模型不得绕过门禁替换默认推理后端。',
    ]
    if (promotion.status !== 'champion') {
      limitations.push('当前模型尚未获得生产 champion 状态。')
    }
    if (!promotion.checks['sealed-evaluation']?.passed) {
      limitations.push('尚未通过不少于 100 首全新歌曲的密封评测。')
    }
    return {
      schemaVersion: 1,
      model: {
        version: promotion.modelVersion,
        family: recipe.modelFamily,
        status: promotion.status,
        createdAt: promotion.createdAt,
      },
      training: {
        recipeId: experiment.recipeId,
        datasetReleaseId: experiment.datasetReleaseId,
        datasetManifestSha256: experiment.datasetManifestSha256,
        codeRevision: experiment.codeRevision,
        runtimeImage: experiment.runtimeImage,
        parameters: { ...experiment.parameters },
        lineageSha256: experiment.lineageSha256,
      },
      artifact: {
        sha256: experiment.artifactSha256,
        sizeBytes: experiment.artifactSizeBytes,
        mediaType: experiment.artifactMediaType,
      },
      metrics: { ...(experiment.metrics ?? {}) },
      checks: { ...promotion.checks },
      limitations,
    }
  }

  listPromotionGateJobs(
    _agent: PromotionGateAgent,
  ): PromotionGateJob[] {
    return this.state.promotions
      .filter(
        (promotion) =>
          promotion.status === 'candidate' ||
          promotion.status === 'shadow' ||
          promotion.status === 'canary',
      )
      .map((promotion) => {
        const experiment = this.state.experiments.find(
          (candidate) => candidate.id === promotion.experimentId,
        )
        if (!experiment) {
          throw new CommunityRequestError('模型晋级实验不存在', 500)
        }
        const reproductions = this.state.experiments
          .filter(
            (candidate) =>
              candidate.id !== experiment.id &&
              candidate.status === 'completed' &&
              candidate.lineageSha256 === experiment.lineageSha256 &&
              candidate.artifactSha256 &&
              candidate.metrics,
          )
          .map((candidate) => ({
            experimentId: candidate.id,
            artifactSha256: candidate.artifactSha256!,
            metrics: { ...candidate.metrics! },
          }))
        return {
          promotion: {
            ...promotion,
            checks: { ...promotion.checks },
            approvals: [...promotion.approvals],
          },
          experiment: this.experimentView(experiment),
          reproductions,
        }
      })
  }

  listSealedPromotionGateJobs(
    agent: PromotionGateAgent,
  ): PromotionGateJob[] {
    if (agent.scope !== 'sealed') {
      throw new CommunityRequestError('密封评测服务权限无效', 403)
    }
    return this.listPromotionGateJobs(agent).filter((job) => {
      const checks = job.promotion.checks
      return (
        checks.lineage?.passed === true &&
        checks.reproduction?.passed === true &&
        checks['public-validation']?.passed === true &&
        checks.robustness?.passed === true &&
        checks['sealed-evaluation'] === undefined
      )
    })
  }

  createPromotion(
    account: CommunityAccount,
    experimentId: string,
  ): ModelPromotion {
    requiredRole(account, 'model-maintainer')
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === experimentId,
    )
    if (
      !experiment ||
      experiment.status !== 'completed' ||
      !experiment.artifactSha256 ||
      !experiment.artifactObjectKey ||
      !experiment.artifactSizeBytes ||
      !experiment.executionEvidence ||
      !experiment.modelVersion
    ) {
      throw new CommunityRequestError('只有已完成且带模型制品的实验可以提名', 409)
    }
    const recipe = RECIPES.find(
      (candidate) => candidate.id === experiment.recipeId,
    )
    if (!recipe?.promotionEligible) {
      throw new CommunityRequestError('该配方的制品不具备模型晋级资格', 409)
    }
    const existing = this.state.promotions.find(
      (promotion) => promotion.experimentId === experimentId,
    )
    if (existing) return { ...existing }
    const promotion: ModelPromotion = {
      id: randomUUID(),
      experimentId,
      modelVersion: experiment.modelVersion,
      submittedBy: account.id,
      status: 'candidate',
      checks: {
        lineage: {
          passed: true,
          summary: '数据、代码、运行环境、执行证据与制品哈希完整',
          recordedAt: this.now(),
          recordedBy: 'system:lineage',
          source: 'system',
          evidenceSha256: digest({
            lineageSha256: experiment.lineageSha256,
            datasetManifestSha256: experiment.datasetManifestSha256,
            codeRevision: experiment.codeRevision,
            runtimeImage: experiment.runtimeImage,
            artifactSha256: experiment.artifactSha256,
            artifactSizeBytes: experiment.artifactSizeBytes,
            executionEvidence: experiment.executionEvidence,
          }),
        },
      },
      approvals: [],
      createdAt: this.now(),
      updatedAt: this.now(),
    }
    this.state.promotions.push(promotion)
    this.persist()
    this.audit(account.id, 'promotion.created', 'promotion', promotion.id, {
      experimentId,
      modelVersion: promotion.modelVersion,
    })
    return promotion
  }

  recordPromotionCheck(
    account: CommunityAccount,
    promotionId: string,
    input: { check: unknown; passed: unknown; summary: unknown },
  ): ModelPromotion {
    requiredRole(account, 'model-maintainer')
    const promotion = this.state.promotions.find(
      (candidate) => candidate.id === promotionId,
    )
    if (!promotion) throw new CommunityRequestError('模型晋级记录不存在', 404)
    const checks: PromotionCheck[] = ['shadow', 'canary']
    if (!checks.includes(input.check as PromotionCheck)) {
      throw new CommunityRequestError('该门禁只能由独立门禁服务记录', 403)
    }
    if (typeof input.passed !== 'boolean') {
      throw new CommunityRequestError('门禁结果必须为布尔值')
    }
    const summary =
      typeof input.summary === 'string' ? input.summary.trim().slice(0, 300) : ''
    if (!summary) throw new CommunityRequestError('必须提供门禁摘要')
    promotion.checks[input.check as PromotionCheck] = {
      passed: input.passed,
      summary,
      recordedAt: this.now(),
      recordedBy: account.id,
      source: 'maintainer',
    }
    if (!input.passed) promotion.status = 'rejected'
    promotion.updatedAt = this.now()
    this.persist()
    this.audit(account.id, 'promotion.check-recorded', 'promotion', promotion.id, {
      check: input.check,
      passed: input.passed,
      summary,
    })
    return { ...promotion }
  }

  recordAutomatedPromotionCheck(
    agent: PromotionGateAgent,
    promotionId: string,
    input: {
      check: unknown
      passed: unknown
      summary: unknown
      evidenceSha256: unknown
    },
  ): ModelPromotion {
    const promotion = this.state.promotions.find(
      (candidate) => candidate.id === promotionId,
    )
    if (!promotion) throw new CommunityRequestError('模型晋级记录不存在', 404)
    const checks: PromotionCheck[] =
      agent.scope === 'sealed'
        ? ['sealed-evaluation']
        : ['reproduction', 'public-validation', 'robustness']
    if (!checks.includes(input.check as PromotionCheck)) {
      throw new CommunityRequestError('自动门禁检查项无效')
    }
    if (typeof input.passed !== 'boolean') {
      throw new CommunityRequestError('门禁结果必须为布尔值')
    }
    const summary =
      typeof input.summary === 'string' ? input.summary.trim().slice(0, 300) : ''
    if (!summary) throw new CommunityRequestError('必须提供门禁摘要')
    if (
      typeof input.evidenceSha256 !== 'string' ||
      !SHA256.test(input.evidenceSha256)
    ) {
      throw new CommunityRequestError('门禁证据 SHA-256 无效')
    }
    promotion.checks[input.check as PromotionCheck] = {
      passed: input.passed,
      summary,
      recordedAt: this.now(),
      recordedBy: `gate:${agent.id}`,
      source: 'gate-service',
      evidenceSha256: input.evidenceSha256,
    }
    if (!input.passed) promotion.status = 'rejected'
    promotion.updatedAt = this.now()
    this.persist()
    this.audit(
      `gate:${agent.id}`,
      'promotion.automated-check-recorded',
      'promotion',
      promotion.id,
      {
        check: input.check,
        passed: input.passed,
        evidenceSha256: input.evidenceSha256,
      },
    )
    return { ...promotion }
  }

  advancePromotion(
    account: CommunityAccount,
    promotionId: string,
    target: unknown,
  ): ModelPromotion {
    requiredRole(account, 'model-maintainer')
    const promotion = this.state.promotions.find(
      (candidate) => candidate.id === promotionId,
    )
    if (!promotion) throw new CommunityRequestError('模型晋级记录不存在', 404)
    if (promotion.status === 'rejected' || promotion.status === 'retired') {
      throw new CommunityRequestError('当前模型不能继续晋级', 409)
    }
    if (target !== 'shadow' && target !== 'canary' && target !== 'champion') {
      throw new CommunityRequestError('目标阶段无效')
    }
    const expectedTarget =
      promotion.status === 'candidate'
        ? 'shadow'
        : promotion.status === 'shadow'
          ? 'canary'
          : promotion.status === 'canary'
            ? 'champion'
            : null
    if (target !== expectedTarget) {
      throw new CommunityRequestError('模型必须按 candidate、shadow、canary 顺序晋级')
    }
    const requiredChecks: PromotionCheck[] =
      target === 'shadow'
        ? [
            'lineage',
            'reproduction',
            'public-validation',
            'sealed-evaluation',
            'robustness',
          ]
        : target === 'canary'
          ? ['shadow']
          : ['canary']
    if (requiredChecks.some((check) => promotion.checks[check]?.passed !== true)) {
      throw new CommunityRequestError('模型尚未通过当前阶段的全部门禁', 409)
    }
    if (
      !promotion.approvals.some(
        (approval) => approval.target === target && approval.accountId === account.id,
      )
    ) {
      promotion.approvals.push({
        target,
        accountId: account.id,
        approvedAt: this.now(),
      })
    }
    const targetApprovals = promotion.approvals.filter(
      (approval) => approval.target === target,
    )
    const requiredApprovals =
      this.state.governanceMode === 'single-maintainer' ? 1 : 2
    if (new Set(targetApprovals.map((approval) => approval.accountId)).size >= requiredApprovals) {
      if (target === 'champion') {
        const previous = this.state.promotions.find(
          (candidate) => candidate.id === this.state.championPromotionId,
        )
        if (previous) {
          previous.status = 'retired'
          previous.updatedAt = this.now()
          promotion.previousChampionId = previous.id
        }
        this.state.championPromotionId = promotion.id
      }
      promotion.status = target
      promotion.updatedAt = this.now()
    }
    this.persist()
    this.audit(account.id, 'promotion.advanced', 'promotion', promotion.id, {
      target,
      approvalCount: targetApprovals.length,
      governanceMode: this.state.governanceMode,
    })
    return { ...promotion }
  }

  rollbackChampion(account: CommunityAccount, reasonInput: unknown): ModelPromotion {
    requiredRole(account, 'model-maintainer')
    const current = this.state.promotions.find(
      (promotion) => promotion.id === this.state.championPromotionId,
    )
    if (!current) throw new CommunityRequestError('当前没有可回滚的生产模型', 409)
    const reason =
      typeof reasonInput === 'string' ? reasonInput.trim().slice(0, 300) : ''
    if (!reason) throw new CommunityRequestError('必须填写回滚原因')
    current.status = 'retired'
    current.updatedAt = this.now()
    const previous = this.state.promotions.find(
      (promotion) => promotion.id === current.previousChampionId,
    )
    if (previous) {
      previous.status = 'champion'
      previous.updatedAt = this.now()
      this.state.championPromotionId = previous.id
    } else {
      this.state.championPromotionId = undefined
    }
    this.persist()
    this.audit(account.id, 'promotion.rolled-back', 'promotion', current.id, {
      reason,
      restoredPromotionId: previous?.id,
    })
    return { ...current }
  }

  configureInferenceDeployment(
    account: CommunityAccount,
    input: {
      shadowSamplePercent?: unknown
      canaryTrafficPercent?: unknown
      errorBudgetPercent?: unknown
      minimumObservations?: unknown
    },
  ): InferenceDeploymentStatus {
    requiredRole(account, 'model-maintainer')
    const current = this.state.deploymentConfig
    this.state.deploymentConfig = {
      shadowSamplePercent: finiteInput(
        input.shadowSamplePercent ?? current.shadowSamplePercent,
        'shadowSamplePercent',
        0,
        100,
        false,
      ),
      canaryTrafficPercent: finiteInput(
        input.canaryTrafficPercent ?? current.canaryTrafficPercent,
        'canaryTrafficPercent',
        0,
        50,
        false,
      ),
      errorBudgetPercent: finiteInput(
        input.errorBudgetPercent ?? current.errorBudgetPercent,
        'errorBudgetPercent',
        0,
        50,
        false,
      ),
      minimumObservations: finiteInput(
        input.minimumObservations ?? current.minimumObservations,
        'minimumObservations',
        1,
        100,
        true,
      ),
    }
    this.persist()
    this.audit(
      account.id,
      'deployment.configured',
      'system',
      'inference',
      this.state.deploymentConfig,
    )
    return this.inferenceDeploymentStatus()
  }

  private promotionArtifactPath(
    promotion: ModelPromotion | undefined,
  ): string | undefined {
    if (!promotion) return undefined
    const experiment = this.state.experiments.find(
      (candidate) => candidate.id === promotion.experimentId,
    )
    if (!experiment?.artifactObjectKey) return undefined
    const artifactPath = resolve(this.root, experiment.artifactObjectKey)
    const relativePath = relative(this.root, artifactPath)
    if (relativePath.startsWith('..') || relativePath === '') return undefined
    return existsSync(artifactPath) ? artifactPath : undefined
  }

  inferenceDeployment(jobId: string): InferenceDeploymentPolicy {
    const champion = this.state.promotions.find(
      (promotion) =>
        promotion.id === this.state.championPromotionId &&
        promotion.status === 'champion',
    )
    const championArtifact = this.promotionArtifactPath(champion)
    const previousChampion = this.state.promotions.find(
      (promotion) => promotion.id === champion?.previousChampionId,
    )
    const previousChampionArtifact =
      this.promotionArtifactPath(previousChampion)
    const active = [...this.state.promotions]
      .filter(
        (promotion) =>
          promotion.status === 'canary' || promotion.status === 'shadow',
      )
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0]
    if (active) {
      const artifactPath = this.promotionArtifactPath(active)
      const samplePercent =
        active.status === 'canary'
          ? this.state.deploymentConfig.canaryTrafficPercent
          : this.state.deploymentConfig.shadowSamplePercent
      const selected =
        Boolean(artifactPath) &&
        percentageBucket(`${active.id}:${jobId}`) < samplePercent
      if (selected) {
        return {
          mode: active.status as 'shadow' | 'canary',
          promotionId: active.id,
          modelVersion: active.modelVersion,
          artifactPath,
          baselineModelVersion: champion?.modelVersion,
          baselineArtifactPath: championArtifact,
          baselineFallbackModelVersion: previousChampion?.modelVersion,
          baselineFallbackArtifactPath: previousChampionArtifact,
        }
      }
    }
    if (champion && championArtifact) {
      return {
        mode: 'champion',
        promotionId: champion.id,
        modelVersion: champion.modelVersion,
        artifactPath: championArtifact,
        baselineModelVersion: previousChampion?.modelVersion,
        baselineArtifactPath: previousChampionArtifact,
      }
    }
    return { mode: 'baseline' }
  }

  recordInferenceObservation(
    input: Omit<InferenceObservation, 'id' | 'recordedAt'>,
  ): InferenceDeploymentStatus {
    const promotion = this.state.promotions.find(
      (candidate) => candidate.id === input.promotionId,
    )
    if (!promotion) throw new CommunityRequestError('推理观测对应模型不存在', 404)
    if (
      input.mode !== 'shadow' &&
      input.mode !== 'canary' &&
      input.mode !== 'champion'
    ) {
      throw new CommunityRequestError('推理观测模式无效')
    }
    if (
      this.state.inferenceObservations.some(
        (observation) =>
          observation.promotionId === input.promotionId &&
          observation.jobId === input.jobId &&
          observation.mode === input.mode,
      )
    ) {
      return this.inferenceDeploymentStatus()
    }
    const observation: InferenceObservation = {
      ...input,
      durationMs: finiteInput(
        input.durationMs,
        'durationMs',
        0,
        86_400_000,
        true,
      ),
      error:
        typeof input.error === 'string'
          ? input.error.trim().slice(0, 300)
          : undefined,
      id: randomUUID(),
      recordedAt: this.now(),
    }
    this.state.inferenceObservations.push(observation)
    this.state.inferenceObservations =
      this.state.inferenceObservations.slice(-2000)

    const recent = this.state.inferenceObservations
      .filter(
        (candidate) =>
          candidate.promotionId === promotion.id &&
          candidate.mode === input.mode,
      )
      .slice(-100)
    const failures = recent.filter(
      (candidate) => !candidate.success || candidate.fallbackUsed,
    ).length
    const errorRate = recent.length === 0 ? 0 : (failures / recent.length) * 100
    const overBudget =
      recent.length >= this.state.deploymentConfig.minimumObservations &&
      errorRate > this.state.deploymentConfig.errorBudgetPercent
    if (
      overBudget &&
      promotion.status === input.mode &&
      (input.mode === 'canary' || input.mode === 'champion')
    ) {
      if (input.mode === 'canary') {
        promotion.status = 'rejected'
      } else {
        promotion.status = 'retired'
        const previous = this.state.promotions.find(
          (candidate) => candidate.id === promotion.previousChampionId,
        )
        if (previous) {
          previous.status = 'champion'
          previous.updatedAt = this.now()
          this.state.championPromotionId = previous.id
        } else {
          this.state.championPromotionId = undefined
        }
      }
      promotion.updatedAt = this.now()
      this.audit(
        'system:inference-router',
        'promotion.auto-rollback',
        'promotion',
        promotion.id,
        { mode: input.mode, errorRate, observations: recent.length },
      )
    }
    this.persist()
    this.audit(
      'system:inference-router',
      'inference.observed',
      'promotion',
      promotion.id,
      {
        jobId: observation.jobId,
        mode: observation.mode,
        success: observation.success,
        fallbackUsed: observation.fallbackUsed,
        durationMs: observation.durationMs,
      },
    )
    return this.inferenceDeploymentStatus()
  }

  inferenceDeploymentStatus(
    account?: CommunityAccount,
  ): InferenceDeploymentStatus {
    if (account) requiredRole(account, 'model-maintainer')
    const staged = [...this.state.promotions]
      .filter(
        (promotion) =>
          promotion.status === 'shadow' || promotion.status === 'canary',
      )
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0]
    const champion = this.state.promotions.find(
      (promotion) =>
        promotion.id === this.state.championPromotionId &&
        promotion.status === 'champion',
    )
    const active = staged ?? champion
    const observations = active
      ? this.state.inferenceObservations.filter(
          (observation) =>
            observation.promotionId === active.id &&
            observation.mode === active.status,
        )
      : []
    const successful = observations.filter(
      (observation) => observation.success && !observation.fallbackUsed,
    ).length
    const fallbackCount = observations.filter(
      (observation) => observation.fallbackUsed,
    ).length
    const failures = observations.filter(
      (observation) => !observation.success || observation.fallbackUsed,
    ).length
    return {
      config: { ...this.state.deploymentConfig },
      ...(active
        ? {
            active: {
              promotionId: active.id,
              modelVersion: active.modelVersion,
              mode: active.status as 'shadow' | 'canary' | 'champion',
            },
          }
        : {}),
      observations: {
        total: observations.length,
        successful,
        fallbackCount,
        errorRate: observations.length === 0 ? 0 : failures / observations.length,
        averageDurationMs:
          observations.length === 0
            ? 0
            : Math.round(
                observations.reduce(
                  (total, observation) => total + observation.durationMs,
                  0,
                ) / observations.length,
              ),
        lastRecordedAt: observations.at(-1)?.recordedAt,
      },
      recent: this.state.inferenceObservations.slice(-20).reverse(),
    }
  }

  setGovernanceMode(
    account: CommunityAccount,
    mode: unknown,
  ): CommunityState['governanceMode'] {
    requiredRole(account, 'governance-admin')
    if (mode !== 'single-maintainer' && mode !== 'team') {
      throw new CommunityRequestError('治理模式无效')
    }
    if (
      mode === 'team' &&
      this.state.accounts.filter(
        (candidate) =>
          !candidate.disabledAt && candidate.roles.includes('model-maintainer'),
      ).length < 2
    ) {
      throw new CommunityRequestError('团队治理至少需要两名模型维护者', 409)
    }
    this.state.governanceMode = mode
    this.persist()
    this.audit(account.id, 'governance.mode-changed', 'system', 'community', {
      mode,
    })
    return mode
  }

  dashboard(): CommunityDashboard {
    this.requeueExpiredLeases()
    const activeSubmissions = this.state.submissions.filter(
      (submission) => !submission.withdrawnAt,
    )
    const consensusTasks = this.state.tasks.filter(
      (task) => task.status === 'consensus',
    )
    const consensusSubmissionIds = new Set(
      consensusTasks.flatMap(
        (task) => task.consensus?.supportingSubmissionIds ?? [],
      ),
    )
    const calibration = activeSubmissions.filter(
      (submission) => submission.calibrationCorrect !== undefined,
    )
    const rightsCoverage =
      this.state.sources.length === 0
        ? 0
        : this.state.sources.filter(
              (source) => source.rightsStatus === 'active' && source.license,
            ).length / this.state.sources.length
    const maintenance: CommunityDashboard['maintenance'] = []
    const escalated = this.state.tasks.filter(
      (task) => task.status === 'escalated',
    ).length
    const blockedReleases = this.state.releases.filter(
      (release) => release.status === 'blocked',
    ).length
    const deployment = this.inferenceDeploymentStatus()
    if (escalated > 0) {
      maintenance.push({
        severity: 'warning',
        message: `${escalated} 个任务存在分歧，等待所有者处理`,
      })
    }
    if (blockedReleases > 0) {
      maintenance.push({
        severity: 'warning',
        message: `${blockedReleases} 个数据版本因来源撤回被阻断`,
      })
    }
    if (
      this.state.experiments.some(
        (experiment) => experiment.status === 'queued',
      )
    ) {
      maintenance.push({
        severity: 'info',
        message: '受控训练队列有待执行任务',
      })
    }
    if (deployment.active) {
      const modeLabel = {
        shadow: '影子',
        canary: '灰度',
        champion: '生产',
      }[deployment.active.mode]
      const observations = deployment.observations
      maintenance.push({
        severity:
          observations.total >= deployment.config.minimumObservations &&
          observations.errorRate >
            deployment.config.errorBudgetPercent / 100
            ? 'warning'
            : 'info',
        message:
          observations.total < deployment.config.minimumObservations
            ? `${modeLabel}部署正在收集观测：${observations.total}/${deployment.config.minimumObservations}`
            : `${modeLabel}部署错误率 ${(observations.errorRate * 100).toFixed(1)}%，预算 ${deployment.config.errorBudgetPercent}%`,
      })
    }
    if (maintenance.length === 0) {
      maintenance.push({
        severity: 'info',
        message: '当前没有需要立即处理的治理事件',
      })
    }

    const phases: CommunityDashboard['phases'] = [
      {
        id: 'C0',
        label: '治理与契约',
        status: 'ready',
        detail: '账号、权限、审计与撤回控制面可用',
      },
      {
        id: 'C1',
        label: '校准与金题',
        status: this.state.tasks.some((task) => task.calibrationAnswer)
          ? 'active'
          : 'waiting',
        detail: '从已批准审核包生成隐藏校准任务',
      },
      {
        id: 'C2',
        label: '社区复核',
        status: this.state.tasks.length > 0 ? 'active' : 'waiting',
        detail: '盲审、共识、争议升级与贡献撤回',
      },
      {
        id: 'C3',
        label: '数据发布',
        status: this.state.releases.some((release) => release.status === 'active')
          ? 'active'
          : 'waiting',
        detail: '冻结可追溯的 owner_verified 数据版本',
      },
      {
        id: 'C4',
        label: '受控训练',
        status: this.state.experiments.length > 0 ? 'active' : 'ready',
        detail: '批准配方、租约执行器、参数边界与可复现运行记录',
      },
      {
        id: 'C5',
        label: '模型晋级',
        status: this.state.promotions.length > 0 ? 'active' : 'waiting',
        detail: deployment.active
          ? `${deployment.active.modelVersion} 正处于${statusLabelForDeployment(deployment.active.mode)}部署`
          : '密封门禁、影子、灰度、生产与回滚',
      },
      {
        id: 'C6',
        label: '长期运营',
        status: 'active',
        detail: `质量、队列与部署持续观测；已保留 ${deployment.recent.length} 条近期推理记录`,
      },
    ]
    return {
      governanceMode: this.state.governanceMode,
      counts: {
        contributors: this.state.accounts.filter(
          (account) => !account.disabledAt,
        ).length,
        openTasks: this.state.tasks.filter((task) => task.status === 'open').length,
        consensusTasks: consensusTasks.length,
        escalatedTasks: escalated,
        submissions: activeSubmissions.length,
        activeReleases: this.state.releases.filter(
          (release) => release.status === 'active',
        ).length,
        queuedExperiments: this.state.experiments.filter(
          (experiment) =>
            experiment.status === 'queued' || experiment.status === 'running',
        ).length,
        completedExperiments: this.state.experiments.filter(
          (experiment) => experiment.status === 'completed',
        ).length,
        promotionCandidates: this.state.promotions.filter(
          (promotion) =>
            promotion.status !== 'champion' &&
            promotion.status !== 'retired' &&
            promotion.status !== 'rejected',
        ).length,
      },
      quality: {
        agreementRate:
          activeSubmissions.length === 0
            ? 0
            : consensusSubmissionIds.size / activeSubmissions.length,
        calibrationAccuracy:
          calibration.length === 0
            ? 0
            : calibration.filter((submission) => submission.calibrationCorrect)
                .length / calibration.length,
        rightsCoverage,
      },
      phases,
      maintenance,
    }
  }

  auditEvents(account: CommunityAccount): AuditEvent[] {
    requiredRole(account, 'governance-admin')
    if (!existsSync(this.auditPath)) return []
    return readFileSync(this.auditPath, 'utf8')
      .split('\n')
      .filter(Boolean)
      .slice(-MAX_AUDIT_EVENTS)
      .map((line) => JSON.parse(line) as AuditEvent)
      .reverse()
  }
}

export function createCommunityStore(
  root?: string,
  options?: CommunityStoreOptions,
): CommunityStore {
  return new CommunityStore(root, options)
}
