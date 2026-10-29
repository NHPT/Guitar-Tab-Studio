import type { Technique } from '@gts/core'

export type {
  AnalysisJob,
  AuthorizedMediaConsent,
  BeatStyle,
  CapabilityState,
  JobStatus,
  MediaBrowserSession,
  PlatformDescriptor,
  PlatformId,
  ProjectRevision,
  ProjectRevisionKind,
  SourceDescriptor,
  StemTrack,
  StudioProject,
  TabBeat,
  TabDocument,
  TabMeasure,
  TabNote,
  Technique,
  TechniqueCandidate,
  TechniqueModelInfo,
  TrackRole,
  TranscriptionModelInfo,
  VisualizationPoint,
} from '@gts/core'

export type VisualizerMode =
  | 'wave'
  | 'glow'
  | 'ecg'
  | 'spectrum'
  | 'ribbon'
  | 'dots'
  | 'reactor'

export interface TrackMix {
  muted: boolean
  solo: boolean
  volume: number
}

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
  technique: Technique
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

export interface ReviewDraft {
  events: ReviewEvent[]
  excluded_ranges: ReviewExcludedRange[]
  checks: ReviewChecks
  capo: number
}

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

export interface CommunityAnswer {
  choice?: 'yes' | 'no' | 'unsure'
  onset?: number
  offset?: number
  pitch?: number
  string?: number
  fret?: number
  technique?: string
}

export interface CommunityAccount {
  id: string
  alias: string
  roles: CommunityRole[]
  governanceMode: 'single-maintainer' | 'team'
  reputation: Partial<
    Record<
      CommunityTaskType,
      {
        submissions: number
        consensusMatches: number
        goldTotal: number
        goldCorrect: number
        goldAccuracy: number
        qualified: boolean
      }
    >
  >
  createdAt: string
  suspendedUntil?: string
  anomalyReason?: string
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
  source: {
    title: string
    subtitle?: string
    audioUrl: string
    dataset: string
    license: string
  }
  submissionCount: number
}

export interface CommunityContribution {
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

export interface CommunityDashboard {
  governanceMode: 'single-maintainer' | 'team'
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
  parameters: Record<
    string,
    {
      type: 'number' | 'integer'
      minimum: number
      maximum: number
      default: number
    }
  >
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
  executionEvidence?: {
    isolation: 'sandbox-exec' | 'podman'
    durationMs: number
    stdoutSha256: string
    stderrSha256: string
    reportSha256: string
  }
  modelVersion?: string
  failureReason?: string
}

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
      | 'lineage'
      | 'reproduction'
      | 'public-validation'
      | 'sealed-evaluation'
      | 'robustness'
      | 'shadow'
      | 'canary',
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

export interface InferenceObservation {
  id: string
  promotionId: string
  jobId: string
  mode: 'shadow' | 'canary' | 'champion'
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
  submission?: {
    submitted_at: string
    content_sha256: string
    admin_user_id?: string
    admin_username?: string
  } | null
  verification_status:
    | 'draft'
    | 'independent_annotation'
    | 'adjudicated_gold'
  benchmark_eligible: boolean
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

export interface BeatReviewDraft {
  beats: BeatReviewMarker[]
  time_signature: [number, number]
  pickup_beats: number
  checks: BeatReviewChecks
}

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
  audio_duration: number
  time_signature: [number, number]
  pickup_beats: number
  phase_status: 'confirmed' | 'uncertain'
  first_downbeat_at: number | null
  meter_changes: BeatMeterChange[]
  checks: BeatStructureChecks
  submission?: {
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

export interface BeatStructureDraft {
  time_signature: [number, number]
  pickup_beats: number
  phase_status: 'confirmed' | 'uncertain'
  first_downbeat_at: number | null
  meter_changes: BeatMeterChange[]
  checks: BeatStructureChecks
}

export interface BeatGoldStatus {
  enabled: boolean
  reviewers: string[]
  manifest: {
    corpusVersion: string
    corpusSha256: string
    immutable: true
    evaluationDuration: number
    context: {
      preRoll: number
      postRoll: number
    }
    recordings: number
    clips: number
    splitCounts: Record<string, number>
    verificationStatus: string
    benchmarkEligible: boolean
  } | null
}

export type AdminRole =
  | 'owner'
  | 'reviewer'
  | 'adjudicator'
  | 'model-maintainer'
  | 'viewer'

export interface AdminUser {
  id: string
  username: string
  displayName: string
  role: AdminRole
  active: boolean
  reviewerAlias?: string
  createdAt: string
  lastLoginAt?: string
}

export interface AdminAuditEvent {
  at: string
  action: string
  actorId: string | null
  actorUsername: string | null
  [key: string]: unknown
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
  split: 'train' | 'validation' | 'test'
  leftStatus: BeatReviewPackage['status']
  rightStatus: BeatReviewPackage['status']
  ready: boolean
  adjudicated: boolean
  independentAccounts: boolean
  beatF1: number | null
  medianErrorMs: number | null
  disagreements: number | null
}

export interface BeatEvaluationReport {
  schemaVersion: 1
  generatedAt: string
  modelVersion: string
  corpusVersion: string
  corpusSha256: string
  coverage: {
    adjudicated: number
    total: number
    complete: boolean
    splitCounts: Record<string, number>
  }
  aggregate: {
    beatMetrics: BeatMetrics
    downbeatMetrics: BeatMetrics
    meterAccuracy: number
    pickupAccuracy: number
  }
  gate: {
    passed: boolean
    checks: Record<string, boolean>
    thresholds: {
      beatF1: number
      medianErrorMs: number
      p95ErrorMs: number
      downbeatF1: number
    }
  }
  items: Array<{
    clipId: string
    split: 'train' | 'validation' | 'test'
    referenceBeats: number
    predictedBeats: number
    beatMetrics: BeatMetrics
    downbeatMetrics: BeatMetrics
    meterCorrect: boolean
    pickupCorrect: boolean
  }>
  reportSha256: string
}

export interface BeatGoldRelease {
  schemaVersion: 1
  version: string
  createdAt: string
  corpusVersion: string
  corpusSha256: string
  clips: number
  splitCounts: Record<string, number>
  files: Array<{
    path: string
    size: number
    sha256: string
  }>
  archive: {
    filename: string
    size: number
    sha256: string
  }
}

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

export interface BeatTrainingStatus {
  minimumTrainClips: number
  adjudicatedTrainClips: number
  datasets: BeatTrainingDataset[]
  jobs: BeatTrainingJob[]
}

export interface ReviewLayerStatus {
  id: 'beat' | 'onset' | 'pitch' | 'string-fret' | 'chord' | 'technique'
  label: string
  order: number
  schemaVersion: 1
  reviewKind: string
  dependsOn?: ReviewLayerStatus['id']
  fields: Array<{
    name: string
    type: string
    required: boolean
  }>
  checks: string[]
  status: 'active' | 'scaffolded' | 'blocked'
  blocker?: string
  reviewed: number
  total: number
}

export interface BeatBackup {
  schemaVersion: 1
  id: string
  createdAt: string
  corpusVersion: string
  corpusSha256: string
  archive: {
    filename: string
    size: number
    sha256: string
  }
  verified: boolean
  restored: boolean
}

export interface AdminSession {
  configured: boolean
  authenticated: boolean
  user: AdminUser | null
}

export interface AdminDashboard {
  generatedAt: string
  corpus: BeatGoldStatus['manifest']
  reviewer: string | null
  reviews: {
    total: number
    draft: number
    submitted: number
    adjudicated: number
    checksComplete: number
  }
  layers: Array<{
    id: string
    label: string
    status: 'ready' | 'blocked'
    dependsOn?: string
  }>
  gates: {
    beatF1: number
    medianErrorMs: number
    p95ErrorMs: number
    downbeatF1: number
  }
}
