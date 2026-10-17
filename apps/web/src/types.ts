import type { Technique } from '@gts/core'

export type {
  AnalysisJob,
  AuthorizedMediaConsent,
  BeatStyle,
  CapabilityState,
  JobStatus,
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
