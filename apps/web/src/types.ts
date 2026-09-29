export type TrackRole = 'vocals' | 'guitar' | 'bass' | 'drums' | 'piano' | 'other'
export type VisualizerMode =
  | 'wave'
  | 'glow'
  | 'ecg'
  | 'spectrum'
  | 'ribbon'
  | 'dots'
  | 'reactor'

export type Technique =
  | 'unknown'
  | 'pick'
  | 'strum-down'
  | 'strum-up'
  | 'hammer-on'
  | 'pull-off'
  | 'slide'
  | 'harmonic'
  | 'palm-mute'
  | 'dead-note'
  | 'slap'
  | 'body-tap'
  | 'tremolo'
  | 'bend'
  | 'pinch-harmonic'
  | 'vibrato'

export type BeatStyle = 'pick' | 'strum' | 'arpeggio' | 'rasgueado' | 'tremolo'

export interface TechniqueCandidate {
  technique: Technique
  confidence: number
  evidence: string[]
}

export interface TechniqueModelInfo {
  name: string
  version: string
  kind: 'heuristic' | 'trained'
  labels: Technique[]
}

export interface TranscriptionModelInfo {
  name: string
  version: string
  kind: 'baseline' | 'trained'
}

export interface CapabilityState {
  ffmpeg: boolean
  ffprobe: boolean
  ytDlp: boolean
  worker: boolean
  mode: 'demo' | 'hybrid' | 'inference'
}

export interface PlatformDescriptor {
  id: string
  label: string
  hosts: string[]
  acquisition: 'public-media' | 'metadata-only'
  note: string
}

export interface StemTrack {
  id: string
  name: string
  role: TrackRole
  color: string
  url: string | null
  waveform: number[]
  level?: number
  available: boolean
}

export interface VisualizationPoint {
  energy: number
  pitch: number
  low: number
  mid: number
  high: number
  impact: number
}

export interface TabNote {
  id: string
  string: 1 | 2 | 3 | 4 | 5 | 6
  fret: number
  at: number
  notationAt?: number
  duration: number
  technique: Technique
  harmonicType?: 'natural' | 'artificial'
  harmonicTouchFret?: number
  confidence: number
  positionConfidence?: number
  positionSource?: string
  techniqueConfidence?: number
  techniqueSource?: string
  techniqueEvidence?: string[]
  techniqueCandidates?: TechniqueCandidate[]
  relatedNoteId?: string
}

export interface TabBeat {
  at: number
  duration: number
  direction?: 'down' | 'up'
  style?: BeatStyle
  notes: TabNote[]
}

export interface TabMeasure {
  number: number
  start: number
  duration: number
  chord: string
  section?: string
  lyric?: string
  beats: TabBeat[]
}

export interface StudioProject {
  id: string
  title: string
  artist: string
  source: {
    kind: string
    label: string
    url?: string
    filename?: string
  }
  bpm: number
  key: string
  duration: number
  createdAt: string
  analysisMode: 'demo' | 'inference'
  transcriptionModel?: TranscriptionModelInfo
  techniqueModel?: TechniqueModelInfo
  tracks: StemTrack[]
  visualization?: VisualizationPoint[]
  tab: {
    tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
    capo: number
    measures: TabMeasure[]
  }
  warnings: string[]
}

export interface AnalysisJob {
  id: string
  status:
    | 'queued'
    | 'validating'
    | 'acquiring'
    | 'separating'
    | 'transcribing'
    | 'mapping'
    | 'completed'
    | 'degraded'
    | 'failed'
  progress: number
  stageLabel: string
  projectId?: string
  error?: string
}

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
