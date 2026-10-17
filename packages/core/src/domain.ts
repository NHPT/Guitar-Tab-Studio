import type { AuthorizedMediaProvider } from './media.js'

export const STUDIO_PROJECT_SCHEMA_VERSION = 1 as const

export type PlatformId =
  | 'upload'
  | 'bilibili'
  | 'douyin'
  | 'netease'
  | 'qqmusic'
  | 'youtube'
  | 'vimeo'
  | 'soundcloud'
  | 'bandcamp'
  | 'jamendo'
  | 'audius'
  | 'internet-archive'
  | 'wikimedia'

export type JobStatus =
  | 'queued'
  | 'validating'
  | 'acquiring'
  | 'separating'
  | 'transcribing'
  | 'mapping'
  | 'completed'
  | 'degraded'
  | 'cancelled'
  | 'failed'

export type TrackRole =
  | 'vocals'
  | 'guitar'
  | 'bass'
  | 'drums'
  | 'piano'
  | 'other'

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

export type BeatStyle =
  | 'pick'
  | 'strum'
  | 'arpeggio'
  | 'rasgueado'
  | 'tremolo'

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
  localAuthorizedMediaDecoder?: boolean
  mode: 'demo' | 'hybrid' | 'inference'
}

export interface AuthorizedMediaConsent {
  accepted: true
  policyVersion: 2
}

export interface AuthorizedMediaGrant {
  provider: AuthorizedMediaProvider
  purpose: 'analysis-and-score-generation'
  decodedOutput: 'ephemeral'
  policyVersion: 1 | 2
}

export interface PlatformDescriptor {
  id: Exclude<PlatformId, 'upload'>
  label: string
  hosts: string[]
  acquisition: 'public-media' | 'metadata-only'
  note: string
}

export interface SourceDescriptor {
  kind: PlatformId
  label: string
  url?: string
  filename?: string
  authorizedMedia?: AuthorizedMediaGrant
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
  tieFromPrevious?: boolean
  tieToNext?: boolean
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

export type TimeSignature = [number, 2 | 4 | 8 | 16]

export interface VocalMelodyNote {
  id: string
  at: number
  duration: number
  pitch: number
  confidence: number
}

export interface TimedLyric {
  at: number
  text: string
}

export interface TabDocument {
  tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
  capo: number
  timeSignature: TimeSignature
  vocalMelody?: VocalMelodyNote[]
  lyrics?: TimedLyric[]
  measures: TabMeasure[]
}

export interface EasyArrangement {
  kind: 'easy'
  strategyVersion: 'pitch-preserving-v1'
  createdAt: string
  tab: TabDocument
}

export interface ProjectProvenance {
  compositionCredit: string
  arrangementCredit: string
  recordingCredit: string
  assetLicense: string
  note?: string
  references: string[]
}

export interface StudioProject {
  schemaVersion: typeof STUDIO_PROJECT_SCHEMA_VERSION
  id: string
  title: string
  artist: string
  source: SourceDescriptor
  bpm: number
  key: string
  duration: number
  createdAt: string
  analysisMode: 'demo' | 'inference'
  transcriptionModel?: TranscriptionModelInfo
  techniqueModel?: TechniqueModelInfo
  tracks: StemTrack[]
  visualization?: VisualizationPoint[]
  tab: TabDocument
  easyArrangement?: EasyArrangement
  provenance?: ProjectProvenance
  warnings: string[]
}

export type ProjectRevisionKind =
  | 'inference'
  | 'manual'
  | 'arrangement'
  | 'restore'
  | 'import'

export interface ProjectRevision {
  schemaVersion: 1
  id: string
  projectId: string
  parentRevisionId?: string
  createdAt: string
  label: string
  kind: ProjectRevisionKind
  project: StudioProject
}

export interface AnalysisJob {
  id: string
  source: SourceDescriptor
  status: JobStatus
  progress: number
  stageLabel: string
  createdAt: string
  updatedAt: string
  projectId?: string
  error?: string
  deployment?: {
    mode: 'baseline' | 'shadow' | 'canary' | 'champion'
    modelVersion?: string
    fallbackUsed?: boolean
  }
}
