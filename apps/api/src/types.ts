export type PlatformId =
  | 'upload'
  | 'bilibili'
  | 'douyin'
  | 'netease'

export type JobStatus =
  | 'queued'
  | 'validating'
  | 'acquiring'
  | 'separating'
  | 'transcribing'
  | 'mapping'
  | 'completed'
  | 'degraded'
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

export interface TabDocument {
  tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
  capo: number
  measures: TabMeasure[]
}

export interface StudioProject {
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
  warnings: string[]
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
}
