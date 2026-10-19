import type {
  GuitarPartRole,
  Technique,
  TechniqueModelInfo,
  TimeSignature,
  TrackRole,
  TranscriptionModelInfo,
  VisualizationPoint,
} from './domain.js'
import {
  CoreValidationError,
  arrayAt,
  enumAt,
  numberAt,
  optionalNumberAt,
  optionalStringAt,
  recordAt,
  stringAt,
  stringsAt,
} from './validation.js'

export const ANALYSIS_SCHEMA_VERSION = 10 as const
export const MINIMUM_ANALYSIS_SCHEMA_VERSION = 1 as const

const trackRoles = [
  'vocals',
  'guitar',
  'bass',
  'drums',
  'piano',
  'other',
] as const satisfies readonly TrackRole[]
const guitarPartRoles = [
  'lead',
  'rhythm',
  'mixed',
  'unknown',
] as const satisfies readonly GuitarPartRole[]
const techniques = [
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
] as const satisfies readonly Technique[]

export interface AnalysisTechniqueCandidate {
  technique: Technique
  confidence: number
  evidence: string[]
}

export interface AnalysisNote {
  start: number
  end: number
  pitch: number
  velocity: number
  string: 1 | 2 | 3 | 4 | 5 | 6
  fret: number
  technique: Technique
  harmonic_type?: 'natural' | 'artificial' | null
  harmonic_touch_fret?: number | null
  confidence: number
  position_confidence?: number
  position_source?: string
  technique_confidence?: number
  technique_source?: string
  technique_evidence?: string[]
  technique_candidates?: AnalysisTechniqueCandidate[]
  related_note_index?: number | null
}

export interface AnalysisStemTrack {
  id: string
  name: string
  role: TrackRole
  path: string
  level?: number
  waveform?: number[]
  separation_confidence?: number
  separation_method?: string
}

export interface AnalysisGuitarPart {
  id: string
  track_id: string
  name: string
  role: GuitarPartRole
  role_confidence?: number
  transcription_model?: TranscriptionModelInfo
  notes: AnalysisNote[]
  chords: Array<{ start: number; duration: number; chord: string }>
}

export interface AnalysisVocalMelodyNote {
  start: number
  end: number
  pitch: number
  confidence: number
}

export interface AnalysisTimedLyric {
  at: number
  text: string
}

export interface AnalysisDocument {
  version: typeof ANALYSIS_SCHEMA_VERSION
  transcription_model?: TranscriptionModelInfo
  technique_model?: TechniqueModelInfo
  metadata?: {
    title?: string
    artist?: string
    duration?: number
    webpage_url?: string
  }
  bpm: number
  time_signature?: TimeSignature
  key?: string
  capo?: number
  beats?: number[]
  onsets?: number[]
  vocal_melody?: AnalysisVocalMelodyNote[]
  lyrics?: AnalysisTimedLyric[]
  stems: Partial<Record<TrackRole, string>>
  stem_tracks?: AnalysisStemTrack[]
  guitar_parts?: AnalysisGuitarPart[]
  levels?: Partial<Record<TrackRole, number>>
  waveforms?: Partial<Record<TrackRole, number[]>>
  visualization?: VisualizationPoint[]
  notes: AnalysisNote[]
  chords: Array<{ start: number; duration: number; chord: string }>
  warnings: string[]
}

function stemTrackAt(value: unknown, path: string): AnalysisStemTrack {
  const track = recordAt(value, path)
  return {
    id: stringAt(track.id, `${path}.id`),
    name: stringAt(track.name, `${path}.name`),
    role: enumAt(track.role, trackRoles, `${path}.role`),
    path: stringAt(track.path, `${path}.path`),
    level: optionalNumberAt(track.level, `${path}.level`),
    waveform:
      track.waveform === undefined
        ? undefined
        : numericArrayAt(track.waveform, `${path}.waveform`),
    separation_confidence: optionalNumberAt(
      track.separation_confidence,
      `${path}.separation_confidence`,
    ),
    separation_method: optionalStringAt(
      track.separation_method,
      `${path}.separation_method`,
    ),
  }
}

function guitarPartAt(value: unknown, path: string): AnalysisGuitarPart {
  const part = recordAt(value, path)
  return {
    id: stringAt(part.id, `${path}.id`),
    track_id: stringAt(part.track_id, `${path}.track_id`),
    name: stringAt(part.name, `${path}.name`),
    role: enumAt(part.role, guitarPartRoles, `${path}.role`),
    role_confidence: optionalNumberAt(
      part.role_confidence,
      `${path}.role_confidence`,
    ),
    transcription_model:
      part.transcription_model === undefined
        ? undefined
        : transcriptionModelAt(
            part.transcription_model,
            `${path}.transcription_model`,
          ),
    notes: arrayAt(part.notes ?? [], `${path}.notes`).map((note, index) =>
      noteAt(note, `${path}.notes[${index}]`),
    ),
    chords: arrayAt(part.chords ?? [], `${path}.chords`).map(
      (chord, index) => chordAt(chord, `${path}.chords[${index}]`),
    ),
  }
}

function numericArrayAt(value: unknown, path: string): number[] {
  return arrayAt(value, path).map((item, index) =>
    numberAt(item, `${path}[${index}]`),
  )
}

function modelVersionAt(value: unknown, path: string): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  throw new CoreValidationError(path, 'expected string or finite number')
}

function trackStringsAt(
  value: unknown,
  path: string,
): Partial<Record<TrackRole, string>> {
  const record = recordAt(value, path)
  return Object.fromEntries(
    trackRoles.flatMap((role) =>
      record[role] === undefined
        ? []
        : [[role, stringAt(record[role], `${path}.${role}`)]],
    ),
  )
}

function trackNumbersAt(
  value: unknown,
  path: string,
): Partial<Record<TrackRole, number>> {
  const record = recordAt(value, path)
  return Object.fromEntries(
    trackRoles.flatMap((role) =>
      record[role] === undefined
        ? []
        : [[role, numberAt(record[role], `${path}.${role}`)]],
    ),
  )
}

function trackWaveformsAt(
  value: unknown,
  path: string,
): Partial<Record<TrackRole, number[]>> {
  const record = recordAt(value, path)
  return Object.fromEntries(
    trackRoles.flatMap((role) =>
      record[role] === undefined
        ? []
        : [[role, numericArrayAt(record[role], `${path}.${role}`)]],
    ),
  )
}

function transcriptionModelAt(
  value: unknown,
  path: string,
): TranscriptionModelInfo {
  const model = recordAt(value, path)
  const kind = enumAt(
    model.kind,
    ['baseline', 'trained', 'hybrid'] as const,
    `${path}.kind`,
  )
  return {
    name: stringAt(model.name, `${path}.name`),
    version: modelVersionAt(model.version, `${path}.version`),
    kind: kind === 'hybrid' ? 'trained' : kind,
  }
}

function techniqueModelAt(value: unknown, path: string): TechniqueModelInfo {
  const model = recordAt(value, path)
  return {
    name: stringAt(model.name, `${path}.name`),
    version: modelVersionAt(model.version, `${path}.version`),
    kind: enumAt(model.kind, ['heuristic', 'trained'] as const, `${path}.kind`),
    labels: arrayAt(model.labels, `${path}.labels`).map((label, index) =>
      enumAt(label, techniques, `${path}.labels[${index}]`),
    ),
  }
}

function metadataAt(
  value: unknown,
  path: string,
): NonNullable<AnalysisDocument['metadata']> {
  const metadata = recordAt(value, path)
  return {
    title: optionalStringAt(metadata.title, `${path}.title`),
    artist: optionalStringAt(metadata.artist, `${path}.artist`),
    duration:
      metadata.duration === null
        ? undefined
        : optionalNumberAt(metadata.duration, `${path}.duration`),
    webpage_url: optionalStringAt(metadata.webpage_url, `${path}.webpage_url`),
  }
}

function visualizationAt(value: unknown, path: string): VisualizationPoint {
  const point = recordAt(value, path)
  return {
    energy: numberAt(point.energy, `${path}.energy`),
    pitch: numberAt(point.pitch, `${path}.pitch`),
    low: numberAt(point.low, `${path}.low`),
    mid: numberAt(point.mid, `${path}.mid`),
    high: numberAt(point.high, `${path}.high`),
    impact: numberAt(point.impact, `${path}.impact`),
  }
}

function candidateAt(
  value: unknown,
  path: string,
): AnalysisTechniqueCandidate {
  const candidate = recordAt(value, path)
  return {
    technique: enumAt(candidate.technique, techniques, `${path}.technique`),
    confidence: numberAt(candidate.confidence, `${path}.confidence`),
    evidence: stringsAt(candidate.evidence, `${path}.evidence`),
  }
}

function nullableNumberAt(
  value: unknown,
  path: string,
): number | null | undefined {
  return value === null ? null : optionalNumberAt(value, path)
}

function noteAt(value: unknown, path: string): AnalysisNote {
  const note = recordAt(value, path)
  const string = numberAt(note.string, `${path}.string`)
  if (!Number.isInteger(string) || string < 1 || string > 6) {
    throw new CoreValidationError(`${path}.string`, 'expected guitar string 1-6')
  }
  const relatedNoteIndex = nullableNumberAt(
    note.related_note_index,
    `${path}.related_note_index`,
  )
  if (
    typeof relatedNoteIndex === 'number' &&
    (!Number.isInteger(relatedNoteIndex) || relatedNoteIndex < 0)
  ) {
    throw new CoreValidationError(
      `${path}.related_note_index`,
      'expected non-negative integer',
    )
  }
  return {
    start: numberAt(note.start, `${path}.start`),
    end: numberAt(note.end, `${path}.end`),
    pitch: numberAt(note.pitch, `${path}.pitch`),
    velocity: numberAt(note.velocity, `${path}.velocity`),
    string: string as AnalysisNote['string'],
    fret: numberAt(note.fret, `${path}.fret`),
    technique: enumAt(note.technique, techniques, `${path}.technique`),
    harmonic_type:
      note.harmonic_type === null || note.harmonic_type === undefined
        ? note.harmonic_type
        : enumAt(
            note.harmonic_type,
            ['natural', 'artificial'] as const,
            `${path}.harmonic_type`,
          ),
    harmonic_touch_fret: nullableNumberAt(
      note.harmonic_touch_fret,
      `${path}.harmonic_touch_fret`,
    ),
    confidence: numberAt(note.confidence, `${path}.confidence`),
    position_confidence: optionalNumberAt(
      note.position_confidence,
      `${path}.position_confidence`,
    ),
    position_source: optionalStringAt(
      note.position_source,
      `${path}.position_source`,
    ),
    technique_confidence: optionalNumberAt(
      note.technique_confidence,
      `${path}.technique_confidence`,
    ),
    technique_source: optionalStringAt(
      note.technique_source,
      `${path}.technique_source`,
    ),
    technique_evidence:
      note.technique_evidence === undefined
        ? undefined
        : stringsAt(note.technique_evidence, `${path}.technique_evidence`),
    technique_candidates:
      note.technique_candidates === undefined
        ? undefined
        : arrayAt(note.technique_candidates, `${path}.technique_candidates`).map(
            (candidate, index) =>
              candidateAt(
                candidate,
                `${path}.technique_candidates[${index}]`,
              ),
          ),
    related_note_index: relatedNoteIndex,
  }
}

function chordAt(
  value: unknown,
  path: string,
): AnalysisDocument['chords'][number] {
  const chord = recordAt(value, path)
  return {
    start: numberAt(chord.start, `${path}.start`),
    duration: numberAt(chord.duration, `${path}.duration`),
    chord: stringAt(chord.chord, `${path}.chord`),
  }
}

function vocalMelodyNoteAt(
  value: unknown,
  path: string,
): AnalysisVocalMelodyNote {
  const note = recordAt(value, path)
  return {
    start: numberAt(note.start, `${path}.start`),
    end: numberAt(note.end, `${path}.end`),
    pitch: numberAt(note.pitch, `${path}.pitch`),
    confidence: numberAt(note.confidence, `${path}.confidence`),
  }
}

function timedLyricAt(value: unknown, path: string): AnalysisTimedLyric {
  const lyric = recordAt(value, path)
  return {
    at: numberAt(lyric.at, `${path}.at`),
    text: stringAt(lyric.text, `${path}.text`),
  }
}

function timeSignatureAt(value: unknown, path: string): TimeSignature {
  const signature = arrayAt(value, path)
  if (signature.length !== 2) {
    throw new CoreValidationError(path, 'expected [numerator, denominator]')
  }
  const numerator = numberAt(signature[0], `${path}[0]`)
  const denominator = numberAt(signature[1], `${path}[1]`)
  if (!Number.isInteger(numerator) || numerator < 1 || numerator > 12) {
    throw new CoreValidationError(`${path}[0]`, 'expected integer 1-12')
  }
  if (![2, 4, 8, 16].includes(denominator)) {
    throw new CoreValidationError(
      `${path}[1]`,
      'expected denominator 2, 4, 8, or 16',
    )
  }
  return [numerator, denominator as TimeSignature[1]]
}

export function parseAnalysisDocument(value: unknown): AnalysisDocument {
  const analysis = recordAt(value, 'analysis')
  const sourceVersion = numberAt(analysis.version, 'analysis.version')
  if (
    !Number.isInteger(sourceVersion) ||
    sourceVersion < MINIMUM_ANALYSIS_SCHEMA_VERSION ||
    sourceVersion > ANALYSIS_SCHEMA_VERSION
  ) {
    throw new CoreValidationError(
      'analysis.version',
      `unsupported version ${sourceVersion}`,
    )
  }
  return {
    version: ANALYSIS_SCHEMA_VERSION,
    transcription_model:
      analysis.transcription_model === undefined
        ? undefined
        : transcriptionModelAt(
            analysis.transcription_model,
            'analysis.transcription_model',
          ),
    technique_model:
      analysis.technique_model === undefined
        ? undefined
        : techniqueModelAt(
            analysis.technique_model,
            'analysis.technique_model',
          ),
    metadata:
      analysis.metadata === undefined
        ? undefined
        : metadataAt(analysis.metadata, 'analysis.metadata'),
    bpm: numberAt(analysis.bpm, 'analysis.bpm'),
    time_signature:
      analysis.time_signature === undefined
        ? undefined
        : timeSignatureAt(
            analysis.time_signature,
            'analysis.time_signature',
          ),
    key: optionalStringAt(analysis.key, 'analysis.key'),
    capo: optionalNumberAt(analysis.capo, 'analysis.capo'),
    beats:
      analysis.beats === undefined
        ? undefined
        : numericArrayAt(analysis.beats, 'analysis.beats'),
    onsets:
      analysis.onsets === undefined
        ? undefined
        : numericArrayAt(analysis.onsets, 'analysis.onsets'),
    vocal_melody:
      analysis.vocal_melody === undefined
        ? undefined
        : arrayAt(analysis.vocal_melody, 'analysis.vocal_melody').map(
            (note, index) =>
              vocalMelodyNoteAt(
                note,
                `analysis.vocal_melody[${index}]`,
              ),
          ),
    lyrics:
      analysis.lyrics === undefined
        ? undefined
        : arrayAt(analysis.lyrics, 'analysis.lyrics').map((lyric, index) =>
            timedLyricAt(lyric, `analysis.lyrics[${index}]`),
          ),
    stems: trackStringsAt(analysis.stems ?? {}, 'analysis.stems'),
    stem_tracks:
      analysis.stem_tracks === undefined
        ? undefined
        : arrayAt(analysis.stem_tracks, 'analysis.stem_tracks').map(
            (track, index) =>
              stemTrackAt(track, `analysis.stem_tracks[${index}]`),
          ),
    guitar_parts:
      analysis.guitar_parts === undefined
        ? undefined
        : arrayAt(analysis.guitar_parts, 'analysis.guitar_parts').map(
            (part, index) =>
              guitarPartAt(part, `analysis.guitar_parts[${index}]`),
          ),
    levels:
      analysis.levels === undefined
        ? undefined
        : trackNumbersAt(analysis.levels, 'analysis.levels'),
    waveforms:
      analysis.waveforms === undefined
        ? undefined
        : trackWaveformsAt(analysis.waveforms, 'analysis.waveforms'),
    visualization:
      analysis.visualization === undefined
        ? undefined
        : arrayAt(analysis.visualization, 'analysis.visualization').map(
            (point, index) =>
              visualizationAt(point, `analysis.visualization[${index}]`),
          ),
    notes: arrayAt(analysis.notes ?? [], 'analysis.notes').map((note, index) =>
      noteAt(note, `analysis.notes[${index}]`),
    ),
    chords: arrayAt(analysis.chords ?? [], 'analysis.chords').map(
      (chord, index) => chordAt(chord, `analysis.chords[${index}]`),
    ),
    warnings: stringsAt(analysis.warnings ?? [], 'analysis.warnings'),
  }
}
