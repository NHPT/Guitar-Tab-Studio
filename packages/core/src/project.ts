import {
  STUDIO_PROJECT_SCHEMA_VERSION,
  type EasyArrangement,
  type GuitarPart,
  type ProjectProvenance,
  type SourceDescriptor,
  type StemTrack,
  type StudioProject,
  type TabBeat,
  type TabDocument,
  type TabMeasure,
  type TabNote,
  type Technique,
  type TechniqueCandidate,
  type TechniqueModelInfo,
  type TimedLyric,
  type TimeSignature,
  type TranscriptionModelInfo,
  type VisualizationPoint,
  type VocalMelodyNote,
} from './domain.js'
import {
  CoreValidationError,
  arrayAt,
  booleanAt,
  enumAt,
  numberAt,
  optionalNumberAt,
  optionalStringAt,
  recordAt,
  stringAt,
  stringsAt,
} from './validation.js'
import { tempoMapAt } from './timing.js'

const platformIds = [
  'upload',
  'bilibili',
  'douyin',
  'netease',
  'qqmusic',
  'youtube',
  'vimeo',
  'soundcloud',
  'bandcamp',
  'jamendo',
  'audius',
  'internet-archive',
  'wikimedia',
] as const
const authorizedMediaProviders = [
  'qqmusic',
  'netease',
  'moo',
  'kuwo',
  'kugou',
  'unknown',
] as const
const trackRoles = [
  'vocals',
  'guitar',
  'bass',
  'drums',
  'piano',
  'other',
] as const
const guitarPartRoles = [
  'lead',
  'rhythm',
  'mixed',
  'unknown',
] as const
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
const beatStyles = [
  'pick',
  'strum',
  'arpeggio',
  'rasgueado',
  'tremolo',
] as const
const tuning = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'] as const

function sourceAt(value: unknown, path: string): SourceDescriptor {
  const source = recordAt(value, path)
  const authorizedMedia =
    source.authorizedMedia === undefined
      ? undefined
      : recordAt(source.authorizedMedia, `${path}.authorizedMedia`)
  return {
    kind: enumAt(source.kind, platformIds, `${path}.kind`),
    label: stringAt(source.label, `${path}.label`),
    url: optionalStringAt(source.url, `${path}.url`),
    filename: optionalStringAt(source.filename, `${path}.filename`),
    authorizedMedia: authorizedMedia
      ? {
          provider: enumAt(
            authorizedMedia.provider,
            authorizedMediaProviders,
            `${path}.authorizedMedia.provider`,
          ),
          purpose: enumAt(
            authorizedMedia.purpose,
            ['analysis-and-score-generation'] as const,
            `${path}.authorizedMedia.purpose`,
          ),
          decodedOutput: enumAt(
            authorizedMedia.decodedOutput,
            ['ephemeral'] as const,
            `${path}.authorizedMedia.decodedOutput`,
          ),
          policyVersion: (() => {
            const version = numberAt(
              authorizedMedia.policyVersion,
              `${path}.authorizedMedia.policyVersion`,
            )
            if (version !== 1 && version !== 2) {
              throw new CoreValidationError(
                `${path}.authorizedMedia.policyVersion`,
                'expected 1 or 2',
              )
            }
            return version as 1 | 2
          })(),
        }
      : undefined,
  }
}

function techniqueCandidateAt(
  value: unknown,
  path: string,
): TechniqueCandidate {
  const candidate = recordAt(value, path)
  return {
    technique: enumAt(candidate.technique, techniques, `${path}.technique`),
    confidence: numberAt(candidate.confidence, `${path}.confidence`),
    evidence: stringsAt(candidate.evidence, `${path}.evidence`),
  }
}

function tabNoteAt(value: unknown, path: string): TabNote {
  const note = recordAt(value, path)
  const string = numberAt(note.string, `${path}.string`)
  if (!Number.isInteger(string) || string < 1 || string > 6) {
    throw new CoreValidationError(`${path}.string`, 'expected guitar string 1-6')
  }
  const harmonicType =
    note.harmonicType === undefined
      ? undefined
      : enumAt(
          note.harmonicType,
          ['natural', 'artificial'] as const,
          `${path}.harmonicType`,
        )
  return {
    id: stringAt(note.id, `${path}.id`),
    string: string as TabNote['string'],
    fret: numberAt(note.fret, `${path}.fret`),
    at: numberAt(note.at, `${path}.at`),
    notationAt: optionalNumberAt(note.notationAt, `${path}.notationAt`),
    notationBeat: optionalNumberAt(
      note.notationBeat,
      `${path}.notationBeat`,
    ),
    duration: numberAt(note.duration, `${path}.duration`),
    technique: enumAt(note.technique, techniques, `${path}.technique`),
    harmonicType,
    harmonicTouchFret: optionalNumberAt(
      note.harmonicTouchFret,
      `${path}.harmonicTouchFret`,
    ),
    confidence: numberAt(note.confidence, `${path}.confidence`),
    positionConfidence: optionalNumberAt(
      note.positionConfidence,
      `${path}.positionConfidence`,
    ),
    positionSource: optionalStringAt(
      note.positionSource,
      `${path}.positionSource`,
    ),
    techniqueConfidence: optionalNumberAt(
      note.techniqueConfidence,
      `${path}.techniqueConfidence`,
    ),
    techniqueSource: optionalStringAt(
      note.techniqueSource,
      `${path}.techniqueSource`,
    ),
    techniqueEvidence:
      note.techniqueEvidence === undefined
        ? undefined
        : stringsAt(note.techniqueEvidence, `${path}.techniqueEvidence`),
    techniqueCandidates:
      note.techniqueCandidates === undefined
        ? undefined
        : arrayAt(note.techniqueCandidates, `${path}.techniqueCandidates`).map(
            (candidate, index) =>
              techniqueCandidateAt(
                candidate,
                `${path}.techniqueCandidates[${index}]`,
              ),
          ),
    relatedNoteId: optionalStringAt(
      note.relatedNoteId,
      `${path}.relatedNoteId`,
    ),
    tieFromPrevious:
      note.tieFromPrevious === undefined
        ? undefined
        : booleanAt(note.tieFromPrevious, `${path}.tieFromPrevious`),
    tieToNext:
      note.tieToNext === undefined
        ? undefined
        : booleanAt(note.tieToNext, `${path}.tieToNext`),
  }
}

function tabBeatAt(value: unknown, path: string): TabBeat {
  const beat = recordAt(value, path)
  return {
    at: numberAt(beat.at, `${path}.at`),
    duration: numberAt(beat.duration, `${path}.duration`),
    direction:
      beat.direction === undefined
        ? undefined
        : enumAt(beat.direction, ['down', 'up'] as const, `${path}.direction`),
    style:
      beat.style === undefined
        ? undefined
        : enumAt(beat.style, beatStyles, `${path}.style`),
    notes: arrayAt(beat.notes, `${path}.notes`).map((note, index) =>
      tabNoteAt(note, `${path}.notes[${index}]`),
    ),
  }
}

function tabMeasureAt(value: unknown, path: string): TabMeasure {
  const measure = recordAt(value, path)
  return {
    number: numberAt(measure.number, `${path}.number`),
    start: numberAt(measure.start, `${path}.start`),
    duration: numberAt(measure.duration, `${path}.duration`),
    chord: stringAt(measure.chord, `${path}.chord`),
    section: optionalStringAt(measure.section, `${path}.section`),
    lyric: optionalStringAt(measure.lyric, `${path}.lyric`),
    beats: arrayAt(measure.beats, `${path}.beats`).map((beat, index) =>
      tabBeatAt(beat, `${path}.beats[${index}]`),
    ),
  }
}

function inferTimeSignature(measures: TabMeasure[]): TimeSignature {
  const beatCounts = new Map<number, number>()
  measures.forEach((measure) => {
    if (measure.beats.length > 0) {
      beatCounts.set(
        measure.beats.length,
        (beatCounts.get(measure.beats.length) ?? 0) + 1,
      )
    }
  })
  const predominantBeatCount =
    [...beatCounts.entries()].sort(
      ([leftCount, leftFrequency], [rightCount, rightFrequency]) =>
        rightFrequency - leftFrequency || rightCount - leftCount,
    )[0]?.[0] ?? 4
  if (predominantBeatCount === 6) return [6, 8]
  if (
    predominantBeatCount >= 1 &&
    predominantBeatCount <= 12 &&
    Number.isInteger(predominantBeatCount)
  ) {
    return [predominantBeatCount, 4]
  }
  return [4, 4]
}

function timeSignatureAt(
  value: unknown,
  path: string,
  measures: TabMeasure[],
): TimeSignature {
  if (value === undefined) return inferTimeSignature(measures)
  const signature = arrayAt(value, path)
  if (signature.length !== 2) {
    throw new CoreValidationError(path, 'expected [numerator, denominator]')
  }
  const numerator = numberAt(signature[0], `${path}[0]`)
  const denominator = numberAt(signature[1], `${path}[1]`)
  if (
    !Number.isInteger(numerator) ||
    numerator < 1 ||
    numerator > 12
  ) {
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

function vocalMelodyNoteAt(
  value: unknown,
  path: string,
): VocalMelodyNote {
  const note = recordAt(value, path)
  return {
    id: stringAt(note.id, `${path}.id`),
    at: numberAt(note.at, `${path}.at`),
    duration: numberAt(note.duration, `${path}.duration`),
    pitch: numberAt(note.pitch, `${path}.pitch`),
    confidence: numberAt(note.confidence, `${path}.confidence`),
  }
}

function timedLyricAt(value: unknown, path: string): TimedLyric {
  const lyric = recordAt(value, path)
  return {
    at: numberAt(lyric.at, `${path}.at`),
    text: stringAt(lyric.text, `${path}.text`),
  }
}

function tabDocumentAt(value: unknown, path: string): TabDocument {
  const tab = recordAt(value, path)
  const parsedTuning = stringsAt(tab.tuning, `${path}.tuning`)
  if (
    parsedTuning.length !== tuning.length ||
    parsedTuning.some((note, index) => note !== tuning[index])
  ) {
    throw new CoreValidationError(`${path}.tuning`, 'unsupported tuning')
  }
  const measures = arrayAt(tab.measures, `${path}.measures`).map(
    (measure, index) =>
      tabMeasureAt(measure, `${path}.measures[${index}]`),
  )
  return {
    tuning: [...tuning],
    capo: numberAt(tab.capo, `${path}.capo`),
    timeSignature: timeSignatureAt(
      tab.timeSignature,
      `${path}.timeSignature`,
      measures,
    ),
    tempoMap:
      tab.tempoMap === undefined
        ? undefined
        : tempoMapAt(tab.tempoMap, `${path}.tempoMap`),
    vocalMelody:
      tab.vocalMelody === undefined
        ? undefined
        : arrayAt(tab.vocalMelody, `${path}.vocalMelody`).map(
            (note, index) =>
              vocalMelodyNoteAt(note, `${path}.vocalMelody[${index}]`),
          ),
    lyrics:
      tab.lyrics === undefined
        ? undefined
        : arrayAt(tab.lyrics, `${path}.lyrics`).map((lyric, index) =>
            timedLyricAt(lyric, `${path}.lyrics[${index}]`),
          ),
    measures,
  }
}

function easyArrangementAt(
  value: unknown,
  path: string,
): EasyArrangement {
  const arrangement = recordAt(value, path)
  return {
    kind: enumAt(arrangement.kind, ['easy'] as const, `${path}.kind`),
    strategyVersion: enumAt(
      arrangement.strategyVersion,
      ['pitch-preserving-v1'] as const,
      `${path}.strategyVersion`,
    ),
    createdAt: stringAt(arrangement.createdAt, `${path}.createdAt`),
    tab: tabDocumentAt(arrangement.tab, `${path}.tab`),
  }
}

function guitarPartAt(value: unknown, path: string): GuitarPart {
  const part = recordAt(value, path)
  return {
    id: stringAt(part.id, `${path}.id`),
    trackId: stringAt(part.trackId, `${path}.trackId`),
    name: stringAt(part.name, `${path}.name`),
    role: enumAt(part.role, guitarPartRoles, `${path}.role`),
    roleConfidence: optionalNumberAt(
      part.roleConfidence,
      `${path}.roleConfidence`,
    ),
    tab: tabDocumentAt(part.tab, `${path}.tab`),
    transcriptionModel:
      part.transcriptionModel === undefined
        ? undefined
        : transcriptionModelAt(
            part.transcriptionModel,
            `${path}.transcriptionModel`,
          ),
  }
}

function stemTrackAt(value: unknown, path: string): StemTrack {
  const track = recordAt(value, path)
  const url =
    track.url === null ? null : stringAt(track.url, `${path}.url`)
  return {
    id: stringAt(track.id, `${path}.id`),
    name: stringAt(track.name, `${path}.name`),
    role: enumAt(track.role, trackRoles, `${path}.role`),
    color: stringAt(track.color, `${path}.color`),
    url,
    waveform: arrayAt(track.waveform, `${path}.waveform`).map(
      (point, index) => numberAt(point, `${path}.waveform[${index}]`),
    ),
    level: optionalNumberAt(track.level, `${path}.level`),
    available: booleanAt(track.available, `${path}.available`),
  }
}

function visualizationPointAt(
  value: unknown,
  path: string,
): VisualizationPoint {
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

function transcriptionModelAt(
  value: unknown,
  path: string,
): TranscriptionModelInfo {
  const model = recordAt(value, path)
  return {
    name: stringAt(model.name, `${path}.name`),
    version: stringAt(model.version, `${path}.version`),
    kind: enumAt(model.kind, ['baseline', 'trained'] as const, `${path}.kind`),
    timingModel: optionalStringAt(
      model.timingModel,
      `${path}.timingModel`,
    ),
    positionOptimizer: optionalStringAt(
      model.positionOptimizer,
      `${path}.positionOptimizer`,
    ),
  }
}

function techniqueModelAt(value: unknown, path: string): TechniqueModelInfo {
  const model = recordAt(value, path)
  return {
    name: stringAt(model.name, `${path}.name`),
    version: stringAt(model.version, `${path}.version`),
    kind: enumAt(model.kind, ['heuristic', 'trained'] as const, `${path}.kind`),
    labels: arrayAt(model.labels, `${path}.labels`).map((label, index) =>
      enumAt(label, techniques, `${path}.labels[${index}]`),
    ),
  }
}

function projectProvenanceAt(
  value: unknown,
  path: string,
): ProjectProvenance {
  const provenance = recordAt(value, path)
  return {
    compositionCredit: stringAt(
      provenance.compositionCredit,
      `${path}.compositionCredit`,
    ),
    arrangementCredit: stringAt(
      provenance.arrangementCredit,
      `${path}.arrangementCredit`,
    ),
    recordingCredit: stringAt(
      provenance.recordingCredit,
      `${path}.recordingCredit`,
    ),
    assetLicense: stringAt(
      provenance.assetLicense,
      `${path}.assetLicense`,
    ),
    note: optionalStringAt(provenance.note, `${path}.note`),
    references: stringsAt(provenance.references, `${path}.references`),
  }
}

function applyTechniqueGate(
  tab: TabDocument,
  model: TechniqueModelInfo | undefined,
): void {
  if (
    model?.kind !== 'heuristic' ||
    model.name !== 'stringtrace-technique-baseline'
  ) {
    return
  }
  for (const measure of tab.measures) {
    for (const beat of measure.beats) {
      if (beat.style === 'tremolo' || beat.style === 'rasgueado') {
        beat.style = 'pick'
        beat.direction = undefined
      }
      for (const note of beat.notes) {
        if (
          note.technique === 'pick' ||
          note.technique === 'unknown' ||
          note.technique === 'harmonic' ||
          (note.techniqueSource === 'ordered-string-attacks-v1' &&
            (note.technique === 'strum-up' ||
              note.technique === 'strum-down'))
        ) {
          continue
        }
        note.technique = 'unknown'
        note.techniqueConfidence = 0
        note.techniqueSource = 'unreviewed-heuristic'
        note.relatedNoteId = undefined
        note.harmonicType = undefined
        note.harmonicTouchFret = undefined
      }
    }
  }
}

export function migrateStudioProject(value: unknown): StudioProject {
  const project = recordAt(value, 'project')
  const schemaVersion =
    project.schemaVersion === undefined
      ? 1
      : numberAt(project.schemaVersion, 'project.schemaVersion')
  if (![1, STUDIO_PROJECT_SCHEMA_VERSION].includes(schemaVersion)) {
    throw new CoreValidationError(
      'project.schemaVersion',
      `unsupported version ${schemaVersion}`,
    )
  }
  const tracks = arrayAt(project.tracks, 'project.tracks').map(
    (track, index) => stemTrackAt(track, `project.tracks[${index}]`),
  )
  const tab = tabDocumentAt(project.tab, 'project.tab')
  const transcriptionModel =
    project.transcriptionModel === undefined
      ? undefined
      : transcriptionModelAt(
          project.transcriptionModel,
          'project.transcriptionModel',
        )
  const parsedGuitarParts =
    project.guitarParts === undefined
      ? []
      : arrayAt(project.guitarParts, 'project.guitarParts').map(
          (part, index) =>
            guitarPartAt(part, `project.guitarParts[${index}]`),
        )
  const guitarParts = parsedGuitarParts
    .filter(
      (part, index, parts) =>
        parts.findIndex((candidate) => candidate.id === part.id) === index,
    )
  if (guitarParts.length === 0) {
    const guitarTrack = tracks.find((track) => track.role === 'guitar')
    guitarParts.push({
      id: guitarTrack?.id ?? 'guitar-1',
      trackId: guitarTrack?.id ?? 'guitar',
      name: guitarTrack?.name ?? '吉他 1',
      role: 'unknown',
      tab,
      ...(transcriptionModel ? { transcriptionModel } : {}),
    })
  }
  const requestedPrimaryPartId = optionalStringAt(
    project.primaryGuitarPartId,
    'project.primaryGuitarPartId',
  )
  const primaryGuitarPartId = guitarParts.some(
    (part) => part.id === requestedPrimaryPartId,
  )
    ? requestedPrimaryPartId!
    : guitarParts[0].id
  const primaryPartIndex = guitarParts.findIndex(
    (part) => part.id === primaryGuitarPartId,
  )
  guitarParts[primaryPartIndex] = {
    ...guitarParts[primaryPartIndex],
    tab,
  }
  const migrated: StudioProject = {
    schemaVersion: STUDIO_PROJECT_SCHEMA_VERSION,
    id: stringAt(project.id, 'project.id'),
    title: stringAt(project.title, 'project.title'),
    artist: stringAt(project.artist, 'project.artist'),
    source: sourceAt(project.source, 'project.source'),
    bpm: numberAt(project.bpm, 'project.bpm'),
    key: stringAt(project.key, 'project.key'),
    duration: numberAt(project.duration, 'project.duration'),
    createdAt: stringAt(project.createdAt, 'project.createdAt'),
    analysisMode: enumAt(
      project.analysisMode,
      ['demo', 'inference'] as const,
      'project.analysisMode',
    ),
    transcriptionModel,
    techniqueModel:
      project.techniqueModel === undefined
        ? undefined
        : techniqueModelAt(project.techniqueModel, 'project.techniqueModel'),
    tracks,
    visualization:
      project.visualization === undefined
        ? undefined
        : arrayAt(project.visualization, 'project.visualization').map(
            (point, index) =>
              visualizationPointAt(point, `project.visualization[${index}]`),
          ),
    tab,
    guitarParts,
    primaryGuitarPartId,
    easyArrangement:
      project.easyArrangement === undefined
        ? undefined
        : easyArrangementAt(
            project.easyArrangement,
            'project.easyArrangement',
          ),
    provenance:
      project.provenance === undefined
        ? undefined
        : projectProvenanceAt(project.provenance, 'project.provenance'),
    warnings: stringsAt(project.warnings, 'project.warnings'),
  }
  migrated.tracks = migrated.tracks.filter(
    (track, index, tracks) =>
      tracks.findIndex((candidate) => candidate.id === track.id) === index,
  ).map((track) =>
    track.url && !track.available ? { ...track, available: true } : track,
  )
  applyTechniqueGate(migrated.tab, migrated.techniqueModel)
  if (migrated.easyArrangement) {
    applyTechniqueGate(migrated.easyArrangement.tab, migrated.techniqueModel)
  }
  for (const part of migrated.guitarParts) {
    applyTechniqueGate(part.tab, migrated.techniqueModel)
  }
  return migrated
}
