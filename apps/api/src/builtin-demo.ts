import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  parseAnalysisDocument,
  type AnalysisDocument,
} from '@gts/core'

export const builtinDemoAudioFilename =
  'wayfaring-stranger-beginner.wav'
export const builtinDemoStemFilenames = {
  vocals: 'wayfaring-stranger-guide-vocals.wav',
  guitar: builtinDemoAudioFilename,
} as const
export const builtinDemoGuitarPartFilenames = [] as const
const MANIFEST_FILENAME = 'wayfaring-stranger-beginner.json'
const moduleDirectory = dirname(fileURLToPath(import.meta.url))
const assetDirectories = [
  resolve(moduleDirectory, '../assets'),
  resolve(moduleDirectory, 'assets'),
]

interface BeginnerEvent {
  tick: number
  durationTicks: number
  string: 1 | 2 | 3 | 4 | 5 | 6
  fret: number
  sourcePitch: string
  part: 'bass' | 'melody'
  tieFromPrevious?: boolean
  tieToNext?: boolean
  lyricSyllable?: string
}

interface BeginnerManifest {
  title: string
  artist: string
  bpm: number
  key: string
  timeSignature: [number, number]
  ticksPerBeat: number
  audio: {
    filename: string
    vocalsFilename: string
  }
  measures: Array<{
    chord: string
    lyric?: string
    events: BeginnerEvent[]
  }>
}

export function builtinDemoAssetPath(filename: string): string {
  if (!/^[a-z0-9][a-z0-9.-]*$/u.test(filename)) {
    throw new Error('Invalid built-in demo asset name')
  }
  const path = assetDirectories
    .map((directory) => resolve(directory, filename))
    .find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Built-in demo asset is unavailable: ${filename}`)
  }
  return path
}

function readManifest(): BeginnerManifest {
  return JSON.parse(
    readFileSync(builtinDemoAssetPath(MANIFEST_FILENAME), 'utf8'),
  ) as BeginnerManifest
}

function buildAnalysis(manifest: BeginnerManifest): AnalysisDocument {
  const beatDuration = 60 / manifest.bpm
  const tickDuration = beatDuration / manifest.ticksPerBeat
  const ticksPerMeasure =
    manifest.ticksPerBeat * manifest.timeSignature[0]
  const openStringPitches = {
    1: 64,
    2: 59,
    3: 55,
    4: 50,
    5: 45,
    6: 40,
  } as const
  const notes: AnalysisDocument['notes'] = []
  const lyrics: NonNullable<AnalysisDocument['lyrics']> = []

  manifest.measures.forEach((measure, measureIndex) => {
    for (const event of measure.events) {
      const absoluteTick = measureIndex * ticksPerMeasure + event.tick
      const start = absoluteTick * tickDuration
      const end = (absoluteTick + event.durationTicks) * tickDuration
      if (event.tieFromPrevious) {
        const previous = [...notes]
          .reverse()
          .find(
            (note) =>
              note.string === event.string &&
              note.fret === event.fret &&
              Math.abs(note.end - start) < 0.0001,
          )
        if (!previous) {
          throw new Error(
            `Built-in demo tie has no preceding note at measure ${measureIndex + 1}`,
          )
        }
        previous.end = end
      } else {
        notes.push({
          start,
          end,
          pitch: openStringPitches[event.string] + event.fret,
          velocity: event.part === 'melody' ? 96 : 78,
          string: event.string,
          fret: event.fret,
          technique: 'pick',
          confidence: 1,
          position_confidence: 1,
          position_source: 'authored-beginner-arrangement-v1',
          technique_confidence: 1,
          technique_source: 'authored-beginner-arrangement-v1',
          technique_evidence: ['authored-public-domain-arrangement'],
        })
      }
      if (event.lyricSyllable) {
        lyrics.push({ at: start, text: event.lyricSyllable })
      }
    }
  })

  const duration =
    manifest.measures.length *
    manifest.timeSignature[0] *
    beatDuration
  const beats = Array.from(
    {
      length:
        manifest.measures.length * manifest.timeSignature[0] + 1,
    },
    (_value, index) => index * beatDuration,
  )
  return parseAnalysisDocument({
    version: 9,
    transcription_model: {
      name: 'authored-beginner-arrangement',
      version: '1',
      kind: 'baseline',
    },
    technique_model: {
      name: 'authored-techniques',
      version: '1',
      kind: 'heuristic',
      labels: ['pick'],
    },
    metadata: {
      title: manifest.title,
      artist: manifest.artist,
      duration,
    },
    bpm: manifest.bpm,
    time_signature: manifest.timeSignature,
    key: manifest.key,
    capo: 0,
    beats,
    onsets: [...new Set(notes.map((note) => note.start))],
    vocal_melody: notes
      .filter((note) => note.velocity === 96)
      .map((note) => ({
        start: note.start,
        end: note.end,
        pitch: note.pitch,
        confidence: 1,
      })),
    lyrics,
    stems: {
      vocals: manifest.audio.vocalsFilename,
      guitar: manifest.audio.filename,
    },
    notes,
    chords: manifest.measures.map((measure, index) => ({
      start: index * manifest.timeSignature[0] * beatDuration,
      duration: manifest.timeSignature[0] * beatDuration,
      chord: measure.chord,
    })),
    warnings: [
      '内置示例为公共领域旋律的原创初学编配与本地合成录音。',
    ],
  })
}

const manifest = readManifest()
export const builtinDemoAnalysis = buildAnalysis(manifest)
export const builtinDemoAudioPath = builtinDemoAssetPath(
  builtinDemoAudioFilename,
)
export const builtinDemoMediaPaths = new Map(
  [
    builtinDemoAudioFilename,
    ...Object.values(builtinDemoStemFilenames),
    ...builtinDemoGuitarPartFilenames,
  ]
    .filter((filename, index, filenames) => filenames.indexOf(filename) === index)
    .map((filename) => [filename, builtinDemoAssetPath(filename)]),
)
