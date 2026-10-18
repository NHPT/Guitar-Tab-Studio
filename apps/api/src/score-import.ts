import { execFile } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { accessSync, constants } from 'node:fs'
import { basename, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import {
  migrateStudioProject,
  STUDIO_PROJECT_SCHEMA_VERSION,
  type StudioProject,
  type TabBeat,
  type TabMeasure,
  type TabNote,
} from '@gts/core'

const executeFile = promisify(execFile)
const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url))

export interface OpticalScoreEvent {
  position: number
  duration: number
  string: number
  fret: number
  confidence: number
  source: 'ocr' | 'visual-marker'
}

export interface OpticalScoreDocument {
  version: 1
  title: string
  pageCount: number
  staffCount: number
  timeSignature: [number, 4]
  bpm: number
  confidence: number
  measures: Array<{
    chord: string
    events: OpticalScoreEvent[]
    source: {
      page: number
      staff: number
    }
  }>
  lyrics: Array<{
    position: number
    text: string
  }>
  warnings: string[]
}

function executable(path: string): boolean {
  try {
    accessSync(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : fallback
}

function parseOpticalDocument(value: unknown): OpticalScoreDocument {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('曲谱识别结果格式无效')
  }
  const document = value as Partial<OpticalScoreDocument>
  if (!Array.isArray(document.measures) || document.measures.length === 0) {
    throw new Error('曲谱中没有可导入的小节')
  }
  return {
    version: 1,
    title:
      typeof document.title === 'string' && document.title.trim()
        ? document.title.trim()
        : '导入曲谱',
    pageCount: Math.max(1, Math.round(finite(document.pageCount, 1))),
    staffCount: Math.max(1, Math.round(finite(document.staffCount, 1))),
    timeSignature: [4, 4],
    bpm: Math.max(40, Math.min(240, finite(document.bpm, 90))),
    confidence: Math.max(0, Math.min(1, finite(document.confidence, 0))),
    measures: document.measures.map((measure) => ({
      chord:
        typeof measure?.chord === 'string' && measure.chord.trim()
          ? measure.chord.trim()
          : 'N.C.',
      events: Array.isArray(measure?.events)
        ? measure.events.flatMap((event) => {
            const string = Math.round(finite(event?.string, 0))
            const fret = Math.round(finite(event?.fret, -1))
            if (string < 1 || string > 6 || fret < 0 || fret > 24) return []
            return [{
              position: Math.max(0, Math.min(3.75, finite(event.position, 0))),
              duration: Math.max(0.125, Math.min(4, finite(event.duration, 1))),
              string,
              fret,
              confidence: Math.max(
                0,
                Math.min(1, finite(event.confidence, 0.3)),
              ),
              source: event.source === 'ocr' ? 'ocr' as const : 'visual-marker' as const,
            }]
          })
        : [],
      source: {
        page: Math.max(1, Math.round(finite(measure?.source?.page, 1))),
        staff: Math.max(1, Math.round(finite(measure?.source?.staff, 1))),
      },
    })),
    lyrics: Array.isArray(document.lyrics)
      ? document.lyrics.flatMap((lyric) =>
          typeof lyric?.text === 'string' && lyric.text.trim()
            ? [{
                position: Math.max(0, finite(lyric.position, 0)),
                text: lyric.text.trim(),
              }]
            : [],
        )
      : [],
    warnings: Array.isArray(document.warnings)
      ? document.warnings.filter(
          (warning): warning is string => typeof warning === 'string',
        )
      : [],
  }
}

export function projectFromOpticalScore(
  documentValue: unknown,
  filename: string,
): StudioProject {
  const document = parseOpticalDocument(documentValue)
  const quarterDuration = 60 / document.bpm
  const beatsPerMeasure = document.timeSignature[0]
  let noteIndex = 0
  const measures: TabMeasure[] = document.measures.map(
    (sourceMeasure, measureIndex) => {
      const start = measureIndex * beatsPerMeasure * quarterDuration
      const beats: TabBeat[] = Array.from(
        { length: beatsPerMeasure },
        (_, beatIndex) => ({
          at: start + beatIndex * quarterDuration,
          duration: quarterDuration,
          style: 'pick',
          notes: [],
        }),
      )
      for (const event of sourceMeasure.events) {
        const beatIndex = Math.max(
          0,
          Math.min(beats.length - 1, Math.floor(event.position)),
        )
        const at = start + event.position * quarterDuration
        const note: TabNote = {
          id: `score-${++noteIndex}`,
          string: event.string as TabNote['string'],
          fret: event.fret,
          at,
          notationAt: at,
          duration: Math.min(
            start + beatsPerMeasure * quarterDuration - at,
            event.duration * quarterDuration,
          ),
          technique: event.source === 'ocr' ? 'pick' : 'unknown',
          confidence: event.confidence,
          positionConfidence: event.confidence,
          positionSource: 'source-score-omr-v1',
          techniqueConfidence: event.source === 'ocr' ? 0.72 : 0,
          techniqueSource: 'source-score-omr-v1',
          techniqueEvidence: [
            event.source === 'ocr'
              ? 'optical-fret-recognition'
              : 'optical-marker-position',
          ],
          techniqueCandidates: [],
        }
        beats[beatIndex].notes.push(note)
      }
      return {
        number: measureIndex + 1,
        start,
        duration: beatsPerMeasure * quarterDuration,
        chord: sourceMeasure.chord,
        section:
          measureIndex === 0
            ? `曲谱第 ${sourceMeasure.source.page} 页`
            : undefined,
        beats,
      }
    },
  )
  const duration = measures.at(-1)!.start + measures.at(-1)!.duration
  const waveform = Array.from({ length: 120 }, (_, index) => {
    const start = duration * index / 120
    const end = duration * (index + 1) / 120
    const events = measures.flatMap((measure) =>
      measure.beats.flatMap((beat) =>
        beat.notes.filter((note) => note.at >= start && note.at < end),
      ),
    )
    return Math.min(1, 0.08 + events.length * 0.12)
  })
  const title = document.title || basename(filename).replace(/\.[^.]+$/u, '')
  const warnings = [
    `已从 ${document.pageCount} 页图片中检测 ${document.staffCount} 行六线谱，整体识别置信度 ${Math.round(document.confidence * 100)}%。`,
    '图片曲谱的弦品和节奏需在编辑器中人工校对后再作为正式谱使用。',
    ...document.warnings,
  ]
  return migrateStudioProject({
    schemaVersion: STUDIO_PROJECT_SCHEMA_VERSION,
    id: randomUUID(),
    title,
    artist: '曲谱导入',
    source: {
      kind: 'upload',
      label: `图片/PDF 曲谱 · ${document.pageCount} 页`,
      filename,
    },
    bpm: document.bpm,
    key: 'Unknown',
    duration,
    createdAt: new Date().toISOString(),
    analysisMode: 'inference',
    transcriptionModel: {
      name: 'optical-tab-parser',
      version: 'v1',
      kind: 'baseline',
    },
    techniqueModel: {
      name: 'source-score-symbols',
      version: 'v1',
      kind: 'heuristic',
      labels: ['unknown', 'pick'],
    },
    tracks: [{
      id: 'score-guitar',
      name: '曲谱合成',
      role: 'guitar',
      color: '#19c37d',
      url: null,
      waveform,
      level: 0.18,
      available: true,
    }],
    tab: {
      tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
      capo: 0,
      timeSignature: document.timeSignature,
      lyrics: document.lyrics.map((lyric) => ({
        at: lyric.position * quarterDuration,
        text: lyric.text,
      })),
      measures,
    },
    warnings,
  })
}

export async function importScoreImages(
  paths: string[],
  filename: string,
): Promise<StudioProject> {
  if (paths.length === 0) throw new Error('请选择曲谱图片或 PDF')
  const python =
    process.env.GTS_SCORE_IMPORT_PYTHON ??
    resolve(repositoryRoot, '.venv/bin/python')
  if (!executable(python)) {
    throw new Error('当前运行环境未包含曲谱识别组件')
  }
  const script = resolve(repositoryRoot, 'services/worker/score_import.py')
  const argumentsList = paths.flatMap((path) => ['--input', resolve(path)])
  try {
    const { stdout } = await executeFile(
      python,
      [script, ...argumentsList, '--name', filename],
      {
        cwd: repositoryRoot,
        encoding: 'utf8',
        env: {
          ...process.env,
          PYTHONUTF8: '1',
          PYTHONIOENCODING: 'utf-8',
        },
        timeout: 2 * 60_000,
        maxBuffer: 8 * 1024 * 1024,
      },
    )
    return projectFromOpticalScore(JSON.parse(stdout), filename)
  } catch (error) {
    const stderr =
      typeof error === 'object' &&
      error !== null &&
      'stderr' in error &&
      typeof error.stderr === 'string'
        ? error.stderr
        : ''
    const detail =
      stderr
        .trim()
        .split(/\r?\n/u)
        .reverse()
        .find((line: string) => line.startsWith('ValueError: '))
        ?.slice('ValueError: '.length) ??
      (error instanceof SyntaxError ? '曲谱识别结果无法解析' : '曲谱识别失败')
    throw new Error(detail)
  }
}
