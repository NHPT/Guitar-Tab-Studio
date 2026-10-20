import { spawn, spawnSync } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { accessSync, constants } from 'node:fs'
import { mkdir, open, readFile, readdir, rm } from 'node:fs/promises'
import {
  basename,
  delimiter,
  dirname,
  relative,
  resolve,
  sep,
} from 'node:path'
import { fileURLToPath } from 'node:url'
import { StringDecoder } from 'node:string_decoder'
import {
  guitarGestures,
  detectStandardAudioSignature,
  notationPositionAt,
  parseAnalysisDocument,
  requiresAuthorizedMediaDecode,
  STUDIO_PROJECT_SCHEMA_VERSION,
  tempoMapFromBeats,
  type AnalysisDocument,
} from '@gts/core'
import type {
  AnalysisJob,
  CapabilityState,
  PlatformDescriptor,
  PlatformId,
  SourceDescriptor,
  StudioProject,
  TabBeat,
  TabMeasure,
  Technique,
  TechniqueCandidate,
  TrackRole,
} from './types.js'
import type {
  InferenceDeploymentPolicy,
  InferenceObservation,
} from './community.js'
import {
  decodeAuthorizedMediaLocally,
  localAuthorizedMediaDecoderAvailable,
} from './authorized-media.js'
import {
  builtinDemoAnalysis,
  builtinDemoStemFilenames,
} from './builtin-demo.js'
import { runtimeDataRoot } from './runtime.js'

export interface PipelineExecutionOptions {
  deployment?: InferenceDeploymentPolicy
  signal?: AbortSignal
  timeoutMs?: number
  skipShadow?: boolean
  onInferenceObservation?: (
    observation: Omit<InferenceObservation, 'id' | 'recordedAt'>,
  ) => void
}

export interface ResolvedInferenceExecution {
  mode: InferenceDeploymentPolicy['mode']
  promotionId?: string
  modelVersion?: string
  primaryModelPath?: string
  fallbackModelPaths: Array<string | undefined>
  shadowModelPath?: string
}

function fallbackModels(...paths: Array<string | undefined>): Array<string | undefined> {
  const artifacts = paths.filter(
    (path, index, values): path is string =>
      typeof path === 'string' && values.indexOf(path) === index,
  )
  return [...artifacts, undefined]
}

export function resolveInferenceExecution(
  deployment?: InferenceDeploymentPolicy,
): ResolvedInferenceExecution {
  if (
    !deployment ||
    deployment.mode === 'baseline' ||
    !deployment.promotionId ||
    !deployment.artifactPath
  ) {
    return { mode: 'baseline', fallbackModelPaths: [] }
  }
  if (deployment.mode === 'shadow') {
    return {
      mode: 'shadow',
      promotionId: deployment.promotionId,
      modelVersion: deployment.modelVersion,
      primaryModelPath: deployment.baselineArtifactPath,
      fallbackModelPaths: deployment.baselineArtifactPath
        ? fallbackModels(deployment.baselineFallbackArtifactPath)
        : [],
      shadowModelPath: deployment.artifactPath,
    }
  }
  return {
    mode: deployment.mode,
    promotionId: deployment.promotionId,
    modelVersion: deployment.modelVersion,
    primaryModelPath: deployment.artifactPath,
    fallbackModelPaths: fallbackModels(
      deployment.baselineArtifactPath,
      deployment.baselineFallbackArtifactPath,
    ),
  }
}

export type WorkerAnalysis = Omit<AnalysisDocument, 'version'> & {
  version: number
}

export interface WorkerProgress {
  progress: number
  status: Extract<
    AnalysisJob['status'],
    'acquiring' | 'separating' | 'transcribing' | 'mapping'
  >
  stageLabel: string
}

export function parseWorkerProgress(line: string): WorkerProgress | null {
  try {
    const message = JSON.parse(line) as {
      type?: string
      progress?: number
      status?: WorkerProgress['status']
      stageLabel?: string
    }
    if (
      message.type !== 'gts-progress' ||
      !Number.isFinite(message.progress) ||
      !['acquiring', 'separating', 'transcribing', 'mapping'].includes(
        message.status ?? '',
      ) ||
      typeof message.stageLabel !== 'string'
    ) {
      return null
    }
    return {
      progress: Math.max(0, Math.min(99, message.progress!)),
      status: message.status!,
      stageLabel: message.stageLabel,
    }
  } catch {
    return null
  }
}

export function describeWorkerFailure(
  stderr: string,
  code: number | null,
): string {
  if (
    /no space left on device|errno\s*28|可用磁盘空间不足/iu.test(stderr)
  ) {
    return '可用磁盘空间不足，无法完成音轨分离。请释放更多磁盘空间后重试。'
  }
  const detail =
    stderr
      .split(/\r?\n/u)
      .reverse()
      .map((line) =>
        /^(?:RuntimeError|ValueError|FileNotFoundError|OSError):\s*(.+)$/u.exec(
          line.trim(),
        )?.[1]?.trim(),
      )
      .find(Boolean) ?? ''
  if (/command failed.*demucs\.separate/iu.test(detail)) {
    return '音轨分离进程异常退出。请确认磁盘空间充足后重试。'
  }
  if (detail && !/[\\/](?:Users|home|var|tmp|private)[\\/]/u.test(detail)) {
    return detail.slice(0, 500)
  }
  return `媒体分析进程异常退出${code === null ? '' : `（代码 ${code}）`}`
}

const projectRoot = fileURLToPath(new URL('../../../', import.meta.url))
const DEFAULT_INFERENCE_TIMEOUT_MS = 2 * 60 * 60 * 1000

function executableFile(path: string): boolean {
  try {
    accessSync(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

function getWorkerCommand(): string | null {
  const configured = process.env.GTS_WORKER_COMMAND
  if (configured && executableFile(configured)) {
    return configured
  }
  const bundled = resolve(projectRoot, '.venv', 'bin', 'guitar-tab-worker')
  return executableFile(bundled) ? bundled : null
}

function getFfmpegCommand(): string | null {
  const configured = process.env.FFMPEG_BINARY
  if (configured && executableFile(configured)) {
    return configured
  }
  const bundled = resolve(projectRoot, 'node_modules', 'ffmpeg-static', 'ffmpeg')
  return executableFile(bundled) ? bundled : commandAvailable('ffmpeg') ? 'ffmpeg' : null
}

function getYtDlpCommand(): string | null {
  const bundled = resolve(projectRoot, '.venv', 'bin', 'yt-dlp')
  return executableFile(bundled) ? bundled : commandAvailable('yt-dlp') ? 'yt-dlp' : null
}

export const platforms: PlatformDescriptor[] = [
  {
    id: 'bilibili',
    label: '哔哩哔哩',
    hosts: ['bilibili.com', 'b23.tv', 'bili2233.cn'],
    acquisition: 'public-media',
    note: '支持提取器能够直接读取的音视频页面。',
  },
  {
    id: 'douyin',
    label: '抖音',
    hosts: ['douyin.com', 'iesdouyin.com'],
    acquisition: 'public-media',
    note: '支持提取器能够直接读取的公开视频页面。',
  },
  {
    id: 'netease',
    label: '网易云音乐',
    hosts: ['music.163.com', 'y.music.163.com', '163cn.tv', '163.fm'],
    acquisition: 'public-media',
    note: '支持可直接返回标准音频流的歌曲链接。',
  },
  {
    id: 'qqmusic',
    label: 'QQ 音乐',
    hosts: ['y.qq.com', 'c.y.qq.com'],
    acquisition: 'public-media',
    note: '支持可直接返回标准音频流的歌曲链接；登录曲目需配置会话。',
  },
  {
    id: 'youtube',
    label: 'YouTube',
    hosts: ['youtube.com', 'youtu.be', 'youtube-nocookie.com'],
    acquisition: 'public-media',
    note: '支持视频、Shorts 与音乐链接；登录或风控页面需配置会话。',
  },
  {
    id: 'vimeo',
    label: 'Vimeo',
    hosts: ['vimeo.com'],
    acquisition: 'public-media',
    note: '支持无需登录即可访问的公开视频。',
  },
  {
    id: 'soundcloud',
    label: 'SoundCloud',
    hosts: ['soundcloud.com', 'on.soundcloud.com'],
    acquisition: 'public-media',
    note: '支持提取器能够直接读取的曲目、集合与播放列表。',
  },
  {
    id: 'bandcamp',
    label: 'Bandcamp',
    hosts: ['bandcamp.com'],
    acquisition: 'public-media',
    note: '支持创作者公开提供试听的曲目与专辑。',
  },
  {
    id: 'jamendo',
    label: 'Jamendo',
    hosts: ['jamendo.com', 'jamen.do'],
    acquisition: 'public-media',
    note: '支持可直接返回标准媒体流的曲目。',
  },
  {
    id: 'audius',
    label: 'Audius',
    hosts: ['audius.co'],
    acquisition: 'public-media',
    note: '支持公开曲目、专辑和播放列表。',
  },
  {
    id: 'internet-archive',
    label: 'Internet Archive',
    hosts: ['archive.org'],
    acquisition: 'public-media',
    note: '支持可直接访问的音视频馆藏。',
  },
  {
    id: 'wikimedia',
    label: 'Wikimedia Commons',
    hosts: ['commons.wikimedia.org', 'upload.wikimedia.org'],
    acquisition: 'public-media',
    note: '支持媒体页面与文件直链。',
  },
]

const unsupportedMusicHosts = [
  'qishui.douyin.com',
  'music.douyin.com',
  'qishui.com',
]

export function normalizeSharedUrl(input: string): string {
  const normalized = input.trim().replaceAll('&amp;', '&')
  const matched = normalized.match(/https?:\/\/[^\s<>"']+/i)?.[0] ?? normalized
  const cleaned = matched.replace(/[，。；、）)\]}]+$/u, '')
  const withProtocol = /^https?:\/\//i.test(cleaned) ? cleaned : `https://${cleaned}`
  const parsed = new URL(withProtocol)
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('仅支持 HTTP 或 HTTPS 链接')
  }
  if (
    parsed.hostname.toLowerCase() === 'y.qq.com' &&
    parsed.pathname.startsWith('/n/ryqq_v2/songDetail/')
  ) {
    parsed.pathname = parsed.pathname.replace(
      '/n/ryqq_v2/songDetail/',
      '/n/ryqq/songDetail/',
    )
  }
  return parsed.toString()
}

export function detectPlatform(rawUrl: string): PlatformDescriptor | null {
  let host: string

  try {
    host = new URL(normalizeSharedUrl(rawUrl)).hostname.toLowerCase()
  } catch {
    return null
  }
  if (
    unsupportedMusicHosts.some(
      (candidate) => host === candidate || host.endsWith(`.${candidate}`),
    )
  ) {
    return null
  }

  const matches = platforms.flatMap((platform) =>
    platform.hosts
      .filter((candidate) => host === candidate || host.endsWith(`.${candidate}`))
      .map((candidate) => ({ platform, specificity: candidate.length })),
  )
  matches.sort((left, right) => right.specificity - left.specificity)
  return matches[0]?.platform ?? null
}

function commandAvailable(command: string): boolean {
  const result = spawnSync('sh', ['-lc', `command -v ${command}`], {
    encoding: 'utf8',
    timeout: 1500,
  })
  return result.status === 0 && result.stdout.trim().length > 0
}

export function getCapabilities(): CapabilityState {
  const ffmpeg = Boolean(getFfmpegCommand())
  const ffprobe = commandAvailable('ffprobe')
  const ytDlp = Boolean(getYtDlpCommand())
  const worker = Boolean(getWorkerCommand() && ffmpeg)

  return {
    ffmpeg,
    ffprobe,
    ytDlp,
    worker,
    localAuthorizedMediaDecoder: localAuthorizedMediaDecoderAvailable(),
    mode: worker ? 'inference' : ffmpeg || ytDlp ? 'hybrid' : 'demo',
  }
}

export function createStudioProject(
  source: SourceDescriptor,
  title =
    builtinDemoAnalysis.metadata?.title ??
    '国际歌',
): StudioProject {
  const analysis = builtinDemoAnalysis
  const tab = tabFromWorker(analysis)
  tab.capo = analysis.capo ?? 0
  tab.measures.forEach((measure, index) => {
    measure.section =
      analysis.pickup_beats && index === 0
        ? '弱起'
        : index === (analysis.pickup_beats ? 1 : 0)
          ? '前奏'
          : undefined
  })
  const guitarArrangement = guitarPartsFromWorker(analysis, tab)
  for (const part of guitarArrangement.guitarParts) {
    part.tab.capo = analysis.capo ?? 0
    part.tab.measures.forEach((measure, index) => {
      measure.section =
        analysis.pickup_beats && index === 0
          ? '弱起'
          : index === (analysis.pickup_beats ? 1 : 0)
            ? '前奏'
            : undefined
    })
  }
  const duration =
    analysis.metadata?.duration ??
    tab.measures.reduce(
      (maximum, measure) =>
        Math.max(maximum, measure.start + measure.duration),
      0,
    )
  const separatedGuitars = confirmedGuitarTracks(analysis)
  return {
    schemaVersion: STUDIO_PROJECT_SCHEMA_VERSION,
    id: randomUUID(),
    title,
    artist: analysis.metadata?.artist ?? '未知艺术家',
    source,
    bpm: analysis.bpm,
    key: analysis.key ?? 'Unknown',
    duration,
    createdAt: new Date().toISOString(),
    analysisMode: 'demo',
    transcriptionModel: analysis.transcription_model,
    techniqueModel: analysis.technique_model
      ? {
          ...analysis.technique_model,
          labels: [
            ...new Set([
              ...analysis.technique_model.labels,
              'strum-up' as const,
              'strum-down' as const,
            ]),
          ],
        }
      : {
          name: 'ordered-string-attacks',
          version: 'heuristic-v1',
          kind: 'heuristic',
          labels: ['unknown', 'strum-up', 'strum-down'],
        },
    visualization: analysis.visualization,
    tracks: [
      ['vocals', '人声', '#ef7d5f'],
      ['guitar', '吉他', '#19c37d'],
      ['bass', '贝斯', '#4d96ff'],
      ['drums', '鼓组', '#f2b84b'],
      ['piano', '键盘', '#9673d3'],
      ['other', '其他', '#8d92a1'],
    ].flatMap(([role, name, color]) => {
      if (role === 'guitar' && separatedGuitars.length > 1) {
        return separatedGuitars.map((part, index) => ({
          id: part.id,
          role: 'guitar' as const,
          name: part.name,
          color: ['#19c37d', '#d97706', '#2563eb'][index % 3],
          url: `/media/builtin/${part.path}`,
          waveform: part.waveform ?? [],
          level: part.level,
          available: true,
        }))
      }
      const filename =
        builtinDemoStemFilenames[role as keyof typeof builtinDemoStemFilenames]
      return [{
        id: role,
        role: role as TrackRole,
        name,
        color,
        url: filename ? `/media/builtin/${filename}` : null,
        waveform: analysis.waveforms?.[role as TrackRole] ?? [],
        level: analysis.levels?.[role as TrackRole],
        available: Boolean(filename),
      }]
    }),
    tab: guitarArrangement.primaryTab,
    guitarParts: guitarArrangement.guitarParts,
    primaryGuitarPartId: guitarArrangement.primaryGuitarPartId,
    warnings: analysis.warnings,
  }
}

export function tabFromWorker(
  analysis: WorkerAnalysis,
  guitarPart?: NonNullable<WorkerAnalysis['guitar_parts']>[number],
): StudioProject['tab'] {
  const sourceNotes = guitarPart?.notes ?? analysis.notes
  const sourceChords = guitarPart?.chords ?? analysis.chords
  const noteIdPrefix = guitarPart?.id ?? 'worker'
  const bpm = analysis.bpm > 0 ? analysis.bpm : 120
  const timeSignature: StudioProject['tab']['timeSignature'] =
    analysis.time_signature ?? [4, 4]
  const beatsPerMeasure = timeSignature[0]
  const beatDuration = 60 / bpm
  const measureDuration = beatDuration * beatsPerMeasure
  const noteDuration = sourceNotes.reduce((maximum, note) => Math.max(maximum, note.end), 0)
  const chordDuration = sourceChords.reduce(
    (maximum, chord) => Math.max(maximum, chord.start + chord.duration),
    0,
  )
  const duration = Math.max(
    noteDuration,
    chordDuration,
    analysis.metadata?.duration ?? 0,
    measureDuration,
  )
  const detectedBeats = (analysis.beats ?? [])
    .filter((beat) => Number.isFinite(beat) && beat >= 0)
    .sort((left, right) => left - right)
    .filter((beat, index, beats) => index === 0 || beat - beats[index - 1] > 0.05)
  const beatStarts =
    detectedBeats.length >= 2
      ? [...detectedBeats]
      : Array.from(
          { length: Math.ceil(duration / beatDuration) + 5 },
          (_, index) => index * beatDuration,
        )
  const pickupBeats =
    detectedBeats.length >= 2 &&
    Number.isInteger(analysis.pickup_beats) &&
    (analysis.pickup_beats ?? 0) > 0
      ? Math.min(beatsPerMeasure - 1, analysis.pickup_beats ?? 0)
      : 0
  while (
    beatStarts.length < 5 ||
    beatStarts.at(-1)! <= duration + measureDuration
  ) {
    const previous = beatStarts.at(-1) ?? 0
    const recentBeats = beatStarts.slice(-9)
    const recentIntervals = recentBeats
      .slice(1)
      .map((beat, index) => beat - recentBeats[index])
      .filter((interval) => interval > 0.2 && interval < 1.5)
      .sort((left, right) => left - right)
    const extrapolatedDuration =
      recentIntervals[Math.floor(recentIntervals.length / 2)] ?? beatDuration
    beatStarts.push(previous + extrapolatedDuration)
  }
  const measureBeatRanges: Array<{
    firstBeatIndex: number
    beatCount: number
  }> = []
  if (pickupBeats > 0 && beatStarts[0] < duration - 0.05) {
    measureBeatRanges.push({
      firstBeatIndex: 0,
      beatCount: pickupBeats,
    })
  }
  for (
    let firstBeatIndex = pickupBeats;
    beatStarts[firstBeatIndex] < duration - 0.05;
    firstBeatIndex += beatsPerMeasure
  ) {
    measureBeatRanges.push({
      firstBeatIndex,
      beatCount: beatsPerMeasure,
    })
  }
  if (measureBeatRanges.length === 0) {
    measureBeatRanges.push({
      firstBeatIndex: 0,
      beatCount: beatsPerMeasure,
    })
  }
  const measures: TabMeasure[] = measureBeatRanges.map((range, measureIndex) => {
    const { firstBeatIndex, beatCount } = range
    const start = beatStarts[firstBeatIndex]
    const end = beatStarts[firstBeatIndex + beatCount]
    const chord =
      sourceChords.find(
        (candidate) => candidate.start <= start && candidate.start + candidate.duration > start,
      )?.chord ?? 'N.C.'

    return {
      number: measureIndex + 1,
      start,
      duration: end - start,
      chord,
      section:
        measureIndex === 0 && pickupBeats > 0
          ? '弱起'
          : measureIndex === (pickupBeats > 0 ? 1 : 0)
            ? '前奏'
            : (measureIndex - (pickupBeats > 0 ? 1 : 0)) % 16 === 0
              ? `段落 ${Math.floor((measureIndex - (pickupBeats > 0 ? 1 : 0)) / 16) + 1}`
            : undefined,
      beats: Array.from({ length: beatCount }, (_, beatIndex): TabBeat => ({
        at: beatStarts[firstBeatIndex + beatIndex],
        duration:
          beatStarts[firstBeatIndex + beatIndex + 1] -
          beatStarts[firstBeatIndex + beatIndex],
        style: 'pick',
        notes: [],
      })),
    }
  })
  const beatSlots = measureBeatRanges.flatMap((range, measureIndex) =>
    Array.from({ length: range.beatCount }, (_, beatIndex) => ({
      globalBeatIndex: range.firstBeatIndex + beatIndex,
      measureIndex,
      beatIndex,
    })),
  )

  const beatIndexAt = (at: number) => {
    let beatLowerBound = 0
    let beatUpperBound = beatStarts.length
    while (beatLowerBound < beatUpperBound) {
      const middle = Math.floor((beatLowerBound + beatUpperBound) / 2)
      if (beatStarts[middle] <= at) {
        beatLowerBound = middle + 1
      } else {
        beatUpperBound = middle
      }
    }
    return Math.max(
      0,
      Math.min(beatSlots.length - 1, beatLowerBound - 1),
    )
  }
  const beatDurationAt = (at: number) => {
    const index = beatIndexAt(at)
    return beatStarts[index + 1] - beatStarts[index]
  }
  const notes: TabBeat['notes'] = sourceNotes.map((workerNote, index) => {
    const techniqueCandidates: TechniqueCandidate[] =
      workerNote.technique_candidates?.map((candidate) => ({
        technique: candidate.technique,
        confidence: candidate.confidence,
        evidence: candidate.evidence,
      })) ?? []
    return {
      id: `${noteIdPrefix}-${index + 1}`,
      string: workerNote.string,
      fret: workerNote.fret,
      at: workerNote.start,
      notationAt: workerNote.start,
      duration: Math.max(0.04, workerNote.end - workerNote.start),
      technique: workerNote.technique,
      harmonicType: workerNote.harmonic_type ?? undefined,
      harmonicTouchFret: workerNote.harmonic_touch_fret ?? undefined,
      confidence: workerNote.confidence,
      positionConfidence:
        workerNote.position_confidence ?? workerNote.confidence,
      positionSource: workerNote.position_source ?? 'legacy-greedy',
      techniqueConfidence:
        workerNote.technique_confidence ??
        techniqueCandidates[0]?.confidence ??
        workerNote.confidence,
      techniqueSource: workerNote.technique_source ?? 'legacy-heuristic',
      techniqueEvidence:
        workerNote.technique_evidence ??
        techniqueCandidates.find(
          (candidate) => candidate.technique === workerNote.technique,
        )?.evidence ??
        [],
      techniqueCandidates,
      relatedNoteId:
        typeof workerNote.related_note_index === 'number'
          ? `${noteIdPrefix}-${workerNote.related_note_index + 1}`
          : undefined,
    }
  })

  const gestures = guitarGestures(notes, beatDurationAt)
  const gestureByNote = new Map(
    gestures.flatMap((gesture) => gesture.notes.map((note) => [note.id, gesture] as const)),
  )
  for (const note of notes) {
    const gesture = gestureByNote.get(note.id)
    const anchor = gesture?.at ?? note.at
    // Keep the performed timestamp immutable. Nearby quarter, eighth, triplet,
    // and sixteenth positions are cleaned only on the notation timeline.
    const performedNotationAt =
      gesture?.style === 'arpeggio' ? note.at : anchor
    if (gesture?.style === 'strum') {
      note.technique = gesture.direction === 'up' ? 'strum-up' : 'strum-down'
      note.techniqueConfidence = Math.min(0.82, ...gesture.notes.map((item) => item.confidence))
      note.techniqueSource = 'ordered-string-attacks-v1'
      note.techniqueEvidence = [
        `ordered-strings:${gesture.notes.map((item) => item.string).join(',')}`,
        `spread-ms:${Math.round((gesture.notes.at(-1)!.at - gesture.at) * 1000)}`,
      ]
    }
    const notation = notationPositionAt(performedNotationAt, beatStarts)
    const assignedIndex = Math.min(
      beatSlots.length - 1,
      notation.beatIndex,
    )
    note.notationAt = notation.notationAt
    note.notationBeat = notation.notationBeat
    const slot = beatSlots[assignedIndex]
    const beat = measures[slot.measureIndex].beats[slot.beatIndex]
    beat.notes.push(note)
  }
  measures.forEach((measure) => {
    measure.beats.forEach((beat) => {
      const gesture = gestureByNote.get(beat.notes[0]?.id)
      if (gesture && (gesture.style === 'arpeggio' || beat.notes.length === gesture.notes.length) &&
        beat.notes.every((note) => gestureByNote.get(note.id) === gesture)) {
        beat.style = gesture.style
        beat.direction = gesture.direction
      } else if (beat.notes.some((note) => note.technique === 'tremolo')) {
        beat.style = 'tremolo'
      }
    })
  })

  return {
    tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
    capo: 0,
    timeSignature,
    tempoMap: analysis.tempo_map
      ? {
          ...analysis.tempo_map,
          points: analysis.tempo_map.points.map((point) => ({ ...point })),
        }
      : tempoMapFromBeats(
          detectedBeats.length >= 2
            ? detectedBeats
            : beatStarts.filter((beat) => beat <= duration),
          bpm,
        ),
    vocalMelody: analysis.vocal_melody?.map((note, index) => ({
      id: `vocal-${index + 1}`,
      at: note.start,
      duration: Math.max(0.04, note.end - note.start),
      pitch: note.pitch,
      confidence: note.confidence,
    })),
    lyrics: analysis.lyrics?.map((lyric) => ({ ...lyric })),
    measures,
  }
}

function guitarPartsFromWorker(
  analysis: WorkerAnalysis,
  fallbackTab: StudioProject['tab'],
): {
  guitarParts: StudioProject['guitarParts']
  primaryGuitarPartId: string
  primaryTab: StudioProject['tab']
} {
  const guitarParts =
    analysis.guitar_parts?.map((part) => ({
      id: part.id,
      trackId: part.track_id,
      name: part.name,
      role: part.role,
      ...(part.role_confidence === undefined
        ? {}
        : { roleConfidence: part.role_confidence }),
      tab: tabFromWorker(analysis, part),
      transcriptionModel:
        part.transcription_model ?? analysis.transcription_model,
    })) ?? []
  if (guitarParts.length === 0) {
    const track =
      analysis.stem_tracks?.find((candidate) => candidate.role === 'guitar')
    const id = track?.id ?? 'guitar-1'
    guitarParts.push({
      id,
      trackId: id,
      name: track?.name ?? '吉他',
      role: 'unknown',
      tab: fallbackTab,
      transcriptionModel: analysis.transcription_model,
    })
  }
  const primary =
    guitarParts
      .filter(
        (part) =>
          part.role === 'lead' && (part.roleConfidence ?? 0) >= 0.6,
      )
      .sort(
        (left, right) =>
          (right.roleConfidence ?? 0) - (left.roleConfidence ?? 0),
      )[0] ??
    guitarParts.reduce((best, part) => {
      const noteCount = part.tab.measures.reduce(
        (total, measure) =>
          total +
          measure.beats.reduce(
            (beatTotal, beat) => beatTotal + beat.notes.length,
            0,
          ),
        0,
      )
      const bestCount = best.tab.measures.reduce(
        (total, measure) =>
          total +
          measure.beats.reduce(
            (beatTotal, beat) => beatTotal + beat.notes.length,
            0,
          ),
        0,
      )
      return noteCount > bestCount ? part : best
    }, guitarParts[0])
  return {
    guitarParts,
    primaryGuitarPartId: primary.id,
    primaryTab: primary.tab,
  }
}

function confirmedGuitarTracks(analysis: WorkerAnalysis) {
  const transcribedTrackIds = new Set(
    (analysis.guitar_parts ?? []).map((part) => part.track_id),
  )
  return (analysis.stem_tracks ?? []).filter(
    (track) =>
      track.role === 'guitar' && transcribedTrackIds.has(track.id),
  )
}

export function workerProject(
  analysis: WorkerAnalysis,
  source: SourceDescriptor,
  jobId: string,
): StudioProject {
  const base = createStudioProject(
    source,
    source.filename?.replace(/\.[^.]+$/, '') ??
      (source.kind === 'upload' ? '未命名录音' : `${source.label} 导入`),
  )
  const tab = tabFromWorker(analysis)
  const guitarArrangement = guitarPartsFromWorker(analysis, tab)
  const duration =
    analysis.metadata?.duration ??
    tab.measures.reduce(
      (maximum, measure) =>
        Math.max(maximum, measure.start + measure.duration),
      0,
    )
  const dataRoot = runtimeDataRoot()
  const separatedGuitars = confirmedGuitarTracks(analysis)
  const baseTrackTemplates = base.tracks.filter(
    (track, index, tracks) =>
      tracks.findIndex((candidate) => candidate.role === track.role) === index,
  )
  const trackUrl = (path: string | undefined) => {
    const relativePath = path
      ? relative(dataRoot, path).split(sep).join('/')
      : ''
    return relativePath.length > 0 && !relativePath.startsWith('..')
      ? `/media/${relativePath}`
      : null
  }

  return {
    ...base,
    id: randomUUID(),
    title: analysis.metadata?.title ?? base.title,
    artist: analysis.metadata?.artist ?? base.artist,
    bpm: analysis.bpm,
    key: analysis.key ?? 'Unknown',
    duration,
    analysisMode: 'inference',
    transcriptionModel: analysis.transcription_model,
    techniqueModel: analysis.technique_model,
    provenance: undefined,
    visualization: analysis.visualization,
    tracks: baseTrackTemplates.flatMap((track) => {
      if (track.role === 'guitar' && separatedGuitars.length > 1) {
        return separatedGuitars.map((part, index) => {
          const url = trackUrl(part.path)
          return {
            ...track,
            id: part.id,
            name: part.name,
            color: ['#19c37d', '#d97706', '#2563eb'][index % 3],
            url,
            waveform: part.waveform ?? [],
            level: part.level,
            available: Boolean(url),
          }
        })
      }
      const stemPath = analysis.stems[track.role]
      const url = trackUrl(stemPath)
      const level = analysis.levels?.[track.role]
      return [{
        ...track,
        url,
        waveform: analysis.waveforms?.[track.role] ?? track.waveform,
        level,
        available: Boolean(url),
      }]
    }),
    tab: guitarArrangement.primaryTab,
    guitarParts: guitarArrangement.guitarParts,
    primaryGuitarPartId: guitarArrangement.primaryGuitarPartId,
    warnings: analysis.warnings,
    source: {
      ...source,
      label: `${source.label} · Worker ${jobId.slice(0, 6)}`,
    },
  }
}

export function workerSourceArguments(
  job: AnalysisJob,
  inputPath?: string,
): string[] {
  return inputPath
    ? ['--input', resolve(inputPath)]
    : job.source.url
      ? [
          '--url',
          job.source.url,
          ...(job.source.mediaSession
            ? [
                '--cookies-from-browser',
                job.source.mediaSession.browser,
              ]
            : []),
        ]
      : []
}

export function transcriptionWorkerArguments(
  environment: NodeJS.ProcessEnv = process.env,
): string[] {
  const requestedProvider = environment.GTS_TRANSCRIPTION_PROVIDER
  const provider = [
    'basic-pitch',
    'muscriptor',
    'guitar-specialist',
    'multitrack-research',
  ].includes(requestedProvider ?? '')
    ? requestedProvider!
    : 'basic-pitch'
  const model = environment.GTS_TRANSCRIPTION_MODEL?.trim()
  const variant = environment.GTS_TRANSCRIPTION_VARIANT
  return [
    '--transcription-provider',
    provider,
    ...((provider === 'muscriptor' ||
      provider === 'multitrack-research') &&
    model
      ? ['--transcription-model', model]
      : []),
    ...(provider === 'muscriptor' &&
    (variant === 'small' || variant === 'medium' || variant === 'large')
      ? ['--transcription-variant', variant]
      : []),
  ]
}

export async function executeWorker(
  workerCommand: string,
  ffmpegCommand: string,
  sourceArguments: string[],
  outputRoot: string,
  modelPath?: string,
  signal?: AbortSignal,
  timeoutMs = DEFAULT_INFERENCE_TIMEOUT_MS,
  onProgress?: (progress: WorkerProgress) => void,
): Promise<number> {
  if (signal?.aborted) throw new Error('任务已取消')
  await mkdir(outputRoot, { recursive: true })
  const startedAt = Date.now()
  const argumentsList = [
    ...sourceArguments,
    '--output',
    outputRoot,
    ...transcriptionWorkerArguments(),
    ...(modelPath ? ['--tab-model', modelPath] : []),
  ]
  await new Promise<void>((resolvePromise, rejectPromise) => {
    const environment: NodeJS.ProcessEnv = {
      ...process.env,
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
      FFMPEG_BINARY: ffmpegCommand,
      PATH: `${dirname(ffmpegCommand)}${delimiter}${process.env.PATH ?? ''}`,
      XDG_CACHE_HOME:
        process.env.XDG_CACHE_HOME ?? resolve(projectRoot, '.cache'),
      TORCH_HOME:
        process.env.TORCH_HOME ?? resolve(projectRoot, '.cache', 'torch'),
      HF_HUB_CACHE:
        process.env.HF_HUB_CACHE ??
        resolve(projectRoot, '.cache', 'huggingface', 'hub'),
    }
    delete environment.STRINGTRACE_TAB_MODEL
    const processHandle = spawn(workerCommand, argumentsList, {
      cwd: process.cwd(),
      env: environment,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let timedOut = false
    let aborted = false
    let killTimer: NodeJS.Timeout | undefined
    const terminate = () => {
      if (processHandle.exitCode !== null || processHandle.pid === undefined) return
      const target = process.platform === 'win32' ? processHandle.pid : -processHandle.pid
      try {
        process.kill(target, 'SIGTERM')
      } catch {
        processHandle.kill('SIGTERM')
      }
      killTimer = setTimeout(() => {
        try {
          process.kill(target, 'SIGKILL')
        } catch {
          processHandle.kill('SIGKILL')
        }
      }, 5000)
    }
    const abortHandler = () => {
      aborted = true
      terminate()
    }
    signal?.addEventListener('abort', abortHandler, { once: true })
    if (signal?.aborted) abortHandler()
    const timeout = setTimeout(() => {
      timedOut = true
      terminate()
    }, Math.max(1000, timeoutMs))
    const cleanup = () => {
      clearTimeout(timeout)
      if (killTimer) clearTimeout(killTimer)
      signal?.removeEventListener('abort', abortHandler)
    }
    let stdout = ''
    const stdoutDecoder = new StringDecoder('utf8')
    processHandle.stdout.on('data', (chunk: Buffer) => {
      stdout += stdoutDecoder.write(chunk)
      const lines = stdout.split(/\r?\n/u)
      stdout = lines.pop() ?? ''
      for (const line of lines) {
        const progress = parseWorkerProgress(line)
        if (progress) onProgress?.(progress)
      }
    })
    let stderr = ''
    const stderrDecoder = new StringDecoder('utf8')
    processHandle.stderr.on('data', (chunk: Buffer) => {
      stderr = `${stderr}${stderrDecoder.write(chunk)}`.slice(-1200)
    })
    processHandle.on('error', (error) => {
      cleanup()
      rejectPromise(error)
    })
    processHandle.on('close', (code) => {
      cleanup()
      stdout += stdoutDecoder.end()
      stderr = `${stderr}${stderrDecoder.end()}`.slice(-1200)
      if (stdout) {
        const progress = parseWorkerProgress(stdout)
        if (progress) onProgress?.(progress)
      }
      if (code === 0) {
        resolvePromise()
      } else {
        rejectPromise(
          new Error(
            aborted
              ? '任务已取消'
              : timedOut
                ? '推理任务超时'
                : describeWorkerFailure(stderr, code),
          ),
        )
      }
    })
  })
  return Date.now() - startedAt
}

function reportInferenceObservation(
  options: PipelineExecutionOptions,
  observation: Omit<InferenceObservation, 'id' | 'recordedAt'>,
): void {
  try {
    options.onInferenceObservation?.(observation)
  } catch {
    // Operational telemetry must not change the user-visible inference result.
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error('任务已取消')
}

async function runShadowInference(
  job: AnalysisJob,
  sourceArguments: string[],
  baseline: WorkerAnalysis,
  workerCommand: string,
  ffmpegCommand: string,
  deployment: ResolvedInferenceExecution,
  options: PipelineExecutionOptions,
): Promise<void> {
  if (
    deployment.mode !== 'shadow' ||
    !deployment.promotionId ||
    !deployment.shadowModelPath
  ) {
    return
  }
  const outputRoot = resolve(
    runtimeDataRoot(),
    'shadow',
    job.id,
    deployment.promotionId,
  )
  const startedAt = Date.now()
  try {
    const durationMs = await executeWorker(
      workerCommand,
      ffmpegCommand,
      sourceArguments,
      outputRoot,
      deployment.shadowModelPath,
      options.signal,
      options.timeoutMs,
    )
    const candidate = parseAnalysisDocument(
      JSON.parse(await readFile(resolve(outputRoot, 'analysis.json'), 'utf8')),
    )
    reportInferenceObservation(options, {
      promotionId: deployment.promotionId,
      jobId: job.id,
      mode: 'shadow',
      success: true,
      fallbackUsed: false,
      durationMs,
      baselineNoteCount: baseline.notes.length,
      candidateNoteCount: candidate.notes.length,
      noteCountDelta: candidate.notes.length - baseline.notes.length,
    })
  } catch (error) {
    reportInferenceObservation(options, {
      promotionId: deployment.promotionId,
      jobId: job.id,
      mode: 'shadow',
      success: false,
      fallbackUsed: false,
      durationMs: Date.now() - startedAt,
      baselineNoteCount: baseline.notes.length,
      error: error instanceof Error ? error.message : '影子推理失败',
    })
  } finally {
    await rm(outputRoot, { recursive: true, force: true }).catch(() => undefined)
  }
}

async function runPreparedInferencePipeline(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  inputPath?: string,
  options: PipelineExecutionOptions = {},
): Promise<StudioProject> {
  const workerCommand = getWorkerCommand()
  const ffmpegCommand = getFfmpegCommand()
  if (!workerCommand) {
    throw new Error('真实分析 Worker 尚未安装或不可执行')
  }
  if (!ffmpegCommand) {
    throw new Error('FFmpeg 尚未安装或不可执行')
  }
  throwIfAborted(options.signal)
  const configuredModelPath =
    process.env.STRINGTRACE_TAB_MODEL?.trim() || undefined

  const outputRoot = resolve(runtimeDataRoot(), 'jobs', job.id)
  const deployment = resolveInferenceExecution(options.deployment)
  Object.assign(job, {
    status: 'acquiring',
    progress: 1,
    stageLabel: inputPath ? '准备分析本地媒体' : '准备获取在线媒体',
    deployment: {
      mode: deployment.mode,
      modelVersion: deployment.modelVersion,
      fallbackUsed: false,
    },
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  throwIfAborted(options.signal)

  const sourceArguments = workerSourceArguments(job, inputPath)
  if (sourceArguments.length === 0) {
    throw new Error('推理任务缺少媒体输入')
  }

  const primaryStartedAt = Date.now()
  const reportWorkerProgress = (progress: WorkerProgress) => {
    if (progress.progress < job.progress) return
    Object.assign(job, {
      ...progress,
      updatedAt: new Date().toISOString(),
    })
    onUpdate(job)
  }
  let analysis: WorkerAnalysis
  try {
    await executeWorker(
      workerCommand,
      ffmpegCommand,
      sourceArguments,
      outputRoot,
      deployment.primaryModelPath ?? configuredModelPath,
      options.signal,
      options.timeoutMs,
      reportWorkerProgress,
    )
    analysis = parseAnalysisDocument(
      JSON.parse(await readFile(resolve(outputRoot, 'analysis.json'), 'utf8')),
    )
    if (
      deployment.mode === 'canary' ||
      deployment.mode === 'champion'
    ) {
      reportInferenceObservation(options, {
        promotionId: deployment.promotionId as string,
        jobId: job.id,
        mode: deployment.mode,
        success: true,
        fallbackUsed: false,
        durationMs: Date.now() - primaryStartedAt,
      })
    }
  } catch (error) {
    if (options.signal?.aborted) throw error
    if (deployment.fallbackModelPaths.length === 0) {
      throw error
    }
    if (
      (deployment.mode === 'canary' || deployment.mode === 'champion') &&
      deployment.promotionId
    ) {
      reportInferenceObservation(options, {
        promotionId: deployment.promotionId,
        jobId: job.id,
        mode: deployment.mode,
        success: false,
        fallbackUsed: true,
        durationMs: Date.now() - primaryStartedAt,
        error: error instanceof Error ? error.message : '候选推理失败',
      })
    }
    let fallbackError: unknown = error
    let fallbackAnalysis: WorkerAnalysis | undefined
    for (const fallbackModelPath of deployment.fallbackModelPaths) {
      try {
        await rm(outputRoot, { recursive: true, force: true })
        await executeWorker(
          workerCommand,
          ffmpegCommand,
          sourceArguments,
          outputRoot,
          fallbackModelPath,
          options.signal,
          options.timeoutMs,
          reportWorkerProgress,
        )
        fallbackAnalysis = parseAnalysisDocument(
          JSON.parse(
            await readFile(resolve(outputRoot, 'analysis.json'), 'utf8'),
          ),
        )
        break
      } catch (cause) {
        fallbackError = cause
      }
    }
    if (!fallbackAnalysis) throw fallbackError
    analysis = fallbackAnalysis
    if (job.deployment) job.deployment.fallbackUsed = true
  }

  throwIfAborted(options.signal)
  Object.assign(job, {
    status: 'mapping',
    progress: Math.max(job.progress, 92),
    stageLabel: '整理分轨与 TAB',
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  const project = workerProject(analysis, job.source, job.id)
  throwIfAborted(options.signal)
  Object.assign(job, {
    status: 'completed',
    progress: 100,
    stageLabel: '分析完成',
    projectId: project.id,
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  if (deployment.mode === 'shadow' && !options.skipShadow) {
    void runShadowInference(
      job,
      sourceArguments,
      analysis,
      workerCommand,
      ffmpegCommand,
      deployment,
      options,
    )
  }
  return project
}

async function readMediaHeader(path: string): Promise<Buffer> {
  const handle = await open(path, 'r')
  const header = Buffer.alloc(16)
  try {
    const { bytesRead } = await handle.read(header, 0, header.length, 0)
    return header.subarray(0, bytesRead)
  } finally {
    await handle.close()
  }
}

async function acquireAuthorizedLink(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  options: PipelineExecutionOptions,
): Promise<{ root: string; path: string; name: string }> {
  const workerCommand = getWorkerCommand()
  const ffmpegCommand = getFfmpegCommand()
  if (!workerCommand || !ffmpegCommand || !job.source.url) {
    throw new Error('本地媒体提取器不可用，无法准备受保护链接')
  }
  const root = resolve(
    runtimeDataRoot(),
    'uploads',
    `${job.id}.authorized-source`,
  )
  Object.assign(job, {
    status: 'acquiring',
    progress: 1,
    stageLabel: '正在将在线媒体下载到本机临时目录',
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  await executeWorker(
    workerCommand,
    ffmpegCommand,
    [...workerSourceArguments(job), '--acquire-only'],
    root,
    undefined,
    options.signal,
    options.timeoutMs,
    (progress) => {
      Object.assign(job, {
        status: 'acquiring',
        progress: Math.max(1, Math.min(8, progress.progress * 0.08)),
        stageLabel: progress.stageLabel,
        updatedAt: new Date().toISOString(),
      })
      onUpdate(job)
    },
  )
  const name = (await readdir(root))
    .filter(
      (candidate) =>
        candidate.startsWith('source.') &&
        !candidate.endsWith('.json') &&
        !candidate.endsWith('.part') &&
        !candidate.endsWith('.ytdl'),
    )
    .sort()[0]
  if (!name) throw new Error('本地媒体提取器未生成可分析文件')
  return { root, path: resolve(root, name), name }
}

async function runInferencePipeline(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  inputPath?: string,
  options: PipelineExecutionOptions = {},
): Promise<StudioProject> {
  const authorization = job.source.authorizedMedia
  if (!authorization) {
    return runPreparedInferencePipeline(
      job,
      onUpdate,
      inputPath,
      options,
    )
  }

  let preparedPath = inputPath
  let preparedName = job.source.filename
  let acquiredRoot: string | undefined
  const decodedPath = resolve(
    runtimeDataRoot(),
    'uploads',
    `${job.id}.authorized-media`,
  )
  try {
    if (!preparedPath) {
      const acquired = await acquireAuthorizedLink(job, onUpdate, options)
      acquiredRoot = acquired.root
      preparedPath = acquired.path
      preparedName = acquired.name
    }
    if (!preparedName) preparedName = basename(preparedPath)
    const encryptedHeader = await readMediaHeader(preparedPath)
    const requiresLocalDecode = requiresAuthorizedMediaDecode(
      preparedName,
      encryptedHeader,
    )
    if (!requiresLocalDecode) {
      return await runPreparedInferencePipeline(
        job,
        onUpdate,
        preparedPath,
        { ...options, skipShadow: true },
      )
    }
    Object.assign(job, {
      status: 'validating',
      progress: Math.max(2, job.progress),
      stageLabel: '正在本机临时解码受保护媒体',
      updatedAt: new Date().toISOString(),
    })
    onUpdate(job)
    await decodeAuthorizedMediaLocally({
      provider: authorization.provider,
      outputPath: decodedPath,
      originalName: preparedName,
      inputPath: preparedPath,
      signal: options.signal,
    })
    const decodedHeader = await readMediaHeader(decodedPath)
    if (!detectStandardAudioSignature(decodedHeader)) {
      throw new Error('本地解码器未生成受支持的标准音频容器')
    }
    Object.assign(job, {
      progress: 4,
      stageLabel: '本地临时媒体已就绪，开始分析',
      updatedAt: new Date().toISOString(),
    })
    onUpdate(job)
    return await runPreparedInferencePipeline(
      job,
      onUpdate,
      decodedPath,
      { ...options, skipShadow: true },
    )
  } finally {
    await rm(decodedPath, { force: true }).catch(() => undefined)
    if (acquiredRoot) {
      await rm(acquiredRoot, { recursive: true, force: true }).catch(
        () => undefined,
      )
    }
  }
}

export function runPipeline(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  inputPath?: string,
  options: PipelineExecutionOptions = {},
): Promise<StudioProject> {
  const capabilities = getCapabilities()
  const platform = platforms.find((candidate) => candidate.id === job.source.kind)
  if (platform?.acquisition === 'metadata-only') {
    return Promise.reject(
      new Error(
        `${platform.label}链接当前只支持读取元数据，未实现音频流获取`,
      ),
    )
  }
  if (!capabilities.worker) {
    return Promise.reject(
      new Error('真实分析 Worker 尚未就绪，任务未执行；系统不会用示例结果替代'),
    )
  }
  return runInferencePipeline(job, onUpdate, inputPath, options).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : '媒体获取失败'
    if (!inputPath && platform) {
      const failure = classifyMediaAcquisitionFailure(platform, message)
      const wrapped = new Error(failure.message) as Error & {
        errorCode?: 'media-auth-required' | 'media-session-unavailable'
      }
      wrapped.errorCode = failure.errorCode
      throw wrapped
    }
    throw error
  })
}

export function classifyMediaAcquisitionFailure(
  platform: PlatformDescriptor,
  message: string,
): {
  message: string
  errorCode?: 'media-auth-required' | 'media-session-unavailable'
} {
  if (/drm|encrypted|encryption/iu.test(message)) {
    return {
      message: `${platform.label}返回了当前解码器不支持的 DRM 或加密媒体流`,
    }
  }
  if (
    /sign in|login|cookie|captcha|confirm you.re not a bot|http error (?:401|403|412|429)/iu.test(
      message,
    )
  ) {
    if (
      /could not (?:copy|find).*cookie|failed to decrypt|keyring|safe storage/iu.test(
        message,
      )
    ) {
      return {
        message: `${platform.label}需要登录，但无法读取或解密系统默认浏览器的 Cookie；请确认已登录并完全退出浏览器后重试`,
        errorCode: 'media-session-unavailable',
      }
    }
    return {
      message: `${platform.label}要求登录会话或人机验证；可在导入窗口使用系统默认浏览器的登录状态重试`,
      errorCode: 'media-auth-required',
    }
  }
  if (/unsupported url|no suitable extractor/iu.test(message)) {
    return {
      message: `${platform.label}链接格式暂未被当前提取器识别，请使用标准歌曲/视频详情页链接`,
    }
  }
  if (/unavailable|private|removed|geo|region|not available/iu.test(message)) {
    return { message: `${platform.label}媒体在当前账号或网络区域不可用` }
  }
  if (/requested format|no video formats|no formats/iu.test(message)) {
    return { message: `${platform.label}没有返回可解码的音频格式` }
  }
  if (
    /unable to download webpage|connection|network|timed out|timeout|ssl|certificate|proxy/iu.test(
      message,
    )
  ) {
    return {
      message: `${platform.label}媒体获取网络请求失败；请检查系统代理、TLS 证书、网络限制或稍后重试`,
    }
  }
  if (
    /extractor|signature|unable to extract|failed to parse|site changed/iu.test(
      message,
    )
  ) {
    return {
      message: `${platform.label}页面结构与当前提取器不兼容；请更新到包含新版 yt-dlp 的应用版本后重试`,
    }
  }
  const detail = message
    .replace(/[\r\n\t]+/gu, ' ')
    .replace(/[\\/](?:Users|home|var|tmp|private)[\\/]\S*/giu, '[local path]')
    .trim()
    .slice(0, 240)
  return {
    message: detail
      ? `${platform.label}媒体获取失败：${detail}`
      : `${platform.label}媒体获取失败；请检查链接状态、登录会话、网络限制或提取器版本`,
  }
}

export function describeMediaAcquisitionFailure(
  platform: PlatformDescriptor,
  message: string,
): string {
  return classifyMediaAcquisitionFailure(platform, message).message
}

export function sourceFromLink(url: string): SourceDescriptor {
  const normalizedUrl = normalizeSharedUrl(url)
  const platform = detectPlatform(normalizedUrl)
  if (!platform) {
    throw new Error(
      `当前支持以下在线来源：${platforms
        .map((candidate) => candidate.label)
        .join('、')}`,
    )
  }

  return {
    kind: platform.id as PlatformId,
    label: platform.label,
    url: normalizedUrl,
  }
}
