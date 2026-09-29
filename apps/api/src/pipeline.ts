import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { accessSync, constants } from 'node:fs'
import { mkdir, readFile } from 'node:fs/promises'
import { dirname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import type {
  AnalysisJob,
  CapabilityState,
  PlatformDescriptor,
  PlatformId,
  SourceDescriptor,
  StudioProject,
  TabBeat,
  TabMeasure,
  TabNote,
  Technique,
  TechniqueCandidate,
  TechniqueModelInfo,
  TranscriptionModelInfo,
  TrackRole,
  VisualizationPoint,
} from './types.js'

interface WorkerTechniqueCandidate {
  technique: Technique
  confidence: number
  evidence: string[]
}

interface WorkerNote {
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
  technique_candidates?: WorkerTechniqueCandidate[]
  related_note_index?: number | null
}

export interface WorkerAnalysis {
  version: number
  transcription_model?: TranscriptionModelInfo
  technique_model?: TechniqueModelInfo
  metadata?: {
    title?: string
    artist?: string
    duration?: number
    webpage_url?: string
  }
  bpm: number
  beats?: number[]
  onsets?: number[]
  stems: Partial<Record<TrackRole, string>>
  levels?: Partial<Record<TrackRole, number>>
  waveforms?: Partial<Record<TrackRole, number[]>>
  visualization?: VisualizationPoint[]
  notes: WorkerNote[]
  chords: Array<{ start: number; duration: number; chord: string }>
  warnings: string[]
}

const projectRoot = fileURLToPath(new URL('../../../', import.meta.url))

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
    note: '仅处理公开且用户有权使用的非 DRM 媒体。',
  },
  {
    id: 'douyin',
    label: '抖音',
    hosts: ['douyin.com', 'iesdouyin.com'],
    acquisition: 'public-media',
    note: '公开页面可交给已配置的媒体 worker，登录内容不抓取。',
  },
  {
    id: 'netease',
    label: '网易云音乐',
    hosts: ['music.163.com', 'y.music.163.com', '163cn.tv', '163.fm'],
    acquisition: 'public-media',
    note: '尝试处理无需登录即可取得的公开非 DRM 媒体，受版权保护的音频不抓取。',
  },
]

const unsupportedMusicHosts = [
  'y.qq.com',
  'c.y.qq.com',
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
    mode: worker ? 'inference' : ffmpeg || ytDlp ? 'hybrid' : 'demo',
  }
}

const chordShapes: Record<string, Array<[1 | 2 | 3 | 4 | 5 | 6, number]>> = {
  Em: [
    [6, 0],
    [5, 2],
    [4, 2],
    [3, 0],
    [2, 0],
    [1, 0],
  ],
  G: [
    [6, 3],
    [5, 2],
    [4, 0],
    [3, 0],
    [2, 0],
    [1, 3],
  ],
  C: [
    [5, 3],
    [4, 2],
    [3, 0],
    [2, 1],
    [1, 0],
  ],
  D: [
    [4, 0],
    [3, 2],
    [2, 3],
    [1, 2],
  ],
}

const techniqueSequence: Technique[] = [
  'pick',
  'hammer-on',
  'pull-off',
  'slide',
  'harmonic',
  'palm-mute',
  'dead-note',
  'slap',
  'tremolo',
  'body-tap',
]

function seededValue(seed: string, index: number): number {
  const digest = createHash('sha256').update(`${seed}:${index}`).digest()
  return digest[index % digest.length] / 255
}

function buildWaveform(role: TrackRole): number[] {
  return Array.from({ length: 120 }, (_, index) => {
    const base = seededValue(role, index)
    const pulse = index % 15 < 3 ? 0.28 : 0
    const roleScale = role === 'drums' ? 0.95 : role === 'guitar' ? 0.78 : 0.58
    return Math.min(1, 0.08 + (base * 0.58 + pulse) * roleScale)
  })
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function buildVisualization(pointCount = 720): VisualizationPoint[] {
  return Array.from({ length: pointCount }, (_, index) => {
    const beat = index % 90
    const impact = beat < 5 ? 0.82 * (1 - beat / 5) : 0
    const phrase = 0.35 + seededValue('visual', index) * 0.45
    const energy = clamp01(phrase + impact * 0.35)
    const pitch = clamp01(0.28 + seededValue('pitch', Math.floor(index / 12)) * 0.44)
    return {
      energy: Number(energy.toFixed(4)),
      pitch: Number(pitch.toFixed(4)),
      low: Number(clamp01(0.32 + impact * 0.62).toFixed(4)),
      mid: Number(clamp01(energy * 0.74).toFixed(4)),
      high: Number(clamp01(energy * 0.38).toFixed(4)),
      impact: Number(impact.toFixed(4)),
    }
  })
}

function buildMeasures(bpm: number): TabMeasure[] {
  const beatDuration = 60 / bpm
  const measureDuration = beatDuration * 4
  const progression = ['Em', 'G', 'C', 'D', 'Em', 'G', 'C', 'D']
  const lyrics = [
    '沿着节拍 慢慢靠近',
    '听见琴弦 清楚回应',
    '每个和弦 留下位置',
    '下一小节 继续前行',
    '拨动旋律 跟随呼吸',
    '扫过节奏 保持稳定',
    '反复练习 记住声音',
    '回到开头 再弹一次',
  ]

  return progression.map((chord, measureIndex) => {
    const shape = chordShapes[chord]
    const beats = Array.from({ length: 4 }, (_, beatIndex): TabBeat => {
      const direction: 'down' | 'up' = beatIndex % 2 === 0 ? 'down' : 'up'
      const picked = beatIndex < 2
      const selected = picked
        ? [shape[(beatIndex * 2) % shape.length], shape[(beatIndex * 2 + 2) % shape.length]]
        : shape
      const notes: TabNote[] = selected.map(([string, fret], noteIndex) => {
        const technique =
          picked && noteIndex === selected.length - 1
            ? techniqueSequence[(measureIndex + beatIndex) % techniqueSequence.length]
            : direction === 'down'
              ? 'strum-down'
              : 'strum-up'
        const harmonicType =
          technique === 'harmonic'
            ? measureIndex % 2 === 0
              ? 'natural'
              : 'artificial'
            : undefined

        return {
          id: `m${measureIndex + 1}-b${beatIndex + 1}-n${noteIndex + 1}`,
          string,
          fret:
            technique === 'harmonic'
              ? harmonicType === 'natural'
                ? 12
                : 7
              : fret,
          at: measureIndex * measureDuration + beatIndex * beatDuration,
          duration: picked ? beatDuration * 0.72 : beatDuration * 0.86,
          technique,
          harmonicType,
          harmonicTouchFret:
            harmonicType === 'artificial' ? 19 : undefined,
          confidence: Number((0.68 + seededValue(chord, measureIndex * 12 + beatIndex + noteIndex) * 0.28).toFixed(2)),
          positionConfidence: Number(
            (
              0.64 +
              seededValue(
                `position-${chord}`,
                measureIndex * 12 + beatIndex + noteIndex,
              ) *
                0.3
            ).toFixed(2),
          ),
          positionSource:
            technique === 'harmonic'
              ? 'harmonic-optimizer-v1'
              : 'playable-optimizer-v2',
          techniqueConfidence: Number(
            (
              0.62 +
              seededValue(
                `technique-${chord}`,
                measureIndex * 12 + beatIndex + noteIndex,
              ) *
                0.3
            ).toFixed(2),
          ),
          techniqueSource: 'demo',
          techniqueEvidence: ['demo-fixture'],
          techniqueCandidates: [
            {
              technique,
              confidence: Number(
                (
                  0.62 +
                  seededValue(
                    `technique-${chord}`,
                    measureIndex * 12 + beatIndex + noteIndex,
                  ) *
                    0.3
                ).toFixed(2),
              ),
              evidence: ['demo-fixture'],
            },
          ],
        }
      })

      return {
        at: measureIndex * measureDuration + beatIndex * beatDuration,
        duration: beatDuration,
        direction,
        style:
          notes.some((note) => note.technique === 'tremolo')
            ? 'tremolo'
            : !picked && measureIndex % 4 === 1 && beatIndex === 2
            ? 'arpeggio'
            : !picked && measureIndex % 4 === 2 && beatIndex === 3
              ? 'rasgueado'
              : !picked
                ? 'strum'
                : 'pick',
        notes,
      }
    })

    return {
      number: measureIndex + 1,
      start: measureIndex * measureDuration,
      duration: measureDuration,
      chord,
      section: measureIndex === 0 ? '前奏' : measureIndex === 4 ? '主歌' : undefined,
      lyric: lyrics[measureIndex],
      beats,
    }
  })
}

export function createStudioProject(
  source: SourceDescriptor,
  title = '午夜练习段落',
): StudioProject {
  const bpm = 96
  const measures = buildMeasures(bpm)
  const duration = measures.reduce((sum, measure) => sum + measure.duration, 0)
  return {
    id: randomUUID(),
    title,
    artist: '本地分析',
    source,
    bpm,
    key: 'E minor',
    duration,
    createdAt: new Date().toISOString(),
    analysisMode: 'demo',
    techniqueModel: {
      name: 'stringtrace-demo-fixture',
      version: 'demo-1',
      kind: 'heuristic',
      labels: techniqueSequence,
    },
    visualization: buildVisualization(),
    tracks: [
      ['vocals', '人声', '#ef7d5f'],
      ['guitar', '吉他', '#19c37d'],
      ['bass', '贝斯', '#4d96ff'],
      ['drums', '鼓组', '#f2b84b'],
      ['piano', '键盘', '#9673d3'],
      ['other', '其他', '#8d92a1'],
    ].map(([role, name, color]) => ({
      id: role,
      role: role as TrackRole,
      name,
      color,
      url: null,
      waveform: buildWaveform(role as TrackRole),
      available: ['guitar', 'bass', 'drums'].includes(role),
    })),
    tab: {
      tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
      capo: 0,
      measures,
    },
    warnings: [
      '当前为演示识别结果；配置 GTS_WORKER_COMMAND 后启用真实模型。',
      '演奏技法是带置信度的建议，需要人工校对。',
    ],
  }
}

export function tabFromWorker(analysis: WorkerAnalysis): StudioProject['tab'] {
  const bpm = analysis.bpm > 0 ? analysis.bpm : 120
  const beatDuration = 60 / bpm
  const measureDuration = beatDuration * 4
  const noteDuration = analysis.notes.reduce((maximum, note) => Math.max(maximum, note.end), 0)
  const chordDuration = analysis.chords.reduce(
    (maximum, chord) => Math.max(maximum, chord.start + chord.duration),
    0,
  )
  const duration = Math.max(noteDuration, chordDuration, measureDuration)
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
  const measureCount = Math.max(
    1,
    Array.from(
      { length: Math.ceil(beatStarts.length / 4) },
      (_, index) => index * 4,
    ).filter((beatIndex) => beatStarts[beatIndex] < duration).length,
  )
  const measures: TabMeasure[] = Array.from({ length: measureCount }, (_, measureIndex) => {
    const firstBeatIndex = measureIndex * 4
    const start = beatStarts[firstBeatIndex]
    const end = beatStarts[firstBeatIndex + 4]
    const chord =
      analysis.chords.find(
        (candidate) => candidate.start <= start && candidate.start + candidate.duration > start,
      )?.chord ?? 'N.C.'

    return {
      number: measureIndex + 1,
      start,
      duration: end - start,
      chord,
      section:
        measureIndex === 0
          ? '前奏'
          : measureIndex % 16 === 0
            ? `段落 ${Math.floor(measureIndex / 16) + 1}`
            : undefined,
      beats: Array.from({ length: 4 }, (_, beatIndex): TabBeat => ({
        at: beatStarts[firstBeatIndex + beatIndex],
        duration:
          beatStarts[firstBeatIndex + beatIndex + 1] -
          beatStarts[firstBeatIndex + beatIndex],
        style: 'pick',
        notes: [],
      })),
    }
  })

  const detectedOnsets = (analysis.onsets ?? [])
    .filter((onset) => Number.isFinite(onset) && onset >= 0)
    .sort((left, right) => left - right)
  const notationTimeFor = (start: number) => {
    if (detectedOnsets.length === 0) return start
    let lowerBound = 0
    let upperBound = detectedOnsets.length
    while (lowerBound < upperBound) {
      const middle = Math.floor((lowerBound + upperBound) / 2)
      if (detectedOnsets[middle] < start) {
        lowerBound = middle + 1
      } else {
        upperBound = middle
      }
    }
    const candidates = [
      detectedOnsets[Math.max(0, lowerBound - 1)],
      detectedOnsets[Math.min(detectedOnsets.length - 1, lowerBound)],
    ]
    const nearest = candidates.reduce((best, candidate) =>
      Math.abs(candidate - start) < Math.abs(best - start) ? candidate : best,
    )
    return Math.abs(nearest - start) <= 0.08 ? nearest : start
  }

  analysis.notes.forEach((workerNote, index) => {
    const notationAt = notationTimeFor(workerNote.start)
    let beatLowerBound = 0
    let beatUpperBound = beatStarts.length
    while (beatLowerBound < beatUpperBound) {
      const middle = Math.floor((beatLowerBound + beatUpperBound) / 2)
      if (beatStarts[middle] <= notationAt) {
        beatLowerBound = middle + 1
      } else {
        beatUpperBound = middle
      }
    }
    const previousBeatIndex = Math.max(0, beatLowerBound - 1)
    const nextBeatIndex = Math.min(beatStarts.length - 1, beatLowerBound)
    const nextBeatDistance = beatStarts[nextBeatIndex] - notationAt
    const previousBeatDuration =
      beatStarts[Math.min(beatStarts.length - 1, previousBeatIndex + 1)] -
      beatStarts[previousBeatIndex]
    const absoluteBeatIndex =
      nextBeatIndex > previousBeatIndex &&
      nextBeatDistance >= 0 &&
      nextBeatDistance <= previousBeatDuration * 0.1
        ? nextBeatIndex
        : previousBeatIndex
    const measureIndex = Math.min(
      measures.length - 1,
      Math.floor(absoluteBeatIndex / 4),
    )
    const beatIndex = Math.min(3, absoluteBeatIndex % 4)
    const beat = measures[measureIndex].beats[beatIndex]
    const techniqueCandidates: TechniqueCandidate[] =
      workerNote.technique_candidates?.map((candidate) => ({
        technique: candidate.technique,
        confidence: candidate.confidence,
        evidence: candidate.evidence,
      })) ?? []
    beat.notes.push({
      id: `worker-${index + 1}`,
      string: workerNote.string,
      fret: workerNote.fret,
      at: workerNote.start,
      notationAt,
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
          ? `worker-${workerNote.related_note_index + 1}`
          : undefined,
    })
  })

  measures.forEach((measure) => {
    measure.beats.forEach((beat) => {
      const notes = [...beat.notes].sort((left, right) => left.at - right.at)
      const uniqueStrings = new Set(notes.map((note) => note.string))
      const spread = notes.length > 1 ? notes.at(-1)!.at - notes[0].at : 0
      const stringDeltas = notes
        .slice(1)
        .map((note, index) => note.string - notes[index].string)
        .filter((delta) => delta !== 0)
      const direction = Math.sign(stringDeltas[0] ?? 0)
      const directionalRun =
        stringDeltas.length >= 2 &&
        direction !== 0 &&
        stringDeltas.every(
          (delta) => Math.sign(delta) === direction && Math.abs(delta) <= 3,
        )
      const maximumOnsetGap = notes
        .slice(1)
        .reduce(
          (maximum, note, index) =>
            Math.max(maximum, note.at - notes[index].at),
          0,
        )
      const containsLegato = notes.some((note) =>
        ['hammer-on', 'pull-off', 'slide'].includes(note.technique),
      )
      const rapidRepeatedString =
        notes.length >= 4 &&
        uniqueStrings.size === 1 &&
        maximumOnsetGap <= Math.min(0.14, beat.duration * 0.3)

      if (
        rapidRepeatedString ||
        notes.filter((note) => note.technique === 'tremolo').length >= 2
      ) {
        beat.style = 'tremolo'
        return
      }
      if (
        uniqueStrings.size < 3 ||
        containsLegato ||
        !directionalRun ||
        maximumOnsetGap > Math.min(0.18, beat.duration * 0.5)
      ) {
        beat.style = 'pick'
        beat.direction = undefined
        return
      }

      if (spread <= 0.035) {
        beat.style = 'pick'
        beat.direction = undefined
        return
      }

      beat.direction = direction < 0 ? 'down' : 'up'
      if (spread <= 0.14) {
        beat.style = 'strum'
      } else if (spread <= Math.min(0.26, beat.duration * 0.55) && notes.length >= 5) {
        beat.style = 'rasgueado'
      } else if (spread <= beat.duration * 0.85) {
        beat.style = 'arpeggio'
      } else {
        beat.style = 'pick'
        beat.direction = undefined
      }
    })
  })

  return {
    tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
    capo: 0,
    measures,
  }
}

function workerProject(
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
  const duration = tab.measures.reduce(
    (maximum, measure) =>
      Math.max(maximum, measure.start + measure.duration),
    0,
  )
  const dataRoot = resolve(process.cwd(), 'data')
  const maximumLevel = Math.max(0, ...Object.values(analysis.levels ?? {}))

  return {
    ...base,
    id: randomUUID(),
    title: source.kind === 'upload' ? base.title : (analysis.metadata?.title ?? base.title),
    artist: source.kind === 'upload' ? base.artist : (analysis.metadata?.artist ?? base.artist),
    bpm: analysis.bpm,
    duration,
    analysisMode: 'inference',
    transcriptionModel: analysis.transcription_model,
    techniqueModel: analysis.technique_model,
    visualization: analysis.visualization,
    tracks: base.tracks.map((track) => {
      const stemPath = analysis.stems[track.role]
      const relativePath = stemPath ? relative(dataRoot, stemPath).split(sep).join('/') : ''
      const isInsideDataRoot = relativePath.length > 0 && !relativePath.startsWith('..')
      const level = analysis.levels?.[track.role]
      return {
        ...track,
        url: isInsideDataRoot ? `/media/${relativePath}` : null,
        waveform: analysis.waveforms?.[track.role] ?? track.waveform,
        level,
        available:
          Boolean(stemPath) &&
          (level === undefined || maximumLevel === 0 || level >= maximumLevel * 0.02),
      }
    }),
    tab,
    warnings: analysis.warnings,
    source: {
      ...source,
      label: `${source.label} · Worker ${jobId.slice(0, 6)}`,
    },
  }
}

async function runInferencePipeline(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  inputPath?: string,
): Promise<StudioProject> {
  const workerCommand = getWorkerCommand()
  const ffmpegCommand = getFfmpegCommand()
  if (!workerCommand) {
    throw new Error('真实分析 Worker 尚未安装或不可执行')
  }
  if (!ffmpegCommand) {
    throw new Error('FFmpeg 尚未安装或不可执行')
  }

  const outputRoot = resolve(process.cwd(), 'data', 'jobs', job.id)
  await mkdir(outputRoot, { recursive: true })
  Object.assign(job, {
    status: 'separating',
    progress: 35,
    stageLabel: 'Worker 正在分离音轨',
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)

  const sourceArguments = inputPath
    ? ['--input', resolve(inputPath)]
    : job.source.url
      ? ['--url', job.source.url]
      : []
  if (sourceArguments.length === 0) {
    throw new Error('推理任务缺少媒体输入')
  }

  await new Promise<void>((resolvePromise, rejectPromise) => {
    const processHandle = spawn(workerCommand, [...sourceArguments, '--output', outputRoot], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        FFMPEG_BINARY: ffmpegCommand,
        PATH: `${dirname(ffmpegCommand)}:${process.env.PATH ?? ''}`,
        XDG_CACHE_HOME: resolve(projectRoot, '.cache'),
        TORCH_HOME: resolve(projectRoot, '.cache', 'torch'),
        HF_HOME: resolve(projectRoot, '.cache', 'huggingface'),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stderr = ''
    processHandle.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString()
    })
    processHandle.on('error', rejectPromise)
    processHandle.on('close', (code) => {
      if (code === 0) {
        resolvePromise()
      } else {
        rejectPromise(new Error(stderr.trim().slice(-1200) || `Worker exited with code ${code}`))
      }
    })
  })

  Object.assign(job, {
    status: 'mapping',
    progress: 92,
    stageLabel: '整理分轨与 TAB',
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  const analysis = JSON.parse(
    await readFile(resolve(outputRoot, 'analysis.json'), 'utf8'),
  ) as WorkerAnalysis
  const project = workerProject(analysis, job.source, job.id)
  Object.assign(job, {
    status: 'completed',
    progress: 100,
    stageLabel: '分析完成',
    projectId: project.id,
    updatedAt: new Date().toISOString(),
  })
  onUpdate(job)
  return project
}

export function runPipeline(
  job: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
  inputPath?: string,
): Promise<StudioProject> {
  const capabilities = getCapabilities()
  const platform = platforms.find((candidate) => candidate.id === job.source.kind)
  if (platform?.acquisition === 'metadata-only') {
    return Promise.reject(
      new Error(
        `${platform.label}链接不提供可授权下载的公开音频；会员可播放权限不等于导出或解密授权，请上传你合法持有的标准音频文件`,
      ),
    )
  }
  if (!capabilities.worker) {
    return Promise.reject(
      new Error('真实分析 Worker 尚未就绪，任务未执行；系统不会用示例结果替代'),
    )
  }
  return runInferencePipeline(job, onUpdate, inputPath).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : '媒体获取失败'
    if (
      !inputPath &&
      platform &&
      /yt-dlp|calledprocesserror|unsupported url|http error|sign in|login|drm/i.test(message)
    ) {
      throw new Error(
        `${platform.label}未返回可直接处理的公开非 DRM 音频。页面可播放或未标 VIP，不代表平台提供可下载媒体流；请上传你合法导出的标准音频文件`,
      )
    }
    throw error
  })
}

export function sourceFromLink(url: string): SourceDescriptor {
  const normalizedUrl = normalizeSharedUrl(url)
  const platform = detectPlatform(normalizedUrl)
  if (!platform) {
    throw new Error('当前仅支持哔哩哔哩、抖音和网易云音乐的公开非 DRM 链接')
  }

  return {
    kind: platform.id as PlatformId,
    label: platform.label,
    url: normalizedUrl,
  }
}
