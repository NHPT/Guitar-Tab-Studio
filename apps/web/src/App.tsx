import {
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Download,
  FileAudio,
  Gauge,
  Guitar,
  Headphones,
  Link2,
  LoaderCircle,
  Maximize2,
  Minimize2,
  Music2,
  Palette,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Scissors,
  SkipBack,
  SlidersHorizontal,
  Upload,
  Users,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  fetchDemoProject,
  fetchHealth,
  fetchJob,
  fetchPlatforms,
  fetchProject,
  submitLink,
  submitUpload,
} from './api'
import type {
  AnalysisJob,
  BeatStyle,
  CapabilityState,
  PlatformDescriptor,
  StudioProject,
  TabNote,
  Technique,
  TrackMix,
  VisualizerMode,
} from './types'
import { AudioVisualizer } from './AudioVisualizer'
import { recommendCapo, transposeChordForCapo } from './capo'
import { ChordDiagram, FullScore } from './FullScore'
import { inferBeatStyle } from './rhythm'
import { usePracticeAudio } from './usePracticeAudio'

const themePresets = [
  { id: 'vermilion', label: '朱砂', accent: '#b43c30', surface: '#ffffff' },
  { id: 'graphite', label: '石墨', accent: '#34383d', surface: '#ffffff' },
  { id: 'cobalt', label: '钴蓝', accent: '#235aa6', surface: '#ffffff' },
  { id: 'amber', label: '琥珀', accent: '#a65f16', surface: '#fffefa' },
  { id: 'jade', label: '青玉', accent: '#18766d', surface: '#fbfffe' },
  { id: 'cherry', label: '樱桃', accent: '#a52f4d', surface: '#fffdfd' },
  { id: 'wisteria', label: '藤紫', accent: '#6a4c93', surface: '#fefeff' },
  { id: 'indigo', label: '靛青', accent: '#3f517e', surface: '#ffffff' },
  { id: 'patina', label: '铜绿', accent: '#35726d', surface: '#fcfffe' },
  { id: 'night', label: '夜谱', accent: '#e26150', surface: '#202326' },
] as const

type ThemeId = (typeof themePresets)[number]['id']
const visualizerModes: Array<{ id: VisualizerMode; label: string }> = [
  { id: 'glow', label: '动感波光' },
  { id: 'wave', label: '动态波形' },
  { id: 'ecg', label: '心电图' },
  { id: 'spectrum', label: '频谱脉冲' },
  { id: 'ribbon', label: '呼吸光带' },
  { id: 'dots', label: '鼓点弹球' },
  { id: 'reactor', label: '反应堆控制杆' },
]

function initialTheme(): ThemeId {
  const stored = window.localStorage.getItem('gts-theme')
  return themePresets.some((theme) => theme.id === stored) ? (stored as ThemeId) : 'vermilion'
}

function initialVisualizer(): VisualizerMode {
  const stored = window.localStorage.getItem('gts-visualizer')
  return visualizerModes.some((mode) => mode.id === stored)
    ? (stored as VisualizerMode)
    : 'glow'
}

function extractSharedUrl(value: string): string | null {
  const normalized = value.trim().replaceAll('&amp;', '&')
  const matched = normalized.match(/https?:\/\/[^\s<>"']+/i)?.[0] ?? normalized
  const cleaned = matched.replace(/[，。；、）)\]}]+$/u, '')
  if (!cleaned) return null
  try {
    return new URL(/^https?:\/\//i.test(cleaned) ? cleaned : `https://${cleaned}`).toString()
  } catch {
    return null
  }
}

function detectInputPlatform(
  value: string,
  platforms: PlatformDescriptor[],
): PlatformDescriptor | null {
  const url = extractSharedUrl(value)
  if (!url) return null
  const host = new URL(url).hostname.toLowerCase()
  const unsupportedHosts = [
    'y.qq.com',
    'c.y.qq.com',
    'qishui.douyin.com',
    'music.douyin.com',
    'qishui.com',
  ]
  if (
    unsupportedHosts.some(
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

const techniqueLabels: Record<Technique, string> = {
  unknown: '待确认',
  pick: '拨弦',
  'strum-down': '下扫',
  'strum-up': '上扫',
  'hammer-on': '击弦',
  'pull-off': '勾弦',
  slide: '滑音',
  harmonic: '泛音',
  'palm-mute': '切音',
  'dead-note': '闷音',
  slap: '拍弦',
  'body-tap': '打板',
  tremolo: '轮指',
  bend: '推弦',
  'pinch-harmonic': '人工泛音',
  vibrato: '揉弦',
}

const techniqueEvidenceLabels: Record<string, string> = {
  'stable-pitched-event': '稳定音高',
  'audible-onset': '清晰起音',
  'natural-harmonic-fret': '自然泛音品位',
  'bright-spectrum': '高频突出',
  'tonal-spectrum': '谐波集中',
  'soft-onset': '弱起音',
  'sustained-note': '持续音',
  'harmonic-overtone-stack': '泛音倍频簇',
  'coincident-partials': '同步谐波分量',
  'hard-transient': '强瞬态',
  'noise-dominant': '噪声成分',
  'very-short-event': '短时事件',
  'hard-onset': '强起音',
  'bright-or-noisy-transient': '明亮瞬态',
  'short-event': '短音',
  'short-low-string-note': '低弦短音',
  'damped-spectrum': '衰减频谱',
  'weak-pitched-content': '弱音高成分',
  'noise-component': '噪声分量',
  'same-string-repeat': '同弦重复',
  'short-onset-gap': '短起音间隔',
  'rearticulated-onset': '重复触弦',
  'same-string': '同弦',
  'connected-notes': '音符衔接',
  'weak-second-onset': '后音弱起音',
  'ascending-fret': '品位上行',
  'descending-fret': '品位下行',
  'large-fret-change': '大跨度品位变化',
  'manual-correction': '人工校对',
  'demo-fixture': '演示数据',
}

const beatStyleLabels: Record<BeatStyle, string> = {
  pick: '拨弦',
  strum: '扫弦',
  arpeggio: '琶音',
  rasgueado: '轮扫',
  tremolo: '轮指',
}

const trackRoleLabels: Record<string, string> = {
  vocals: '人声',
  guitar: '吉他',
  bass: '贝斯',
  drums: '鼓组',
  piano: '钢琴 / 键盘',
  other: '其他乐器',
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safeSeconds / 60)
  const remainder = safeSeconds % 60
  return `${minutes}:${remainder.toString().padStart(2, '0')}`
}

function initialMixes(project: StudioProject): Record<string, TrackMix> {
  return Object.fromEntries(
    project.tracks.map((track) => [
      track.id,
      {
        muted: false,
        solo: false,
        volume: track.role === 'vocals' ? 0.72 : 0.86,
      },
    ]),
  )
}

function recommendedCapoForProject(project: StudioProject): number {
  const containsHarmonics = project.tab.measures.some((measure) =>
    measure.beats.some((beat) =>
      beat.notes.some((note) => note.technique === 'harmonic'),
    ),
  )
  return containsHarmonics ? 0 : recommendCapo(project.tab.measures)
}

function App() {
  const [project, setProject] = useState<StudioProject | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilityState | null>(null)
  const [platforms, setPlatforms] = useState<PlatformDescriptor[]>([])
  const [mixes, setMixes] = useState<Record<string, TrackMix>>({})
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [loopStartMeasure, setLoopStartMeasure] = useState(1)
  const [loopEndMeasure, setLoopEndMeasure] = useState(1)
  const [scorePart, setScorePart] = useState<'guitar' | 'bass' | 'drums'>('guitar')
  const [scoreExpanded, setScoreExpanded] = useState(false)
  const [capoEnabled, setCapoEnabled] = useState(true)
  const [capoFret, setCapoFret] = useState(0)
  const [theme, setTheme] = useState<ThemeId>(initialTheme)
  const [visualizerMode, setVisualizerMode] = useState<VisualizerMode>(initialVisualizer)
  const [themeOpen, setThemeOpen] = useState(false)
  const [selectedNoteId, setSelectedNoteId] = useState<string>()
  const [selectedMeasureNumber, setSelectedMeasureNumber] = useState<number>()
  const [selectedBeatIndex, setSelectedBeatIndex] = useState<number>()
  const [importOpen, setImportOpen] = useState(false)
  const [importMode, setImportMode] = useState<'upload' | 'link'>('upload')
  const [linkValue, setLinkValue] = useState('')
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const [error, setError] = useState('')
  const playbackPositionRef = useRef(0)
  const animationRef = useRef<number | null>(null)
  const previousFrameRef = useRef<number | null>(null)
  const previousUiFrameRef = useRef(0)
  const timeReadoutRef = useRef<HTMLElement>(null)

  const loadProject = useCallback((nextProject: StudioProject) => {
    const finalMeasure = Math.max(1, nextProject.tab.measures.length)
    setProject(nextProject)
    setMixes(initialMixes(nextProject))
    playbackPositionRef.current = 0
    setPosition(0)
    setPlaying(false)
    setLoopEnabled(false)
    setLoopStartMeasure(1)
    setLoopEndMeasure(finalMeasure)
    setScorePart('guitar')
    setScoreExpanded(false)
    setCapoEnabled(true)
    setCapoFret(nextProject.tab.capo || recommendedCapoForProject(nextProject))
    setSelectedNoteId(undefined)
    setSelectedMeasureNumber(undefined)
    setSelectedBeatIndex(undefined)
  }, [])

  useEffect(() => {
    let active = true
    const requestedProject = new URLSearchParams(window.location.search).get('project')
    void Promise.all([
      requestedProject ? fetchProject(requestedProject) : fetchDemoProject(),
      fetchHealth(),
      fetchPlatforms(),
    ])
      .then(([loadedProject, capabilityState, platformList]) => {
        if (!active) return
        loadProject(loadedProject)
        setCapabilities(capabilityState)
        setPlatforms(platformList)
      })
      .catch((loadError: unknown) => {
        if (!active) return
        setError(loadError instanceof Error ? loadError.message : '无法连接本地服务')
      })
    return () => {
      active = false
    }
  }, [loadProject])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('gts-theme', theme)
  }, [theme])

  useEffect(() => {
    window.localStorage.setItem('gts-visualizer', visualizerMode)
  }, [visualizerMode])

  const loopRange = useMemo(() => {
    if (!project) {
      return { start: 0, end: 0 }
    }
    const start = project.tab.measures[loopStartMeasure - 1]?.start ?? 0
    const endMeasure = project.tab.measures[loopEndMeasure - 1]
    return {
      start,
      end: endMeasure ? endMeasure.start + endMeasure.duration : project.duration,
    }
  }, [loopEndMeasure, loopStartMeasure, project])

  const { analyserRef, activateAudio, seekAudio } = usePracticeAudio({
    project,
    playing,
    position,
    speed,
    mixes,
  })

  const seekPosition = useCallback(
    (seconds: number) => {
      playbackPositionRef.current = seconds
      setPosition(seconds)
      seekAudio(seconds)
    },
    [seekAudio],
  )

  useEffect(() => {
    if (!playing || !project) {
      previousFrameRef.current = null
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
      }
      return
    }

    const tick = (now: number) => {
      const previous = previousFrameRef.current ?? now
      const delta = ((now - previous) / 1000) * speed
      previousFrameRef.current = now
      let next = playbackPositionRef.current + delta
      let forceUiUpdate = false
      if (loopEnabled && next >= loopRange.end) {
        next = loopRange.start
        seekAudio(next)
        forceUiUpdate = true
      } else if (next >= project.duration) {
        next = 0
        seekAudio(0)
        setPlaying(false)
        forceUiUpdate = true
      }
      playbackPositionRef.current = next
      if (now - previousUiFrameRef.current >= 50) {
        previousUiFrameRef.current = now
        if (timeReadoutRef.current) {
          timeReadoutRef.current.textContent = formatTime(next)
        }
      }
      if (forceUiUpdate) {
        setPosition(next)
      }
      animationRef.current = requestAnimationFrame(tick)
    }

    animationRef.current = requestAnimationFrame(tick)
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [
    loopEnabled,
    loopRange.end,
    loopRange.start,
    playing,
    project,
    seekAudio,
    speed,
  ])

  const allNotes = useMemo(
    () =>
      project?.tab.measures.flatMap((measure) =>
        measure.beats.flatMap((beat) => beat.notes),
      ) ?? [],
    [project],
  )
  const selectedNote = allNotes.find((note) => note.id === selectedNoteId)
  const detectedPlatform = useMemo(
    () => detectInputPlatform(linkValue, platforms),
    [linkValue, platforms],
  )
  const selectedMeasure = project?.tab.measures.find(
    (measure) => measure.number === selectedMeasureNumber,
  )
  const selectedBeat =
    selectedMeasure && selectedBeatIndex !== undefined
      ? selectedMeasure.beats[selectedBeatIndex]
      : undefined
  const selectedBeatStyle = selectedBeat ? inferBeatStyle(selectedBeat) : undefined
  const recommendedCapo = useMemo(
    () => (project ? recommendedCapoForProject(project) : 0),
    [project],
  )
  const activeCapo = capoEnabled ? capoFret : 0
  const visibleTracks = useMemo(() => {
    if (!project) return []
    const available = project.tracks.filter((track) => track.available)
    if (project.analysisMode !== 'inference') return available

    const detected = available.filter((track) => {
      if (track.waveform.length === 0) return false
      const average =
        track.waveform.reduce((total, sample) => total + Math.abs(sample), 0) /
        track.waveform.length
      const activeRatio =
        track.waveform.filter((sample) => Math.abs(sample) >= 0.04).length /
        track.waveform.length
      return average >= 0.02 && activeRatio >= 0.02
    })
    return detected.length > 0 ? detected : available.slice(0, 1)
  }, [project])

  const updateSelectedNote = (update: Partial<TabNote>) => {
    if (!project || !selectedNoteId) {
      return
    }
    setProject({
      ...project,
      tab: {
        ...project.tab,
        measures: project.tab.measures.map((measure) => ({
          ...measure,
          beats: measure.beats.map((beat) => ({
            ...beat,
            notes: beat.notes.map((note) =>
              note.id === selectedNoteId ? { ...note, ...update } : note,
            ),
          })),
        })),
      },
    })
  }

  const updateSelectedChord = (chord: string) => {
    if (!project || !selectedMeasureNumber) return
    setProject({
      ...project,
      tab: {
        ...project.tab,
        measures: project.tab.measures.map((measure) =>
          measure.number === selectedMeasureNumber ? { ...measure, chord } : measure,
        ),
      },
    })
  }

  const updateSelectedBeatStyle = (style: BeatStyle) => {
    if (!project || !selectedMeasureNumber || selectedBeatIndex === undefined) return
    setProject({
      ...project,
      tab: {
        ...project.tab,
        measures: project.tab.measures.map((measure) =>
          measure.number === selectedMeasureNumber
            ? {
                ...measure,
                beats: measure.beats.map((beat, index) =>
                  index === selectedBeatIndex ? { ...beat, style } : beat,
                ),
              }
            : measure,
        ),
      },
    })
  }

  const updateMix = (trackId: string, update: Partial<TrackMix>) => {
    setMixes((current) => ({
      ...current,
      [trackId]: { ...current[trackId], ...update },
    }))
  }

  const selectChord = useCallback((measureNumber: number) => {
    setSelectedMeasureNumber(measureNumber)
    setSelectedBeatIndex(undefined)
    setSelectedNoteId(undefined)
  }, [])

  const selectBeat = useCallback((measureNumber: number, beatIndex: number) => {
    setSelectedMeasureNumber(measureNumber)
    setSelectedBeatIndex(beatIndex)
    setSelectedNoteId(undefined)
  }, [])

  const selectNote = useCallback((noteId: string) => {
    setSelectedNoteId(noteId)
    setSelectedMeasureNumber(undefined)
    setSelectedBeatIndex(undefined)
  }, [])

  const togglePlayback = () => {
    if (playing) {
      setPosition(playbackPositionRef.current)
      setPlaying(false)
      return
    }
    const currentPosition = playbackPositionRef.current
    if (
      loopEnabled &&
      (currentPosition < loopRange.start || currentPosition >= loopRange.end)
    ) {
      seekPosition(loopRange.start)
    }
    void activateAudio()
    setPlaying(true)
  }

  const pollJob = useCallback(
    async (jobId: string) => {
      let nextJob = await fetchJob(jobId)
      setJob(nextJob)
      while (!['completed', 'degraded', 'failed'].includes(nextJob.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 450))
        nextJob = await fetchJob(jobId)
        setJob(nextJob)
      }

      if (nextJob.status === 'failed') {
        setError(nextJob.error ?? '分析任务失败')
        setJob(null)
        return
      }

      if (nextJob.projectId) {
        const nextProject = await fetchProject(nextJob.projectId)
        loadProject(nextProject)
        window.history.replaceState(null, '', `?project=${nextProject.id}`)
        window.setTimeout(() => {
          setImportOpen(false)
          setJob(null)
        }, 700)
      }
    },
    [loadProject],
  )

  const startLinkImport = async () => {
    setError('')
    try {
      const normalizedUrl = extractSharedUrl(linkValue)
      if (!normalizedUrl || !detectedPlatform) {
        throw new Error('未识别该链接来源，请粘贴受支持平台的完整分享链接')
      }
      const nextJob = await submitLink(normalizedUrl)
      setJob(nextJob)
      await pollJob(nextJob.id)
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : '链接导入失败')
    }
  }

  const startUpload = async (file: File) => {
    setError('')
    try {
      const nextJob = await submitUpload(file)
      setJob(nextJob)
      await pollJob(nextJob.id)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : '上传失败')
    }
  }

  if (!project) {
    return (
      <main className="loading-screen">
        <AudioLines aria-hidden="true" />
        <LoaderCircle className="spin" aria-hidden="true" />
        <p>{error || '正在连接本地工作台'}</p>
      </main>
    )
  }

  return (
    <div className={`app-shell${selectedNote || selectedMeasure ? ' has-inspector' : ''}`}>
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">
            <Guitar aria-hidden="true" strokeWidth={2.5} />
          </span>
          <div>
            <strong>Guitar Tab Studio</strong>
            <span>HackAll · AI 吉他谱与分轨</span>
          </div>
        </div>
        <div className="header-status">
          <span className={`status-dot ${capabilities?.mode ?? 'demo'}`} />
          <span>
            {capabilities?.mode === 'inference'
              ? '模型已就绪'
              : capabilities?.mode === 'hybrid'
                ? '混合处理'
                : '演示引擎'}
          </span>
        </div>
        <div className="header-actions">
          <button
            className="icon-button"
            type="button"
            aria-label="打开社区协作"
            title="社区协作"
            onClick={() => {
              window.location.href = '/community'
            }}
          >
            <Users aria-hidden="true" />
          </button>
          <div className="theme-picker">
            <button
              className="icon-button"
              type="button"
              aria-label="选择界面配色"
              title="选择界面配色"
              aria-expanded={themeOpen}
              onClick={() => setThemeOpen((current) => !current)}
            >
              <Palette aria-hidden="true" />
            </button>
            {themeOpen ? (
              <div className="theme-menu" role="menu" aria-label="界面配色">
                <strong>界面配色</strong>
                <div>
                  {themePresets.map((preset) => (
                    <button
                      type="button"
                      role="menuitemradio"
                      aria-checked={theme === preset.id}
                      className={theme === preset.id ? 'is-active' : ''}
                      key={preset.id}
                      onClick={() => {
                        setTheme(preset.id)
                        setThemeOpen(false)
                      }}
                    >
                      <i
                        style={{
                          '--theme-accent': preset.accent,
                          '--theme-surface': preset.surface,
                        } as React.CSSProperties}
                      />
                      <span>{preset.label}</span>
                      {theme === preset.id ? <Check aria-hidden="true" /> : null}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="打印或导出 PDF"
            title="打印或导出 PDF"
            onClick={() => window.print()}
          >
            <Download aria-hidden="true" />
          </button>
          <button className="primary-button" type="button" onClick={() => setImportOpen(true)}>
            <Plus aria-hidden="true" />
            导入歌曲
          </button>
        </div>
      </header>

      <aside className="library-rail">
        <div className="rail-heading">
          <span>项目</span>
          <button className="icon-button compact" type="button" onClick={() => setImportOpen(true)}>
            <Plus aria-label="新建项目" />
          </button>
        </div>
        <button className="session-item is-current" type="button">
          <span className="session-art">
            <Music2 aria-hidden="true" />
          </span>
          <span>
            <strong>{project.title}</strong>
            <small>{project.source.label}</small>
          </span>
          <i />
        </button>
        <div className="rail-section">
          <span>来源</span>
          <button type="button" onClick={() => setImportOpen(true)}>
            <Upload aria-hidden="true" />
            本地音视频
          </button>
          <button
            type="button"
            onClick={() => {
              setImportMode('link')
              setImportOpen(true)
            }}
          >
            <Link2 aria-hidden="true" />
            在线链接
          </button>
        </div>
        <div className="rail-capabilities">
          <span>处理能力</span>
          <p>
            <i className={capabilities?.ffmpeg ? 'ok' : ''} />
            媒体标准化
          </p>
          <p>
            <i className={capabilities?.ytDlp ? 'ok' : ''} />
            公开链接解析
          </p>
          <p>
            <i className={capabilities?.worker ? 'ok' : ''} />
            AI 推理 Worker
          </p>
        </div>
      </aside>

      <main className="workspace">
        <section className="project-bar">
          <div>
            <span className="eyebrow">{project.artist}</span>
            <h1>{project.title}</h1>
          </div>
          <dl>
            <div>
              <dt>速度</dt>
              <dd>{project.bpm} BPM</dd>
            </div>
            <div>
              <dt>调性</dt>
              <dd>{project.key}</dd>
            </div>
            <div>
              <dt>拍号</dt>
              <dd>4 / 4</dd>
            </div>
            <div>
              <dt>转录模型</dt>
              <dd
                title={
                  project.analysisMode === 'demo'
                    ? '内置演示项目'
                    : project.transcriptionModel
                    ? `${project.transcriptionModel.name} ${project.transcriptionModel.version}`
                    : '历史项目未记录模型版本'
                }
              >
                {project.analysisMode === 'demo'
                  ? '演示数据'
                  : project.transcriptionModel?.kind === 'trained'
                  ? '弦级模型'
                  : '通用基线'}
              </dd>
            </div>
          </dl>
        </section>

        <section className="transport-section">
          <div className="transport">
            <button
              className="icon-button"
              type="button"
              title="回到开始"
              aria-label="回到开始"
              onClick={() => seekPosition(loopEnabled ? loopRange.start : 0)}
            >
              <SkipBack aria-hidden="true" />
            </button>
            <button
              className="play-button"
              type="button"
              aria-label={playing ? '暂停' : '播放'}
              onClick={togglePlayback}
            >
              {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
            </button>
            <div className="time-readout">
              <strong ref={timeReadoutRef}>{formatTime(position)}</strong>
              <span>/ {formatTime(project.duration)}</span>
            </div>
            <div className="speed-control">
              <Gauge aria-hidden="true" />
              <input
                aria-label="播放速度"
                type="range"
                min="0.5"
                max="1.25"
                step="0.05"
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value))}
              />
              <strong>{speed.toFixed(2)}×</strong>
            </div>
            <div className={`loop-control${loopEnabled ? ' is-active' : ''}`}>
              <button
                className="loop-toggle"
                type="button"
                aria-label="练习循环"
                aria-pressed={loopEnabled}
                title={loopEnabled ? '关闭练习循环' : '开启练习循环'}
                onClick={() => setLoopEnabled((current) => !current)}
              >
                <RotateCcw aria-hidden="true" />
                <span>练习循环</span>
              </button>
              <div className="loop-range" aria-label="循环小节范围">
                <label className="loop-measure-picker">
                  <span>起</span>
                  <select
                    aria-label="循环起始小节"
                    value={loopStartMeasure}
                    disabled={!loopEnabled}
                    onChange={(event) => {
                      const nextMeasure = Number(event.target.value)
                      setLoopStartMeasure(nextMeasure)
                      setLoopEndMeasure((current) =>
                        Math.max(current, nextMeasure),
                      )
                    }}
                  >
                    {project.tab.measures.map((measure) => (
                      <option value={measure.number} key={measure.number}>
                        第 {measure.number} 小节
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
                <span className="loop-range-separator">至</span>
                <label className="loop-measure-picker">
                  <span>止</span>
                  <select
                    aria-label="循环结束小节"
                    value={loopEndMeasure}
                    disabled={!loopEnabled}
                    onChange={(event) => {
                      const nextMeasure = Number(event.target.value)
                      setLoopEndMeasure(nextMeasure)
                      setLoopStartMeasure((current) =>
                        Math.min(current, nextMeasure),
                      )
                    }}
                  >
                    {project.tab.measures.map((measure) => (
                      <option value={measure.number} key={measure.number}>
                        第 {measure.number} 小节
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
              </div>
            </div>
            <label className="visualizer-picker">
              <AudioLines aria-hidden="true" />
              <select
                aria-label="播放可视化模式"
                value={visualizerMode}
                onChange={(event) =>
                  setVisualizerMode(event.target.value as VisualizerMode)
                }
              >
                {visualizerModes.map((mode) => (
                  <option value={mode.id} key={mode.id}>
                    {mode.label}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
          </div>
          <AudioVisualizer
            project={project}
            position={position}
            playbackPositionRef={playbackPositionRef}
            playing={playing}
            mode={visualizerMode}
            analyserRef={analyserRef}
            onSeek={seekPosition}
          />
        </section>

        <section className="tab-workspace">
          <header className="section-heading">
            <div>
              <span className="eyebrow">和弦 · 节奏 · 六线谱 · 技法为实验性建议</span>
              <h2>{scorePart === 'guitar' ? '吉他谱' : scorePart === 'bass' ? '贝斯谱' : '鼓谱'}</h2>
            </div>
            <div className="score-view-controls">
              <div className="capo-control">
                <label className="capo-toggle">
                  <input
                    type="checkbox"
                    checked={capoEnabled}
                    onChange={(event) => setCapoEnabled(event.target.checked)}
                  />
                  <i aria-hidden="true" />
                  <span>
                    <strong>变调夹</strong>
                    <small>
                      {capoEnabled
                        ? capoFret === recommendedCapo
                          ? '智能推荐'
                          : '手动位置'
                        : '原始指法'}
                    </small>
                  </span>
                </label>
                <label className="capo-fret-field">
                  <span>位置</span>
                  <select
                    aria-label="变调夹位置"
                    disabled={!capoEnabled}
                    value={capoFret}
                    onChange={(event) => setCapoFret(Number(event.target.value))}
                  >
                    {Array.from({ length: 13 }, (_, fret) => (
                      <option value={fret} key={fret}>
                        {fret === 0 ? '不夹' : `${fret} 品`}
                        {fret === recommendedCapo ? ' · 推荐' : ''}
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
              </div>
              <div className="part-selector" aria-label="选择曲谱声部">
                {[
                  ['guitar', '吉他'],
                  ['bass', '贝斯'],
                  ['drums', '鼓'],
                ].map(([value, label]) => (
                  <button
                    type="button"
                    key={value}
                    className={scorePart === value ? 'is-active' : ''}
                    onClick={() => {
                      setScorePart(value as typeof scorePart)
                      setSelectedNoteId(undefined)
                      setSelectedMeasureNumber(undefined)
                      setSelectedBeatIndex(undefined)
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button
                className="score-expand-button"
                type="button"
                onClick={() => setScoreExpanded((current) => !current)}
              >
                {scoreExpanded ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}
                {scoreExpanded ? '单行谱' : '完整曲谱'}
              </button>
            </div>
          </header>

          {scorePart === 'guitar' ? (
            <FullScore
              project={project}
              positionSnapshot={position}
              playbackPositionRef={playbackPositionRef}
              playing={playing}
              speed={speed}
              expanded={scoreExpanded}
              capo={activeCapo}
              selectedMeasureNumber={selectedMeasureNumber}
              selectedNoteId={selectedNoteId}
              onSelectChord={selectChord}
              onSelectBeat={selectBeat}
              onSelectNote={selectNote}
            />
          ) : (
            <div className="part-empty-state">
              <Music2 aria-hidden="true" />
              <strong>{scorePart === 'bass' ? '贝斯谱' : '鼓谱'}尚未生成</strong>
              <span>当前项目已保留对应分轨，可继续播放、静音和独奏。</span>
            </div>
          )}
        </section>

        <section className="mixer-section">
          <header className="section-heading">
            <div>
              <span className="eyebrow">实时练习</span>
              <h2>音轨混音</h2>
            </div>
            <span className="quiet-label">
              <SlidersHorizontal aria-hidden="true" />
              {project.analysisMode === 'inference'
                ? `${visibleTracks.length} 条有效音轨`
                : '内置示例音色'}
            </span>
          </header>
          <div className="mixer-grid">
            {visibleTracks.map((track) => {
              const mix = mixes[track.id]
              const trackName = trackRoleLabels[track.role] ?? track.name
              return (
                <div className="track-strip" key={track.id}>
                  <div className="track-title">
                    <i style={{ background: track.color }} />
                    <span>
                      <strong>{trackName}</strong>
                      <small>{track.role.toUpperCase()}</small>
                    </span>
                  </div>
                  <div className="mini-wave" aria-hidden="true">
                    {track.waveform.slice(0, 36).map((value, index) => (
                      <i key={`${track.id}-${index}`} style={{ height: `${12 + value * 78}%` }} />
                    ))}
                  </div>
                  <div className="track-controls">
                    <button
                      className={mix?.muted ? 'is-active' : ''}
                      type="button"
                      aria-label={`${trackName}${mix?.muted ? '取消静音' : '静音'}`}
                      title="静音"
                      onClick={() => updateMix(track.id, { muted: !mix?.muted })}
                    >
                      {mix?.muted ? <VolumeX aria-hidden="true" /> : <Volume2 aria-hidden="true" />}
                    </button>
                    <button
                      className={mix?.solo ? 'is-active' : ''}
                      type="button"
                      aria-label={`${trackName}${mix?.solo ? '取消独奏' : '独奏'}`}
                      title="独奏"
                      onClick={() => updateMix(track.id, { solo: !mix?.solo })}
                    >
                      <Headphones aria-hidden="true" />
                    </button>
                    <input
                      aria-label={`${trackName}音量`}
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={mix?.volume ?? 0.8}
                      onChange={(event) =>
                        updateMix(track.id, { volume: Number(event.target.value) })
                      }
                    />
                    <span>{Math.round((mix?.volume ?? 0.8) * 100)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      </main>

      {selectedNote || selectedMeasure ? (
        <aside className="inspector">
          <header>
            <span>
              <Scissors aria-hidden="true" />
              {selectedBeat ? '节奏技法校对' : selectedMeasure ? '和弦校对' : '单音校对'}
            </span>
            <button
              className="icon-button compact"
              type="button"
              aria-label="关闭校谱检查器"
              onClick={() => {
                setSelectedNoteId(undefined)
                setSelectedMeasureNumber(undefined)
                setSelectedBeatIndex(undefined)
              }}
            >
              <X aria-hidden="true" />
            </button>
          </header>
          {selectedMeasure ? (
            <div className="inspector-body chord-inspector">
              <ChordDiagram
                name={transposeChordForCapo(selectedMeasure.chord, activeCapo)}
              />
              <label className="field">
                <span>实际和弦（原调）</span>
                <input
                  type="text"
                  value={selectedMeasure.chord}
                  onChange={(event) => updateSelectedChord(event.target.value)}
                />
              </label>
              {selectedBeat ? (
                <label className="field">
                  <span>节奏技法</span>
                  <select
                    value={selectedBeatStyle}
                    onChange={(event) =>
                      updateSelectedBeatStyle(event.target.value as BeatStyle)
                    }
                  >
                    {Object.entries(beatStyleLabels).map(([value, label]) => (
                      <option value={value} key={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
              ) : null}
              <div className="note-meta">
                <span>小节</span>
                <strong>{selectedMeasure.number}</strong>
                <span>变调夹指法</span>
                <strong>{transposeChordForCapo(selectedMeasure.chord, activeCapo)}</strong>
                {selectedBeat ? (
                  <>
                    <span>拍内位置</span>
                    <strong>{selectedBeatIndex! + 1}</strong>
                  </>
                ) : null}
              </div>
            </div>
          ) : selectedNote ? (
            <div className="inspector-body">
              <div className="confidence">
                <span>音符置信度</span>
                <strong>{Math.round(selectedNote.confidence * 100)}%</strong>
                <i>
                  <b style={{ width: `${selectedNote.confidence * 100}%` }} />
                </i>
              </div>
              <div className="confidence">
                <span>技法置信度</span>
                <strong>
                  {Math.round(
                    (selectedNote.techniqueConfidence ?? selectedNote.confidence) *
                      100,
                  )}
                  %
                </strong>
                <i>
                  <b
                    style={{
                      width: `${
                        (selectedNote.techniqueConfidence ??
                          selectedNote.confidence) * 100
                      }%`,
                    }}
                  />
                </i>
              </div>
              <div className="confidence">
                <span>指板位置置信度</span>
                <strong>
                  {Math.round(
                    (selectedNote.positionConfidence ?? selectedNote.confidence) *
                      100,
                  )}
                  %
                </strong>
                <i>
                  <b
                    style={{
                      width: `${
                        (selectedNote.positionConfidence ??
                          selectedNote.confidence) * 100
                      }%`,
                    }}
                  />
                </i>
              </div>
              <label className="field">
                <span>演奏技法</span>
                <select
                  value={selectedNote.technique}
                  onChange={(event) => {
                    const technique = event.target.value as Technique
                    updateSelectedNote({
                      technique,
                      harmonicType:
                        technique === 'harmonic'
                          ? (selectedNote.harmonicType ?? 'natural')
                          : undefined,
                      techniqueConfidence: 1,
                      techniqueSource: 'user',
                      techniqueEvidence: ['manual-correction'],
                      techniqueCandidates: [
                        {
                          technique,
                          confidence: 1,
                          evidence: ['manual-correction'],
                        },
                      ],
                      relatedNoteId: ['hammer-on', 'pull-off', 'slide'].includes(
                        technique,
                      )
                        ? selectedNote.relatedNoteId
                        : undefined,
                    })
                  }}
                >
                  {Object.entries(techniqueLabels).map(([value, label]) => (
                    <option value={value} key={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <ChevronDown aria-hidden="true" />
              </label>
              {selectedNote.technique === 'harmonic' ? (
                <label className="field">
                  <span>泛音类型</span>
                  <select
                    value={selectedNote.harmonicType ?? 'natural'}
                    onChange={(event) => {
                      const harmonicType = event.target.value as
                        | 'natural'
                        | 'artificial'
                      updateSelectedNote({
                        harmonicType,
                        harmonicTouchFret:
                          harmonicType === 'artificial'
                            ? (selectedNote.harmonicTouchFret ??
                              selectedNote.fret + 12)
                            : undefined,
                        techniqueConfidence: 1,
                        techniqueSource: 'user',
                      })
                    }}
                  >
                    <option value="natural">自然泛音</option>
                    <option value="artificial">人工泛音</option>
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
              ) : null}
              {selectedNote.technique === 'harmonic' &&
              selectedNote.harmonicType === 'artificial' ? (
                <label className="field">
                  <span>人工泛音触弦品</span>
                  <input
                    type="number"
                    min="12"
                    max="36"
                    value={
                      selectedNote.harmonicTouchFret ?? selectedNote.fret + 12
                    }
                    onChange={(event) =>
                      updateSelectedNote({
                        harmonicTouchFret: Math.max(
                          selectedNote.fret + 12,
                          Math.min(36, Number(event.target.value)),
                        ),
                        positionConfidence: 1,
                        positionSource: 'user',
                      })
                    }
                  />
                </label>
              ) : null}
              <div className="field-row">
                <label className="field">
                  <span>弦（1 为最细弦）</span>
                  <select
                    value={selectedNote.string}
                    onChange={(event) =>
                      updateSelectedNote({
                        string: Number(event.target.value) as TabNote['string'],
                        positionConfidence: 1,
                        positionSource: 'user',
                      })
                    }
                  >
                    <option value="1">1 弦 · 高音 e</option>
                    <option value="2">2 弦 · B</option>
                    <option value="3">3 弦 · G</option>
                    <option value="4">4 弦 · D</option>
                    <option value="5">5 弦 · A</option>
                    <option value="6">6 弦 · 低音 E</option>
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
                <label className="field">
                  <span>{activeCapo > 0 ? '品位（相对变调夹）' : '品位'}</span>
                  <input
                    type="number"
                    min="0"
                    max="24"
                    value={Math.max(0, selectedNote.fret - activeCapo)}
                    onChange={(event) => {
                      const fret =
                        Math.max(0, Math.min(24, Number(event.target.value))) +
                        activeCapo
                      updateSelectedNote({
                        fret,
                        harmonicTouchFret:
                          selectedNote.technique === 'harmonic' &&
                          selectedNote.harmonicType === 'artificial'
                            ? fret + 12
                            : selectedNote.harmonicTouchFret,
                        positionConfidence: 1,
                        positionSource: 'user',
                      })
                    }}
                  />
                </label>
              </div>
              <div className="note-meta">
                <span>时间</span>
                <strong>{selectedNote.at.toFixed(2)}s</strong>
                <span>时值</span>
                <strong>{selectedNote.duration.toFixed(2)}s</strong>
                <span>技法来源</span>
                <strong>
                  {selectedNote.techniqueSource === 'user'
                    ? '人工校对'
                    : selectedNote.techniqueSource?.startsWith('source-score')
                      ? '来源谱验证'
                    : selectedNote.techniqueSource?.startsWith('transition')
                      ? '音符关系'
                      : selectedNote.techniqueSource?.startsWith('acoustic')
                        ? '声学分析'
                        : '兼容结果'}
                </strong>
                <span>位置来源</span>
                <strong>
                  {selectedNote.positionSource === 'user'
                    ? '人工校对'
                    : selectedNote.positionSource?.startsWith('source-score')
                      ? '来源谱验证'
                    : selectedNote.positionSource?.startsWith('harmonic')
                      ? '泛音映射'
                      : selectedNote.positionSource?.startsWith('legato')
                        ? '连奏把位优化'
                      : selectedNote.positionSource?.startsWith('playable')
                        ? '和弦把位优化'
                        : '兼容结果'}
                </strong>
              </div>
              {selectedNote.techniqueCandidates?.length ? (
                <div className="technique-candidates">
                  <span>技法候选</span>
                  {selectedNote.techniqueCandidates.slice(0, 3).map((candidate) => (
                    <div key={candidate.technique}>
                      <strong>{techniqueLabels[candidate.technique]}</strong>
                      <b>{Math.round(candidate.confidence * 100)}%</b>
                    </div>
                  ))}
                  {selectedNote.techniqueEvidence?.length ? (
                    <p>
                      {selectedNote.techniqueEvidence
                        .map((evidence) => techniqueEvidenceLabels[evidence] ?? evidence)
                        .join(' · ')}
                    </p>
                  ) : null}
                </div>
              ) : null}
              <button
                className="secondary-button"
                type="button"
                onClick={() =>
                  updateSelectedNote({
                    confidence: 1,
                    positionConfidence: 1,
                    positionSource: 'user',
                    techniqueConfidence: 1,
                    techniqueSource: 'user',
                    techniqueEvidence: ['manual-correction'],
                    techniqueCandidates: [
                      {
                        technique: selectedNote.technique,
                        confidence: 1,
                        evidence: ['manual-correction'],
                      },
                    ],
                  })
                }
              >
                <Check aria-hidden="true" />
                确认此音符
              </button>
            </div>
          ) : null}
          <div className="analysis-note">
            <CircleAlert aria-hidden="true" />
            <p>只修改当前选中元素；自动识别结果仍需结合听感确认。</p>
          </div>
        </aside>
      ) : null}

      {importOpen ? (
        <div className="modal-backdrop" role="presentation">
          <section className="import-dialog" role="dialog" aria-modal="true" aria-label="导入歌曲">
            <header>
              <div>
                <span className="eyebrow">新建分析任务</span>
                <h2>导入歌曲</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label="关闭"
                onClick={() => {
                  if (!job) setImportOpen(false)
                }}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            {job ? (
              <div className="job-progress">
                <span className="job-icon">
                  {job.status === 'completed' || job.status === 'degraded' ? (
                    <Check aria-hidden="true" />
                  ) : (
                    <LoaderCircle className="spin" aria-hidden="true" />
                  )}
                </span>
                <strong>{job.stageLabel}</strong>
                <p>{job.status === 'degraded' ? '已生成可编辑演示项目' : '处理期间可保持此窗口打开'}</p>
                <i
                  role="progressbar"
                  aria-label="歌曲分析进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(Math.max(0, Math.min(100, job.progress)))}
                  className={
                    job.status === 'completed' ||
                    job.status === 'degraded' ||
                    job.status === 'failed'
                      ? ''
                      : 'is-animated'
                  }
                >
                  <b style={{ width: `${job.progress}%` }} />
                </i>
                <span className="job-progress-value" aria-live="polite">
                  {Math.round(Math.max(0, Math.min(100, job.progress)))}%
                </span>
              </div>
            ) : (
              <>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={importMode === 'upload' ? 'is-active' : ''}
                    onClick={() => setImportMode('upload')}
                  >
                    <Upload aria-hidden="true" />
                    本地上传
                  </button>
                  <button
                    type="button"
                    className={importMode === 'link' ? 'is-active' : ''}
                    onClick={() => setImportMode('link')}
                  >
                    <Link2 aria-hidden="true" />
                    在线链接
                  </button>
                </div>
                {importMode === 'upload' ? (
                  <label className="drop-zone">
                    <FileAudio aria-hidden="true" />
                    <strong>选择音频或视频</strong>
                    <span>MP3、WAV、FLAC、APE、M4A、AAC、OGG、MP4、MOV、WEBM，最大 500 MB</span>
                    <small>MGG / MFLAC 仅在内容实际为标准音频时可用；加密容器和 NCM 无法处理。</small>
                    <input
                      type="file"
                      accept=".mp3,.wav,.flac,.ape,.m4a,.aac,.ogg,.mp4,.mov,.webm,.mgg,.mflac,audio/*,video/*"
                      onChange={(event) => {
                        const file = event.target.files?.[0]
                        if (file) void startUpload(file)
                      }}
                    />
                  </label>
                ) : (
                  <div className="link-import">
                    <label>
                      <span>歌曲链接</span>
                      <div>
                        <Link2 aria-hidden="true" />
                        <input
                          value={linkValue}
                          onChange={(event) => {
                            setLinkValue(event.target.value)
                            setError('')
                          }}
                          onKeyDown={(event) => {
                            if (
                              (event.metaKey || event.ctrlKey) &&
                              event.key.toLowerCase() === 'a'
                            ) {
                              event.preventDefault()
                              event.currentTarget.select()
                            }
                          }}
                          placeholder="粘贴哔哩哔哩、抖音或网易云音乐公开链接"
                          spellCheck={false}
                        />
                      </div>
                    </label>
                    <div className="platform-list">
                      {platforms.map((platform) => (
                        <span
                          key={platform.id}
                          className={detectedPlatform?.id === platform.id ? 'is-detected' : ''}
                        >
                          {detectedPlatform?.id === platform.id ? <Check aria-hidden="true" /> : null}
                          {platform.label}
                        </span>
                      ))}
                    </div>
                    <button
                      className="primary-button wide"
                      type="button"
                      disabled={!linkValue.trim() || !detectedPlatform}
                      onClick={() => void startLinkImport()}
                    >
                      开始分析
                    </button>
                    {linkValue.trim() ? (
                      <p className={`source-detection${detectedPlatform ? ' is-valid' : ''}`}>
                        {detectedPlatform
                          ? `已识别：${detectedPlatform.label}`
                          : '尚未识别链接来源'}
                      </p>
                    ) : null}
                    <p className="policy-copy">
                      QQ 音乐和汽水音乐需要登录凭据或无稳定公开媒体接口，因此不提供在线导入。
                    </p>
                  </div>
                )}
              </>
            )}
            {error ? <p className="dialog-error">{error}</p> : null}
          </section>
        </div>
      ) : null}
    </div>
  )
}

export default App
