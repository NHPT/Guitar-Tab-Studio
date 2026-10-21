import {
  ArrowLeft,
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Cloud,
  Cookie,
  Download,
  Drum,
  ExternalLink,
  FileAudio,
  FileImage,
  FolderOpen,
  Gauge,
  Guitar,
  HardDrive,
  Headphones,
  History,
  KeyRound,
  Languages,
  Link2,
  Laptop,
  LoaderCircle,
  Maximize2,
  Minus,
  Minimize2,
  Music2,
  Palette,
  Pause,
  Play,
  Plus,
  PackagePlus,
  RotateCcw,
  Save,
  Scissors,
  SkipBack,
  SlidersHorizontal,
  TimerReset,
  Upload,
  Users,
  Volume2,
  VolumeX,
  Trash2,
  WandSparkles,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import {
  arrangementPreservesPitch,
  authorizedMediaProviderForContent,
  countInBeatDurations,
  createEasyArrangement,
  formatTimeSignature,
  PRACTICE_RHYTHMS,
  practiceTempoProfile,
  requiresAuthorizedMediaDecode,
  rhythmCompatibilityForTempoProfile,
  type DesktopLocalFileSource,
  type DesktopRecentProject,
  type DesktopRuntimeStatus,
  type AppLanguage,
  type AppLanguagePreference,
  type DesktopApplicationSettings,
  type AuthorizedMediaProvider,
  type InstalledModelPackage,
  type MediaBrowserDescriptor,
  type MediaBrowserSession,
  type PercussionVoice,
  type PracticeRhythm,
  type ProjectRevision,
  type RhythmGroup,
  type RhythmHit,
  type RhythmKit,
  type StorageCleanupStatus,
  type StudioApplication,
  type TimeSignature,
  type TranscriptionModelVariant,
  type TranscriptionProviderId,
  type TranscriptionSetupProgress,
} from '@gts/core'
import './App.css'
import { studioApplication } from './api'
import type { DesktopStudioControls } from './desktop'
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
import { BrandMark } from './BrandMark'
import { recommendCapo, transposeChordForCapo } from './capo'
import { ChordDiagram, FullScore } from './FullScore'
import {
  LiveTranscriptionWorkspace,
  TunerWorkspace,
} from './LiveGuitarWorkspaces'
import { inferBeatStyle } from './rhythm'
import { useLiveGuitar } from './useLiveGuitar'
import { usePracticeAudio } from './usePracticeAudio'
import {
  initialLanguage,
  resolveLanguage,
  systemLanguage,
  translate,
  translatePracticeMessage,
} from './i18n'

const themePresets = [
  { id: 'vermilion', label: '朱砂', accent: '#b43c30', surface: '#ffffff' },
  { id: 'graphite', label: '石墨', accent: '#F59E0B', surface: '#30343D' },
  { id: 'cobalt', label: '钴蓝', accent: '#235aa6', surface: '#ffffff' },
  { id: 'amber', label: '琥珀', accent: '#F59E0B', surface: '#ffffff' },
  { id: 'jade', label: '青玉', accent: '#0F766E', surface: '#F2FBF8' },
  { id: 'cherry', label: '樱桃', accent: '#a52f4d', surface: '#fffdfd' },
  { id: 'wisteria', label: '藤紫', accent: '#6a4c93', surface: '#fefeff' },
  { id: 'indigo', label: '靛青', accent: '#3f517e', surface: '#ffffff' },
  { id: 'patina', label: '铜绿', accent: '#3e625d', surface: '#fffefa' },
  { id: 'moss', label: '苔庭', accent: '#53662e', surface: '#fcfdfb' },
  { id: 'glacier', label: '冰川', accent: '#38BDF8', surface: '#F4FAFF' },
  { id: 'ember', label: '余烬', accent: '#C65D3A', surface: '#FFF8F4' },
  { id: 'night', label: '夜谱', accent: '#A78BFA', surface: '#1D212B' },
  { id: 'deepsea', label: '深海', accent: '#2DD4BF', surface: '#1B2C31' },
  { id: 'pine', label: '松夜', accent: '#A3E635', surface: '#202C22' },
  { id: 'wine', label: '酒红', accent: '#FB7185', surface: '#30212A' },
  { id: 'moonstone', label: '月岩', accent: '#60A5FA', surface: '#242B36' },
  { id: 'scarlet-night', label: '赤夜', accent: '#EC4141', surface: '#242424' },
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

const countInBarOptions = [0, 1, 2, 3, 4] as const
const countInBarsStorageKey = 'gts-count-in-bars-v2'
const countInKitOptions = [
  { id: 'click', label: '电子滴答' },
  { id: 'wood', label: '木质节拍' },
] as const
type CountInKit = (typeof countInKitOptions)[number]['id']
type ProtectedImportRequest =
  | {
      kind: 'upload'
      file: File
      provider: AuthorizedMediaProvider
    }
  | {
      kind: 'link'
      url: string
      provider: 'qqmusic' | 'netease'
    }
interface MediaSessionRecovery {
  url: string
  authorized: boolean
  detecting: boolean
  browser?: MediaBrowserDescriptor
}
const protectedProviderLabels: Record<AuthorizedMediaProvider, string> = {
  qqmusic: 'QQ 音乐',
  netease: '网易云音乐',
  moo: 'MOO 音乐',
  kuwo: '酷我音乐',
  kugou: '酷狗音乐',
  unknown: '合作平台',
}
const huggingFaceOAuthApplicationUrl =
  'https://huggingface.co/settings/applications/new'
const huggingFaceOAuthRedirectUri =
  'http://127.0.0.1/oauth/callback'
type DesktopMediaSource = Exclude<DesktopLocalFileSource, 'default'>
const desktopMediaSourceOptions: Array<{
  id: DesktopMediaSource
  label: string
}> = [
  { id: 'qqmusic', label: 'QQ 音乐' },
  { id: 'netease', label: '网易云音乐' },
  { id: 'moo', label: 'MOO 音乐' },
  { id: 'kuwo', label: '酷我音乐' },
  { id: 'kugou', label: '酷狗音乐' },
]
const rhythmKitOptions: Array<{
  id: 'pattern' | RhythmKit
  label: string
}> = [
  { id: 'pattern', label: '跟随节奏型' },
  { id: 'acoustic', label: '原声鼓' },
  { id: 'electronic', label: '电子鼓' },
  { id: 'brush', label: '刷奏' },
  { id: 'wood', label: '木质' },
]
type RhythmKitSelection = (typeof rhythmKitOptions)[number]['id']
type StandalonePracticeMode = 'metronome' | 'drums'
type WorkspaceView = 'analysis' | 'practice' | 'tuner' | 'live-transcription'
type PracticeSubdivision = 1 | 2 | 3 | 4
type PracticeLaneId = 'kick' | 'snare' | 'hat' | 'percussion'

const practiceTimeSignatures: TimeSignature[] = [
  [4, 4],
  [3, 4],
  [2, 4],
  [6, 8],
]
const practiceLanes: Array<{
  id: PracticeLaneId
  label: string
  voices: PercussionVoice[]
  defaultVoice: PercussionVoice
  velocity: number
}> = [
  {
    id: 'kick',
    label: '底鼓',
    voices: ['kick'],
    defaultVoice: 'kick',
    velocity: 0.92,
  },
  {
    id: 'snare',
    label: '军鼓',
    voices: ['snare', 'clap', 'rim'],
    defaultVoice: 'snare',
    velocity: 0.88,
  },
  {
    id: 'hat',
    label: '踩镲',
    voices: ['closed-hat', 'open-hat', 'crash', 'shaker'],
    defaultVoice: 'closed-hat',
    velocity: 0.56,
  },
  {
    id: 'percussion',
    label: '打击乐',
    voices: ['tom-high', 'tom-low', 'cowbell'],
    defaultVoice: 'tom-high',
    velocity: 0.72,
  },
]
const customRhythmId = 'custom-drums'
const customRhythmsStorageKey = 'gts-custom-rhythms-v1'
const percussionVoices = new Set(
  practiceLanes.flatMap((lane) => lane.voices),
)

function customRhythmSteps(signature: TimeSignature): number {
  return signature[0] * (signature[1] === 8 ? 2 : 4)
}

function initialCustomRhythms(): Record<string, RhythmHit[]> {
  try {
    const value = JSON.parse(
      window.localStorage.getItem(customRhythmsStorageKey) ?? '{}',
    ) as Record<string, unknown>
    return Object.fromEntries(
      Object.entries(value).flatMap(([signature, hits]) => {
        if (!Array.isArray(hits)) return []
        const parsed = hits.flatMap((hit) => {
          if (
            !hit ||
            typeof hit !== 'object' ||
            !Number.isInteger((hit as RhythmHit).step) ||
            (hit as RhythmHit).step < 0 ||
            !percussionVoices.has((hit as RhythmHit).voice) ||
            typeof (hit as RhythmHit).velocity !== 'number' ||
            (hit as RhythmHit).velocity <= 0 ||
            (hit as RhythmHit).velocity > 1
          ) {
            return []
          }
          return [{ ...(hit as RhythmHit) }]
        })
        return [[signature, parsed]]
      }),
    )
  } catch {
    return {}
  }
}
const rhythmGroupOrder: RhythmGroup[] = [
  'metronome',
  'rock',
  'pop',
  'funk',
  'blues',
  'electronic',
  'world',
  'other-meter',
]
const rhythmGroupLabels: Record<RhythmGroup, string> = {
  metronome: '节拍器',
  rock: '摇滚',
  pop: '流行',
  funk: '放克与灵魂',
  blues: '布鲁斯',
  electronic: '电子',
  world: '世界节奏',
  'other-meter': '其他拍号',
}

interface CountInProgress {
  bar: number
  beat: number
  totalBars: number
  beatsPerBar: number
  tempo: number
}

const projectListDefaultHeight = 464
const projectListMinimumHeight = 116
const projectListReservedViewportHeight = 408

function projectListMaximumHeight(): number {
  return Math.max(
    projectListMinimumHeight,
    window.innerHeight - projectListReservedViewportHeight,
  )
}

function initialProjectListHeight(): number {
  const stored = Number(window.localStorage.getItem('gts-project-list-height'))
  const preferred =
    Number.isFinite(stored) && stored >= projectListMinimumHeight
      ? stored
      : projectListDefaultHeight
  return Math.min(preferred, projectListMaximumHeight())
}

function initialTheme(): ThemeId {
  const stored = window.localStorage.getItem('gts-theme')
  return themePresets.some((theme) => theme.id === stored) ? (stored as ThemeId) : 'amber'
}

function initialVisualizer(): VisualizerMode {
  const stored = window.localStorage.getItem('gts-visualizer')
  return visualizerModes.some((mode) => mode.id === stored)
    ? (stored as VisualizerMode)
    : 'glow'
}

function initialCountInBars(): number {
  const persisted = window.localStorage.getItem(countInBarsStorageKey)
  if (persisted === null) return 2
  const stored = Number(persisted)
  return countInBarOptions.includes(
    stored as (typeof countInBarOptions)[number],
  )
    ? stored
    : 2
}

function initialRhythmId(): string {
  const stored = window.localStorage.getItem('gts-rhythm-id')
  return PRACTICE_RHYTHMS.some((rhythm) => rhythm.id === stored)
    ? stored!
    : 'rock-basic'
}

function initialRhythmVolume(): number {
  const stored = Number(window.localStorage.getItem('gts-rhythm-volume'))
  return Number.isFinite(stored) && stored >= 0.1 && stored <= 1 ? stored : 0.68
}

function initialCountInVolume(): number {
  const persisted = window.localStorage.getItem('gts-count-in-volume')
  if (persisted === null) return 0.8
  const stored = Number(persisted)
  return Number.isFinite(stored) && stored >= 0 && stored <= 1 ? stored : 0.8
}

function initialCountInKit(): CountInKit {
  const stored = window.localStorage.getItem('gts-count-in-kit')
  return countInKitOptions.some((option) => option.id === stored)
    ? (stored as CountInKit)
    : 'click'
}

function initialRhythmKit(): RhythmKitSelection {
  const stored = window.localStorage.getItem('gts-rhythm-kit')
  return rhythmKitOptions.some((option) => option.id === stored)
    ? (stored as RhythmKitSelection)
    : 'pattern'
}

function initialStandaloneBpm(): number {
  const stored = Number(window.localStorage.getItem('gts-practice-bpm'))
  return Number.isFinite(stored) && stored >= 30 && stored <= 260
    ? Math.round(stored)
    : 100
}

function createMetronomeRhythm(
  timeSignature: TimeSignature,
  subdivision: PracticeSubdivision,
  accent: boolean,
): PracticeRhythm {
  const stepsPerBar = timeSignature[0] * subdivision
  return {
    id: `standalone-click-${formatTimeSignature(timeSignature)}-${subdivision}`,
    label: '独立节拍器',
    group: 'metronome',
    kit: 'click',
    timeSignature,
    stepsPerBar,
    bpmRange: [30, 260],
    swing: 0.5,
    hits: Array.from({ length: stepsPerBar }, (_, step) => {
      const beatStart = step % subdivision === 0
      return {
        step,
        voice:
          accent && step === 0
            ? ('click-high' as const)
            : ('click-low' as const),
        velocity:
          accent && step === 0 ? 1 : beatStart ? 0.72 : 0.32,
      }
    }),
  }
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

function formatBpm(bpm: number): string {
  const rounded = Math.round(bpm * 10) / 10
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)
}

function formatStorageSize(bytes: number): string {
  const gibibytes = bytes / 1024 ** 3
  if (gibibytes >= 1) return `${gibibytes.toFixed(1)} GB`
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`
}

function initialMixes(project: StudioProject): Record<string, TrackMix> {
  return Object.fromEntries(
    project.tracks.map((track) => [
      track.id,
      {
        muted: false,
        solo: false,
        volume:
          project.analysisMode === 'demo'
            ? 0.86
            : track.role === 'vocals' ? 0.72 : 0.86,
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

export interface AppProps {
  application?: StudioApplication<File> & Partial<DesktopStudioControls>
}

function App({ application = studioApplication }: AppProps) {
  const desktopControls = application as StudioApplication<File> &
    Partial<DesktopStudioControls>
  const [project, setProject] = useState<StudioProject | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilityState | null>(null)
  const [platforms, setPlatforms] = useState<PlatformDescriptor[]>([])
  const [mixes, setMixes] = useState<Record<string, TrackMix>>({})
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [countInBars, setCountInBars] = useState(initialCountInBars)
  const [countInKit, setCountInKit] = useState<CountInKit>(initialCountInKit)
  const [countInVolume, setCountInVolume] = useState(initialCountInVolume)
  const [countInProgress, setCountInProgress] =
    useState<CountInProgress | null>(null)
  const [rhythmEnabled, setRhythmEnabled] = useState(
    () => window.localStorage.getItem('gts-rhythm-enabled') === 'true',
  )
  const [rhythmId, setRhythmId] = useState(initialRhythmId)
  const [rhythmKit, setRhythmKit] =
    useState<RhythmKitSelection>(initialRhythmKit)
  const [rhythmVolume, setRhythmVolume] = useState(initialRhythmVolume)
  const [workspaceView, setWorkspaceView] =
    useState<WorkspaceView>('analysis')
  const [practiceMode, setPracticeMode] =
    useState<StandalonePracticeMode>('metronome')
  const [practicePlaying, setPracticePlaying] = useState(false)
  const [practiceBpm, setPracticeBpm] = useState(initialStandaloneBpm)
  const [practiceTimeSignature, setPracticeTimeSignature] =
    useState<TimeSignature>([4, 4])
  const [practiceSubdivision, setPracticeSubdivision] =
    useState<PracticeSubdivision>(1)
  const [practiceAccent, setPracticeAccent] = useState(true)
  const [practiceRhythmId, setPracticeRhythmId] =
    useState('rock-basic')
  const [practiceCustomEnabled, setPracticeCustomEnabled] =
    useState(false)
  const [customPracticeHits, setCustomPracticeHits] =
    useState<Record<string, RhythmHit[]>>(initialCustomRhythms)
  const [practiceKit, setPracticeKit] =
    useState<RhythmKit>('click')
  const [practiceVolume, setPracticeVolume] = useState(0.7)
  const [practiceStep, setPracticeStep] = useState(-1)
  const [mutedPracticeLanes, setMutedPracticeLanes] =
    useState<PracticeLaneId[]>([])
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [loopStartMeasure, setLoopStartMeasure] = useState(1)
  const [loopEndMeasure, setLoopEndMeasure] = useState(1)
  const [scorePart, setScorePart] = useState<'guitar' | 'bass'>('guitar')
  const [scoreGuitarPartId, setScoreGuitarPartId] = useState('overview')
  const [scoreExpanded, setScoreExpanded] = useState(false)
  const [capoEnabled, setCapoEnabled] = useState(true)
  const [capoFret, setCapoFret] = useState(0)
  const [theme, setTheme] = useState<ThemeId>(initialTheme)
  const [language, setLanguage] = useState<AppLanguage>(initialLanguage)
  const [languagePreference, setLanguagePreference] =
    useState<AppLanguagePreference>('system')
  const [detectedSystemLanguage, setDetectedSystemLanguage] =
    useState<AppLanguage>(() => systemLanguage(window.navigator.language))
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsBusy, setSettingsBusy] = useState(false)
  const [settingsError, setSettingsError] = useState('')
  const [visualizerMode, setVisualizerMode] = useState<VisualizerMode>(initialVisualizer)
  const [themeOpen, setThemeOpen] = useState(false)
  const [selectedNoteId, setSelectedNoteId] = useState<string>()
  const [selectedMeasureNumber, setSelectedMeasureNumber] = useState<number>()
  const [selectedBeatIndex, setSelectedBeatIndex] = useState<number>()
  const [importOpen, setImportOpen] = useState(false)
  const [runtimeOpen, setRuntimeOpen] = useState(false)
  const [runtimeBusy, setRuntimeBusy] = useState(false)
  const [runtimeError, setRuntimeError] = useState('')
  const [runtimeNotice, setRuntimeNotice] = useState('')
  const [runtimeStatus, setRuntimeStatus] =
    useState<DesktopRuntimeStatus | null>(null)
  const [runtimeLocation, setRuntimeLocation] = useState<'local' | 'cloud'>(
    'local',
  )
  const [transcriptionScheme, setTranscriptionScheme] =
    useState<TranscriptionProviderId>('basic-pitch')
  const [transcriptionChoice, setTranscriptionChoice] = useState<
    'auto' | TranscriptionModelVariant
  >('auto')
  const [cloudApiRoot, setCloudApiRoot] = useState('')
  const [oauthSetup, setOauthSetup] = useState(false)
  const [huggingFaceOAuthClientId, setHuggingFaceOAuthClientId] =
    useState('')
  const [transcriptionLicenseAccepted, setTranscriptionLicenseAccepted] =
    useState(false)
  const [transcriptionProgress, setTranscriptionProgress] =
    useState<TranscriptionSetupProgress | null>(null)
  const [transcriptionSetupError, setTranscriptionSetupError] = useState('')
  const [storageCleanup, setStorageCleanup] =
    useState<StorageCleanupStatus | null>(null)
  const [cleanupEnabled, setCleanupEnabled] = useState(true)
  const [cleanupTriggerGiB, setCleanupTriggerGiB] = useState('5')
  const [cleanupTargetGiB, setCleanupTargetGiB] = useState('8')
  const [cleanupRetentionHours, setCleanupRetentionHours] = useState('24')
  const [revisions, setRevisions] = useState<ProjectRevision[]>([])
  const [localProjects, setLocalProjects] = useState<DesktopRecentProject[]>(
    [],
  )
  const [builtinProject, setBuiltinProject] =
    useState<StudioProject | null>(null)
  const [projectListHeight, setProjectListHeight] = useState(
    initialProjectListHeight,
  )
  const [projectListHeightMaximum, setProjectListHeightMaximum] = useState(
    projectListMaximumHeight,
  )
  const [projectContextMenu, setProjectContextMenu] = useState<{
    project: DesktopRecentProject
    x: number
    y: number
  } | null>(null)
  const [revisionBusy, setRevisionBusy] = useState(false)
  const [revisionNotice, setRevisionNotice] = useState('')
  const [scoreLayer, setScoreLayer] = useState<'original' | 'easy'>('original')
  const [modelPackages, setModelPackages] = useState<InstalledModelPackage[]>([])
  const [modelManifestUrl, setModelManifestUrl] = useState('')
  const [importMode, setImportMode] = useState<'upload' | 'score' | 'link'>('upload')
  const [linkValue, setLinkValue] = useState('')
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const [protectedImport, setProtectedImport] =
    useState<ProtectedImportRequest | null>(null)
  const [mediaSessionRecovery, setMediaSessionRecovery] =
    useState<MediaSessionRecovery | null>(null)
  const [scoreImportBusy, setScoreImportBusy] = useState(false)
  const [error, setError] = useState('')
  const [mediaKeyNotice, setMediaKeyNotice] = useState('')
  const playbackPositionRef = useRef(0)
  const animationRef = useRef<number | null>(null)
  const previousFrameRef = useRef<number | null>(null)
  const previousUiFrameRef = useRef(0)
  const timeReadoutRef = useRef<HTMLElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const countInConsumedRef = useRef(false)
  const practiceTapTimesRef = useRef<number[]>([])
  const projectListResizeRef = useRef<{
    pointerId: number
    startY: number
    startHeight: number
  } | null>(null)
  const projectId = project?.id
  const t = useCallback(
    (source: string) => translate(language, source),
    [language],
  )
  const liveGuitar = useLiveGuitar()
  const stopLiveInput = liveGuitar.stop

  const loadProject = useCallback((nextProject: StudioProject) => {
    const finalMeasure = Math.max(1, nextProject.tab.measures.length)
    stopLiveInput()
    setPracticePlaying(false)
    setWorkspaceView('analysis')
    setProject(nextProject)
    setMixes(initialMixes(nextProject))
    playbackPositionRef.current = 0
    countInConsumedRef.current = false
    setPosition(0)
    setPlaying(false)
    setCountInProgress(null)
    setLoopEnabled(false)
    setLoopStartMeasure(1)
    setLoopEndMeasure(finalMeasure)
    setScorePart('guitar')
    setScoreGuitarPartId(
      nextProject.guitarParts.length > 1
        ? 'overview'
        : nextProject.primaryGuitarPartId,
    )
    setScoreExpanded(false)
    setCapoEnabled(true)
    setCapoFret(nextProject.tab.capo || recommendedCapoForProject(nextProject))
    setSelectedNoteId(undefined)
    setSelectedMeasureNumber(undefined)
    setSelectedBeatIndex(undefined)
    setScoreLayer('original')
    setRevisionNotice('')
  }, [stopLiveInput])

  const refreshLocalProjects = useCallback(async () => {
    if (!desktopControls.localProjects) return []
    const items = await desktopControls.localProjects()
    setLocalProjects(items)
    return items
  }, [desktopControls])

  const openLocalProject = useCallback(
    async (id: string) => {
      const nextProject = await application.getProject(id)
      loadProject(nextProject)
      window.history.replaceState(null, '', `?project=${nextProject.id}`)
      await refreshLocalProjects()
    },
    [application, loadProject, refreshLocalProjects],
  )

  useEffect(() => {
    let active = true
    const requestedProject = new URLSearchParams(window.location.search).get('project')
    const demoProject = application.getDemoProject()
    void Promise.all([
      requestedProject
        ? application.getProject(requestedProject)
        : demoProject,
      application.getCapabilities(),
      application.listPlatforms(),
      demoProject,
    ])
      .then(([loadedProject, capabilityState, platformList, loadedDemo]) => {
        if (!active) return
        setBuiltinProject(loadedDemo)
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
  }, [application, loadProject])

  useEffect(() => {
    if (!projectId) return
    let active = true
    void application
      .listProjectRevisions(projectId)
      .then((items) => {
        if (active) setRevisions(items)
      })
      .catch(() => {
        if (active) setRevisions([])
      })
    return () => {
      active = false
    }
  }, [application, projectId])

  useEffect(() => {
    if (!desktopControls.localProjects) return
    let active = true
    void desktopControls
      .localProjects()
      .then((items) => {
        if (active) setLocalProjects(items)
      })
      .catch(() => {
        if (active) setLocalProjects([])
      })
    return () => {
      active = false
    }
  }, [desktopControls, projectId])

  useEffect(() => {
    if (!projectContextMenu) return
    const closeMenu = () => setProjectContextMenu(null)
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeMenu()
    }
    window.addEventListener('pointerdown', closeMenu)
    window.addEventListener('blur', closeMenu)
    window.addEventListener('keydown', closeOnEscape)
    window.addEventListener('scroll', closeMenu, true)
    return () => {
      window.removeEventListener('pointerdown', closeMenu)
      window.removeEventListener('blur', closeMenu)
      window.removeEventListener('keydown', closeOnEscape)
      window.removeEventListener('scroll', closeMenu, true)
    }
  }, [projectContextMenu])

  useEffect(() => {
    if (!desktopControls.onTranscriptionProgress) return
    return desktopControls.onTranscriptionProgress((progress) => {
      setTranscriptionProgress(progress)
      setRuntimeNotice(progress.stageLabel)
    })
  }, [desktopControls])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('gts-theme', theme)
  }, [theme])

  useEffect(() => {
    document.documentElement.lang = language
    window.localStorage.setItem('gts-language', language)
  }, [language])

  useEffect(() => {
    if (!desktopControls.applicationSettings) return
    let active = true
    void desktopControls.applicationSettings().then((settings) => {
      if (!active) return
      setLanguage(settings.language)
      setLanguagePreference(settings.languagePreference)
      setDetectedSystemLanguage(settings.systemLanguage)
    })
    return () => {
      active = false
    }
  }, [desktopControls])

  useEffect(() => {
    if (!desktopControls.onOpenSettings) return
    return desktopControls.onOpenSettings(() => setSettingsOpen(true))
  }, [desktopControls])

  useEffect(() => {
    if (!desktopControls.onLanguageChanged) return
    return desktopControls.onLanguageChanged(
      (settings: DesktopApplicationSettings) => {
        setLanguage(settings.language)
        setLanguagePreference(settings.languagePreference)
        setDetectedSystemLanguage(settings.systemLanguage)
      },
    )
  }, [desktopControls])

  useEffect(() => {
    if (!desktopControls.setBackgroundJobState) return
    const active = Boolean(
      job &&
        !['completed', 'degraded', 'failed', 'cancelled'].includes(job.status),
    )
    void desktopControls.setBackgroundJobState(
      active && job
        ? {
            active: true,
            jobId: job.id,
            progress: job.progress,
            stageLabel: job.stageLabel,
          }
        : { active: false },
    )
  }, [desktopControls, job])

  useEffect(() => {
    const updateMaximumHeight = () => {
      const maximum = projectListMaximumHeight()
      setProjectListHeightMaximum(maximum)
      setProjectListHeight((current) => Math.min(current, maximum))
    }
    window.addEventListener('resize', updateMaximumHeight)
    return () => window.removeEventListener('resize', updateMaximumHeight)
  }, [])

  useEffect(() => {
    window.localStorage.setItem(
      'gts-project-list-height',
      String(Math.round(projectListHeight)),
    )
  }, [projectListHeight])

  useEffect(
    () =>
      application.subscribe?.((event) => {
        if (event.type === 'open-media') {
          setImportMode('upload')
          setImportOpen(true)
          window.setTimeout(() => fileInputRef.current?.click(), 0)
          return
        }
        void application
          .getProject(event.projectId)
          .then((nextProject) => {
            loadProject(nextProject)
            window.history.replaceState(
              null,
              '',
              `?project=${nextProject.id}`,
            )
          })
          .catch((loadError: unknown) =>
            setError(
              loadError instanceof Error
                ? loadError.message
                : '项目加载失败',
            ),
          )
      }),
    [application, loadProject],
  )

  const refreshDesktopRuntime = useCallback(async (
    preserveTranscriptionScheme = false,
  ) => {
    if (!desktopControls.runtimeStatus || !desktopControls.modelPackages) return
    const [status, packages, cleanup] = await Promise.all([
      desktopControls.runtimeStatus(project?.id),
      desktopControls.modelPackages(),
      desktopControls.storageCleanupStatus?.() ?? Promise.resolve(null),
    ])
    setRuntimeStatus(status)
    setRuntimeLocation(status.providerMode)
    if (!preserveTranscriptionScheme) {
      setTranscriptionScheme(status.transcription.preference.providerId)
    }
    setCloudApiRoot(status.cloudApiRoot ?? '')
    setTranscriptionChoice(
      status.transcription.preference.selectionMode === 'manual' &&
        status.transcription.preference.variant
        ? status.transcription.preference.variant
        : 'auto',
    )
    setTranscriptionLicenseAccepted(
      status.transcription.preference.licenseAcceptedVersion ===
        'CC-BY-NC-4.0',
    )
    setHuggingFaceOAuthClientId(
      status.transcription.oauthClientId ?? '',
    )
    setModelPackages(packages)
    setStorageCleanup(cleanup)
    if (cleanup) {
      setCleanupEnabled(cleanup.config.enabled)
      setCleanupTriggerGiB(
        (cleanup.config.triggerFreeBytes / 1024 ** 3).toString(),
      )
      setCleanupTargetGiB(
        (cleanup.config.targetFreeBytes / 1024 ** 3).toString(),
      )
      setCleanupRetentionHours(cleanup.config.retentionHours.toString())
    }
  }, [desktopControls, project?.id])

  const runDesktopAction = async (
    action: () => Promise<unknown>,
    options: {
      errorScope?: 'transcription'
      preserveTranscriptionScheme?: boolean
    } = {},
  ) => {
    setRuntimeBusy(true)
    setRuntimeError('')
    setRuntimeNotice('')
    if (options.errorScope === 'transcription') {
      setTranscriptionSetupError('')
    }
    try {
      await action()
      await refreshDesktopRuntime(
        options.preserveTranscriptionScheme === true,
      )
      if (options.errorScope === 'transcription') {
        setTranscriptionProgress(null)
      }
      setCapabilities(await application.getCapabilities())
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : '桌面运行时操作失败'
      setTranscriptionProgress(null)
      if (message.includes('已取消') && message.includes('模型准备')) {
        setRuntimeNotice(message)
        setRuntimeError('')
        setTranscriptionSetupError('')
      } else if (options.errorScope === 'transcription') {
        setRuntimeNotice('')
        setRuntimeError('')
        setTranscriptionSetupError(message)
      } else {
        setRuntimeNotice('')
        setRuntimeError(message)
      }
    } finally {
      setRuntimeBusy(false)
    }
  }

  useEffect(() => {
    window.localStorage.setItem('gts-visualizer', visualizerMode)
  }, [visualizerMode])

  useEffect(() => {
    window.localStorage.setItem(countInBarsStorageKey, String(countInBars))
    window.localStorage.setItem('gts-count-in-kit', countInKit)
    window.localStorage.setItem(
      'gts-count-in-volume',
      String(countInVolume),
    )
  }, [countInBars, countInKit, countInVolume])

  useEffect(() => {
    window.localStorage.setItem('gts-rhythm-enabled', String(rhythmEnabled))
    window.localStorage.setItem('gts-rhythm-id', rhythmId)
    window.localStorage.setItem('gts-rhythm-kit', rhythmKit)
    window.localStorage.setItem('gts-rhythm-volume', String(rhythmVolume))
  }, [rhythmEnabled, rhythmId, rhythmKit, rhythmVolume])

  useEffect(() => {
    window.localStorage.setItem('gts-practice-bpm', String(practiceBpm))
  }, [practiceBpm])

  useEffect(() => {
    window.localStorage.setItem(
      customRhythmsStorageKey,
      JSON.stringify(customPracticeHits),
    )
  }, [customPracticeHits])

  const scoreProject = useMemo(() => {
    if (!project) return project
    const selectedPart =
      project.guitarParts.find(
        (part) => part.id === scoreGuitarPartId,
      ) ??
      project.guitarParts.find(
        (part) => part.id === project.primaryGuitarPartId,
      )
    if (
      scoreLayer === 'easy' &&
      project.easyArrangement &&
      selectedPart?.id === project.primaryGuitarPartId
    ) {
      return { ...project, tab: project.easyArrangement.tab }
    }
    const selectedTab =
      selectedPart?.id === project.primaryGuitarPartId
        ? project.tab
        : selectedPart?.tab
    return selectedPart
      ? {
          ...project,
          tab: selectedTab ?? project.tab,
          transcriptionModel:
            selectedPart.transcriptionModel ??
            project.transcriptionModel,
        }
      : project
  }, [project, scoreGuitarPartId, scoreLayer])

  const loopRange = useMemo(() => {
    if (!scoreProject) {
      return { start: 0, end: 0 }
    }
    const start = scoreProject.tab.measures[loopStartMeasure - 1]?.start ?? 0
    const endMeasure = scoreProject.tab.measures[loopEndMeasure - 1]
    return {
      start,
      end: endMeasure
        ? endMeasure.start + endMeasure.duration
        : scoreProject.duration,
    }
  }, [loopEndMeasure, loopStartMeasure, scoreProject])

  const timeSignature =
    project?.tab.timeSignature ?? PRACTICE_RHYTHMS[0].timeSignature
  const tempoProfile = useMemo(() => {
    const fallback = (project?.bpm ?? 120) * speed
    return (
      practiceTempoProfile(project?.tab.measures ?? [], speed) ?? {
        low: fallback,
        typical: fallback,
        high: fallback,
      }
    )
  }, [project, speed])
  const tempoLabel =
    Math.round(tempoProfile.high) - Math.round(tempoProfile.low) >= 2
      ? `${Math.round(tempoProfile.low)}–${Math.round(tempoProfile.high)}`
      : String(Math.round(tempoProfile.typical))
  const requestedRhythm =
    PRACTICE_RHYTHMS.find((rhythm) => rhythm.id === rhythmId) ??
    PRACTICE_RHYTHMS[0]
  const selectedRhythm =
    rhythmCompatibilityForTempoProfile(
      requestedRhythm,
      tempoProfile,
      timeSignature,
    ).compatible
      ? requestedRhythm
      : PRACTICE_RHYTHMS.find(
          (rhythm) =>
            rhythm.group === 'metronome' &&
            rhythmCompatibilityForTempoProfile(
              rhythm,
              tempoProfile,
              timeSignature,
            ).compatible,
        ) ??
        PRACTICE_RHYTHMS.find(
          (rhythm) =>
            rhythmCompatibilityForTempoProfile(
              rhythm,
              tempoProfile,
              timeSignature,
            ).compatible,
        ) ??
        requestedRhythm
  const compatibleRhythmCount = PRACTICE_RHYTHMS.filter(
    (rhythm) =>
      rhythmCompatibilityForTempoProfile(
        rhythm,
        tempoProfile,
        timeSignature,
      ).compatible,
  ).length
  const standaloneDrumRhythms = useMemo(
    () =>
      PRACTICE_RHYTHMS.filter(
        (rhythm) =>
          rhythm.group !== 'metronome' &&
          rhythm.timeSignature[0] === practiceTimeSignature[0] &&
          rhythm.timeSignature[1] === practiceTimeSignature[1],
      ),
    [practiceTimeSignature],
  )
  const selectedStandaloneDrumRhythm =
    standaloneDrumRhythms.find(
      (rhythm) => rhythm.id === practiceRhythmId,
    ) ??
    standaloneDrumRhythms[0] ??
    PRACTICE_RHYTHMS.find((rhythm) => rhythm.id === 'rock-basic')!
  const customPatternKey = formatTimeSignature(practiceTimeSignature)
  const customPracticeRhythm = useMemo<PracticeRhythm>(
    () => ({
      id: customRhythmId,
      label: '自定义节奏',
      group: 'rock',
      kit: practiceKit === 'click' ? 'acoustic' : practiceKit,
      timeSignature: practiceTimeSignature,
      stepsPerBar: customRhythmSteps(practiceTimeSignature),
      bpmRange: [30, 260],
      swing: 0.5,
      hits: customPracticeHits[customPatternKey] ?? [],
    }),
    [
      customPatternKey,
      customPracticeHits,
      practiceKit,
      practiceTimeSignature,
    ],
  )
  const standaloneDefinition = useMemo(
    () =>
      practiceMode === 'metronome'
        ? createMetronomeRhythm(
            practiceTimeSignature,
            practiceSubdivision,
            practiceAccent,
          )
        : practiceCustomEnabled
          ? customPracticeRhythm
          : selectedStandaloneDrumRhythm,
    [
      customPracticeRhythm,
      practiceAccent,
      practiceCustomEnabled,
      practiceMode,
      practiceSubdivision,
      practiceTimeSignature,
      selectedStandaloneDrumRhythm,
    ],
  )
  const mutedPracticeVoices = useMemo(
    () =>
      practiceLanes
        .filter((lane) => mutedPracticeLanes.includes(lane.id))
        .flatMap((lane) => lane.voices),
    [mutedPracticeLanes],
  )
  const handlePracticeStep = useCallback((step: number) => {
    setPracticeStep(step)
  }, [])
  const enableCustomPattern = useCallback(() => {
    setCustomPracticeHits((current) =>
      customPatternKey in current
        ? current
        : {
            ...current,
            [customPatternKey]: selectedStandaloneDrumRhythm.hits.map(
              (hit) => ({ ...hit }),
            ),
          },
    )
    setPracticeCustomEnabled(true)
  }, [customPatternKey, selectedStandaloneDrumRhythm.hits])
  const toggleCustomPatternHit = useCallback(
    (lane: (typeof practiceLanes)[number], step: number) => {
      setCustomPracticeHits((current) => {
        const source =
          current[customPatternKey] ??
          selectedStandaloneDrumRhythm.hits.map((hit) => ({ ...hit }))
        const hasHit = source.some(
          (hit) => hit.step === step && lane.voices.includes(hit.voice),
        )
        const hits = source.filter(
          (hit) => hit.step !== step || !lane.voices.includes(hit.voice),
        )
        if (!hasHit) {
          hits.push({
            step,
            voice: lane.defaultVoice,
            velocity: lane.velocity,
          })
        }
        hits.sort(
          (left, right) =>
            left.step - right.step ||
            left.voice.localeCompare(right.voice),
        )
        return { ...current, [customPatternKey]: hits }
      })
      setPracticeCustomEnabled(true)
    },
    [customPatternKey, selectedStandaloneDrumRhythm.hits],
  )
  const clearCustomPattern = useCallback(() => {
    setCustomPracticeHits((current) => ({
      ...current,
      [customPatternKey]: [],
    }))
    setPracticeCustomEnabled(true)
  }, [customPatternKey])

  const {
    analyserRef,
    activateAudio,
    seekAudio,
    startCountIn,
    cancelCountIn,
  } = usePracticeAudio({
    project,
    playing,
    position,
    playbackPositionRef,
    speed,
    mixes,
    rhythm: {
      enabled: rhythmEnabled,
      definition: selectedRhythm,
      kit: rhythmKit === 'pattern' ? selectedRhythm.kit : rhythmKit,
      volume: rhythmVolume,
    },
    standalone: {
      playing: practicePlaying,
      bpm: practiceBpm,
      definition: standaloneDefinition,
      kit: practiceKit,
      volume: practiceVolume,
      mutedVoices: mutedPracticeVoices,
      onStep: handlePracticeStep,
    },
  })

  useEffect(() => {
    cancelCountIn()
  }, [cancelCountIn, projectId])

  const seekPosition = useCallback(
    (seconds: number) => {
      cancelCountIn()
      setCountInProgress(null)
      const firstMeasureStart = scoreProject?.tab.measures[0]?.start ?? 0
      if (seconds <= firstMeasureStart + 0.05) {
        countInConsumedRef.current = false
      }
      playbackPositionRef.current = seconds
      setPosition(seconds)
      seekAudio(seconds)
    },
    [cancelCountIn, scoreProject, seekAudio],
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
        countInConsumedRef.current = false
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
      scoreProject?.tab.measures.flatMap((measure) =>
        measure.beats.flatMap((beat) => beat.notes),
      ) ?? [],
    [scoreProject],
  )
  const selectedNote = allNotes.find((note) => note.id === selectedNoteId)
  const detectedPlatform = useMemo(
    () => detectInputPlatform(linkValue, platforms),
    [linkValue, platforms],
  )
  const jobTerminal = Boolean(
    job &&
      ['completed', 'degraded', 'failed', 'cancelled'].includes(job.status),
  )
  const selectedMeasure = scoreProject?.tab.measures.find(
    (measure) => measure.number === selectedMeasureNumber,
  )
  const selectedBeat =
    selectedMeasure && selectedBeatIndex !== undefined
      ? selectedMeasure.beats[selectedBeatIndex]
      : undefined
  const selectedBeatStyle = selectedBeat ? inferBeatStyle(selectedBeat) : undefined
  const recommendedCapo = useMemo(
    () => (scoreProject ? recommendedCapoForProject(scoreProject) : 0),
    [scoreProject],
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

  const updateActiveTab = (
    update: (tab: StudioProject['tab']) => StudioProject['tab'],
  ) => {
    setProject((current) => {
      if (!current) return current
      const activePartId =
        scoreGuitarPartId === 'overview'
          ? current.primaryGuitarPartId
          : scoreGuitarPartId
      if (
        scoreLayer === 'easy' &&
        current.easyArrangement &&
        activePartId === current.primaryGuitarPartId
      ) {
        return {
          ...current,
          easyArrangement: {
            ...current.easyArrangement,
            tab: update(current.easyArrangement.tab),
          },
        }
      }
      let updatedPrimaryTab = current.tab
      const guitarParts = current.guitarParts.map((part) => {
        if (part.id !== activePartId) return part
        const tab = update(part.tab)
        if (part.id === current.primaryGuitarPartId) {
          updatedPrimaryTab = tab
        }
        return { ...part, tab }
      })
      return {
        ...current,
        tab: updatedPrimaryTab,
        guitarParts,
      }
    })
    setRevisionNotice('有未保存的谱面修改')
  }

  const updateSelectedNote = (update: Partial<TabNote>) => {
    if (!project || !selectedNoteId) {
      return
    }
    updateActiveTab((tab) => ({
      ...tab,
      measures: tab.measures.map((measure) => ({
          ...measure,
          beats: measure.beats.map((beat) => ({
            ...beat,
            notes: beat.notes.map((note) =>
              note.id === selectedNoteId ? { ...note, ...update } : note,
            ),
          })),
        })),
    }))
  }

  const updateSelectedChord = (chord: string) => {
    if (!project || !selectedMeasureNumber) return
    updateActiveTab((tab) => ({
      ...tab,
      measures: tab.measures.map((measure) =>
          measure.number === selectedMeasureNumber ? { ...measure, chord } : measure,
        ),
    }))
  }

  const updateSelectedBeatStyle = (style: BeatStyle) => {
    if (!project || !selectedMeasureNumber || selectedBeatIndex === undefined) return
    updateActiveTab((tab) => ({
      ...tab,
      measures: tab.measures.map((measure) =>
          measure.number === selectedMeasureNumber
            ? {
                ...measure,
                beats: measure.beats.map((beat, index) =>
                  index === selectedBeatIndex ? { ...beat, style } : beat,
                ),
              }
            : measure,
        ),
    }))
  }

  const saveRevision = async (
    nextProject: StudioProject,
    label: string,
    kind: 'manual' | 'arrangement' = 'manual',
  ) => {
    setRevisionBusy(true)
    setRevisionNotice('')
    try {
      const revision = await application.saveProjectRevision(
        nextProject,
        label,
        kind,
        revisions[0]?.id ?? null,
      )
      setProject(revision.project)
      setRevisions((current) => [
        revision,
        ...current.filter((item) => item.id !== revision.id),
      ])
      setRevisionNotice('修订已保存')
    } catch (cause) {
      setRevisionNotice(
        cause instanceof Error ? cause.message : '保存修订失败',
      )
    } finally {
      setRevisionBusy(false)
    }
  }

  const generateEasyArrangement = () => {
    if (!project) return
    const easyArrangement = createEasyArrangement(project)
    if (!arrangementPreservesPitch(project.tab, easyArrangement.tab)) {
      setRevisionNotice('易弹版音高校验失败，未应用更改')
      return
    }
    setProject({ ...project, easyArrangement })
    setScoreLayer('easy')
    setRevisionNotice('易弹版已生成，保存修订后持久化')
  }

  const restoreRevision = async (revisionId: string) => {
    if (!project || !revisionId) return
    setRevisionBusy(true)
    setRevisionNotice('')
    try {
      const revision = await application.restoreProjectRevision(
        project.id,
        revisionId,
      )
      loadProject(revision.project)
      const items = await application.listProjectRevisions(project.id)
      setRevisions(items)
      setRevisionNotice('已恢复为新修订，历史记录未被覆盖')
    } catch (cause) {
      setRevisionNotice(
        cause instanceof Error ? cause.message : '恢复修订失败',
      )
    } finally {
      setRevisionBusy(false)
    }
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

  const cancelPreparation = () => {
    cancelCountIn()
    setCountInProgress(null)
  }

  const openImportDialog = (
    mode: 'upload' | 'score' | 'link' = 'upload',
  ) => {
    setImportMode(mode)
    setImportOpen(true)
  }

  const openAnalysisWorkspace = () => {
    setPracticePlaying(false)
    stopLiveInput()
    setWorkspaceView('analysis')
  }

  const openPracticeWorkspace = () => {
    if (playing) {
      setPosition(playbackPositionRef.current)
      setPlaying(false)
    }
    cancelPreparation()
    stopLiveInput()
    setWorkspaceView('practice')
  }

  const openLiveWorkspace = (
    view: Extract<WorkspaceView, 'tuner' | 'live-transcription'>,
  ) => {
    if (playing) {
      setPosition(playbackPositionRef.current)
      setPlaying(false)
    }
    setPracticePlaying(false)
    cancelPreparation()
    stopLiveInput()
    setWorkspaceView(view)
  }

  const changePlaybackSpeed = (nextSpeed: number) => {
    cancelPreparation()
    setSpeed(nextSpeed)
  }

  const changePracticeTempo = (nextTempo: number) => {
    if (!Number.isFinite(nextTempo)) return
    setPracticeBpm(Math.max(30, Math.min(260, Math.round(nextTempo))))
  }

  const tapPracticeTempo = () => {
    const now = performance.now()
    const recent = [
      ...practiceTapTimesRef.current.filter((time) => now - time <= 2_000),
      now,
    ].slice(-6)
    practiceTapTimesRef.current = recent
    if (recent.length < 2) return
    const intervals = recent.slice(1).map((time, index) => time - recent[index])
    const average =
      intervals.reduce((total, interval) => total + interval, 0) /
      intervals.length
    changePracticeTempo(60_000 / average)
  }

  const togglePracticePlayback = () => {
    if (practicePlaying) {
      setPracticePlaying(false)
      return
    }
    setPracticePlaying(true)
    void activateAudio().catch((cause: unknown) =>
      setError(
        cause instanceof Error ? cause.message : '无法启动练习音频',
      ),
    )
  }

  const togglePracticeLane = (lane: PracticeLaneId) => {
    setMutedPracticeLanes((current) =>
      current.includes(lane)
        ? current.filter((candidate) => candidate !== lane)
        : [...current, lane],
    )
  }

  const togglePlayback = () => {
    if (!project) return
    if (countInProgress) {
      cancelPreparation()
      return
    }
    if (playing) {
      setPosition(playbackPositionRef.current)
      setPlaying(false)
      return
    }
    const currentPosition = playbackPositionRef.current
    const firstMeasureStart = scoreProject?.tab.measures[0]?.start ?? 0
    if (
      loopEnabled &&
      (currentPosition < loopRange.start || currentPosition >= loopRange.end)
    ) {
      seekPosition(loopRange.start)
    } else if (countInBars > 0) {
      if (currentPosition < firstMeasureStart) {
        seekPosition(firstMeasureStart)
      }
    }
    const startsAtSongOpening =
      Math.abs(playbackPositionRef.current - firstMeasureStart) <= 0.05
    const shouldCountIn =
      countInBars > 0 &&
      startsAtSongOpening &&
      !countInConsumedRef.current
    if (!shouldCountIn) {
      countInConsumedRef.current = true
      void activateAudio()
      setPlaying(true)
      return
    }
    const beatDurations = countInBeatDurations(
      project.bpm,
      timeSignature[0],
    )
    const beatsPerBar = beatDurations.length
    const averageBeatDuration =
      beatDurations.reduce((total, duration) => total + duration, 0) /
      beatsPerBar
    const countInTempo = Math.round((60 * speed) / averageBeatDuration)
    setCountInProgress({
      bar: 1,
      beat: 0,
      totalBars: countInBars,
      beatsPerBar,
      tempo: countInTempo,
    })
    void startCountIn({
      bars: countInBars,
      speed,
      beatDurations,
      kit: countInKit,
      volume: countInVolume,
      onBeat: (bar, beat) =>
        setCountInProgress({
          bar,
          beat,
          totalBars: countInBars,
          beatsPerBar,
          tempo: countInTempo,
        }),
      onComplete: () => {
        countInConsumedRef.current = true
        previousFrameRef.current = null
        setCountInProgress(null)
        setPlaying(true)
      },
    }).catch((cause: unknown) => {
      setCountInProgress(null)
      setError(
        cause instanceof Error ? cause.message : '无法启动预备节拍',
      )
    })
  }

  const pollJob = useCallback(
    async (jobId: string) => {
      let nextJob = await application.getJob(jobId)
      setJob(nextJob)
      while (
        !['completed', 'degraded', 'failed', 'cancelled'].includes(
          nextJob.status,
        )
      ) {
        await new Promise((resolve) => window.setTimeout(resolve, 450))
        nextJob = await application.getJob(jobId)
        setJob(nextJob)
      }

      if (nextJob.status === 'failed') {
        if (
          nextJob.source.url &&
          (nextJob.errorCode === 'media-auth-required' ||
            nextJob.errorCode === 'media-session-unavailable') &&
          desktopControls.detectDefaultMediaBrowser
        ) {
          setMediaSessionRecovery({
            url: nextJob.source.url,
            authorized: Boolean(nextJob.source.authorizedMedia),
            detecting: false,
          })
        } else {
          setMediaSessionRecovery(null)
        }
        setError(nextJob.error ?? '分析任务失败')
        setJob(null)
        return
      }
      if (nextJob.status === 'cancelled') return

      if (nextJob.projectId) {
        const nextProject = await application.getProject(nextJob.projectId)
        loadProject(nextProject)
        window.history.replaceState(null, '', `?project=${nextProject.id}`)
        window.setTimeout(() => {
          setImportOpen(false)
          setJob(null)
        }, 700)
      }
    },
    [application, desktopControls, loadProject],
  )

  const authorizedMediaConsent = {
    accepted: true,
    policyVersion: 2,
  } as const

  const startLinkImport = async (
    authorized = false,
    preparedUrl?: string,
    mediaSession?: MediaBrowserSession,
  ) => {
    setError('')
    setMediaSessionRecovery(null)
    try {
      const normalizedUrl = preparedUrl ?? extractSharedUrl(linkValue)
      if (!normalizedUrl || !detectInputPlatform(normalizedUrl, platforms)) {
        throw new Error('未识别该链接来源，请粘贴受支持平台的完整分享链接')
      }
      const nextJob = await application.submitLink(
        normalizedUrl,
        authorized ? authorizedMediaConsent : undefined,
        mediaSession,
      )
      setProtectedImport(null)
      setJob(nextJob)
      await pollJob(nextJob.id)
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : '链接导入失败')
    }
  }

  const startUpload = async (file: File, authorized = false) => {
    setError('')
    try {
      const nextJob = await application.submitUpload(
        file,
        authorized ? authorizedMediaConsent : undefined,
      )
      setProtectedImport(null)
      setJob(nextJob)
      await pollJob(nextJob.id)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : '上传失败')
    }
  }

  const requestUpload = async (file: File) => {
    setError('')
    const header = new Uint8Array(
      await file.slice(0, 16).arrayBuffer(),
    )
    const provider = authorizedMediaProviderForContent(file.name, header)
    if (provider) {
      if (requiresAuthorizedMediaDecode(file.name, header)) {
        setProtectedImport({ kind: 'upload', file, provider })
        return
      }
    }
    await startUpload(file)
  }

  const requestLinkImport = async () => {
    setError('')
    setMediaSessionRecovery(null)
    try {
      const normalizedUrl = extractSharedUrl(linkValue)
      if (!normalizedUrl || !detectedPlatform) {
        throw new Error('未识别该链接来源，请粘贴受支持平台的完整分享链接')
      }
      if (
        detectedPlatform.id === 'qqmusic' ||
        detectedPlatform.id === 'netease'
      ) {
        setProtectedImport({
          kind: 'link',
          url: normalizedUrl,
          provider: detectedPlatform.id,
        })
        return
      }
      await startLinkImport(false, normalizedUrl)
    } catch (importError) {
      setError(
        importError instanceof Error ? importError.message : '链接导入失败',
      )
    }
  }

  const detectMediaSessionForRetry = async () => {
    if (
      !mediaSessionRecovery ||
      !desktopControls.detectDefaultMediaBrowser
    ) {
      return
    }
    setMediaSessionRecovery((current) =>
      current ? { ...current, detecting: true } : current,
    )
    try {
      const result = await desktopControls.detectDefaultMediaBrowser()
      if (!result.supported || !result.browser) {
        throw new Error(
          result.detectedLabel
            ? `系统默认浏览器“${result.detectedLabel}”暂不支持自动读取登录状态`
            : '无法识别系统默认浏览器，请确认已设置默认浏览器',
        )
      }
      setMediaSessionRecovery((current) =>
        current
          ? { ...current, detecting: false, browser: result.browser }
          : current,
      )
    } catch (cause) {
      setMediaSessionRecovery((current) =>
        current ? { ...current, detecting: false } : current,
      )
      setError(
        cause instanceof Error ? cause.message : '无法识别系统默认浏览器',
      )
    }
  }

  const retryWithDefaultBrowserSession = async () => {
    if (!mediaSessionRecovery?.browser) return
    const recovery = mediaSessionRecovery
    const browser = mediaSessionRecovery.browser
    await startLinkImport(
      recovery.authorized,
      recovery.url,
      { browser: browser.id },
    )
  }

  const startScoreUpload = async (file: File) => {
    setError('')
    setScoreImportBusy(true)
    try {
      const nextProject = await application.submitScore(file)
      loadProject(nextProject)
      window.history.replaceState(null, '', `?project=${nextProject.id}`)
      setImportOpen(false)
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : '曲谱识别失败',
      )
    } finally {
      setScoreImportBusy(false)
    }
  }

  const selectDesktopFile = async (
    kind: 'media' | 'score',
    source: DesktopLocalFileSource = 'default',
  ) => {
    if (!desktopControls.selectLocalFile) return
    setError('')
    try {
      const file = await desktopControls.selectLocalFile(kind, source)
      if (!file) return
      if (kind === 'media') await requestUpload(file)
      else await startScoreUpload(file)
    } catch (selectionError) {
      setError(
        selectionError instanceof Error
          ? selectionError.message
          : '无法打开所选文件',
      )
    }
  }

  const importMediaKeyDatabase = async (
    provider: 'kuwo' | 'kugou',
  ) => {
    if (!desktopControls.importMediaKeyDatabase) return
    setError('')
    setMediaKeyNotice('')
    try {
      const result =
        await desktopControls.importMediaKeyDatabase(provider)
      if (!result) return
      setMediaKeyNotice(
        `${provider === 'kuwo' ? '酷我' : '酷狗'}密钥库已载入 ${result.count} 条密钥；密钥仅保留在本次桌面运行内存中。`,
      )
    } catch (importError) {
      setError(
        importError instanceof Error
          ? importError.message
          : '密钥库导入失败',
      )
    }
  }

  const removeLocalProject = async (item: DesktopRecentProject) => {
    setProjectContextMenu(null)
    if (
      !desktopControls.removeProject ||
      !window.confirm(`确定删除本地项目“${item.title}”吗？`)
    ) {
      return
    }
    setError('')
    try {
      await desktopControls.removeProject(item.id)
      const remaining = await refreshLocalProjects()
      if (project?.id !== item.id) return
      const next = remaining[0]
      if (next) {
        await openLocalProject(next.id)
        return
      }
      const demoProject = await application.getDemoProject()
      setBuiltinProject(demoProject)
      loadProject(demoProject)
      window.history.replaceState(null, '', window.location.pathname)
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : '无法删除本地项目',
      )
    }
  }

  const changeLanguage = async (
    preference: AppLanguagePreference,
  ) => {
    setSettingsBusy(true)
    setSettingsError('')
    try {
      const settings = desktopControls.setLanguage
        ? await desktopControls.setLanguage(preference)
        : {
            languagePreference: preference,
            systemLanguage: systemLanguage(window.navigator.language),
            language: resolveLanguage(
              preference,
              systemLanguage(window.navigator.language),
            ),
          }
      setLanguage(settings.language)
      setLanguagePreference(settings.languagePreference)
      setDetectedSystemLanguage(settings.systemLanguage)
      window.localStorage.setItem(
        'gts-language-preference',
        settings.languagePreference,
      )
    } catch (languageError) {
      setSettingsError(
        languageError instanceof Error
          ? languageError.message
          : 'Unable to change language',
      )
    } finally {
      setSettingsBusy(false)
    }
  }

  if (!project) {
    return (
      <main className="loading-screen">
        <AudioLines aria-hidden="true" />
        <LoaderCircle className="spin" aria-hidden="true" />
        <p>
          {error ||
            (language === 'en-US'
              ? 'Connecting to the local workspace'
              : '正在连接本地工作台')}
        </p>
      </main>
    )
  }
  const displayedProject = scoreProject ?? project
  const selectedTranscriptionVariant =
    transcriptionChoice === 'auto'
      ? runtimeStatus?.transcription.recommendation.variant
      : transcriptionChoice
  const selectedTranscriptionWeightSource = selectedTranscriptionVariant
    ? runtimeStatus?.transcription.variantWeights[
        selectedTranscriptionVariant
      ]
    : undefined
  const selectedTranscriptionDescriptor =
    runtimeStatus?.transcription.providers
      .find((provider) => provider.id === 'muscriptor')
      ?.variants.find(
        (variant) => variant.id === selectedTranscriptionVariant,
      )
  const muscriptorComponent = runtimeStatus?.transcription.pipeline.find(
    (component) => component.id === 'muscriptor',
  )
  const noise2FretComponent = runtimeStatus?.transcription.pipeline.find(
    (component) => component.id === 'noise2fret',
  )
  const mrMt3Component = runtimeStatus?.transcription.pipeline.find(
    (component) => component.id === 'mr-mt3',
  )
  const mrMt3Requested =
    selectedTranscriptionVariant !== 'small' &&
    runtimeStatus?.transcription.device.accelerator === 'cuda'
  const mrMt3Selected =
    mrMt3Requested && mrMt3Component?.runtimeAvailable === true
  const mrMt3Fallback =
    mrMt3Requested && mrMt3Component?.runtimeAvailable === false
  const highAccuracyRuntimeAvailable =
    muscriptorComponent?.runtimeAvailable === true
  const highAccuracyDownloadRequired =
    selectedTranscriptionWeightSource === 'download-required' ||
    noise2FretComponent?.ready === false ||
    (mrMt3Selected && mrMt3Component?.ready === false)
  const highAccuracyDownloadBytes =
    (selectedTranscriptionWeightSource === 'download-required'
      ? selectedTranscriptionDescriptor?.weightSizeBytes ?? 0
      : 0) +
    (noise2FretComponent?.ready === false
      ? noise2FretComponent.weightSizeBytes ?? 0
      : 0) +
    (mrMt3Selected && mrMt3Component?.ready === false
      ? mrMt3Component.weightSizeBytes ?? 0
      : 0)
  const highAccuracyProviderAvailable =
    runtimeStatus?.transcription.providers.some(
      (provider) => provider.id === 'muscriptor',
    ) === true
  const hasResearchTranscriptionOptions =
    runtimeStatus?.transcription.providers.some(
      (provider) => provider.id !== 'basic-pitch',
    ) === true
  const activeHighAccuracyVariant =
    runtimeStatus?.transcription.resolvedVariant ??
    (runtimeStatus?.transcription.preference.providerId === 'muscriptor'
      ? runtimeStatus.transcription.preference.selectionMode === 'manual'
        ? runtimeStatus.transcription.preference.variant
        : runtimeStatus.transcription.recommendation.variant
      : undefined)
  const highAccuracySelectionActive =
    runtimeStatus?.transcription.preference.providerId === 'muscriptor' &&
    activeHighAccuracyVariant === selectedTranscriptionVariant &&
    highAccuracyRuntimeAvailable &&
    !highAccuracyDownloadRequired
  const selectedVariantExceedsMemory =
    transcriptionChoice !== 'auto' &&
    selectedTranscriptionDescriptor !== undefined &&
    runtimeStatus !== null &&
    runtimeStatus.transcription.device.totalMemoryBytes <
      selectedTranscriptionDescriptor.minimumMemoryBytes
  const guitarSpecialistComponent =
    runtimeStatus?.transcription.pipeline.find(
      (component) => component.id === 'stringtrace',
    )
  const multitrackResearchComponent =
    runtimeStatus?.transcription.pipeline.find(
      (component) => component.id === 'yourmt3',
    )
  const transcriptionSetupActive =
    Boolean(
      transcriptionProgress &&
        transcriptionProgress.phase !== 'ready',
    )
  const transcriptionAuthorizationActive =
    transcriptionSetupActive &&
    transcriptionProgress?.phase === 'authorizing'
  const cloudModeLabel = !runtimeStatus?.cloudAvailable
    ? '云端 · 配置'
    : runtimeStatus.cloudReady
      ? '云端'
      : '云端 · 不可用'
  const cloudModeTitle = !runtimeStatus?.cloudAvailable
    ? '尚未配置云端服务地址'
    : runtimeStatus.cloudReady
      ? '切换到云端推理，待分析音频将上传到已配置服务'
      : `云端服务当前不可用${
          runtimeStatus.cloudError ? `：${runtimeStatus.cloudError}` : ''
        }`
  const libraryProjects = [
    ...(builtinProject
      ? [
          {
            id: builtinProject.id,
            title: builtinProject.title,
            sourceLabel: builtinProject.source.label,
            openedAt: builtinProject.createdAt,
          },
        ]
      : []),
    ...localProjects,
    {
      id: project.id,
      title: project.title,
      sourceLabel: project.source.label,
      openedAt: project.createdAt,
    },
  ].filter(
    (item, index, items) =>
      items.findIndex((candidate) => candidate.id === item.id) === index,
  )

  return (
    <div
      className={`app-shell${
        workspaceView === 'analysis' && (selectedNote || selectedMeasure)
          ? ' has-inspector'
          : ''
      }`}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes('Files')) {
          event.preventDefault()
          event.dataTransfer.dropEffect = 'copy'
        }
      }}
      onDrop={(event) => {
        const file = event.dataTransfer.files[0]
        if (!file) return
        event.preventDefault()
        setImportOpen(true)
        const scoreFile =
          file.type === 'application/pdf' ||
          file.type.startsWith('image/') ||
          /\.(?:pdf|png|jpe?g|webp)$/iu.test(file.name)
        setImportMode(scoreFile ? 'score' : 'upload')
        if (scoreFile) {
          void startScoreUpload(file)
        } else {
          void requestUpload(file)
        }
      }}
    >
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">
            <BrandMark />
          </span>
          <div>
            <strong>Guitar Tab Studio</strong>
            <span>{t('AI 吉他谱与分轨')}</span>
          </div>
        </div>
        <div className="header-status">
          <span className={`status-dot ${capabilities?.mode ?? 'demo'}`} />
          <span>
            {capabilities?.mode === 'inference'
              ? t('模型已就绪')
              : capabilities?.mode === 'hybrid'
                ? t('混合处理')
                : t('演示引擎')}
          </span>
        </div>
        <div className="header-actions">
          {job && !jobTerminal ? (
            <button
              className="background-job-button"
              type="button"
              aria-label={t('打开分析进度')}
              title={t('后台分析任务')}
              onClick={() => setImportOpen(true)}
            >
              <LoaderCircle className="spin" aria-hidden="true" />
              <span>{Math.round(job.progress)}%</span>
            </button>
          ) : null}
          {desktopControls.runtimeStatus && (
            <button
              className="icon-button"
              type="button"
              aria-label={t('桌面运行时')}
              title={t('桌面运行时')}
              onClick={() => {
                setRuntimeOpen(true)
                setRuntimeLocation(runtimeStatus?.providerMode ?? 'local')
                setRuntimeError('')
                setRuntimeNotice('')
                void refreshDesktopRuntime()
              }}
            >
              <HardDrive aria-hidden="true" />
            </button>
          )}
          <button
            className="icon-button"
            type="button"
            aria-label={t('打开社区协作')}
            title={t('社区协作')}
            onClick={() => {
              window.location.href = '/community'
            }}
          >
            <Users aria-hidden="true" />
          </button>
          <button
            className="icon-button"
            type="button"
            aria-label={t('设置')}
            title={t('设置')}
            onClick={() => setSettingsOpen(true)}
          >
            <Languages aria-hidden="true" />
          </button>
          <div className="theme-picker">
            <button
              className="icon-button"
              type="button"
              aria-label={t('选择界面配色')}
              title={t('选择界面配色')}
              aria-expanded={themeOpen}
              onClick={() => setThemeOpen((current) => !current)}
            >
              <Palette aria-hidden="true" />
            </button>
            {themeOpen ? (
              <div className="theme-menu" role="menu" aria-label={t('界面配色')}>
                <strong>{t('界面配色')}</strong>
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
                      <span>{t(preset.label)}</span>
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
            aria-label={t('打印或导出 PDF')}
            title={t('打印或导出 PDF')}
            onClick={() => window.print()}
          >
            <Download aria-hidden="true" />
          </button>
          <button
            className="primary-button"
            type="button"
            onClick={() => openImportDialog()}
          >
            <Plus aria-hidden="true" />
            {t('导入歌曲')}
          </button>
        </div>
      </header>

      <aside className="library-rail">
        <div className="rail-heading">
          <span>{t('项目')}</span>
          <button
            className="icon-button compact"
            type="button"
            onClick={() => openImportDialog()}
          >
            <Plus aria-label={t('新建项目')} />
          </button>
        </div>
        <div
          className="project-list"
          style={{ height: projectListHeight }}
        >
          {libraryProjects.map((item) => (
            <button
              className={`session-item${
                item.id === project.id ? ' is-current' : ''
              }`}
              type="button"
              key={item.id}
              onClick={() => {
                openAnalysisWorkspace()
                if (item.id === builtinProject?.id) {
                  loadProject(builtinProject)
                  window.history.replaceState(
                    null,
                    '',
                    window.location.pathname,
                  )
                  return
                }
                void openLocalProject(item.id)
              }}
              onContextMenu={(event) => {
                if (
                  !desktopControls.removeProject ||
                  !localProjects.some((local) => local.id === item.id)
                ) {
                  return
                }
                event.preventDefault()
                setProjectContextMenu({
                  project: item,
                  x: Math.min(event.clientX, window.innerWidth - 156),
                  y: Math.min(event.clientY, window.innerHeight - 52),
                })
              }}
            >
              <span className="session-art">
                <Music2 aria-hidden="true" />
              </span>
              <span>
                <strong>{item.title}</strong>
                <small>{item.sourceLabel}</small>
              </span>
              {item.id === project.id ? <i /> : null}
            </button>
          ))}
        </div>
        <div
          className="project-list-resizer"
          role="separator"
          aria-label="调整项目列表高度"
          aria-orientation="horizontal"
          aria-valuemin={projectListMinimumHeight}
          aria-valuemax={projectListHeightMaximum}
          aria-valuenow={Math.round(projectListHeight)}
          tabIndex={0}
          title="拖拽调整项目列表高度"
          onPointerDown={(event) => {
            if (event.button !== 0) return
            projectListResizeRef.current = {
              pointerId: event.pointerId,
              startY: event.clientY,
              startHeight: projectListHeight,
            }
            event.currentTarget.setPointerCapture(event.pointerId)
            event.preventDefault()
          }}
          onPointerMove={(event) => {
            const resize = projectListResizeRef.current
            if (!resize || resize.pointerId !== event.pointerId) return
            setProjectListHeight(
              Math.min(
                projectListHeightMaximum,
                Math.max(
                  projectListMinimumHeight,
                  resize.startHeight + event.clientY - resize.startY,
                ),
              ),
            )
          }}
          onPointerUp={(event) => {
            if (projectListResizeRef.current?.pointerId !== event.pointerId) {
              return
            }
            projectListResizeRef.current = null
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
          }}
          onPointerCancel={() => {
            projectListResizeRef.current = null
          }}
          onKeyDown={(event) => {
            const step = event.shiftKey ? 58 : 12
            if (event.key === 'ArrowUp') {
              setProjectListHeight((current) =>
                Math.max(projectListMinimumHeight, current - step),
              )
            } else if (event.key === 'ArrowDown') {
              setProjectListHeight((current) =>
                Math.min(projectListHeightMaximum, current + step),
              )
            } else if (event.key === 'Home') {
              setProjectListHeight(projectListMinimumHeight)
            } else if (event.key === 'End') {
              setProjectListHeight(projectListHeightMaximum)
            } else {
              return
            }
            event.preventDefault()
          }}
        />
        <div className="rail-section">
          <span>{t('来源')}</span>
          <button type="button" onClick={() => openImportDialog('upload')}>
            <Upload aria-hidden="true" />
            {t('本地音视频')}
          </button>
          <button
            type="button"
            onClick={() => openImportDialog('link')}
          >
            <Link2 aria-hidden="true" />
            {t('在线链接')}
          </button>
        </div>
        <div className="rail-section rail-tools">
          <span>{t('工具')}</span>
          <button
            type="button"
            className={workspaceView === 'practice' ? 'is-active' : ''}
            aria-pressed={workspaceView === 'practice'}
            onClick={openPracticeWorkspace}
          >
            <TimerReset aria-hidden="true" />
            {t('节拍器 / 鼓机')}
          </button>
          <button
            type="button"
            className={workspaceView === 'tuner' ? 'is-active' : ''}
            aria-pressed={workspaceView === 'tuner'}
            onClick={() => openLiveWorkspace('tuner')}
          >
            <Gauge aria-hidden="true" />
            {t('调音器')}
          </button>
          <button
            type="button"
            className={
              workspaceView === 'live-transcription' ? 'is-active' : ''
            }
            aria-pressed={workspaceView === 'live-transcription'}
            onClick={() => openLiveWorkspace('live-transcription')}
          >
            <Guitar aria-hidden="true" />
            {t('实时记谱')}
          </button>
        </div>
        <div className="rail-capabilities">
          <span>{t('处理能力')}</span>
          <p>
            <i className={capabilities?.ffmpeg ? 'ok' : ''} />
            {t('媒体标准化')}
          </p>
          <p>
            <i className={capabilities?.ytDlp ? 'ok' : ''} />
            {t('公开链接解析')}
          </p>
          <p>
            <i className={capabilities?.worker ? 'ok' : ''} />
            {t('AI 推理 Worker')}
          </p>
        </div>
      </aside>
      {projectContextMenu ? (
        <div
          className="project-context-menu"
          role="menu"
          style={{
            left: projectContextMenu.x,
            top: projectContextMenu.y,
          }}
          onPointerDown={(event) => event.stopPropagation()}
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => void removeLocalProject(projectContextMenu.project)}
          >
            <Trash2 aria-hidden="true" />
            {t('删除项目')}
          </button>
        </div>
      ) : null}

      <main
        className={`workspace${
          workspaceView === 'analysis'
            ? ''
            : ` is-tool-workspace is-${workspaceView}-workspace`
        }`}
      >
        <section className="project-bar">
          <div>
            <span className="eyebrow">{project.artist}</span>
            <h1>{project.title}</h1>
          </div>
          <dl>
            <div>
              <dt>{t('速度')}</dt>
              <dd>{formatBpm(project.bpm)} BPM</dd>
            </div>
            <div>
              <dt>{t('调性')}</dt>
              <dd>{project.key}</dd>
            </div>
            <div>
              <dt>{t('拍号')}</dt>
              <dd>4 / 4</dd>
            </div>
            <div>
              <dt>{t('转录模型')}</dt>
              <dd
                title={
                  project.analysisMode === 'demo'
                    ? t('内置演示项目')
                    : project.transcriptionModel
                    ? `${project.transcriptionModel.name} ${project.transcriptionModel.version}`
                    : t('历史项目未记录模型版本')
                }
              >
                {project.analysisMode === 'demo'
                  ? t('待复核')
                  : project.transcriptionModel?.kind === 'trained'
                  ? t('弦级模型')
                  : t('通用基线')}
              </dd>
            </div>
          </dl>
          <div className="project-revision-controls">
            <div
              className="score-layer-switch"
              role="group"
              aria-label={t('谱面版本')}
            >
              <button
                type="button"
                className={scoreLayer === 'original' ? 'is-active' : ''}
                onClick={() => setScoreLayer('original')}
              >
                {t('原始谱')}
              </button>
              <button
                type="button"
                className={scoreLayer === 'easy' ? 'is-active' : ''}
                onClick={() => {
                  if (project.easyArrangement) setScoreLayer('easy')
                  else generateEasyArrangement()
                }}
              >
                <WandSparkles aria-hidden="true" />
                {t('易弹版')}
              </button>
            </div>
            <label className="revision-picker">
              <History aria-hidden="true" />
              <select
                aria-label={t('恢复项目修订')}
                value=""
                disabled={revisionBusy || revisions.length === 0}
                onChange={(event) => void restoreRevision(event.target.value)}
              >
                <option value="">{t('修订历史')}</option>
                {revisions.map((revision) => (
                  <option value={revision.id} key={revision.id}>
                    {revision.label} ·{' '}
                    {new Date(revision.createdAt).toLocaleString(language, {
                      month: '2-digit',
                      day: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
            <button
              className="secondary-button save-revision-button"
              type="button"
              disabled={revisionBusy}
              onClick={() =>
                void saveRevision(
                  project,
                  scoreLayer === 'easy' ? '易弹版修订' : '人工校订',
                  scoreLayer === 'easy' ? 'arrangement' : 'manual',
                )
              }
            >
              {revisionBusy ? (
                <LoaderCircle className="spin" aria-hidden="true" />
              ) : (
                <Save aria-hidden="true" />
              )}
              {t('保存修订')}
            </button>
            {revisionNotice ? (
              <span className="revision-notice" role="status">
                {revisionNotice}
              </span>
            ) : null}
          </div>
        </section>

        <section className="transport-section">
          <div className="transport">
            <button
              className="icon-button"
              type="button"
              title={t('回到开始')}
              aria-label={t('回到开始')}
              onClick={() => seekPosition(loopEnabled ? loopRange.start : 0)}
            >
              <SkipBack aria-hidden="true" />
            </button>
            <button
              className="play-button"
              type="button"
              aria-label={
                countInProgress
                  ? t('取消预备')
                  : playing
                    ? t('暂停')
                    : t('播放')
              }
              onClick={togglePlayback}
            >
              {playing || countInProgress ? (
                <Pause aria-hidden="true" />
              ) : (
                <Play aria-hidden="true" />
              )}
            </button>
            <div className="time-readout">
              <strong ref={timeReadoutRef}>{formatTime(position)}</strong>
              <span>/ {formatTime(project.duration)}</span>
            </div>
            <div className="speed-control">
              <Gauge aria-hidden="true" />
              <input
                aria-label={t('播放速度')}
                type="range"
                min="0.5"
                max="1.25"
                step="0.05"
                value={speed}
                onChange={(event) =>
                  changePlaybackSpeed(Number(event.target.value))
                }
              />
              <strong>{speed.toFixed(2)}×</strong>
            </div>
            <div className={`loop-control${loopEnabled ? ' is-active' : ''}`}>
              <button
                className="loop-toggle"
                type="button"
                aria-label={t('练习循环')}
                aria-pressed={loopEnabled}
                title={
                  loopEnabled ? t('关闭练习循环') : t('开启练习循环')
                }
                onClick={() => setLoopEnabled((current) => !current)}
              >
                <RotateCcw aria-hidden="true" />
                <span>{t('练习循环')}</span>
              </button>
              <div className="loop-range" aria-label={t('循环小节范围')}>
                <label className="loop-measure-picker">
                  <span>{t('起')}</span>
                  <select
                    aria-label={t('循环起始小节')}
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
                    {displayedProject.tab.measures.map((measure) => (
                      <option value={measure.number} key={measure.number}>
                        {language === 'en-US'
                          ? `Measure ${measure.number}`
                          : `第 ${measure.number} 小节`}
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
                <span className="loop-range-separator">{t('至')}</span>
                <label className="loop-measure-picker">
                  <span>{t('止')}</span>
                  <select
                    aria-label={t('循环结束小节')}
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
                    {displayedProject.tab.measures.map((measure) => (
                      <option value={measure.number} key={measure.number}>
                        {language === 'en-US'
                          ? `Measure ${measure.number}`
                          : `第 ${measure.number} 小节`}
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
                aria-label={t('播放可视化模式')}
                value={visualizerMode}
                onChange={(event) =>
                  setVisualizerMode(event.target.value as VisualizerMode)
                }
              >
                {visualizerModes.map((mode) => (
                  <option value={mode.id} key={mode.id}>
                    {t(mode.label)}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
          </div>
        </section>
        <section className="practice-section">
          <div className="practice-controls">
            <label className="practice-picker count-in-picker">
              <TimerReset aria-hidden="true" />
              <span>{t('预备')}</span>
              <select
                aria-label={t('预备小节数')}
                value={countInBars}
                onChange={(event) => {
                  cancelPreparation()
                  setCountInBars(Number(event.target.value))
                }}
              >
                {countInBarOptions.map((bars) => (
                  <option value={bars} key={bars}>
                    {bars === 0
                      ? t('关闭')
                      : language === 'en-US'
                        ? `${bars} measure${bars === 1 ? '' : 's'}`
                        : `${bars} 小节`}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
            <label className="practice-picker count-in-sound">
              <span>{t('预备音色')}</span>
              <select
                aria-label={t('预备节拍音色')}
                value={countInKit}
                disabled={countInBars === 0}
                onChange={(event) =>
                  setCountInKit(event.target.value as CountInKit)
                }
              >
                {countInKitOptions.map((option) => (
                  <option value={option.id} key={option.id}>
                    {t(option.label)}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
            <label className="rhythm-volume count-in-volume">
              <Volume2 aria-hidden="true" />
              <input
                aria-label={t('预备节拍音量')}
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={countInVolume}
                disabled={countInBars === 0}
                onChange={(event) =>
                  setCountInVolume(Number(event.target.value))
                }
              />
            </label>
            <button
              className={`rhythm-toggle${rhythmEnabled ? ' is-active' : ''}`}
              type="button"
              aria-label={t('节奏伴奏')}
              aria-pressed={rhythmEnabled}
              onClick={() => setRhythmEnabled((current) => !current)}
            >
              <Drum aria-hidden="true" />
              <span>{t('节奏伴奏')}</span>
            </button>
            <label className="practice-picker rhythm-picker">
              <span>{t('节奏')}</span>
              <select
                aria-label={t('节奏型')}
                value={selectedRhythm.id}
                onChange={(event) => setRhythmId(event.target.value)}
              >
                {rhythmGroupOrder.map((group) => (
                  <optgroup label={t(rhythmGroupLabels[group])} key={group}>
                    {PRACTICE_RHYTHMS.filter(
                      (rhythm) => rhythm.group === group,
                    ).map((rhythm) => {
                      const compatibility = rhythmCompatibilityForTempoProfile(
                        rhythm,
                        tempoProfile,
                        timeSignature,
                      )
                      return (
                        <option
                          value={rhythm.id}
                          disabled={!compatibility.compatible}
                          key={rhythm.id}
                        >
                          {t(rhythm.label)}
                          {compatibility.message
                            ? ` · ${translatePracticeMessage(
                                language,
                                compatibility.message,
                              )}`
                            : ''}
                        </option>
                      )
                    })}
                  </optgroup>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
            <label className="practice-picker rhythm-kit-picker">
              <span>{t('鼓组音色')}</span>
              <select
                aria-label={t('鼓组音色')}
                value={rhythmKit}
                disabled={!rhythmEnabled}
                onChange={(event) =>
                  setRhythmKit(
                    event.target.value as RhythmKitSelection,
                  )
                }
              >
                {rhythmKitOptions.map((option) => (
                  <option value={option.id} key={option.id}>
                    {t(option.label)}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>
            <label className="rhythm-volume">
              <Volume2 aria-hidden="true" />
              <input
                aria-label={t('节奏伴奏音量')}
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={rhythmVolume}
                disabled={!rhythmEnabled}
                onChange={(event) =>
                  setRhythmVolume(Number(event.target.value))
                }
              />
            </label>
            <span className="rhythm-availability" role="status">
              {formatTimeSignature(timeSignature)} · {tempoLabel} BPM ·{' '}
              {compatibleRhythmCount}/{PRACTICE_RHYTHMS.length}{' '}
              {language === 'en-US' ? 'available' : '可用'}
            </span>
          </div>
          {countInProgress ? (
            <div className="count-in-status" role="status" aria-live="polite">
              <div>
                <TimerReset aria-hidden="true" />
                <span>
                  {t('预备')} {countInProgress.bar}/
                  {countInProgress.totalBars}{' '}
                  {language === 'en-US' ? 'measures' : '小节'}
                  {' · '}
                  {countInProgress.tempo} BPM
                </span>
              </div>
              <strong>
                {countInProgress.beat || t('准备')}
              </strong>
              <div className="count-in-beats" aria-hidden="true">
                {Array.from(
                  { length: countInProgress.beatsPerBar },
                  (_, index) => (
                    <i
                      className={
                        index + 1 === countInProgress.beat
                          ? 'is-current'
                          : index + 1 < countInProgress.beat
                            ? 'is-past'
                            : ''
                      }
                      key={index}
                    />
                  ),
                )}
              </div>
            </div>
          ) : null}
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

        <section
          className={`standalone-practice${practicePlaying ? ' is-playing' : ''}`}
          aria-labelledby="standalone-practice-title"
        >
          <header>
            <button
              className="practice-back-button"
              type="button"
              aria-label={t('返回分析')}
              title={t('返回分析')}
              onClick={openAnalysisWorkspace}
            >
              <ArrowLeft aria-hidden="true" />
            </button>
            <div className="practice-title">
              <span className="eyebrow">{t('独立练习')}</span>
              <h2 id="standalone-practice-title">
                {t('节拍器 / 鼓机')}
              </h2>
            </div>
            <div
              className="practice-mode-switch"
              role="group"
              aria-label={t('节拍器 / 鼓机模式')}
            >
              <button
                type="button"
                className={practiceMode === 'metronome' ? 'is-active' : ''}
                aria-pressed={practiceMode === 'metronome'}
                onClick={() => {
                  setPracticeMode('metronome')
                  if (practiceKit !== 'wood') setPracticeKit('click')
                }}
              >
                <TimerReset aria-hidden="true" />
                {t('节拍器')}
              </button>
              <button
                type="button"
                className={practiceMode === 'drums' ? 'is-active' : ''}
                aria-pressed={practiceMode === 'drums'}
                onClick={() => {
                  setPracticeMode('drums')
                  if (practiceKit === 'click') {
                    setPracticeKit(selectedStandaloneDrumRhythm.kit)
                  }
                }}
              >
                <Drum aria-hidden="true" />
                {t('鼓机')}
              </button>
            </div>
            <button
              className="practice-master-toggle"
              type="button"
              aria-label={
                practicePlaying
                  ? t('停止节拍器 / 鼓机')
                  : t('启动节拍器 / 鼓机')
              }
              aria-pressed={practicePlaying}
              onClick={() => void togglePracticePlayback()}
            >
              {practicePlaying ? (
                <Pause aria-hidden="true" />
              ) : (
                <Play aria-hidden="true" />
              )}
              <span>{practicePlaying ? t('停止') : t('开始')}</span>
            </button>
          </header>

          <div className="standalone-practice-controls">
            <div className="practice-tempo-control">
              <span>{t('速度')}</span>
              <button
                type="button"
                aria-label={t('降低练习速度')}
                onClick={() => changePracticeTempo(practiceBpm - 1)}
              >
                <Minus aria-hidden="true" />
              </button>
              <label>
                <input
                  aria-label={t('练习速度')}
                  type="number"
                  min="30"
                  max="260"
                  value={practiceBpm}
                  onChange={(event) =>
                    changePracticeTempo(Number(event.target.value))
                  }
                />
                <small>BPM</small>
              </label>
              <button
                type="button"
                aria-label={t('提高练习速度')}
                onClick={() => changePracticeTempo(practiceBpm + 1)}
              >
                <Plus aria-hidden="true" />
              </button>
              <button
                className="tap-tempo-button"
                type="button"
                onClick={tapPracticeTempo}
              >
                TAP
              </button>
            </div>

            <label className="practice-tool-picker">
              <span>{t('拍号')}</span>
              <select
                aria-label={t('练习拍号')}
                value={formatTimeSignature(practiceTimeSignature)}
                onChange={(event) => {
                  const next =
                    practiceTimeSignatures.find(
                      (signature) =>
                        formatTimeSignature(signature) === event.target.value,
                    ) ?? practiceTimeSignatures[0]
                  setPracticeTimeSignature(next)
                  setPracticeCustomEnabled(false)
                  const firstPattern = PRACTICE_RHYTHMS.find(
                    (rhythm) =>
                      rhythm.group !== 'metronome' &&
                      rhythm.timeSignature[0] === next[0] &&
                      rhythm.timeSignature[1] === next[1],
                  )
                  if (firstPattern) {
                    setPracticeRhythmId(firstPattern.id)
                  }
                }}
              >
                {practiceTimeSignatures.map((signature) => (
                  <option
                    value={formatTimeSignature(signature)}
                    key={formatTimeSignature(signature)}
                  >
                    {formatTimeSignature(signature)}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>

            {practiceMode === 'metronome' ? (
              <>
                <label className="practice-tool-picker">
                  <span>{t('细分')}</span>
                  <select
                    aria-label={t('节拍细分')}
                    value={practiceSubdivision}
                    onChange={(event) =>
                      setPracticeSubdivision(
                        Number(event.target.value) as PracticeSubdivision,
                      )
                    }
                  >
                    <option value="1">{t('四分音符')}</option>
                    <option value="2">{t('八分音符')}</option>
                    <option value="3">{t('三连音')}</option>
                    <option value="4">{t('十六分音符')}</option>
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
                <button
                  className={`practice-accent-toggle${practiceAccent ? ' is-active' : ''}`}
                  type="button"
                  aria-pressed={practiceAccent}
                  onClick={() => setPracticeAccent((current) => !current)}
                >
                  {t('首拍重音')}
                </button>
              </>
            ) : (
              <label className="practice-tool-picker practice-pattern-picker">
                <span>{t('节奏型')}</span>
                <select
                  aria-label={t('独立鼓机节奏型')}
                  value={
                    practiceCustomEnabled
                      ? customRhythmId
                      : selectedStandaloneDrumRhythm.id
                  }
                  onChange={(event) => {
                    if (event.target.value === customRhythmId) {
                      enableCustomPattern()
                      return
                    }
                    const next =
                      standaloneDrumRhythms.find(
                        (rhythm) => rhythm.id === event.target.value,
                      ) ?? selectedStandaloneDrumRhythm
                    setPracticeCustomEnabled(false)
                    setPracticeRhythmId(next.id)
                    setPracticeKit(next.kit)
                  }}
                >
                  <option value={customRhythmId}>
                    {t('自定义节奏')}
                  </option>
                  {standaloneDrumRhythms.map((rhythm) => (
                    <option value={rhythm.id} key={rhythm.id}>
                      {t(rhythm.label)}
                    </option>
                  ))}
                </select>
                <ChevronDown aria-hidden="true" />
              </label>
            )}

            <label className="practice-tool-picker">
              <span>{t('音色')}</span>
              <select
                aria-label={t('独立练习音色')}
                value={practiceKit}
                onChange={(event) =>
                  setPracticeKit(event.target.value as RhythmKit)
                }
              >
                {(practiceMode === 'metronome'
                  ? countInKitOptions
                  : rhythmKitOptions.filter(
                      (option) =>
                        option.id !== 'pattern' && option.id !== 'click',
                    )
                ).map((option) => (
                  <option value={option.id} key={option.id}>
                    {t(option.label)}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </label>

            <label className="practice-tool-volume">
              <Volume2 aria-hidden="true" />
              <input
                aria-label={t('独立练习音量')}
                type="range"
                min="0.05"
                max="1"
                step="0.05"
                value={practiceVolume}
                onChange={(event) =>
                  setPracticeVolume(Number(event.target.value))
                }
              />
              <span>{Math.round(practiceVolume * 100)}%</span>
            </label>
          </div>

          {practiceMode === 'drums' ? (
            <div
              className="practice-step-sequencer"
              style={{
                '--practice-step-count': standaloneDefinition.stepsPerBar,
              } as CSSProperties}
            >
              {practiceCustomEnabled ? (
                <header className="practice-sequencer-toolbar">
                  <strong>{t('自定义节奏')}</strong>
                  <div>
                    <button
                      type="button"
                      aria-label={t('恢复预设')}
                      title={t('恢复预设')}
                      onClick={() => setPracticeCustomEnabled(false)}
                    >
                      <RotateCcw aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      aria-label={t('清空节奏')}
                      title={t('清空节奏')}
                      onClick={clearCustomPattern}
                    >
                      <Trash2 aria-hidden="true" />
                    </button>
                  </div>
                </header>
              ) : null}
              {practiceLanes.map((lane) => {
                const muted = mutedPracticeLanes.includes(lane.id)
                const laneSteps = new Set(
                  standaloneDefinition.hits
                    .filter((hit) => lane.voices.includes(hit.voice))
                    .map((hit) => hit.step),
                )
                return (
                  <div className={muted ? 'is-muted' : ''} key={lane.id}>
                    <button
                      className="practice-lane-toggle"
                      type="button"
                      aria-label={`${t(lane.label)}${muted ? t('取消静音') : t('静音')}`}
                      aria-pressed={muted}
                      onClick={() => togglePracticeLane(lane.id)}
                    >
                      {t(lane.label)}
                    </button>
                    <div className="practice-step-grid">
                      {Array.from(
                        { length: standaloneDefinition.stepsPerBar },
                        (_, step) => {
                          const hasHit = laneSteps.has(step)
                          const beatStep =
                            standaloneDefinition.stepsPerBar /
                            practiceTimeSignature[0]
                          return (
                          <button
                            type="button"
                            className={`${hasHit ? 'has-hit' : ''}${
                              practiceStep === step ? ' is-current' : ''
                            }${step % beatStep === 0 ? ' is-beat' : ''}`}
                            aria-label={
                              language === 'en-US'
                                ? `${t(lane.label)}, step ${step + 1}`
                                : `${t(lane.label)}，第 ${step + 1} 步`
                            }
                            aria-pressed={hasHit}
                            title={
                              language === 'en-US'
                                ? `${t(lane.label)}, step ${step + 1}`
                                : `${t(lane.label)}，第 ${step + 1} 步`
                            }
                            onClick={() =>
                              toggleCustomPatternHit(lane, step)
                            }
                            key={step}
                          />
                          )
                        },
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div
              className="metronome-pulse-strip"
              style={{
                '--practice-step-count': standaloneDefinition.stepsPerBar,
              } as CSSProperties}
              aria-hidden="true"
            >
              {Array.from(
                { length: standaloneDefinition.stepsPerBar },
                (_, step) => (
                  <i
                    className={`${step % practiceSubdivision === 0 ? 'is-beat' : ''}${
                      practiceStep === step ? ' is-current' : ''
                    }`}
                    key={step}
                  />
                ),
              )}
            </div>
          )}
        </section>

        <TunerWorkspace
          controller={liveGuitar}
          onBack={openAnalysisWorkspace}
          t={t}
        />

        <LiveTranscriptionWorkspace
          controller={liveGuitar}
          onBack={openAnalysisWorkspace}
          t={t}
        />

        <section className="tab-workspace">
          {project.analysisMode === 'demo' ? (
            <p className="score-review-status" role="note">
              {t('当前为音频分析候选谱，整首逐音准确性尚未通过独立验证。')}
            </p>
          ) : null}
          <header className="section-heading">
            <div>
              <span className="eyebrow">
                {t('和弦 · 节奏 · 六线谱 · 技法为实验性建议')}
              </span>
              <h2>
                {scorePart === 'guitar'
                  ? t('吉他谱')
                  : t('贝斯谱')}
              </h2>
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
                    <strong>{t('变调夹')}</strong>
                    <small>
                      {capoEnabled
                        ? capoFret === recommendedCapo
                          ? t('智能推荐')
                          : t('手动位置')
                        : t('原始指法')}
                    </small>
                  </span>
                </label>
                <label className="capo-fret-field">
                  <span>{t('位置')}</span>
                  <select
                    aria-label={t('变调夹位置')}
                    disabled={!capoEnabled}
                    value={capoFret}
                    onChange={(event) => setCapoFret(Number(event.target.value))}
                  >
                    {Array.from({ length: 13 }, (_, fret) => (
                      <option value={fret} key={fret}>
                        {fret === 0
                          ? t('不夹')
                          : language === 'en-US'
                            ? `Fret ${fret}`
                            : `${fret} 品`}
                        {fret === recommendedCapo
                          ? language === 'en-US'
                            ? ' · Recommended'
                            : ' · 推荐'
                          : ''}
                      </option>
                    ))}
                  </select>
                  <ChevronDown aria-hidden="true" />
                </label>
              </div>
              <div className="part-selector" aria-label={t('选择曲谱声部')}>
                {[
                  ['guitar', '吉他'],
                  ['bass', '贝斯'],
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
                    {t(label)}
                  </button>
                ))}
              </div>
              {scorePart === 'guitar' && project.guitarParts.length > 1 ? (
                <div
                  className="guitar-part-selector"
                  aria-label={t('选择吉他声部')}
                >
                  <button
                    type="button"
                    className={
                      scoreGuitarPartId === 'overview' ? 'is-active' : ''
                    }
                    onClick={() => {
                      setScoreGuitarPartId('overview')
                      setScoreLayer('original')
                      setSelectedNoteId(undefined)
                      setSelectedMeasureNumber(undefined)
                      setSelectedBeatIndex(undefined)
                    }}
                  >
                    {t('总谱')}
                  </button>
                  {project.guitarParts.map((part, index) => {
                    const roleLabel =
                      (part.roleConfidence ?? 0) >= 0.6
                        ? part.role === 'lead'
                          ? t('主音吉他')
                          : part.role === 'rhythm'
                            ? t('节奏吉他')
                            : part.name
                        : `Guitar ${index + 1}`
                    return (
                      <button
                        type="button"
                        key={part.id}
                        className={
                          scoreGuitarPartId === part.id
                            ? 'is-active'
                            : ''
                        }
                        onClick={() => {
                          setScoreGuitarPartId(part.id)
                          if (part.id !== project.primaryGuitarPartId) {
                            setScoreLayer('original')
                          }
                          setSelectedNoteId(undefined)
                          setSelectedMeasureNumber(undefined)
                          setSelectedBeatIndex(undefined)
                        }}
                      >
                        {roleLabel}
                      </button>
                    )
                  })}
                </div>
              ) : null}
              <button
                className="score-expand-button"
                type="button"
                onClick={() => setScoreExpanded((current) => !current)}
              >
                {scoreExpanded ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}
                {scoreExpanded ? t('单行谱') : t('完整曲谱')}
              </button>
            </div>
          </header>

          {scorePart === 'guitar' ? (
            scoreGuitarPartId === 'overview' &&
            project.guitarParts.length > 1 ? (
              <div className="multi-guitar-score">
                <p className="multi-guitar-score-note">
                  {t('每路吉他保持独立弦品与演奏关系；总谱不合并为一把六弦琴。')}
                </p>
                {project.guitarParts.map((part, index) => {
                  const roleLabel =
                    (part.roleConfidence ?? 0) >= 0.6
                      ? part.role === 'lead'
                        ? t('主音吉他')
                        : part.role === 'rhythm'
                          ? t('节奏吉他')
                          : part.name
                      : `Guitar ${index + 1}`
                  return (
                    <section className="guitar-part-score" key={part.id}>
                      <header>
                        <strong>{roleLabel}</strong>
                        <span>
                          {part.role === 'unknown'
                            ? t('角色待校订')
                            : `角色置信度 ${Math.round(
                                (part.roleConfidence ?? 0) * 100,
                              )}%`}
                        </span>
                      </header>
                      <FullScore
                        project={{
                          ...displayedProject,
                          tab:
                            part.id === project.primaryGuitarPartId
                              ? project.tab
                              : part.tab,
                          transcriptionModel:
                            part.transcriptionModel ??
                            project.transcriptionModel,
                        }}
                        positionSnapshot={position}
                        playbackPositionRef={playbackPositionRef}
                        playing={playing}
                        speed={speed}
                        expanded={scoreExpanded}
                        language={language}
                        capo={activeCapo}
                        onSelectChord={() => undefined}
                        onSelectBeat={() => undefined}
                        onSelectNote={() => undefined}
                      />
                    </section>
                  )
                })}
              </div>
            ) : (
              <FullScore
                project={displayedProject}
                positionSnapshot={position}
                playbackPositionRef={playbackPositionRef}
                playing={playing}
                speed={speed}
                expanded={scoreExpanded}
                language={language}
                capo={activeCapo}
                selectedMeasureNumber={selectedMeasureNumber}
                selectedNoteId={selectedNoteId}
                onSelectChord={selectChord}
                onSelectBeat={selectBeat}
                onSelectNote={selectNote}
              />
            )
          ) : (
            <div className="part-empty-state">
              <Music2 aria-hidden="true" />
              <strong>{t('贝斯谱尚未生成')}</strong>
              <span>当前项目已保留对应分轨，可继续播放、静音和独奏。</span>
            </div>
          )}
        </section>

        <section className="mixer-section">
          <header className="section-heading">
            <div>
              <span className="eyebrow">{t('实时练习')}</span>
              <h2>{t('音轨混音')}</h2>
            </div>
            <span className="quiet-label">
              <SlidersHorizontal aria-hidden="true" />
              {project.analysisMode === 'inference'
                ? language === 'en-US'
                  ? `${visibleTracks.length} active tracks`
                  : `${visibleTracks.length} 条有效音轨`
                : language === 'en-US'
                  ? `${visibleTracks.length} separate tracks`
                  : `${visibleTracks.length} 条独立分轨`}
            </span>
          </header>
          <div className="mixer-grid">
            {visibleTracks.map((track) => {
              const mix = mixes[track.id]
              const hasRoleSiblings =
                visibleTracks.filter((candidate) => candidate.role === track.role)
                  .length > 1
              const trackName =
                hasRoleSiblings || language === 'zh-CN'
                  ? track.name
                  : t(trackRoleLabels[track.role] ?? track.name)
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
                      aria-label={
                        language === 'en-US'
                          ? `${trackName} ${mix?.muted ? 'unmute' : 'mute'}`
                          : `${trackName}${mix?.muted ? '取消静音' : '静音'}`
                      }
                      title={t('静音')}
                      onClick={() => updateMix(track.id, { muted: !mix?.muted })}
                    >
                      {mix?.muted ? <VolumeX aria-hidden="true" /> : <Volume2 aria-hidden="true" />}
                    </button>
                    <button
                      className={mix?.solo ? 'is-active' : ''}
                      type="button"
                      aria-label={
                        language === 'en-US'
                          ? `${trackName} ${mix?.solo ? 'unsolo' : 'solo'}`
                          : `${trackName}${mix?.solo ? '取消独奏' : '独奏'}`
                      }
                      title={t('独奏')}
                      onClick={() => updateMix(track.id, { solo: !mix?.solo })}
                    >
                      <Headphones aria-hidden="true" />
                    </button>
                    <input
                      aria-label={
                        language === 'en-US'
                          ? `${trackName} volume`
                          : `${trackName}音量`
                      }
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

      {workspaceView === 'analysis' && (selectedNote || selectedMeasure) ? (
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
                        : selectedNote.positionSource?.startsWith(
                              'onset-position',
                            )
                          ? '弦品模型'
                          : selectedNote.positionSource?.startsWith(
                                'audio-spectrum',
                              )
                            ? '频谱支持 · 弦品待复核'
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
          <section
            className="import-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={t('导入歌曲')}
          >
            <header>
              <div>
                <span className="eyebrow">{t('新建分析任务')}</span>
                <h2>{t('导入歌曲')}</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label={t('关闭窗口')}
                onClick={() => {
                  setImportOpen(false)
                  setProtectedImport(null)
                  setMediaSessionRecovery(null)
                  if (!job || jobTerminal) setJob(null)
                }}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            {protectedImport ? (
              <div className="protected-media-consent">
                <CircleAlert aria-hidden="true" />
                <div>
                  <span className="eyebrow">{t('受保护媒体')}</span>
                  <h3>{t('版权与处理确认')}</h3>
                  <p>
                    已识别为
                    {protectedProviderLabels[protectedImport.provider]}
                    {protectedImport.kind === 'upload'
                      ? ` 加密文件“${protectedImport.file.name}”`
                      : ' 可能受保护的媒体链接'}
                    。继续前请确认你拥有该媒体的版权、有效许可或其他合法处理权。
                  </p>
                </div>
                <dl>
                  <div>
                    <dt>{t('用途限制')}</dt>
                    <dd>{t('仅用于本次音轨分析和曲谱生成')}</dd>
                  </div>
                  <div>
                    <dt>{t('临时处理')}</dt>
                    <dd>{t('解码结果不提供下载，任务结束后删除')}</dd>
                  </div>
                  <div>
                    <dt>{t('责任说明')}</dt>
                    <dd>{t('工具不核验或授予权利，权利责任由用户承担')}</dd>
                  </div>
                  {protectedImport.provider === 'qqmusic' &&
                    desktopControls.selectLocalFile && (
                      <div>
                        <dt>账号授权</dt>
                        <dd>新版 QMC2 会使用本机登录态向 QQ 音乐官方接口验证当前账号权限，凭据与密钥不保存</dd>
                      </div>
                    )}
                </dl>
                <p className="consent-notice">
                  点击“确认并继续”即表示你已理解并同意上述内容，并对所提交媒体的合法来源及处理权限负责。
                </p>
                <p
                  className={`decoder-status${
                    capabilities?.localAuthorizedMediaDecoder ? ' is-ready' : ''
                  }`}
                >
                  {capabilities?.localAuthorizedMediaDecoder
                    ? desktopControls.selectLocalFile
                      ? '桌面处理端离线解码器已就绪'
                      : 'API 服务端离线解码器已就绪'
                    : desktopControls.selectLocalFile
                      ? '桌面处理端离线解码器不可用'
                      : 'API 服务端离线解码器不可用'}
                </p>
                <div className="protected-media-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      setProtectedImport(null)
                      if (fileInputRef.current) {
                        fileInputRef.current.value = ''
                      }
                    }}
                  >
                    {t('取消')}
                  </button>
                  <button
                    type="button"
                    className="primary-button"
                    disabled={!capabilities?.localAuthorizedMediaDecoder}
                    onClick={() => {
                      if (protectedImport.kind === 'upload') {
                        void startUpload(protectedImport.file, true)
                      } else {
                        void startLinkImport(
                          true,
                          protectedImport.url,
                        )
                      }
                    }}
                  >
                    {t('确认并继续')}
                  </button>
                </div>
              </div>
            ) : scoreImportBusy ? (
              <div className="job-progress">
                <span className="job-icon">
                  <LoaderCircle className="spin" aria-hidden="true" />
                </span>
                <strong>{t('正在识别曲谱')}</strong>
                <p>{t('检测谱线、小节、弦品、和弦与文字')}</p>
                <i
                  role="progressbar"
                  aria-label={t('曲谱识别进度')}
                  className="is-animated is-indeterminate"
                >
                  <b />
                </i>
              </div>
            ) : job ? (
              <div className="job-progress">
                <span className="job-icon">
                  {job.status === 'completed' || job.status === 'degraded' ? (
                    <Check aria-hidden="true" />
                  ) : job.status === 'cancelled' ||
                    job.status === 'failed' ? (
                    <X aria-hidden="true" />
                  ) : (
                    <LoaderCircle className="spin" aria-hidden="true" />
                  )}
                </span>
                <strong>{job.stageLabel}</strong>
                <p>
                  {job.status === 'degraded'
                    ? t('已生成可编辑演示项目')
                    : t('分析继续在后台进行，可以关闭此窗口')}
                </p>
                <i
                  role="progressbar"
                  aria-label={t('歌曲分析进度')}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(Math.max(0, Math.min(100, job.progress)))}
                  className={
                    job.status === 'completed' ||
                    job.status === 'degraded' ||
                    job.status === 'failed' ||
                    job.status === 'cancelled'
                      ? ''
                      : 'is-animated'
                  }
                >
                  <b style={{ width: `${job.progress}%` }} />
                </i>
                <span className="job-progress-value" aria-live="polite">
                  {Math.round(Math.max(0, Math.min(100, job.progress)))}%
                </span>
                {!jobTerminal && (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() =>
                      void application.cancelJob(job.id).then(setJob).catch(
                        (cancelError: unknown) =>
                          setError(
                            cancelError instanceof Error
                              ? cancelError.message
                              : '取消任务失败',
                          ),
                      )
                    }
                  >
                    <X aria-hidden="true" />
                    {t('取消任务')}
                  </button>
                )}
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
                    {t('本地上传')}
                  </button>
                  <button
                    type="button"
                    className={importMode === 'score' ? 'is-active' : ''}
                    onClick={() => setImportMode('score')}
                  >
                    <FileImage aria-hidden="true" />
                    {t('图片 / PDF')}
                  </button>
                  <button
                    type="button"
                    className={importMode === 'link' ? 'is-active' : ''}
                    onClick={() => setImportMode('link')}
                  >
                    <Link2 aria-hidden="true" />
                    {t('在线链接')}
                  </button>
                </div>
                {importMode === 'upload' ? (
                  desktopControls.selectLocalFile ? (
                    <div className="desktop-media-picker">
                      <button
                        className="drop-zone"
                        type="button"
                        onClick={() => void selectDesktopFile('media')}
                      >
                        <FileAudio aria-hidden="true" />
                        <strong>{t('选择音频或视频')}</strong>
                        <span>MP3、WAV、FLAC、APE、M4A、AAC、OGG、MP4、MOV、WEBM，最大 500 MB</span>
                        <small>使用系统原生文件窗口；加密容器经授权确认后在本机临时解码。</small>
                      </button>
                      <div
                        className="media-source-shortcuts"
                        aria-label={t('音乐平台下载目录')}
                      >
                        {desktopMediaSourceOptions.map((source) => (
                          <button
                            className="media-source-button"
                            type="button"
                            onClick={() =>
                              void selectDesktopFile('media', source.id)
                            }
                            key={source.id}
                          >
                            <FolderOpen aria-hidden="true" />
                            {source.label}
                          </button>
                        ))}
                      </div>
                      {desktopControls.importMediaKeyDatabase ? (
                        <>
                          <div
                            className="media-source-shortcuts"
                            aria-label="分离密钥库"
                          >
                            <button
                              className="media-source-button"
                              type="button"
                              onClick={() =>
                                void importMediaKeyDatabase('kuwo')
                              }
                            >
                              <KeyRound aria-hidden="true" />
                              导入酷我密钥库
                            </button>
                            <button
                              className="media-source-button"
                              type="button"
                              onClick={() =>
                                void importMediaKeyDatabase('kugou')
                              }
                            >
                              <KeyRound aria-hidden="true" />
                              导入酷狗密钥库
                            </button>
                          </div>
                          {mediaKeyNotice ? (
                            <p className="policy-copy">{mediaKeyNotice}</p>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  ) : (
                    <label className="drop-zone">
                      <FileAudio aria-hidden="true" />
                      <strong>{t('选择音频或视频')}</strong>
                      <span>MP3、WAV、FLAC、APE、M4A、AAC、OGG、MP4、MOV、WEBM，最大 500 MB</span>
                      <small>加密容器经确认后由 API 服务端临时解码；无内嵌密钥的新版 QMC2 需使用 macOS 或 Windows 桌面版。</small>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".mp3,.wav,.flac,.ape,.m4a,.aac,.ogg,.mp4,.mov,.webm,.qmc0,.qmc2,.qmc3,.qmcflac,.qmcogg,.tkm,.tm0,.tm2,.tm3,.tm6,.bkcmp3,.bkcflac,.mflac,.mflac0,.mgg,.mgg0,.mgg1,.ncm,.kwm,.kgg,.kgm,.kgma,.vpr,audio/*,video/*"
                        onChange={(event) => {
                          const file = event.target.files?.[0]
                          if (file) void requestUpload(file)
                        }}
                      />
                    </label>
                  )
                ) : importMode === 'score' ? (
                  desktopControls.selectLocalFile ? (
                    <button
                      className="drop-zone"
                      type="button"
                      onClick={() => void selectDesktopFile('score')}
                    >
                      <FileImage aria-hidden="true" />
                      <strong>{t('选择吉他谱图片或 PDF')}</strong>
                      <span>PNG、JPG、WEBP、PDF，最多 24 页</span>
                      <small>使用系统原生文件窗口；导入后可逐音修改并直接合成播放。</small>
                    </button>
                  ) : (
                    <label className="drop-zone">
                      <FileImage aria-hidden="true" />
                      <strong>{t('选择吉他谱图片或 PDF')}</strong>
                      <span>PNG、JPG、WEBP、PDF，最多 24 页</span>
                      <small>识别六线谱、小节、品位、和弦与歌词；导入后可逐音修改并直接合成播放。</small>
                      <input
                        type="file"
                        accept=".png,.jpg,.jpeg,.webp,.pdf,image/png,image/jpeg,image/webp,application/pdf"
                        onChange={(event) => {
                          const file = event.target.files?.[0]
                          if (file) void startScoreUpload(file)
                        }}
                      />
                    </label>
                  )
                ) : (
                  <div className="link-import">
                    <label>
                      <span>{t('歌曲链接')}</span>
                      <div>
                        <Link2 aria-hidden="true" />
                        <input
                          value={linkValue}
                          onChange={(event) => {
                            setLinkValue(event.target.value)
                            setError('')
                            setMediaSessionRecovery(null)
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
                          placeholder="粘贴 YouTube 或其他受支持的公开音视频链接"
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
                      onClick={() => void requestLinkImport()}
                    >
                      {t('开始分析')}
                    </button>
                    {linkValue.trim() ? (
                      <p className={`source-detection${detectedPlatform ? ' is-valid' : ''}`}>
                        {detectedPlatform
                          ? `已识别：${detectedPlatform.label}`
                          : t('尚未识别链接来源')}
                      </p>
                    ) : null}
                    <p className="policy-copy">
                      在线导入取决于平台能否返回可解码媒体流；登录、验证、地区限制或 DRM 会显示具体原因。
                    </p>
                  </div>
                )}
              </>
            )}
            {error ? (
              <>
                <p className="dialog-error">{error}</p>
                {mediaSessionRecovery ? (
                  <div className="media-session-recovery">
                    <Cookie aria-hidden="true" />
                    {mediaSessionRecovery.browser ? (
                      <>
                        <div>
                          <strong>
                            {t('使用系统默认浏览器登录状态')}
                          </strong>
                          <p>
                            {language === 'en-US'
                              ? `Read ${mediaSessionRecovery.browser.label}'s default profile cookies for this retry only. Cookies are not stored or uploaded.`
                              : `仅为本次重试读取 ${mediaSessionRecovery.browser.label} 默认配置中的站点 Cookie，不保存也不上传。`}
                          </p>
                        </div>
                        <div className="media-session-actions">
                          {desktopControls.openExternalUrl ? (
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() =>
                                void desktopControls.openExternalUrl?.(
                                  mediaSessionRecovery.url,
                                )
                              }
                            >
                              <ExternalLink aria-hidden="true" />
                              {t('打开原链接登录')}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="secondary-button"
                            onClick={() => setMediaSessionRecovery(null)}
                          >
                            {t('取消')}
                          </button>
                          <button
                            type="button"
                            className="primary-button"
                            onClick={() =>
                              void retryWithDefaultBrowserSession()
                            }
                          >
                            {t('确认并重试')}
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <strong>{t('此链接需要登录或验证')}</strong>
                          <p>
                            {t(
                              '应用尚未读取任何浏览器数据。确认后将识别系统默认浏览器，再由你决定是否仅为本次任务读取登录状态。',
                            )}
                          </p>
                        </div>
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={mediaSessionRecovery.detecting}
                          onClick={() => void detectMediaSessionForRetry()}
                        >
                          {mediaSessionRecovery.detecting ? (
                            <LoaderCircle className="spin" aria-hidden="true" />
                          ) : (
                            <Cookie aria-hidden="true" />
                          )}
                          {t('使用默认浏览器登录状态重试')}
                        </button>
                      </>
                    )}
                  </div>
                ) : null}
                {error.includes('完全磁盘访问权限') &&
                desktopControls.openFullDiskAccessSettings ? (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() =>
                      void desktopControls
                        .openFullDiskAccessSettings?.()
                        .catch((settingsError: unknown) =>
                          setError(
                            settingsError instanceof Error
                              ? settingsError.message
                              : '无法打开 macOS 权限设置',
                          ),
                        )
                    }
                  >
                    <HardDrive aria-hidden="true" />
                    打开权限设置
                  </button>
                ) : null}
              </>
            ) : null}
          </section>
        </div>
      ) : null}

      {settingsOpen ? (
        <div className="modal-backdrop" role="presentation">
          <section
            className="settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={t('应用设置')}
          >
            <header>
              <div>
                <span className="eyebrow">Guitar Tab Studio</span>
                <h2>{t('应用设置')}</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label={t('关闭窗口')}
                onClick={() => setSettingsOpen(false)}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            <div className="settings-language">
              <div>
                <Languages aria-hidden="true" />
                <span>
                  <strong>{t('界面语言')}</strong>
                  <small>
                    {t('系统语言')}:{' '}
                    {detectedSystemLanguage === 'zh-CN'
                      ? t('简体中文')
                      : 'English'}
                  </small>
                </span>
              </div>
              <div
                className="language-segmented"
                role="group"
                aria-label={t('界面语言')}
              >
                {(
                  [
                    ['system', '跟随系统'],
                    ['zh-CN', '简体中文'],
                    ['en-US', 'English'],
                  ] as const
                ).map(([preference, label]) => (
                  <button
                    type="button"
                    className={
                      languagePreference === preference ? 'is-active' : ''
                    }
                    aria-pressed={languagePreference === preference}
                    disabled={settingsBusy}
                    onClick={() => void changeLanguage(preference)}
                    key={preference}
                  >
                    {t(label)}
                  </button>
                ))}
              </div>
              <p>{t('设置会立即应用并在下次启动时保留。')}</p>
              {settingsError ? (
                <p className="settings-error" role="alert">
                  {settingsError}
                </p>
              ) : null}
            </div>
            <footer>
              <button
                className="primary-button"
                type="button"
                onClick={() => setSettingsOpen(false)}
              >
                {t('完成')}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {runtimeOpen && desktopControls.runtimeStatus ? (
        <div className="modal-backdrop" role="presentation">
          <section
            className="runtime-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="桌面运行时"
          >
            <header>
              <div>
                <span className="eyebrow">Desktop runtime</span>
                <h2>运行模型与存储</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label="关闭"
                onClick={() => setRuntimeOpen(false)}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            <div
              className={`runtime-mode ${
                runtimeLocation === 'cloud' ? 'is-cloud' : ''
              }`}
            >
              <div className="runtime-control runtime-location-control">
                <span>执行位置</span>
                <div role="group" aria-label="执行位置">
                  <button
                    type="button"
                    className={
                      runtimeLocation === 'local' ? 'is-active' : ''
                    }
                    disabled={runtimeBusy || !desktopControls.setRuntimeMode}
                    onClick={() => {
                      setRuntimeLocation('local')
                      setRuntimeError('')
                      setRuntimeNotice('')
                      if (runtimeStatus?.providerMode === 'local') return
                      void runDesktopAction(() =>
                        desktopControls.setRuntimeMode!('local', project?.id),
                      )
                    }}
                  >
                    <Laptop aria-hidden="true" />
                    本地
                  </button>
                  <button
                    type="button"
                    className={
                      runtimeLocation === 'cloud' ? 'is-active' : ''
                    }
                    disabled={
                      runtimeBusy ||
                    (!runtimeStatus?.cloudReady &&
                      !desktopControls.configureCloudRuntime) ||
                    (runtimeStatus?.cloudReady &&
                      !desktopControls.setRuntimeMode)
                    }
                    title={cloudModeTitle}
                    onClick={() => {
                      setRuntimeLocation('cloud')
                      setRuntimeError('')
                      setRuntimeNotice('')
                      if (!runtimeStatus?.cloudReady) return
                      if (runtimeStatus.providerMode === 'cloud') return
                      if (
                        !window.confirm(
                          '云端推理会将待分析音频上传到已配置服务。是否切换？',
                        )
                      ) {
                        return
                      }
                      void runDesktopAction(() =>
                        desktopControls.setRuntimeMode!(
                          'cloud',
                          project?.id,
                        ),
                      )
                    }}
                  >
                    <Cloud aria-hidden="true" />
                    {cloudModeLabel}
                  </button>
                </div>
              </div>
              {runtimeLocation === 'local' ? (
                <>
                  {hasResearchTranscriptionOptions ? (
                    <>
                      <label className="runtime-control">
                        <span>转录方案</span>
                        <select
                          aria-label="转录方案"
                          value={transcriptionScheme}
                          disabled={
                            runtimeBusy ||
                            !runtimeStatus ||
                            !desktopControls.setTranscriptionProvider
                          }
                          onChange={(event) => {
                            const providerId =
                              event.target.value as TranscriptionProviderId
                            setTranscriptionScheme(providerId)
                            setTranscriptionProgress(null)
                            setTranscriptionSetupError('')
                            if (providerId !== 'basic-pitch') {
                              return
                            }
                            void runDesktopAction(() =>
                              desktopControls.setTranscriptionProvider!({
                                providerId: 'basic-pitch',
                              }),
                            )
                          }}
                        >
                          {runtimeStatus?.transcription.providers.map(
                            (provider) => (
                              <option value={provider.id} key={provider.id}>
                                {provider.label}
                              </option>
                            ),
                          )}
                        </select>
                      </label>
                      <label className="runtime-control">
                        <span>模型档位</span>
                        <select
                          aria-label="模型档位"
                          value={transcriptionChoice}
                          disabled={
                            runtimeBusy || transcriptionScheme !== 'muscriptor'
                          }
                          onChange={(event) => {
                            setTranscriptionChoice(
                              event.target.value as
                                | 'auto'
                                | TranscriptionModelVariant,
                            )
                            setTranscriptionSetupError('')
                          }}
                        >
                          <option value="auto">
                            自动 ·{' '}
                            {runtimeStatus?.transcription.recommendation.variant.toUpperCase() ??
                              '检测中'}
                          </option>
                          <option value="small">Small · 103M</option>
                          <option value="medium">Medium · 307M</option>
                          <option value="large">Large · 1.4B</option>
                        </select>
                      </label>
                    </>
                  ) : (
                    <div className="runtime-mode-summary runtime-engine-summary">
                      <span>转录引擎</span>
                      <strong>Basic Pitch</strong>
                      <small>生产默认 · 内置运行 · 无需额外权重</small>
                    </div>
                  )}
                  <p>
                    {runtimeStatus
                      ? `${runtimeStatus.resourceNote} · 预计 ${Math.ceil(runtimeStatus.estimatedSeconds.minimum / 60)}–${Math.ceil(runtimeStatus.estimatedSeconds.maximum / 60)} 分钟`
                      : '正在检查运行环境'}
                  </p>
                </>
              ) : (
                <div className="runtime-mode-summary" aria-live="polite">
                  <span>云端推理</span>
                  <strong>
                    {runtimeStatus?.cloudReady
                      ? runtimeStatus.providerMode === 'cloud'
                        ? '当前项目正在使用'
                        : '服务已连接'
                      : runtimeStatus?.cloudAvailable
                        ? '连接异常'
                        : '等待配置'}
                  </strong>
                  <small>音频由云端处理，项目与编辑结果保留在本机</small>
                </div>
              )}
            </div>
            {runtimeLocation === 'cloud' ? (
              <section
                className="runtime-cloud-panel"
                aria-label="云端运行配置"
              >
                <div className="runtime-cloud-heading">
                  <span>
                    <strong>云端 API</strong>
                    <small>
                      仅接受 HTTPS 地址，保存前会验证服务能力
                    </small>
                  </span>
                  <i
                    className={
                      runtimeStatus?.cloudReady
                        ? 'is-ready'
                        : runtimeStatus?.cloudAvailable
                          ? 'is-error'
                          : ''
                    }
                  >
                    {runtimeStatus?.cloudReady
                      ? '已连接'
                      : runtimeStatus?.cloudAvailable
                        ? '连接失败'
                        : '未配置'}
                  </i>
                </div>
                <form
                  className="runtime-cloud-config"
                onSubmit={(event) => {
                  event.preventDefault()
                  void runDesktopAction(async () => {
                    await desktopControls.configureCloudRuntime?.(
                      cloudApiRoot.trim(),
                      project?.id,
                    )
                    setRuntimeNotice('云端服务连接成功，已切换为云端推理')
                  })
                }}
                >
                  <label>
                    <span>服务地址</span>
                    <input
                      type="url"
                      required
                      aria-label="云端 API 地址"
                      placeholder="https://api.example.com"
                      value={cloudApiRoot}
                      disabled={runtimeBusy}
                      onChange={(event) =>
                        setCloudApiRoot(event.target.value)
                      }
                    />
                  </label>
                  <button
                    type="submit"
                    className="primary-button"
                    disabled={
                      runtimeBusy ||
                      !desktopControls.configureCloudRuntime ||
                      !cloudApiRoot.trim()
                    }
                  >
                    <Cloud aria-hidden="true" />
                    {runtimeStatus?.cloudReady &&
                    runtimeStatus.providerMode === 'cloud'
                      ? '重新验证并保存'
                      : '保存并切换'}
                  </button>
                </form>
                <dl className="runtime-cloud-features">
                  <div>
                    <dt>连接校验</dt>
                    <dd>HTTPS · /health 能力检查</dd>
                  </div>
                  <div>
                    <dt>音频处理</dt>
                    <dd>待分析音频上传至已配置服务</dd>
                  </div>
                  <div>
                    <dt>项目数据</dt>
                    <dd>曲谱、编辑记录与项目文件保留本机</dd>
                  </div>
                  <div>
                    <dt>应用范围</dt>
                    <dd>{project ? '当前项目' : '后续新项目默认值'}</dd>
                  </div>
                </dl>
                {runtimeStatus?.cloudError ? (
                  <p className="runtime-warning">
                    云端连接失败，本地项目与已下载模型不受影响：
                    {runtimeStatus.cloudError}
                  </p>
                ) : null}
              </section>
            ) : null}
            <section
              className="runtime-transcription"
              hidden={runtimeLocation !== 'local'}
            >
              {transcriptionScheme === 'muscriptor' &&
              highAccuracyProviderAvailable ? (
                <div className="runtime-transcription-setup">
                  <div className="runtime-model-recommendation">
                    <strong>
                      推荐{' '}
                      {runtimeStatus.transcription.recommendation.variant.toUpperCase()}
                    </strong>
                    <span>
                      {runtimeStatus.transcription.recommendation.reason}
                    </span>
                    <small>
                      {runtimeStatus.transcription.device.accelerator.toUpperCase()}{' '}
                      ·{' '}
                      {formatStorageSize(
                        runtimeStatus.transcription.device.totalMemoryBytes,
                      )}{' '}
                      内存 ·{' '}
                      {runtimeStatus.transcription.device.logicalCpuCount} 线程
                    </small>
                  </div>
                  <p className="runtime-model-state">
                    {highAccuracySelectionActive
                      ? `已启用研究融合 · ${
                          selectedTranscriptionVariant?.toUpperCase() ??
                          '自动档位'
                        }`
                      : !highAccuracyRuntimeAvailable
                      ? '当前 Worker 未包含 MuScriptor 运行组件'
                      : highAccuracyDownloadRequired
                        ? `尚未准备，启用时将从官方模型源下载并校验${
                            highAccuracyDownloadBytes > 0
                              ? `（约 ${formatStorageSize(
                                  highAccuracyDownloadBytes,
                                )}）`
                              : ''
                          }`
                        : selectedTranscriptionWeightSource ===
                          'local-import'
                        ? '已导入本地权重'
                        : selectedTranscriptionWeightSource ===
                            'managed-cache'
                          ? '推荐档位权重已缓存'
                          : '研究模型已准备'}
                  </p>
                  {selectedVariantExceedsMemory ? (
                    <p className="runtime-warning">
                      所选档位高于当前硬件建议，仍可手动下载并启用，但分析可能明显变慢或因内存不足失败。
                    </p>
                  ) : null}
                  {mrMt3Fallback ? (
                    <p className="runtime-warning">
                      当前 Worker 未包含 MR-MT3，将使用 MuScriptor、Basic Pitch 与 Noise2Fret 完成分析。
                    </p>
                  ) : null}
                  {transcriptionProgress &&
                  transcriptionProgress.phase !== 'ready' ? (
                    <div className="runtime-model-progress" aria-live="polite">
                      <div>
                        <span>{transcriptionProgress.stageLabel}</span>
                        <strong>
                          {Math.round(transcriptionProgress.progress)}%
                        </strong>
                      </div>
                      <progress
                        max="100"
                        value={transcriptionProgress.progress}
                      />
                      {transcriptionProgress.phase === 'authorizing' ? (
                        <small>
                          已打开系统浏览器
                          {transcriptionProgress.userCode
                            ? ` · 验证码 ${transcriptionProgress.userCode}`
                            : ''}
                        </small>
                      ) : null}
                    </div>
                  ) : null}
                  {transcriptionSetupError ? (
                    <div className="runtime-model-error" role="alert">
                      <CircleAlert aria-hidden="true" />
                      <span>
                        <strong>启用失败</strong>
                        <small>{transcriptionSetupError}</small>
                      </span>
                    </div>
                  ) : null}
                  <p className="runtime-model-stack">
                    <strong>当前组合</strong>
                    <span>
                      MuScriptor 主转录 ·{' '}
                      {mrMt3Selected ? 'MR-MT3 与 ' : ''}Basic Pitch 复核补漏 ·
                      Noise2Fret 弦品优化
                    </span>
                  </p>
                  {selectedTranscriptionWeightSource ===
                    'download-required' &&
                  (!runtimeStatus.transcription.oauthConfigured ||
                    oauthSetup) ? (
                    <form
                      className="runtime-inline-config runtime-oauth-config"
                      onSubmit={(event) => {
                        event.preventDefault()
                        void runDesktopAction(
                          async () => {
                            await desktopControls.configureHuggingFaceOAuth?.(
                              huggingFaceOAuthClientId.trim(),
                            )
                            setHuggingFaceOAuthClientId('')
                            setOauthSetup(false)
                            setRuntimeNotice(
                              'Hugging Face OAuth 已配置，可以开始授权下载',
                            )
                          },
                          {
                            errorScope: 'transcription',
                            preserveTranscriptionScheme: true,
                          },
                        )
                      }}
                    >
                      <div className="runtime-oauth-intro">
                        <span>
                          <strong>OAuth 应用配置</strong>
                          <small>
                            此处填写应用的 Client ID，不是浏览器返回的
                            Authorization Code
                          </small>
                        </span>
                        <a
                          className="secondary-button"
                          href={huggingFaceOAuthApplicationUrl}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <ExternalLink aria-hidden="true" />
                          创建 OAuth 应用
                        </a>
                      </div>
                      <div className="runtime-oauth-redirect">
                        <span>应用类型</span>
                        <strong>Public · 不创建 Client Secret</strong>
                        <span>Redirect URI</span>
                        <code>{huggingFaceOAuthRedirectUri}</code>
                      </div>
                      <label>
                        <span>OAuth 客户端 ID</span>
                        <input
                          type="text"
                          required
                          aria-label="Hugging Face OAuth Client ID"
                          placeholder="创建应用后复制 Client ID"
                          value={huggingFaceOAuthClientId}
                          disabled={runtimeBusy}
                          onChange={(event) =>
                            setHuggingFaceOAuthClientId(event.target.value)
                          }
                        />
                      </label>
                      <button
                        type="submit"
                        className="secondary-button"
                        disabled={
                          runtimeBusy ||
                          !desktopControls.configureHuggingFaceOAuth ||
                          !huggingFaceOAuthClientId.trim()
                        }
                      >
                        <KeyRound aria-hidden="true" />
                        保存 OAuth 配置
                      </button>
                    </form>
                  ) : selectedTranscriptionWeightSource ===
                      'download-required' ? (
                    <div className="runtime-oauth-ready">
                      <span>Hugging Face OAuth 已配置</span>
                      <button
                        type="button"
                        disabled={runtimeBusy}
                        onClick={() => setOauthSetup(true)}
                      >
                        修改
                      </button>
                    </div>
                  ) : null}
                  <label className="runtime-license-confirmation">
                    <input
                      type="checkbox"
                      checked={transcriptionLicenseAccepted}
                      disabled={
                        runtimeBusy ||
                        (Boolean(
                          runtimeStatus.transcription.preference
                            .licenseAcceptedAt,
                        ) &&
                          runtimeStatus.transcription.preference
                            .licenseAcceptedVersion ===
                            'CC-BY-NC-4.0')
                      }
                      onChange={(event) =>
                        setTranscriptionLicenseAccepted(event.target.checked)
                      }
                    />
                    <span>
                      我确认 MuScriptor 官方权重采用 CC BY-NC 4.0，仅用于非商用场景
                    </span>
                  </label>
                  <div className="runtime-model-actions">
                    {transcriptionAuthorizationActive ? (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={
                          !desktopControls.openTranscriptionAuthorization
                        }
                        onClick={() =>
                          void desktopControls
                            .openTranscriptionAuthorization?.()
                            .catch((authorizationError: unknown) =>
                              setRuntimeError(
                                authorizationError instanceof Error
                                  ? authorizationError.message
                                  : '无法重新打开登录页',
                              ),
                            )
                        }
                      >
                        <Link2 aria-hidden="true" />
                        重新打开登录页
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={
                          runtimeBusy ||
                          !desktopControls.openTranscriptionProviderPage
                        }
                        onClick={() =>
                          void desktopControls.openTranscriptionProviderPage?.(
                            transcriptionChoice === 'auto'
                              ? runtimeStatus.transcription.recommendation
                                  .variant
                              : transcriptionChoice,
                          )
                        }
                      >
                        <Link2 aria-hidden="true" />
                        官方授权页
                      </button>
                    )}
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={
                        runtimeBusy ||
                        !transcriptionLicenseAccepted ||
                        !desktopControls.importTranscriptionWeights
                      }
                      onClick={() =>
                        void runDesktopAction(
                          async () => {
                            const variant =
                              transcriptionChoice === 'auto'
                                ? runtimeStatus.transcription.recommendation
                                    .variant
                                : transcriptionChoice
                            const imported =
                              await desktopControls.importTranscriptionWeights?.(
                                variant,
                                transcriptionLicenseAccepted,
                              )
                            if (imported) {
                              await desktopControls.setTranscriptionProvider?.({
                                providerId: 'muscriptor',
                                selectionMode:
                                  transcriptionChoice === 'auto'
                                    ? 'auto'
                                    : 'manual',
                                ...(transcriptionChoice === 'auto'
                                  ? {}
                                  : { variant }),
                                licenseAccepted:
                                  transcriptionLicenseAccepted,
                              })
                              setRuntimeNotice(
                                `MuScriptor ${variant.toUpperCase()} 权重已导入`,
                              )
                            }
                          },
                          { errorScope: 'transcription' },
                        )
                      }
                    >
                      <FolderOpen aria-hidden="true" />
                      导入权重
                    </button>
                    {transcriptionSetupActive ? (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={!desktopControls.cancelTranscriptionSetup}
                        onClick={() => {
                          setRuntimeNotice('正在取消模型准备')
                          void desktopControls
                            .cancelTranscriptionSetup?.()
                            .then((result) => {
                              if (result.cancelled) return
                              setRuntimeBusy(false)
                              setTranscriptionProgress(null)
                              setRuntimeNotice('模型准备已结束，可以重新尝试')
                            })
                            .catch((cancelError: unknown) =>
                              setRuntimeError(
                                cancelError instanceof Error
                                  ? cancelError.message
                                  : '无法取消模型准备',
                              ),
                            )
                        }}
                      >
                        <X aria-hidden="true" />
                        {transcriptionAuthorizationActive
                          ? '取消等待'
                          : '取消下载'}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="primary-button"
                        disabled={
                          runtimeBusy ||
                          !transcriptionLicenseAccepted ||
                          (selectedTranscriptionWeightSource ===
                            'download-required' &&
                            !runtimeStatus.transcription.oauthConfigured) ||
                          !highAccuracyRuntimeAvailable ||
                          highAccuracySelectionActive ||
                          !desktopControls.setTranscriptionProvider
                        }
                        onClick={() =>
                          void runDesktopAction(
                            async () => {
                              setTranscriptionProgress({
                                phase: 'checking',
                                progress: 1,
                                stageLabel: '正在检查研究模型和登录状态',
                              })
                              setRuntimeNotice(
                                highAccuracyDownloadRequired
                                  ? '正在检查并下载研究模型，完成前不会切换转录策略'
                                  : '正在检查研究模型完整性',
                              )
                              await desktopControls.setTranscriptionProvider?.({
                                providerId: 'muscriptor',
                                selectionMode:
                                  transcriptionChoice === 'auto'
                                    ? 'auto'
                                    : 'manual',
                                ...(transcriptionChoice === 'auto'
                                  ? {}
                                  : { variant: transcriptionChoice }),
                                licenseAccepted:
                                  transcriptionLicenseAccepted,
                              })
                              setRuntimeNotice(
                                `研究融合已启用，当前使用 ${
                                  transcriptionChoice === 'auto'
                                    ? `自动推荐的 ${runtimeStatus.transcription.recommendation.variant.toUpperCase()}`
                                    : transcriptionChoice.toUpperCase()
                                }`,
                              )
                            },
                            { errorScope: 'transcription' },
                          )
                        }
                      >
                        <Check aria-hidden="true" />
                        {runtimeBusy
                          ? '正在准备模型'
                          : highAccuracySelectionActive
                            ? '研究融合已启用'
                          : highAccuracyDownloadRequired
                            ? '下载并启用'
                            : '启用研究融合'}
                      </button>
                    )}
                  </div>
                </div>
              ) : transcriptionScheme === 'guitar-specialist' ? (
                <div className="runtime-transcription-setup">
                  <div className="runtime-model-recommendation">
                    <strong>吉他专用 · StringTrace</strong>
                    <span>
                      起音与弦品联合识别，随后执行可演奏性指法约束
                    </span>
                    <small>适用于分离后的吉他音轨 · 研究模式</small>
                  </div>
                  <p className="runtime-model-state">
                    {guitarSpecialistComponent?.ready
                      ? '吉他专用模型包已准备'
                      : '需要安装包含 StringTrace 转录模型的研究模型包'}
                  </p>
                  <p className="runtime-model-stack">
                    <strong>当前组合</strong>
                    <span>
                      StringTrace 起音/弦品联合模型 · 可演奏性指法解码
                    </span>
                  </p>
                  <div className="runtime-model-actions">
                    <button
                      type="button"
                      className="primary-button"
                      disabled={
                        runtimeBusy ||
                        !guitarSpecialistComponent?.runtimeAvailable ||
                        (!guitarSpecialistComponent?.ready &&
                          (!desktopControls.installModelPackage ||
                            !desktopControls.activateModelPackage)) ||
                        !desktopControls.setTranscriptionProvider
                      }
                      onClick={() =>
                        void runDesktopAction(async () => {
                          if (!guitarSpecialistComponent?.ready) {
                            const installed =
                              await desktopControls.installModelPackage?.()
                            if (!installed) return
                            await desktopControls.activateModelPackage?.(
                              installed.manifest.id,
                              installed.manifest.version,
                            )
                          }
                          await desktopControls.setTranscriptionProvider?.({
                            providerId: 'guitar-specialist',
                          })
                          setRuntimeNotice('吉他专用方案已启用')
                        })
                      }
                    >
                      <Guitar aria-hidden="true" />
                      {guitarSpecialistComponent?.ready
                        ? '启用吉他专用'
                        : '安装模型包并启用'}
                    </button>
                  </div>
                </div>
              ) : transcriptionScheme === 'multitrack-research' ? (
                <div className="runtime-transcription-setup">
                  <div className="runtime-model-recommendation">
                    <strong>多轨研究 · YourMT3+</strong>
                    <span>
                      多乐器事件转录，提取吉他声部后与 Basic Pitch 融合
                    </span>
                    <small>约 536 MB · 研究模式 · 首次使用自动下载</small>
                  </div>
                  <p className="runtime-model-state">
                    {multitrackResearchComponent?.ready
                      ? 'YourMT3+ 权重已缓存'
                      : multitrackResearchComponent?.runtimeAvailable
                        ? '尚未准备，启用时将下载并校验研究模型'
                        : '当前 Worker 未包含 YourMT3+ 运行组件'}
                  </p>
                  {transcriptionProgress &&
                  transcriptionProgress.phase !== 'ready' ? (
                    <div className="runtime-model-progress" aria-live="polite">
                      <div>
                        <span>{transcriptionProgress.stageLabel}</span>
                        <strong>
                          {Math.round(transcriptionProgress.progress)}%
                        </strong>
                      </div>
                      <progress
                        max="100"
                        value={transcriptionProgress.progress}
                      />
                    </div>
                  ) : null}
                  <p className="runtime-model-stack">
                    <strong>当前组合</strong>
                    <span>
                      YourMT3+ 多轨转录 · Basic Pitch 补漏 · Noise2Fret 弦品优化
                    </span>
                  </p>
                  <div className="runtime-model-actions">
                    {transcriptionSetupActive ? (
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={!desktopControls.cancelTranscriptionSetup}
                        onClick={() => {
                          setRuntimeNotice('正在取消模型准备')
                          void desktopControls.cancelTranscriptionSetup?.()
                        }}
                      >
                        <X aria-hidden="true" />
                        取消下载
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="primary-button"
                        disabled={
                          runtimeBusy ||
                          !multitrackResearchComponent?.runtimeAvailable ||
                          !desktopControls.setTranscriptionProvider
                        }
                        onClick={() =>
                          void runDesktopAction(async () => {
                            setTranscriptionProgress({
                              phase: 'checking',
                              progress: 1,
                              stageLabel: '正在检查多轨研究模型',
                            })
                            await desktopControls.setTranscriptionProvider?.({
                              providerId: 'multitrack-research',
                            })
                            setRuntimeNotice('多轨研究方案已启用')
                          })
                        }
                      >
                        <Check aria-hidden="true" />
                        {multitrackResearchComponent?.ready
                          ? '启用多轨研究'
                          : '下载并启用'}
                      </button>
                    )}
                  </div>
                </div>
              ) : null}
            </section>
            <dl
              className="runtime-status runtime-status-local"
              hidden={runtimeLocation !== 'local'}
            >
              <div>
                <dt>本地 API</dt>
                <dd>{runtimeStatus?.apiReady ? '可用' : '检查中'}</dd>
              </div>
              <div>
                <dt>项目存储</dt>
                <dd>{runtimeStatus?.storageReady ? 'SQLite' : '检查中'}</dd>
              </div>
              <div>
                <dt>推理 Worker</dt>
                <dd>{runtimeStatus?.workerReady ? '可用' : '未安装'}</dd>
              </div>
              <div>
                <dt>活动模型</dt>
                <dd>
                  {runtimeStatus?.activeModel
                    ? `${runtimeStatus.activeModel.id} ${runtimeStatus.activeModel.version} · ${
                        runtimeStatus.activeModel.releaseChannel === 'production'
                          ? '生产'
                          : '研究'
                      }`
                    : '内置基线'}
                </dd>
              </div>
            </dl>
            {runtimeLocation === 'local' && runtimeStatus?.modelWarning ? (
              <p className="runtime-warning">{runtimeStatus.modelWarning}</p>
            ) : null}
            <div
              className="runtime-actions"
              hidden={runtimeLocation !== 'local'}
            >
              <button
                type="button"
                className="secondary-button"
                disabled={runtimeBusy || !desktopControls.installModelPackage}
                onClick={() =>
                  void runDesktopAction(async () => {
                    await desktopControls.installModelPackage?.()
                  })
                }
              >
                <PackagePlus aria-hidden="true" />
                安装模型包
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={runtimeBusy || !desktopControls.rollbackModelPackage}
                onClick={() =>
                  void runDesktopAction(async () => {
                    await desktopControls.rollbackModelPackage?.()
                  })
                }
              >
                <RotateCcw aria-hidden="true" />
                回滚模型
              </button>
            </div>
            <form
              className="runtime-download"
              hidden={runtimeLocation !== 'local'}
              onSubmit={(event) => {
                event.preventDefault()
                void runDesktopAction(async () => {
                  await desktopControls.downloadModelPackage?.(
                    modelManifestUrl.trim(),
                  )
                  setModelManifestUrl('')
                  setRuntimeNotice('模型包已下载并校验')
                })
              }}
            >
              <label>
                <span>模型清单地址</span>
                <input
                  type="url"
                  value={modelManifestUrl}
                  placeholder="https://…/manifest.json"
                  onChange={(event) => setModelManifestUrl(event.target.value)}
                />
              </label>
              <button
                type="submit"
                className="secondary-button"
                disabled={
                  runtimeBusy ||
                  !desktopControls.downloadModelPackage ||
                  !modelManifestUrl.trim()
                }
              >
                <Download aria-hidden="true" />
                下载
              </button>
            </form>
            <div
              className="runtime-package-list"
              hidden={runtimeLocation !== 'local'}
            >
              {modelPackages.map((item) => (
                <article key={`${item.manifest.id}:${item.manifest.version}`}>
                  <span>
                    <strong>{item.manifest.id}</strong>
                    <small>
                      {item.manifest.version} · {item.manifest.runtime.platform}{' '}
                      {item.manifest.runtime.arch} ·{' '}
                      {item.manifest.releaseChannel === 'production'
                        ? '生产'
                        : '研究'}
                    </small>
                  </span>
                  {item.active ? (
                    <i>当前</i>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={
                          runtimeBusy ||
                          !desktopControls.activateModelPackage
                        }
                        onClick={() =>
                          void runDesktopAction(() =>
                            desktopControls.activateModelPackage!(
                              item.manifest.id,
                              item.manifest.version,
                            ),
                          )
                        }
                      >
                        启用
                      </button>
                      <button
                        type="button"
                        className="icon-button"
                        aria-label={`删除模型 ${item.manifest.id} ${item.manifest.version}`}
                        disabled={
                          runtimeBusy ||
                          !desktopControls.removeModelPackage
                        }
                        onClick={() =>
                          void runDesktopAction(() =>
                            desktopControls.removeModelPackage!(
                              item.manifest.id,
                              item.manifest.version,
                            ),
                          )
                        }
                      >
                        <Trash2 aria-hidden="true" />
                      </button>
                    </>
                  )}
                </article>
              ))}
              {modelPackages.length === 0 && (
                <p>未安装额外模型包，当前使用内置基线。</p>
              )}
            </div>
            {runtimeLocation === 'local' &&
            storageCleanup &&
            desktopControls.setStorageCleanupConfig ? (
              <form
                className="runtime-cleanup"
                onSubmit={(event) => {
                  event.preventDefault()
                  void runDesktopAction(async () => {
                    const result =
                      await desktopControls.setStorageCleanupConfig?.({
                        enabled: cleanupEnabled,
                        triggerFreeBytes: Math.round(
                          Number(cleanupTriggerGiB) * 1024 ** 3,
                        ),
                        targetFreeBytes: Math.round(
                          Number(cleanupTargetGiB) * 1024 ** 3,
                        ),
                        retentionHours: Number(cleanupRetentionHours),
                      })
                    if (result) {
                      setStorageCleanup(result)
                      setRuntimeNotice('自动清理配置已保存')
                    }
                  })
                }}
              >
                <div className="runtime-cleanup-heading">
                  <span>
                    <strong>自动清理</strong>
                    <small>
                      可用 {formatStorageSize(storageCleanup.availableBytes)} ·
                      可回收{' '}
                      {formatStorageSize(storageCleanup.reclaimableBytes)}
                    </small>
                  </span>
                  <label className="runtime-cleanup-toggle">
                    <input
                      type="checkbox"
                      checked={cleanupEnabled}
                      disabled={runtimeBusy}
                      onChange={(event) =>
                        setCleanupEnabled(event.target.checked)
                      }
                    />
                    <i aria-hidden="true" />
                    <span>{cleanupEnabled ? '已启用' : '已关闭'}</span>
                  </label>
                </div>
                <div className="runtime-cleanup-fields">
                  <label>
                    <span>启动阈值</span>
                    <span>
                      <input
                        type="number"
                        min="0.25"
                        step="0.25"
                        value={cleanupTriggerGiB}
                        disabled={runtimeBusy}
                        onChange={(event) =>
                          setCleanupTriggerGiB(event.target.value)
                        }
                      />
                      GB
                    </span>
                  </label>
                  <label>
                    <span>清理目标</span>
                    <span>
                      <input
                        type="number"
                        min="0.25"
                        step="0.25"
                        value={cleanupTargetGiB}
                        disabled={runtimeBusy}
                        onChange={(event) =>
                          setCleanupTargetGiB(event.target.value)
                        }
                      />
                      GB
                    </span>
                  </label>
                  <label>
                    <span>保留时间</span>
                    <span>
                      <input
                        type="number"
                        min="0"
                        max="8760"
                        step="1"
                        value={cleanupRetentionHours}
                        disabled={runtimeBusy}
                        onChange={(event) =>
                          setCleanupRetentionHours(event.target.value)
                        }
                      />
                      小时
                    </span>
                  </label>
                </div>
                <div className="runtime-cleanup-footer">
                  <small>
                    {storageCleanup.lastRun
                      ? `最近运行 ${new Date(
                          storageCleanup.lastRun.completedAt,
                        ).toLocaleString('zh-CN')} · 释放 ${formatStorageSize(
                          storageCleanup.lastRun.removedBytes,
                        )}`
                      : '尚未运行'}
                  </small>
                  <span>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={
                        runtimeBusy || !desktopControls.runStorageCleanup
                      }
                      onClick={() =>
                        void runDesktopAction(async () => {
                          const result =
                            await desktopControls.runStorageCleanup?.()
                          if (result) {
                            setStorageCleanup(result)
                            setRuntimeNotice(
                              `已释放 ${formatStorageSize(
                                result.lastRun?.removedBytes ?? 0,
                              )}`,
                            )
                          }
                        })
                      }
                    >
                      <Trash2 aria-hidden="true" />
                      立即清理
                    </button>
                    <button
                      type="submit"
                      className="primary-button"
                      disabled={
                        runtimeBusy ||
                        !Number.isFinite(Number(cleanupTriggerGiB)) ||
                        !Number.isFinite(Number(cleanupTargetGiB)) ||
                        !Number.isFinite(Number(cleanupRetentionHours)) ||
                        Number(cleanupTriggerGiB) < 0.25 ||
                        Number(cleanupTargetGiB) <
                          Number(cleanupTriggerGiB) ||
                        Number(cleanupRetentionHours) < 0
                      }
                    >
                      <Save aria-hidden="true" />
                      保存
                    </button>
                  </span>
                </div>
              </form>
            ) : null}
            <div className="runtime-maintenance">
              <strong>项目与数据</strong>
              <div>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={
                    runtimeBusy ||
                    !project ||
                    !desktopControls.exportProject
                  }
                  onClick={() =>
                    void runDesktopAction(async () => {
                      const result = await desktopControls.exportProject?.(
                        project!.id,
                      )
                      if (result) setRuntimeNotice('项目已导出')
                    })
                  }
                >
                  <Download aria-hidden="true" />
                  导出项目
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={runtimeBusy || !desktopControls.importProject}
                  onClick={() =>
                    void runDesktopAction(async () => {
                      const imported = await desktopControls.importProject?.()
                      if (!imported) return
                      loadProject(imported)
                      window.history.replaceState(
                        null,
                        '',
                        `?project=${imported.id}`,
                      )
                      setRuntimeNotice('项目已导入')
                    })
                  }
                >
                  <Upload aria-hidden="true" />
                  导入项目
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={runtimeBusy || !desktopControls.createBackup}
                  onClick={() =>
                    void runDesktopAction(async () => {
                      const result = await desktopControls.createBackup?.()
                      if (result) {
                        setRuntimeNotice(`备份已创建：${result.name}`)
                      }
                    })
                  }
                >
                  <HardDrive aria-hidden="true" />
                  创建备份
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={runtimeBusy || !desktopControls.restoreBackup}
                  onClick={() => {
                    if (!window.confirm('恢复备份将替换当前本地数据，是否继续？')) {
                      return
                    }
                    void runDesktopAction(async () => {
                      const result = await desktopControls.restoreBackup?.()
                      if (result) window.location.reload()
                    })
                  }}
                >
                  <RotateCcw aria-hidden="true" />
                  恢复备份
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={runtimeBusy || !desktopControls.clearCache}
                  onClick={() => {
                    if (!window.confirm('清理推理缓存和未处理上传文件？')) return
                    void runDesktopAction(async () => {
                      const result = await desktopControls.clearCache?.()
                      if (result) {
                        setRuntimeNotice(
                          `已释放 ${(result.removedBytes / 1024 / 1024).toFixed(1)} MB`,
                        )
                      }
                    })
                  }}
                >
                  <Trash2 aria-hidden="true" />
                  清理缓存
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={runtimeBusy || !desktopControls.exportDiagnostics}
                  onClick={() =>
                    void runDesktopAction(async () => {
                      const result =
                        await desktopControls.exportDiagnostics?.()
                      if (result) setRuntimeNotice('诊断信息已导出')
                    })
                  }
                >
                  <Download aria-hidden="true" />
                  导出诊断
                </button>
              </div>
            </div>
            {runtimeNotice && (
              <p className="runtime-notice">{runtimeNotice}</p>
            )}
            {runtimeError && <p className="dialog-error">{runtimeError}</p>}
          </section>
        </div>
      ) : null}
    </div>
  )
}

export default App
