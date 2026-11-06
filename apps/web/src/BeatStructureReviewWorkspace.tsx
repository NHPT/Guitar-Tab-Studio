import {
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Flag,
  Headphones,
  LoaderCircle,
  Pause,
  PencilLine,
  Play,
  Plus,
  RotateCcw,
  Save,
  SkipBack,
  Target,
  Trash2,
} from 'lucide-react'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  apiAssetUrl,
  fetchBeatStructureReview,
  fetchBeatStructureReviews,
  reopenBeatStructureReview,
  saveBeatStructureReview,
  submitBeatStructureReview,
} from './api'
import type {
  BeatMeterChange,
  BeatStructureDocument,
  BeatStructureDraft,
  BeatStructureSummary,
} from './types'
import './BeatStructureReviewWorkspace.css'

const structureChecks: Array<{
  key: keyof BeatStructureDraft['checks']
  label: string
}> = [
  { key: 'meter', label: '主拍号' },
  { key: 'downbeat_phase', label: '强拍相位' },
  { key: 'pickup', label: '弱起时值' },
  { key: 'meter_changes', label: '变拍位置' },
]

function draftFromDocument(
  document: BeatStructureDocument,
): BeatStructureDraft {
  return {
    time_signature: [...document.review.time_signature],
    pickup_beats: document.review.pickup_beats,
    phase_status: document.review.phase_status,
    first_downbeat_at: document.review.first_downbeat_at,
    meter_changes: document.review.meter_changes.map((change) => ({
      at: change.at,
      time_signature: [...change.time_signature],
    })),
    excluded_candidate_beats: [
      ...(document.review.excluded_candidate_beats ?? []),
    ],
    added_candidate_beats: [
      ...(document.review.added_candidate_beats ?? []),
    ],
    checks: { ...document.review.checks },
  }
}

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60)
  const seconds = value - minutes * 60
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`
}

function greatestCommonDivisor(left: number, right: number): number {
  let currentLeft = Math.abs(left)
  let currentRight = Math.abs(right)
  while (currentRight > 0) {
    const remainder = currentLeft % currentRight
    currentLeft = currentRight
    currentRight = remainder
  }
  return currentLeft
}

function noteValueLabel(beats: number, denominator: number): string {
  const units = Math.round(beats * 4)
  const rawDenominator = denominator * 4
  const divisor = greatestCommonDivisor(units, rawDenominator)
  const numerator = units / divisor
  const reducedDenominator = rawDenominator / divisor
  const noteNames: Record<number, string> = {
    1: '全音符',
    2: '二分音符',
    4: '四分音符',
    8: '八分音符',
    16: '十六分音符',
    32: '三十二分音符',
    64: '六十四分音符',
  }
  if (numerator === 1 && noteNames[reducedDenominator]) {
    return noteNames[reducedDenominator]
  }
  if (
    numerator === 3 &&
    noteNames[reducedDenominator / 2]
  ) {
    return `附点${noteNames[reducedDenominator / 2]}`
  }
  return `${numerator}/${reducedDenominator} 音符时值`
}

function formatPickupBeats(beats: number, denominator: number): string {
  if (beats === 0) return '无'
  const beatCount = Number.isInteger(beats)
    ? beats.toFixed(0)
    : beats.toFixed(2).replace(/0$/u, '')
  return `${beatCount} 拍（${noteValueLabel(beats, denominator)}）`
}

function waveformPath(values: readonly number[]): string {
  if (values.length === 0) return ''
  const upper = values.map((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * 1000
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${(
      60 -
      value * 49
    ).toFixed(2)}`
  })
  const lower = [...values].reverse().map((value, reverseIndex) => {
    const index = values.length - reverseIndex - 1
    const x = (index / Math.max(1, values.length - 1)) * 1000
    return `L${x.toFixed(2)},${(60 + value * 49).toFixed(2)}`
  })
  return `${upper.join(' ')} ${lower.join(' ')} Z`
}

function nearestBeatIndex(beats: readonly number[], at: number): number {
  if (beats.length === 0) return 0
  let best = 0
  for (let index = 1; index < beats.length; index += 1) {
    if (Math.abs(beats[index] - at) < Math.abs(beats[best] - at)) {
      best = index
    }
  }
  return best
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus
}

function estimatedBpm(beats: readonly number[]): number | null {
  const intervals = beats
    .slice(1)
    .map((beat, index) => beat - beats[index])
    .filter((interval) => interval > 0)
    .sort((left, right) => left - right)
  if (intervals.length === 0) return null
  const middle = Math.floor(intervals.length / 2)
  const median =
    intervals.length % 2 === 0
      ? (intervals[middle - 1] + intervals[middle]) / 2
      : intervals[middle]
  return median > 0 ? 60 / median : null
}

function sameBeatAt(left: number, right: number): boolean {
  return Math.abs(left - right) <= 0.001
}

function sameBeatList(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((beat, index) => sameBeatAt(beat, right[index]))
  )
}

type BeatBatchMode = 'halve' | 'quarter' | 'double' | 'quadruple'
type BeatBatchScope = 'all' | 'from-current'

interface BeatCorrectionSnapshot {
  excludedCandidateBeats: number[]
  addedCandidateBeats: number[]
  firstDownbeatAt: number | null
  meterChanges: BeatMeterChange[]
  selectedAt: number
}

function batchFactor(mode: BeatBatchMode): number {
  return mode === 'halve' || mode === 'double' ? 2 : 4
}

function isReductionMode(mode: BeatBatchMode): boolean {
  return mode === 'halve' || mode === 'quarter'
}

function batchPhaseLabel(mode: BeatBatchMode, phase: number): string {
  if (mode === 'halve') {
    return phase === 0
      ? '删 2、4 · 留 1、3'
      : '删 1、3 · 留 2、4'
  }
  const deleted = [1, 2, 3, 4]
    .filter((position) => position !== phase + 1)
    .join('、')
  return `仅留 ${phase + 1} · 删 ${deleted}`
}

function uniqueSortedBeats(beats: readonly number[]): number[] {
  return beats
    .map((beat) => Math.round(beat * 1_000_000) / 1_000_000)
    .sort((left, right) => left - right)
    .filter(
      (beat, index, values) =>
        index === 0 || !sameBeatAt(beat, values[index - 1]),
    )
}

function transformBeatGrid(
  beats: readonly number[],
  mode: BeatBatchMode,
  phase: number,
): number[] {
  const factor = batchFactor(mode)
  if (isReductionMode(mode)) {
    return beats.filter((_, index) => index % factor === phase)
  }
  if (beats.length < 2) return [...beats]
  const intervals = beats
    .slice(1)
    .map((beat, index) => beat - beats[index])
    .filter((interval) => interval > 0)
    .sort((left, right) => left - right)
  const medianInterval =
    intervals[Math.floor(intervals.length / 2)] ?? Number.POSITIVE_INFINITY
  const maximumFillInterval = medianInterval * 2.5
  const expanded: number[] = []
  for (let index = 0; index < beats.length - 1; index += 1) {
    const left = beats[index]
    const right = beats[index + 1]
    expanded.push(left)
    if (
      right - left <= maximumFillInterval &&
      (right - left) / factor >= 0.03
    ) {
      for (let step = 1; step < factor; step += 1) {
        expanded.push(left + ((right - left) * step) / factor)
      }
    }
  }
  expanded.push(beats[beats.length - 1])
  return uniqueSortedBeats(expanded)
}

function isKeyboardEditingTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName))
  )
}

interface BeatStructureReviewWorkspaceProps {
  reviewer: string
  onReviewsChange: (reviews: BeatStructureSummary[]) => void
}

export default function BeatStructureReviewWorkspace({
  reviewer,
  onReviewsChange,
}: BeatStructureReviewWorkspaceProps) {
  const [reviews, setReviews] = useState<BeatStructureSummary[]>([])
  const [selectedRecordingId, setSelectedRecordingId] = useState<string | null>(
    null,
  )
  const [document, setDocument] = useState<BeatStructureDocument | null>(null)
  const [draft, setDraft] = useState<BeatStructureDraft | null>(null)
  const [waveform, setWaveform] = useState<number[]>([])
  const [selectedBeatIndex, setSelectedBeatIndex] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [windowDuration, setWindowDuration] = useState(4)
  const [changeSignature, setChangeSignature] = useState<[number, number]>([
    4, 4,
  ])
  const [batchMode, setBatchMode] = useState<BeatBatchMode>('halve')
  const [batchScope, setBatchScope] = useState<BeatBatchScope>('all')
  const [batchPhase, setBatchPhase] = useState(0)
  const [batchUndo, setBatchUndo] =
    useState<BeatCorrectionSnapshot | null>(null)
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)
  const previewEndRef = useRef<number | null>(null)
  const selectedRecordingIdRef = useRef<string | null>(null)
  const excludedCandidateBeatsRef = useRef<number[]>([])
  const addedCandidateBeatsRef = useRef<number[]>([])

  useEffect(() => {
    selectedRecordingIdRef.current = selectedRecordingId
  }, [selectedRecordingId])

  const refreshReviews = useCallback(async () => {
    const next = await fetchBeatStructureReviews(reviewer)
    setReviews(next)
    onReviewsChange(next)
    return next
  }, [onReviewsChange, reviewer])

  useEffect(() => {
    let active = true
    fetchBeatStructureReviews(reviewer)
      .then((next) => {
        if (!active) return
        setReviews(next)
        onReviewsChange(next)
        setSelectedRecordingId(
          next.find((review) => review.status === 'draft')?.recordingId ??
            next[0]?.recordingId ??
            null,
        )
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(
            reason instanceof Error ? reason.message : '全曲结构队列加载失败',
          )
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [onReviewsChange, reviewer])

  useEffect(() => {
    if (!selectedRecordingId) return
    let active = true
    fetchBeatStructureReview(selectedRecordingId, reviewer)
      .then((next) => {
        if (!active) return
        const nextDraft = draftFromDocument(next)
        const anchor =
          nextDraft.first_downbeat_at ?? next.candidateBeats[0] ?? 0
        excludedCandidateBeatsRef.current = [
          ...nextDraft.excluded_candidate_beats,
        ]
        addedCandidateBeatsRef.current = [
          ...nextDraft.added_candidate_beats,
        ]
        setDocument(next)
        setDraft(nextDraft)
        setWaveform([])
        setSelectedBeatIndex(nearestBeatIndex(next.candidateBeats, anchor))
        setCurrentTime(anchor)
        setBatchMode('halve')
        setBatchScope('all')
        setBatchPhase(0)
        setBatchUndo(null)
        setCorrectionOpen(false)
        setDirty(false)
        setNotice('')
        setError('')
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(
            reason instanceof Error ? reason.message : '全曲结构加载失败',
          )
        }
      })
    return () => {
      active = false
    }
  }, [reviewer, selectedRecordingId])

  useEffect(() => {
    if (!document) return
    const controller = new AbortController()
    fetch(apiAssetUrl(document.audioUrl), {
      credentials: 'include',
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error('全曲音频加载失败')
        return response.arrayBuffer()
      })
      .then(async (buffer) => {
        const context = new AudioContext()
        try {
          const audio = await context.decodeAudioData(buffer)
          const channel = audio.getChannelData(0)
          const bins = 1000
          const step = Math.max(1, Math.floor(channel.length / bins))
          const values = Array.from({ length: bins }, (_, bin) => {
            const start = bin * step
            const end = Math.min(channel.length, start + step)
            let peak = 0
            for (let index = start; index < end; index += 1) {
              peak = Math.max(peak, Math.abs(channel[index]))
            }
            return peak
          })
          const peak = Math.max(0.001, ...values)
          if (!controller.signal.aborted) {
            setWaveform(values.map((value) => value / peak))
          }
        } finally {
          await context.close()
        }
      })
      .catch((reason: unknown) => {
        if (
          !controller.signal.aborted &&
          !(reason instanceof DOMException && reason.name === 'AbortError')
        ) {
          setError(reason instanceof Error ? reason.message : '波形加载失败')
        }
      })
    return () => controller.abort()
  }, [document])

  useEffect(() => {
    if (!playing) return
    let frame = 0
    const update = () => {
      const audio = audioRef.current
      if (!audio || audio.paused) return
      const at = audio.currentTime
      setCurrentTime(at)
      if (
        previewEndRef.current !== null &&
        at >= previewEndRef.current
      ) {
        previewEndRef.current = null
        audio.pause()
        return
      }
      frame = window.requestAnimationFrame(update)
    }
    frame = window.requestAnimationFrame(update)
    return () => window.cancelAnimationFrame(frame)
  }, [playing])

  const candidateBeats = useMemo(() => {
    if (!document) return []
    const rawCandidateBeats =
      document.rawCandidateBeats ?? document.candidateBeats
    const excludedCandidateBeats =
      draft?.excluded_candidate_beats ?? []
    return [
      ...rawCandidateBeats.filter(
        (beat) =>
          !excludedCandidateBeats.some((excluded) =>
            sameBeatAt(beat, excluded),
          ),
      ),
      ...(draft?.added_candidate_beats ?? []),
    ].sort((left, right) => left - right)
  }, [
    document,
    draft?.added_candidate_beats,
    draft?.excluded_candidate_beats,
  ])
  const selectedBeatAt = candidateBeats[selectedBeatIndex] ?? 0
  const reviewedBpm = useMemo(
    () => estimatedBpm(candidateBeats),
    [candidateBeats],
  )
  const batchPreview = useMemo(() => {
    if (!document || candidateBeats.length < 2) return null
    const startIndex =
      batchScope === 'all'
        ? 0
        : Math.max(0, Math.min(selectedBeatIndex, candidateBeats.length - 1))
    const prefix = candidateBeats.slice(0, startIndex)
    const source = candidateBeats.slice(startIndex)
    const nextScopedBeats = transformBeatGrid(
      source,
      batchMode,
      batchPhase,
    )
    const targetBeats = uniqueSortedBeats([
      ...prefix,
      ...nextScopedBeats,
    ])
    const rawCandidateBeats =
      document.rawCandidateBeats ?? document.candidateBeats
    const excludedCandidateBeats = rawCandidateBeats.filter(
      (beat) => !targetBeats.some((target) => sameBeatAt(beat, target)),
    )
    const addedCandidateBeats = targetBeats.filter(
      (beat) => !rawCandidateBeats.some((raw) => sameBeatAt(beat, raw)),
    )
    return {
      targetBeats,
      excludedCandidateBeats,
      addedCandidateBeats,
      bpm: estimatedBpm(targetBeats),
      changed: !sameBeatList(candidateBeats, targetBeats),
    }
  }, [
    batchMode,
    batchPhase,
    batchScope,
    candidateBeats,
    document,
    selectedBeatIndex,
  ])
  const pickupOptions = useMemo(
    () =>
      Array.from(
        { length: (draft?.time_signature[0] ?? 0) * 4 },
        (_, index) => index / 4,
      ),
    [draft?.time_signature],
  )
  const hasBeatCorrections =
    (draft?.excluded_candidate_beats.length ?? 0) > 0 ||
    (draft?.added_candidate_beats.length ?? 0) > 0
  const waveformShape = useMemo(() => waveformPath(waveform), [waveform])
  const readOnly = document?.review.status !== 'draft'
  const selectedQueueIndex = reviews.findIndex(
    (review) => review.recordingId === selectedRecordingId,
  )
  const windowStart = document
    ? Math.max(
        0,
        Math.min(
          document.review.audio_duration - windowDuration,
          currentTime - windowDuration / 2,
        ),
      )
    : 0
  const windowEnd = document
    ? Math.min(document.review.audio_duration, windowStart + windowDuration)
    : windowDuration
  const visibleBeats = useMemo(() => {
    if (!document) return []
    return candidateBeats
      .map((at, index) => ({ at, index }))
      .filter((beat) => beat.at >= windowStart && beat.at <= windowEnd)
  }, [candidateBeats, document, windowEnd, windowStart])
  const subdivisionTicks = useMemo(() => {
    const anchor =
      draft?.phase_status === 'confirmed'
        ? draft.first_downbeat_at
        : null
    const bpm = reviewedBpm ?? document?.candidateBpm ?? null
    if (
      anchor === null ||
      bpm === null ||
      bpm <= 0 ||
      windowEnd <= windowStart
    ) {
      return []
    }
    const subdivisionDuration = 15 / bpm
    const firstUnit = Math.ceil(
      (windowStart - anchor) / subdivisionDuration - 0.000001,
    )
    const lastUnit = Math.floor(
      (windowEnd - anchor) / subdivisionDuration + 0.000001,
    )
    if (lastUnit < firstUnit || lastUnit - firstUnit > 2_000) {
      return []
    }
    return Array.from(
      { length: lastUnit - firstUnit + 1 },
      (_, index) => {
        const unit = firstUnit + index
        return {
          at: anchor + unit * subdivisionDuration,
          quarter: positiveModulo(unit, 4) === 0,
          unit,
        }
      },
    )
  }, [
    document?.candidateBpm,
    draft,
    reviewedBpm,
    windowEnd,
    windowStart,
  ])

  const beatLabel = useCallback(
    (index: number): number | null => {
      if (!document || !draft || draft.phase_status === 'uncertain') {
        return null
      }
      const sections = [
        {
          at: draft.first_downbeat_at ?? candidateBeats[0],
          timeSignature: draft.time_signature,
        },
        ...draft.meter_changes.map((change) => ({
          at: change.at,
          timeSignature: change.time_signature,
        })),
      ].sort((left, right) => left.at - right.at)
      const at = candidateBeats[index]
      const section =
        [...sections].reverse().find((candidate) => candidate.at <= at) ??
        sections[0]
      const anchorIndex = nearestBeatIndex(
        candidateBeats,
        section.at,
      )
      return (
        positiveModulo(index - anchorIndex, section.timeSignature[0]) + 1
      )
    },
    [candidateBeats, document, draft],
  )

  const representativePoints = useMemo(() => {
    if (!document || !draft || candidateBeats.length === 0) {
      return []
    }
    const candidates =
      draft.phase_status === 'confirmed'
        ? candidateBeats
            .map((at, index) => ({ at, index }))
            .filter(({ index }) => beatLabel(index) === 1)
        : candidateBeats.map((at, index) => ({ at, index }))
    const pool =
      candidates.length > 0
        ? candidates
        : candidateBeats.map((at, index) => ({ at, index }))
    const targets = [
      {
        label: '开头',
        at: draft.first_downbeat_at ?? candidateBeats[0],
      },
      { label: '中段', at: document.review.audio_duration * 0.5 },
      { label: '结尾', at: document.review.audio_duration * 0.88 },
    ]
    return targets.map((target) => {
      const point = pool.reduce((closest, candidate) =>
        Math.abs(candidate.at - target.at) <
        Math.abs(closest.at - target.at)
          ? candidate
          : closest,
      )
      return { ...point, label: target.label }
    })
  }, [beatLabel, candidateBeats, document, draft])

  const updateDraft = useCallback(
    (update: (current: BeatStructureDraft) => BeatStructureDraft) => {
      if (saving) return
      setDraft((current) => (current ? update(current) : current))
      setBatchUndo(null)
      setDirty(true)
      setNotice('')
    },
    [saving],
  )

  const seek = useCallback((at: number) => {
    if (!document) return
    previewEndRef.current = null
    const bounded = Math.max(0, Math.min(document.review.audio_duration, at))
    if (audioRef.current) audioRef.current.currentTime = bounded
    setCurrentTime(bounded)
  }, [document])

  const togglePlayback = useCallback(async () => {
    const audio = audioRef.current
    if (!audio) return
    previewEndRef.current = null
    if (audio.paused) await audio.play()
    else audio.pause()
  }, [])

  const playRepresentativePreview = async (index: number, at: number) => {
    const audio = audioRef.current
    if (!audio || !document) return
    const start = Math.max(0, at - 1.5)
    previewEndRef.current = Math.min(document.review.audio_duration, at + 6.5)
    setSelectedBeatIndex(index)
    setCurrentTime(start)
    audio.currentTime = start
    await audio.play()
  }

  const selectRecording = (recordingId: string) => {
    if (saving) return
    if (dirty && !window.confirm('当前全曲结构尚未保存，确认切换吗？')) {
      return
    }
    audioRef.current?.pause()
    selectedRecordingIdRef.current = recordingId
    setSelectedRecordingId(recordingId)
  }

  const setAnchor = () => {
    updateDraft((current) => ({
      ...current,
      phase_status: 'confirmed',
      first_downbeat_at: selectedBeatAt,
    }))
  }

  const addMeterChange = () => {
    if (!draft || selectedBeatAt <= 0) return
    updateDraft((current) => {
      const meterChanges: BeatMeterChange[] = [
        ...current.meter_changes.filter(
          (change) => Math.abs(change.at - selectedBeatAt) > 0.001,
        ),
        {
          at: selectedBeatAt,
          time_signature: [
            changeSignature[0],
            changeSignature[1],
          ],
        },
      ]
      return {
        ...current,
        meter_changes: meterChanges.sort((left, right) => left.at - right.at),
      }
    })
  }

  const applyBatchCorrection = useCallback(() => {
    if (
      !document ||
      !draft ||
      !batchPreview ||
      !batchPreview.changed ||
      batchPreview.targetBeats.length < 2 ||
      batchPreview.targetBeats.length > 10_000 ||
      readOnly ||
      saving
    ) {
      return
    }
    const targetBeats = batchPreview.targetBeats
    const nearestTarget = (at: number): number =>
      targetBeats[nearestBeatIndex(targetBeats, at)]
    const mappedMeterChanges = draft.meter_changes
      .map((change) => ({
        ...change,
        at: nearestTarget(change.at),
      }))
      .sort((left, right) => left.at - right.at)
    if (
      mappedMeterChanges.some(
        (change, index) =>
          index > 0 &&
          sameBeatAt(change.at, mappedMeterChanges[index - 1].at),
      )
    ) {
      setError('批量修正会合并多个变拍点，请先调整变拍位置')
      return
    }
    const snapshot: BeatCorrectionSnapshot = {
      excludedCandidateBeats: [
        ...excludedCandidateBeatsRef.current,
      ],
      addedCandidateBeats: [...addedCandidateBeatsRef.current],
      firstDownbeatAt: draft.first_downbeat_at,
      meterChanges: draft.meter_changes.map((change) => ({
        ...change,
        time_signature: [...change.time_signature],
      })),
      selectedAt: selectedBeatAt,
    }
    excludedCandidateBeatsRef.current = [
      ...batchPreview.excludedCandidateBeats,
    ]
    addedCandidateBeatsRef.current = [
      ...batchPreview.addedCandidateBeats,
    ]
    updateDraft((current) => ({
      ...current,
      first_downbeat_at:
        current.first_downbeat_at === null
          ? null
          : nearestTarget(current.first_downbeat_at),
      meter_changes: mappedMeterChanges,
      excluded_candidate_beats: [
        ...batchPreview.excludedCandidateBeats,
      ],
      added_candidate_beats: [...batchPreview.addedCandidateBeats],
    }))
    setBatchUndo(snapshot)
    const nextSelectedAt = nearestTarget(selectedBeatAt)
    setSelectedBeatIndex(nearestBeatIndex(targetBeats, nextSelectedAt))
    seek(nextSelectedAt)
    setError('')
    setNotice(
      `批量修正完成：${candidateBeats.length} → ${targetBeats.length} 拍`,
    )
  }, [
    batchPreview,
    candidateBeats.length,
    document,
    draft,
    readOnly,
    saving,
    seek,
    selectedBeatAt,
    updateDraft,
  ])

  const undoBatchCorrection = useCallback(() => {
    if (!document || !batchUndo || readOnly || saving) return
    const rawCandidateBeats =
      document.rawCandidateBeats ?? document.candidateBeats
    const restoredBeats = [
      ...rawCandidateBeats.filter(
        (beat) =>
          !batchUndo.excludedCandidateBeats.some((excluded) =>
            sameBeatAt(beat, excluded),
          ),
      ),
      ...batchUndo.addedCandidateBeats,
    ].sort((left, right) => left - right)
    excludedCandidateBeatsRef.current = [
      ...batchUndo.excludedCandidateBeats,
    ]
    addedCandidateBeatsRef.current = [...batchUndo.addedCandidateBeats]
    updateDraft((current) => ({
      ...current,
      first_downbeat_at: batchUndo.firstDownbeatAt,
      meter_changes: batchUndo.meterChanges.map((change) => ({
        ...change,
        time_signature: [...change.time_signature],
      })),
      excluded_candidate_beats: [
        ...batchUndo.excludedCandidateBeats,
      ],
      added_candidate_beats: [...batchUndo.addedCandidateBeats],
    }))
    setBatchUndo(null)
    const restoredIndex = nearestBeatIndex(
      restoredBeats,
      batchUndo.selectedAt,
    )
    setSelectedBeatIndex(restoredIndex)
    seek(restoredBeats[restoredIndex] ?? 0)
    setNotice('已撤销上一次批量修正')
  }, [batchUndo, document, readOnly, saving, seek, updateDraft])

  const alignBatchPhaseToSelected = () => {
    const scopeStartIndex =
      batchScope === 'all' ? 0 : selectedBeatIndex
    const phase = positiveModulo(
      selectedBeatIndex - scopeStartIndex,
      batchFactor(batchMode),
    )
    setBatchPhase(phase)
    setNotice(
      `已将 ${formatTime(selectedBeatAt)} 设为批量保留拍`,
    )
  }

  const deleteCandidateBeat = useCallback(() => {
    if (!draft || readOnly || saving || candidateBeats.length <= 2) return
    const deletedAt = selectedBeatAt
    const deletedAddedBeat = addedCandidateBeatsRef.current.some((beat) =>
      sameBeatAt(beat, deletedAt),
    )
    const nextCandidateBeats = candidateBeats.filter(
      (beat) => !sameBeatAt(beat, deletedAt),
    )
    const nextSelectedIndex = Math.min(
      selectedBeatIndex,
      nextCandidateBeats.length - 1,
    )
    const anchorDeleted =
      draft.first_downbeat_at !== null &&
      sameBeatAt(draft.first_downbeat_at, deletedAt)
    const nextAnchor =
      nextCandidateBeats.find((beat) => beat > deletedAt + 0.001) ??
      nextCandidateBeats[nextCandidateBeats.length - 1] ??
      null
    const nextExcludedCandidateBeats = deletedAddedBeat
      ? [...excludedCandidateBeatsRef.current]
      : [...excludedCandidateBeatsRef.current, deletedAt].sort(
          (left, right) => left - right,
        )
    const nextAddedCandidateBeats = deletedAddedBeat
      ? addedCandidateBeatsRef.current.filter(
          (beat) => !sameBeatAt(beat, deletedAt),
        )
      : [...addedCandidateBeatsRef.current]
    excludedCandidateBeatsRef.current = nextExcludedCandidateBeats
    addedCandidateBeatsRef.current = nextAddedCandidateBeats
    updateDraft((current) => ({
      ...current,
      first_downbeat_at: anchorDeleted
        ? nextAnchor
        : current.first_downbeat_at,
      meter_changes: current.meter_changes.filter(
        (change) => !sameBeatAt(change.at, deletedAt),
      ),
      excluded_candidate_beats: nextExcludedCandidateBeats,
      added_candidate_beats: nextAddedCandidateBeats,
    }))
    setSelectedBeatIndex(nextSelectedIndex)
    if (nextCandidateBeats[nextSelectedIndex] !== undefined) {
      seek(nextCandidateBeats[nextSelectedIndex])
    }
    setNotice(
      anchorDeleted
        ? `已删除 ${formatTime(deletedAt)}，首个强拍已移至 ${formatTime(nextAnchor ?? 0)}`
        : deletedAddedBeat
          ? `已删除 ${formatTime(deletedAt)} 新增拍点`
          : `已删除 ${formatTime(deletedAt)}，后续拍位已重新计算`,
    )
  }, [
    candidateBeats,
    draft,
    readOnly,
    saving,
    seek,
    selectedBeatAt,
    selectedBeatIndex,
    updateDraft,
  ])

  const restoreCandidateBeat = useCallback(
    (restoredAt: number) => {
      if (!document || !draft || readOnly || saving) return
      const nextExcluded = excludedCandidateBeatsRef.current.filter(
        (beat) => !sameBeatAt(beat, restoredAt),
      )
      excludedCandidateBeatsRef.current = nextExcluded
      const rawCandidateBeats =
        document.rawCandidateBeats ?? document.candidateBeats
      const nextCandidateBeats = [
        ...rawCandidateBeats.filter(
          (beat) =>
            !nextExcluded.some((excluded) => sameBeatAt(beat, excluded)),
        ),
        ...addedCandidateBeatsRef.current,
      ].sort((left, right) => left - right)
      const restoredIndex = nearestBeatIndex(nextCandidateBeats, restoredAt)
      updateDraft((current) => ({
        ...current,
        excluded_candidate_beats: current.excluded_candidate_beats.filter(
          (beat) => !sameBeatAt(beat, restoredAt),
        ),
      }))
      setSelectedBeatIndex(restoredIndex)
      seek(restoredAt)
      setNotice(`已恢复 ${formatTime(restoredAt)} 候选拍`)
    },
    [document, draft, readOnly, saving, seek, updateDraft],
  )

  const addCandidateBeat = useCallback(() => {
    if (!document || !draft || readOnly || saving) return
    const mediaTime = audioRef.current?.currentTime ?? currentTime
    const addedAt =
      Math.round(
        Math.max(
          0,
          Math.min(document.review.audio_duration, mediaTime),
        ) * 1_000_000,
      ) / 1_000_000
    const excludedRawBeat = excludedCandidateBeatsRef.current.find((beat) =>
      sameBeatAt(beat, addedAt),
    )
    if (excludedRawBeat !== undefined) {
      restoreCandidateBeat(excludedRawBeat)
      return
    }
    const existingBeat = candidateBeats.find((beat) =>
      sameBeatAt(beat, addedAt),
    )
    if (existingBeat !== undefined) {
      setSelectedBeatIndex(nearestBeatIndex(candidateBeats, existingBeat))
      seek(existingBeat)
      setNotice(`播放位置已有 ${formatTime(existingBeat)} 拍点`)
      return
    }
    if (candidateBeats.length >= 10_000) {
      setError('候选拍点数量已达到上限')
      return
    }
    const nextAddedCandidateBeats = [
      ...addedCandidateBeatsRef.current,
      addedAt,
    ].sort((left, right) => left - right)
    const nextCandidateBeats = [...candidateBeats, addedAt].sort(
      (left, right) => left - right,
    )
    addedCandidateBeatsRef.current = nextAddedCandidateBeats
    updateDraft((current) => ({
      ...current,
      added_candidate_beats: nextAddedCandidateBeats,
    }))
    setSelectedBeatIndex(
      nextCandidateBeats.findIndex((beat) => sameBeatAt(beat, addedAt)),
    )
    seek(addedAt)
    setError('')
    setNotice(`已在 ${formatTime(addedAt)} 添加拍点`)
  }, [
    candidateBeats,
    currentTime,
    document,
    draft,
    readOnly,
    restoreCandidateBeat,
    saving,
    seek,
    updateDraft,
  ])

  const persist = useCallback(
    async (
      submit: boolean,
      nextDraft = draft,
      confirmation = '提交后全曲结构将锁定，确认提交吗？',
    ) => {
      if (!nextDraft || !selectedRecordingId || saving) return
      if (
        submit &&
        !window.confirm(confirmation)
      ) {
        return
      }
      setSaving(true)
      setError('')
      setNotice('')
      try {
        const persistedDraft = {
          ...nextDraft,
          excluded_candidate_beats: [
            ...excludedCandidateBeatsRef.current,
          ],
          added_candidate_beats: [
            ...addedCandidateBeatsRef.current,
          ],
        }
        const next = submit
          ? await submitBeatStructureReview(
              selectedRecordingId,
              reviewer,
              persistedDraft,
            )
          : await saveBeatStructureReview(
              selectedRecordingId,
              reviewer,
              persistedDraft,
            )
        const returnedExcludedCandidateBeats =
          next.review.excluded_candidate_beats ?? []
        const returnedAddedCandidateBeats =
          next.review.added_candidate_beats ?? []
        if (
          !sameBeatList(
            persistedDraft.excluded_candidate_beats,
            returnedExcludedCandidateBeats,
          ) ||
          !sameBeatList(
            persistedDraft.added_candidate_beats,
            returnedAddedCandidateBeats,
          )
        ) {
          throw new Error('拍点修正未被服务器保存，当前修改已保留，请刷新后重试')
        }
        excludedCandidateBeatsRef.current = [
          ...returnedExcludedCandidateBeats,
        ]
        addedCandidateBeatsRef.current = [
          ...returnedAddedCandidateBeats,
        ]
        setDocument(next)
        setDraft(draftFromDocument(next))
        setDirty(false)
        setNotice(submit ? '全曲结构已提交' : '结构草稿已保存')
        await refreshReviews()
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : '结构审核保存失败')
      } finally {
        setSaving(false)
      }
    },
    [draft, refreshReviews, reviewer, saving, selectedRecordingId],
  )

  const reopen = async () => {
    if (!selectedRecordingId || !readOnly || saving) return
    const recordingId = selectedRecordingId
    const reason = window.prompt(
      '请输入撤回原因',
      '发现全曲结构标注错误',
    )
    if (!reason?.trim()) return
    if (
      !window.confirm(
        '撤回后，该歌曲的关联片段将退回草稿，已有金标准将失效归档。确认继续吗？',
      )
    ) {
      return
    }
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const result = await reopenBeatStructureReview(
        recordingId,
        reviewer,
        reason.trim(),
      )
      if (selectedRecordingIdRef.current !== recordingId) {
        await refreshReviews()
        return
      }
      excludedCandidateBeatsRef.current = [
        ...(result.document.review.excluded_candidate_beats ?? []),
      ]
      addedCandidateBeatsRef.current = [
        ...(result.document.review.added_candidate_beats ?? []),
      ]
      setDocument(result.document)
      setDraft(draftFromDocument(result.document))
      setCorrectionOpen(true)
      setDirty(false)
      setNotice(
        `已撤回全曲结构，${result.clipsInvalidated} 个关联片段已退回草稿` +
          (result.goldInvalidated > 0
            ? `，${result.goldInvalidated} 个金标准已失效归档`
            : ''),
      )
      await refreshReviews()
    } catch (reasonValue) {
      setError(
        reasonValue instanceof Error
          ? reasonValue.message
          : '全曲结构撤回失败',
      )
    } finally {
      setSaving(false)
    }
  }

  const checkedDraft = (): BeatStructureDraft | null =>
    draft
      ? {
          ...draft,
          checks: {
            meter: true,
            downbeat_phase: true,
            pickup: true,
            meter_changes: true,
          },
        }
      : null

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if (!document || !draft) return
      const saveShortcut =
        (event.metaKey || event.ctrlKey) &&
        event.key.toLowerCase() === 's'
      if (saveShortcut) {
        if (readOnly) return
        event.preventDefault()
        if (!event.repeat && dirty && !saving) {
          void persist(false)
        }
        return
      }
      if (
        isKeyboardEditingTarget(event.target) ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey
      ) {
        return
      }

      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault()
        const direction = event.key === 'ArrowLeft' ? -1 : 1
        const nextIndex = Math.max(
          0,
          Math.min(
            selectedBeatIndex + direction,
            candidateBeats.length - 1,
          ),
        )
        setSelectedBeatIndex(nextIndex)
        if (candidateBeats[nextIndex] !== undefined) {
          seek(candidateBeats[nextIndex])
        }
        return
      }

      if (event.key === 'Home' || event.key === 'End') {
        event.preventDefault()
        const nextIndex =
          event.key === 'Home' ? 0 : candidateBeats.length - 1
        setSelectedBeatIndex(nextIndex)
        if (candidateBeats[nextIndex] !== undefined) {
          seek(candidateBeats[nextIndex])
        }
        return
      }

      if (event.key === ' ') {
        event.preventDefault()
        if (!event.repeat) void togglePlayback()
        return
      }

      if (event.key === 'Escape') {
        const audio = audioRef.current
        if (audio && !audio.paused) {
          event.preventDefault()
          audio.pause()
        }
        return
      }

      if (
        (event.key === 'Backspace' || event.key === 'Delete') &&
        correctionOpen &&
        !readOnly &&
        !saving
      ) {
        event.preventDefault()
        if (!event.repeat) deleteCandidateBeat()
        return
      }

      if (
        event.key.toLowerCase() === 'a' &&
        correctionOpen &&
        !readOnly &&
        !saving
      ) {
        event.preventDefault()
        if (!event.repeat) addCandidateBeat()
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [
    addCandidateBeat,
    candidateBeats,
    correctionOpen,
    deleteCandidateBeat,
    dirty,
    document,
    draft,
    persist,
    readOnly,
    saving,
    seek,
    selectedBeatIndex,
    togglePlayback,
  ])

  if (loading) {
    return (
      <div className="beat-review-loading">
        <LoaderCircle aria-hidden="true" />
        <span>正在加载全曲结构队列</span>
      </div>
    )
  }

  if (error && !document) {
    return (
      <div className="beat-review-loading is-error">
        <CircleAlert aria-hidden="true" />
        <strong>{error}</strong>
      </div>
    )
  }

  return (
    <div
      className="beat-structure-shell"
      data-review-dirty={dirty}
      aria-keyshortcuts="ArrowLeft ArrowRight Home End Space A Delete Backspace Control+S Meta+S"
    >
      <aside className="beat-structure-queue">
        <header>
          <strong>全曲结构</strong>
          <span>
            {reviews.filter((review) => review.status === 'submitted').length}/
            {reviews.length}
          </span>
        </header>
        <div>
          {reviews.map((review) => (
            <button
              className={
                review.recordingId === selectedRecordingId
                  ? 'is-current'
                  : ''
              }
              type="button"
              key={review.recordingId}
              onClick={() => selectRecording(review.recordingId)}
            >
              <span>
                <strong>{review.recordingId}</strong>
                <small>
                  {formatTime(review.duration)} ·{' '}
                  {review.timeSignature.join('/')}
                </small>
              </span>
              <i className={`is-${review.status}`}>
                {review.status === 'submitted'
                  ? '已提交'
                  : `${
                      structureChecks.filter(
                        (check) => review.checks[check.key],
                      ).length
                    }/4`}
              </i>
            </button>
          ))}
        </div>
      </aside>

      <main className="beat-structure-workspace">
        {document && draft ? (
          <>
            <header className="beat-structure-title">
              <button
                type="button"
                aria-label="上一首"
                disabled={selectedQueueIndex <= 0}
                onClick={() =>
                  selectRecording(reviews[selectedQueueIndex - 1].recordingId)}
              >
                <ChevronLeft aria-hidden="true" />
              </button>
              <div>
                <span>
                  {
                    reviews.find(
                      (review) =>
                        review.recordingId === selectedRecordingId,
                    )?.split
                  }
                </span>
                <h1>{document.review.recording_id}</h1>
                <small>
                  原始 {document.candidateBpm.toFixed(2)} BPM
                  {hasBeatCorrections && reviewedBpm !== null
                    ? ` · 复核 ${reviewedBpm.toFixed(2)} BPM`
                    : ''}{' '}
                  ·{' '}
                  {candidateBeats.length} 拍
                  {hasBeatCorrections
                    ? `（删除 ${draft.excluded_candidate_beats.length}，新增 ${draft.added_candidate_beats.length}）`
                    : ''}
                </small>
              </div>
              <button
                type="button"
                aria-label="下一首"
                disabled={selectedQueueIndex >= reviews.length - 1}
                onClick={() =>
                  selectRecording(reviews[selectedQueueIndex + 1].recordingId)}
              >
                <ChevronRight aria-hidden="true" />
              </button>
            </header>

            <audio
              ref={audioRef}
              src={apiAssetUrl(document.audioUrl)}
              crossOrigin="use-credentials"
              preload="auto"
              onEnded={(event) => {
                setCurrentTime(event.currentTarget.currentTime)
                setPlaying(false)
              }}
              onLoadedMetadata={(event) => {
                event.currentTarget.playbackRate = playbackRate
                event.currentTarget.currentTime = currentTime
              }}
              onPause={(event) => {
                setCurrentTime(event.currentTarget.currentTime)
                setPlaying(false)
              }}
              onPlay={() => setPlaying(true)}
              onTimeUpdate={(event) => {
                const at = event.currentTarget.currentTime
                setCurrentTime(at)
                if (
                  previewEndRef.current !== null &&
                  at >= previewEndRef.current
                ) {
                  previewEndRef.current = null
                  event.currentTarget.pause()
                }
              }}
            />

            <div className="beat-structure-transport">
              <button
                type="button"
                aria-label="回到开头"
                title="回到开头"
                onClick={() => seek(0)}
              >
                <SkipBack aria-hidden="true" />
              </button>
              <button
                type="button"
                aria-label="跳到首个强拍"
                title="跳到首个强拍"
                disabled={draft.first_downbeat_at === null}
                onClick={() => seek(draft.first_downbeat_at ?? 0)}
              >
                <Target aria-hidden="true" />
              </button>
              <button
                className="is-primary"
                type="button"
                aria-label={playing ? '暂停' : '播放'}
                aria-keyshortcuts="Space"
                onClick={() => void togglePlayback()}
              >
                {playing ? (
                  <Pause aria-hidden="true" />
                ) : (
                  <Play aria-hidden="true" />
                )}
              </button>
              <strong>{formatTime(currentTime)}</strong>
              <span>/ {formatTime(document.review.audio_duration)}</span>
              <select
                aria-label="播放速度"
                value={playbackRate}
                onChange={(event) => {
                  const next = Number(event.target.value)
                  setPlaybackRate(next)
                  if (audioRef.current) audioRef.current.playbackRate = next
                }}
              >
                {[0.5, 0.75, 1, 1.25].map((rate) => (
                  <option value={rate} key={rate}>
                    {rate}×
                  </option>
                ))}
              </select>
            </div>

            <div
              className="beat-structure-overview"
              onClick={(event) => {
                const bounds = event.currentTarget.getBoundingClientRect()
                seek(
                  ((event.clientX - bounds.left) / bounds.width) *
                    document.review.audio_duration,
                )
              }}
            >
              <svg
                aria-label="全曲波形"
                preserveAspectRatio="none"
                viewBox="0 0 1000 120"
              >
                <path d={waveformShape} />
              </svg>
              {draft.first_downbeat_at !== null && (
                <i
                  className="is-anchor"
                  title={`首个强拍 ${formatTime(draft.first_downbeat_at)}`}
                  style={{
                    left: `${
                      (draft.first_downbeat_at /
                        document.review.audio_duration) *
                      100
                    }%`,
                  }}
                />
              )}
              {draft.meter_changes.map((change) => (
                <i
                  className="is-change"
                  title={`${formatTime(change.at)} · ${change.time_signature.join('/')}`}
                  key={change.at}
                  style={{
                    left: `${
                      (change.at / document.review.audio_duration) * 100
                    }%`,
                  }}
                />
              ))}
              <b
                style={{
                  left: `${
                    (currentTime / document.review.audio_duration) * 100
                  }%`,
                }}
              />
              <span
                style={{
                  left: `${
                    (windowStart / document.review.audio_duration) * 100
                  }%`,
                  width: `${
                    ((windowEnd - windowStart) /
                      document.review.audio_duration) *
                    100
                  }%`,
                }}
              />
            </div>

            <div className="beat-structure-detail-header">
              <span>
                {formatTime(windowStart)}–{formatTime(windowEnd)}
              </span>
              <select
                aria-label="时间窗"
                value={windowDuration}
                onChange={(event) =>
                  setWindowDuration(Number(event.target.value))}
              >
                {[4, 8, 12, 20, 40].map((duration) => (
                  <option value={duration} key={duration}>
                    {duration} 秒
                  </option>
                ))}
              </select>
            </div>
            <div
              className="beat-structure-detail"
              onClick={(event) => {
                const bounds = event.currentTarget.getBoundingClientRect()
                seek(
                  windowStart +
                    ((event.clientX - bounds.left) / bounds.width) *
                      (windowEnd - windowStart),
                )
              }}
            >
              <div
                className="beat-structure-subdivision-grid"
                aria-hidden="true"
              >
                {subdivisionTicks.map((tick) => (
                  <i
                    className={tick.quarter ? 'is-quarter' : ''}
                    key={tick.unit}
                    style={{
                      left: `${
                        ((tick.at - windowStart) /
                          (windowEnd - windowStart)) *
                        100
                      }%`,
                    }}
                  />
                ))}
              </div>
              {visibleBeats.map((beat) => {
                const label = beatLabel(beat.index)
                return (
                  <button
                    className={[
                      label === 1 ? 'is-downbeat' : '',
                      beat.index === selectedBeatIndex ? 'is-selected' : '',
                      draft.added_candidate_beats.some((added) =>
                        sameBeatAt(added, beat.at),
                      )
                        ? 'is-added'
                        : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                    type="button"
                    key={beat.index}
                    style={{
                      left: `${
                        ((beat.at - windowStart) /
                          (windowEnd - windowStart)) *
                        100
                      }%`,
                    }}
                    aria-label={`${formatTime(beat.at)}，${
                      draft.added_candidate_beats.some((added) =>
                        sameBeatAt(added, beat.at),
                      )
                        ? '新增拍点，'
                        : ''
                    }${label === null ? '拍位待确认' : `第 ${label} 拍`}`}
                    onClick={(event) => {
                      event.stopPropagation()
                      setSelectedBeatIndex(beat.index)
                      seek(beat.at)
                    }}
                  >
                    <i />
                    <span>{label ?? '?'}</span>
                  </button>
                )
              })}
              <b
                style={{
                  left: `${
                    ((currentTime - windowStart) /
                      (windowEnd - windowStart)) *
                    100
                  }%`,
                }}
              />
            </div>
            <div className="beat-structure-selection">
              <span>当前候选拍</span>
              <strong>{formatTime(selectedBeatAt)}</strong>
              <i>第 {beatLabel(selectedBeatIndex) ?? '?'} 拍</i>
            </div>
          </>
        ) : null}
      </main>

      <aside className="beat-structure-inspector">
        {document && draft ? (
          <>
            <section className="beat-structure-candidate">
              <header>
                <span>自动分析结果</span>
                <i className={readOnly ? 'is-submitted' : ''}>
                  {readOnly ? '已提交' : '待确认'}
                </i>
              </header>
              <dl>
                <div>
                  <dt>主拍号</dt>
                  <dd>{draft.time_signature.join('/')}</dd>
                </div>
                <div>
                  <dt>
                    {hasBeatCorrections
                      ? '原始速度'
                      : '速度'}
                  </dt>
                  <dd>{document.candidateBpm.toFixed(1)} BPM</dd>
                </div>
                {hasBeatCorrections &&
                  reviewedBpm !== null && (
                    <div>
                      <dt>复核后速度</dt>
                      <dd>{reviewedBpm.toFixed(1)} BPM</dd>
                    </div>
                  )}
                <div>
                  <dt>弱起</dt>
                  <dd>
                    {formatPickupBeats(
                      draft.pickup_beats,
                      draft.time_signature[1],
                    )}
                  </dd>
                </div>
                <div>
                  <dt>首个强拍</dt>
                  <dd>
                    {draft.first_downbeat_at === null
                      ? '无法确认'
                      : formatTime(draft.first_downbeat_at)}
                  </dd>
                </div>
                <div>
                  <dt>变拍</dt>
                  <dd>
                    {draft.meter_changes.length === 0
                      ? '无'
                      : `${draft.meter_changes.length} 处`}
                  </dd>
                </div>
              </dl>
              <div className="beat-structure-preview-points">
                {representativePoints.map((point) => (
                  <button
                    type="button"
                    key={point.label}
                    aria-label={`试听${point.label} ${formatTime(point.at)}`}
                    onClick={() =>
                      void playRepresentativePreview(point.index, point.at)}
                  >
                    <Headphones aria-hidden="true" />
                    <span>{point.label}</span>
                    <small>{formatTime(point.at)}</small>
                  </button>
                ))}
              </div>
              <small>自动候选未附量化置信度</small>
              {!readOnly && (
                <div className="beat-structure-review-actions">
                  <button
                    className="is-primary"
                    type="button"
                    disabled={
                      saving ||
                      dirty ||
                      draft.phase_status !== 'confirmed' ||
                      draft.first_downbeat_at === null
                    }
                    onClick={() =>
                      void persist(
                        true,
                        checkedDraft(),
                        '已试听代表位置，确认当前分析正确并提交吗？',
                      )}
                  >
                    <Check aria-hidden="true" />
                    分析正确，确认并提交
                  </button>
                  <button
                    type="button"
                    aria-expanded={correctionOpen}
                    onClick={() => setCorrectionOpen((current) => !current)}
                  >
                    <PencilLine aria-hidden="true" />
                    {correctionOpen ? '收起纠正' : '需要纠正'}
                  </button>
                </div>
              )}
              {readOnly && (
                <div className="beat-structure-review-actions">
                  <button
                    className="is-reopen"
                    type="button"
                    disabled={saving}
                    onClick={() => void reopen()}
                  >
                    <RotateCcw aria-hidden="true" />
                    撤回并重新复核
                  </button>
                </div>
              )}
            </section>

            {correctionOpen && !readOnly && (
              <>
                <section className="beat-structure-beat-correction">
                  <span>纠正候选拍点</span>
                  <div className="beat-structure-add-beat">
                    <div>
                      <small>当前播放位置</small>
                      <strong>{formatTime(currentTime)}</strong>
                    </div>
                    <button
                      type="button"
                      aria-label="在播放位置添加拍点"
                      aria-keyshortcuts="A"
                      onClick={addCandidateBeat}
                    >
                      <Plus aria-hidden="true" />
                      添加拍点
                    </button>
                  </div>
                  <div className="beat-structure-delete-beat">
                    <div>
                      <small>当前候选拍</small>
                      <strong>{formatTime(selectedBeatAt)}</strong>
                    </div>
                    <button
                      type="button"
                      disabled={candidateBeats.length <= 2}
                      aria-label={`删除当前候选拍 ${formatTime(selectedBeatAt)}`}
                      aria-keyshortcuts="Backspace Delete"
                      onClick={deleteCandidateBeat}
                    >
                      <Trash2 aria-hidden="true" />
                      删除误检拍
                    </button>
                  </div>
                  <div className="beat-structure-batch-correction">
                    <div className="beat-structure-batch-controls">
                      <label>
                        <span>倍率</span>
                        <select
                          aria-label="批量节拍倍率"
                          value={batchMode}
                          onChange={(event) => {
                            setBatchMode(
                              event.target.value as BeatBatchMode,
                            )
                            setBatchPhase(0)
                          }}
                        >
                          <option value="halve">拍点 ÷2</option>
                          <option value="quarter">拍点 ÷4</option>
                          <option value="double">拍点 ×2</option>
                          <option value="quadruple">拍点 ×4</option>
                        </select>
                      </label>
                      <label>
                        <span>范围</span>
                        <select
                          aria-label="批量修正范围"
                          value={batchScope}
                          onChange={(event) =>
                            setBatchScope(
                              event.target.value as BeatBatchScope,
                            )}
                        >
                          <option value="all">全曲</option>
                          <option value="from-current">
                            当前拍至结尾
                          </option>
                        </select>
                      </label>
                    </div>
                    {isReductionMode(batchMode) && (
                      <div className="beat-structure-batch-phase">
                        <span>删除模式</span>
                        <div
                          role="group"
                          aria-label="批量删除模式"
                        >
                          {Array.from(
                            { length: batchFactor(batchMode) },
                            (_, index) => (
                              <button
                                type="button"
                                key={index}
                                aria-pressed={batchPhase === index}
                                onClick={() => setBatchPhase(index)}
                              >
                                {batchPhaseLabel(batchMode, index)}
                              </button>
                            ),
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={alignBatchPhaseToSelected}
                        >
                          <Target aria-hidden="true" />
                          指定当前拍为保留拍
                        </button>
                      </div>
                    )}
                    <div
                      className="beat-structure-batch-preview"
                      aria-live="polite"
                    >
                      <strong>
                        {candidateBeats.length}
                        <span>→</span>
                        {batchPreview?.targetBeats.length ?? 0} 拍
                      </strong>
                      <span>
                        {reviewedBpm?.toFixed(1) ?? '—'}
                        <i>→</i>
                        {batchPreview?.bpm?.toFixed(1) ?? '—'} BPM
                      </span>
                      <small>
                        删除{' '}
                        {batchPreview?.excludedCandidateBeats.length ?? 0}
                        {' · '}新增{' '}
                        {batchPreview?.addedCandidateBeats.length ?? 0}
                      </small>
                    </div>
                    <div className="beat-structure-batch-actions">
                      <button
                        type="button"
                        disabled={
                          !batchPreview?.changed ||
                          (batchPreview?.targetBeats.length ?? 0) < 2 ||
                          (batchPreview?.targetBeats.length ?? 0) > 10_000
                        }
                        onClick={applyBatchCorrection}
                      >
                        应用批量修正
                      </button>
                      <button
                        type="button"
                        disabled={!batchUndo}
                        onClick={undoBatchCorrection}
                      >
                        <RotateCcw aria-hidden="true" />
                        撤销
                      </button>
                    </div>
                  </div>
                  <small>新增或删除后，后续拍位会立即重新计算。</small>
                  {draft.excluded_candidate_beats.length > 0 && (
                    <div className="beat-structure-excluded-beats">
                      {draft.excluded_candidate_beats
                        .slice(0, 12)
                        .map((beat) => (
                        <button
                          type="button"
                          key={beat}
                          aria-label={`恢复 ${formatTime(beat)} 候选拍`}
                          onClick={() => restoreCandidateBeat(beat)}
                        >
                          <RotateCcw aria-hidden="true" />
                          <span>{formatTime(beat)}</span>
                        </button>
                        ))}
                      {draft.excluded_candidate_beats.length > 12 && (
                        <small>
                          另有{' '}
                          {draft.excluded_candidate_beats.length - 12}{' '}
                          个删除拍点
                        </small>
                      )}
                    </div>
                  )}
                  {draft.added_candidate_beats.length > 0 && (
                    <div className="beat-structure-added-beats">
                      {draft.added_candidate_beats
                        .slice(0, 12)
                        .map((beat) => (
                        <button
                          type="button"
                          key={beat}
                          aria-label={`定位到新增拍点 ${formatTime(beat)}`}
                          onClick={() => {
                            setSelectedBeatIndex(
                              nearestBeatIndex(candidateBeats, beat),
                            )
                            seek(beat)
                          }}
                        >
                          <Plus aria-hidden="true" />
                          <span>{formatTime(beat)}</span>
                        </button>
                        ))}
                      {draft.added_candidate_beats.length > 12 && (
                        <small>
                          另有 {draft.added_candidate_beats.length - 12}{' '}
                          个新增拍点
                        </small>
                      )}
                    </div>
                  )}
                </section>

                <section>
                  <span>纠正拍号与弱起</span>
                  <div className="beat-structure-meter">
                    <input
                      aria-label="每小节拍数"
                      type="number"
                      min={1}
                      max={16}
                      value={draft.time_signature[0]}
                      onChange={(event) => {
                        const numerator = Math.max(
                          1,
                          Math.min(16, Number(event.target.value)),
                        )
                        updateDraft((current) => ({
                          ...current,
                          time_signature: [
                            numerator,
                            current.time_signature[1],
                          ],
                          pickup_beats: Math.min(
                            current.pickup_beats,
                            numerator - 0.25,
                          ),
                        }))
                      }}
                    />
                    <span>/</span>
                    <select
                      aria-label="拍值"
                      value={draft.time_signature[1]}
                      onChange={(event) =>
                        updateDraft((current) => ({
                          ...current,
                          time_signature: [
                            current.time_signature[0],
                            Number(event.target.value),
                          ],
                        }))}
                    >
                      {[2, 4, 8, 16].map((value) => (
                        <option value={value} key={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </div>
                  <label>
                    <span>弱起时值</span>
                    <select
                      aria-label="弱起时值"
                      value={draft.pickup_beats}
                      onChange={(event) =>
                        updateDraft((current) => ({
                          ...current,
                          pickup_beats: Number(event.target.value),
                        }))}
                    >
                      {pickupOptions.map((value) => (
                        <option value={value} key={value}>
                          {value === 0
                            ? '无弱起'
                            : formatPickupBeats(
                                value,
                                draft.time_signature[1],
                              )}
                        </option>
                      ))}
                    </select>
                  </label>
                </section>

                <section>
                  <span>纠正强拍相位</span>
                  <div className="beat-structure-phase-status">
                    <button
                      className={
                        draft.phase_status === 'confirmed' ? 'is-active' : ''
                      }
                      type="button"
                      onClick={() =>
                        updateDraft((current) => ({
                          ...current,
                          phase_status: 'confirmed',
                          first_downbeat_at:
                            current.first_downbeat_at ?? selectedBeatAt,
                        }))}
                    >
                      可确认
                    </button>
                    <button
                      className={
                        draft.phase_status === 'uncertain' ? 'is-active' : ''
                      }
                      type="button"
                      onClick={() =>
                        updateDraft((current) => ({
                          ...current,
                          phase_status: 'uncertain',
                          first_downbeat_at: null,
                        }))}
                    >
                      无法确认
                    </button>
                  </div>
                  <button
                    className="beat-structure-command"
                    type="button"
                    disabled={draft.phase_status === 'uncertain'}
                    onClick={setAnchor}
                  >
                    <Target aria-hidden="true" />
                    当前拍设为首个强拍
                  </button>
                  <small>
                    {draft.first_downbeat_at === null
                      ? '未设置'
                      : formatTime(draft.first_downbeat_at)}
                  </small>
                </section>

                <details
                  className="beat-structure-advanced"
                  open={draft.meter_changes.length > 0}
                >
                  <summary>
                    <span>变拍设置</span>
                    <small>
                      {draft.meter_changes.length === 0
                        ? '无'
                        : `${draft.meter_changes.length} 处`}
                    </small>
                  </summary>
                  <div className="beat-structure-advanced-body">
                    <div className="beat-structure-change-input">
                      <input
                        aria-label="变拍分子"
                        type="number"
                        min={1}
                        max={16}
                        value={changeSignature[0]}
                        onChange={(event) =>
                          setChangeSignature([
                            Math.max(
                              1,
                              Math.min(16, Number(event.target.value)),
                            ),
                            changeSignature[1],
                          ])}
                      />
                      <span>/</span>
                      <select
                        aria-label="变拍拍值"
                        value={changeSignature[1]}
                        onChange={(event) =>
                          setChangeSignature([
                            changeSignature[0],
                            Number(event.target.value),
                          ])}
                      >
                        {[2, 4, 8, 16].map((value) => (
                          <option value={value} key={value}>
                            {value}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        aria-label="添加变拍点"
                        title="添加变拍点"
                        disabled={selectedBeatAt <= 0}
                        onClick={addMeterChange}
                      >
                        <Flag aria-hidden="true" />
                      </button>
                    </div>
                    <div className="beat-structure-changes">
                      {draft.meter_changes.length === 0 ? (
                        <small>无变拍</small>
                      ) : (
                        draft.meter_changes.map((change) => (
                          <div key={change.at}>
                            <button
                              type="button"
                              onClick={() => seek(change.at)}
                            >
                              {formatTime(change.at)}
                            </button>
                            <span>{change.time_signature.join('/')}</span>
                            <button
                              type="button"
                              aria-label={`删除 ${formatTime(change.at)} 变拍点`}
                              onClick={() =>
                                updateDraft((current) => ({
                                  ...current,
                                  meter_changes:
                                    current.meter_changes.filter(
                                      (candidate) =>
                                        candidate.at !== change.at,
                                    ),
                                }))}
                            >
                              <Trash2 aria-hidden="true" />
                            </button>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </details>
              </>
            )}

            {error && (
              <div className="beat-structure-message is-error">
                <CircleAlert aria-hidden="true" />
                <span>{error}</span>
              </div>
            )}
            {notice && (
              <div className="beat-structure-message">
                <Check aria-hidden="true" />
                <span>{notice}</span>
              </div>
            )}

            {correctionOpen && !readOnly && (
              <footer>
                <button
                  type="button"
                  disabled={saving || !dirty}
                  aria-keyshortcuts="Control+S Meta+S"
                  onClick={() => void persist(false)}
                >
                  <Save aria-hidden="true" />
                  保存草稿
                </button>
                <button
                  className="is-primary"
                  type="button"
                  disabled={saving}
                  onClick={() =>
                    void persist(
                      true,
                      checkedDraft(),
                      draft.phase_status === 'uncertain'
                        ? '将以“强拍相位无法确认”提交，确认继续吗？'
                        : '确认提交纠正后的全曲结构吗？',
                    )}
                >
                  <Check aria-hidden="true" />
                  提交纠正结果
                </button>
              </footer>
            )}
          </>
        ) : null}
      </aside>
    </div>
  )
}
