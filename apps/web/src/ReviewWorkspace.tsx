import {
  ArrowLeft,
  AudioLines,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Clock3,
  Guitar,
  LocateFixed,
  Pause,
  Play,
  Plus,
  Save,
  SkipBack,
  Trash2,
  Undo2,
  Unlock,
  Volume2,
  VolumeX,
  X,
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
  approveReview,
  fetchReview,
  fetchReviews,
  reopenReview,
  saveReview,
} from './api'
import './ReviewWorkspace.css'
import type {
  ReviewChecks,
  ReviewDocument,
  ReviewDraft,
  ReviewEvent,
  ReviewExcludedRange,
  ReviewSummary,
  Technique,
} from './types'

const techniqueOptions: Array<{ value: Technique; label: string }> = [
  { value: 'unknown', label: '待确认' },
  { value: 'pick', label: '拨弦' },
  { value: 'strum-down', label: '下扫' },
  { value: 'strum-up', label: '上扫' },
  { value: 'hammer-on', label: '击弦' },
  { value: 'pull-off', label: '勾弦' },
  { value: 'slide', label: '滑音' },
  { value: 'harmonic', label: '泛音' },
  { value: 'palm-mute', label: '掌根制音' },
  { value: 'dead-note', label: '闷音' },
  { value: 'slap', label: '拍弦' },
  { value: 'body-tap', label: '打板' },
  { value: 'tremolo', label: '轮指' },
  { value: 'bend', label: '推弦' },
  { value: 'pinch-harmonic', label: '人工泛音' },
  { value: 'vibrato', label: '揉弦' },
]

const techniqueLabel = Object.fromEntries(
  techniqueOptions.map((option) => [option.value, option.label]),
) as Record<Technique, string>

const checkOptions: Array<{
  key: keyof ReviewChecks
  label: string
  detail: string
  required: boolean
}> = [
  {
    key: 'timing',
    label: '时间边界',
    detail: '起音与终止位置',
    required: true,
  },
  {
    key: 'string_fret',
    label: '弦与品位',
    detail: '1 弦为最细弦',
    required: true,
  },
  {
    key: 'completeness',
    label: '事件完整性',
    detail: '补漏并删除误报',
    required: true,
  },
  {
    key: 'technique',
    label: '演奏技法',
    detail: '本项可选',
    required: false,
  },
]

const playbackRates = [0.25, 0.5, 0.75, 1, 1.25]

function draftFromDocument(document: ReviewDocument): ReviewDraft {
  return {
    events: document.review.events.map((event) => ({
      ...event,
      ...(event.candidate_provenance
        ? { candidate_provenance: { ...event.candidate_provenance } }
        : {}),
    })),
    excluded_ranges: (document.review.excluded_ranges ?? []).map((range) => ({
      ...range,
    })),
    checks: { ...document.review.checks },
    capo: document.review.capo,
  }
}

function formatSeconds(value: number): string {
  return `${value.toFixed(4)}s`
}

function formatMeasures(measures: number[]): string {
  if (measures.length === 1) return `第 ${measures[0]} 小节`
  return `第 ${measures[0]}-${measures[measures.length - 1]} 小节`
}

function completionCount(checks: ReviewChecks): number {
  return checkOptions.filter((option) => checks[option.key]).length
}

interface DetectedSegment {
  start: number
  end: number
  eventIndexes: number[]
  kind: 'note' | 'chord'
}

interface DraftHistoryEntry {
  draft: ReviewDraft
  selectedIndex: number
}

function detectSegments(
  events: ReviewEvent[],
  duration: number,
): DetectedSegment[] {
  const groups: Array<{
    start: number
    events: Array<{ event: ReviewEvent; index: number }>
  }> = []
  events
    .map((event, index) => ({ event, index }))
    .sort((left, right) => left.event.onset - right.event.onset)
    .forEach((entry) => {
      const current = groups.at(-1)
      if (!current || entry.event.onset - current.start > 0.08) {
        groups.push({ start: entry.event.onset, events: [entry] })
      } else {
        current.events.push(entry)
      }
    })
  return groups.map((group) => ({
    start: group.start,
    end: Math.min(
      duration,
      Math.max(...group.events.map(({ event }) => event.offset)),
    ),
    eventIndexes: group.events.map(({ index }) => index),
    kind:
      new Set(group.events.map(({ event }) => event.string)).size > 1
        ? 'chord'
        : 'note',
  }))
}

export default function ReviewWorkspace() {
  const [reviewQueue] = useState(
    () =>
      new URLSearchParams(window.location.search).get('queue') ?? undefined,
  )
  const residualOnsetQueue =
    reviewQueue === 'residual-onset' || reviewQueue === 'residual-onset-v2'
  const [reviews, setReviews] = useState<ReviewSummary[]>([])
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [document, setDocument] = useState<ReviewDocument | null>(null)
  const [draft, setDraft] = useState<ReviewDraft | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [volume, setVolume] = useState(1)
  const [muted, setMuted] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [showApproval, setShowApproval] = useState(false)
  const [undoCount, setUndoCount] = useState(0)
  const [rangeStart, setRangeStart] = useState('')
  const [rangeEnd, setRangeEnd] = useState('')
  const [rangeReason, setRangeReason] = useState('')
  const [reviewerAlias, setReviewerAlias] = useState(
    () => window.localStorage.getItem('gts-reviewer-alias') ?? '',
  )
  const audioRef = useRef<HTMLAudioElement>(null)
  const eventStopTimer = useRef<number | null>(null)
  const progressFrame = useRef<number | null>(null)
  const undoHistory = useRef<DraftHistoryEntry[]>([])
  const savedDraftSignature = useRef('')

  const resetDraftHistory = useCallback((nextDraft: ReviewDraft) => {
    undoHistory.current = []
    savedDraftSignature.current = JSON.stringify(nextDraft)
    setUndoCount(0)
  }, [])

  useEffect(() => {
    globalThis.document.documentElement.classList.add('review-page')
    return () =>
      globalThis.document.documentElement.classList.remove('review-page')
  }, [])

  const refreshReviews = useCallback(async () => {
    const values = await fetchReviews(reviewQueue)
    setReviews(values)
    return values
  }, [reviewQueue])

  useEffect(() => {
    let active = true
    fetchReviews(reviewQueue)
      .then((values) => {
        if (!active) return
        setReviews(values)
        setSelectedName(
          (current) =>
            current ??
            values.find((review) => review.status === 'draft')?.name ??
            values[0]?.name ??
            null,
        )
        if (values.length === 0) setLoading(false)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '审核包加载失败')
          setLoading(false)
        }
      })
    return () => {
      active = false
    }
  }, [reviewQueue])

  useEffect(() => {
    if (!selectedName) return
    let active = true
    fetchReview(selectedName, reviewQueue)
      .then((value) => {
        if (!active) return
        const nextDraft = draftFromDocument(value)
        setDocument(value)
        setDraft(nextDraft)
        setSelectedIndex(0)
        setCurrentTime(0)
        setRangeStart('')
        setRangeEnd('')
        setRangeReason('')
        setDirty(false)
        resetDraftHistory(nextDraft)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '审核包加载失败')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [resetDraftHistory, reviewQueue, selectedName])

  useEffect(
    () => () => {
      if (eventStopTimer.current !== null) {
        window.clearTimeout(eventStopTimer.current)
      }
    },
    [],
  )

  useEffect(() => {
    if (!isPlaying) return
    const updateProgress = () => {
      if (audioRef.current) setCurrentTime(audioRef.current.currentTime)
      progressFrame.current = window.requestAnimationFrame(updateProgress)
    }
    progressFrame.current = window.requestAnimationFrame(updateProgress)
    return () => {
      if (progressFrame.current !== null) {
        window.cancelAnimationFrame(progressFrame.current)
        progressFrame.current = null
      }
    }
  }, [isPlaying])

  const selectedEvent = draft?.events[selectedIndex] ?? null
  const readOnly = document?.review.status === 'approved'
  const requiredChecksComplete =
    draft !== null &&
    draft.checks.timing &&
    (residualOnsetQueue || draft.checks.string_fret) &&
    draft.checks.completeness
  const approvedCount = reviews.filter(
    (review) => review.status === 'approved',
  ).length
  const selectedSummary = reviews.find(
    (review) => review.name === selectedName,
  )
  const unknownTechniqueCount = useMemo(
    () =>
      draft?.events.filter((event) => event.technique === 'unknown').length ?? 0,
    [draft],
  )
  const excludedDuration = useMemo(
    () =>
      draft?.excluded_ranges.reduce(
        (total, range) => total + range.end - range.start,
        0,
      ) ?? 0,
    [draft?.excluded_ranges],
  )
  const detectedSegments = useMemo(
    () => detectSegments(draft?.events ?? [], document?.review.duration ?? 0),
    [document?.review.duration, draft?.events],
  )
  const activeSegmentIndex = useMemo(() => {
    if (detectedSegments.length === 0) return -1
    const containing = detectedSegments.findIndex(
      (segment) =>
        currentTime >= segment.start - 0.0001 &&
        currentTime <= segment.end + 0.0001,
    )
    if (containing >= 0) return containing
    const previous = detectedSegments.findLastIndex(
      (segment) => segment.start <= currentTime,
    )
    return Math.max(0, previous)
  }, [currentTime, detectedSegments])
  const canUndo = undoCount > 0

  const stopPlaybackTimer = useCallback(() => {
    if (eventStopTimer.current !== null) {
      window.clearTimeout(eventStopTimer.current)
      eventStopTimer.current = null
    }
  }, [])

  const seekTo = useCallback((seconds: number) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = Math.max(0, seconds)
    setCurrentTime(audio.currentTime)
  }, [])

  const focusDetectedSegment = useCallback(
    (index: number) => {
      const segment = detectedSegments[index]
      if (!segment) return
      setSelectedIndex(segment.eventIndexes[0])
      seekTo(segment.start)
    },
    [detectedSegments, seekTo],
  )

  const playEvent = useCallback(
    async (event: ReviewEvent) => {
      const audio = audioRef.current
      if (!audio) return
      stopPlaybackTimer()
      audio.pause()
      const start = Math.max(0, event.onset - 0.06)
      const end = Math.min(document?.review.duration ?? event.offset, event.offset + 0.08)
      audio.currentTime = start
      audio.playbackRate = playbackRate
      try {
        await audio.play()
        eventStopTimer.current = window.setTimeout(
          () => {
            audio.pause()
            audio.currentTime = event.onset
            setCurrentTime(event.onset)
          },
          Math.max(80, ((end - start) / playbackRate) * 1000),
        )
      } catch {
        setError('浏览器未允许音频播放，请先点击完整片段播放器')
      }
    },
    [document?.review.duration, playbackRate, stopPlaybackTimer],
  )

  const toggleFullPlayback = useCallback(async () => {
    const audio = audioRef.current
    if (!audio) return
    stopPlaybackTimer()
    if (!audio.paused) {
      audio.pause()
      return
    }
    audio.playbackRate = playbackRate
    try {
      await audio.play()
    } catch {
      setError('浏览器未允许音频播放')
    }
  }, [playbackRate, stopPlaybackTimer])

  const restartPlayback = useCallback(async () => {
    const audio = audioRef.current
    if (!audio) return
    stopPlaybackTimer()
    audio.pause()
    audio.currentTime = 0
    audio.playbackRate = playbackRate
    setCurrentTime(0)
    try {
      await audio.play()
    } catch {
      setError('浏览器未允许音频播放')
    }
  }, [playbackRate, stopPlaybackTimer])

  const commitDraft = useCallback(
    (nextDraft: ReviewDraft, nextSelectedIndex = selectedIndex) => {
      if (!draft || readOnly) return
      undoHistory.current = [
        ...undoHistory.current.slice(-99),
        { draft, selectedIndex },
      ]
      setDraft(nextDraft)
      setSelectedIndex(nextSelectedIndex)
      setDirty(JSON.stringify(nextDraft) !== savedDraftSignature.current)
      setNotice(null)
      setUndoCount(undoHistory.current.length)
    },
    [draft, readOnly, selectedIndex],
  )

  const updateSelectedEvent = useCallback(
    (key: keyof ReviewEvent, value: number | Technique) => {
      if (!draft || readOnly || !draft.events[selectedIndex]) return
      const events = [...draft.events]
      events[selectedIndex] = { ...events[selectedIndex], [key]: value }
      commitDraft({ ...draft, events })
    },
    [commitDraft, draft, readOnly, selectedIndex],
  )

  const updateCheck = useCallback(
    (key: keyof ReviewChecks, value: boolean) => {
      if (!draft || readOnly) return
      commitDraft({
        ...draft,
        checks: { ...draft.checks, [key]: value },
      })
    },
    [commitDraft, draft, readOnly],
  )

  const addEvent = useCallback(() => {
    if (!draft || !document || readOnly) return
    const onset = Math.min(
      Math.max(0, audioRef.current?.currentTime ?? currentTime),
      Math.max(0, document.review.duration - 0.02),
    )
    const event: ReviewEvent = {
      onset: Number(onset.toFixed(4)),
      offset: Number(
        Math.min(document.review.duration, onset + 0.25).toFixed(4),
      ),
      string: 1,
      fret: 0,
      technique: 'unknown',
      confidence: 1,
    }
    const events = [...draft.events, event].sort(
      (left, right) => left.onset - right.onset || left.string - right.string,
    )
    commitDraft({ ...draft, events }, events.indexOf(event))
  }, [commitDraft, currentTime, document, draft, readOnly])

  const deleteSelectedEvent = useCallback(() => {
    if (!draft || !selectedEvent || readOnly) return
    const events = draft.events.filter((_event, index) => index !== selectedIndex)
    commitDraft(
      { ...draft, events },
      Math.max(0, Math.min(selectedIndex, events.length - 1)),
    )
  }, [commitDraft, draft, readOnly, selectedEvent, selectedIndex])

  const setCurrentTimeForRange = useCallback(
    (boundary: 'start' | 'end') => {
      const value = Math.min(
        document?.review.duration ?? currentTime,
        Math.max(0, audioRef.current?.currentTime ?? currentTime),
      ).toFixed(4)
      if (boundary === 'start') {
        setRangeStart(value)
      } else {
        setRangeEnd(value)
      }
    },
    [currentTime, document?.review.duration],
  )

  const addExcludedRange = useCallback(() => {
    if (!draft || !document || readOnly) return
    const start = Number(rangeStart)
    const end = Number(rangeEnd)
    const reason = rangeReason.trim()
    if (
      rangeStart === '' ||
      rangeEnd === '' ||
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      start < 0 ||
      start >= end ||
      end > document.review.duration
    ) {
      setError('排除区间必须位于片段内，且终点晚于起点')
      return
    }
    if (!reason || reason.length > 200) {
      setError('排除原因必须为 1 到 200 个字符')
      return
    }
    if (
      draft.excluded_ranges.some(
        (range) => range.start < end && start < range.end,
      )
    ) {
      setError('排除区间不能与已有区间重叠')
      return
    }
    const excludedRange: ReviewExcludedRange = { start, end, reason }
    const excludedRanges = [...draft.excluded_ranges, excludedRange].sort(
      (left, right) => left.start - right.start || left.end - right.end,
    )
    commitDraft({
      ...draft,
      excluded_ranges: excludedRanges,
      checks: { ...draft.checks, completeness: false },
    })
    setRangeStart('')
    setRangeEnd('')
    setRangeReason('')
    setError(null)
  }, [
    commitDraft,
    document,
    draft,
    rangeEnd,
    rangeReason,
    rangeStart,
    readOnly,
  ])

  const removeExcludedRange = useCallback(
    (index: number) => {
      if (!draft || readOnly) return
      commitDraft({
        ...draft,
        excluded_ranges: draft.excluded_ranges.filter(
          (_range, rangeIndex) => rangeIndex !== index,
        ),
        checks: { ...draft.checks, completeness: false },
      })
    },
    [commitDraft, draft, readOnly],
  )

  const undoDraft = useCallback(() => {
    if (!draft || readOnly) return
    const previous = undoHistory.current.pop()
    if (!previous) return
    setDraft(previous.draft)
    setSelectedIndex(
      Math.max(
        0,
        Math.min(previous.selectedIndex, previous.draft.events.length - 1),
      ),
    )
    setDirty(
      JSON.stringify(previous.draft) !== savedDraftSignature.current,
    )
    setNotice('已撤回上一步')
    setUndoCount(undoHistory.current.length)
  }, [draft, readOnly])

  useEffect(() => {
    const handleDeleteKey = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      if (event.repeat) return
      const target = event.target
      if (
        target instanceof HTMLElement &&
        target.closest('input, select, textarea, [contenteditable="true"]')
      ) {
        return
      }
      if (!selectedEvent || readOnly) return
      event.preventDefault()
      deleteSelectedEvent()
    }
    window.addEventListener('keydown', handleDeleteKey)
    return () => window.removeEventListener('keydown', handleDeleteKey)
  }, [deleteSelectedEvent, readOnly, selectedEvent])

  useEffect(() => {
    const handleUndoKey = (event: KeyboardEvent) => {
      if (
        event.key.toLowerCase() !== 'z' ||
        (!event.metaKey && !event.ctrlKey)
      ) {
        return
      }
      const target = event.target
      if (
        target instanceof HTMLElement &&
        target.closest('input, select, textarea, [contenteditable="true"]')
      ) {
        return
      }
      if (undoHistory.current.length === 0 || readOnly) return
      event.preventDefault()
      undoDraft()
    }
    window.addEventListener('keydown', handleUndoKey)
    return () => window.removeEventListener('keydown', handleUndoKey)
  }, [readOnly, undoDraft])

  const acceptDocument = useCallback((value: ReviewDocument) => {
    const nextDraft = draftFromDocument(value)
    setDocument(value)
    setDraft(nextDraft)
    setDirty(false)
    setSelectedIndex((index) =>
      Math.max(0, Math.min(index, value.review.events.length - 1)),
    )
    setRangeStart('')
    setRangeEnd('')
    setRangeReason('')
    resetDraftHistory(nextDraft)
  }, [resetDraftHistory])

  const handleSave = useCallback(async () => {
    if (!selectedName || !draft || readOnly) return
    setSaving(true)
    setError(null)
    try {
      const value = await saveReview(selectedName, draft, reviewQueue)
      acceptDocument(value)
      await refreshReviews()
      setNotice('草稿已保存')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '草稿保存失败')
    } finally {
      setSaving(false)
    }
  }, [
    acceptDocument,
    draft,
    readOnly,
    refreshReviews,
    reviewQueue,
    selectedName,
  ])

  const handleApprove = useCallback(async () => {
    if (!selectedName || !draft || !requiredChecksComplete || readOnly) return
    setSaving(true)
    setError(null)
    try {
      const value = await approveReview(
        selectedName,
        draft,
        reviewerAlias,
        reviewQueue,
      )
      window.localStorage.setItem('gts-reviewer-alias', reviewerAlias.trim())
      acceptDocument(value)
      await refreshReviews()
      setShowApproval(false)
      setNotice('审核包已批准并锁定')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '审核包批准失败')
    } finally {
      setSaving(false)
    }
  }, [
    acceptDocument,
    draft,
    readOnly,
    refreshReviews,
    requiredChecksComplete,
    reviewQueue,
    reviewerAlias,
    selectedName,
  ])

  const handleReopen = useCallback(async () => {
    if (!selectedName || !readOnly) return
    if (!window.confirm('重新打开后需要再次确认并批准，是否继续？')) return
    setSaving(true)
    setError(null)
    try {
      const value = await reopenReview(
        selectedName,
        '人工修订',
        reviewQueue,
      )
      acceptDocument(value)
      await refreshReviews()
      setNotice('审核包已重新打开，时间边界需要重新确认')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '重新打开失败')
    } finally {
      setSaving(false)
    }
  }, [
    acceptDocument,
    readOnly,
    refreshReviews,
    reviewQueue,
    selectedName,
  ])

  const chooseReview = useCallback(
    (name: string) => {
      if (name === selectedName) return
      if (dirty && !window.confirm('当前草稿尚未保存，确认切换审核包？')) return
      audioRef.current?.pause()
      stopPlaybackTimer()
      setLoading(true)
      setError(null)
      setNotice(null)
      setSelectedName(name)
    },
    [dirty, selectedName, stopPlaybackTimer],
  )

  if (!document || !draft) {
    return (
      <main className="review-loading">
        <AudioLines aria-hidden="true" />
        <strong>{loading ? '正在加载审核包' : '没有可用审核包'}</strong>
        {error && <span>{error}</span>}
      </main>
    )
  }

  return (
    <div className="review-shell">
      <header className="review-header">
        <div className="review-brand">
          <span className="review-brand-mark">
            <Guitar aria-hidden="true" strokeWidth={2.5} />
          </span>
          <div>
            <strong>Guitar Tab Review</strong>
            <span>{residualOnsetQueue ? 'Residual Onset · HackAll' : 'Gold Review · HackAll'}</span>
          </div>
        </div>
        <div className="review-progress" aria-label="审核进度">
          <span>{approvedCount} / {reviews.length} 已批准</span>
          <i>
            <b
              style={{
                width: `${reviews.length ? (approvedCount / reviews.length) * 100 : 0}%`,
              }}
            />
          </i>
        </div>
        <div className="review-header-actions">
          <a className="review-icon-button" href="/" title="返回工作台">
            <ArrowLeft aria-hidden="true" />
            <span className="sr-only">返回工作台</span>
          </a>
          <span className={`review-save-state ${dirty ? 'is-dirty' : ''}`}>
            {readOnly ? '已锁定' : dirty ? '未保存' : '已保存'}
          </span>
          <button
            className="review-icon-button review-undo-button"
            type="button"
            disabled={!canUndo || readOnly}
            title="撤回上一步"
            aria-label="撤回上一步"
            onClick={undoDraft}
          >
            <Undo2 aria-hidden="true" />
          </button>
          {readOnly ? (
            <button
              className="review-secondary-button"
              type="button"
              disabled={saving}
              onClick={() => void handleReopen()}
            >
              <Unlock aria-hidden="true" />
              重新编辑
            </button>
          ) : (
            <button
              className="review-secondary-button"
              type="button"
              disabled={!dirty || saving}
              onClick={() => void handleSave()}
            >
              <Save aria-hidden="true" />
              保存草稿
            </button>
          )}
          <button
            className="review-primary-button"
            type="button"
            disabled={
              saving ||
              readOnly ||
              !requiredChecksComplete ||
              reviewerAlias.trim().length === 0
            }
            onClick={() => setShowApproval(true)}
          >
            <CheckCircle2 aria-hidden="true" />
            批准
          </button>
        </div>
      </header>

      <aside className="review-package-rail" aria-label="审核包列表">
        <div className="review-rail-heading">
          <strong>
            {residualOnsetQueue ? '训练标注片段' : '待审核片段'}
          </strong>
          <span>{reviews.reduce((total, review) => total + review.eventCount, 0)} 个事件</span>
        </div>
        <div className="review-package-list">
          {reviews.map((review, index) => (
            <button
              className={`review-package ${
                review.name === selectedName ? 'is-active' : ''
              }`}
              key={review.name}
              type="button"
              onClick={() => chooseReview(review.name)}
            >
              <span className="review-package-index">
                {review.status === 'approved' ? (
                  <Check aria-hidden="true" />
                ) : (
                  String(index + 1).padStart(2, '0')
                )}
              </span>
              <span className="review-package-copy">
                <strong>
                  {review.projectTitle ?? formatMeasures(review.measureNumbers)}
                </strong>
                <small>
                  {residualOnsetQueue && review.projectArtist
                    ? review.projectArtist
                    : formatMeasures(review.measureNumbers)}
                  {' · '}
                  {review.eventCount} 音符
                </small>
              </span>
              <span
                className={`review-package-status is-${review.status}`}
                title={review.status === 'approved' ? '已批准' : '草稿'}
              />
            </button>
          ))}
        </div>
      </aside>

      <main className="review-main">
        <section className="review-player-band">
          <div className="review-title-row">
            <div>
              <span className="review-kicker">{document.name}</span>
              <h1>
                {selectedSummary?.projectTitle ??
                  formatMeasures(document.review.measure_numbers)}
              </h1>
              <span className="review-clip-context">
                {residualOnsetQueue
                  ? selectedSummary?.projectArtist ??
                    formatMeasures(document.review.measure_numbers)
                  : `${formatMeasures(document.review.measure_numbers)}${
                      selectedSummary?.projectArtist
                        ? ` · ${selectedSummary.projectArtist}`
                        : ''
                    }`}
              </span>
            </div>
            <dl>
              <div>
                <dt>BPM</dt>
                <dd>{document.review.bpm.toFixed(2)}</dd>
              </div>
              <div>
                <dt>变调夹</dt>
                <dd>{draft.capo === 0 ? '无' : `${draft.capo} 品`}</dd>
              </div>
              <div>
                <dt>事件</dt>
                <dd>{draft.events.length}</dd>
              </div>
            </dl>
          </div>

          <div className="review-transport">
            <button
              className="review-play-button"
              type="button"
              onClick={() => void toggleFullPlayback()}
              aria-label={isPlaying ? '暂停完整片段' : '播放完整片段'}
            >
              {isPlaying ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
            </button>
            <button
              className="review-restart-button"
              type="button"
              title="从头播放"
              aria-label="从头播放"
              onClick={() => void restartPlayback()}
            >
              <SkipBack aria-hidden="true" />
            </button>
            <div className="review-precise-progress">
              <input
                type="range"
                min="0"
                max={document.review.duration}
                step="0.0001"
                value={Math.min(currentTime, document.review.duration)}
                aria-label="精确播放进度"
                aria-valuetext={formatSeconds(currentTime)}
                onChange={(event) => seekTo(Number(event.target.value))}
              />
              <div className="review-precise-time">
                <output>{formatSeconds(currentTime)}</output>
                <span>/</span>
                <span>{formatSeconds(document.review.duration)}</span>
              </div>
            </div>
            <div className="review-transport-options">
              <div className="review-segment-nav">
                <button
                  type="button"
                  disabled={activeSegmentIndex <= 0}
                  title="上一个检测音组"
                  aria-label="上一个检测音组"
                  onClick={() => focusDetectedSegment(activeSegmentIndex - 1)}
                >
                  <ChevronLeft aria-hidden="true" />
                </button>
                <span>
                  {activeSegmentIndex >= 0
                    ? `${
                        detectedSegments[activeSegmentIndex].kind === 'chord'
                          ? '和弦'
                          : '单音'
                      } ${activeSegmentIndex + 1}/${detectedSegments.length}`
                    : '无音组'}
                </span>
                <button
                  type="button"
                  disabled={
                    activeSegmentIndex < 0 ||
                    activeSegmentIndex >= detectedSegments.length - 1
                  }
                  title="下一个检测音组"
                  aria-label="下一个检测音组"
                  onClick={() => focusDetectedSegment(activeSegmentIndex + 1)}
                >
                  <ChevronRight aria-hidden="true" />
                </button>
              </div>
              <label className="review-speed">
                <span>速度</span>
                <select
                  value={playbackRate}
                  onChange={(event) => {
                    const rate = Number(event.target.value)
                    setPlaybackRate(rate)
                    if (audioRef.current) audioRef.current.playbackRate = rate
                  }}
                >
                  {playbackRates.map((rate) => (
                    <option key={rate} value={rate}>{rate}x</option>
                  ))}
                </select>
              </label>
              <div className="review-volume">
                <button
                  type="button"
                  aria-label={muted ? '取消静音' : '静音'}
                  title={muted ? '取消静音' : '静音'}
                  onClick={() => {
                    const nextMuted = !muted
                    setMuted(nextMuted)
                    if (audioRef.current) audioRef.current.muted = nextMuted
                  }}
                >
                  {muted || volume === 0 ? (
                    <VolumeX aria-hidden="true" />
                  ) : (
                    <Volume2 aria-hidden="true" />
                  )}
                </button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={volume}
                  aria-label="音量"
                  onChange={(event) => {
                    const nextVolume = Number(event.target.value)
                    setVolume(nextVolume)
                    setMuted(false)
                    if (audioRef.current) {
                      audioRef.current.volume = nextVolume
                      audioRef.current.muted = false
                    }
                  }}
                />
              </div>
            </div>
            <audio
              className="review-audio-engine"
              ref={audioRef}
              preload="metadata"
              src={apiAssetUrl(document.audioUrl)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => {
                setIsPlaying(false)
                setCurrentTime(document.review.duration)
              }}
              onTimeUpdate={(event) =>
                setCurrentTime(event.currentTarget.currentTime)
              }
            />
          </div>

          <div
            className="review-timeline"
            aria-label="六弦事件时间轴"
            onClick={(event) => {
              if (event.target !== event.currentTarget) return
              const bounds = event.currentTarget.getBoundingClientRect()
              seekTo(
                ((event.clientX - bounds.left) / bounds.width) *
                  document.review.duration,
              )
            }}
          >
            <div className="review-segment-guides" aria-label="检测音组边界">
              {detectedSegments.map((segment, index) => (
                <button
                  className={`${segment.kind === 'chord' ? 'is-chord' : ''} ${
                    activeSegmentIndex === index ? 'is-active' : ''
                  }`}
                  key={`${index}-${segment.start}`}
                  type="button"
                  style={{
                    left: `${(segment.start / document.review.duration) * 100}%`,
                    width: `${Math.max(
                      0.8,
                      ((segment.end - segment.start) /
                        document.review.duration) *
                        100,
                    )}%`,
                  }}
                  title={`${segment.kind === 'chord' ? '和弦' : '单音'} ${
                    index + 1
                  } · ${formatSeconds(segment.start)}–${formatSeconds(segment.end)}`}
                  aria-label={`跳到${segment.kind === 'chord' ? '和弦' : '单音'} ${
                    index + 1
                  }，${formatSeconds(segment.start)}到${formatSeconds(segment.end)}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    focusDetectedSegment(index)
                  }}
                >
                  <span>{index + 1}</span>
                </button>
              ))}
            </div>
            <div className="review-excluded-range-layer" aria-hidden="true">
              {draft.excluded_ranges.map((range, index) => (
                <span
                  key={`${index}-${range.start}-${range.end}`}
                  style={{
                    left: `${(range.start / document.review.duration) * 100}%`,
                    width: `${
                      ((range.end - range.start) / document.review.duration) *
                      100
                    }%`,
                  }}
                >
                  <i>排除</i>
                </span>
              ))}
            </div>
            <span
              className="review-timeline-playhead"
              style={{
                left: `${Math.min(100, (currentTime / document.review.duration) * 100)}%`,
              }}
            />
            {[1, 2, 3, 4, 5, 6].map((string) => (
              <div className="review-string-lane" key={string}>
                <span>{string}</span>
                <i />
                {draft.events.map((note, index) =>
                  note.string === string ? (
                    <button
                      className={`review-event-mark ${
                        selectedIndex === index ? 'is-selected' : ''
                      }`}
                      key={`${index}-${note.onset}-${note.string}`}
                      type="button"
                      style={{
                        left: `${(note.onset / document.review.duration) * 100}%`,
                        width: `${Math.max(
                          0.6,
                          ((note.offset - note.onset) /
                            document.review.duration) *
                            100,
                        )}%`,
                      }}
                      title={`事件 ${index + 1}，${note.string} 弦 ${note.fret} 品`}
                      aria-label={`选择事件 ${index + 1}`}
                      onClick={() => {
                        setSelectedIndex(index)
                        seekTo(note.onset)
                      }}
                    />
                  ) : null,
                )}
              </div>
            ))}
          </div>
          <div className="review-time-axis">
            <span>0.0000s</span>
            <span>{formatSeconds(document.review.duration / 2)}</span>
            <span>{formatSeconds(document.review.duration)}</span>
          </div>
        </section>

        {(error || notice) && (
          <div className={`review-message ${error ? 'is-error' : 'is-success'}`}>
            {error ? <CircleAlert aria-hidden="true" /> : <Check aria-hidden="true" />}
            <span>{error ?? notice}</span>
            <button
              type="button"
              aria-label="关闭提示"
              onClick={() => {
                setError(null)
                setNotice(null)
              }}
            >
              <X aria-hidden="true" />
            </button>
          </div>
        )}

        <section className="review-event-section">
          <div className="review-section-heading">
            <div>
              <h2>音符事件</h2>
              <span>{unknownTechniqueCount} 个技法待确认</span>
            </div>
            <button
              className="review-secondary-button"
              type="button"
              disabled={readOnly}
              onClick={addEvent}
            >
              <Plus aria-hidden="true" />
              添加事件
            </button>
          </div>
          <div className="review-table-wrap">
            <table className="review-event-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>起点</th>
                  <th>终点</th>
                  <th>弦</th>
                  <th>品</th>
                  <th>技法</th>
                  <th><span className="sr-only">播放</span></th>
                </tr>
              </thead>
              <tbody>
                {draft.events.map((note, index) => (
                  <tr
                    className={selectedIndex === index ? 'is-selected' : ''}
                    key={`${index}-${note.onset}-${note.string}`}
                    tabIndex={selectedIndex === index ? 0 : -1}
                    aria-selected={selectedIndex === index}
                    aria-keyshortcuts="Delete Backspace"
                    onClick={(event) => {
                      setSelectedIndex(index)
                      seekTo(note.onset)
                      event.currentTarget.focus()
                    }}
                  >
                    <td>{String(index + 1).padStart(3, '0')}</td>
                    <td>{note.onset.toFixed(4)}</td>
                    <td>{note.offset.toFixed(4)}</td>
                    <td>{note.string}</td>
                    <td>{note.fret}</td>
                    <td>
                      <span className={`review-technique is-${note.technique}`}>
                        {techniqueLabel[note.technique]}
                      </span>
                    </td>
                    <td>
                      <button
                        className="review-row-play"
                        type="button"
                        title="播放此事件"
                        aria-label={`播放事件 ${index + 1}`}
                        onClick={(event) => {
                          event.stopPropagation()
                          setSelectedIndex(index)
                          void playEvent(note)
                        }}
                      >
                        <Play aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <aside className="review-inspector">
        <section>
          <div className="review-inspector-heading">
            <div>
              <span>当前事件</span>
              <strong>
                {selectedEvent ? `#${String(selectedIndex + 1).padStart(3, '0')}` : '无'}
              </strong>
            </div>
            <div>
              <button
                className="review-icon-button"
                type="button"
                disabled={!selectedEvent}
                title="播放所选事件"
                aria-label="播放所选事件"
                onClick={() => selectedEvent && void playEvent(selectedEvent)}
              >
                <Play aria-hidden="true" />
              </button>
              <button
                className="review-icon-button is-danger"
                type="button"
                disabled={!selectedEvent || readOnly}
                title="删除所选事件"
                aria-label="删除所选事件"
                onClick={deleteSelectedEvent}
              >
                <Trash2 aria-hidden="true" />
              </button>
            </div>
          </div>

          {selectedEvent && (
            <div className="review-event-editor">
              <label>
                <span>起点（秒）</span>
                <input
                  type="number"
                  min="0"
                  max={document.review.duration}
                  step="0.0001"
                  value={selectedEvent.onset}
                  disabled={readOnly}
                  onChange={(event) =>
                    event.target.value &&
                    updateSelectedEvent('onset', Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>终点（秒）</span>
                <input
                  type="number"
                  min="0.0001"
                  max={document.review.duration + 0.05}
                  step="0.0001"
                  value={selectedEvent.offset}
                  disabled={readOnly}
                  onChange={(event) =>
                    event.target.value &&
                    updateSelectedEvent('offset', Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>弦</span>
                <select
                  value={selectedEvent.string}
                  disabled={readOnly}
                  onChange={(event) =>
                    updateSelectedEvent('string', Number(event.target.value))
                  }
                >
                  {[1, 2, 3, 4, 5, 6].map((string) => (
                    <option key={string} value={string}>
                      {string} 弦
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>绝对品位</span>
                <input
                  type="number"
                  min="0"
                  max="24"
                  step="1"
                  value={selectedEvent.fret}
                  disabled={readOnly}
                  onChange={(event) =>
                    event.target.value &&
                    updateSelectedEvent('fret', Number(event.target.value))
                  }
                />
              </label>
              <label className="review-editor-wide">
                <span>演奏技法</span>
                <select
                  value={selectedEvent.technique}
                  disabled={readOnly}
                  onChange={(event) =>
                    updateSelectedEvent(
                      'technique',
                      event.target.value as Technique,
                    )
                  }
                >
                  {techniqueOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <div className="review-editor-source review-editor-wide">
                <span>候选来源</span>
                <strong>
                  {selectedEvent.candidate_provenance?.position_source ?? '人工新增'}
                </strong>
              </div>
            </div>
          )}
        </section>

        <section className="review-exclusion-section">
          <div className="review-check-title">
            <div>
              <span>不确定区间</span>
              <strong>{draft.excluded_ranges.length}</strong>
            </div>
            <CircleAlert aria-hidden="true" />
          </div>
          {draft.excluded_ranges.length > 0 ? (
            <>
              <div className="review-exclusion-summary">
                <strong>{formatSeconds(excludedDuration)} 不参与评测</strong>
                <small>按事件起音过滤参考与预测结果</small>
              </div>
              <div className="review-exclusion-list">
                {draft.excluded_ranges.map((range, index) => (
                  <div key={`${index}-${range.start}-${range.end}`}>
                    <button
                      type="button"
                      title="跳到区间起点"
                      onClick={() => seekTo(range.start)}
                    >
                      <strong>
                        {range.start.toFixed(4)}–{range.end.toFixed(4)}
                      </strong>
                      <small>{range.reason}</small>
                    </button>
                    <button
                      className="review-exclusion-delete"
                      type="button"
                      disabled={readOnly}
                      title="删除排除区间"
                      aria-label={`删除排除区间 ${index + 1}`}
                      onClick={() => removeExcludedRange(index)}
                    >
                      <Trash2 aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="review-exclusion-empty">当前片段全部纳入评测</p>
          )}
          {!readOnly && (
            <div className="review-exclusion-editor">
              <label>
                <span>起点（秒）</span>
                <span className="review-range-input">
                  <input
                    type="number"
                    min="0"
                    max={document.review.duration}
                    step="0.0001"
                    value={rangeStart}
                    aria-label="排除区间起点"
                    onChange={(event) => setRangeStart(event.target.value)}
                  />
                  <button
                    type="button"
                    title="使用当前播放位置作为起点"
                    aria-label="使用当前播放位置作为起点"
                    onClick={() => setCurrentTimeForRange('start')}
                  >
                    <LocateFixed aria-hidden="true" />
                  </button>
                </span>
              </label>
              <label>
                <span>终点（秒）</span>
                <span className="review-range-input">
                  <input
                    type="number"
                    min="0.0001"
                    max={document.review.duration}
                    step="0.0001"
                    value={rangeEnd}
                    aria-label="排除区间终点"
                    onChange={(event) => setRangeEnd(event.target.value)}
                  />
                  <button
                    type="button"
                    title="使用当前播放位置作为终点"
                    aria-label="使用当前播放位置作为终点"
                    onClick={() => setCurrentTimeForRange('end')}
                  >
                    <LocateFixed aria-hidden="true" />
                  </button>
                </span>
              </label>
              <label className="review-exclusion-reason">
                <span>原因</span>
                <input
                  type="text"
                  maxLength={200}
                  value={rangeReason}
                  aria-label="排除原因"
                  placeholder="例如：轻扫过弱，无法确认实际触弦"
                  onChange={(event) => setRangeReason(event.target.value)}
                />
              </label>
              <button
                className="review-secondary-button review-exclusion-add"
                type="button"
                onClick={addExcludedRange}
              >
                <Plus aria-hidden="true" />
                添加区间
              </button>
            </div>
          )}
        </section>

        <section className="review-check-section">
          <div className="review-check-title">
            <div>
              <span>片段检查</span>
              <strong>{completionCount(draft.checks)} / 4</strong>
            </div>
            <Clock3 aria-hidden="true" />
          </div>
          <div className="review-check-list">
            {checkOptions.map((option) => (
              <label key={option.key}>
                <input
                  type="checkbox"
                  checked={draft.checks[option.key]}
                  disabled={readOnly}
                  onChange={(event) =>
                    updateCheck(option.key, event.target.checked)
                  }
                />
                <span>
                  <strong>
                    {option.label}
                    {option.required &&
                      !(
                        residualOnsetQueue &&
                        option.key === 'string_fret'
                      ) && <i>必选</i>}
                  </strong>
                  <small>
                    {option.key === 'completeness' &&
                    draft.excluded_ranges.length > 0
                      ? '未排除区间已补漏并清理误报'
                      : residualOnsetQueue &&
                          option.key === 'string_fret'
                        ? '同音异弦可保留预填结果'
                      : option.detail}
                  </small>
                </span>
              </label>
            ))}
          </div>
          <label className="reviewer-field">
            <span>审核人代号</span>
            <input
              type="text"
              value={
                readOnly
                  ? document.review.approval?.reviewer_alias ?? ''
                  : reviewerAlias
              }
              disabled={readOnly}
              maxLength={64}
              placeholder="例如 reviewer-1"
              onChange={(event) => setReviewerAlias(event.target.value)}
            />
          </label>
          {readOnly && document.review.approval && (
            <div className="review-lock-record">
              <CheckCircle2 aria-hidden="true" />
              <span>
                <strong>内容已锁定</strong>
                <small>
                  {new Date(document.review.approval.reviewed_at).toLocaleString(
                    'zh-CN',
                  )}
                </small>
              </span>
            </div>
          )}
        </section>
      </aside>

      {showApproval && (
        <div
          className="review-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !saving) {
              setShowApproval(false)
            }
          }}
        >
          <section
            className="review-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="approval-title"
          >
            <div className="review-dialog-icon">
              <CheckCircle2 aria-hidden="true" />
            </div>
            <h2 id="approval-title">批准并锁定此审核包？</h2>
            <p>
              将绑定 {draft.events.length} 个事件、音频与项目快照。批准后不可在此界面修改。
            </p>
            <dl>
              <div><dt>审核人</dt><dd>{reviewerAlias.trim()}</dd></div>
              <div>
                <dt>片段</dt>
                <dd>
                  {residualOnsetQueue
                    ? selectedSummary?.projectArtist ??
                      formatMeasures(document.review.measure_numbers)
                    : formatMeasures(document.review.measure_numbers)}
                </dd>
              </div>
              <div>
                <dt>排除</dt>
                <dd>
                  {draft.excluded_ranges.length > 0
                    ? `${draft.excluded_ranges.length} 段 / ${formatSeconds(excludedDuration)}`
                    : '无'}
                </dd>
              </div>
              <div><dt>技法</dt><dd>{draft.checks.technique ? '已审核' : '未纳入审核'}</dd></div>
            </dl>
            <div className="review-dialog-actions">
              <button
                className="review-secondary-button"
                type="button"
                disabled={saving}
                onClick={() => setShowApproval(false)}
              >
                取消
              </button>
              <button
                className="review-primary-button"
                type="button"
                disabled={saving}
                onClick={() => void handleApprove()}
              >
                <Check aria-hidden="true" />
                {saving ? '正在批准' : '确认批准'}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
