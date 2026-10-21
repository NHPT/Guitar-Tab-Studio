import {
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Clock3,
  ListFilter,
  LoaderCircle,
  Pause,
  Play,
  Plus,
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
  fetchBeatGoldStatus,
  fetchBeatReview,
  fetchBeatReviews,
  saveBeatReview,
  submitBeatReview,
} from './api'
import type {
  BeatGoldStatus,
  BeatReviewDocument,
  BeatReviewDraft,
  BeatReviewMarker,
  BeatReviewSummary,
} from './types'
import './BeatReviewWorkspace.css'

type SplitFilter = 'all' | 'train' | 'validation' | 'test'
type StatusFilter = 'all' | 'draft' | 'submitted' | 'adjudicated'

const checkOptions: Array<{
  key: keyof BeatReviewDraft['checks']
  label: string
}> = [
  { key: 'beat_times', label: '拍点时间' },
  { key: 'downbeats', label: '强拍位置' },
  { key: 'meter', label: '拍号与弱起' },
  { key: 'completeness', label: '片段完整性' },
]

function draftFromDocument(document: BeatReviewDocument): BeatReviewDraft {
  return {
    beats: document.review.beats.map((marker) => ({ ...marker })),
    time_signature: [...document.review.time_signature],
    pickup_beats: document.review.pickup_beats,
    checks: { ...document.review.checks },
  }
}

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60)
  const seconds = value - minutes * 60
  return `${minutes}:${seconds.toFixed(3).padStart(6, '0')}`
}

function rephaseMarkers(
  markers: readonly BeatReviewMarker[],
  numerator: number,
  anchorIndex: number,
  anchorBeat: number,
): BeatReviewMarker[] {
  if (markers.length === 0) return []
  const next = markers.map((marker) => ({ ...marker }))
  const safeAnchor = Math.max(0, Math.min(anchorIndex, next.length - 1))
  const selected = next[safeAnchor]
  selected.beat = Math.max(1, Math.min(numerator, anchorBeat))
  selected.downbeat = selected.beat === 1
  if (selected.downbeat && selected.measure === 0) selected.measure = 1
  for (let index = safeAnchor - 1; index >= 0; index -= 1) {
    const after = next[index + 1]
    const beat = after.beat === 1 ? numerator : after.beat - 1
    const measure = after.measure - (after.beat === 1 ? 1 : 0)
    next[index] = {
      ...next[index],
      beat,
      measure: Math.max(0, measure),
      downbeat: beat === 1,
    }
  }
  for (let index = safeAnchor + 1; index < next.length; index += 1) {
    const before = next[index - 1]
    const beat = before.beat % numerator + 1
    next[index] = {
      ...next[index],
      beat,
      measure: before.measure + (beat === 1 ? 1 : 0),
      downbeat: beat === 1,
    }
  }
  return next
}

function partialMeasureRanges(
  markers: readonly BeatReviewMarker[],
  start: number,
  end: number,
): Array<{ start: number; end: number }> {
  const downbeats = markers
    .filter((marker) => marker.downbeat)
    .map((marker) => marker.at)
  const first = downbeats.find((at) => at >= start)
  const last = [...downbeats].reverse().find((at) => at < end)
  const ranges: Array<{ start: number; end: number }> = []
  if (first !== undefined && first > start + 0.001) {
    ranges.push({ start, end: Math.min(end, first) })
  }
  const nextAfterLast =
    last === undefined ? undefined : downbeats.find((at) => at > last)
  if (
    last !== undefined &&
    last < end - 0.001 &&
    (nextAfterLast === undefined || nextAfterLast > end + 0.001)
  ) {
    ranges.push({ start: Math.max(start, last), end })
  }
  return ranges.filter((range) => range.end - range.start > 0.001)
}

function completedChecks(review: BeatReviewSummary): number {
  return checkOptions.filter((option) => review.checks[option.key]).length
}

function waveformPath(values: readonly number[]): string {
  if (values.length === 0) return ''
  const upper = values.map((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * 1000
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${(
      70 -
      value * 58
    ).toFixed(2)}`
  })
  const lower = [...values].reverse().map((value, reverseIndex) => {
    const index = values.length - reverseIndex - 1
    const x = (index / Math.max(1, values.length - 1)) * 1000
    return `L${x.toFixed(2)},${(70 + value * 58).toFixed(2)}`
  })
  return `${upper.join(' ')} ${lower.join(' ')} Z`
}

export default function BeatReviewWorkspace() {
  const requestedReviewer =
    new URLSearchParams(window.location.search).get('reviewer') ?? 'reviewer-a'
  const [reviewer, setReviewer] = useState(requestedReviewer)
  const [status, setStatus] = useState<BeatGoldStatus | null>(null)
  const [reviews, setReviews] = useState<BeatReviewSummary[]>([])
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null)
  const [document, setDocument] = useState<BeatReviewDocument | null>(null)
  const [draft, setDraft] = useState<BeatReviewDraft | null>(null)
  const [selectedBeat, setSelectedBeat] = useState(0)
  const [waveform, setWaveform] = useState<number[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [splitFilter, setSplitFilter] = useState<SplitFilter>('all')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    globalThis.document.documentElement.classList.add('beat-review-page')
    return () =>
      globalThis.document.documentElement.classList.remove('beat-review-page')
  }, [])

  const refreshReviews = useCallback(async () => {
    const next = await fetchBeatReviews(reviewer)
    setReviews(next)
    return next
  }, [reviewer])

  useEffect(() => {
    let active = true
    Promise.all([fetchBeatGoldStatus(), fetchBeatReviews(reviewer)])
      .then(([nextStatus, nextReviews]) => {
        if (!active) return
        if (!nextStatus.enabled) {
          throw new Error('节拍 Pilot 尚未启用')
        }
        setStatus(nextStatus)
        setReviews(nextReviews)
        setSelectedClipId(
          nextReviews.find((review) => review.status === 'draft')?.clipId ??
            nextReviews[0]?.clipId ??
            null,
        )
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '审核队列加载失败')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [reviewer])

  useEffect(() => {
    if (!selectedClipId) return
    let active = true
    fetchBeatReview(selectedClipId, reviewer)
      .then((next) => {
        if (!active) return
        setWaveform([])
        setDocument(next)
        setDraft(draftFromDocument(next))
        setSelectedBeat(
          Math.max(
            0,
            next.review.beats.findIndex(
              (marker) =>
                marker.at >= next.review.evaluation_start &&
                marker.at < next.review.evaluation_end,
            ),
          ),
        )
        setCurrentTime(0)
        setDirty(false)
        setNotice('')
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '审核片段加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [reviewer, selectedClipId])

  useEffect(() => {
    if (!document) return
    const controller = new AbortController()
    fetch(apiAssetUrl(document.audioUrl), {
      credentials: 'include',
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error('波形音频加载失败')
        return response.arrayBuffer()
      })
      .then(async (buffer) => {
        const context = new AudioContext()
        try {
          const audio = await context.decodeAudioData(buffer)
          const channel = audio.getChannelData(0)
          const bins = 520
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

  const markers = useMemo(() => draft?.beats ?? [], [draft?.beats])
  const selectedMarker = markers[selectedBeat]
  const readOnly = document?.review.status !== 'draft'
  const selectedInEvaluation =
    Boolean(document && selectedMarker) &&
    selectedMarker.at >= document!.review.evaluation_start &&
    selectedMarker.at < document!.review.evaluation_end
  const filteredReviews = useMemo(() => {
    const query = search.trim().toLowerCase()
    return reviews.filter(
      (review) =>
        (splitFilter === 'all' || review.split === splitFilter) &&
        (statusFilter === 'all' || review.status === statusFilter) &&
        (!query ||
          review.clipId.toLowerCase().includes(query) ||
          review.recordingId.toLowerCase().includes(query)),
    )
  }, [reviews, search, splitFilter, statusFilter])
  const selectedQueueIndex = filteredReviews.findIndex(
    (review) => review.clipId === selectedClipId,
  )
  const submittedCount = reviews.filter(
    (review) => review.status !== 'draft',
  ).length
  const checksDone = draft
    ? checkOptions.filter((option) => draft.checks[option.key]).length
    : 0
  const partialRanges = useMemo(
    () =>
      document
        ? partialMeasureRanges(
            markers,
            document.review.evaluation_start,
            document.review.evaluation_end,
          )
        : [],
    [document, markers],
  )

  const updateDraft = useCallback(
    (update: (current: BeatReviewDraft) => BeatReviewDraft) => {
      setDraft((current) => (current ? update(current) : current))
      setDirty(true)
      setNotice('')
    },
    [],
  )

  const selectClip = useCallback(
    (clipId: string) => {
      if (
        dirty &&
        !window.confirm('当前片段尚未保存，确认切换吗？')
      ) {
        return
      }
      audioRef.current?.pause()
      setError('')
      setWaveform([])
      setSelectedClipId(clipId)
    },
    [dirty],
  )

  const moveQueue = (offset: number) => {
    const next = filteredReviews[selectedQueueIndex + offset]
    if (next) selectClip(next.clipId)
  }

  const seek = (at: number) => {
    const audio = audioRef.current
    if (!audio || !document) return
    const next = Math.max(0, Math.min(document.review.audio_duration, at))
    audio.currentTime = next
    setCurrentTime(next)
  }

  const togglePlayback = async () => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      await audio.play()
    } else {
      audio.pause()
    }
  }

  const updateBeatTime = (index: number, value: number) => {
    if (!draft || !document || readOnly || !markers[index]) return
    if (
      markers[index].at < document.review.evaluation_start ||
      markers[index].at >= document.review.evaluation_end
    ) {
      return
    }
    const lower = Math.max(
      document.review.evaluation_start,
      index > 0 ? draft.beats[index - 1].at + 0.03 : 0,
    )
    const upper =
      index + 1 < draft.beats.length
        ? Math.min(
            document.review.evaluation_end - 0.001,
            draft.beats[index + 1].at - 0.03,
          )
        : document.review.evaluation_end - 0.001
    const at = Math.max(lower, Math.min(upper, value))
    updateDraft((current) => {
      const beats = [...current.beats]
      beats[index] = {
        ...beats[index],
        at: Math.round(at * 1_000_000) / 1_000_000,
      }
      return { ...current, beats }
    })
    seek(at)
    return at
  }

  const updateSelectedBeatTime = (value: number) => {
    updateBeatTime(selectedBeat, value)
  }

  const commitBeatTimeInput = (input: HTMLInputElement) => {
    const value = Number(input.value.trim())
    if (input.value.trim() === '' || !Number.isFinite(value)) {
      input.value = selectedMarker?.at.toFixed(3) ?? ''
      return
    }
    const committed = updateBeatTime(selectedBeat, value)
    input.value = (committed ?? selectedMarker?.at ?? 0).toFixed(3)
  }

  const addBeat = () => {
    if (!draft || !document || readOnly) return
    if (
      currentTime < document.review.evaluation_start ||
      currentTime >= document.review.evaluation_end
    ) {
      setError('只能在亮色评测窗内添加拍点')
      return
    }
    const at = Math.max(
      document.review.evaluation_start,
      Math.min(document.review.evaluation_end - 0.001, currentTime),
    )
    if (draft.beats.some((beat) => Math.abs(beat.at - at) < 0.03)) {
      setError('新拍点与现有拍点距离不足 30ms')
      return
    }
    const anchor = draft.beats[0] ?? {
      at,
      beat: 1,
      measure: 1,
      downbeat: true,
    }
    const beats = [
      ...draft.beats,
      {
        at: Math.round(at * 1_000_000) / 1_000_000,
        beat: anchor.beat,
        measure: anchor.measure,
        downbeat: anchor.downbeat,
      },
    ].sort((left, right) => left.at - right.at)
    const index = beats.findIndex((beat) => Math.abs(beat.at - at) < 0.000001)
    const phased = rephaseMarkers(
      beats,
      draft.time_signature[0],
      0,
      beats[0].beat,
    )
    updateDraft((current) => ({ ...current, beats: phased }))
    setSelectedBeat(index)
  }

  const deleteBeat = () => {
    if (!draft || readOnly || !selectedMarker || !selectedInEvaluation) return
    const beats = draft.beats.filter((_, index) => index !== selectedBeat)
    const phased =
      beats.length > 0
        ? rephaseMarkers(
            beats,
            draft.time_signature[0],
            0,
            beats[0].beat,
          )
        : []
    updateDraft((current) => ({ ...current, beats: phased }))
    setSelectedBeat(Math.max(0, Math.min(selectedBeat, beats.length - 1)))
  }

  const updateSelectedBeatPhase = (beat: number) => {
    if (!draft || readOnly || !selectedMarker) return
    updateDraft((current) => ({
      ...current,
      beats: rephaseMarkers(
        current.beats,
        current.time_signature[0],
        selectedBeat,
        beat,
      ),
    }))
  }

  const persist = async (submit: boolean) => {
    if (!draft || !selectedClipId || saving) return
    if (
      submit &&
      !window.confirm('提交后该标注将锁定，确认提交独立审核吗？')
    ) {
      return
    }
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const next = submit
        ? await submitBeatReview(selectedClipId, reviewer, draft)
        : await saveBeatReview(selectedClipId, reviewer, draft)
      setDocument(next)
      setDraft(draftFromDocument(next))
      setDirty(false)
      setNotice(submit ? '独立审核已提交' : '草稿已保存')
      await refreshReviews()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '审核保存失败')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="beat-review-loading">
        <LoaderCircle aria-hidden="true" />
        <span>正在加载节拍审核队列</span>
      </div>
    )
  }

  if (error && !document) {
    return (
      <div className="beat-review-loading is-error">
        <CircleAlert aria-hidden="true" />
        <strong>{error}</strong>
        <a href="/admin">返回仪表盘</a>
      </div>
    )
  }

  return (
    <div className="beat-review-shell">
      <header className="beat-review-header">
        <div className="beat-review-corpus">
          <strong>Pilot v2</strong>
          <span>{status?.manifest?.corpusVersion ?? '未准备'}</span>
        </div>
        <div className="beat-review-progress">
          <span>
            {submittedCount} / {reviews.length}
          </span>
          <i>
            <b
              style={{
                width: `${reviews.length > 0
                  ? (submittedCount / reviews.length) * 100
                  : 0}%`,
              }}
            />
          </i>
        </div>
        <label>
          <span>审核者</span>
          <select
            value={reviewer}
            onChange={(event) => {
              const next = event.target.value
              setLoading(true)
              setError('')
              setDocument(null)
              setDraft(null)
              setReviewer(next)
              const url = new URL(window.location.href)
              url.searchParams.set('reviewer', next)
              window.history.replaceState(null, '', url)
            }}
          >
            {(status?.reviewers ?? [reviewer]).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </header>

      <div className="beat-review-body">
        <aside className="beat-review-queue">
          <div className="beat-review-queue-tools">
            <label>
              <ListFilter aria-hidden="true" />
              <input
                type="search"
                placeholder="筛选片段"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <div aria-label="数据分区">
              {(['all', 'train', 'validation', 'test'] as const).map(
                (value) => (
                  <button
                    className={splitFilter === value ? 'is-active' : ''}
                    type="button"
                    key={value}
                    onClick={() => setSplitFilter(value)}
                  >
                    {value === 'all' ? '全部' : value}
                  </button>
                ),
              )}
            </div>
            <select
              aria-label="审核状态"
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(event.target.value as StatusFilter)}
            >
              <option value="all">全部状态</option>
              <option value="draft">待审核</option>
              <option value="submitted">已提交</option>
              <option value="adjudicated">已裁决</option>
            </select>
          </div>
          <div className="beat-review-list">
            {filteredReviews.map((review) => (
              <button
                className={
                  review.clipId === selectedClipId ? 'is-current' : ''
                }
                type="button"
                key={review.clipId}
                onClick={() => selectClip(review.clipId)}
              >
                <span>
                  <strong>{review.recordingId}</strong>
                  <small>
                    {formatTime(review.sourceStart)} · {review.beatCount} 拍
                  </small>
                </span>
                <i className={`is-${review.status}`}>
                  {review.status === 'draft'
                    ? `${completedChecks(review)}/4`
                    : review.status === 'submitted'
                      ? '已提交'
                      : '金标准'}
                </i>
              </button>
            ))}
          </div>
        </aside>

        <section className="beat-review-workspace">
          {document && draft ? (
            <>
              <header className="beat-review-title">
                <button
                  type="button"
                  aria-label="上一个片段"
                  title="上一个片段"
                  disabled={selectedQueueIndex <= 0}
                  onClick={() => moveQueue(-1)}
                >
                  <ChevronLeft aria-hidden="true" />
                </button>
                <div>
                  <span>
                    {reviews.find(
                      (review) => review.clipId === selectedClipId,
                    )?.split ?? ''}
                  </span>
                  <h1>{document.review.clip_id}</h1>
                  <small>{document.review.tags.join(' · ')}</small>
                </div>
                <button
                  type="button"
                  aria-label="下一个片段"
                  title="下一个片段"
                  disabled={
                    selectedQueueIndex < 0 ||
                    selectedQueueIndex >= filteredReviews.length - 1
                  }
                  onClick={() => moveQueue(1)}
                >
                  <ChevronRight aria-hidden="true" />
                </button>
              </header>

              <audio
                ref={audioRef}
                src={apiAssetUrl(document.audioUrl)}
                crossOrigin="use-credentials"
                preload="auto"
                onEnded={() => setPlaying(false)}
                onLoadedMetadata={(event) => {
                  event.currentTarget.playbackRate = playbackRate
                }}
                onPause={() => setPlaying(false)}
                onPlay={() => setPlaying(true)}
                onTimeUpdate={(event) =>
                  setCurrentTime(event.currentTarget.currentTime)}
              />

              <div className="beat-review-transport">
                <button
                  type="button"
                  aria-label="回到片段开始"
                  title="回到片段开始"
                  onClick={() => seek(0)}
                >
                  <SkipBack aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="跳到评测窗起点"
                  title="跳到评测窗起点"
                  onClick={() => seek(document.review.evaluation_start)}
                >
                  <Target aria-hidden="true" />
                </button>
                <button
                  className="is-primary"
                  type="button"
                  aria-label={playing ? '暂停' : '播放'}
                  title={playing ? '暂停' : '播放'}
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
                    const value = Number(event.target.value)
                    setPlaybackRate(value)
                    if (audioRef.current) {
                      audioRef.current.playbackRate = value
                    }
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
                className="beat-review-timeline"
                onClick={(event) => {
                  const bounds = event.currentTarget.getBoundingClientRect()
                  seek(
                    ((event.clientX - bounds.left) / bounds.width) *
                      document.review.audio_duration,
                  )
                }}
              >
                <div
                  className="beat-review-context is-before"
                  title="试听上下文，仅用于判断拍位和边界"
                  style={{
                    width: `${
                      (document.review.evaluation_start /
                        document.review.audio_duration) *
                      100
                    }%`,
                  }}
                >
                  <span>不计分</span>
                </div>
                <div
                  className="beat-review-context is-after"
                  title="试听上下文，仅用于判断拍位和边界"
                  style={{
                    left: `${
                      (document.review.evaluation_end /
                        document.review.audio_duration) *
                      100
                    }%`,
                    width: `${
                      ((document.review.audio_duration -
                        document.review.evaluation_end) /
                        document.review.audio_duration) *
                      100
                    }%`,
                  }}
                >
                  <span>不计分</span>
                </div>
                {partialRanges.map((range) => (
                  <div
                    className="beat-review-partial-measure"
                    key={`${range.start}-${range.end}`}
                    style={{
                      left: `${
                        (range.start / document.review.audio_duration) * 100
                      }%`,
                      width: `${
                        ((range.end - range.start) /
                          document.review.audio_duration) *
                        100
                      }%`,
                    }}
                  />
                ))}
                <svg
                  aria-label="音频波形"
                  preserveAspectRatio="none"
                  viewBox="0 0 1000 140"
                >
                  <path d={waveformPath(waveform)} />
                </svg>
                <div
                  className="beat-review-playhead"
                  style={{
                    left: `${
                      (currentTime / document.review.audio_duration) * 100
                    }%`,
                  }}
                />
                {markers.map((marker, index) => (
                  <button
                    className={[
                      marker.downbeat ? 'is-downbeat' : '',
                      selectedBeat === index ? 'is-selected' : '',
                      marker.at < document.review.evaluation_start ||
                      marker.at >= document.review.evaluation_end
                        ? 'is-context'
                        : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                    type="button"
                    aria-label={`第 ${marker.measure} 小节第 ${marker.beat} 拍，${marker.at.toFixed(3)} 秒`}
                    title={
                      marker.at < document.review.evaluation_start ||
                      marker.at >= document.review.evaluation_end
                        ? `上下文参考拍点，不计分；第 ${marker.measure} 小节第 ${marker.beat} 拍`
                        : `第 ${marker.measure} 小节第 ${marker.beat} 拍`
                    }
                    key={`${document.review.clip_id}-${index}`}
                    style={{
                      left: `${
                        (marker.at / document.review.audio_duration) * 100
                      }%`,
                    }}
                    onClick={(event) => {
                      event.stopPropagation()
                      setSelectedBeat(index)
                      seek(marker.at)
                    }}
                    onPointerDown={(event) => {
                      if (
                        readOnly ||
                        event.button !== 0 ||
                        marker.at < document.review.evaluation_start ||
                        marker.at >= document.review.evaluation_end
                      ) {
                        return
                      }
                      event.preventDefault()
                      event.stopPropagation()
                      setSelectedBeat(index)
                      event.currentTarget.setPointerCapture(event.pointerId)
                    }}
                    onPointerMove={(event) => {
                      if (
                        readOnly ||
                        !event.currentTarget.hasPointerCapture(event.pointerId)
                      ) {
                        return
                      }
                      const timeline = event.currentTarget.parentElement
                      if (!timeline) return
                      const bounds = timeline.getBoundingClientRect()
                      updateBeatTime(
                        index,
                        ((event.clientX - bounds.left) / bounds.width) *
                          document.review.audio_duration,
                      )
                    }}
                  >
                    <i />
                    <span>{marker.beat}</span>
                  </button>
                ))}
              </div>

              <div className="beat-review-editor">
                <div className="beat-review-marker-summary">
                  <Clock3 aria-hidden="true" />
                  <span>
                    {selectedMarker
                      ? selectedInEvaluation
                        ? `第 ${selectedMarker.measure} 小节 · 第 ${selectedMarker.beat} 拍`
                        : `上下文参考 · 第 ${selectedMarker.measure} 小节 · 第 ${selectedMarker.beat} 拍`
                      : '未选择拍点'}
                  </span>
                  <strong>
                    {
                      markers.filter(
                        (marker) =>
                          marker.at >= document.review.evaluation_start &&
                          marker.at < document.review.evaluation_end,
                      ).length
                    }{' '}
                    评测拍点
                  </strong>
                </div>
                <div className="beat-review-marker-controls">
                  <button
                    type="button"
                    aria-label="向前移动 10 毫秒"
                    title="向前移动 10 毫秒"
                    disabled={readOnly || !selectedMarker || !selectedInEvaluation}
                    onClick={() =>
                      updateSelectedBeatTime((selectedMarker?.at ?? 0) - 0.01)}
                  >
                    <ChevronLeft aria-hidden="true" />
                    10ms
                  </button>
                  <label>
                    <span>拍点时间</span>
                    <input
                      type="number"
                      min={document.review.evaluation_start}
                      max={document.review.evaluation_end}
                      step={0.001}
                      disabled={
                        readOnly || !selectedMarker || !selectedInEvaluation
                      }
                      defaultValue={selectedMarker?.at.toFixed(3) ?? ''}
                      key={`${selectedClipId}-${selectedBeat}-${selectedMarker?.at ?? 'none'}`}
                      onBlur={(event) =>
                        commitBeatTimeInput(event.currentTarget)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          event.currentTarget.blur()
                        } else if (event.key === 'Escape') {
                          event.preventDefault()
                          event.currentTarget.value =
                            selectedMarker?.at.toFixed(3) ?? ''
                          event.currentTarget.blur()
                        }
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    aria-label="向后移动 10 毫秒"
                    title="向后移动 10 毫秒"
                    disabled={readOnly || !selectedMarker || !selectedInEvaluation}
                    onClick={() =>
                      updateSelectedBeatTime((selectedMarker?.at ?? 0) + 0.01)}
                  >
                    10ms
                    <ChevronRight aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    aria-label="在播放位置添加拍点"
                    title="在播放位置添加拍点"
                    disabled={
                      readOnly ||
                      currentTime < document.review.evaluation_start ||
                      currentTime >= document.review.evaluation_end
                    }
                    onClick={addBeat}
                  >
                    <Plus aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    aria-label="删除当前拍点"
                    title="删除当前拍点"
                    disabled={readOnly || !selectedMarker || !selectedInEvaluation}
                    onClick={deleteBeat}
                  >
                    <Trash2 aria-hidden="true" />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="beat-review-empty">未选择审核片段</div>
          )}
        </section>

        <aside className="beat-review-inspector">
          {document && draft ? (
            <>
              <section>
                <span>节拍结构</span>
                <div className="beat-review-meter-fields">
                  <label>
                    <span>每小节</span>
                    <input
                      type="number"
                      min={1}
                      max={16}
                      disabled={readOnly}
                      value={draft.time_signature[0]}
                      onChange={(event) => {
                        const numerator = Math.max(
                          1,
                          Math.min(16, Number(event.target.value)),
                        )
                        updateDraft((current) => {
                          const anchorIndex = Math.min(
                            selectedBeat,
                            Math.max(0, current.beats.length - 1),
                          )
                          const anchorBeat = Math.min(
                            numerator,
                            current.beats[anchorIndex]?.beat ?? 1,
                          )
                          return {
                            ...current,
                            time_signature: [
                              numerator,
                              current.time_signature[1],
                            ],
                            pickup_beats: Math.min(
                              current.pickup_beats,
                              Math.max(0, numerator - 1),
                            ),
                            beats: rephaseMarkers(
                              current.beats,
                              numerator,
                              anchorIndex,
                              anchorBeat,
                            ),
                          }
                        })
                      }}
                    />
                  </label>
                  <label>
                    <span>拍值</span>
                    <select
                      disabled={readOnly}
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
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="beat-review-phase-fields">
                  <label>
                    <span>当前拍位</span>
                    <select
                      disabled={readOnly || !selectedMarker}
                      value={selectedMarker?.beat ?? 1}
                      onChange={(event) =>
                        updateSelectedBeatPhase(Number(event.target.value))}
                    >
                      {Array.from(
                        { length: draft.time_signature[0] },
                        (_, index) => index + 1,
                      ).map((value) => (
                        <option key={value} value={value}>
                          第 {value} 拍
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    disabled={readOnly || !selectedMarker}
                    onClick={() => updateSelectedBeatPhase(1)}
                  >
                    <Target aria-hidden="true" />
                    设为强拍
                  </button>
                </div>
                <label className="beat-review-pickup">
                  <span>全曲弱起拍数</span>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, draft.time_signature[0] - 1)}
                    disabled={readOnly}
                    value={draft.pickup_beats}
                    onChange={(event) =>
                      updateDraft((current) => ({
                        ...current,
                        pickup_beats: Number(event.target.value),
                      }))}
                  />
                </label>
              </section>

              <section>
                <span>审核检查</span>
                <div className="beat-review-checks">
                  {checkOptions.map((option) => (
                    <label key={option.key}>
                      <input
                        type="checkbox"
                        checked={draft.checks[option.key]}
                        disabled={readOnly}
                        onChange={(event) =>
                          updateDraft((current) => ({
                            ...current,
                            checks: {
                              ...current.checks,
                              [option.key]: event.target.checked,
                            },
                          }))}
                      />
                      <span>{option.label}</span>
                      {draft.checks[option.key] ? (
                        <Check aria-hidden="true" />
                      ) : null}
                    </label>
                  ))}
                </div>
                <strong className="beat-review-check-count">
                  {checksDone} / 4
                </strong>
              </section>

              <section className="beat-review-status">
                <span>审核状态</span>
                <strong className={`is-${document.review.status}`}>
                  {document.review.status === 'draft'
                    ? '待审核'
                    : document.review.status === 'submitted'
                      ? '已提交'
                      : '已裁决'}
                </strong>
                <small>
                  {document.review.annotator_alias} ·{' '}
                  {document.review.corpus_version}
                </small>
              </section>

              {error ? (
                <p className="beat-review-message is-error" role="alert">
                  <CircleAlert aria-hidden="true" />
                  {error}
                </p>
              ) : null}
              {notice ? (
                <p className="beat-review-message is-success">
                  <Check aria-hidden="true" />
                  {notice}
                </p>
              ) : null}

              <div className="beat-review-actions">
                <button
                  type="button"
                  disabled={readOnly || saving || !dirty}
                  onClick={() => void persist(false)}
                >
                  {saving ? (
                    <LoaderCircle className="spin" aria-hidden="true" />
                  ) : (
                    <Save aria-hidden="true" />
                  )}
                  保存草稿
                </button>
                <button
                  className="is-primary"
                  type="button"
                  disabled={readOnly || saving || checksDone < 4}
                  onClick={() => void persist(true)}
                >
                  <Check aria-hidden="true" />
                  提交独立审核
                </button>
              </div>
            </>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
