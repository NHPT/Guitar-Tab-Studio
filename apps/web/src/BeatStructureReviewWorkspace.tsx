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
  { key: 'pickup', label: '弱起拍数' },
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
    checks: { ...document.review.checks },
  }
}

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60)
  const seconds = value - minutes * 60
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`
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
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)
  const previewEndRef = useRef<number | null>(null)

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
        setDocument(next)
        setDraft(nextDraft)
        setWaveform([])
        setSelectedBeatIndex(nearestBeatIndex(next.candidateBeats, anchor))
        setCurrentTime(anchor)
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

  const selectedBeatAt = document?.candidateBeats[selectedBeatIndex] ?? 0
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
    return document.candidateBeats
      .map((at, index) => ({ at, index }))
      .filter((beat) => beat.at >= windowStart && beat.at <= windowEnd)
  }, [document, windowEnd, windowStart])

  const beatLabel = useCallback(
    (index: number): number | null => {
      if (!document || !draft || draft.phase_status === 'uncertain') {
        return null
      }
      const sections = [
        {
          at: draft.first_downbeat_at ?? document.candidateBeats[0],
          timeSignature: draft.time_signature,
        },
        ...draft.meter_changes.map((change) => ({
          at: change.at,
          timeSignature: change.time_signature,
        })),
      ].sort((left, right) => left.at - right.at)
      const at = document.candidateBeats[index]
      const section =
        [...sections].reverse().find((candidate) => candidate.at <= at) ??
        sections[0]
      const anchorIndex = nearestBeatIndex(
        document.candidateBeats,
        section.at,
      )
      return (
        positiveModulo(index - anchorIndex, section.timeSignature[0]) + 1
      )
    },
    [document, draft],
  )

  const representativePoints = useMemo(() => {
    if (!document || !draft || document.candidateBeats.length === 0) {
      return []
    }
    const candidates =
      draft.phase_status === 'confirmed'
        ? document.candidateBeats
            .map((at, index) => ({ at, index }))
            .filter(({ index }) => beatLabel(index) === 1)
        : document.candidateBeats.map((at, index) => ({ at, index }))
    const pool =
      candidates.length > 0
        ? candidates
        : document.candidateBeats.map((at, index) => ({ at, index }))
    const targets = [
      {
        label: '开头',
        at: draft.first_downbeat_at ?? document.candidateBeats[0],
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
  }, [beatLabel, document, draft])

  const updateDraft = useCallback(
    (update: (current: BeatStructureDraft) => BeatStructureDraft) => {
      setDraft((current) => (current ? update(current) : current))
      setDirty(true)
      setNotice('')
    },
    [],
  )

  const seek = (at: number) => {
    if (!document) return
    previewEndRef.current = null
    const bounded = Math.max(0, Math.min(document.review.audio_duration, at))
    if (audioRef.current) audioRef.current.currentTime = bounded
    setCurrentTime(bounded)
  }

  const togglePlayback = async () => {
    const audio = audioRef.current
    if (!audio) return
    previewEndRef.current = null
    if (audio.paused) await audio.play()
    else audio.pause()
  }

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
    if (dirty && !window.confirm('当前全曲结构尚未保存，确认切换吗？')) {
      return
    }
    audioRef.current?.pause()
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

  const persist = async (
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
      const next = submit
        ? await submitBeatStructureReview(
            selectedRecordingId,
            reviewer,
            nextDraft,
          )
        : await saveBeatStructureReview(
            selectedRecordingId,
            reviewer,
            nextDraft,
          )
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
    <div className="beat-structure-shell" data-review-dirty={dirty}>
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
                  候选 {document.candidateBpm.toFixed(2)} BPM ·{' '}
                  {document.candidateBeats.length} 拍
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
              onEnded={() => setPlaying(false)}
              onLoadedMetadata={(event) => {
                event.currentTarget.playbackRate = playbackRate
                event.currentTarget.currentTime = currentTime
              }}
              onPause={() => setPlaying(false)}
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
                <path d={waveformPath(waveform)} />
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
              {visibleBeats.map((beat) => {
                const label = beatLabel(beat.index)
                return (
                  <button
                    className={[
                      label === 1 ? 'is-downbeat' : '',
                      beat.index === selectedBeatIndex ? 'is-selected' : '',
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
                    aria-label={`${formatTime(beat.at)}，${label === null ? '拍位待确认' : `第 ${label} 拍`}`}
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
                  <dt>速度</dt>
                  <dd>{document.candidateBpm.toFixed(1)} BPM</dd>
                </div>
                <div>
                  <dt>弱起</dt>
                  <dd>
                    {draft.pickup_beats === 0
                      ? '无'
                      : `${draft.pickup_beats} 拍`}
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
            </section>

            {correctionOpen && !readOnly && (
              <>
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
                            numerator - 1,
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
                    <span>弱起拍数</span>
                    <input
                      aria-label="弱起拍数"
                      type="number"
                      min={0}
                      max={draft.time_signature[0] - 1}
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
