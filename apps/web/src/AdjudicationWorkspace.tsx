import {
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  LoaderCircle,
  Pause,
  Play,
  Save,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  adjudicateBeat,
  apiAssetUrl,
  fetchBeatAdjudicationQueue,
  fetchBeatComparison,
} from './api'
import type {
  BeatAdjudicationSummary,
  BeatReviewComparison,
  BeatReviewDraft,
  BeatReviewMarker,
} from './types'
import './AdjudicationWorkspace.css'

function draftFrom(
  comparison: BeatReviewComparison,
  side: 'left' | 'right',
): BeatReviewDraft {
  const review = comparison[side].review
  return {
    beats: review.beats.map((marker) => ({ ...marker })),
    time_signature: [...review.time_signature],
    pickup_beats: review.pickup_beats,
    checks: {
      beat_times: true,
      downbeats: true,
      meter: true,
      completeness: true,
    },
  }
}

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60)
  return `${minutes}:${(value - minutes * 60).toFixed(3).padStart(6, '0')}`
}

function rephase(
  markers: readonly BeatReviewMarker[],
  numerator: number,
  anchorIndex: number,
  anchorBeat: number,
): BeatReviewMarker[] {
  if (markers.length === 0) return []
  const next = markers.map((marker) => ({ ...marker }))
  const anchor = Math.max(0, Math.min(anchorIndex, next.length - 1))
  next[anchor].beat = anchorBeat
  next[anchor].downbeat = anchorBeat === 1
  for (let index = anchor - 1; index >= 0; index -= 1) {
    const after = next[index + 1]
    const beat = after.beat === 1 ? numerator : after.beat - 1
    next[index] = {
      ...next[index],
      beat,
      measure: Math.max(
        0,
        after.measure - (after.beat === 1 ? 1 : 0),
      ),
      downbeat: beat === 1,
    }
  }
  for (let index = anchor + 1; index < next.length; index += 1) {
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

export default function AdjudicationWorkspace() {
  const [queue, setQueue] = useState<BeatAdjudicationSummary[]>([])
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null)
  const [comparison, setComparison] =
    useState<BeatReviewComparison | null>(null)
  const [draft, setDraft] = useState<BeatReviewDraft | null>(null)
  const [baseline, setBaseline] = useState<'left' | 'right'>('left')
  const [selectedMarker, setSelectedMarker] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)

  const refreshQueue = async () => {
    const next = await fetchBeatAdjudicationQueue()
    setQueue(next)
    return next
  }

  useEffect(() => {
    let active = true
    fetchBeatAdjudicationQueue()
      .then((items) => {
        if (!active) return
        setQueue(items)
        setSelectedClipId(
          items.find((item) => item.ready && !item.adjudicated)?.clipId ??
            items[0]?.clipId ??
            null,
        )
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '裁决队列加载失败')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const summary = queue.find((item) => item.clipId === selectedClipId)
    if (!selectedClipId || !summary?.ready) {
      return
    }
    let active = true
    fetchBeatComparison(selectedClipId)
      .then((value) => {
        if (!active) return
        setComparison(value)
        setBaseline('left')
        setDraft(draftFrom(value, 'left'))
        const first = value.left.review.beats.findIndex(
          (marker) => marker.at >= value.left.review.evaluation_start,
        )
        setSelectedMarker(Math.max(0, first))
        setError('')
        setNotice('')
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '差异数据加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [queue, selectedClipId])

  const summary = queue.find((item) => item.clipId === selectedClipId)
  const review = comparison?.left.review
  const marker = draft?.beats[selectedMarker]
  const disagreements = useMemo(
    () =>
      comparison?.differences.filter(
        (difference) =>
          difference.left === null ||
          difference.right === null ||
          !difference.phaseAgreement ||
          Math.abs(difference.deltaMs ?? 0) > 20,
      ) ?? [],
    [comparison],
  )

  const seek = (at: number) => {
    if (!audioRef.current || !review) return
    const next = Math.max(0, Math.min(review.audio_duration, at))
    audioRef.current.currentTime = next
    setCurrentTime(next)
  }

  const chooseBaseline = (side: 'left' | 'right') => {
    if (!comparison) return
    setBaseline(side)
    const next = draftFrom(comparison, side)
    setDraft(next)
    const first = next.beats.findIndex(
      (candidate) => candidate.at >= comparison.left.review.evaluation_start,
    )
    setSelectedMarker(Math.max(0, first))
  }

  const updateMarker = (update: Partial<BeatReviewMarker>) => {
    if (!draft || !marker) return
    setDraft((current) => {
      if (!current) return current
      const beats = current.beats.map((candidate, index) =>
        index === selectedMarker ? { ...candidate, ...update } : candidate,
      )
      return { ...current, beats }
    })
  }

  const submit = async () => {
    if (!draft || !selectedClipId || saving) return
    if (!window.confirm('确认以当前时间轴生成裁决金标准吗？')) return
    setSaving(true)
    setError('')
    try {
      await adjudicateBeat(selectedClipId, draft)
      setNotice('裁决金标准已生成')
      await refreshQueue()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '裁决提交失败')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="adjudication-loading">
        <LoaderCircle className="spin" />
        正在加载裁决队列
      </div>
    )
  }

  return (
    <div className="adjudication-page">
      <aside className="adjudication-queue">
        <header>
          <span>双人独立审核</span>
          <strong>
            {queue.filter((item) => item.ready).length} / {queue.length}
          </strong>
        </header>
        <div>
          {queue.map((item) => (
            <button
              className={item.clipId === selectedClipId ? 'is-current' : ''}
              type="button"
              key={item.clipId}
              onClick={() => {
                setComparison(null)
                setDraft(null)
                setSelectedClipId(item.clipId)
              }}
            >
              <span>
                <strong>{item.recordingId}</strong>
                <small>{item.clipId}</small>
              </span>
              <i
                className={
                  item.adjudicated
                    ? 'is-gold'
                    : item.ready
                      ? 'is-ready'
                      : ''
                }
              >
                {item.adjudicated
                  ? '金标准'
                  : item.ready
                    ? `${item.disagreements ?? 0} 差异`
                    : `${item.leftStatus === 'submitted' ? 'A✓' : 'A·'} ${
                        item.rightStatus === 'submitted' ? 'B✓' : 'B·'
                      }`}
              </i>
            </button>
          ))}
        </div>
      </aside>

      <section className="adjudication-main">
        {comparison && draft && review ? (
          <>
            <header className="adjudication-title">
              <div>
                <span>{summary?.split}</span>
                <h1>{comparison.clipId}</h1>
              </div>
              <div className="adjudication-baseline">
                <button
                  className={baseline === 'left' ? 'is-active' : ''}
                  type="button"
                  onClick={() => chooseBaseline('left')}
                >
                  采用 A
                </button>
                <button
                  className={baseline === 'right' ? 'is-active' : ''}
                  type="button"
                  onClick={() => chooseBaseline('right')}
                >
                  采用 B
                </button>
              </div>
            </header>
            <audio
              ref={audioRef}
              src={apiAssetUrl(comparison.left.audioUrl)}
              crossOrigin="use-credentials"
              onPause={() => setPlaying(false)}
              onPlay={() => setPlaying(true)}
              onTimeUpdate={(event) =>
                setCurrentTime(event.currentTarget.currentTime)}
            />
            <div className="adjudication-transport">
              <button
                type="button"
                aria-label={playing ? '暂停' : '播放'}
                onClick={() => {
                  if (!audioRef.current) return
                  if (audioRef.current.paused) void audioRef.current.play()
                  else audioRef.current.pause()
                }}
              >
                {playing ? <Pause /> : <Play />}
              </button>
              <strong>{formatTime(currentTime)}</strong>
              <span>/ {formatTime(review.audio_duration)}</span>
            </div>
            <div
              className="adjudication-timeline"
              onClick={(event) => {
                const bounds = event.currentTarget.getBoundingClientRect()
                seek(
                  ((event.clientX - bounds.left) / bounds.width) *
                    review.audio_duration,
                )
              }}
            >
              <span className="is-a">A</span>
              <span className="is-b">B</span>
              <div
                className="adjudication-playhead"
                style={{
                  left: `${(currentTime / review.audio_duration) * 100}%`,
                }}
              />
              {comparison.left.review.beats.map((candidate, index) => (
                <button
                  className={`is-a${candidate.downbeat ? ' is-downbeat' : ''}`}
                  type="button"
                  key={`a-${index}-${candidate.at}`}
                  style={{
                    left: `${(candidate.at / review.audio_duration) * 100}%`,
                  }}
                  onClick={(event) => {
                    event.stopPropagation()
                    seek(candidate.at)
                  }}
                  aria-label={`A ${candidate.at.toFixed(3)} 秒`}
                />
              ))}
              {comparison.right.review.beats.map((candidate, index) => (
                <button
                  className={`is-b${candidate.downbeat ? ' is-downbeat' : ''}`}
                  type="button"
                  key={`b-${index}-${candidate.at}`}
                  style={{
                    left: `${(candidate.at / review.audio_duration) * 100}%`,
                  }}
                  onClick={(event) => {
                    event.stopPropagation()
                    seek(candidate.at)
                  }}
                  aria-label={`B ${candidate.at.toFixed(3)} 秒`}
                />
              ))}
            </div>
            <div className="adjudication-marker-editor">
              <button
                type="button"
                aria-label="上一个拍点"
                disabled={selectedMarker <= 0}
                onClick={() => setSelectedMarker((value) => value - 1)}
              >
                <ChevronLeft />
              </button>
              <label>
                <span>裁决拍点时间</span>
                <input
                  type="number"
                  step={0.001}
                  value={marker?.at.toFixed(3) ?? ''}
                  onChange={(event) =>
                    updateMarker({ at: Number(event.target.value) })}
                />
              </label>
              <label>
                <span>拍位</span>
                <select
                  value={marker?.beat ?? 1}
                  onChange={(event) => {
                    if (!draft) return
                    const beat = Number(event.target.value)
                    setDraft({
                      ...draft,
                      beats: rephase(
                        draft.beats,
                        draft.time_signature[0],
                        selectedMarker,
                        beat,
                      ),
                    })
                  }}
                >
                  {Array.from(
                    { length: draft.time_signature[0] },
                    (_, index) => index + 1,
                  ).map((value) => (
                    <option value={value} key={value}>
                      第 {value} 拍
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                aria-label="下一个拍点"
                disabled={selectedMarker >= draft.beats.length - 1}
                onClick={() => setSelectedMarker((value) => value + 1)}
              >
                <ChevronRight />
              </button>
            </div>
            <div className="adjudication-differences">
              <header>
                <span>需关注差异</span>
                <strong>{disagreements.length}</strong>
              </header>
              <div>
                {disagreements.slice(0, 80).map((difference, index) => (
                  <button
                    type="button"
                    key={`${difference.left?.at}-${difference.right?.at}-${index}`}
                    onClick={() =>
                      seek(difference.left?.at ?? difference.right?.at ?? 0)}
                  >
                    <code>
                      {difference.left?.at.toFixed(3) ?? '—'} /{' '}
                      {difference.right?.at.toFixed(3) ?? '—'}
                    </code>
                    <span>
                      {difference.deltaMs === null
                        ? '缺失拍点'
                        : `${difference.deltaMs > 0 ? '+' : ''}${difference.deltaMs.toFixed(1)}ms`}
                    </span>
                    <i>
                      {difference.phaseAgreement ? '拍位一致' : '拍位不同'}
                    </i>
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="adjudication-empty">
            {summary?.adjudicated
              ? '该片段已完成裁决'
              : '等待 A、B 两名独立复核员提交'}
          </div>
        )}
      </section>

      <aside className="adjudication-inspector">
        <section>
          <span>一致性</span>
          <dl>
            <div>
              <dt>Beat F1</dt>
              <dd>
                {comparison ? comparison.beatMetrics.f1.toFixed(3) : '—'}
              </dd>
            </div>
            <div>
              <dt>中位误差</dt>
              <dd>
                {comparison?.beatMetrics.median_error_ms?.toFixed(1) ?? '—'}ms
              </dd>
            </div>
            <div>
              <dt>拍号</dt>
              <dd>{comparison?.meterAgreement ? '一致' : '不同'}</dd>
            </div>
            <div>
              <dt>独立账号</dt>
              <dd>{comparison?.independentAccounts ? '已验证' : '未通过'}</dd>
            </div>
          </dl>
        </section>
        {error ? (
          <p className="adjudication-message is-error" role="alert">
            <CircleAlert />
            {error}
          </p>
        ) : null}
        {notice ? (
          <p className="adjudication-message is-success">
            <Check />
            {notice}
          </p>
        ) : null}
        <button
          className="adjudication-submit"
          type="button"
          disabled={!comparison || !draft || saving || summary?.adjudicated}
          onClick={() => void submit()}
        >
          {saving ? <LoaderCircle className="spin" /> : <Save />}
          生成裁决金标准
        </button>
      </aside>
    </div>
  )
}
