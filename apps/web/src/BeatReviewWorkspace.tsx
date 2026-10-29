import { Clock3, ListChecks } from 'lucide-react'
import { useEffect, useState } from 'react'
import { fetchBeatStructureReviews } from './api'
import BeatClipReviewWorkspace from './BeatClipReviewWorkspace'
import BeatStructureReviewWorkspace from './BeatStructureReviewWorkspace'
import type { BeatStructureSummary } from './types'
import './BeatReviewWorkspace.css'

type ReviewStage = 'structure' | 'clips'

export default function BeatReviewWorkspace() {
  const parameters = new URLSearchParams(window.location.search)
  const [requestedReviewer, setRequestedReviewer] = useState(
    parameters.get('reviewer') ?? 'reviewer-a',
  )
  const [requestedStage] = useState(parameters.get('stage'))
  const [stage, setStage] = useState<ReviewStage>('structure')
  const [structures, setStructures] = useState<BeatStructureSummary[]>([])
  const submitted = structures.filter(
    (review) => review.status === 'submitted',
  ).length

  useEffect(() => {
    let active = true
    fetchBeatStructureReviews(requestedReviewer)
      .then((reviews) => {
        if (!active) return
        const confirmed = reviews.filter(
          (review) => review.status === 'submitted',
        ).length
        setStructures(reviews)
        if (
          requestedStage === 'clips' &&
          confirmed > 0
        ) {
          setStage('clips')
        }
      })
      .catch(() => {
        if (active) setStructures([])
      })
    return () => {
      active = false
    }
  }, [requestedReviewer, requestedStage])

  const selectStage = (next: ReviewStage) => {
    if (
      globalThis.document.querySelector('[data-review-dirty="true"]') &&
      !window.confirm('当前草稿尚未保存，确认切换复核阶段吗？')
    ) {
      return
    }
    setStage(next)
    const url = new URL(window.location.href)
    url.searchParams.set('stage', next)
    window.history.replaceState(null, '', url)
  }

  return (
    <div className="beat-review-stage-shell">
      <nav className="beat-review-stage-nav" aria-label="节拍复核阶段">
        <button
          className={stage === 'structure' ? 'is-active' : ''}
          type="button"
          onClick={() => selectStage('structure')}
        >
          <ListChecks aria-hidden="true" />
          <span>全曲结构</span>
          <i>
            {submitted}/{structures.length || 10}
          </i>
        </button>
        <button
          className={stage === 'clips' ? 'is-active' : ''}
          type="button"
          disabled={submitted === 0}
          title={submitted > 0 ? '复核已确认歌曲的片段' : '先确认一首歌曲的全曲结构'}
          onClick={() => selectStage('clips')}
        >
          <Clock3 aria-hidden="true" />
          <span>片段时间</span>
          <i>{submitted > 0 ? `${submitted} 首已开放` : '待结构确认'}</i>
        </button>
      </nav>
      {stage === 'structure' ? (
        <BeatStructureReviewWorkspace
          reviewer={requestedReviewer}
          onReviewsChange={setStructures}
        />
      ) : (
        <BeatClipReviewWorkspace
          reviewer={requestedReviewer}
          structures={structures}
          onReviewerChange={(next) => {
            setStructures([])
            setRequestedReviewer(next)
            setStage('structure')
            const url = new URL(window.location.href)
            url.searchParams.set('reviewer', next)
            url.searchParams.set('stage', 'structure')
            window.history.replaceState(null, '', url)
          }}
        />
      )}
    </div>
  )
}
