import type {
  BeatReviewSummary,
  BeatStructureDocument,
  BeatStructureSummary,
  BeatReviewDraft,
  BeatReviewDocument,
} from './types'

export function coreBeatClipIds(
  reviews: readonly BeatReviewSummary[],
  structures: readonly BeatStructureSummary[],
): Set<string> {
  const recordings = new Map<string, BeatReviewSummary[]>()
  for (const review of reviews) {
    const clips = recordings.get(review.recordingId) ?? []
    clips.push(review)
    recordings.set(review.recordingId, clips)
  }
  const selected = new Set<string>()
  for (const [recordingId, clips] of recordings) {
    clips.sort(
      (left, right) =>
        left.sourceStart - right.sourceStart ||
        left.clipId.localeCompare(right.clipId),
    )
    const structure = structures.find(
      (item) => item.recordingId === recordingId,
    )
    const tags = new Set(clips.flatMap((clip) => clip.tags))
    const changing = [
      'rubato',
      'expressive-timing',
      'tempo-change',
      'mixed-texture',
      'mixed-meter-candidate',
    ].some((tag) => tags.has(tag))
    const count = Math.min(
      clips.length,
      structure?.phaseStatus === 'uncertain' ||
        (structure?.meterChangeCount ?? 0) > 0
        ? 5
        : changing
          ? 3
          : 2,
    )
    for (let index = 0; index < count; index += 1) {
      const position =
        count === 1
          ? 0
          : Math.round((index * (clips.length - 1)) / (count - 1))
      selected.add(clips[position].clipId)
    }
    for (const clip of clips) {
      if (
        clip.status !== 'draft' ||
        Object.values(clip.checks).some(Boolean)
      ) {
        selected.add(clip.clipId)
      }
    }
  }
  return selected
}

export function draftWithConfirmedStructure(
  document: BeatReviewDocument,
  structure: BeatStructureDocument,
  draft: BeatReviewDraft,
): BeatReviewDraft | null {
  const review = structure.review
  if (
    review.status !== 'submitted' ||
    document.review.status !== 'draft' ||
    review.phase_status !== 'confirmed' ||
    review.first_downbeat_at === null ||
    structure.candidateBeats.length === 0 ||
    document.review.group_id !== `${review.recording_id}-recording` ||
    document.review.annotator_alias !== review.annotator_alias ||
    document.review.corpus_version !== review.corpus_version
  ) {
    return null
  }
  const start = document.review.audio_source_start
  const end = document.review.audio_source_end
  if (review.meter_changes.some((change) => change.at > start && change.at < end)) {
    return null
  }
  const beats = structure.candidateBeats
  const nearestIndex = (at: number) => {
    let nearest = 0
    for (let index = 1; index < beats.length; index += 1) {
      if (Math.abs(beats[index] - at) < Math.abs(beats[nearest] - at)) {
        nearest = index
      }
    }
    return nearest
  }
  const initialIndex = nearestIndex(review.first_downbeat_at)
  const sections = [
    {
      index: initialIndex % review.time_signature[0],
      signature: review.time_signature,
      measure: 1,
    },
  ]
  for (const change of review.meter_changes) {
    if (change.at > start) break
    const previous = sections[sections.length - 1]
    const index = nearestIndex(change.at)
    sections.push({
      index,
      signature: change.time_signature,
      measure:
        previous.measure +
        Math.ceil((index - previous.index) / previous.signature[0]),
    })
  }
  const section = sections[sections.length - 1]
  const firstIndex = nearestIndex(start + (draft.beats[0]?.at ?? 0))
  return {
    ...draft,
    time_signature: [...section.signature],
    pickup_beats: Math.min(review.pickup_beats, section.signature[0] - 1),
    beats: draft.beats.map((marker, index) => {
      const offset = firstIndex + index - section.index
      const numerator = section.signature[0]
      const beat = ((offset % numerator) + numerator) % numerator + 1
      return {
        ...marker,
        beat,
        measure: Math.max(0, section.measure + Math.floor(offset / numerator)),
        downbeat: beat === 1,
      }
    }),
    checks: { ...draft.checks, downbeats: false, meter: false },
  }
}
