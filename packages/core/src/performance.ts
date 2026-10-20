import type { TabBeat, TabNote } from './domain.js'

export interface GuitarGesture {
  at: number
  notes: TabNote[]
  style: 'strum' | 'arpeggio'
  direction: 'up' | 'down'
}

/**
 * Infer a single sweep from ordered string attacks, independently of beat
 * boundaries. Simultaneous notes provide no evidence of picking direction.
 * This is an onset heuristic, not a classifier for rasgueado or tremolo.
 */
export function guitarGestures(
  input: readonly TabNote[],
  beatDurationAt: (at: number) => number,
): GuitarGesture[] {
  const notes = [...input].sort((a, b) => a.at - b.at)
  const gestures: GuitarGesture[] = []
  const eligible = (note: TabNote) =>
    ['unknown', 'pick', 'strum-up', 'strum-down'].includes(note.technique) &&
    !note.relatedNoteId && !note.tieFromPrevious
  for (let first = 0; first < notes.length; first += 1) {
    if (!eligible(notes[first])) continue
    const beatDuration = beatDurationAt(notes[first].at)
    const maximumSpan = Math.min(0.36, beatDuration * 0.8)
    let best: GuitarGesture | undefined
    const strings = new Set<number>()
    for (let last = first; last < notes.length; last += 1) {
      const note = notes[last]
      if (
        !eligible(note) ||
        strings.has(note.string) ||
        (best?.style === 'strum' && note.at - notes[last - 1].at > 0.075) ||
        note.at - notes[first].at > maximumSpan ||
        (last > first && note.at - notes[last - 1].at > Math.min(0.18, beatDuration * 0.4))
      ) break
      strings.add(note.string)
      if (strings.size < 3) continue
      const group = notes.slice(first, last + 1)
      const directions = new Set<number>()
      let resolvedPairs = 0
      for (let a = 0; a < group.length; a += 1) {
        for (let b = a + 1; b < group.length; b += 1) {
          if (group[b].at - group[a].at <= 0.01) continue
          directions.add(Math.sign(group[b].string - group[a].string))
          resolvedPairs += 1
        }
      }
      if (directions.size > 1) break
      const spread = note.at - notes[first].at
      if (directions.size !== 1 || resolvedPairs < 2 || spread <= 0.025) continue
      const onsetClusters = group.reduce(
        (count, item, index) =>
          index > 0 && item.at - group[index - 1].at > 0.01
            ? count + 1
            : count,
        1,
      )
      const orderedStrings = [...strings].sort((a, b) => a - b)
      const contiguousStrings =
        orderedStrings.at(-1)! - orderedStrings[0] + 1 ===
        orderedStrings.length
      // With only two observed onset layers, skipped strings can manufacture a
      // direction from a simultaneous dyad and an unrelated following note.
      if (onsetClusters < 3 && !contiguousStrings) continue
      const maximumGap = Math.max(...group.slice(1).map((item, index) => item.at - group[index].at))
      best = {
        at: group[0].at,
        notes: group,
        direction: directions.has(1) ? 'up' : 'down',
        style: spread <= Math.min(0.14, beatDuration * 0.35) && maximumGap <= 0.075
          ? 'strum' : 'arpeggio',
      }
    }
    if (!best) continue
    gestures.push(best)
    first += best.notes.length - 1
  }
  return gestures
}

export interface RhythmValue {
  beats: number
  flags: number
  dotted: boolean
  hollow: boolean
}

/** Notated attack spacing is distinct from an acoustic note's decay. */
export function rhythmValue(beats: number): RhythmValue {
  const values = [
    { beats: 4, flags: 0, dotted: false, hollow: true },
    { beats: 3, flags: 0, dotted: true, hollow: true },
    { beats: 2, flags: 0, dotted: false, hollow: true },
    { beats: 1.5, flags: 0, dotted: true, hollow: false },
    { beats: 1, flags: 0, dotted: false, hollow: false },
    { beats: 0.75, flags: 1, dotted: true, hollow: false },
    { beats: 0.5, flags: 1, dotted: false, hollow: false },
    { beats: 0.375, flags: 2, dotted: true, hollow: false },
    { beats: 0.25, flags: 2, dotted: false, hollow: false },
    { beats: 0.125, flags: 3, dotted: false, hollow: false },
  ]
  // Rubato / onset uncertainty must not turn a near-quarter into a dotted
  // eighth. Dotted values require closer evidence than ordinary divisions.
  return values.filter((value) => !value.dotted || Math.abs(beats / value.beats - 1) <= 0.08)
    .reduce((best, value) =>
    Math.abs(Math.log(Math.max(beats, 0.001) / value.beats)) <
    Math.abs(Math.log(Math.max(beats, 0.001) / best.beats)) ? value : best,
  )
}

/** Integrate the local tempo map; do not assume constant BPM across measures. */
export function beatPosition(at: number, beats: readonly Pick<TabBeat, 'at' | 'duration'>[]): number {
  if (!beats.length) return 0
  let low = 0
  let high = beats.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (beats[middle].at <= at) low = middle + 1
    else high = middle
  }
  const index = Math.max(0, low - 1)
  return index + (at - beats[index].at) / Math.max(0.01, beats[index].duration)
}
