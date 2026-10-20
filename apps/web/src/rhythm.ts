import { beatPosition, guitarGestures, rhythmValue } from '@gts/core'
import type { RhythmValue } from '@gts/core'
import type { BeatStyle, TabBeat, TabDocument, TabNote } from './types'

export function inferBeatStyle(beat: TabBeat): BeatStyle {
  if (beat.style) return beat.style
  if (beat.notes.some((note) => note.technique === 'tremolo')) return 'tremolo'
  const gestures = guitarGestures(beat.notes, () => beat.duration)
  return gestures.length === 1 && gestures[0].notes.length === beat.notes.length
    ? gestures[0].style
    : 'pick'
}

export interface ScoreTiming {
  values: ReadonlyMap<string, RhythmValue>
  displayTimes: ReadonlyMap<string, number>
  attacks: ReadonlyMap<number, readonly ScoreAttack[]>
  segments: ReadonlyMap<number, readonly ScoreRhythmSegment[]>
  heldBeatTimes: ReadonlySet<number>
  notes: readonly TabNote[]
}

export interface ScoreAttack {
  at: number
  position: number
  beatIndex: number
  notes: TabNote[]
  tuplet?: 3
}

export interface ScoreRhythmSegment {
  kind: 'attack' | 'continuation' | 'rest'
  at: number
  endAt: number
  position: number
  value: RhythmValue
  notes: TabNote[]
}

interface MeasureRange {
  measure: TabDocument['measures'][number]
  start: number
  end: number
  capacity: number
  quarterScale: number
}

interface InternalAttack extends ScoreAttack {
  range: MeasureRange
  globalPosition: number
  tripletUnit?: number
}

const GRID = 0.125
const EPSILON = 0.000_001
const NOTATED_DURATIONS = [4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.125]
const BINARY_SIXTEENTHS = [0, 0.25, 0.5, 0.75]
const BINARY_THIRTY_SECONDS = [
  0,
  0.125,
  0.25,
  0.375,
  0.5,
  0.625,
  0.75,
  0.875,
]
const TRIPLETS = [0, 1 / 3, 2 / 3]

interface BeatQuantization {
  positions: number[]
  assignments: number[]
  tuplet?: 3
  unit?: number
}

interface BeatQuantizationCandidate {
  positions: number[]
  tolerance: number
  penalty: number
  merged?: boolean
  tuplet?: 3
  unit?: number
}

function combinations(values: readonly number[], size: number): number[][] {
  if (size === 0) return [[]]
  if (size > values.length) return []
  return values.flatMap((value, index) =>
    combinations(values.slice(index + 1), size - 1).map((tail) => [
      value,
      ...tail,
    ]),
  )
}

function quantizeBeatPositions(
  observed: readonly number[],
  validAssignments: (assignments: readonly number[]) => boolean,
): BeatQuantization | undefined {
  if (observed.length === 0 || observed.length > BINARY_THIRTY_SECONDS.length) {
    return undefined
  }
  const exactTemplates: BeatQuantizationCandidate[] = [
    ...(observed.length <= BINARY_SIXTEENTHS.length
      ? combinations(BINARY_SIXTEENTHS, observed.length).map((positions) => ({
          positions,
          tolerance: 0.18,
          penalty: 0,
        }))
      : []),
    ...(observed.length <= TRIPLETS.length
      ? combinations(TRIPLETS, observed.length).map((positions) => ({
          positions,
          tolerance: 0.14,
          penalty: 0.002,
          tuplet: 3 as const,
          unit: 1 / 3,
        }))
      : []),
    ...combinations(BINARY_THIRTY_SECONDS, observed.length).map((positions) => ({
      positions,
      tolerance: 0.1,
      penalty: 0.005 + Math.max(0, observed.length - 2) * 0.015,
    })),
  ]
  const mergedTemplates: BeatQuantizationCandidate[] = observed.length > 2
    ? Array.from(
        { length: Math.min(BINARY_SIXTEENTHS.length, observed.length - 1) - 1 },
        (_, index) => index + 2,
      ).flatMap((size) =>
        combinations(BINARY_SIXTEENTHS, size).map((positions) => ({
          positions,
          tolerance: 0.18,
          penalty: (observed.length - size) * 0.01,
          merged: true,
        }))
      )
    : []
  const candidates = [...exactTemplates, ...mergedTemplates].flatMap(
    (candidate) => {
      const assignments = candidate.merged
        ? observed.map((position) =>
            candidate.positions.reduce(
              (best, target, index) =>
                Math.abs(target - position) <
                Math.abs(candidate.positions[best] - position)
                  ? index
                  : best,
              0,
            )
          )
        : observed.map((_position, index) => index)
      if (
        candidate.positions.some((_, index) => !assignments.includes(index)) ||
        !validAssignments(assignments)
      ) {
        return []
      }
      const errors = observed.map((position, index) =>
        Math.abs(position - candidate.positions[assignments[index]])
      )
      const maximumError = Math.max(...errors)
      if (maximumError > candidate.tolerance) return []
      return [{
        ...candidate,
        score:
          errors.reduce((sum, error) => sum + error * error, 0) /
          errors.length +
          candidate.penalty,
        assignments,
      }]
    }
  )
  const best = candidates.sort((left, right) => left.score - right.score)[0]
  return best
    ? {
        positions: best.assignments.map((index) => best.positions[index]),
        assignments: best.assignments,
        tuplet: best.tuplet,
        unit: best.unit,
      }
    : undefined
}

function valueForDuration(beats: number): RhythmValue {
  const canonical = NOTATED_DURATIONS.find(
    (duration) => Math.abs(duration - beats) <= EPSILON,
  )
  if (canonical !== undefined) return rhythmValue(canonical)
  return {
    beats,
    flags: beats < 0.5 ? 2 : beats < 1 ? 1 : 0,
    dotted: false,
    hollow: beats >= 2,
  }
}

export function scoreTiming(tab: TabDocument): ScoreTiming {
  const beats = tab.measures.flatMap((measure) => measure.beats)
  const quarterScale = 4 / tab.timeSignature[1]
  let offset = 0
  const ranges = tab.measures.map((measure) => {
    const capacity = Math.max(
      quarterScale,
      measure.beats.length * quarterScale,
    )
    const range = {
      measure,
      start: offset,
      end: offset + capacity,
      capacity,
      quarterScale,
    }
    offset = range.end
    return range
  })

  const timeAt = (position: number) => {
    const range = ranges.find((candidate) => position < candidate.end - EPSILON) ??
      ranges.at(-1)
    if (!range) return 0
    const local = Math.max(0, Math.min(range.capacity, position - range.start))
    if (local >= range.capacity - EPSILON) {
      return range.measure.start + range.measure.duration
    }
    const beatPositionInMeasure = local / range.quarterScale
    const beatIndex = Math.min(
      range.measure.beats.length - 1,
      Math.max(0, Math.floor(beatPositionInMeasure)),
    )
    const beat = range.measure.beats[beatIndex]
    return beat.at + (beatPositionInMeasure - beatIndex) * beat.duration
  }
  const positionAt = (at: number) => {
    const range = ranges.find((candidate) =>
      at < candidate.measure.start + candidate.measure.duration - EPSILON
    ) ?? ranges.at(-1)
    if (!range) return 0
    const local = beatPosition(at, range.measure.beats) * range.quarterScale
    return range.start + Math.max(0, Math.min(range.capacity, local))
  }

  const allAttacks: InternalAttack[] = []
  const attacks = new Map<number, ScoreAttack[]>()
  const displayTimes = new Map<string, number>()
  for (const range of ranges) {
    const rawGroups: Array<{
      at: number
      beatIndex: number
      notes: TabNote[]
      rawPosition: number
      targetPosition: number
      tuplet?: 3
      tripletUnit?: number
    }> = []
    range.measure.beats.forEach((beat, beatIndex) => {
      const collapse = beat.notes.length > 0 &&
        ['strum', 'rasgueado'].includes(inferBeatStyle(beat)) &&
        !beat.notes.some((note) => note.relatedNoteId)
      const anchor = collapse
        ? Math.min(...beat.notes.map((note) => note.notationAt ?? note.at))
        : 0
      const ordered = beat.notes
        .map((note) => ({
          note,
          at: collapse ? anchor : note.notationAt ?? note.at,
        }))
        .sort((left, right) => left.at - right.at)
      for (const candidate of ordered) {
        const previous = rawGroups.at(-1)
        if (
          previous &&
          previous.beatIndex === beatIndex &&
          candidate.at - previous.at <= 0.04 &&
          previous.notes.every((note) => note.string !== candidate.note.string)
        ) {
          previous.notes.push(candidate.note)
        } else {
          const rawPosition = Math.max(
            0,
            Math.min(
              range.capacity - GRID,
              beatPosition(candidate.at, range.measure.beats) *
                range.quarterScale,
            ),
          )
          rawGroups.push({
            at: candidate.at,
            beatIndex,
            notes: [candidate.note],
            rawPosition,
            targetPosition: Math.round(rawPosition / GRID) * GRID,
          })
        }
      }
    })

    for (let beatIndex = 0; beatIndex < range.measure.beats.length; beatIndex += 1) {
      const beatGroups = rawGroups.filter((group) => group.beatIndex === beatIndex)
      const beat = range.measure.beats[beatIndex]
      const observed = beatGroups.map(
        (group) => (group.at - beat.at) / beat.duration,
      )
      const quantization = quantizeBeatPositions(
        observed,
        (assignments) => {
          const buckets = new Map<number, TabNote[]>()
          assignments.forEach((assignment, index) => {
            buckets.set(
              assignment,
              [
                ...(buckets.get(assignment) ?? []),
                ...beatGroups[index].notes,
              ],
            )
          })
          return [...buckets.values()].every((notes) =>
            new Set(notes.map((note) => note.string)).size === notes.length
          )
        },
      )
      if (quantization) {
        beatGroups.forEach((group, index) => {
          group.targetPosition =
            beatIndex * range.quarterScale +
            quantization.positions[index] * range.quarterScale
          if (quantization.tuplet && quantization.unit) {
            group.tuplet = quantization.tuplet
            group.tripletUnit = quantization.unit * range.quarterScale
          }
        })
        const retainedGroups = new Map<number, typeof beatGroups[number]>()
        beatGroups.forEach((group, index) => {
          const assignment = quantization.assignments[index]
          const retained = retainedGroups.get(assignment)
          if (retained) {
            retained.notes.push(...group.notes)
            rawGroups.splice(rawGroups.indexOf(group), 1)
          } else {
            retainedGroups.set(assignment, group)
          }
        })
      }
    }

    rawGroups.sort((left, right) =>
      left.rawPosition - right.rawPosition || left.at - right.at
    )
    const minimumStep = Math.min(
      GRID,
      range.capacity / Math.max(1, rawGroups.length),
    )
    let previousPosition = -minimumStep
    const measureAttacks = rawGroups.map((group, index) => {
      const lower = Math.max(0, previousPosition + minimumStep)
      const upper = Math.max(
        lower,
        range.capacity - (rawGroups.length - index) * minimumStep,
      )
      const position = Math.max(lower, Math.min(upper, group.targetPosition))
      previousPosition = position
      const globalPosition = range.start + position
      const at = timeAt(globalPosition)
      group.notes.forEach((note) => displayTimes.set(note.id, at))
      const attack: InternalAttack = {
        at,
        position,
        beatIndex: group.beatIndex,
        notes: group.notes,
        tuplet: group.tuplet,
        tripletUnit: group.tripletUnit,
        range,
        globalPosition,
      }
      allAttacks.push(attack)
      return attack
    })
    attacks.set(range.measure.number, measureAttacks)
  }

  allAttacks.sort((left, right) => left.globalPosition - right.globalPosition)
  const values = new Map<string, RhythmValue>()
  const heldBeatTimes = new Set<number>()
  const segments = new Map<number, ScoreRhythmSegment[]>()
  const appendInterval = (
    kind: 'sound' | 'rest',
    start: number,
    end: number,
    attack?: InternalAttack,
  ) => {
    let cursor = start
    let firstSoundPiece = true
    while (cursor < end - EPSILON) {
      const range = ranges.find((candidate) =>
        cursor < candidate.end - EPSILON
      ) ?? ranges.at(-1)
      if (!range) break
      const sectionEnd = Math.min(end, range.end)
      while (cursor < sectionEnd - EPSILON) {
        const remaining = sectionEnd - cursor
        const preferredTriplet =
          kind === 'sound' &&
          firstSoundPiece &&
          attack?.tripletUnit &&
          attack.tripletUnit <= remaining + EPSILON
            ? attack.tripletUnit
            : undefined
        const duration = preferredTriplet ??
          NOTATED_DURATIONS.find((candidate) =>
            candidate <= remaining + EPSILON
          ) ??
          remaining
        const exactDuration = Math.min(duration, remaining)
        const value = valueForDuration(exactDuration)
        const segmentKind =
          kind === 'rest'
            ? 'rest'
            : firstSoundPiece
              ? 'attack'
              : 'continuation'
        const pieces = segments.get(range.measure.number) ?? []
        pieces.push({
          kind: segmentKind,
          at: timeAt(cursor),
          endAt: timeAt(cursor + exactDuration),
          position: cursor - range.start,
          value,
          notes: attack?.notes ?? [],
        })
        segments.set(range.measure.number, pieces)
        if (segmentKind === 'attack' && attack) {
          attack.notes.forEach((note) => values.set(note.id, value))
        }
        if (kind === 'sound' && attack) {
          range.measure.beats.forEach((beat, beatIndex) => {
            const beatPositionInScore =
              range.start + beatIndex * range.quarterScale
            if (
              beatPositionInScore > attack.globalPosition + EPSILON &&
              beatPositionInScore >= cursor - EPSILON &&
              beatPositionInScore < cursor + exactDuration - EPSILON
            ) {
              heldBeatTimes.add(beat.at)
            }
          })
          firstSoundPiece = false
        }
        cursor += exactDuration
      }
    }
  }

  if (allAttacks.length === 0) {
    appendInterval('rest', 0, offset)
  } else {
    appendInterval('rest', 0, allAttacks[0].globalPosition)
    allAttacks.forEach((attack, index) => {
      const next = allAttacks[index + 1]?.globalPosition ?? offset
      const decay = Math.min(
        next,
        positionAt(
          Math.max(...attack.notes.map((note) => note.at + note.duration)),
        ),
      )
      const gap = next - attack.globalPosition
      const decayGap = next - decay
      const quantizedDecay = Math.round(decay / GRID) * GRID
      // Short release gaps are articulation. Longer gaps become explicit rests.
      const soundEnd =
        gap <= 1.75 || decayGap <= 0.5
          ? next
          : Math.max(
              attack.globalPosition + Math.min(GRID, gap),
              Math.min(next, quantizedDecay),
            )
      appendInterval('sound', attack.globalPosition, soundEnd, attack)
      appendInterval('rest', soundEnd, next)
    })
  }

  const notes = beats
    .flatMap((beat) => beat.notes)
    .sort((left, right) =>
      (displayTimes.get(left.id) ?? left.notationAt ?? left.at) -
      (displayTimes.get(right.id) ?? right.notationAt ?? right.at)
    )
  return {
    values,
    displayTimes,
    attacks,
    segments,
    heldBeatTimes,
    notes,
  }
}
