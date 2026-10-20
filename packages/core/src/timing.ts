import type { TempoMap, TempoMapPoint } from './domain.js'
import {
  CoreValidationError,
  arrayAt,
  enumAt,
  numberAt,
  recordAt,
} from './validation.js'

const notationFractions = [0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1]

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function quantile(sortedValues: readonly number[], fraction: number): number {
  const position = (sortedValues.length - 1) * fraction
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  const weight = position - lower
  return (
    sortedValues[lower] * (1 - weight) +
    sortedValues[upper] * weight
  )
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  return quantile(sorted, 0.5)
}

export function tempoMapFromBeats(
  beatTimes: readonly number[],
  fallbackBpm = 120,
): TempoMap {
  const beats = beatTimes
    .filter((beat) => Number.isFinite(beat) && beat >= 0)
    .filter(
      (beat, index, values) =>
        index === 0 || beat - values[index - 1] > 0.05,
    )
  const safeFallback =
    Number.isFinite(fallbackBpm) && fallbackBpm > 0 ? fallbackBpm : 120
  if (beats.length < 2) {
    return {
      mode: 'steady',
      medianBpm: safeFallback,
      variation: 0,
      points: [{
        at: beats[0] ?? 0,
        beat: 0,
        bpm: safeFallback,
        confidence: 0,
      }],
    }
  }

  const intervals = beats
    .slice(1)
    .map((beat, index) => beat - beats[index])
  const points: TempoMapPoint[] = beats.map((at, index) => {
    const from = Math.max(0, index - 2)
    const to = Math.min(intervals.length, index + 3)
    const localIntervals = intervals
      .slice(from, to)
      .filter((interval) => interval >= 0.15 && interval <= 3)
    const localMedian =
      localIntervals.length > 0 ? median(localIntervals) : 60 / safeFallback
    const deviations = localIntervals.map((interval) =>
      Math.abs(interval - localMedian),
    )
    const relativeDeviation =
      deviations.length > 0
        ? median(deviations) / Math.max(localMedian, 1e-6)
        : 1
    return {
      at,
      beat: index,
      bpm: 60 / localMedian,
      confidence: clamp(1 - relativeDeviation * 6, 0, 1),
    }
  })
  const tempos = points
    .map((point) => point.bpm)
    .filter((bpm) => Number.isFinite(bpm) && bpm >= 20 && bpm <= 320)
    .sort((left, right) => left - right)
  const medianBpm = tempos.length > 0 ? quantile(tempos, 0.5) : safeFallback
  const variation =
    tempos.length > 1
      ? (quantile(tempos, 0.9) - quantile(tempos, 0.1)) /
        Math.max(medianBpm, 1)
      : 0
  return {
    mode: variation >= 0.04 ? 'variable' : 'steady',
    medianBpm,
    variation,
    points,
  }
}

export function tempoMapAt(value: unknown, path: string): TempoMap {
  const map = recordAt(value, path)
  const points = arrayAt(map.points, `${path}.points`).map(
    (value, index): TempoMapPoint => {
      const point = recordAt(value, `${path}.points[${index}]`)
      return {
        at: numberAt(point.at, `${path}.points[${index}].at`),
        beat: numberAt(point.beat, `${path}.points[${index}].beat`),
        bpm: numberAt(point.bpm, `${path}.points[${index}].bpm`),
        confidence: numberAt(
          point.confidence,
          `${path}.points[${index}].confidence`,
        ),
      }
    },
  )
  points.forEach((point, index) => {
    if (
      point.at < 0 ||
      point.beat < 0 ||
      point.bpm < 20 ||
      point.bpm > 320 ||
      point.confidence < 0 ||
      point.confidence > 1
    ) {
      throw new CoreValidationError(
        `${path}.points[${index}]`,
        'invalid tempo-map point',
      )
    }
    if (
      index > 0 &&
      (point.at <= points[index - 1].at ||
        point.beat <= points[index - 1].beat)
    ) {
      throw new CoreValidationError(
        `${path}.points[${index}]`,
        'tempo-map points must be strictly increasing',
      )
    }
  })
  const medianBpm = numberAt(
    map.medianBpm ?? map.median_bpm,
    `${path}.medianBpm`,
  )
  const variation = numberAt(map.variation, `${path}.variation`)
  if (
    points.length === 0 ||
    medianBpm < 20 ||
    medianBpm > 320 ||
    variation < 0
  ) {
    throw new CoreValidationError(path, 'invalid tempo map')
  }
  return {
    mode: enumAt(map.mode, ['steady', 'variable'] as const, `${path}.mode`),
    medianBpm,
    variation,
    points,
  }
}

export interface NotationPosition {
  beatIndex: number
  notationAt: number
  notationBeat: number
  snapped: boolean
}

export function notationPositionAt(
  at: number,
  beatStarts: readonly number[],
): NotationPosition {
  if (beatStarts.length < 2) {
    return {
      beatIndex: 0,
      notationAt: at,
      notationBeat: 0,
      snapped: false,
    }
  }
  if (at <= beatStarts[0]) {
    return {
      beatIndex: 0,
      notationAt: beatStarts[0],
      notationBeat: 0,
      snapped: true,
    }
  }
  let lower = 0
  let upper = beatStarts.length
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2)
    if (beatStarts[middle] <= at) {
      lower = middle + 1
    } else {
      upper = middle
    }
  }
  const physicalBeat = clamp(lower - 1, 0, beatStarts.length - 2)
  const start = beatStarts[physicalBeat]
  const end = beatStarts[physicalBeat + 1]
  const duration = Math.max(0.05, end - start)
  const performedFraction = (at - start) / duration
  const nearestFraction = notationFractions.reduce((best, candidate) =>
    Math.abs(candidate - performedFraction) <
    Math.abs(best - performedFraction)
      ? candidate
      : best,
  )
  const candidateAt = start + nearestFraction * duration
  const tolerance = Math.max(0.015, Math.min(0.035, duration * 0.06))
  const snapped = Math.abs(candidateAt - at) <= tolerance
  const notationFraction = snapped
    ? nearestFraction
    : clamp(performedFraction, 0, 0.999999)
  const notationBeat = physicalBeat + notationFraction
  const beatIndex = Math.min(
    beatStarts.length - 2,
    Math.floor(notationBeat + 1e-8),
  )
  return {
    beatIndex,
    notationAt: snapped ? candidateAt : at,
    notationBeat,
    snapped,
  }
}
