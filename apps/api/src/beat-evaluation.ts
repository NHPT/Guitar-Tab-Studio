import { createHash } from 'node:crypto'
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import {
  beatGoldStatus,
  BeatReviewRequestError,
  defaultBeatGoldRoot,
  evaluateBeatTimes,
  listBeatGold,
  type BeatMetrics,
  type BeatReviewMarker,
  type BeatReviewPackage,
} from './beat-reviews.js'

const VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const THRESHOLDS = {
  beatF1: 0.95,
  medianErrorMs: 20,
  p95ErrorMs: 50,
  downbeatF1: 0.95,
} as const

export interface BeatEvaluationItem {
  clipId: string
  split: 'train' | 'validation' | 'test'
  referenceBeats: number
  predictedBeats: number
  beatMetrics: BeatMetrics
  downbeatMetrics: BeatMetrics
  meterCorrect: boolean
  pickupCorrect: boolean
}

export interface BeatEvaluationReport {
  schemaVersion: 1
  generatedAt: string
  modelVersion: string
  corpusVersion: string
  corpusSha256: string
  coverage: {
    adjudicated: number
    total: number
    complete: boolean
    splitCounts: Record<string, number>
  }
  aggregate: {
    beatMetrics: BeatMetrics
    downbeatMetrics: BeatMetrics
    meterAccuracy: number
    pickupAccuracy: number
  }
  gate: {
    passed: boolean
    checks: {
      coverage: boolean
      beatF1: boolean
      medianError: boolean
      p95Error: boolean
      downbeatF1: boolean
      meter: boolean
      pickup: boolean
    }
    thresholds: typeof THRESHOLDS
  }
  items: BeatEvaluationItem[]
  reportSha256: string
}

function splitFor(review: BeatReviewPackage): BeatEvaluationItem['split'] {
  const split = review.tags.find(
    (tag) => tag === 'train' || tag === 'validation' || tag === 'test',
  )
  if (!split) {
    throw new BeatReviewRequestError('金标准缺少数据分区', 500)
  }
  return split
}

function analysisPath(root: string, review: BeatReviewPackage): string {
  const recordingId = review.group_id.replace(/-recording$/u, '')
  if (!VERSION_PATTERN.test(recordingId)) {
    throw new BeatReviewRequestError('录音标识无效', 500)
  }
  const path = resolve(root, 'analysis', `${recordingId}.json`)
  if (
    !existsSync(path) ||
    lstatSync(path).isSymbolicLink() ||
    !lstatSync(path).isFile()
  ) {
    throw new BeatReviewRequestError(`缺少 ${recordingId} 的候选分析`, 500)
  }
  return path
}

function markersFromAnalysis(
  analysis: Record<string, unknown>,
  review: BeatReviewPackage,
): BeatReviewMarker[] {
  const signature = Array.isArray(analysis.time_signature)
    ? analysis.time_signature
    : [4, 4]
  const numerator = Number(signature[0])
  const pickup = Number(analysis.pickup_beats ?? 0)
  const beats = Array.isArray(analysis.beats) ? analysis.beats : []
  return beats.flatMap((rawAt, index) => {
    const sourceAt = Number(rawAt)
    if (
      !Number.isFinite(sourceAt) ||
      sourceAt < review.audio_source_start ||
      sourceAt >= review.audio_source_end
    ) {
      return []
    }
    const beat =
      index < pickup
        ? numerator - pickup + index + 1
        : ((index - pickup) % numerator) + 1
    const measure =
      index < pickup ? 0 : Math.floor((index - pickup) / numerator) + 1
    return [
      {
        at: sourceAt - review.audio_source_start,
        beat,
        measure,
        downbeat: beat === 1,
      },
    ]
  })
}

function evaluationWindow(
  review: BeatReviewPackage,
  markers: readonly BeatReviewMarker[],
): BeatReviewMarker[] {
  return markers.filter(
    (marker) =>
      marker.at >= review.evaluation_start &&
      marker.at < review.evaluation_end,
  )
}

function canonicalDigest(value: unknown): string {
  return createHash('sha256')
    .update(JSON.stringify(value), 'utf8')
    .digest('hex')
}

export function evaluateBeatGold(
  modelVersion = 'current-candidate',
  root = defaultBeatGoldRoot(),
): BeatEvaluationReport {
  if (!VERSION_PATTERN.test(modelVersion)) {
    throw new BeatReviewRequestError('模型版本标识无效')
  }
  const status = beatGoldStatus(root)
  if (!status.manifest) {
    throw new BeatReviewRequestError('节拍 Pilot 尚未准备', 404)
  }
  const gold = listBeatGold(root)
  const items: BeatEvaluationItem[] = []
  const aggregateReference: BeatReviewMarker[] = []
  const aggregatePredicted: BeatReviewMarker[] = []
  const aggregateReferenceDownbeats: BeatReviewMarker[] = []
  const aggregatePredictedDownbeats: BeatReviewMarker[] = []

  gold.forEach((review, index) => {
    const analysis = JSON.parse(
      readFileSync(analysisPath(root, review), 'utf8'),
    ) as Record<string, unknown>
    const reference = evaluationWindow(review, review.beats)
    const predicted = evaluationWindow(
      review,
      markersFromAnalysis(analysis, review),
    )
    const referenceDownbeats = reference.filter((marker) => marker.downbeat)
    const predictedDownbeats = predicted.filter((marker) => marker.downbeat)
    const offset = index * 1000
    aggregateReference.push(
      ...reference.map((marker) => ({ ...marker, at: marker.at + offset })),
    )
    aggregatePredicted.push(
      ...predicted.map((marker) => ({ ...marker, at: marker.at + offset })),
    )
    aggregateReferenceDownbeats.push(
      ...referenceDownbeats.map((marker) => ({
        ...marker,
        at: marker.at + offset,
      })),
    )
    aggregatePredictedDownbeats.push(
      ...predictedDownbeats.map((marker) => ({
        ...marker,
        at: marker.at + offset,
      })),
    )
    const signature = Array.isArray(analysis.time_signature)
      ? analysis.time_signature.map(Number)
      : [4, 4]
    items.push({
      clipId: review.clip_id,
      split: splitFor(review),
      referenceBeats: reference.length,
      predictedBeats: predicted.length,
      beatMetrics: evaluateBeatTimes(reference, predicted),
      downbeatMetrics: evaluateBeatTimes(
        referenceDownbeats,
        predictedDownbeats,
      ),
      meterCorrect:
        signature[0] === review.time_signature[0] &&
        signature[1] === review.time_signature[1],
      pickupCorrect:
        Number(analysis.pickup_beats ?? 0) === review.pickup_beats,
    })
  })

  const beatMetrics = evaluateBeatTimes(
    aggregateReference,
    aggregatePredicted,
  )
  const downbeatMetrics = evaluateBeatTimes(
    aggregateReferenceDownbeats,
    aggregatePredictedDownbeats,
  )
  const meterAccuracy =
    items.filter((item) => item.meterCorrect).length / Math.max(1, items.length)
  const pickupAccuracy =
    items.filter((item) => item.pickupCorrect).length / Math.max(1, items.length)
  const coverage = gold.length === status.manifest.clips
  const checks = {
    coverage,
    beatF1: beatMetrics.f1 >= THRESHOLDS.beatF1,
    medianError:
      beatMetrics.median_error_ms !== null &&
      beatMetrics.median_error_ms <= THRESHOLDS.medianErrorMs,
    p95Error:
      beatMetrics.p95_error_ms !== null &&
      beatMetrics.p95_error_ms <= THRESHOLDS.p95ErrorMs,
    downbeatF1: downbeatMetrics.f1 >= THRESHOLDS.downbeatF1,
    meter: items.length > 0 && meterAccuracy === 1,
    pickup: items.length > 0 && pickupAccuracy === 1,
  }
  const body = {
    schemaVersion: 1 as const,
    generatedAt: new Date().toISOString(),
    modelVersion,
    corpusVersion: status.manifest.corpusVersion,
    corpusSha256: status.manifest.corpusSha256,
    coverage: {
      adjudicated: gold.length,
      total: status.manifest.clips,
      complete: coverage,
      splitCounts: Object.fromEntries(
        ['train', 'validation', 'test'].map((split) => [
          split,
          items.filter((item) => item.split === split).length,
        ]),
      ),
    },
    aggregate: {
      beatMetrics,
      downbeatMetrics,
      meterAccuracy,
      pickupAccuracy,
    },
    gate: {
      passed: Object.values(checks).every(Boolean),
      checks,
      thresholds: THRESHOLDS,
    },
    items,
  }
  return {
    ...body,
    reportSha256: canonicalDigest(body),
  }
}

function evaluationsRoot(root: string): string {
  return resolve(root, 'evaluations')
}

export function saveBeatEvaluationRun(
  modelVersion: string,
  root = defaultBeatGoldRoot(),
): BeatEvaluationReport {
  const report = evaluateBeatGold(modelVersion, root)
  const directory = evaluationsRoot(root)
  mkdirSync(directory, { recursive: true })
  const path = resolve(directory, `${modelVersion}.json`)
  if (existsSync(path)) {
    throw new BeatReviewRequestError('该模型版本已有冻结评测报告', 409)
  }
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(report, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  })
  renameSync(temporary, path)
  return report
}

export function listBeatEvaluationRuns(
  root = defaultBeatGoldRoot(),
): BeatEvaluationReport[] {
  const directory = evaluationsRoot(root)
  if (!existsSync(directory)) return []
  return readdirSync(directory)
    .filter((name) => VERSION_PATTERN.test(name.replace(/\.json$/u, '')))
    .filter((name) => name.endsWith('.json'))
    .map((name) => {
      const path = resolve(directory, name)
      if (lstatSync(path).isSymbolicLink()) {
        throw new BeatReviewRequestError('评测报告路径无效', 500)
      }
      return JSON.parse(readFileSync(path, 'utf8')) as BeatEvaluationReport
    })
    .sort((left, right) => right.generatedAt.localeCompare(left.generatedAt))
}
