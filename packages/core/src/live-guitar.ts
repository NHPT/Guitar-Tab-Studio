const noteNames = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
] as const

const openStringMidi = [64, 59, 55, 50, 45, 40] as const

export interface DetectedPitch {
  frequency: number
  confidence: number
  level: number
}

export interface TunerReading extends DetectedPitch {
  midi: number
  note: string
  octave: number
  cents: number
}

export interface GuitarPosition {
  string: 1 | 2 | 3 | 4 | 5 | 6
  fret: number
}

export interface TunerStabilityOptions {
  minimumConfidence?: number
  minimumReadings?: number
  switchReadings?: number
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

export function midiFrequency(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12)
}

export function tunerReading(
  detection: DetectedPitch,
): TunerReading | null {
  if (
    !Number.isFinite(detection.frequency) ||
    detection.frequency <= 0 ||
    detection.confidence < 0 ||
    detection.confidence > 1
  ) {
    return null
  }
  const exactMidi = 69 + 12 * Math.log2(detection.frequency / 440)
  const midi = Math.round(exactMidi)
  return {
    ...detection,
    midi,
    note: noteNames[(midi + 120) % 12],
    octave: Math.floor(midi / 12) - 1,
    cents: clamp((exactMidi - midi) * 100, -50, 50),
  }
}

export function tunerReadingForTarget(
  reading: TunerReading,
  targetMidi: number,
  maximumDistanceCents = 80,
): TunerReading | null {
  if (!Number.isInteger(targetMidi)) return null
  const targetFrequency = midiFrequency(targetMidi)
  let bestFrequency = reading.frequency
  let bestDistance = Math.abs(
    1200 * Math.log2(bestFrequency / targetFrequency),
  )
  for (let harmonic = 2; harmonic <= 6; harmonic += 1) {
    const candidate = reading.frequency / harmonic
    const distance = Math.abs(
      1200 * Math.log2(candidate / targetFrequency),
    )
    if (distance < bestDistance) {
      bestDistance = distance
      bestFrequency = candidate
    }
  }
  if (bestDistance > maximumDistanceCents) return null
  const exactMidi = 69 + 12 * Math.log2(bestFrequency / 440)
  return {
    ...reading,
    frequency: bestFrequency,
    midi: targetMidi,
    note: noteNames[(targetMidi + 120) % 12],
    octave: Math.floor(targetMidi / 12) - 1,
    cents: clamp((exactMidi - targetMidi) * 100, -50, 50),
  }
}

export function stabilizeTunerReadings(
  readings: readonly TunerReading[],
  previous?: TunerReading | null,
  options: TunerStabilityOptions = {},
): TunerReading | null {
  const minimumConfidence = options.minimumConfidence ?? 0.72
  const minimumReadings = Math.max(1, options.minimumReadings ?? 3)
  const switchReadings = Math.max(
    minimumReadings,
    options.switchReadings ?? 4,
  )
  const usable = readings
    .filter(
      (reading) =>
        reading.confidence >= minimumConfidence &&
        Number.isFinite(reading.frequency) &&
        reading.frequency > 0,
    )
    .slice(-9)
  if (usable.length < minimumReadings) return previous ?? null

  const switchingTail = usable.slice(-switchReadings)
  const forcedMidi =
    switchingTail.length === switchReadings &&
    switchingTail.every(
      (reading) => reading.midi === switchingTail[0].midi,
    )
      ? switchingTail[0].midi
      : undefined
  const groups = new Map<number, TunerReading[]>()
  for (const reading of usable) {
    const group = groups.get(reading.midi) ?? []
    group.push(reading)
    groups.set(reading.midi, group)
  }
  const candidates = [...groups.entries()]
    .filter(([, group]) => group.length >= minimumReadings)
    .sort(([leftMidi, left], [rightMidi, right]) => {
      const score = (midi: number, group: readonly TunerReading[]) =>
        group.length * 2 +
        group.reduce((sum, reading) => sum + reading.confidence, 0) +
        (previous?.midi === midi ? 1.5 : 0)
      return score(rightMidi, right) - score(leftMidi, left)
    })
  const selectedMidi =
    forcedMidi ??
    candidates[0]?.[0] ??
    previous?.midi
  const selected = selectedMidi === undefined
    ? []
    : (groups.get(selectedMidi) ?? [])
  if (selected.length < minimumReadings) return previous ?? null

  const latest = selected.at(-1)!
  const measuredFrequency = median(
    selected.slice(-5).map((reading) => reading.frequency),
  )
  const frequency =
    previous?.midi === selectedMidi
      ? previous.frequency * 0.58 + measuredFrequency * 0.42
      : measuredFrequency
  return tunerReading({
    frequency,
    confidence: median(
      selected.slice(-5).map((reading) => reading.confidence),
    ),
    level: latest.level,
  })
}

export function detectFundamentalFrequency(
  input: Float32Array,
  sampleRate: number,
  minimumFrequency = 70,
  maximumFrequency = 1200,
  preferredFrequencies: readonly number[] = [],
): DetectedPitch | null {
  if (
    input.length < 512 ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0
  ) {
    return null
  }
  const decimation = sampleRate >= 32_000 ? 2 : 1
  const length = Math.min(
    Math.floor(input.length / decimation),
    4096,
  )
  const samples = new Float32Array(length)
  let mean = 0
  for (let index = 0; index < length; index += 1) {
    const value = input[index * decimation]
    samples[index] = value
    mean += value
  }
  mean /= length
  let energy = 0
  for (let index = 0; index < length; index += 1) {
    samples[index] -= mean
    energy += samples[index] ** 2
  }
  const level = Math.sqrt(energy / length)
  if (level < 0.006) return null

  const effectiveRate = sampleRate / decimation
  const minimumLag = Math.max(
    2,
    Math.floor(effectiveRate / maximumFrequency),
  )
  const maximumLag = Math.min(
    length - 2,
    Math.ceil(effectiveRate / minimumFrequency),
  )
  const difference = new Float64Array(maximumLag + 1)
  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    let sum = 0
    const count = length - lag
    for (let index = 0; index < count; index += 1) {
      const delta = samples[index] - samples[index + lag]
      sum += delta * delta
    }
    difference[lag] = sum
  }

  const normalized = new Float64Array(maximumLag + 1)
  let cumulative = 0
  let selectedLag = -1
  let selectedValue = Number.POSITIVE_INFINITY
  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    cumulative += difference[lag]
    normalized[lag] =
      cumulative > 0
        ? (difference[lag] * (lag - minimumLag + 1)) / cumulative
        : 1
  }
  if (preferredFrequencies.length > 0) {
    const toleranceRatio = 2 ** (70 / 1200)
    const candidates: Array<{ lag: number; value: number }> = []
    for (const frequency of preferredFrequencies) {
      if (
        !Number.isFinite(frequency) ||
        frequency < minimumFrequency ||
        frequency > maximumFrequency
      ) {
        continue
      }
      const start = Math.max(
        minimumLag,
        Math.floor(effectiveRate / (frequency * toleranceRatio)),
      )
      const end = Math.min(
        maximumLag,
        Math.ceil(effectiveRate / (frequency / toleranceRatio)),
      )
      let candidateLag = -1
      let candidateValue = Number.POSITIVE_INFINITY
      for (let lag = start; lag <= end; lag += 1) {
        if (normalized[lag] < candidateValue) {
          candidateLag = lag
          candidateValue = normalized[lag]
        }
      }
      if (candidateLag >= minimumLag) {
        candidates.push({ lag: candidateLag, value: candidateValue })
      }
    }
    const reliable = candidates
      .filter((candidate) => candidate.value <= 0.03)
      .sort((left, right) => left.lag - right.lag)
    const fallback = [...candidates].sort(
      (left, right) => left.value - right.value,
    )
    const selected = reliable[0] ?? fallback[0]
    if (selected) {
      selectedLag = selected.lag
      selectedValue = selected.value
    }
  } else {
    for (let lag = minimumLag + 1; lag < maximumLag; lag += 1) {
      if (
        normalized[lag] < 0.16 &&
        normalized[lag] <= normalized[lag - 1] &&
        normalized[lag] <= normalized[lag + 1]
      ) {
        selectedLag = lag
        selectedValue = normalized[lag]
        break
      }
    }
    if (selectedLag < 0) {
      for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
        if (normalized[lag] < selectedValue) {
          selectedLag = lag
          selectedValue = normalized[lag]
        }
      }
    }
  }
  if (selectedLag < minimumLag || selectedValue > 0.32) return null

  const left = normalized[Math.max(minimumLag, selectedLag - 1)]
  const center = normalized[selectedLag]
  const right = normalized[Math.min(maximumLag, selectedLag + 1)]
  const denominator = left - 2 * center + right
  const correction =
    Math.abs(denominator) > 1e-9
      ? clamp(0.5 * (left - right) / denominator, -0.5, 0.5)
      : 0
  const lag = selectedLag + correction
  return {
    frequency: effectiveRate / lag,
    confidence: clamp(1 - selectedValue, 0, 1),
    level,
  }
}

export function detectStandardGuitarFrequency(
  input: Float32Array,
  sampleRate: number,
): DetectedPitch | null {
  return detectFundamentalFrequency(
    input,
    sampleRate,
    70,
    400,
    openStringMidi.map(midiFrequency),
  )
}

export function guitarPositionForPitch(
  midi: number,
  previous?: GuitarPosition,
): GuitarPosition | null {
  const candidates = openStringMidi.flatMap((openPitch, index) => {
    const fret = midi - openPitch
    return Number.isInteger(fret) && fret >= 0 && fret <= 24
      ? [{
          string: (index + 1) as GuitarPosition['string'],
          fret,
        }]
      : []
  })
  if (candidates.length === 0) return null
  return candidates.reduce((best, candidate) => {
    const cost = (position: GuitarPosition) =>
      position.fret * 0.2 +
      Math.max(0, position.fret - 12) * 1.8 +
      (position.fret === 0 ? -1.5 : 0) +
      (previous
        ? Math.abs(position.fret - previous.fret) * 0.7 +
          Math.abs(position.string - previous.string) * 0.2
        : 0)
    return cost(candidate) < cost(best) ? candidate : best
  })
}
