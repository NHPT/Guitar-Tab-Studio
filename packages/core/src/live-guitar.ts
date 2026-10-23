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

export type TunerInstrumentId = 'guitar' | 'bass' | 'ukulele'

export interface TunerString {
  midi: number
  note: string
  string: number
}

export interface TunerInstrument {
  id: TunerInstrumentId
  label: string
  strings: readonly TunerString[]
}

export const TUNER_INSTRUMENTS: readonly TunerInstrument[] = [
  {
    id: 'guitar',
    label: '吉他',
    strings: [
      { midi: 40, note: 'E2', string: 6 },
      { midi: 45, note: 'A2', string: 5 },
      { midi: 50, note: 'D3', string: 4 },
      { midi: 55, note: 'G3', string: 3 },
      { midi: 59, note: 'B3', string: 2 },
      { midi: 64, note: 'E4', string: 1 },
    ],
  },
  {
    id: 'bass',
    label: '贝斯',
    strings: [
      { midi: 28, note: 'E1', string: 4 },
      { midi: 33, note: 'A1', string: 3 },
      { midi: 38, note: 'D2', string: 2 },
      { midi: 43, note: 'G2', string: 1 },
    ],
  },
  {
    id: 'ukulele',
    label: '尤克里里',
    strings: [
      { midi: 67, note: 'G4', string: 4 },
      { midi: 60, note: 'C4', string: 3 },
      { midi: 64, note: 'E4', string: 2 },
      { midi: 69, note: 'A4', string: 1 },
    ],
  },
]

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

function frequencyDistanceCents(left: number, right: number): number {
  return 1200 * Math.log2(left / right)
}

export function midiFrequency(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12)
}

export function tunerInstrument(
  instrumentId: TunerInstrumentId,
): TunerInstrument {
  return (
    TUNER_INSTRUMENTS.find(
      (instrument) => instrument.id === instrumentId,
    ) ?? TUNER_INSTRUMENTS[0]
  )
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
  maximumDistanceCents = 420,
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

export function tunerReadingForNearestTarget(
  reading: TunerReading,
  targetMidis: readonly number[],
  maximumDistanceCents = 250,
): TunerReading | null {
  const exactMidi = 69 + 12 * Math.log2(reading.frequency / 440)
  const targetMidi = targetMidis
    .filter(Number.isInteger)
    .map((midi) => ({
      midi,
      distance: Math.abs((exactMidi - midi) * 100),
    }))
    .sort((left, right) => left.distance - right.distance)[0]
  if (!targetMidi || targetMidi.distance > maximumDistanceCents) return null
  return {
    ...reading,
    midi: targetMidi.midi,
    note: noteNames[(targetMidi.midi + 120) % 12],
    octave: Math.floor(targetMidi.midi / 12) - 1,
    cents: clamp((exactMidi - targetMidi.midi) * 100, -50, 50),
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
  const recent = selected.slice(-5)
  const medianFrequency = median(recent.map((reading) => reading.frequency))
  const deviations = recent.map((reading) =>
    Math.abs(frequencyDistanceCents(reading.frequency, medianFrequency)),
  )
  const deviationLimit = Math.max(7, median(deviations) * 3.5)
  const latestIsOutlier =
    recent.length >= 4 &&
    Math.abs(frequencyDistanceCents(latest.frequency, medianFrequency)) >
      deviationLimit
  const measuredFrequency = latestIsOutlier
    ? medianFrequency
    : latest.frequency
  const frequency =
    previous?.midi === selectedMidi
      ? (() => {
          const delta = Math.abs(
            frequencyDistanceCents(measuredFrequency, previous.frequency),
          )
          const trend = selected.slice(-3)
          const trendSteps = trend.slice(1).map((reading, index) =>
            frequencyDistanceCents(
              reading.frequency,
              trend[index].frequency,
            ),
          )
          const coherentTrend =
            trendSteps.length === 2 &&
            trendSteps.every((step) => Math.abs(step) <= 12) &&
            Math.sign(trendSteps[0]) === Math.sign(trendSteps[1])
          const response = latestIsOutlier
            ? 0.24
            : coherentTrend
              ? 0.82
              : delta <= 1.5
                ? 0.48
                : delta <= 5
                  ? 0.68
                  : 0.78
          return (
            previous.frequency *
            (measuredFrequency / previous.frequency) ** response
          )
        })()
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
  minimumLevel = 0.006,
  preferredToleranceCents = 70,
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
    const sourceIndex = index * decimation
    const value =
      decimation === 1
        ? input[sourceIndex]
        : (input[sourceIndex] + input[sourceIndex + 1]) * 0.5
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
  if (level < minimumLevel) return null

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
  const localMinima: Array<{ lag: number; value: number }> = []
  for (let lag = minimumLag + 1; lag < maximumLag; lag += 1) {
    if (
      normalized[lag] <= normalized[lag - 1] &&
      normalized[lag] <= normalized[lag + 1]
    ) {
      localMinima.push({ lag, value: normalized[lag] })
      if (selectedLag < 0 && normalized[lag] < 0.16) {
        selectedLag = lag
        selectedValue = normalized[lag]
      }
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
  const usablePreferences = preferredFrequencies.filter(
    (frequency) =>
      Number.isFinite(frequency) &&
      frequency >= minimumFrequency &&
      frequency <= maximumFrequency,
  )
  if (selectedLag >= minimumLag && usablePreferences.length > 0) {
    const spectralRatios = new Map<number, number>()
    const spectralRatio = (lag: number) => {
      const cached = spectralRatios.get(lag)
      if (cached !== undefined) return cached
      let real = 0
      let imaginary = 0
      let weightSum = 0
      for (let index = 0; index < samples.length; index += 1) {
        const weight =
          0.5 -
          0.5 * Math.cos((2 * Math.PI * index) / (samples.length - 1))
        const phase = (2 * Math.PI * index) / lag
        real += samples[index] * weight * Math.cos(phase)
        imaginary -= samples[index] * weight * Math.sin(phase)
        weightSum += weight
      }
      const ratio =
        (2 * Math.hypot(real, imaginary)) /
        Math.max(weightSum * level, 1e-9)
      spectralRatios.set(lag, ratio)
      return ratio
    }
    const targetDistance = (lag: number) => {
      const frequency = effectiveRate / lag
      return Math.min(
        ...usablePreferences.map((preferred) =>
          Math.abs(frequencyDistanceCents(frequency, preferred)),
        ),
      )
    }
    const selectedDistance = targetDistance(selectedLag)
    const harmonicCorrection = localMinima
      .filter(
        (candidate) =>
          candidate.lag > selectedLag &&
          candidate.value <= selectedValue * 0.65 &&
          spectralRatio(candidate.lag) >= 0.02 &&
          targetDistance(candidate.lag) <= preferredToleranceCents &&
          (targetDistance(candidate.lag) + 60 < selectedDistance ||
            candidate.value <= selectedValue * 0.2),
      )
      .sort(
        (left, right) =>
          targetDistance(left.lag) - targetDistance(right.lag) ||
          left.value - right.value,
      )[0]
    if (harmonicCorrection) {
      selectedLag = harmonicCorrection.lag
      selectedValue = harmonicCorrection.value
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

export function detectTunerFrequency(
  input: Float32Array,
  sampleRate: number,
  targetMidis: readonly number[],
): DetectedPitch | null {
  const targetFrequencies = targetMidis
    .filter(Number.isInteger)
    .map(midiFrequency)
  if (targetFrequencies.length === 0) return null
  const rangeMargin = 2 ** (480 / 1200)
  return detectFundamentalFrequency(
    input,
    sampleRate,
    Math.min(...targetFrequencies) / rangeMargin,
    Math.max(...targetFrequencies) * rangeMargin,
    targetFrequencies,
    0.00018,
    420,
  )
}

export function detectStandardGuitarFrequency(
  input: Float32Array,
  sampleRate: number,
): DetectedPitch | null {
  return detectTunerFrequency(input, sampleRate, openStringMidi)
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
