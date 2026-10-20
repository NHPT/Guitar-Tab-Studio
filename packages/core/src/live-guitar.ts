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

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
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

export function detectFundamentalFrequency(
  input: Float32Array,
  sampleRate: number,
  minimumFrequency = 70,
  maximumFrequency = 1200,
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
    if (
      lag > minimumLag &&
      normalized[lag - 1] < 0.16 &&
      normalized[lag - 1] <= normalized[lag - 2] &&
      normalized[lag - 1] <= normalized[lag]
    ) {
      selectedLag = lag - 1
      selectedValue = normalized[selectedLag]
      break
    }
    if (normalized[lag] < selectedValue) {
      selectedLag = lag
      selectedValue = normalized[lag]
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
