import { useEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import type { StudioProject, VisualizationPoint, VisualizerMode } from './types'

interface AudioVisualizerProps {
  project: StudioProject
  position: number
  playbackPositionRef: MutableRefObject<number>
  playing: boolean
  mode: VisualizerMode
  analyserRef: MutableRefObject<AnalyserNode | null>
  onSeek: (seconds: number) => void
}

interface AudioFeature {
  energy: number
  pitch: number
  low: number
  mid: number
  high: number
  impact: number
  bounce: number
}

const HISTORY_SIZE = 720
const SILENCE_RMS = 0.004
const BOUNCE_GRAVITY = 11
const BOUNCE_CEILING = 0.94
const REACTOR_ROWS = 6

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function stableNoise(index: number, row: number, salt = 0): number {
  let value =
    Math.imul(index + 1, 374761393) ^
    Math.imul(row + 1, 668265263) ^
    Math.imul(salt + 1, -2048144789)
  value = Math.imul(value ^ (value >>> 13), 1274126177)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

function resizeCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const ratio = Math.min(2, window.devicePixelRatio || 1)
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio))
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  const context = canvas.getContext('2d')
  context?.setTransform(ratio, 0, 0, ratio, 0, 0)
  return context
}

function averageBand(
  frequency: Uint8Array,
  sampleRate: number,
  fftSize: number,
  minimumHz: number,
  maximumHz: number,
): number {
  const binHz = sampleRate / fftSize
  const start = Math.max(1, Math.floor(minimumHz / binHz))
  const end = Math.min(frequency.length - 1, Math.ceil(maximumHz / binHz))
  if (end <= start) return 0
  let total = 0
  for (let index = start; index <= end; index += 1) total += frequency[index]
  return total / (end - start + 1) / 255
}

function analyseAudio(
  samples: Uint8Array,
  frequency: Uint8Array,
  sampleRate: number,
  fftSize: number,
): AudioFeature {
  const rms = Math.sqrt(
    samples.reduce((total, value) => {
      const normalized = (value - 128) / 128
      return total + normalized * normalized
    }, 0) / Math.max(1, samples.length),
  )
  const energy = rms <= SILENCE_RMS ? 0 : clamp((rms - SILENCE_RMS) / 0.16)
  if (energy === 0) {
    return { energy: 0, pitch: 0.5, low: 0, mid: 0, high: 0, impact: 0, bounce: 0 }
  }

  const binHz = sampleRate / fftSize
  const minimumBin = Math.max(1, Math.floor(70 / binHz))
  const maximumBin = Math.min(frequency.length - 1, Math.ceil(2200 / binHz))
  let dominantBin = minimumBin
  let dominantLevel = 0
  for (let index = minimumBin; index <= maximumBin; index += 1) {
    if (frequency[index] > dominantLevel) {
      dominantLevel = frequency[index]
      dominantBin = index
    }
  }
  const dominantHz = dominantBin * binHz
  const pitch = clamp(Math.log2(dominantHz / 70) / Math.log2(2200 / 70))

  return {
    energy,
    pitch,
    low: averageBand(frequency, sampleRate, fftSize, 40, 250),
    mid: averageBand(frequency, sampleRate, fftSize, 250, 2000),
    high: averageBand(frequency, sampleRate, fftSize, 2000, 8000),
    impact: 0,
    bounce: 0,
  }
}

function seedHistory(
  duration: number,
  visualization: VisualizationPoint[] | undefined,
  tracks: StudioProject['tracks'],
): Array<AudioFeature | undefined> {
  const source: VisualizationPoint[] =
    visualization ??
    tracks
      .filter((track) => track.available && track.waveform.length > 0)
      .flatMap((track, trackIndex, tracks) =>
        trackIndex === 0
          ? track.waveform.map((_, index) => {
              const energy =
                tracks.reduce((total, candidate) => total + (candidate.waveform[index] ?? 0), 0) /
                tracks.length
              return {
                energy,
                pitch: 0.5,
                low: energy * 0.72,
                mid: energy * 0.48,
                high: energy * 0.24,
                impact: 0,
              }
            })
          : [],
      )
  if (source.length === 0) return Array.from({ length: HISTORY_SIZE })

  const history: Array<AudioFeature | undefined> = Array.from({ length: HISTORY_SIZE })
  let bounce = 0
  let velocity = 0
  let previousImpact = 0
  const step = duration / HISTORY_SIZE
  for (let index = 0; index < HISTORY_SIZE; index += 1) {
    const sourceIndex = Math.min(
      source.length - 1,
      Math.floor((index / HISTORY_SIZE) * source.length),
    )
    const point = source[sourceIndex]
    if (point.impact > 0.32 && previousImpact <= 0.32 && bounce <= 0.08) {
      velocity = Math.max(velocity, 2.3 + point.impact * 1.2 + point.pitch * 0.3)
    } else if (bounce <= 0.01 && point.energy > 0.18) {
      velocity = Math.max(velocity, 1.3 + point.energy * 0.7)
    }
    velocity -= BOUNCE_GRAVITY * step
    bounce += velocity * step
    if (bounce < 0) {
      bounce = 0
      velocity = Math.abs(velocity) * 0.58
      if (velocity < 0.4) velocity = 0
    } else if (bounce > BOUNCE_CEILING) {
      bounce = BOUNCE_CEILING
      velocity = Math.min(0, velocity)
    }
    history[index] = { ...point, bounce: clamp(bounce) }
    previousImpact = point.impact
  }
  return history
}

function historyX(index: number, width: number): number {
  return (index / (HISTORY_SIZE - 1)) * width
}

function drawGlowHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  current: AudioFeature,
): void {
  const baseline = height * 0.84
  const barWidth = Math.max(1, width / HISTORY_SIZE - 0.5)
  context.save()
  context.fillStyle = accent
  for (let index = 0; index <= currentIndex; index += 1) {
    const feature = history[index]
    if (!feature || feature.energy === 0) continue
    const pulse = 0.9 + current.energy * 0.22
    const barHeight =
      height * (0.12 + feature.pitch * 0.68) * (0.45 + feature.energy * 0.55) * pulse
    context.globalAlpha = index === currentIndex ? 1 : 0.58
    context.shadowColor = accent
    context.shadowBlur = index === currentIndex ? 8 : 0
    context.fillRect(historyX(index, width), baseline - barHeight, barWidth, barHeight)
  }
  context.restore()
}

function drawWaveHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  muted: string,
  current: AudioFeature,
): void {
  const center = height / 2
  const points = Array.from({ length: currentIndex + 1 }, (_, index) => ({
    index,
    feature: history[index],
  })).filter(
    (
      point,
    ): point is {
      index: number
      feature: AudioFeature
    } => Boolean(point.feature),
  )
  if (points.length === 0) return

  const amplitude = (feature: AudioFeature) =>
    feature.energy === 0
      ? 0
      : Math.max(
          1,
          feature.energy * height * 0.39 * (0.9 + current.energy * 0.16),
        )

  context.save()
  context.fillStyle = accent
  context.globalAlpha = 0.09
  context.beginPath()
  context.moveTo(historyX(points[0].index, width), center)
  points.forEach(({ index, feature }) => {
    context.lineTo(historyX(index, width), center - amplitude(feature))
  })
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const point = points[index]
    context.lineTo(
      historyX(point.index, width),
      center + amplitude(point.feature),
    )
  }
  context.closePath()
  context.fill()

  context.strokeStyle = muted
  context.globalAlpha = 0.34
  context.lineWidth = 1
  context.beginPath()
  context.moveTo(0, center)
  context.lineTo(historyX(currentIndex, width), center)
  context.stroke()

  context.strokeStyle = accent
  context.globalAlpha = 0.74
  context.lineWidth = 1.25
  context.lineJoin = 'round'
  for (const direction of [-1, 1]) {
    context.beginPath()
    points.forEach(({ index, feature }, pointIndex) => {
      const x = historyX(index, width)
      const y = center + direction * amplitude(feature)
      if (pointIndex === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    })
    context.stroke()
  }
  context.restore()
}

function drawEcgHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  muted: string,
): void {
  const center = height / 2
  context.save()
  context.strokeStyle = muted
  context.globalAlpha = 0.45
  context.beginPath()
  context.moveTo(0, center)
  context.lineTo(historyX(currentIndex, width), center)
  context.stroke()

  context.strokeStyle = accent
  context.globalAlpha = 0.9
  context.lineWidth = 1.5
  context.lineJoin = 'round'
  context.beginPath()
  let started = false
  for (let index = 0; index <= currentIndex; index += 1) {
    const feature = history[index]
    if (!feature) continue
    const y =
      feature.energy === 0
        ? center
        : center + (0.5 - feature.pitch) * height * 0.76
    const x = historyX(index, width)
    if (!started) {
      context.moveTo(x, y)
      started = true
    } else {
      context.lineTo(x, y)
    }
  }
  context.stroke()
  context.restore()
}

function drawSpectrumHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
): void {
  const thresholds = [0.58, 0.42, 0.24]
  const laneCenters = [height * 0.22, height * 0.5, height * 0.78]
  const levelFor = (feature: AudioFeature, lane: number) =>
    lane === 0 ? feature.low : lane === 1 ? feature.mid : feature.high
  const bitFor = (feature: AudioFeature, lane: number) =>
    levelFor(feature, lane) >= thresholds[lane] ||
    feature.impact >= 0.78 - lane * 0.08

  context.save()
  context.strokeStyle = accent
  context.lineWidth = 1.5
  context.lineJoin = 'miter'
  laneCenters.forEach((laneCenter, lane) => {
    const highY = laneCenter - height * 0.07
    const lowY = laneCenter + height * 0.07
    let previousState = false
    let started = false
    context.globalAlpha = 0.9 - lane * 0.2
    context.beginPath()
    for (let index = 0; index <= currentIndex; index += 1) {
      const feature = history[index]
      if (!feature) continue
      const state = bitFor(feature, lane)
      const x = historyX(index, width)
      const y = state ? highY : lowY
      if (!started) {
        context.moveTo(x, y)
        previousState = state
        started = true
        continue
      }
      context.lineTo(x, previousState ? highY : lowY)
      if (state !== previousState) context.lineTo(x, y)
      previousState = state
    }
    context.stroke()

    const current = history[currentIndex]
    if (current) {
      const x = Math.max(0, historyX(currentIndex, width) - 3)
      const y = bitFor(current, lane) ? highY : lowY
      context.fillStyle = accent
      context.globalAlpha = 1 - lane * 0.18
      context.shadowColor = accent
      context.shadowBlur = 4
      context.fillRect(x, y - 2.5, 3, 5)
    }
  })
  context.restore()
}

function drawRibbonHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  current: AudioFeature,
): void {
  const center = height / 2
  const points = Array.from({ length: currentIndex + 1 }, (_, index) => ({
    index,
    feature: history[index],
  })).filter(
    (
      point,
    ): point is {
      index: number
      feature: AudioFeature
    } => Boolean(point.feature),
  )
  if (points.length === 0) return

  const breath = clamp(
    0.22 + current.energy * 0.92 + current.impact * 0.48,
    0.22,
    1.45,
  )
  const amplitude = (feature: AudioFeature, scale: number) =>
    (1.5 + feature.energy * height * 0.22) * breath * scale
  const drawBand = (scale: number, alpha: number, shadowBlur = 0) => {
    context.globalAlpha = alpha * (0.72 + breath * 0.28)
    context.shadowColor = accent
    context.shadowBlur = shadowBlur
    context.beginPath()
    context.moveTo(historyX(points[0].index, width), center)
    points.forEach(({ index, feature }) => {
      context.lineTo(
        historyX(index, width),
        center - amplitude(feature, scale),
      )
    })
    for (let index = points.length - 1; index >= 0; index -= 1) {
      const point = points[index]
      context.lineTo(
        historyX(point.index, width),
        center + amplitude(point.feature, scale),
      )
    }
    context.closePath()
    context.fill()
  }

  context.save()
  context.fillStyle = accent
  drawBand(1.35, 0.06, 14 * breath)
  drawBand(0.86, 0.2, 6 * breath)
  drawBand(0.34, 0.58)
  context.strokeStyle = accent
  context.globalAlpha = 0.38 + current.energy * 0.28
  context.lineWidth = 1
  context.beginPath()
  context.moveTo(0, center)
  context.lineTo(historyX(currentIndex, width), center)
  context.stroke()
  context.restore()
}

function drawCurrentDot(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  feature: AudioFeature | undefined,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  muted: string,
): void {
  const x = historyX(currentIndex, width)
  const ground = height * 0.82
  context.save()
  context.strokeStyle = muted
  context.globalAlpha = 0.38
  context.beginPath()
  context.moveTo(0, ground)
  context.lineTo(x, ground)
  context.stroke()

  context.strokeStyle = accent
  context.globalAlpha = 0.42
  context.lineWidth = 2.2
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.beginPath()
  let started = false
  for (let index = 0; index <= currentIndex; index += 1) {
    const point = history[index]
    if (!point) continue
    const pointX = historyX(index, width)
    const pointY = ground - point.bounce * height * 0.68
    if (!started) {
      context.moveTo(pointX, pointY)
      started = true
    } else {
      context.lineTo(pointX, pointY)
    }
  }
  context.stroke()

  const y = ground - (feature?.bounce ?? 0) * height * 0.68
  context.fillStyle = accent
  context.globalAlpha = feature?.energy ? 1 : 0.42
  context.shadowColor = accent
  context.shadowBlur = feature?.energy ? 9 : 0
  context.beginPath()
  context.arc(x, y, feature?.energy ? 5.2 : 3.4, 0, Math.PI * 2)
  context.fill()
  context.restore()
}

function drawReactorHistory(
  context: CanvasRenderingContext2D,
  history: Array<AudioFeature | undefined>,
  currentIndex: number,
  width: number,
  height: number,
  accent: string,
  muted: string,
): void {
  const current = history[currentIndex]
  if (!current) return

  const playedWidth = historyX(currentIndex, width)
  const rowCenters = Array.from(
    { length: REACTOR_ROWS },
    (_, row) => 9 + (row * (height - 18)) / (REACTOR_ROWS - 1),
  )
  const thresholds = [0.36, 0.34, 0.32, 0.3, 0.27, 0.33]
  const cellGap = 10
  const levelsFor = (feature: AudioFeature) => [
    Math.max(feature.low, feature.impact * 0.72),
    Math.max((feature.low + feature.mid) / 2, feature.impact * 0.64),
    Math.max(feature.mid, feature.impact * 0.58),
    Math.max((feature.mid + feature.high) / 2, feature.impact * 0.52),
    Math.max(feature.high, feature.impact * 0.46),
    Math.max(feature.energy, feature.impact * 0.4),
  ]
  const liveLevels = levelsFor(current)
  const motionPhase = Math.floor(currentIndex / 3) + Math.round(current.impact * 7)

  context.save()
  context.lineWidth = 1
  for (let x = 5; x <= playedWidth - 4; x += cellGap) {
    const index = Math.min(
      currentIndex,
      Math.floor((x / Math.max(1, width)) * (HISTORY_SIZE - 1)),
    )
    const feature = history[index]
    if (!feature) continue
    const levels = levelsFor(feature)
    const isCurrentBank = x >= playedWidth - cellGap * 5
    levels.forEach((level, row) => {
      const liveLevel = liveLevels[row]
      const cellNoise = stableNoise(index, row)
      const fillNoise = stableNoise(index, row, 1)
      const motionNoise = stableNoise(motionPhase, row, 2)
      const rowPulse = clamp(
        0.5 +
          current.energy * 0.42 +
          liveLevel * 0.5 +
          current.impact * 0.22 +
          (motionNoise - 0.5) * 0.32,
        0.44,
        1.68,
      )
      const size = clamp(
        (2.7 + level * 4.1) * rowPulse * (0.8 + cellNoise * 0.42),
        2.2,
        9.4,
      )
      const y =
        rowCenters[row] +
        (0.5 - level) * 2.8 +
        (0.5 - liveLevel) * 6 +
        current.impact * (row % 2 === 0 ? -2 : 2) +
        (cellNoise - 0.5) * 3.4
      const activation =
        level * 0.68 +
        feature.energy * 0.26 +
        feature.impact * 0.38 +
        (fillNoise - 0.5) * 0.22
      const filled = activation >= thresholds[row]
      const left = x + (cellNoise - 0.5) * 2.2 - size / 2
      const top = y - size / 2

      context.shadowColor = accent
      context.shadowBlur = isCurrentBank ? 7 : 0
      if (filled) {
        context.fillStyle = accent
        context.globalAlpha = isCurrentBank ? 1 : 0.42 + activation * 0.32
        context.fillRect(left, top, size, size)
      } else {
        context.fillStyle = accent
        context.globalAlpha = isCurrentBank ? 0.18 : 0.07 + activation * 0.12
        context.fillRect(left, top, size, size)
        context.strokeStyle = isCurrentBank ? accent : muted
        context.globalAlpha = isCurrentBank ? 0.78 : 0.42
        context.strokeRect(left, top, size, size)
      }
    })
  }
  context.restore()
}

export function AudioVisualizer({
  project,
  position,
  playbackPositionRef,
  playing,
  mode,
  analyserRef,
  onSeek,
}: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cursorRef = useRef<HTMLSpanElement>(null)
  const seededHistory = useMemo(
    () => seedHistory(project.duration, project.visualization, project.tracks),
    [project.duration, project.tracks, project.visualization],
  )
  const historyRef = useRef<Array<AudioFeature | undefined>>(seededHistory)

  useEffect(() => {
    historyRef.current = seededHistory
  }, [seededHistory])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let animation = 0
    let frame = 0
    let previousEnergy = 0
    let previousImpact = 0
    let bounce = 0
    let velocity = 0
    let previousTime = performance.now()
    const samples = new Uint8Array(2048)
    const frequency = new Uint8Array(1024)

    const draw = (now: number) => {
      const context = resizeCanvas(canvas)
      if (!context) return
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const styles = getComputedStyle(canvas)
      const accent = styles.getPropertyValue('--accent-strong').trim() || '#b43c30'
      const muted = styles.getPropertyValue('--border-strong').trim() || '#bfc1be'
      const analyser = analyserRef.current
      const progress = clamp(playbackPositionRef.current / project.duration)
      const currentIndex = Math.min(
        HISTORY_SIZE - 1,
        Math.max(0, Math.floor(progress * (HISTORY_SIZE - 1))),
      )

      let feature: AudioFeature = {
        energy: 0,
        pitch: 0.5,
        low: 0,
        mid: 0,
        high: 0,
        impact: 0,
        bounce: 0,
      }
      let signalSource = 'silence'
      if (playing && analyser) {
        analyser.getByteTimeDomainData(samples)
        analyser.getByteFrequencyData(frequency)
        feature = analyseAudio(samples, frequency, analyser.context.sampleRate, analyser.fftSize)
        signalSource = 'audio'
      }

      if (playing) {
        const elapsed = clamp((now - previousTime) / 1000, 0, 0.05)
        feature.impact = clamp(
          Math.max(0, feature.energy - previousEnergy) * 4.2 + feature.low * 0.45,
        )
        if (feature.impact > 0.3 && previousImpact <= 0.3 && bounce <= 0.08) {
          velocity = Math.max(
            velocity,
            2.3 + feature.impact * 1.2 + feature.pitch * 0.3,
          )
        } else if (bounce <= 0.01 && feature.energy > 0.16) {
          velocity = Math.max(velocity, 1.3 + feature.energy * 0.7)
        }
        velocity -= BOUNCE_GRAVITY * elapsed
        bounce += velocity * elapsed
        if (bounce < 0) {
          bounce = 0
          velocity = Math.abs(velocity) * 0.58
          if (velocity < 0.4) velocity = 0
        } else if (bounce > BOUNCE_CEILING) {
          bounce = BOUNCE_CEILING
          velocity = Math.min(0, velocity)
        }
        feature.bounce = clamp(bounce)
        historyRef.current[currentIndex] = feature
        previousEnergy = feature.energy
        previousImpact = feature.impact
        previousTime = now
      }
      const currentFeature = historyRef.current[currentIndex] ?? feature
      context.clearRect(0, 0, width, height)

      if (mode === 'glow') {
        drawGlowHistory(
          context,
          historyRef.current,
          currentIndex,
          width,
          height,
          accent,
          currentFeature,
        )
      } else if (mode === 'wave') {
        drawWaveHistory(
          context,
          historyRef.current,
          currentIndex,
          width,
          height,
          accent,
          muted,
          currentFeature,
        )
      } else if (mode === 'ecg') {
        drawEcgHistory(
          context,
          historyRef.current,
          currentIndex,
          width,
          height,
          accent,
          muted,
        )
      } else if (mode === 'spectrum') {
        drawSpectrumHistory(context, historyRef.current, currentIndex, width, height, accent)
      } else if (mode === 'ribbon') {
        drawRibbonHistory(
          context,
          historyRef.current,
          currentIndex,
          width,
          height,
          accent,
          currentFeature,
        )
      } else if (mode === 'reactor') {
        drawReactorHistory(
          context,
          historyRef.current,
          currentIndex,
          width,
          height,
          accent,
          muted,
        )
      } else {
        drawCurrentDot(
          context,
          historyRef.current,
          historyRef.current[currentIndex],
          currentIndex,
          width,
          height,
          accent,
          muted,
        )
      }

      if (cursorRef.current) cursorRef.current.style.left = `${progress * 100}%`
      canvas.dataset.frame = String(frame)
      canvas.dataset.signalSource = signalSource
      canvas.dataset.energy = feature.energy.toFixed(4)
      canvas.dataset.pitch = feature.pitch.toFixed(4)
      canvas.dataset.impact = feature.impact.toFixed(4)
      canvas.dataset.motionScope = 'current-only'
      canvas.dataset.futureVisible = 'false'
      canvas.dataset.reactorRows = mode === 'reactor' ? String(REACTOR_ROWS) : '0'
      frame += 1
      animation = requestAnimationFrame(draw)
    }

    animation = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animation)
  }, [analyserRef, mode, playbackPositionRef, playing, project.duration])

  const progress = clamp(position / project.duration) * 100

  return (
    <button
      className={`waveform visualizer-${mode}${playing ? ' is-playing' : ''}`}
      type="button"
      aria-label="拖动播放位置"
      onClick={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect()
        onSeek(((event.clientX - bounds.left) / bounds.width) * project.duration)
      }}
    >
      <canvas ref={canvasRef} className="visualizer-canvas" aria-hidden="true" />
      {mode === 'ecg' ? (
        <span ref={cursorRef} className="waveform-cursor" style={{ left: `${progress}%` }} />
      ) : null}
    </button>
  )
}
