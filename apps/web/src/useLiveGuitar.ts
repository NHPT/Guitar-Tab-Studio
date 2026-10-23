import {
  detectFundamentalFrequency,
  detectTunerFrequency,
  guitarPositionForPitch,
  stabilizeTunerReadings,
  tunerInstrument as getTunerInstrument,
  tunerReading,
  type GuitarPosition,
  type TunerInstrumentId,
  type TunerReading,
} from '@gts/core'
import { useCallback, useEffect, useRef, useState } from 'react'

export type LiveInputMode = 'tuner' | 'transcription'
export type LiveInputStatus =
  | 'idle'
  | 'requesting'
  | 'listening'
  | 'error'

export interface LiveTabNote {
  id: string
  at: number
  duration: number
  midi: number
  note: string
  octave: number
  cents: number
  string: GuitarPosition['string']
  fret: number
  confidence: number
}

interface ActiveNote {
  note: LiveTabNote
  startedAt: number
  lastSeenAt: number
}

export interface LiveGuitarController {
  status: LiveInputStatus
  mode: LiveInputMode | null
  tunerInstrument: TunerInstrumentId
  reading: TunerReading | null
  level: number
  notes: LiveTabNote[]
  error: string
  start: (mode: LiveInputMode) => Promise<void>
  setTunerInstrument: (instrument: TunerInstrumentId) => void
  stop: () => void
  clear: () => void
}

function audioLevel(samples: Float32Array): number {
  let mean = 0
  for (const sample of samples) mean += sample
  mean /= samples.length
  let energy = 0
  for (const sample of samples) energy += (sample - mean) ** 2
  return Math.sqrt(energy / samples.length)
}

export function useLiveGuitar(): LiveGuitarController {
  const [status, setStatus] = useState<LiveInputStatus>('idle')
  const [mode, setMode] = useState<LiveInputMode | null>(null)
  const [tunerInstrument, setTunerInstrumentState] =
    useState<TunerInstrumentId>('guitar')
  const [reading, setReading] = useState<TunerReading | null>(null)
  const [level, setLevel] = useState(0)
  const [notes, setNotes] = useState<LiveTabNote[]>([])
  const [error, setError] = useState('')
  const contextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const frameRef = useRef<number | undefined>(undefined)
  const startedAtRef = useRef(0)
  const lastUiUpdateRef = useRef(0)
  const previousLevelRef = useRef(0)
  const lastAttackRef = useRef(0)
  const candidateMidiRef = useRef<number | undefined>(undefined)
  const candidateFramesRef = useRef(0)
  const activeNoteRef = useRef<ActiveNote | undefined>(undefined)
  const previousPositionRef = useRef<GuitarPosition | undefined>(undefined)
  const tunerHistoryRef = useRef<TunerReading[]>([])
  const stableTunerReadingRef = useRef<TunerReading | null>(null)
  const tunerInstrumentRef = useRef<TunerInstrumentId>('guitar')
  const tunerGateOpenRef = useRef(false)
  const tunerNoiseFloorRef = useRef(0.0002)
  const tunerLastSoundAtRef = useRef(0)
  const smoothedTunerLevelRef = useRef(0)

  const publishActiveNote = useCallback((active: ActiveNote) => {
    const note = {
      ...active.note,
      duration: Math.max(0.06, active.lastSeenAt - active.startedAt),
    }
    setNotes((current) => {
      const existing = current.findIndex(
        (candidate) => candidate.id === note.id,
      )
      if (existing < 0) return [...current, note].slice(-500)
      const next = [...current]
      next[existing] = note
      return next
    })
  }, [])

  const finishActiveNote = useCallback(
    (at?: number) => {
      const active = activeNoteRef.current
      if (!active) return
      if (at !== undefined) active.lastSeenAt = Math.max(active.startedAt, at)
      publishActiveNote(active)
      previousPositionRef.current = {
        string: active.note.string,
        fret: active.note.fret,
      }
      activeNoteRef.current = undefined
    },
    [publishActiveNote],
  )

  const stop = useCallback(() => {
    if (frameRef.current !== undefined) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = undefined
    }
    finishActiveNote(contextRef.current?.currentTime)
    sourceRef.current?.disconnect()
    sourceRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (contextRef.current) {
      void contextRef.current.close()
      contextRef.current = null
    }
    candidateMidiRef.current = undefined
    candidateFramesRef.current = 0
    tunerHistoryRef.current = []
    stableTunerReadingRef.current = null
    tunerGateOpenRef.current = false
    tunerNoiseFloorRef.current = 0.0002
    tunerLastSoundAtRef.current = 0
    smoothedTunerLevelRef.current = 0
    setReading(null)
    setLevel(0)
    setMode(null)
    setStatus('idle')
  }, [finishActiveNote])

  const start = useCallback(
    async (nextMode: LiveInputMode) => {
      stop()
      setStatus('requesting')
      setMode(nextMode)
      setError('')
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('当前运行环境不支持音频输入')
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: 1,
          },
          video: false,
        })
        const context = new AudioContext({ latencyHint: 'interactive' })
        await context.resume()
        const source = context.createMediaStreamSource(stream)
        const analyser = context.createAnalyser()
        analyser.fftSize = 8192
        analyser.smoothingTimeConstant = 0
        source.connect(analyser)
        contextRef.current = context
        streamRef.current = stream
        sourceRef.current = source
        startedAtRef.current = context.currentTime
        previousLevelRef.current = 0
        lastAttackRef.current = context.currentTime
        lastUiUpdateRef.current = 0
        setStatus('listening')

        const samples = new Float32Array(analyser.fftSize)
        const update = () => {
          if (contextRef.current !== context) return
          analyser.getFloatTimeDomainData(samples)
          const rawLevel = audioLevel(samples)
          const currentTuner = getTunerInstrument(
            tunerInstrumentRef.current,
          )
          const detection =
            nextMode === 'tuner'
              ? detectTunerFrequency(
                  samples,
                  context.sampleRate,
                  currentTuner.strings.map((string) => string.midi),
                )
              : detectFundamentalFrequency(samples, context.sampleRate)
          const now = context.currentTime
          const detectedReading = detection ? tunerReading(detection) : null
          let nextReading = detectedReading
          let nextLevel = detection?.level ?? 0
          if (nextMode === 'tuner') {
            const previousLevel = smoothedTunerLevelRef.current
            nextLevel =
              rawLevel >= previousLevel
                ? previousLevel * 0.28 + rawLevel * 0.72
                : previousLevel * 0.88 + rawLevel * 0.12
            smoothedTunerLevelRef.current =
              nextLevel < 0.00001 ? 0 : nextLevel
            const noiseFloor = tunerNoiseFloorRef.current
            const gateOpen = tunerGateOpenRef.current
            const attackFloor = Math.max(0.00075, noiseFloor * 3.2)
            const sustainFloor = Math.max(0.00016, noiseFloor * 1.15)
            const trackingFloor = gateOpen
              ? sustainFloor
              : attackFloor
            const minimumConfidence = gateOpen ? 0.54 : 0.68
            const reliableReading =
              detectedReading &&
              detectedReading.confidence >= minimumConfidence &&
              rawLevel >= trackingFloor
                ? detectedReading
                : null
            if (reliableReading) {
              tunerGateOpenRef.current = true
              tunerLastSoundAtRef.current = now
              tunerHistoryRef.current = [
                ...tunerHistoryRef.current,
                reliableReading,
              ].slice(-12)
              stableTunerReadingRef.current = stabilizeTunerReadings(
                tunerHistoryRef.current,
                stableTunerReadingRef.current,
                {
                  minimumConfidence: 0.54,
                  minimumReadings: 2,
                  switchReadings: 3,
                },
              )
            } else if (!tunerGateOpenRef.current) {
              tunerNoiseFloorRef.current = Math.max(
                0.00003,
                Math.min(
                  0.003,
                  noiseFloor * 0.985 + rawLevel * 0.015,
                ),
              )
            }
            if (tunerGateOpenRef.current) {
              if (now - tunerLastSoundAtRef.current >= 1.4) {
                tunerGateOpenRef.current = false
                tunerHistoryRef.current = []
                stableTunerReadingRef.current = null
              }
            }
            nextReading = stableTunerReadingRef.current
          }
          if (now - lastUiUpdateRef.current >= 0.05) {
            lastUiUpdateRef.current = now
            setReading(nextReading)
            setLevel(nextLevel)
          }

          if (nextMode === 'transcription') {
            if (nextReading && nextReading.confidence >= 0.72) {
              if (candidateMidiRef.current === nextReading.midi) {
                candidateFramesRef.current += 1
              } else {
                candidateMidiRef.current = nextReading.midi
                candidateFramesRef.current = 1
              }
              const attack =
                nextLevel >= 0.012 &&
                nextLevel > previousLevelRef.current * 1.55 &&
                now - lastAttackRef.current >= 0.09
              if (attack) lastAttackRef.current = now
              if (candidateFramesRef.current >= 2) {
                const active = activeNoteRef.current
                const changedPitch =
                  active && active.note.midi !== nextReading.midi
                const repeatedAttack =
                  active &&
                  active.note.midi === nextReading.midi &&
                  attack &&
                  now - active.startedAt >= 0.09
                if (changedPitch || repeatedAttack) finishActiveNote(now)
                if (!activeNoteRef.current) {
                  const position = guitarPositionForPitch(
                    nextReading.midi,
                    previousPositionRef.current,
                  )
                  if (position) {
                    const relativeAt = now - startedAtRef.current
                    activeNoteRef.current = {
                      startedAt: now,
                      lastSeenAt: now,
                      note: {
                        id: `live-${Math.round(now * 1000)}`,
                        at: relativeAt,
                        duration: 0.06,
                        midi: nextReading.midi,
                        note: nextReading.note,
                        octave: nextReading.octave,
                        cents: nextReading.cents,
                        string: position.string,
                        fret: position.fret,
                        confidence: nextReading.confidence,
                      },
                    }
                  }
                } else {
                  activeNoteRef.current.lastSeenAt = now
                  activeNoteRef.current.note.cents = nextReading.cents
                  activeNoteRef.current.note.confidence = Math.max(
                    activeNoteRef.current.note.confidence,
                    nextReading.confidence,
                  )
                  if (now - lastUiUpdateRef.current <= 0.06) {
                    publishActiveNote(activeNoteRef.current)
                  }
                }
              }
            } else {
              candidateFramesRef.current = 0
              if (
                activeNoteRef.current &&
                now - activeNoteRef.current.lastSeenAt >= 0.18
              ) {
                finishActiveNote(activeNoteRef.current.lastSeenAt)
              }
            }
          }
          previousLevelRef.current = nextLevel
          frameRef.current = requestAnimationFrame(update)
        }
        frameRef.current = requestAnimationFrame(update)
      } catch (cause) {
        stop()
        setStatus('error')
        setMode(nextMode)
        setError(
          cause instanceof DOMException && cause.name === 'NotAllowedError'
            ? '未获得麦克风权限，请在系统设置中允许音频输入'
            : cause instanceof Error
              ? cause.message
              : '无法启动音频输入',
        )
      }
    },
    [finishActiveNote, publishActiveNote, stop],
  )

  const setTunerInstrument = useCallback(
    (instrument: TunerInstrumentId) => {
      tunerInstrumentRef.current = instrument
      setTunerInstrumentState(instrument)
      tunerHistoryRef.current = []
      stableTunerReadingRef.current = null
      tunerGateOpenRef.current = false
      tunerLastSoundAtRef.current = 0
      setReading(null)
    },
    [],
  )

  const clear = useCallback(() => {
    activeNoteRef.current = undefined
    previousPositionRef.current = undefined
    setNotes([])
  }, [])

  useEffect(() => stop, [stop])

  return {
    status,
    mode,
    tunerInstrument,
    reading,
    level,
    notes,
    error,
    start,
    setTunerInstrument,
    stop,
    clear,
  }
}
