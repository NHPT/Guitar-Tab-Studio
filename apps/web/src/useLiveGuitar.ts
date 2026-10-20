import {
  detectFundamentalFrequency,
  guitarPositionForPitch,
  tunerReading,
  type GuitarPosition,
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
  reading: TunerReading | null
  level: number
  notes: LiveTabNote[]
  error: string
  start: (mode: LiveInputMode) => Promise<void>
  stop: () => void
  clear: () => void
}

export function useLiveGuitar(): LiveGuitarController {
  const [status, setStatus] = useState<LiveInputStatus>('idle')
  const [mode, setMode] = useState<LiveInputMode | null>(null)
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
        setStatus('listening')

        const samples = new Float32Array(analyser.fftSize)
        const update = () => {
          if (contextRef.current !== context) return
          analyser.getFloatTimeDomainData(samples)
          const detection = detectFundamentalFrequency(
            samples,
            context.sampleRate,
          )
          const now = context.currentTime
          const nextReading = detection ? tunerReading(detection) : null
          const nextLevel = detection?.level ?? 0
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

  const clear = useCallback(() => {
    activeNoteRef.current = undefined
    previousPositionRef.current = undefined
    setNotes([])
  }, [])

  useEffect(() => stop, [stop])

  return {
    status,
    mode,
    reading,
    level,
    notes,
    error,
    start,
    stop,
    clear,
  }
}
