import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import {
  PRACTICE_OUTPUT_GAIN,
  rhythmEventPosition,
} from '@gts/core'
import type {
  PercussionVoice,
  PracticeRhythm,
  RhythmKit,
} from '@gts/core'
import type {
  StudioProject,
  TabNote,
  TrackMix,
  TrackRole,
} from './types'

interface PracticeAudioOptions {
  project: StudioProject | null
  playing: boolean
  position: number
  playbackPositionRef: MutableRefObject<number>
  speed: number
  mixes: Record<string, TrackMix>
  rhythm: {
    enabled: boolean
    definition: PracticeRhythm
    kit: RhythmKit
    volume: number
  }
  standalone: {
    playing: boolean
    bpm: number
    definition: PracticeRhythm
    kit: RhythmKit
    volume: number
    mutedVoices: PercussionVoice[]
    onStep: (step: number) => void
  }
}

interface StartCountInOptions {
  bars: number
  speed: number
  beatDurations: number[]
  kit: Extract<RhythmKit, 'click' | 'wood'>
  volume: number
  onBeat: (bar: number, beat: number) => void
  onComplete: () => void
}

interface PracticeAudioState {
  analyserRef: MutableRefObject<AnalyserNode | null>
  activateAudio: () => Promise<void>
  seekAudio: (seconds: number) => void
  startCountIn: (options: StartCountInOptions) => Promise<void>
  cancelCountIn: () => void
}

const API_ROOT = import.meta.env.VITE_API_ROOT ?? 'http://127.0.0.1:8787'

function mediaUrl(url: string): string {
  return url.startsWith('/') ? `${API_ROOT}${url}` : url
}

function midiFrequency(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12)
}

const OPEN_STRING_MIDI: Record<TabNote['string'], number> = {
  1: 64,
  2: 59,
  3: 55,
  4: 50,
  5: 45,
  6: 40,
}

function playTone(
  context: AudioContext,
  frequency: number,
  volume: number,
  duration: number,
  type: OscillatorType,
  destination: AudioNode,
  at = context.currentTime,
): OscillatorNode {
  const oscillator = context.createOscillator()
  const gain = context.createGain()
  oscillator.type = type
  oscillator.frequency.value = frequency
  gain.gain.setValueAtTime(Math.max(0.0001, volume), at)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + duration)
  oscillator.connect(gain).connect(destination)
  oscillator.start(at)
  oscillator.stop(at + duration)
  return oscillator
}

function playNoise(
  context: AudioContext,
  volume: number,
  duration: number,
  destination: AudioNode,
  at = context.currentTime,
): AudioBufferSourceNode {
  const frameCount = Math.max(1, Math.floor(context.sampleRate * duration))
  const buffer = context.createBuffer(1, frameCount, context.sampleRate)
  const data = buffer.getChannelData(0)
  for (let index = 0; index < frameCount; index += 1) {
    data[index] = Math.random() * 2 - 1
  }

  const source = context.createBufferSource()
  const gain = context.createGain()
  source.buffer = buffer
  gain.gain.setValueAtTime(volume, at)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + duration)
  source.connect(gain).connect(destination)
  source.start(at)
  return source
}

const noiseBuffers = new WeakMap<AudioContext, AudioBuffer>()

function sharedNoiseBuffer(context: AudioContext): AudioBuffer {
  const existing = noiseBuffers.get(context)
  if (existing) return existing
  const buffer = context.createBuffer(1, context.sampleRate, context.sampleRate)
  const data = buffer.getChannelData(0)
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.random() * 2 - 1
  }
  noiseBuffers.set(context, buffer)
  return buffer
}

function scheduleNoiseBurst(
  context: AudioContext,
  at: number,
  duration: number,
  volume: number,
  destination: AudioNode,
  filterType: BiquadFilterType,
  frequency: number,
): AudioBufferSourceNode {
  const source = context.createBufferSource()
  const filter = context.createBiquadFilter()
  const gain = context.createGain()
  source.buffer = sharedNoiseBuffer(context)
  filter.type = filterType
  filter.frequency.value = frequency
  gain.gain.setValueAtTime(Math.max(0.0001, volume), at)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + duration)
  source.connect(filter).connect(gain).connect(destination)
  source.start(at)
  source.stop(at + duration)
  return source
}

function schedulePercussion(
  context: AudioContext,
  voice: PercussionVoice,
  kit: RhythmKit,
  velocity: number,
  at: number,
  destination: AudioNode,
): AudioScheduledSourceNode[] {
  const level = Math.max(0.0001, Math.min(1, velocity))
  if (voice === 'click-high' || voice === 'click-low') {
    const high = voice === 'click-high'
    if (kit === 'wood') {
      return [
        playTone(
          context,
          high ? 980 : 640,
          level * 0.9,
          0.11,
          'triangle',
          destination,
          at,
        ),
        playTone(
          context,
          high ? 1470 : 960,
          level * 0.32,
          0.055,
          'sine',
          destination,
          at + 0.004,
        ),
      ]
    }
    return [
      playTone(
        context,
        high ? 2100 : 1320,
        level,
        0.045,
        'square',
        destination,
        at,
      ),
    ]
  }
  if (voice === 'kick') {
    const profile = {
      acoustic: {
        start: 132,
        end: 46,
        duration: 0.21,
        level: 0.58,
        type: 'sine' as const,
      },
      electronic: {
        start: 196,
        end: 38,
        duration: 0.3,
        level: 0.72,
        type: 'sine' as const,
      },
      brush: {
        start: 102,
        end: 52,
        duration: 0.16,
        level: 0.38,
        type: 'triangle' as const,
      },
      wood: {
        start: 118,
        end: 72,
        duration: 0.12,
        level: 0.46,
        type: 'triangle' as const,
      },
    }[kit === 'click' ? 'acoustic' : kit]
    const oscillator = context.createOscillator()
    const gain = context.createGain()
    oscillator.type = profile.type
    oscillator.frequency.setValueAtTime(profile.start, at)
    oscillator.frequency.exponentialRampToValueAtTime(
      profile.end,
      at + profile.duration * 0.72,
    )
    gain.gain.setValueAtTime(level * profile.level, at)
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      at + profile.duration,
    )
    oscillator.connect(gain).connect(destination)
    oscillator.start(at)
    oscillator.stop(at + profile.duration + 0.01)
    const sources: AudioScheduledSourceNode[] = [oscillator]
    if (kit === 'electronic') {
      sources.push(
        playTone(
          context,
          2400,
          level * 0.18,
          0.018,
          'square',
          destination,
          at,
        ),
      )
    } else if (kit === 'brush') {
      sources.push(
        scheduleNoiseBurst(
          context,
          at,
          0.075,
          level * 0.11,
          destination,
          'lowpass',
          520,
        ),
      )
    } else if (kit === 'wood') {
      sources.push(
        playTone(
          context,
          330,
          level * 0.24,
          0.07,
          'square',
          destination,
          at,
        ),
      )
    }
    return sources
  }
  if (voice === 'snare' || voice === 'clap') {
    if (kit === 'wood') {
      return [
        playTone(
          context,
          voice === 'clap' ? 1380 : 920,
          level * 0.5,
          voice === 'clap' ? 0.045 : 0.075,
          'square',
          destination,
          at,
        ),
        playTone(
          context,
          voice === 'clap' ? 1880 : 1320,
          level * 0.22,
          0.035,
          'triangle',
          destination,
          at + 0.012,
        ),
      ]
    }
    const electronic = kit === 'electronic'
    const brushed = kit === 'brush'
    const sources: AudioScheduledSourceNode[] = [
      scheduleNoiseBurst(
        context,
        at,
        voice === 'clap'
          ? electronic
            ? 0.07
            : 0.1
          : brushed
            ? 0.24
            : electronic
              ? 0.095
              : 0.15,
        level * (brushed ? 0.24 : electronic ? 0.45 : 0.38),
        destination,
        electronic ? 'bandpass' : 'highpass',
        brushed ? 720 : electronic ? 2800 : 1450,
      ),
    ]
    if (voice === 'snare') {
      sources.push(
        playTone(
          context,
          electronic ? 245 : brushed ? 150 : 185,
          level * (electronic ? 0.22 : brushed ? 0.07 : 0.14),
          electronic ? 0.075 : 0.11,
          electronic ? 'square' : 'triangle',
          destination,
          at,
        ),
      )
    } else {
      sources.push(
        scheduleNoiseBurst(
          context,
          at + (electronic ? 0.026 : 0.018),
          electronic ? 0.045 : 0.06,
          level * (electronic ? 0.3 : 0.2),
          destination,
          electronic ? 'bandpass' : 'highpass',
          electronic ? 3600 : 1700,
        ),
      )
    }
    return sources
  }
  if (
    voice === 'closed-hat' ||
    voice === 'open-hat' ||
    voice === 'shaker' ||
    voice === 'crash'
  ) {
    if (kit === 'wood') {
      const frequency = {
        'closed-hat': 2100,
        'open-hat': 1640,
        shaker: 2520,
        crash: 1180,
      }[voice]
      return [
        playTone(
          context,
          frequency,
          level * (voice === 'crash' ? 0.34 : 0.24),
          voice === 'crash' ? 0.2 : voice === 'open-hat' ? 0.12 : 0.055,
          'triangle',
          destination,
          at,
        ),
      ]
    }
    const electronic = kit === 'electronic'
    const brushed = kit === 'brush'
    const duration =
      voice === 'crash'
        ? brushed
          ? 0.58
          : electronic
            ? 0.32
            : 0.46
        : voice === 'open-hat'
          ? brushed
            ? 0.28
            : electronic
              ? 0.13
              : 0.2
          : voice === 'shaker'
            ? brushed
              ? 0.1
              : 0.065
            : brushed
              ? 0.08
              : electronic
                ? 0.032
                : 0.05
    const frequency =
      voice === 'crash'
        ? brushed
          ? 1750
          : electronic
            ? 4200
            : 2800
        : voice === 'shaker'
          ? brushed
            ? 2500
            : 4200
          : electronic
            ? 8900
            : brushed
              ? 3200
              : 6500
    const sources: AudioScheduledSourceNode[] = [
      scheduleNoiseBurst(
        context,
        at,
        duration,
        level *
          (voice === 'crash'
            ? brushed
              ? 0.17
              : 0.25
            : voice === 'open-hat'
              ? brushed
                ? 0.12
                : 0.18
              : brushed
                ? 0.09
                : 0.13),
        destination,
        electronic ? 'bandpass' : 'highpass',
        frequency,
      ),
    ]
    if (electronic && voice !== 'shaker') {
      sources.push(
        playTone(
          context,
          voice === 'crash' ? 4200 : 7200,
          level * 0.07,
          Math.min(0.12, duration),
          'square',
          destination,
          at,
        ),
      )
    }
    return sources
  }
  const resolvedKit = kit === 'click' ? 'acoustic' : kit
  const frequency = {
    acoustic: {
      rim: 930,
      'tom-high': 176,
      'tom-low': 112,
      cowbell: 620,
    },
    electronic: {
      rim: 1580,
      'tom-high': 248,
      'tom-low': 138,
      cowbell: 880,
    },
    brush: {
      rim: 760,
      'tom-high': 154,
      'tom-low': 96,
      cowbell: 540,
    },
    wood: {
      rim: 1320,
      'tom-high': 680,
      'tom-low': 420,
      cowbell: 1040,
    },
  }[resolvedKit][voice]
  const typeByKit = ({
    acoustic: voice === 'rim' || voice === 'cowbell' ? 'square' : 'sine',
    electronic: voice === 'tom-low' ? 'sawtooth' : 'square',
    brush: 'triangle',
    wood: 'triangle',
  } satisfies Record<typeof resolvedKit, OscillatorType>)[resolvedKit]
  return [
    playTone(
      context,
      frequency,
      level *
        (resolvedKit === 'electronic'
          ? 0.3
          : resolvedKit === 'brush'
            ? 0.16
            : voice === 'cowbell'
              ? 0.2
              : 0.25),
      voice.startsWith('tom')
        ? resolvedKit === 'electronic'
          ? 0.24
          : resolvedKit === 'wood'
            ? 0.11
            : 0.19
        : resolvedKit === 'wood'
          ? 0.06
          : 0.08,
      typeByKit,
      destination,
      at,
    ),
  ]
}

function triggerStem(
  context: AudioContext,
  role: TrackRole,
  beatIndex: number,
  volume: number,
  speed: number,
  destination: AudioNode,
  at = context.currentTime,
): AudioScheduledSourceNode[] {
  const scaled = Math.max(0.03, volume * 0.16)
  const duration = 0.34 / speed

  if (role === 'drums') {
    return [
      playTone(
        context,
        beatIndex % 2 === 0 ? 58 : 92,
        scaled * 1.4,
        0.12,
        'sine',
        destination,
        at,
      ),
      playNoise(
        context,
        scaled * (beatIndex % 2 === 0 ? 0.38 : 0.85),
        0.08,
        destination,
        at,
      ),
    ]
  }
  if (role === 'bass') {
    return [
      playTone(
        context,
        midiFrequency(40 + (beatIndex % 4) * 2),
        scaled,
        duration * 1.4,
        'triangle',
        destination,
        at,
      ),
    ]
  }
  if (role === 'guitar') {
    const chord = [52, 55, 59, 64]
    return chord
      .slice(0, beatIndex % 2 === 0 ? 4 : 2)
      .map((midi, index) =>
        playTone(
          context,
          midiFrequency(midi),
          scaled * 0.72,
          duration,
          'triangle',
          destination,
          at + index * 0.018,
        ),
      )
  }
  if (role === 'vocals') {
    return beatIndex % 4 === 0
      ? [
          playTone(
            context,
            midiFrequency(64),
            scaled * 0.44,
            duration * 2.6,
            'sine',
            destination,
            at,
          ),
        ]
      : []
  }
  return beatIndex % 2 === 0
    ? [
        playTone(
          context,
          midiFrequency(67),
          scaled * 0.42,
          duration * 1.6,
          'sine',
          destination,
          at,
        ),
      ]
    : []
}

function triggerTabNote(
  context: AudioContext,
  note: TabNote,
  volume: number,
  speed: number,
  destination: AudioNode,
  at: number,
): OscillatorNode {
  return playTone(
    context,
    midiFrequency(OPEN_STRING_MIDI[note.string] + note.fret),
    Math.max(0.015, volume * 0.12 * Math.max(0.3, note.confidence)),
    Math.max(0.07, Math.min(2.4, note.duration / speed)),
    'triangle',
    destination,
    at,
  )
}

export function usePracticeAudio({
  project,
  playing,
  position,
  playbackPositionRef,
  speed,
  mixes,
  rhythm,
  standalone,
}: PracticeAudioOptions): PracticeAudioState {
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const practiceGainRef = useRef<GainNode | null>(null)
  const practiceLimiterRef = useRef<DynamicsCompressorNode | null>(null)
  const analyserConnectedRef = useRef(false)
  const lastBeatRef = useRef<string>('')
  const lastSyntheticPositionRef = useRef(0)
  const countInTimersRef = useRef<number[]>([])
  const countInNodesRef = useRef<AudioScheduledSourceNode[]>([])
  const countInGenerationRef = useRef(0)
  const preScheduledRhythmKeysRef = useRef<Set<string>>(new Set())
  const mediaStartedByCountInRef = useRef(false)
  const mediaRef = useRef<Map<string, HTMLAudioElement>>(new Map())
  const mediaSourcesRef = useRef<Map<string, MediaElementAudioSourceNode>>(new Map())
  const mediaGainsRef = useRef<Map<string, GainNode>>(new Map())
  const {
    playing: standalonePlaying,
    bpm: standaloneBpm,
    definition: standaloneDefinition,
    kit: standaloneKit,
    volume: standaloneVolume,
    mutedVoices: standaloneMutedVoices,
    onStep: onStandaloneStep,
  } = standalone
  const hasRealStems = useMemo(
    () => Boolean(project?.tracks.some((track) => track.url && track.available)),
    [project],
  )
  const sourceScoreNotes = useMemo(
    () =>
      project?.tab.measures
        .flatMap((measure) => measure.beats.flatMap((beat) => beat.notes))
        .filter((note) => note.positionSource?.startsWith('source-score'))
        .sort((left, right) => left.at - right.at) ?? [],
    [project],
  )

  const ensureAudioGraph = useCallback(() => {
    let context = contextRef.current
    if (!context || context.state === 'closed') {
      context = new AudioContext({
        latencyHint: 'interactive',
      })
      contextRef.current = context
      analyserRef.current = null
      practiceGainRef.current = null
      practiceLimiterRef.current = null
      analyserConnectedRef.current = false
    }
    const analyser = analyserRef.current ?? context.createAnalyser()
    analyser.fftSize = 2048
    analyser.smoothingTimeConstant = 0.72
    analyser.minDecibels = -92
    analyser.maxDecibels = -12
    analyserRef.current = analyser
    if (!analyserConnectedRef.current) {
      analyser.connect(context.destination)
      analyserConnectedRef.current = true
    }
    const practiceGain =
      practiceGainRef.current ?? context.createGain()
    const practiceLimiter =
      practiceLimiterRef.current ?? context.createDynamicsCompressor()
    practiceGain.gain.value = PRACTICE_OUTPUT_GAIN
    practiceLimiter.threshold.value = -7
    practiceLimiter.knee.value = 8
    practiceLimiter.ratio.value = 8
    practiceLimiter.attack.value = 0.002
    practiceLimiter.release.value = 0.08
    if (!practiceGainRef.current || !practiceLimiterRef.current) {
      practiceGain.connect(practiceLimiter).connect(analyser)
    }
    practiceGainRef.current = practiceGain
    practiceLimiterRef.current = practiceLimiter
    return { context, analyser, practiceOutput: practiceGain }
  }, [])
  const activateAudio = useCallback(async () => {
    const { context } = ensureAudioGraph()
    if (context.state !== 'running') {
      await context.resume()
    }
  }, [ensureAudioGraph])
  const seekAudio = useCallback((seconds: number) => {
    mediaRef.current.forEach((audio) => {
      if (audio.readyState > 0) audio.currentTime = seconds
    })
  }, [])

  const cancelCountIn = useCallback(() => {
    countInGenerationRef.current += 1
    preScheduledRhythmKeysRef.current.clear()
    mediaStartedByCountInRef.current = false
    countInTimersRef.current.forEach((timer) => window.clearTimeout(timer))
    countInTimersRef.current = []
    countInNodesRef.current.forEach((node) => {
      try {
        node.stop()
      } catch {
        // A source that has already ended cannot be stopped again.
      }
    })
    countInNodesRef.current = []
  }, [])

  const startCountIn = useCallback(
    async ({
      bars,
      speed: playbackSpeed,
      beatDurations,
      kit,
      volume,
      onBeat,
      onComplete,
    }: StartCountInOptions) => {
      cancelCountIn()
      const generation = countInGenerationRef.current
      const { context, analyser, practiceOutput } = ensureAudioGraph()
      if (context.state !== 'running') {
        await Promise.race([
          context.resume(),
          new Promise<void>((resolveDelay) =>
            window.setTimeout(resolveDelay, 250),
          ),
        ])
      }
      if (generation !== countInGenerationRef.current) return
      const beatsPerBar = beatDurations.length
      const startsAt = context.currentTime + 0.06
      const totalBeats = bars * beatsPerBar
      const songPosition = playbackPositionRef.current
      const timers: number[] = []
      const nodes: AudioScheduledSourceNode[] = []
      let elapsed = 0

      for (let beatIndex = 0; beatIndex < totalBeats; beatIndex += 1) {
        const bar = Math.floor(beatIndex / beatsPerBar) + 1
        const beat = (beatIndex % beatsPerBar) + 1
        const at = startsAt + elapsed
        nodes.push(
          ...schedulePercussion(
            context,
            beat === 1 ? 'click-high' : 'click-low',
            kit,
            (beat === 1 ? 1 : 0.76) * volume,
            at,
            practiceOutput,
          ),
        )
        timers.push(
          window.setTimeout(
            () => {
              if (generation === countInGenerationRef.current) {
                onBeat(bar, beat)
              }
            },
            Math.max(0, (at - context.currentTime) * 1000),
          ),
        )
        elapsed += beatDurations[beat - 1] / playbackSpeed
      }
      const songStartsAt = startsAt + elapsed
      const songStartNodes = schedulePercussion(
        context,
        'click-high',
        kit,
        0.82 * volume,
        songStartsAt,
        practiceOutput,
      )
      nodes.push(...songStartNodes)

      if (project && !hasRealStems && sourceScoreNotes.length === 0) {
        const beatIndex = Math.floor(songPosition / (60 / project.bpm))
        const hasSolo = Object.values(mixes).some((mix) => mix.solo)
        project.tracks.forEach((track) => {
          const mix = mixes[track.id]
          if (
            !track.available ||
            !mix ||
            mix.muted ||
            (hasSolo && !mix.solo)
          ) {
            return
          }
          nodes.push(
            ...triggerStem(
              context,
              track.role,
              beatIndex,
              mix.volume,
              playbackSpeed,
              analyser,
              songStartsAt,
            ),
          )
        })
        lastBeatRef.current = `${project.id}:${beatIndex}`
        lastSyntheticPositionRef.current = songPosition
      }

      if (project && rhythm.enabled) {
        const scheduledKeys = new Set<string>()
        project.tab.measures.forEach((measure) => {
          if (
            songPosition < measure.start - 0.025 ||
            songPosition > measure.start + measure.duration
          ) {
            return
          }
          rhythm.definition.hits.forEach((hit, hitIndex) => {
            const eventPosition = rhythmEventPosition(
              measure,
              rhythm.definition,
              hit.step,
            )
            if (Math.abs(eventPosition - songPosition) > 0.025) return
            scheduledKeys.add(`${measure.number}:${hitIndex}`)
            nodes.push(
              ...schedulePercussion(
                context,
                hit.voice,
                rhythm.kit,
                hit.velocity * rhythm.volume,
                songStartsAt,
                practiceOutput,
              ),
            )
          })
        })
        preScheduledRhythmKeysRef.current = scheduledKeys
      }

      let completed = false
      const finishCountIn = () => {
        if (completed || generation !== countInGenerationRef.current) return
        completed = true
        if (hasRealStems) {
          mediaStartedByCountInRef.current = true
          mediaRef.current.forEach((audio) => {
            if (audio.readyState > 0) audio.currentTime = songPosition
            void audio.play()
          })
        }
        countInTimersRef.current = []
        countInNodesRef.current = []
        onComplete()
      }
      songStartNodes[0]?.addEventListener('ended', finishCountIn, {
        once: true,
      })
      timers.push(
        window.setTimeout(
          finishCountIn,
          Math.max(0, (songStartsAt - context.currentTime) * 1000),
        ),
      )
      countInTimersRef.current = timers
      countInNodesRef.current = nodes
    },
    [
      cancelCountIn,
      ensureAudioGraph,
      hasRealStems,
      mixes,
      playbackPositionRef,
      project,
      rhythm,
      sourceScoreNotes.length,
    ],
  )

  useEffect(() => {
    const previous = mediaRef.current
    previous.forEach((audio) => {
      audio.pause()
      audio.src = ''
    })
    mediaSourcesRef.current.forEach((source) => source.disconnect())
    mediaGainsRef.current.forEach((gain) => gain.disconnect())
    mediaSourcesRef.current = new Map()
    mediaGainsRef.current = new Map()
    const next = new Map<string, HTMLAudioElement>()
    const { context, analyser } = ensureAudioGraph()
    project?.tracks.forEach((track) => {
      if (!track.url || !track.available) return
      const audio = new Audio()
      audio.crossOrigin = 'anonymous'
      audio.src = mediaUrl(track.url)
      audio.preload = 'auto'
      const source = context.createMediaElementSource(audio)
      const gain = context.createGain()
      source.connect(gain).connect(analyser)
      next.set(track.id, audio)
      mediaSourcesRef.current.set(track.id, source)
      mediaGainsRef.current.set(track.id, gain)
    })
    mediaRef.current = next
    return () => {
      next.forEach((audio) => {
        audio.pause()
        audio.src = ''
      })
      mediaSourcesRef.current.forEach((source) => source.disconnect())
      mediaGainsRef.current.forEach((gain) => gain.disconnect())
    }
  }, [ensureAudioGraph, project])

  useEffect(() => {
    const hasSolo = Object.values(mixes).some((mix) => mix.solo)
    mediaRef.current.forEach((audio, trackId) => {
      const mix = mixes[trackId]
      const gain = mediaGainsRef.current.get(trackId)
      const audible = !mix?.muted && (!hasSolo || mix?.solo)
      if (gain) {
        gain.gain.value = audible ? Math.max(0, Math.min(1, mix?.volume ?? 0.8)) : 0
      }
      audio.volume = 1
      audio.muted = false
      audio.playbackRate = speed
    })
  }, [mixes, speed])

  useEffect(() => {
    if (!hasRealStems) return
    const { context } = ensureAudioGraph()
    if (!playing) {
      mediaStartedByCountInRef.current = false
      mediaRef.current.forEach((audio) => {
        audio.pause()
      })
      return
    }
    const startedByCountIn = mediaStartedByCountInRef.current
    mediaStartedByCountInRef.current = false
    let active = true
    let synchronizationTimer: number | undefined
    void context.resume().then(async () => {
      if (!active) return
      const audios = [...mediaRef.current.values()]
      if (!startedByCountIn) {
        audios.forEach((audio) => {
          audio.currentTime = playbackPositionRef.current
        })
        await Promise.all(audios.map((audio) => audio.play()))
      }
      const synchronize = () => {
        const master = audios.find((audio) => !audio.paused && !audio.ended)
        if (!master) return
        for (const audio of audios) {
          if (audio === master || audio.paused || audio.ended) continue
          if (Math.abs(audio.currentTime - master.currentTime) > 0.035) {
            audio.currentTime = master.currentTime
          }
        }
      }
      synchronize()
      synchronizationTimer = window.setInterval(synchronize, 180)
    })
    return () => {
      active = false
      if (synchronizationTimer !== undefined) {
        window.clearInterval(synchronizationTimer)
      }
    }
  }, [
    ensureAudioGraph,
    hasRealStems,
    playbackPositionRef,
    playing,
    project?.id,
  ])

  useEffect(() => {
    if (!hasRealStems) return
    mediaRef.current.forEach((audio) => {
      if (!playing || Math.abs(audio.currentTime - position) > 0.18) {
        audio.currentTime = position
      }
    })
  }, [hasRealStems, playing, position])

  useEffect(() => {
    if (!playing || !project || hasRealStems) {
      lastBeatRef.current = ''
      lastSyntheticPositionRef.current = playbackPositionRef.current
      return
    }

    const { context, analyser } = ensureAudioGraph()
    if (sourceScoreNotes.length > 0) {
      const guitarTrack = project.tracks.find(
        (track) => track.available && track.role === 'guitar',
      )
      const mix = guitarTrack ? mixes[guitarTrack.id] : undefined
      const hasSolo = Object.values(mixes).some((candidate) => candidate.solo)
      const audible = Boolean(
        guitarTrack &&
        mix &&
        !mix.muted &&
        (!hasSolo || mix.solo),
      )
      const scheduled = new Set<string>()
      const nodes = new Set<AudioScheduledSourceNode>()
      let previousPosition = playbackPositionRef.current
      let timer: number | undefined
      let active = true
      const scheduleAhead = () => {
        const currentPosition = playbackPositionRef.current
        if (currentPosition + 0.04 < previousPosition) scheduled.clear()
        previousPosition = currentPosition
        if (!audible || !mix) return
        const horizon = currentPosition + 0.18 * speed
        for (const note of sourceScoreNotes) {
          if (
            note.at < currentPosition - 0.025 ||
            note.at > horizon ||
            scheduled.has(note.id)
          ) {
            continue
          }
          scheduled.add(note.id)
          const node = triggerTabNote(
            context,
            note,
            mix.volume,
            speed,
            analyser,
            context.currentTime + Math.max(0, (note.at - currentPosition) / speed),
          )
          nodes.add(node)
          node.addEventListener('ended', () => nodes.delete(node), {
            once: true,
          })
        }
      }
      void context.resume().then(() => {
        if (!active) return
        scheduleAhead()
        timer = window.setInterval(scheduleAhead, 24)
      })
      return () => {
        active = false
        if (timer !== undefined) window.clearInterval(timer)
        nodes.forEach((node) => {
          try {
            node.stop()
          } catch {
            // The pluck may already have ended.
          }
        })
      }
    }

    const secondsPerBeat = 60 / project.bpm
    const hasSolo = Object.values(mixes).some((mix) => mix.solo)
    const triggerCurrentBeat = () => {
      const currentPosition = playbackPositionRef.current
      if (currentPosition + 0.04 < lastSyntheticPositionRef.current) {
        lastBeatRef.current = ''
      }
      lastSyntheticPositionRef.current = currentPosition
      const beatIndex = Math.floor(currentPosition / secondsPerBeat)
      const beatKey = `${project.id}:${beatIndex}`
      if (beatKey === lastBeatRef.current) return
      lastBeatRef.current = beatKey
      project.tracks.forEach((track) => {
        const mix = mixes[track.id]
        if (!track.available || !mix || mix.muted || (hasSolo && !mix.solo)) return
        triggerStem(context, track.role, beatIndex, mix.volume, speed, analyser)
      })
    }
    let interval: number | undefined
    void context.resume().then(() => {
      triggerCurrentBeat()
      interval = window.setInterval(triggerCurrentBeat, 24)
    })
    return () => {
      if (interval !== undefined) window.clearInterval(interval)
    }
  }, [
    ensureAudioGraph,
    hasRealStems,
    mixes,
    playbackPositionRef,
    playing,
    project,
    sourceScoreNotes,
    speed,
  ])

  useEffect(() => {
    if (!playing || !project || !rhythm.enabled) return
    const { context, practiceOutput } = ensureAudioGraph()
    const scheduledKeys = new Set(preScheduledRhythmKeysRef.current)
    preScheduledRhythmKeysRef.current.clear()
    const scheduledNodes = new Set<AudioScheduledSourceNode>()
    let anchorContextTime = context.currentTime
    let anchorPosition = playbackPositionRef.current
    let previousPosition = playbackPositionRef.current
    let interval: number | undefined
    let active = true

    const trackNode = (node: AudioScheduledSourceNode) => {
      scheduledNodes.add(node)
      node.addEventListener('ended', () => scheduledNodes.delete(node), {
        once: true,
      })
    }
    const scheduleAhead = () => {
      const masterAudio = [...mediaRef.current.values()].find(
        (audio) => !audio.paused && !audio.ended,
      )
      const currentPosition = masterAudio
        ? masterAudio.currentTime
        : anchorPosition + (context.currentTime - anchorContextTime) * speed
      if (masterAudio) {
        playbackPositionRef.current = currentPosition
      }
      if (currentPosition + 0.04 < previousPosition) {
        scheduledKeys.clear()
        anchorContextTime = context.currentTime
        anchorPosition = playbackPositionRef.current
      }
      previousPosition = currentPosition
      const horizon = currentPosition + 0.2 * speed

      project.tab.measures.forEach((measure) => {
        const measureEnd = measure.start + measure.duration
        if (measureEnd < currentPosition || measure.start > horizon) return
        rhythm.definition.hits.forEach((hit, hitIndex) => {
          const eventPosition = rhythmEventPosition(
            measure,
            rhythm.definition,
            hit.step,
          )
          if (
            eventPosition < currentPosition - 0.025 ||
            eventPosition > horizon
          ) {
            return
          }
          const eventKey = `${measure.number}:${hitIndex}`
          if (scheduledKeys.has(eventKey)) return
          scheduledKeys.add(eventKey)
          const startsAt =
            context.currentTime +
            Math.max(0, (eventPosition - currentPosition) / speed)
          schedulePercussion(
            context,
            hit.voice,
            rhythm.kit,
            hit.velocity * rhythm.volume,
            startsAt,
            practiceOutput,
          ).forEach(trackNode)
        })
      })
    }

    void context.resume().then(() => {
      if (!active) return
      scheduleAhead()
      interval = window.setInterval(scheduleAhead, 24)
    })
    return () => {
      active = false
      if (interval !== undefined) window.clearInterval(interval)
      scheduledNodes.forEach((node) => {
        try {
          node.stop()
        } catch {
          // A source that has already ended cannot be stopped again.
        }
      })
    }
  }, [
    ensureAudioGraph,
    playbackPositionRef,
    playing,
    position,
    project,
    rhythm.definition,
    rhythm.enabled,
    rhythm.kit,
    rhythm.volume,
    speed,
  ])

  useEffect(() => {
    if (!standalonePlaying) {
      onStandaloneStep(-1)
      return
    }
    const { context, practiceOutput } = ensureAudioGraph()
    const scheduledNodes = new Set<AudioScheduledSourceNode>()
    const stepTimers = new Set<number>()
    const mutedVoices = new Set(standaloneMutedVoices)
    const bpm = Math.max(30, Math.min(260, standaloneBpm))
    const [beatsPerBar, beatUnit] = standaloneDefinition.timeSignature
    const barDuration = (60 / bpm) * beatsPerBar * (4 / beatUnit)
    const baseStepDuration =
      barDuration / standaloneDefinition.stepsPerBar
    const swing = Math.max(
      0.5,
      Math.min(0.75, standaloneDefinition.swing),
    )
    let nextStep = 0
    let nextAt = context.currentTime + 0.06
    let timer: number | undefined
    let active = true

    const trackNode = (node: AudioScheduledSourceNode) => {
      scheduledNodes.add(node)
      node.addEventListener('ended', () => scheduledNodes.delete(node), {
        once: true,
      })
    }
    const scheduleAhead = () => {
      const horizon = context.currentTime + 0.16
      while (nextAt <= horizon) {
        const step = nextStep
        standaloneDefinition.hits
          .filter(
            (hit) =>
              hit.step === step && !mutedVoices.has(hit.voice),
          )
          .forEach((hit) => {
            schedulePercussion(
              context,
              hit.voice,
              standaloneKit,
              hit.velocity * standaloneVolume,
              nextAt,
              practiceOutput,
            ).forEach(trackNode)
          })
        const callbackTimer = window.setTimeout(() => {
          stepTimers.delete(callbackTimer)
          if (active) onStandaloneStep(step)
        }, Math.max(0, (nextAt - context.currentTime) * 1000))
        stepTimers.add(callbackTimer)
        const swingFactor =
            standaloneDefinition.stepsPerBar > beatsPerBar &&
            standaloneDefinition.stepsPerBar % 2 === 0
            ? nextStep % 2 === 0
              ? swing * 2
              : (1 - swing) * 2
            : 1
        nextAt += baseStepDuration * swingFactor
        nextStep =
          (nextStep + 1) % standaloneDefinition.stepsPerBar
      }
    }

    void context.resume().then(() => {
      if (!active) return
      scheduleAhead()
      timer = window.setInterval(scheduleAhead, 40)
    })
    return () => {
      active = false
      onStandaloneStep(-1)
      if (timer !== undefined) window.clearInterval(timer)
      stepTimers.forEach((stepTimer) => window.clearTimeout(stepTimer))
      scheduledNodes.forEach((node) => {
        try {
          node.stop()
        } catch {
          // A scheduled sound may already have ended.
        }
      })
    }
  }, [
    ensureAudioGraph,
    onStandaloneStep,
    standaloneBpm,
    standaloneDefinition,
    standaloneKit,
    standaloneMutedVoices,
    standalonePlaying,
    standaloneVolume,
  ])

  useEffect(
    () => () => {
      cancelCountIn()
      const context = contextRef.current
      contextRef.current = null
      analyserRef.current = null
      practiceGainRef.current = null
      practiceLimiterRef.current = null
      analyserConnectedRef.current = false
      if (context && context.state !== 'closed') {
        void context.close()
      }
    },
    [cancelCountIn],
  )

  return {
    analyserRef,
    activateAudio,
    seekAudio,
    startCountIn,
    cancelCountIn,
  }
}
