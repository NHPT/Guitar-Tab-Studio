import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import { rhythmEventPosition } from '@gts/core'
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
    return [
      playTone(
        context,
        voice === 'click-high' ? 1760 : 1120,
        level * (kit === 'wood' ? 0.78 : 0.9),
        kit === 'wood' ? 0.08 : 0.065,
        kit === 'wood' ? 'sine' : 'square',
        destination,
        at,
      ),
    ]
  }
  if (voice === 'kick') {
    const oscillator = context.createOscillator()
    const gain = context.createGain()
    oscillator.type = 'sine'
    oscillator.frequency.setValueAtTime(
      kit === 'electronic' ? 168 : 132,
      at,
    )
    oscillator.frequency.exponentialRampToValueAtTime(46, at + 0.12)
    gain.gain.setValueAtTime(level * 0.52, at)
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.2)
    oscillator.connect(gain).connect(destination)
    oscillator.start(at)
    oscillator.stop(at + 0.21)
    return [oscillator]
  }
  if (voice === 'snare' || voice === 'clap') {
    const sources: AudioScheduledSourceNode[] = [
      scheduleNoiseBurst(
        context,
        at,
        voice === 'clap' ? 0.095 : 0.14,
        level * (kit === 'brush' ? 0.2 : 0.34),
        destination,
        'highpass',
        kit === 'brush' ? 950 : 1450,
      ),
    ]
    if (voice === 'snare') {
      sources.push(
        playTone(
          context,
          kit === 'electronic' ? 215 : 185,
          level * 0.12,
          0.1,
          'triangle',
          destination,
          at,
        ),
      )
    } else {
      sources.push(
        scheduleNoiseBurst(
          context,
          at + 0.018,
          0.06,
          level * 0.2,
          destination,
          'highpass',
          1700,
        ),
      )
    }
    return sources
  }
  if (
    voice === 'closed-hat' ||
    voice === 'open-hat' ||
    voice === 'shaker'
  ) {
    const duration =
      voice === 'open-hat' ? 0.18 : voice === 'shaker' ? 0.065 : 0.045
    const frequency = voice === 'shaker' ? 4200 : 6500
    return [
      scheduleNoiseBurst(
        context,
        at,
        duration,
        level * (voice === 'open-hat' ? 0.16 : 0.11),
        destination,
        'highpass',
        frequency,
      ),
    ]
  }
  const frequency = {
    rim: kit === 'wood' ? 1320 : 930,
    'tom-high': 176,
    'tom-low': 112,
    cowbell: 620,
  }[voice]
  const type: OscillatorType =
    voice === 'rim' ? 'square' : voice === 'cowbell' ? 'square' : 'sine'
  return [
    playTone(
      context,
      frequency,
      level * (voice === 'cowbell' ? 0.16 : 0.22),
      voice.startsWith('tom') ? 0.18 : 0.075,
      type,
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
}: PracticeAudioOptions): PracticeAudioState {
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
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
    return { context, analyser }
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
      const { context, analyser } = ensureAudioGraph()
      if (context.state !== 'running') await context.resume()
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
            analyser,
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
      nodes.push(
        ...schedulePercussion(
          context,
          'click-high',
          kit,
          0.82 * volume,
          songStartsAt,
          analyser,
        ),
      )

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
                analyser,
              ),
            )
          })
        })
        preScheduledRhythmKeysRef.current = scheduledKeys
      }

      timers.push(
        window.setTimeout(
          () => {
            if (generation !== countInGenerationRef.current) return
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
          },
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
    const { context, analyser } = ensureAudioGraph()
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
            analyser,
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

  useEffect(
    () => () => {
      cancelCountIn()
      const context = contextRef.current
      contextRef.current = null
      analyserRef.current = null
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
