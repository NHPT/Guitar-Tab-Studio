import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import type { StudioProject, TrackMix, TrackRole } from './types'

interface PracticeAudioOptions {
  project: StudioProject | null
  playing: boolean
  position: number
  speed: number
  mixes: Record<string, TrackMix>
}

interface PracticeAudioState {
  analyserRef: MutableRefObject<AnalyserNode | null>
  activateAudio: () => Promise<void>
  seekAudio: (seconds: number) => void
}

const API_ROOT = import.meta.env.VITE_API_ROOT ?? 'http://127.0.0.1:8787'

function mediaUrl(url: string): string {
  return url.startsWith('/') ? `${API_ROOT}${url}` : url
}

function midiFrequency(midi: number): number {
  return 440 * 2 ** ((midi - 69) / 12)
}

function playTone(
  context: AudioContext,
  frequency: number,
  volume: number,
  duration: number,
  type: OscillatorType,
  destination: AudioNode,
): void {
  const oscillator = context.createOscillator()
  const gain = context.createGain()
  oscillator.type = type
  oscillator.frequency.value = frequency
  gain.gain.setValueAtTime(Math.max(0.0001, volume), context.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + duration)
  oscillator.connect(gain).connect(destination)
  oscillator.start()
  oscillator.stop(context.currentTime + duration)
}

function playNoise(
  context: AudioContext,
  volume: number,
  duration: number,
  destination: AudioNode,
): void {
  const frameCount = Math.max(1, Math.floor(context.sampleRate * duration))
  const buffer = context.createBuffer(1, frameCount, context.sampleRate)
  const data = buffer.getChannelData(0)
  for (let index = 0; index < frameCount; index += 1) {
    data[index] = Math.random() * 2 - 1
  }

  const source = context.createBufferSource()
  const gain = context.createGain()
  source.buffer = buffer
  gain.gain.setValueAtTime(volume, context.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + duration)
  source.connect(gain).connect(destination)
  source.start()
}

function triggerStem(
  context: AudioContext,
  role: TrackRole,
  beatIndex: number,
  volume: number,
  speed: number,
  destination: AudioNode,
): void {
  const scaled = Math.max(0.03, volume * 0.16)
  const duration = 0.34 / speed

  if (role === 'drums') {
    playTone(
      context,
      beatIndex % 2 === 0 ? 58 : 92,
      scaled * 1.4,
      0.12,
      'sine',
      destination,
    )
    playNoise(
      context,
      scaled * (beatIndex % 2 === 0 ? 0.38 : 0.85),
      0.08,
      destination,
    )
    return
  }
  if (role === 'bass') {
    playTone(
      context,
      midiFrequency(40 + (beatIndex % 4) * 2),
      scaled,
      duration * 1.4,
      'triangle',
      destination,
    )
    return
  }
  if (role === 'guitar') {
    const chord = [52, 55, 59, 64]
    chord.slice(0, beatIndex % 2 === 0 ? 4 : 2).forEach((midi, index) => {
      window.setTimeout(
        () =>
          playTone(
            context,
            midiFrequency(midi),
            scaled * 0.72,
            duration,
            'triangle',
            destination,
          ),
        index * 18,
      )
    })
    return
  }
  if (role === 'vocals') {
    if (beatIndex % 4 === 0) {
      playTone(
        context,
        midiFrequency(64),
        scaled * 0.44,
        duration * 2.6,
        'sine',
        destination,
      )
    }
    return
  }
  if (beatIndex % 2 === 0) {
    playTone(
      context,
      midiFrequency(67),
      scaled * 0.42,
      duration * 1.6,
      'sine',
      destination,
    )
  }
}

export function usePracticeAudio({
  project,
  playing,
  position,
  speed,
  mixes,
}: PracticeAudioOptions): PracticeAudioState {
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const analyserConnectedRef = useRef(false)
  const lastBeatRef = useRef<string>('')
  const mediaRef = useRef<Map<string, HTMLAudioElement>>(new Map())
  const mediaSourcesRef = useRef<Map<string, MediaElementAudioSourceNode>>(new Map())
  const mediaGainsRef = useRef<Map<string, GainNode>>(new Map())
  const positionRef = useRef(position)
  const hasRealStems = useMemo(
    () => Boolean(project?.tracks.some((track) => track.url && track.available)),
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

  useEffect(() => {
    positionRef.current = position
  }, [position])

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
      mediaRef.current.forEach((audio) => {
        audio.pause()
      })
      return
    }
    void context.resume().then(() => {
      mediaRef.current.forEach((audio) => {
        audio.currentTime = positionRef.current
        void audio.play()
      })
    })
  }, [ensureAudioGraph, hasRealStems, playing, project?.id])

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
      return
    }

    const { context, analyser } = ensureAudioGraph()

    const secondsPerBeat = 60 / project.bpm
    const beatIndex = Math.floor(position / secondsPerBeat)
    const beatKey = `${project.id}:${beatIndex}`
    if (beatKey === lastBeatRef.current) return
    lastBeatRef.current = beatKey

    const hasSolo = Object.values(mixes).some((mix) => mix.solo)
    void context.resume().then(() => {
      project.tracks.forEach((track) => {
        const mix = mixes[track.id]
        if (!track.available || !mix || mix.muted || (hasSolo && !mix.solo)) return
        triggerStem(context, track.role, beatIndex, mix.volume, speed, analyser)
      })
    })
  }, [ensureAudioGraph, hasRealStems, mixes, playing, position, project, speed])

  useEffect(
    () => () => {
      const context = contextRef.current
      contextRef.current = null
      analyserRef.current = null
      analyserConnectedRef.current = false
      if (context && context.state !== 'closed') {
        void context.close()
      }
    },
    [],
  )

  return { analyserRef, activateAudio, seekAudio }
}
