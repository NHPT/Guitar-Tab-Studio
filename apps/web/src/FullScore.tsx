import { useCallback, useLayoutEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import type { StudioProject, TabMeasure, TabNote, Technique } from './types'
import { transposeChordForCapo } from './capo'
import { inferBeatStyle, scoreTiming } from './rhythm'
import type { ScoreTiming } from './rhythm'

type Fret = number | 'x'

interface ChordShape {
  frets: [Fret, Fret, Fret, Fret, Fret, Fret]
  fingers?: [number, number, number, number, number, number]
  baseFret?: number
  barres?: Array<{ fret: number; from: number; to: number }>
}

const chordShapes: Record<string, ChordShape> = {
  C: { frets: ['x', 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0] },
  C7: { frets: ['x', 3, 2, 3, 1, 0], fingers: [0, 3, 2, 4, 1, 0] },
  Cmaj7: { frets: ['x', 3, 2, 0, 0, 0], fingers: [0, 3, 2, 0, 0, 0] },
  D: { frets: ['x', 'x', 0, 2, 3, 2], fingers: [0, 0, 0, 1, 3, 2] },
  Dm: { frets: ['x', 'x', 0, 2, 3, 1], fingers: [0, 0, 0, 2, 3, 1] },
  D7: { frets: ['x', 'x', 0, 2, 1, 2], fingers: [0, 0, 0, 2, 1, 3] },
  Dm7: { frets: ['x', 'x', 0, 2, 1, 1], fingers: [0, 0, 0, 2, 1, 1] },
  Eb: {
    frets: ['x', 6, 8, 8, 8, 6],
    fingers: [0, 1, 3, 3, 3, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  Ebm: {
    frets: ['x', 6, 8, 8, 7, 6],
    fingers: [0, 1, 3, 4, 2, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  Eb7: {
    frets: ['x', 6, 8, 6, 8, 6],
    fingers: [0, 1, 3, 1, 4, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  'D#': {
    frets: ['x', 6, 8, 8, 8, 6],
    fingers: [0, 1, 3, 3, 3, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  'D#m': {
    frets: ['x', 6, 8, 8, 7, 6],
    fingers: [0, 1, 3, 4, 2, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  'D#7': {
    frets: ['x', 6, 8, 6, 8, 6],
    fingers: [0, 1, 3, 1, 4, 1],
    baseFret: 6,
    barres: [{ fret: 6, from: 1, to: 5 }],
  },
  Db: {
    frets: ['x', 4, 6, 6, 6, 4],
    fingers: [0, 1, 3, 3, 3, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  Dbm: {
    frets: ['x', 4, 6, 6, 5, 4],
    fingers: [0, 1, 3, 4, 2, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  Db7: {
    frets: ['x', 4, 6, 4, 6, 4],
    fingers: [0, 1, 3, 1, 4, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  'C#': {
    frets: ['x', 4, 6, 6, 6, 4],
    fingers: [0, 1, 3, 3, 3, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  'C#m': {
    frets: ['x', 4, 6, 6, 5, 4],
    fingers: [0, 1, 3, 4, 2, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  'C#7': {
    frets: ['x', 4, 6, 4, 6, 4],
    fingers: [0, 1, 3, 1, 4, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 1, to: 5 }],
  },
  E: { frets: [0, 2, 2, 1, 0, 0], fingers: [0, 2, 3, 1, 0, 0] },
  Em: { frets: [0, 2, 2, 0, 0, 0], fingers: [0, 2, 3, 0, 0, 0] },
  E7: { frets: [0, 2, 0, 1, 0, 0], fingers: [0, 2, 0, 1, 0, 0] },
  Em7: { frets: [0, 2, 0, 0, 0, 0], fingers: [0, 2, 0, 0, 0, 0] },
  F: {
    frets: [1, 3, 3, 2, 1, 1],
    fingers: [1, 3, 4, 2, 1, 1],
    barres: [{ fret: 1, from: 0, to: 5 }],
  },
  Fmaj7: { frets: ['x', 'x', 3, 2, 1, 0], fingers: [0, 0, 3, 2, 1, 0] },
  'F#': {
    frets: [2, 4, 4, 3, 2, 2],
    fingers: [1, 3, 4, 2, 1, 1],
    baseFret: 2,
    barres: [{ fret: 2, from: 0, to: 5 }],
  },
  'F#m': {
    frets: [2, 4, 4, 2, 2, 2],
    fingers: [1, 3, 4, 1, 1, 1],
    baseFret: 2,
    barres: [{ fret: 2, from: 0, to: 5 }],
  },
  'F#7': {
    frets: [2, 4, 2, 3, 2, 2],
    fingers: [1, 3, 1, 2, 1, 1],
    baseFret: 2,
    barres: [{ fret: 2, from: 0, to: 5 }],
  },
  G: { frets: [3, 2, 0, 0, 0, 3], fingers: [2, 1, 0, 0, 0, 3] },
  G7: { frets: [3, 2, 0, 0, 0, 1], fingers: [3, 2, 0, 0, 0, 1] },
  Ab: {
    frets: [4, 6, 6, 5, 4, 4],
    fingers: [1, 3, 4, 2, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  Abm: {
    frets: [4, 6, 6, 4, 4, 4],
    fingers: [1, 3, 4, 1, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  Ab7: {
    frets: [4, 6, 4, 5, 4, 4],
    fingers: [1, 3, 1, 2, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  'G#': {
    frets: [4, 6, 6, 5, 4, 4],
    fingers: [1, 3, 4, 2, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  'G#m': {
    frets: [4, 6, 6, 4, 4, 4],
    fingers: [1, 3, 4, 1, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  'G#7': {
    frets: [4, 6, 4, 5, 4, 4],
    fingers: [1, 3, 1, 2, 1, 1],
    baseFret: 4,
    barres: [{ fret: 4, from: 0, to: 5 }],
  },
  A: { frets: ['x', 0, 2, 2, 2, 0], fingers: [0, 0, 1, 2, 3, 0] },
  Am: { frets: ['x', 0, 2, 2, 1, 0], fingers: [0, 0, 2, 3, 1, 0] },
  Am9: { frets: ['x', 0, 2, 4, 1, 0], fingers: [0, 0, 1, 3, 2, 0] },
  A7: { frets: ['x', 0, 2, 0, 2, 0], fingers: [0, 0, 2, 0, 3, 0] },
  Am7: { frets: ['x', 0, 2, 0, 1, 0], fingers: [0, 0, 2, 0, 1, 0] },
  Bb: {
    frets: ['x', 1, 3, 3, 3, 1],
    fingers: [0, 1, 3, 3, 3, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  Bbm: {
    frets: ['x', 1, 3, 3, 2, 1],
    fingers: [0, 1, 3, 4, 2, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  Bb7: {
    frets: ['x', 1, 3, 1, 3, 1],
    fingers: [0, 1, 3, 1, 4, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  'A#': {
    frets: ['x', 1, 3, 3, 3, 1],
    fingers: [0, 1, 3, 3, 3, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  'A#m': {
    frets: ['x', 1, 3, 3, 2, 1],
    fingers: [0, 1, 3, 4, 2, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  'A#7': {
    frets: ['x', 1, 3, 1, 3, 1],
    fingers: [0, 1, 3, 1, 4, 1],
    barres: [{ fret: 1, from: 1, to: 5 }],
  },
  B: {
    frets: ['x', 2, 4, 4, 4, 2],
    fingers: [0, 1, 3, 3, 3, 1],
    baseFret: 2,
    barres: [{ fret: 2, from: 1, to: 5 }],
  },
  B7: { frets: ['x', 2, 1, 2, 0, 2], fingers: [0, 2, 1, 3, 0, 4] },
  Bm: {
    frets: ['x', 2, 4, 4, 3, 2],
    fingers: [0, 1, 3, 4, 2, 1],
    baseFret: 2,
    barres: [{ fret: 2, from: 1, to: 5 }],
  },
}

const techniqueMarks: Partial<Record<Technique, string>> = {
  'hammer-on': 'H',
  'pull-off': 'P',
  slide: 'S',
  'palm-mute': 'P.M.',
  slap: 'SLAP',
  'body-tap': 'PERC.',
  tremolo: 'TR',
  bend: 'B',
  'pinch-harmonic': 'P.H.',
  vibrato: '~',
}

const relationalTechniques = new Set<Technique>([
  'hammer-on',
  'pull-off',
  'slide',
])

function getChordShape(name: string): ChordShape | null {
  return chordShapes[name.split('/')[0]] ?? null
}

export function ChordDiagram({ name }: { name: string }) {
  const shape = getChordShape(name)
  if (!shape) {
    return (
      <figure className="chord-diagram is-empty">
        <figcaption>{name}</figcaption>
        <span>{name === 'N.C.' ? '无和弦' : '待校对'}</span>
      </figure>
    )
  }

  const left = 9
  const top = 18
  const stringGap = 7
  const fretGap = 8
  const fretted = shape.frets.filter((fret): fret is number => typeof fret === 'number' && fret > 0)
  const baseFret = shape.baseFret ?? (fretted.length > 0 && Math.max(...fretted) > 4 ? Math.min(...fretted) : 1)

  return (
    <figure className="chord-diagram">
      <figcaption>{name}</figcaption>
      <svg viewBox="0 0 54 60" role="img" aria-label={`${name} 和弦指法图`}>
        {Array.from({ length: 6 }, (_, index) => (
          <line
            key={`string-${index}`}
            x1={left + index * stringGap}
            y1={top}
            x2={left + index * stringGap}
            y2={top + fretGap * 4}
            className="chord-string"
          />
        ))}
        {Array.from({ length: 5 }, (_, index) => (
          <line
            key={`fret-${index}`}
            x1={left}
            y1={top + index * fretGap}
            x2={left + stringGap * 5}
            y2={top + index * fretGap}
            className={index === 0 && baseFret === 1 ? 'chord-nut' : 'chord-fret'}
          />
        ))}
        {baseFret > 1 ? (
          <text x="3" y={top + 6} className="chord-position">
            {baseFret}
          </text>
        ) : null}
        {shape.barres?.map((barre) => {
          const y = top + (barre.fret - baseFret + 0.5) * fretGap
          return (
            <line
              key={`${barre.fret}-${barre.from}-${barre.to}`}
              x1={left + barre.from * stringGap}
              y1={y}
              x2={left + barre.to * stringGap}
              y2={y}
              className="chord-barre"
            />
          )
        })}
        {shape.frets.map((fret, index) => {
          const x = left + index * stringGap
          if (fret === 'x' || fret === 0) {
            return (
              <text key={`open-${index}`} x={x} y={top - 5} className="chord-open">
                {fret === 'x' ? '×' : '○'}
              </text>
            )
          }
          return (
            <g key={`dot-${index}`}>
              <circle
                cx={x}
                cy={top + (fret - baseFret + 0.5) * fretGap}
                r="3.3"
                className="chord-dot"
              />
              {shape.fingers?.[index] ? (
                <text
                  x={x}
                  y={top + (fret - baseFret + 0.5) * fretGap + 1.8}
                  className="chord-finger"
                >
                  {shape.fingers[index]}
                </text>
              ) : null}
            </g>
          )
        })}
        <text x={left} y="58" className="chord-string-number">
          6
        </text>
        <text x={left + stringGap * 5} y="58" className="chord-string-number">
          1
        </text>
      </svg>
    </figure>
  )
}

function rhythmKind(measure: TabMeasure, timing: ScoreTiming): string {
  const styles = measure.beats
    .filter((beat) => beat.notes.length > 0)
    .flatMap((beat) => inferBeatStyle(beat) === 'pick' &&
      beat.notes.some((note) => note.technique.startsWith('strum-'))
      ? ['strum', 'pick'] : [inferBeatStyle(beat)])
  if (styles.length === 0) return measure.beats.some((beat) => timing.heldBeatTimes.has(beat.at)) ? '延音' : '休止'
  if (styles.every((style) => style === 'strum')) return '扫弦'
  if (styles.every((style) => style === 'arpeggio')) return '琶音'
  if (styles.every((style) => style === 'rasgueado')) return '轮扫'
  if (styles.every((style) => style === 'tremolo')) return '轮指'
  if (styles.every((style) => !style || style === 'pick')) return '拨弦'
  return '综合'
}

function relativeFret(note: TabNote, capo: number): number {
  return capo > 0 && note.fret >= capo ? note.fret - capo : note.fret
}

function noteLabel(note: TabNote, capo: number): string {
  if (note.technique === 'dead-note' || note.technique === 'body-tap') {
    return 'x'
  }
  const fret = relativeFret(note, capo)
  if (note.technique === 'harmonic') {
    if (note.harmonicType === 'artificial') {
      const touchFret = Math.max(
        0,
        (note.harmonicTouchFret ?? note.fret + 12) - capo,
      )
      return `${fret}<${touchFret}>`
    }
    return `<${fret}>`
  }
  return String(fret)
}

const numberedPitchClasses: Record<string, number> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11,
}

function numberedPitch(
  pitch: number,
  keySignature: string,
  tonicPitch: number,
): { label: string; octave: number } {
  const root = keySignature.match(/^[A-G](?:#|b)?/)?.[0] ?? 'C'
  const rootPitchClass = numberedPitchClasses[root] ?? 0
  const relative = (Math.round(pitch) - rootPitchClass + 120) % 12
  const minor = /\bminor\b|\bmin\b/u.test(keySignature)
  const labels = minor
    ? ['1', '#1', '2', '3', '#3', '4', '#4', '5', '6', '#6', '7', '#7']
    : ['1', '#1', '2', '#2', '3', '4', '#4', '5', '#5', '6', '#6', '7']
  return {
    label: labels[relative],
    octave: Math.round((pitch - tonicPitch) / 12),
  }
}

function TabNotation({
  measure,
  tuning,
  capo,
  timing,
  timeSignature,
  relatedNotes,
  relationSourceIds,
  vocalMelody,
  lyrics,
  keySignature,
  melodyTonicPitch,
  showLabels,
  timelineWidth,
  selectedNoteId,
  onSelectBeat,
  onSelectNote,
}: {
  measure: TabMeasure
  tuning: StudioProject['tab']['tuning']
  capo: number
  timing: ScoreTiming
  timeSignature: StudioProject['tab']['timeSignature']
  relatedNotes: ReadonlyMap<string, TabNote>
  relationSourceIds: ReadonlySet<string>
  vocalMelody: StudioProject['tab']['vocalMelody']
  lyrics: StudioProject['tab']['lyrics']
  keySignature: string
  melodyTonicPitch: number
  showLabels: boolean
  timelineWidth?: number
  selectedNoteId?: string
  onSelectBeat: (beatIndex: number) => void
  onSelectNote: (id: string) => void
}) {
  const width = timelineWidth ?? 300
  const staffTop = 48
  const stringGap = 12
  const staffBottom = staffTop + stringGap * 5
  const rhythmBeamY = 135
  const rhythmBeamGap = 5
  const strumTop = staffTop - 8
  const strumBottom = staffBottom + 6
  const lineStart = showLabels ? 32 : 0
  const timelineInset = timelineWidth === undefined ? 0 : 14
  const timelineStart =
    timelineWidth === undefined
      ? lineStart + (measure.number === 1 ? 26 : 20)
      : measure.number === 1
        ? 58
        : timelineInset
  const timelineEnd =
    timelineWidth === undefined ? width - 20 : width + timelineInset
  const xAtTime = (at: number) => {
    const ratio = Math.max(
      0,
      Math.min(1, (at - measure.start) / Math.max(0.01, measure.duration)),
    )
    return timelineStart + ratio * (timelineEnd - timelineStart)
  }
  const measureNotes = measure.beats.flatMap((beat) => beat.notes)
  const notationAt = (note: TabNote) =>
    timing.displayTimes.get(note.id) ?? note.notationAt ?? note.at
  const performanceNotationAt = (note: TabNote) => note.notationAt ?? note.at
  const noteX = (note: TabNote) => xAtTime(notationAt(note))
  const rhythmGroups = measure.beats.map((_, beatIndex) =>
    (timing.attacks.get(measure.number) ?? []).filter(
      (attack) => attack.beatIndex === beatIndex,
    ),
  )
  const rhythmSegments = timing.segments.get(measure.number) ?? []
  const quarterScale = 4 / timeSignature[1]
  const measureCapacity = timeSignature[0] * quarterScale
  const melodyPositionAt = (at: number) => {
    const beatIndex = Math.max(
      0,
      measure.beats.findLastIndex((beat) => beat.at <= at),
    )
    const beat = measure.beats[beatIndex]
    return Math.max(
      0,
      Math.min(
        measureCapacity,
        (beatIndex + (at - beat.at) / Math.max(0.01, beat.duration)) *
          quarterScale,
      ),
    )
  }
  const measureMelody = (vocalMelody ?? []).filter(
    (note) =>
      note.at >= measure.start &&
      note.at < measure.start + measure.duration,
  )
  const measureLyrics = (lyrics ?? []).filter(
    (lyric) =>
      lyric.at >= measure.start &&
      lyric.at < measure.start + measure.duration,
  )
  const lyricAnchors = measureLyrics.flatMap((lyric) => {
    const note = measureMelody.find(
      (candidate) => Math.abs(candidate.at - lyric.at) <= 0.025,
    ) ?? [...measureMelody].sort(
      (left, right) =>
        Math.abs(left.at - lyric.at) - Math.abs(right.at - lyric.at),
    )[0]
    return note ? [{ lyric, note }] : []
  })
  const anchorNotes = [
    ...lyricAnchors.map(({ note, lyric }) => ({ note, lyric })),
    ...measureMelody
      .filter(
        (note) =>
          !lyricAnchors.some(
            ({ note: lyricNote }) => lyricNote.id === note.id,
          ) &&
          !lyricAnchors.some(
            ({ note: lyricNote }) =>
              Math.abs(
                melodyPositionAt(lyricNote.at) - melodyPositionAt(note.at),
              ) < 0.125,
          ),
      )
      .map((note) => ({ note, lyric: undefined })),
  ].sort((left, right) => left.note.at - right.note.at)
  const minimumMelodyStep = 0.25
  let previousMelodyPosition = -minimumMelodyStep
  const melodyAnchors = anchorNotes.map((anchor, index) => {
    const lower = previousMelodyPosition + minimumMelodyStep
    const upper = Math.max(
      lower,
      measureCapacity -
        (anchorNotes.length - index) * minimumMelodyStep,
    )
    const position = Math.max(
      lower,
      Math.min(
        upper,
        Math.round(
          melodyPositionAt(anchor.note.at) / minimumMelodyStep,
        ) * minimumMelodyStep,
      ),
    )
    previousMelodyPosition = position
    return { ...anchor, position }
  })
  const numberedSegments: Array<{
    id: string
    kind: 'note' | 'hold' | 'rest'
    position: number
    duration: number
    note?: NonNullable<StudioProject['tab']['vocalMelody']>[number]
    lyric?: NonNullable<StudioProject['tab']['lyrics']>[number]
  }> = []
  const canonicalDurations = [4, 3, 2, 1.5, 1, 0.75, 0.5, 0.25]
  const appendNumberedInterval = (
    kind: 'note' | 'rest',
    start: number,
    end: number,
    anchor?: (typeof melodyAnchors)[number],
  ) => {
    let cursor = start
    let first = true
    while (cursor < end - 0.0001) {
      const remaining = end - cursor
      const duration =
        canonicalDurations.find(
          (candidate) => candidate <= remaining + 0.0001,
        ) ?? remaining
      numberedSegments.push({
        id: `${measure.number}-${kind}-${cursor}-${numberedSegments.length}`,
        kind: kind === 'rest' ? 'rest' : first ? 'note' : 'hold',
        position: cursor,
        duration,
        note: first ? anchor?.note : undefined,
        lyric: first ? anchor?.lyric : undefined,
      })
      cursor += duration
      first = false
    }
  }
  let numberedCursor = 0
  melodyAnchors.forEach((anchor, index) => {
    if (anchor.position > numberedCursor + 0.0001) {
      appendNumberedInterval('rest', numberedCursor, anchor.position)
    }
    const nextPosition =
      melodyAnchors[index + 1]?.position ?? measureCapacity
    const rawEnd = Math.max(
      anchor.position + minimumMelodyStep,
      Math.round(
        melodyPositionAt(anchor.note.at + anchor.note.duration) /
          minimumMelodyStep,
      ) * minimumMelodyStep,
    )
    const soundEnd =
      nextPosition - rawEnd <= minimumMelodyStep
        ? nextPosition
        : Math.min(nextPosition, rawEnd)
    appendNumberedInterval('note', anchor.position, soundEnd, anchor)
    numberedCursor = soundEnd
    if (soundEnd < nextPosition - 0.0001) {
      appendNumberedInterval('rest', soundEnd, nextPosition)
      numberedCursor = nextPosition
    }
  })
  if (numberedCursor < measureCapacity - 0.0001) {
    appendNumberedInterval('rest', numberedCursor, measureCapacity)
  }
  const numberedX = (position: number) =>
    Math.max(
      timelineStart + 4,
      Math.min(
        timelineEnd - 4,
        timelineStart +
          (position / Math.max(0.01, measureCapacity)) *
            (timelineEnd - timelineStart),
      ),
    )
  const lyricLayout = measureLyrics.map((lyric) => {
    const segment = numberedSegments.find(
      (candidate) => candidate.lyric === lyric,
    )
    return {
      lyric,
      x: segment ? numberedX(segment.position) : xAtTime(lyric.at),
      width: Math.max(18, lyric.text.length * 7.2),
      lane: 0,
    }
  })
  const lyricLaneEnds = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]
  for (const item of lyricLayout) {
    const left = item.x - item.width / 2
    const lane = left >= lyricLaneEnds[0] + 5 ? 0 : 1
    item.lane = lane
    lyricLaneEnds[lane] = item.x + item.width / 2
  }
  const hasWrappedLyrics = lyricLayout.some((item) => item.lane > 0)
  const showVocalGuide =
    (vocalMelody?.length ?? 0) > 0 || (lyrics?.length ?? 0) > 0
  const sourceMelodyNotes = measureNotes
    .filter((note) => note.positionSource?.startsWith('source-score'))
    .sort((left, right) => performanceNotationAt(left) - performanceNotationAt(right))
  const tripletGap =
    measure.duration / Math.max(1, measure.beats.length) * (2 / 3)
  const crossBeatTriplets: TabNote[][] = []
  for (let index = 0; index + 2 < sourceMelodyNotes.length; index += 1) {
    const notes = sourceMelodyNotes.slice(index, index + 3)
    const firstGap =
      performanceNotationAt(notes[1]) - performanceNotationAt(notes[0])
    const secondGap =
      performanceNotationAt(notes[2]) - performanceNotationAt(notes[1])
    if (
      Math.abs(firstGap - tripletGap) <= 0.025 &&
      Math.abs(secondGap - tripletGap) <= 0.025
    ) {
      crossBeatTriplets.push(notes)
      index += 2
    }
  }
  const crossBeatTripletNoteIds = new Set(
    crossBeatTriplets.flatMap((notes) => notes.map((note) => note.id)),
  )
  const positionedNotes = measure.beats.flatMap((beat) =>
    beat.notes.map((note) => ({
      note,
      x: noteX(note),
      y: staffTop + (note.string - 1) * stringGap,
    })),
  )
  const performedGroups = measure.beats.flatMap((beat, beatIndex) => {
    const style = inferBeatStyle(beat)
    if (['arpeggio', 'rasgueado', 'tremolo'].includes(style)) {
      return [{ beat, beatIndex, at: rhythmGroups[beatIndex][0]?.at ?? beat.at }]
    }
    return rhythmGroups[beatIndex].map((group) => {
      const technique = group.notes[0]?.technique
      const strum = style === 'strum' ||
        (['strum-up', 'strum-down'].includes(technique) &&
          group.notes.every((note) => note.technique === technique))
      return {
        beat: {
          ...beat,
          notes: group.notes,
          style: strum ? 'strum' as const : 'pick' as const,
          direction: strum ? technique === 'strum-up' ? 'up' as const : beat.direction ?? 'down' as const : undefined,
        },
        beatIndex,
        at: group.at,
      }
    })
  })
  const notePositions = new Map(
    positionedNotes.map((position) => [position.note.id, position]),
  )
  const ties = positionedNotes.flatMap((position, index) => {
    const { note, x, y } = position
    const segments: Array<{
      id: string
      startX: number
      endX: number
      y: number
    }> = []
    const previous = positionedNotes
      .slice(0, index)
      .findLast(
        (candidate) =>
          candidate.note.tieToNext &&
          candidate.note.string === note.string &&
          candidate.note.fret === note.fret,
      )
    if (note.tieFromPrevious && !previous) {
      segments.push({
        id: `${note.id}-in`,
        startX: lineStart + 4,
        endX: x,
        y,
      })
    }
    if (!note.tieToNext) return segments
    const next = positionedNotes
      .slice(index + 1)
      .find(
        (candidate) =>
          candidate.note.tieFromPrevious &&
          candidate.note.string === note.string &&
          candidate.note.fret === note.fret,
      )
    segments.push({
      id: `${note.id}-out`,
      startX: x,
      endX: next?.x ?? width - 4,
      y,
    })
    return segments
  })

  return (
    <svg
      className={`tab-notation${showVocalGuide ? ' has-vocal-guide' : ''}${
        hasWrappedLyrics ? ' has-wrapped-lyrics' : ''
      }`}
      viewBox={`0 0 ${width} ${
        showVocalGuide ? hasWrappedLyrics ? 246 : 228 : 146
      }`}
      data-timeline-start={timelineStart}
      data-timeline-end={timelineEnd}
      role="img"
      aria-label={`第 ${measure.number} 小节 ${rhythmKind(measure, timing)}六线谱`}
    >
      {showLabels ? (
        <>
          <text x="7" y="76" className="tab-clef">
            T
          </text>
          <text x="7" y="88" className="tab-clef">
            A
          </text>
          <text x="7" y="100" className="tab-clef">
            B
          </text>
        </>
      ) : null}
      {tuning.map((tune, index) => {
        const y = staffTop + index * stringGap
        return (
          <g key={`${measure.number}-${tune}`}>
            {showLabels ? (
              <text x="17" y={y + 3} className="tab-tuning">
                {index + 1} {index === 0 ? 'e' : tune.replace(/\d/, '')}
              </text>
            ) : null}
            <line x1={lineStart} y1={y} x2={width} y2={y} className="tab-string-line" />
          </g>
        )
      })}
      {measure.number === 1 ? (
        <g className="time-signature" aria-label={`${timeSignature[0]}/${timeSignature[1]}拍`}>
          <text x={lineStart + 14} y={staffTop + 25}>
            {timeSignature[0]}
          </text>
          <text x={lineStart + 14} y={staffTop + 45}>
            {timeSignature[1]}
          </text>
        </g>
      ) : null}
      <line
        x1={lineStart}
        y1={staffTop - 3}
        x2={lineStart}
        y2={staffBottom + 3}
        className="tab-bar-line"
      />
      <line
        x1={width - 1}
        y1={staffTop - 3}
        x2={width - 1}
        y2={staffBottom + 3}
        className="tab-bar-line"
      />

      {ties.map((tie) => {
        const centerX = (tie.startX + tie.endX) / 2
        return (
          <path
            key={tie.id}
            className="tie-connection"
            data-tie="true"
            d={`M ${tie.startX} ${tie.y - 5} Q ${centerX} ${
              tie.y - 16
            } ${tie.endX} ${tie.y - 5}`}
          />
        )
      })}

      {positionedNotes.map(({ note, x, y }) => {
        if (!note.relatedNoteId || !relationalTechniques.has(note.technique)) {
          return null
        }
        const related = notePositions.get(note.relatedNoteId)
        const externalRelated = relatedNotes.get(note.relatedNoteId)
        if (!related && !externalRelated) return null
        const label = techniqueMarks[note.technique]
        const relationStartX = related?.x ?? lineStart + 4
        const relatedFret = related?.note.fret ?? externalRelated!.fret
        const centerX = (relationStartX + x) / 2
        const continuationClass = related ? '' : ' is-continuation'
        if (note.technique === 'slide') {
          const ascending = note.fret >= relatedFret
          return (
            <g
              key={`relation-${note.id}`}
              className={`technique-connection${continuationClass}`}
            >
              <line
                x1={relationStartX}
                y1={y + (ascending ? 5 : -5)}
                x2={x}
                y2={y + (ascending ? -5 : 5)}
              />
              <text x={centerX} y={y - 8} className="technique-label">
                {label}
              </text>
            </g>
          )
        }
        return (
          <g
            key={`relation-${note.id}`}
            className={`technique-connection${continuationClass}`}
          >
            <path
              d={`M ${relationStartX} ${y - 5} Q ${centerX} ${y - 22} ${x} ${y - 5}`}
            />
            <text x={centerX} y={y - 16} className="technique-label">
              {label}
            </text>
          </g>
        )
      })}

      {performedGroups.map(({ beat, beatIndex, at }) => {
        const style = inferBeatStyle(beat)
        const hasRelationDetail = beat.notes.some(
          (note) =>
            relationalTechniques.has(note.technique) ||
            relationSourceIds.has(note.id),
        )
        const isStrum =
          (style === 'strum' || style === 'rasgueado') &&
          !hasRelationDetail
        const x = xAtTime(at)
        const isArpeggio = style === 'arpeggio'
        const isRasgueado = style === 'rasgueado'
        const isTremolo =
          style === 'tremolo' ||
          beat.notes.some((note) => note.technique === 'tremolo')
        const visibleNotes = beat.notes
        return (
          <g key={`${measure.number}-beat-${beatIndex}-${at}`}>
            {isStrum ? (
              <g
                data-gesture-at={at}
                data-notation-at={at}
                data-direction={beat.direction}
                className={`strum-mark${
                  beat.notes.some((note) => note.id === selectedNoteId) ? ' is-selected' : ''
                }`}
                role="button"
                tabIndex={0}
                aria-label={`第 ${measure.number} 小节 ${
                  beat.direction === 'up' ? '上扫' : '下扫'
                }`}
                onClick={() => onSelectBeat(beatIndex)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    onSelectBeat(beatIndex)
                  }
                }}
              >
                <title>{beat.direction === 'up' ? '上扫：1弦向6弦' : '下扫：6弦向1弦'}</title>
                <line
                  x1={x}
                  y1={beat.direction === 'up' ? strumBottom : strumTop}
                  x2={x}
                  y2={beat.direction === 'up' ? strumTop : strumBottom}
                />
                <path
                  d={
                    beat.direction === 'up'
                      ? `M ${x - 4} ${strumTop + 6} L ${x} ${strumTop} L ${
                          x + 4
                        } ${strumTop + 6}`
                      : `M ${x - 4} ${strumBottom - 6} L ${x} ${strumBottom} L ${
                          x + 4
                        } ${strumBottom - 6}`
                  }
                />
                {isRasgueado ? (
                  <>
                    <line x1={x - 4} y1={staffTop - 2} x2={x - 4} y2={staffBottom + 2} />
                    <line x1={x + 4} y1={staffTop - 2} x2={x + 4} y2={staffBottom + 2} />
                  </>
                ) : null}
              </g>
            ) : null}
            {isArpeggio ? (
              <g className="arpeggio-mark" aria-hidden="true">
                <path
                  d={`M ${x - 7} ${staffTop - 2} q 5 4 0 8 t 0 8 t 0 8 t 0 8 t 0 8 t 0 8 t 0 8`}
                />
                <path
                  d={
                    beat.direction === 'up'
                      ? `M ${x - 11} ${staffTop + 3} L ${x - 7} ${staffTop - 3} L ${
                          x - 3
                        } ${staffTop + 3}`
                      : `M ${x - 11} ${staffBottom - 3} L ${x - 7} ${
                          staffBottom + 3
                        } L ${x - 3} ${staffBottom - 3}`
                  }
                />
              </g>
            ) : null}
            {isArpeggio || isRasgueado || isTremolo ? (
              <g
                className="beat-style-control"
                role="button"
                tabIndex={0}
                aria-label={`第 ${measure.number} 小节${
                  isArpeggio ? '琶音' : isRasgueado ? '轮扫' : '轮指'
                }`}
                onClick={() => onSelectBeat(beatIndex)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    onSelectBeat(beatIndex)
                  }
                }}
              >
                <rect x={x - 18} y="6" width="36" height="15" rx="2" />
                <text x={x} y="17" className="beat-style-label">
                  {isArpeggio ? 'ARP.' : isRasgueado ? 'RASG.' : 'TR'}
                </text>
              </g>
            ) : null}
            {visibleNotes.map((note) => {
              const notePositionX = noteX(note)
              const y = staffTop + (note.string - 1) * stringGap
              const label = noteLabel(note, capo)
              const labelWidth = Math.max(14, label.length * 6.2 + 4)
              const relationResolved = Boolean(
                note.relatedNoteId &&
                  (notePositions.has(note.relatedNoteId) ||
                    relatedNotes.has(note.relatedNoteId)),
              )
              const techniqueLabel =
                note.technique === 'harmonic'
                  ? note.harmonicType === 'artificial'
                    ? 'A.H.'
                    : 'Harm.'
                  : techniqueMarks[note.technique]
              const uncertain =
                note.confidence < 0.76 ||
                (note.technique !== 'pick' &&
                  (note.techniqueConfidence ?? note.confidence) < 0.62)
              return (
                <g key={note.id}>
                  {techniqueLabel &&
                  !(relationalTechniques.has(note.technique) && relationResolved) ? (
                    <text
                      x={notePositionX}
                      y={note.technique === 'harmonic' ? 17 : 29}
                      className="technique-label"
                    >
                      {techniqueLabel}
                    </text>
                  ) : null}
                  <g
                    data-note-id={note.id}
                    data-note-at={note.at}
                    data-notation-at={notationAt(note)}
                    className={`tab-glyph${
                      selectedNoteId === note.id ? ' is-selected' : ''
                    }${uncertain ? ' is-uncertain' : ''}`}
                    role="button"
                    tabIndex={0}
                    aria-label={`${note.string}弦${relativeFret(note, capo)}品 ${
                      techniqueLabel ?? ''
                    }`}
                    onClick={() => onSelectNote(note.id)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onSelectNote(note.id)
                      }
                    }}
                  >
                    <rect
                      x={notePositionX - labelWidth / 2}
                      y={y - 7}
                      width={labelWidth}
                      height="14"
                      rx="1"
                    />
                    <text x={notePositionX} y={y + 4}>
                      {label}
                    </text>
                  </g>
                </g>
              )
            })}
            {isTremolo ? (
              <g className="tremolo-strokes" aria-hidden="true">
                <line x1={x - 5} y1="120" x2={x + 5} y2="116" />
                <line x1={x - 5} y1="125" x2={x + 5} y2="121" />
                <line x1={x - 5} y1="130" x2={x + 5} y2="126" />
              </g>
            ) : null}
          </g>
        )
      })}

      {rhythmSegments.filter((segment) => segment.kind === 'continuation').map(
        ({ at, position, value, notes }) => (
          <g
            key={`held-${notes[0].id}-${at}`}
            className="rhythm-continuation"
            data-duration-beats={value.beats}
            data-rhythm-position={position}
            data-rhythm-x={xAtTime(at)}
          />
        ),
      )}
      {rhythmSegments.filter((segment) => segment.kind === 'rest').map(({ at, endAt, position, value }, index) => {
        const x = (xAtTime(at) + xAtTime(endAt)) / 2
        return (
          <g
            key={`rest-${measure.number}-${position}-${index}`}
            className="rhythm-rest-event"
            data-duration-beats={value.beats}
            data-rhythm-position={position}
          >
            <path
              className="rhythm-rest"
              d={`M ${x - 3} 113 l 6 4 -5 5 5 5 -4 7`}
            />
            {value.dotted ? (
              <circle cx={x + 7} cy="126" r="1.5" className="rhythm-dot" />
            ) : null}
          </g>
        )
      })}
      {measure.beats.map((beat, beatIndex) => {
        const groups = rhythmGroups[beatIndex]
        if (groups.length === 0) {
          const slotStart = beatIndex * quarterScale
          const slotEnd = slotStart + quarterScale
          const sustained = rhythmSegments
            .filter((segment) =>
              segment.kind !== 'rest' &&
              segment.position < slotEnd - 0.0001 &&
              segment.position + segment.value.beats > slotStart + 0.0001
            )
            .flatMap((segment) => segment.notes)
            .filter((note, index, notes) =>
              notes.findIndex((candidate) => candidate.id === note.id) === index
            )
          const isSustain = sustained.length > 0
          return (
            <g
              key={`${measure.number}-rhythm-${beatIndex}`}
              className={`rhythm-beat-slot ${isSustain ? 'is-sustain' : 'is-rest'}`}
              data-beat-index={beatIndex}
              data-event-count="0"
            >
              {sustained.map((note) => {
                const y = staffTop + (note.string - 1) * stringGap - 5
                const startX = xAtTime(beat.at)
                const endX = xAtTime(beat.at + beat.duration)
                return (
                  <g key={`sustain-${note.id}`}>
                    <path
                      className="tie-connection"
                      data-sustain-note={note.id}
                      d={`M ${startX} ${y} Q ${(startX + endX) / 2} ${y - 10} ${endX} ${y}`}
                    />
                  </g>
                )
              })}
            </g>
          )
        }

        const groupXs = groups.map((group) => xAtTime(group.at))
        const isTriplet =
          groups.length === 3 &&
          groups.every((group) => group.tuplet === 3)
        const isCrossBeatTripletSegment = groups.every((group) =>
          group.notes.some((note) => crossBeatTripletNoteIds.has(note.id)),
        )
        const values = groups.map((group) =>
          timing.values.get(group.notes[0].id) ??
          { beats: 1, flags: 0, dotted: false, hollow: false },
        )
        const flags = values.map((value) =>
          isCrossBeatTripletSegment ? 0 : isTriplet ? 1 : value.flags,
        )
        return (
          <g
            key={`${measure.number}-rhythm-${beatIndex}`}
            className="rhythm-beat-slot"
            data-beat-index={beatIndex}
            data-event-count={groups.length}
            data-tuplet={isTriplet ? '3' : undefined}
          >
            {groupXs.map((x, groupIndex) => {
              const value = values[groupIndex]
              const dottedString = Math.min(
                ...groups[groupIndex].notes.map((note) => note.string),
              )
              return (
                <g
                  key={`${measure.number}-${beatIndex}-${x}-${groupIndex}`}
                  className="rhythm-event"
                  data-duration-beats={value.beats}
                  data-rhythm-position={groups[groupIndex].position}
                  data-rhythm-x={x}
                  data-flags={flags[groupIndex]}
                  data-dotted={value.dotted && !isTriplet ? 'true' : undefined}
                >
                  {(value.beats < 4 || isTriplet) ? <line
                    x1={x}
                    y1={staffBottom + 1}
                    x2={x}
                    y2={rhythmBeamY}
                    className="rhythm-stem"
                  /> : null}
                  {value.hollow && !isTriplet ? (
                    <ellipse cx={x + 3} cy={staffBottom + 7} rx="3" ry="2" className="rhythm-hollow" />
                  ) : null}
                  {value.dotted && !isTriplet ? (
                    <circle
                      cx={x + 8}
                      cy={staffTop + (dottedString - 1) * stringGap + 1}
                      r="1.5"
                      className="rhythm-dot"
                    />
                  ) : null}
                  {Array.from({ length: flags[groupIndex] }, (_, level) => {
                    const connectsNext =
                      (flags[groupIndex + 1] ?? 0) > level &&
                      Math.abs(
                        groups[groupIndex].position +
                          value.beats -
                          groups[groupIndex + 1].position,
                      ) <= 0.001
                    const connectsPrevious =
                      (flags[groupIndex - 1] ?? 0) > level &&
                      Math.abs(
                        groups[groupIndex - 1].position +
                          values[groupIndex - 1].beats -
                          groups[groupIndex].position,
                      ) <= 0.001
                    if (connectsPrevious && !connectsNext) return null
                    const hookDirection =
                      groupIndex > 0 &&
                      (flags[groupIndex - 1] ?? 0) > 0 &&
                      Math.abs(
                        groups[groupIndex - 1].position +
                          values[groupIndex - 1].beats -
                          groups[groupIndex].position,
                      ) <= 0.001
                        ? -1
                        : 1
                    return connectsNext ? (
                      <line
                        key={level}
                        x1={x}
                        y1={rhythmBeamY - level * rhythmBeamGap}
                        x2={groupXs[groupIndex + 1]}
                        y2={rhythmBeamY - level * rhythmBeamGap}
                        className="rhythm-beam"
                        data-beam-level={level}
                      />
                    ) : (
                      <line
                        key={level}
                        x1={x}
                        y1={rhythmBeamY - level * rhythmBeamGap}
                        x2={x + 9 * hookDirection}
                        y2={rhythmBeamY - level * rhythmBeamGap}
                        className="rhythm-flag"
                        data-beam-level={level}
                      />
                    )
                  })}
                </g>
              )
            })}
            {isTriplet ? (
              <text
                x={(groupXs[0] + groupXs.at(-1)!) / 2}
                y="143"
                className="rhythm-tuplet"
              >
                3
              </text>
            ) : null}
          </g>
        )
      })}
      {crossBeatTriplets.map((notes, index) => {
        const startX = noteX(notes[0])
        const endX = noteX(notes[2])
        const centerX = (startX + endX) / 2
        return (
          <g
            key={`${measure.number}-cross-triplet-${index}`}
            className="rhythm-tuplet-group"
            data-tuplet="3"
          >
            <path
              className="rhythm-tuplet-bracket"
              d={`M ${startX} 137 v 3 H ${centerX - 5} M ${
                centerX + 5
              } 140 H ${endX} v -3`}
            />
            <text x={centerX} y="143" className="rhythm-tuplet">
              3
            </text>
          </g>
        )
      })}
      {showVocalGuide ? (
        <g className="vocal-guide" aria-label="简谱与歌词">
          <line
            x1={timelineStart}
            y1="149"
            x2={timelineEnd}
            y2="149"
            className="vocal-guide-rule"
          />
          {numberedSegments.map((segment) => {
            const x = numberedX(segment.position)
            const numbered = segment.note
              ? numberedPitch(
                  segment.note.pitch,
                  keySignature,
                  melodyTonicPitch,
                )
              : {
                  label: segment.kind === 'rest' ? '0' : '−',
                  octave: 0,
                }
            const accidental = numbered.label.startsWith('#') ? '#' : ''
            const numeral = accidental
              ? numbered.label.slice(1)
              : numbered.label
            const dotCount =
              segment.kind === 'note'
                ? Math.min(2, Math.abs(numbered.octave))
                : 0
            const durationBeats = segment.duration
            const durationLines =
              durationBeats <= 0.3 ? 2 : durationBeats <= 0.7 ? 1 : 0
            const extensionCount =
              durationBeats >= 3.5
                ? 3
                : durationBeats >= 2.5
                  ? 2
                  : durationBeats >= 1.75
                    ? 1
                    : 0
            const dotted =
              Math.abs(durationBeats - 1.5) <= 0.18 ||
              Math.abs(durationBeats - 3) <= 0.25
            return (
              <g
                key={segment.id}
                className={`numbered-note is-${segment.kind}`}
                data-vocal-note={segment.note?.id}
                data-note-at={segment.note?.at}
                data-numbered-kind={segment.kind}
                data-numbered-duration={segment.duration}
              >
                {accidental ? (
                  <text
                    x={x - 6.5}
                    y="164"
                    className="numbered-accidental"
                  >
                    {accidental}
                  </text>
                ) : null}
                <text x={x} y="170" className="numbered-digit">
                  {numeral}
                </text>
                {Array.from({ length: extensionCount }, (_, index) => (
                  <line
                    key={`extension-${index}`}
                    x1={x + 9 + index * 8}
                    y1="166"
                    x2={x + 14 + index * 8}
                    y2="166"
                    className="numbered-extension"
                  />
                ))}
                {dotted ? (
                  <circle cx={x + 8} cy="166" r="1.4" className="numbered-dot" />
                ) : null}
                {Array.from({ length: durationLines }, (_, index) => (
                  <line
                    key={`duration-${index}`}
                    x1={x - 5}
                    y1={177 + index * 5}
                    x2={x + 5}
                    y2={177 + index * 5}
                    className="numbered-duration"
                  />
                ))}
                {Array.from({ length: dotCount }, (_, index) => (
                  <circle
                    key={index}
                    cx={x}
                    cy={
                      numbered.octave > 0
                        ? 157 - index * 5
                        : 188 + index * 5
                    }
                    r="1.1"
                  />
                ))}
              </g>
            )
          })}
          {lyricLayout.map(({ lyric, x, lane }, index) => (
            <text
              key={`${lyric.at}-${lyric.text}-${index}`}
              x={x}
              y={216 + lane * 18}
              className="lyric-token"
              data-lyric-at={lyric.at}
            >
              {lyric.text}
            </text>
          ))}
        </g>
      ) : null}
      <text x={width - 9} y="13" className="measure-index">
        {measure.number}
      </text>
    </svg>
  )
}

function ScoreMeasure({
  measure,
  tuning,
  capo,
  timing,
  timeSignature,
  relatedNotes,
  relationSourceIds,
  vocalMelody,
  lyrics,
  keySignature,
  melodyTonicPitch,
  showLabels,
  timelineWidth,
  selectedMeasureNumber,
  selectedNoteId,
  onSelectChord,
  onSelectBeat,
  onSelectNote,
}: {
  measure: TabMeasure
  tuning: StudioProject['tab']['tuning']
  capo: number
  timing: ScoreTiming
  timeSignature: StudioProject['tab']['timeSignature']
  relatedNotes: ReadonlyMap<string, TabNote>
  relationSourceIds: ReadonlySet<string>
  vocalMelody: StudioProject['tab']['vocalMelody']
  lyrics: StudioProject['tab']['lyrics']
  keySignature: string
  melodyTonicPitch: number
  showLabels: boolean
  timelineWidth?: number
  selectedMeasureNumber?: number
  selectedNoteId?: string
  onSelectChord: (measureNumber: number) => void
  onSelectBeat: (measureNumber: number, beatIndex: number) => void
  onSelectNote: (id: string) => void
}) {
  const chord = transposeChordForCapo(measure.chord, capo)

  return (
    <article
      className={`score-measure${
        selectedMeasureNumber === measure.number ? ' is-chord-selected' : ''
      }`}
      data-measure={measure.number}
      style={
        timelineWidth === undefined
          ? undefined
          : {
              width: `${timelineWidth}px`,
              minWidth: `${timelineWidth}px`,
              flexBasis: `${timelineWidth}px`,
            }
      }
    >
      {measure.section ? <span className="score-section-label">{measure.section}</span> : null}
      <header className="measure-chord">
        <button
          className="chord-diagram-button"
          type="button"
          aria-label={`校对第 ${measure.number} 小节 ${chord} 和弦`}
          onClick={() => onSelectChord(measure.number)}
        >
          <ChordDiagram name={chord} />
        </button>
        <span>
          <small>{rhythmKind(measure, timing)}</small>
        </span>
      </header>
      <TabNotation
        measure={measure}
        tuning={tuning}
        capo={capo}
        timing={timing}
        timeSignature={timeSignature}
        relatedNotes={relatedNotes}
        relationSourceIds={relationSourceIds}
        vocalMelody={vocalMelody}
        lyrics={lyrics}
        keySignature={keySignature}
        melodyTonicPitch={melodyTonicPitch}
        showLabels={showLabels}
        timelineWidth={timelineWidth}
        selectedNoteId={selectedNoteId}
        onSelectBeat={(beatIndex) => onSelectBeat(measure.number, beatIndex)}
        onSelectNote={onSelectNote}
      />
      {measure.lyric ? <p className="measure-lyric">{measure.lyric}</p> : null}
    </article>
  )
}

export function FullScore({
  project,
  positionSnapshot,
  playbackPositionRef,
  playing,
  speed,
  expanded,
  capo,
  selectedMeasureNumber,
  selectedNoteId,
  onSelectChord,
  onSelectBeat,
  onSelectNote,
}: {
  project: StudioProject
  positionSnapshot: number
  playbackPositionRef: MutableRefObject<number>
  playing: boolean
  speed: number
  expanded: boolean
  capo: number
  selectedMeasureNumber?: number
  selectedNoteId?: string
  onSelectChord: (measureNumber: number) => void
  onSelectBeat: (measureNumber: number, beatIndex: number) => void
  onSelectNote: (id: string) => void
}) {
  const scoreFlowRef = useRef<HTMLDivElement>(null)
  const scoreTrackRef = useRef<HTMLDivElement>(null)
  const timelineAnchorsRef = useRef<Array<{ at: number; x: number }>>([])
  const activeMeasureRef = useRef<number | undefined>(undefined)
  const visualPositionRef = useRef(positionSnapshot)
  const timing = useMemo(() => scoreTiming(project.tab), [project.tab])
  const relatedNotes = useMemo(
    () =>
      new Map(
        project.tab.measures.flatMap((measure) =>
          measure.beats.flatMap((beat) =>
            beat.notes.map((note) => [note.id, note] as const),
          ),
        ),
      ),
    [project.tab.measures],
  )
  const relationSourceIds = useMemo(
    () =>
      new Set(
        [...relatedNotes.values()]
          .map((note) => note.relatedNoteId)
          .filter((id): id is string => Boolean(id)),
      ),
    [relatedNotes],
  )
  const melodyTonicPitch = useMemo(() => {
    const pitches = (project.tab.vocalMelody ?? [])
      .map((note) => note.pitch)
      .sort((left, right) => left - right)
    const center = pitches[Math.floor(pitches.length / 2)] ?? 60
    const root = project.key.match(/^[A-G](?:#|b)?/)?.[0] ?? 'C'
    const pitchClass = numberedPitchClasses[root] ?? 0
    return pitchClass + 12 * Math.round((center - pitchClass) / 12)
  }, [project.key, project.tab.vocalMelody])

  const syncScorePosition = useCallback((position: number) => {
    const flow = scoreFlowRef.current
    const track = scoreTrackRef.current
    const firstMeasure = project.tab.measures[0]
    const lastMeasure = project.tab.measures.at(-1)
    const matchedMeasureIndex = project.tab.measures.findIndex(
      (item) => position >= item.start && position < item.start + item.duration,
    )
    const measureIndex =
      matchedMeasureIndex >= 0
        ? matchedMeasureIndex
        : position < (firstMeasure?.start ?? 0)
          ? 0
          : Math.max(0, project.tab.measures.length - 1)
    const measure = project.tab.measures[measureIndex] ?? lastMeasure
    const activeMeasure = measure?.number ?? 1
    const target = track?.querySelector<HTMLElement>(`[data-measure="${activeMeasure}"]`)
    if (!flow || !track || !target) return

    const measureChanged = activeMeasureRef.current !== activeMeasure
    if (measureChanged) {
      flow.querySelector('.score-measure.is-active')?.classList.remove('is-active')
      target.classList.add('is-active')
      activeMeasureRef.current = activeMeasure
      if (expanded) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' })
      }
    }

    if (expanded) {
      track.style.removeProperty('transform')
      return
    }
    // Bar-line spacing has no musical duration. Interpolate between the
    // rendered event positions so crossing that spacing remains continuous
    // without moving the playhead ahead of notes late in a measure.
    const anchors = timelineAnchorsRef.current
    let lower = 0
    let upper = anchors.length
    while (lower < upper) {
      const middle = Math.floor((lower + upper) / 2)
      if (anchors[middle].at <= position) lower = middle + 1
      else upper = middle
    }
    const before = anchors[Math.max(0, lower - 1)]
    const after = anchors[Math.min(anchors.length - 1, lower)]
    const progress =
      before && after && after.at > before.at
        ? Math.max(0, Math.min(1, (position - before.at) / (after.at - before.at)))
        : 0
    const timelineX =
      before && after
        ? before.x + (after.x - before.x) * progress
        : target.offsetLeft
    const translation = flow.clientWidth / 2 - timelineX
    track.style.transform = `translate3d(${translation}px, 0, 0)`
  }, [expanded, project.tab.measures])

  useLayoutEffect(() => {
    const flow = scoreFlowRef.current
    const track = scoreTrackRef.current
    if (!flow || !track || expanded) return
    const measureAnchors = () => {
      const trackLeft = track.getBoundingClientRect().left
      const anchors: Array<{ at: number; x: number; event: boolean }> = []
      for (const measure of project.tab.measures) {
        const notation = track.querySelector<SVGSVGElement>(
          `[data-measure="${measure.number}"] .tab-notation`,
        )
        const matrix = notation?.getScreenCTM()
        if (!notation || !matrix) continue
        anchors.push({
          at: measure.start,
          x: new DOMPoint(Number(notation.dataset.timelineStart), 0)
            .matrixTransform(matrix).x - trackLeft,
          event: false,
        })
        if (measure === project.tab.measures.at(-1)) {
          anchors.push({
            at: measure.start + measure.duration,
            x: new DOMPoint(Number(notation.dataset.timelineEnd), 0)
              .matrixTransform(matrix).x - trackLeft,
            event: false,
          })
        }
      }
      for (const glyph of track.querySelectorAll<SVGGraphicsElement>(
        '.tab-glyph[data-notation-at], .strum-mark[data-notation-at]',
      )) {
        const bounds = (glyph.querySelector('text') ?? glyph).getBoundingClientRect()
        anchors.push({
          at: Number(glyph.dataset.notationAt),
          x: (bounds.left + bounds.right) / 2 - trackLeft,
          event: true,
        })
      }
      timelineAnchorsRef.current = anchors
        .sort((left, right) => left.at - right.at || Number(right.event) - Number(left.event))
        .filter((anchor, index, all) => index === 0 || anchor.at !== all[index - 1].at)
      syncScorePosition(playbackPositionRef.current)
    }
    measureAnchors()
    const observer = new ResizeObserver(measureAnchors)
    observer.observe(flow)
    return () => observer.disconnect()
  }, [capo, expanded, playbackPositionRef, project.tab.measures, syncScorePosition])

  useLayoutEffect(() => {
    if (playing) return
    syncScorePosition(playbackPositionRef.current)
  }, [playbackPositionRef, playing, positionSnapshot, syncScorePosition])

  useLayoutEffect(() => {
    if (!playing) return
    let animation = 0
    let previousFrame: number | undefined
    let visualPosition = playbackPositionRef.current
    visualPositionRef.current = visualPosition
    const tick = (now: number) => {
      const elapsed = previousFrame === undefined ? 0 : (now - previousFrame) / 1000
      previousFrame = now
      visualPosition += elapsed * speed
      const playbackDrift = playbackPositionRef.current - visualPosition
      const shouldReanchor = Math.abs(playbackDrift) > 0.08
      if (shouldReanchor) {
        visualPosition = playbackPositionRef.current
      }
      visualPositionRef.current = visualPosition
      syncScorePosition(visualPosition)
      animation = requestAnimationFrame(tick)
    }
    animation = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animation)
  }, [playbackPositionRef, playing, speed, syncScorePosition])

  const measures = useMemo(
    () =>
      project.tab.measures.map((measure, index) => (
        <ScoreMeasure
          key={measure.number}
          measure={measure}
          tuning={project.tab.tuning}
          capo={capo}
          timing={timing}
          timeSignature={project.tab.timeSignature}
          relatedNotes={relatedNotes}
          relationSourceIds={relationSourceIds}
          vocalMelody={project.tab.vocalMelody}
          lyrics={project.tab.lyrics}
          keySignature={project.key}
          melodyTonicPitch={melodyTonicPitch}
          showLabels={expanded || index === 0}
          timelineWidth={
            expanded
              ? undefined
              : measure.duration * 150 + (measure.number === 1 ? 44 : 0)
          }
          selectedMeasureNumber={selectedMeasureNumber}
          selectedNoteId={selectedNoteId}
          onSelectChord={onSelectChord}
          onSelectBeat={onSelectBeat}
          onSelectNote={onSelectNote}
        />
      )),
    [
      capo,
      expanded,
      onSelectBeat,
      onSelectChord,
      onSelectNote,
      project.tab.measures,
      project.tab.lyrics,
      project.tab.tuning,
      project.tab.timeSignature,
      project.tab.vocalMelody,
      project.key,
      melodyTonicPitch,
      timing,
      relatedNotes,
      relationSourceIds,
      selectedMeasureNumber,
      selectedNoteId,
    ],
  )

  return (
    <div
      className={`full-score${expanded ? ' is-expanded' : ' is-timeline'}${
        playing ? ' is-playing' : ''
      }`}
    >
      {!expanded ? <i className="score-playhead" aria-hidden="true" /> : null}
      <div className="score-flow" ref={scoreFlowRef}>
        <div
          className="score-track"
          ref={scoreTrackRef}
        >
          {measures}
        </div>
      </div>
    </div>
  )
}
