import { useCallback, useLayoutEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import type { StudioProject, TabMeasure, TabNote, Technique } from './types'
import { transposeChordForCapo } from './capo'
import { inferBeatStyle } from './rhythm'

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

function rhythmKind(measure: TabMeasure): string {
  const styles = measure.beats
    .filter((beat) => beat.notes.length > 0)
    .map(inferBeatStyle)
  if (styles.length === 0) return '休止'
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

function TabNotation({
  measure,
  tuning,
  capo,
  chordShape,
  relatedNotes,
  relationSourceIds,
  showLabels,
  timelineWidth,
  selectedNoteId,
  onSelectBeat,
  onSelectNote,
}: {
  measure: TabMeasure
  tuning: StudioProject['tab']['tuning']
  capo: number
  chordShape: ChordShape | null
  relatedNotes: ReadonlyMap<string, TabNote>
  relationSourceIds: ReadonlySet<string>
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
  const lineStart = showLabels ? 32 : 0
  const timelineStart =
    lineStart + (measure.number === 1 ? 26 : showLabels ? 8 : 0)
  const timelineEnd = width
  const xAtTime = (at: number) => {
    const ratio = Math.max(
      0,
      Math.min(1, (at - measure.start) / Math.max(0.01, measure.duration)),
    )
    return timelineStart + ratio * (timelineEnd - timelineStart)
  }
  const measureNotes = measure.beats.flatMap((beat) => beat.notes)
  const legatoTargets = measureNotes.filter(
    (note) =>
      note.relatedNoteId && relationalTechniques.has(note.technique),
  )
  const notationAtById = new Map(
    measureNotes.map((note) => {
      if (note.notationAt !== undefined) {
        return [note.id, note.notationAt] as const
      }
      if (
        relationSourceIds.has(note.id) ||
        !['pick', 'palm-mute'].includes(note.technique)
      ) {
        return [note.id, note.at] as const
      }
      const nearbyTarget = legatoTargets
        .filter(
          (target) =>
            Math.abs((target.notationAt ?? target.at) - note.at) <= 0.085,
        )
        .sort(
          (left, right) =>
            Math.abs((left.notationAt ?? left.at) - note.at) -
            Math.abs((right.notationAt ?? right.at) - note.at),
        )[0]
      return [
        note.id,
        nearbyTarget?.notationAt ?? nearbyTarget?.at ?? note.at,
      ] as const
    }),
  )
  const notationAt = (note: TabNote) =>
    notationAtById.get(note.id) ?? note.at
  const noteX = (note: TabNote) => xAtTime(notationAt(note))
  const rhythmGroups = measure.beats.map((beat) => {
    const notes = [...beat.notes].sort(
      (left, right) => notationAt(left) - notationAt(right),
    )
    const style = inferBeatStyle(beat)
    const hasRelationDetail = notes.some(
      (note) =>
        relationalTechniques.has(note.technique) ||
        relationSourceIds.has(note.id),
    )
    if (
      notes.length > 0 &&
      (style === 'strum' || style === 'rasgueado') &&
      !hasRelationDetail
    ) {
      return [{ at: notationAt(notes[0]), notes }]
    }

    return notes.reduce<Array<{ at: number; notes: TabNote[] }>>(
      (groups, note) => {
        const previous = groups.at(-1)
        const at = notationAt(note)
        if (previous && at - previous.at <= 0.04) {
          previous.notes.push(note)
        } else {
          groups.push({ at, notes: [note] })
        }
        return groups
      },
      [],
    )
  })
  const positionedNotes = measure.beats.flatMap((beat) => {
    const style = inferBeatStyle(beat)
    const hasRelationDetail = beat.notes.some(
      (note) =>
        relationalTechniques.has(note.technique) ||
        relationSourceIds.has(note.id),
    )
    const hiddenByStrum =
      chordShape !== null &&
      (style === 'strum' || style === 'rasgueado') &&
      !hasRelationDetail
    if (hiddenByStrum) return []
    return beat.notes.map((note) => ({
      note,
      x: noteX(note),
      y: staffTop + (note.string - 1) * stringGap,
    }))
  })
  const notePositions = new Map(
    positionedNotes.map((position) => [position.note.id, position]),
  )

  return (
    <svg
      className="tab-notation"
      viewBox={`0 0 ${width} 154`}
      data-timeline-start={timelineStart}
      data-timeline-end={timelineEnd}
      role="img"
      aria-label={`第 ${measure.number} 小节 ${rhythmKind(measure)}六线谱`}
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
        <g className="time-signature" aria-label="四四拍">
          <text x={lineStart + 14} y={staffTop + 25}>
            4
          </text>
          <text x={lineStart + 14} y={staffTop + 45}>
            4
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

      {measure.beats.map((beat, beatIndex) => {
        const groups = rhythmGroups[beatIndex]
        const x = xAtTime(
          groups[0]?.at ?? beat.at + beat.duration / 2,
        )
        const style = inferBeatStyle(beat)
        const hasRelationDetail = beat.notes.some(
          (note) =>
            relationalTechniques.has(note.technique) ||
            relationSourceIds.has(note.id),
        )
        const isStrum =
          chordShape !== null &&
          (style === 'strum' || style === 'rasgueado') &&
          !hasRelationDetail
        const isArpeggio = style === 'arpeggio'
        const isRasgueado = style === 'rasgueado'
        const isTremolo =
          style === 'tremolo' ||
          beat.notes.some((note) => note.technique === 'tremolo')
        const visibleNotes = isStrum ? [] : beat.notes
        return (
          <g key={`${measure.number}-beat-${beatIndex}`}>
            {isStrum ? (
              <g
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
                <line
                  x1={x}
                  y1={beat.direction === 'up' ? staffBottom + 4 : staffTop - 4}
                  x2={x}
                  y2={beat.direction === 'up' ? staffTop - 4 : staffBottom + 4}
                />
                <path
                  d={
                    beat.direction === 'up'
                      ? `M ${x - 4} ${staffTop + 2} L ${x} ${staffTop - 4} L ${
                          x + 4
                        } ${staffTop + 2}`
                      : `M ${x - 4} ${staffBottom - 2} L ${x} ${
                          staffBottom + 4
                        } L ${x + 4} ${staffBottom - 2}`
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

      {measure.beats.map((beat, beatIndex) => {
        const groups = rhythmGroups[beatIndex]
        const beatCenterX = xAtTime(beat.at + beat.duration / 2)
        if (groups.length === 0) {
          return (
            <g
              key={`${measure.number}-rhythm-${beatIndex}`}
              className="rhythm-beat-slot is-rest"
              data-beat-index={beatIndex}
              data-event-count="0"
            >
              <path
                className="rhythm-rest"
                d={`M ${beatCenterX - 3} 117 l 6 4 -5 5 5 5 -4 7`}
              />
            </g>
          )
        }

        const groupXs = groups.map((group) => xAtTime(group.at))
        const normalizedGaps = groups
          .slice(1)
          .map(
            (group, index) =>
              (group.at - groups[index].at) / Math.max(0.01, beat.duration),
          )
        const minimumGap =
          normalizedGaps.length > 0 ? Math.min(...normalizedGaps) : 1
        const shortestDuration = Math.min(
          ...groups[0].notes.map((note) => note.duration),
        )
        const isTriplet = groups.length === 3
        const beamCount =
          isTriplet
            ? 1
            : groups.length >= 2
            ? minimumGap <= 0.34
              ? 2
              : 1
            : shortestDuration <= beat.duration * 0.35
              ? 2
              : shortestDuration <= beat.duration * 0.75
                ? 1
                : 0
        return (
          <g
            key={`${measure.number}-rhythm-${beatIndex}`}
            className="rhythm-beat-slot"
            data-beat-index={beatIndex}
            data-event-count={groups.length}
            data-tuplet={isTriplet ? '3' : undefined}
          >
            {groupXs.map((x) => (
              <line
                key={`${measure.number}-${beatIndex}-${x}`}
                x1={x}
                y1={staffBottom + 5}
                x2={x}
                y2="137"
                className="rhythm-stem"
              />
            ))}
            {groups.length >= 2
              ? Array.from({ length: beamCount }, (_, beamIndex) => (
                  <line
                    key={`${measure.number}-${beatIndex}-beam-${beamIndex}`}
                    x1={groupXs[0]}
                    y1={137 + beamIndex * 4}
                    x2={groupXs.at(-1)}
                    y2={137 + beamIndex * 4}
                    className="rhythm-beam"
                  />
                ))
              : Array.from({ length: beamCount }, (_, flagIndex) => (
                  <line
                    key={`${measure.number}-${beatIndex}-flag-${flagIndex}`}
                    x1={groupXs[0]}
                    y1={137 + flagIndex * 4}
                    x2={groupXs[0] + 8}
                    y2={140 + flagIndex * 4}
                    className="rhythm-flag"
                  />
                ))}
            {isTriplet ? (
              <text
                x={(groupXs[0] + groupXs.at(-1)!) / 2}
                y="150"
                className="rhythm-tuplet"
              >
                3
              </text>
            ) : null}
          </g>
        )
      })}
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
  relatedNotes,
  relationSourceIds,
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
  relatedNotes: ReadonlyMap<string, TabNote>
  relationSourceIds: ReadonlySet<string>
  showLabels: boolean
  timelineWidth?: number
  selectedMeasureNumber?: number
  selectedNoteId?: string
  onSelectChord: (measureNumber: number) => void
  onSelectBeat: (measureNumber: number, beatIndex: number) => void
  onSelectNote: (id: string) => void
}) {
  const chord = transposeChordForCapo(measure.chord, capo)
  const chordShape = getChordShape(chord)

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
          <small>{rhythmKind(measure)}</small>
        </span>
      </header>
      <TabNotation
        measure={measure}
        tuning={tuning}
        capo={capo}
        chordShape={chordShape}
        relatedNotes={relatedNotes}
        relationSourceIds={relationSourceIds}
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
  const activeMeasureRef = useRef<number | undefined>(undefined)
  const visualPositionRef = useRef(positionSnapshot)
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
    const measureProgress = measure
      ? Math.max(0, Math.min(1, (position - measure.start) / measure.duration))
      : 0
    const notation = target.querySelector<SVGSVGElement>('.tab-notation')
    const viewBoxWidth = notation?.viewBox.baseVal.width ?? 0
    const timelineStart = Number(notation?.dataset.timelineStart)
    const timelineEnd = Number(notation?.dataset.timelineEnd)
    const renderedTimelineX = (
      measureElement: HTMLElement,
      timelineCoordinate: number,
    ) =>
      measureElement.offsetLeft +
      (timelineCoordinate / viewBoxWidth) * measureElement.clientWidth
    const startX =
      viewBoxWidth > 0 && Number.isFinite(timelineStart)
        ? renderedTimelineX(target, timelineStart)
        : target.offsetLeft
    const nextTarget = track.querySelector<HTMLElement>(
      `[data-measure="${activeMeasure + 1}"]`,
    )
    const nextNotation =
      nextTarget?.querySelector<SVGSVGElement>('.tab-notation')
    const nextTimelineStart = Number(nextNotation?.dataset.timelineStart)
    const endX =
      nextTarget &&
      nextNotation &&
      viewBoxWidth > 0 &&
      Number.isFinite(nextTimelineStart)
        ? renderedTimelineX(nextTarget, nextTimelineStart)
        : viewBoxWidth > 0 && Number.isFinite(timelineEnd)
          ? renderedTimelineX(target, timelineEnd)
          : target.offsetLeft + target.offsetWidth
    const timelineX = startX + (endX - startX) * measureProgress
    const translation = flow.clientWidth / 2 - timelineX
    track.style.transform = `translate3d(${translation}px, 0, 0)`
  }, [expanded, project.tab.measures])

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
          relatedNotes={relatedNotes}
          relationSourceIds={relationSourceIds}
          showLabels={expanded || index === 0}
          timelineWidth={
            expanded
              ? undefined
              : measure.duration * 150 + (measure.number === 1 ? 58 : 0)
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
      project.tab.tuning,
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
