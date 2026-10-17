import type {
  EasyArrangement,
  StudioProject,
  TabBeat,
  TabDocument,
  TabNote,
} from './domain.js'

const OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40] as const
const MAX_FRET = 24

interface Position {
  string: TabNote['string']
  fret: number
}

export function noteMidiPitch(note: Pick<TabNote, 'string' | 'fret'>): number {
  return OPEN_STRING_MIDI[note.string - 1] + note.fret
}

function positionsFor(note: TabNote, related?: Position): Position[] {
  if (note.technique === 'harmonic' || !Number.isInteger(note.fret)) {
    return [{ string: note.string, fret: note.fret }]
  }
  const pitch = noteMidiPitch(note)
  return OPEN_STRING_MIDI.flatMap((openPitch, index) => {
    const fret = pitch - openPitch
    return Number.isInteger(fret) && fret >= 0 && fret <= MAX_FRET
      ? [{ string: (index + 1) as TabNote['string'], fret }]
      : []
  }).sort((left, right) => {
    const score = (position: Position) =>
      position.fret +
      Math.max(0, position.fret - 12) * 3 -
      (position.fret === 0 ? 5 : 0) +
      (related && position.string !== related.string ? 100 : 0)
    return score(left) - score(right)
  })
}

function assignmentCost(
  notes: TabNote[],
  assignment: Position[],
  previousFret: number,
  assignedNotes: Map<string, Position>,
): number {
  const fretted = assignment
    .map((position) => position.fret)
    .filter((fret) => fret > 0)
  const center =
    fretted.length > 0
      ? fretted.reduce((total, fret) => total + fret, 0) / fretted.length
      : previousFret
  const spread =
    fretted.length > 1 ? Math.max(...fretted) - Math.min(...fretted) : 0
  return assignment.reduce((total, position, index) => {
    const note = notes[index]
    const related = note.relatedNoteId
      ? assignedNotes.get(note.relatedNoteId)
      : undefined
    return (
      total +
      position.fret * 0.35 +
      Math.max(0, position.fret - 12) * 2.8 +
      Math.abs(position.fret - previousFret) * 0.12 +
      (position.fret === 0 ? -2.5 : 0) +
      (position.string === note.string ? -0.2 : 0) +
      (related && related.string !== position.string ? 100 : 0)
    )
  }, spread * 0.9 + Math.abs(center - previousFret) * 0.45)
}

function optimizeBeat(
  beat: TabBeat,
  previousFret: number,
  assignedNotes: Map<string, Position>,
): Position[] {
  if (beat.notes.length === 0 || beat.notes.length > 6) {
    return beat.notes.map(({ string, fret }) => ({ string, fret }))
  }
  const candidates = beat.notes.map((note) =>
    positionsFor(
      note,
      note.relatedNoteId ? assignedNotes.get(note.relatedNoteId) : undefined,
    ).slice(0, 5),
  )
  let best: Position[] | undefined
  let bestCost = Number.POSITIVE_INFINITY

  const visit = (index: number, current: Position[], used: Set<number>) => {
    if (index === candidates.length) {
      const cost = assignmentCost(
        beat.notes,
        current,
        previousFret,
        assignedNotes,
      )
      if (cost < bestCost) {
        bestCost = cost
        best = current.map((position) => ({ ...position }))
      }
      return
    }
    for (const candidate of candidates[index]) {
      if (used.has(candidate.string)) continue
      used.add(candidate.string)
      current.push(candidate)
      visit(index + 1, current, used)
      current.pop()
      used.delete(candidate.string)
    }
  }
  visit(0, [], new Set())
  return (
    best ??
    beat.notes.map(({ string, fret }) => ({
      string,
      fret,
    }))
  )
}

export function createEasyTab(original: TabDocument): TabDocument {
  const assignedNotes = new Map<string, Position>()
  let previousFret = 3
  const measures = original.measures.map((measure) => ({
    ...measure,
    beats: measure.beats.map((beat) => {
      const positions = optimizeBeat(beat, previousFret, assignedNotes)
      const fretted = positions.filter((position) => position.fret > 0)
      if (fretted.length > 0) {
        previousFret =
          fretted.reduce((total, position) => total + position.fret, 0) /
          fretted.length
      }
      const notes = beat.notes.map((note, index) => {
        const position = positions[index]
        assignedNotes.set(note.id, position)
        return {
          ...note,
          ...position,
          positionSource: 'easy-arrangement-v1',
        }
      })
      return { ...beat, notes }
    }),
  }))
  return {
    tuning: [...original.tuning],
    capo: original.capo,
    timeSignature: [...original.timeSignature],
    vocalMelody: original.vocalMelody?.map((note) => ({ ...note })),
    lyrics: original.lyrics?.map((lyric) => ({ ...lyric })),
    measures,
  }
}

export function createEasyArrangement(
  project: StudioProject,
  createdAt = new Date().toISOString(),
): EasyArrangement {
  return {
    kind: 'easy',
    strategyVersion: 'pitch-preserving-v1',
    createdAt,
    tab: createEasyTab(project.tab),
  }
}

export function arrangementPreservesPitch(
  original: TabDocument,
  arrangement: TabDocument,
): boolean {
  const originalNotes = original.measures.flatMap((measure) =>
    measure.beats.flatMap((beat) => beat.notes),
  )
  const arrangedNotes = arrangement.measures.flatMap((measure) =>
    measure.beats.flatMap((beat) => beat.notes),
  )
  return (
    originalNotes.length === arrangedNotes.length &&
    originalNotes.every((note, index) => {
      const arranged = arrangedNotes[index]
      return (
        arranged?.id === note.id &&
        noteMidiPitch(arranged) === noteMidiPitch(note)
      )
    })
  )
}
