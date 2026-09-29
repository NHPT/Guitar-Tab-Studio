import type { TabMeasure } from './types'

const noteValues: Record<string, number> = {
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

const sharpNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
const flatNames = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

const shapeCosts: Record<string, number> = {
  C: 1,
  C7: 1,
  Cmaj7: 1,
  D: 1,
  Dm: 1,
  D7: 1,
  Dm7: 1,
  E: 1,
  Em: 1,
  E7: 1,
  Em7: 1,
  F: 6,
  Fmaj7: 1,
  G: 1,
  G7: 1,
  A: 1,
  Am: 1,
  Am9: 1,
  A7: 1,
  Am7: 1,
  B7: 3,
  Bm: 6,
}

function parseChord(name: string): { root: string; suffix: string } | null {
  const match = /^([A-G](?:#|b)?)(.*)$/.exec(name.trim())
  return match ? { root: match[1], suffix: match[2] } : null
}

export function transposeChordForCapo(name: string, capo: number): string {
  const parsed = parseChord(name)
  if (!parsed || capo === 0) return name
  const value = noteValues[parsed.root]
  if (value === undefined) return name
  const names = parsed.root.includes('b') ? flatNames : sharpNames
  return `${names[(value - capo + 120) % 12]}${parsed.suffix}`
}

export function recommendCapo(measures: TabMeasure[]): number {
  const chords = measures.map((measure) => measure.chord).filter((chord) => parseChord(chord))
  if (chords.length === 0) return 0

  const score = (capo: number) =>
    chords.reduce(
      (total, chord) => total + (shapeCosts[transposeChordForCapo(chord, capo)] ?? 12),
      0,
    ) +
    capo * 0.18

  let bestCapo = 0
  let bestScore = score(0)
  for (let capo = 1; capo <= 7; capo += 1) {
    const nextScore = score(capo)
    if (nextScore < bestScore - 0.5) {
      bestCapo = capo
      bestScore = nextScore
    }
  }
  return bestCapo
}
