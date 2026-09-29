import type { BeatStyle, TabBeat } from './types'

export function inferBeatStyle(beat: TabBeat): BeatStyle {
  if (beat.style) return beat.style

  const notes = [...beat.notes].sort((left, right) => left.at - right.at)
  const uniqueStrings = new Set(notes.map((note) => note.string))
  const spread = notes.length > 1 ? notes.at(-1)!.at - notes[0].at : 0
  const stringDeltas = notes
    .slice(1)
    .map((note, index) => note.string - notes[index].string)
    .filter((delta) => delta !== 0)
  const direction = Math.sign(stringDeltas[0] ?? 0)
  const directionalRun =
    stringDeltas.length >= 2 &&
    direction !== 0 &&
    stringDeltas.every(
      (delta) => Math.sign(delta) === direction && Math.abs(delta) <= 3,
    )
  const maximumOnsetGap = notes
    .slice(1)
    .reduce(
      (maximum, note, index) =>
        Math.max(maximum, note.at - notes[index].at),
      0,
    )
  const containsLegato = notes.some((note) =>
    ['hammer-on', 'pull-off', 'slide'].includes(note.technique),
  )
  const rapidRepeatedString =
    notes.length >= 4 &&
    uniqueStrings.size === 1 &&
    maximumOnsetGap <= Math.min(0.14, beat.duration * 0.3)

  if (
    rapidRepeatedString ||
    notes.filter((note) => note.technique === 'tremolo').length >= 2
  ) {
    return 'tremolo'
  }
  if (
    uniqueStrings.size < 3 ||
    containsLegato ||
    !directionalRun ||
    maximumOnsetGap > Math.min(0.18, beat.duration * 0.5)
  ) {
    return 'pick'
  }
  if (spread <= 0.035) return 'pick'
  if (spread <= 0.14) return 'strum'
  if (spread <= Math.min(0.26, beat.duration * 0.55) && notes.length >= 5) {
    return 'rasgueado'
  }
  return spread <= beat.duration * 0.85 ? 'arpeggio' : 'pick'
}
