import assert from 'node:assert/strict'
import test from 'node:test'
import { beatPosition, guitarGestures, rhythmValue } from '../src/performance.js'
import type { TabNote } from '../src/domain.js'

function note(at: number, string: TabNote['string'], technique: TabNote['technique'] = 'unknown'): TabNote {
  return { id: `${at}-${string}`, at, string, fret: 3, duration: 0.2, technique, confidence: 0.85 }
}

test('a sweep crossing a beat boundary preserves all string attacks and excludes the next bass pick', () => {
  const notes = [note(0.46, 1), note(0.49, 2), note(0.515, 3), note(0.53, 4), note(0.69, 6)]
  const before = structuredClone(notes)
  const gestures = guitarGestures(notes, () => 0.5)
  assert.equal(gestures.length, 1)
  assert.equal(gestures[0].style, 'strum')
  assert.equal(gestures[0].direction, 'up')
  assert.deepEqual(gestures[0].notes.map((n) => n.string), [1, 2, 3, 4])
  assert.deepEqual(notes, before)
})

test('simultaneous string pairs do not reverse the independently observed sweep direction', () => {
  const gesture = guitarGestures([note(0.1, 5), note(0.1, 6), note(0.15, 4)], () => 0.5)[0]
  assert.equal(gesture.direction, 'down')
  assert.equal(gesture.style, 'strum')
})

test('a sparse simultaneous dyad followed by one note is not enough sweep evidence', () => {
  assert.deepEqual(
    guitarGestures(
      [note(0.1, 1), note(0.1, 3), note(0.15, 4)],
      () => 0.5,
    ),
    [],
  )
})

test('simultaneous plucks, repeated picks and legato do not acquire sweep labels', () => {
  assert.deepEqual(guitarGestures([note(0.1, 6), note(0.1, 5), note(0.1, 4)], () => 0.5), [])
  assert.deepEqual(guitarGestures([0.1, 0.15, 0.2, 0.25].map((at) => note(at, 3)), () => 0.5), [])
  assert.deepEqual(guitarGestures([note(0.1, 6), note(0.15, 5), note(0.2, 4, 'hammer-on')], () => 0.5), [])
})

test('two separate sweeps in one beat remain two gestures', () => {
  const notes = [note(0.03, 6), note(0.065, 5), note(0.1, 4),
    note(0.27, 3), note(0.305, 4), note(0.34, 5)]
  assert.deepEqual(guitarGestures(notes, () => 0.5).map((g) => g.direction), ['down', 'up'])
})

test('slower ordered attacks remain arpeggios, without inferring rasgueado', () => {
  const gesture = guitarGestures([note(0.1, 6), note(0.23, 5), note(0.36, 4)], () => 0.5)[0]
  assert.equal(gesture.style, 'arpeggio')
})

test('rhythmic values represent quarters, dotted eighths and sustained halves', () => {
  assert.deepEqual(rhythmValue(0.94), { beats: 1, flags: 0, dotted: false, hollow: false })
  assert.deepEqual(rhythmValue(0.72), { beats: 0.75, flags: 1, dotted: true, hollow: false })
  assert.deepEqual(rhythmValue(2.1), { beats: 2, flags: 0, dotted: false, hollow: true })
})

test('beat coordinates integrate changing local tempo instead of global BPM', () => {
  const beats = [{ at: 0, duration: 0.4 }, { at: 0.4, duration: 0.8 }, { at: 1.2, duration: 0.6 }]
  assert.equal(beatPosition(0.2, beats), 0.5)
  assert.equal(beatPosition(1, beats), 1.75)
  assert.equal(beatPosition(1.5, beats), 2.5)
})
