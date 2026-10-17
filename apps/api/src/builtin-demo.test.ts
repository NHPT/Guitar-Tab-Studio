import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import {
  builtinDemoAnalysis,
  builtinDemoAudioPath,
  builtinDemoMediaPaths,
  builtinDemoStemFilenames,
} from './builtin-demo.js'

test('bundles the deterministic public-domain beginner recording', () => {
  const audio = readFileSync(builtinDemoAudioPath)
  assert.equal(audio.subarray(0, 4).toString('ascii'), 'RIFF')
  assert.equal(audio.subarray(8, 12).toString('ascii'), 'WAVE')
  assert.equal(audio.byteLength, 15_394_954)
  assert.equal(
    createHash('sha256').update(audio).digest('hex'),
    '37e75840946e3bcdece68c1df7dd8f164a53ccd2942e650411f32704a928483e',
  )
})

test('builds the authored beginner TAB analysis', () => {
  const analysis = builtinDemoAnalysis
  assert.equal(analysis.metadata?.title, 'Wayfaring Stranger')
  assert.equal(
    analysis.metadata?.artist,
    'American Traditional · Guitar Tab Studio 初学编配',
  )
  assert.ok(Math.abs(analysis.metadata!.duration! - 174.54545) < 0.001)
  assert.equal(analysis.bpm, 88)
  assert.equal(analysis.key, 'E minor')
  assert.equal(analysis.capo, 0)
  assert.equal(analysis.chords.length, 64)
  assert.ok(analysis.notes.length > 200)
  assert.equal(
    analysis.transcription_model?.name,
    'authored-beginner-arrangement',
  )
  assert.ok(
    analysis.notes.every(
      (note) =>
        note.string >= 1 &&
        note.string <= 6 &&
        note.fret >= 0 &&
        note.fret <= 24 &&
        note.confidence === 1 &&
        note.pitch ===
          ({ 1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40 } as const)[
            note.string
          ] +
            note.fret,
    ),
  )
})

test('publishes independent authored guitar and guide-vocal tracks', () => {
  const guitar = readFileSync(
    builtinDemoMediaPaths.get(builtinDemoStemFilenames.guitar)!,
  )
  const vocals = readFileSync(
    builtinDemoMediaPaths.get(builtinDemoStemFilenames.vocals)!,
  )
  assert.notEqual(
    createHash('sha256').update(guitar).digest('hex'),
    createHash('sha256').update(vocals).digest('hex'),
  )
  assert.equal(
    createHash('sha256').update(vocals).digest('hex'),
    '8bc008b2badbad083b46ab95977ab51466049ee6abf0a479fc0a48a0cec627be',
  )
  for (const media of [guitar, vocals]) {
    assert.equal(media.subarray(0, 4).toString('ascii'), 'RIFF')
    assert.equal(media.subarray(8, 12).toString('ascii'), 'WAVE')
  }
  assert.equal(builtinDemoAnalysis.stem_tracks, undefined)
})
