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

test('bundles the requested NetEase default recording', () => {
  const audio = readFileSync(builtinDemoAudioPath)
  assert.equal(audio.subarray(0, 3).toString('ascii'), 'ID3')
  assert.equal(audio.byteLength, 3_107_570)
  assert.equal(
    createHash('sha256').update(audio).digest('hex'),
    '0fb7faa72934b35b7f5a3adb65b1ac7cbb49b128692da6fe42c4cc8b90624608',
  )
})

test('loads the generated Country Roads TAB analysis', () => {
  const analysis = builtinDemoAnalysis
  assert.equal(analysis.metadata?.title, 'Take Me Home, Country Roads')
  assert.equal(analysis.metadata?.artist, 'John Denver')
  assert.ok(Math.abs(analysis.metadata!.duration! - 194.14667) < 0.001)
  assert.equal(analysis.bpm, 161.5)
  assert.equal(analysis.key, 'A major')
  assert.equal(analysis.capo, 0)
  assert.equal(analysis.chords.length, 130)
  assert.equal(analysis.notes.length, 1789)
  assert.equal(analysis.transcription_model?.name, 'spotify-basic-pitch')
  assert.ok(
    analysis.notes.every(
      (note) =>
        note.string >= 1 &&
        note.string <= 6 &&
        note.fret >= 0 &&
        note.fret <= 24 &&
        note.confidence >= 0 &&
        note.confidence <= 1 &&
        note.pitch ===
          ({ 1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40 } as const)[
            note.string
          ] +
            note.fret,
    ),
  )
})

test('publishes independent M4A stems and separated guitar parts', () => {
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
    '05a349e17268dafc12e1b780542bb2e930fcdc29f9c53493c37475f37d880c75',
  )
  for (const media of builtinDemoMediaPaths.values()) {
    const contents = readFileSync(media)
    if (contents.subarray(0, 3).toString('ascii') === 'ID3') continue
    assert.equal(contents.subarray(4, 8).toString('ascii'), 'ftyp')
  }
  assert.equal(builtinDemoAnalysis.stem_tracks?.length, 2)
  assert.equal(builtinDemoMediaPaths.size, 9)
})
