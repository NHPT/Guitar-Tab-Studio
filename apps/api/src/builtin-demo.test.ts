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

test('bundles the requested Internationale recording', () => {
  const audio = readFileSync(builtinDemoAudioPath)
  assert.equal(audio.subarray(0, 4).toString('ascii'), 'OggS')
  assert.equal(audio.byteLength, 3_340_717)
  assert.equal(
    createHash('sha256').update(audio).digest('hex'),
    'ad7b28cd453b33a45e84932d97ccc27ec2fe896b3ac7b8d4ab898eb3de5af181',
  )
})

test('loads the generated Internationale TAB analysis', () => {
  const analysis = builtinDemoAnalysis
  assert.equal(analysis.metadata?.title, '国际歌')
  assert.equal(analysis.metadata?.artist, '唐朝')
  assert.ok(Math.abs(analysis.metadata!.duration! - 266.73333) < 0.001)
  assert.equal(analysis.bpm, 98.1)
  assert.equal(analysis.key, 'G major')
  assert.equal(analysis.capo, 0)
  assert.equal(analysis.pickup_beats, 1)
  assert.equal(analysis.beats?.[0], 0.3317)
  assert.equal(analysis.beats?.[1], 0.9433)
  assert.equal(analysis.chords.length, 107)
  assert.equal(analysis.notes.length, 1462)
  assert.equal(analysis.tempo_map?.mode, 'steady')
  assert.equal(analysis.tempo_map?.points.length, 437)
  assert.equal(
    analysis.transcription_model?.positionOptimizer,
    'sequence-optimizer-v3',
  )
  assert.equal(
    analysis.transcription_model?.timingModel,
    'percussion-downbeat-grid-v2',
  )
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
    '6ffff90356b97b53dff273ad1e946e313c803008ccfda868cf289df124d9cae0',
  )
  for (const media of builtinDemoMediaPaths.values()) {
    const contents = readFileSync(media)
    if (contents.subarray(0, 4).toString('ascii') === 'OggS') continue
    assert.equal(contents.subarray(4, 8).toString('ascii'), 'ftyp')
  }
  assert.equal(builtinDemoAnalysis.stem_tracks?.length, 0)
  assert.equal(builtinDemoAnalysis.guitar_parts?.length, 1)
  assert.equal(builtinDemoMediaPaths.size, 7)
})
