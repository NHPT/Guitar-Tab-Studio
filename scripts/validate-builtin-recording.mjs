import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const assets = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../apps/api/assets',
)
const read = (name) => readFileSync(resolve(assets, name))
const sha256 = (value) => createHash('sha256').update(value).digest('hex')
const analysis = JSON.parse(
  read('internationale-analysis.json'),
)
const expectedHashes = {
  'internationale.ogg':
    'ad7b28cd453b33a45e84932d97ccc27ec2fe896b3ac7b8d4ab898eb3de5af181',
  'internationale-vocals.m4a':
    '6ffff90356b97b53dff273ad1e946e313c803008ccfda868cf289df124d9cae0',
  'internationale-guitar.m4a':
    'eb96bde063b5205a1cbbf4a87a3600e0bfbbb733714eaa9637daaf58dcf17433',
  'internationale-bass.m4a':
    '60cb1dc3a2740ede1f2a3f6526f4981b5f0b56814a676ea10e2b4c7b7aa56167',
  'internationale-drums.m4a':
    '13b281b5dbc91e0df6d59d100b5346fd97773458da56041f114b9c0517a8f53e',
  'internationale-piano.m4a':
    'ed4ac00ff3b4274e180bd1a6b8cf0e7042e4e7aa9be10d79230394c1fd2b4620',
  'internationale-other.m4a':
    'c164a556eb960f7bce2a57691b0788e72d4fe523d80ef5a59c1515916d85cb45',
}

assert.equal(analysis.metadata.title, '国际歌')
assert.equal(analysis.metadata.artist, '唐朝')
assert.ok(Math.abs(analysis.metadata.duration - 266.73333) < 0.001)
assert.equal(analysis.bpm, 98.1)
assert.equal(analysis.key, 'G major')
assert.equal(analysis.notes.length, 1462)
assert.equal(analysis.pickup_beats, 1)
assert.equal(analysis.beats[0], 0.3317)
assert.equal(analysis.beats[1], 0.9433)
assert.equal(analysis.chords.length, 107)
assert.equal(analysis.tempo_map.mode, 'steady')
assert.equal(analysis.tempo_map.points.length, analysis.beats.length)
assert.equal(
  analysis.transcription_model.position_optimizer,
  'sequence-optimizer-v3',
)
assert.equal(
  analysis.transcription_model.timing_model,
  'percussion-downbeat-grid-v2',
)
assert.equal(
  analysis.transcription_model.rearticulation_filter,
  'acoustic-onset-gate-v1',
)
assert.equal(analysis.stem_tracks.length, 0)
assert.equal(analysis.guitar_parts.length, 1)
assert.ok(
  analysis.warnings.includes(
    '默认曲目：唐朝《国际歌》，由本机授权媒体生成。',
  ),
)
assert.ok(
  analysis.notes.every(
    (note) =>
      note.string >= 1 &&
      note.string <= 6 &&
      note.fret >= 0 &&
      note.fret <= 24,
  ),
)
const onsetGroups = []
for (const note of analysis.notes) {
  const current = onsetGroups.at(-1)
  if (!current || note.start - current[0].start > 0.045) {
    onsetGroups.push([note])
  } else {
    current.push(note)
  }
}
assert.ok(
  onsetGroups.every(
    (group) =>
      new Set(group.map((note) => note.string)).size === group.length,
  ),
)
assert.ok(
  Object.values(analysis.stems).every(
    (filename) => !String(filename).includes('/'),
  ),
)

for (const [filename, expectedHash] of Object.entries(expectedHashes)) {
  const media = read(filename)
  assert.equal(sha256(media), expectedHash, filename)
  if (filename.endsWith('.ogg')) {
    assert.equal(media.subarray(0, 4).toString('ascii'), 'OggS', filename)
  } else {
    assert.equal(media.subarray(4, 8).toString('ascii'), 'ftyp', filename)
  }
}

console.log(
  `Internationale demo verified: ${analysis.notes.length} notes, ` +
    `${analysis.chords.length} chord regions and ${analysis.guitar_parts.length} guitar part.`,
)
