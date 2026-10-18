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
  read('take-me-home-country-roads-analysis.json'),
)
const expectedHashes = {
  'take-me-home-country-roads.mp3':
    '0fb7faa72934b35b7f5a3adb65b1ac7cbb49b128692da6fe42c4cc8b90624608',
  'take-me-home-country-roads-vocals.m4a':
    '05a349e17268dafc12e1b780542bb2e930fcdc29f9c53493c37475f37d880c75',
  'take-me-home-country-roads-guitar.m4a':
    '2bba021b4366e65c20424dde9bf89c1ff464578666f1f1cab42e75abb588d753',
  'take-me-home-country-roads-guitar-1.m4a':
    '00ef2142b4c2a574a04b13513dc3a93cc1068cdb3ab45e14a42b2e8d47af3fa2',
  'take-me-home-country-roads-guitar-2.m4a':
    '593eead54fc1c09af3f9aca1cdbfa1a042814a65896d11992f7e4e4bc1a4dde9',
  'take-me-home-country-roads-bass.m4a':
    '0cfd32caa6eb89d345a9aeb1d6a4b759b01fa731ea6f749c1c438cfc02cdacbc',
  'take-me-home-country-roads-drums.m4a':
    'b4f359a75d081e0cb3cdb58779478b2bf92d58fc76bec30591c36855123aa6b2',
  'take-me-home-country-roads-piano.m4a':
    '3e01367173e04e3b35acf12763ab4cf2371aca6417a77a5820c74a1e71411e3f',
  'take-me-home-country-roads-other.m4a':
    '42dadcb12432610658ae25e70103dd909411d711c9beb5ae52383dbe6c1b48fa',
}

assert.equal(analysis.metadata.title, 'Take Me Home, Country Roads')
assert.equal(analysis.metadata.artist, 'John Denver')
assert.ok(Math.abs(analysis.metadata.duration - 194.14667) < 0.001)
assert.equal(analysis.bpm, 161.5)
assert.equal(analysis.key, 'A major')
assert.equal(analysis.notes.length, 1789)
assert.equal(analysis.chords.length, 130)
assert.equal(analysis.stem_tracks.length, 2)
assert.ok(
  analysis.warnings.includes(
    '默认曲目来源：https://music.163.com/#/song?id=1478037',
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
assert.ok(
  Object.values(analysis.stems).every(
    (filename) => !String(filename).includes('/'),
  ),
)

for (const [filename, expectedHash] of Object.entries(expectedHashes)) {
  const media = read(filename)
  assert.equal(sha256(media), expectedHash, filename)
  if (filename.endsWith('.mp3')) {
    assert.equal(media.subarray(0, 3).toString('ascii'), 'ID3', filename)
  } else {
    assert.equal(media.subarray(4, 8).toString('ascii'), 'ftyp', filename)
  }
}

console.log(
  `Country Roads demo verified: ${analysis.notes.length} notes, ` +
    `${analysis.chords.length} chord regions and ${analysis.stem_tracks.length} guitar parts.`,
)
