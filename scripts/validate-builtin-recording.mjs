import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const assets = resolve(dirname(fileURLToPath(import.meta.url)), '../apps/api/assets')
const read = (name) => readFileSync(resolve(assets, name))
const manifest = JSON.parse(read('wayfaring-stranger-beginner.json'))
const abc = read('wayfaring-stranger.abc')
const guitar = read(manifest.audio.filename)
const vocals = read(manifest.audio.vocalsFilename)
const sha256 = (value) => createHash('sha256').update(value).digest('hex')

function validatePcmWav(buffer, filename) {
  assert.equal(buffer.toString('ascii', 0, 4), 'RIFF', filename)
  assert.equal(buffer.toString('ascii', 8, 12), 'WAVE', filename)
  assert.equal(buffer.readUInt16LE(20), 1, `${filename} must use PCM`)
  assert.equal(buffer.readUInt16LE(22), 1, `${filename} must be mono`)
  assert.equal(
    buffer.readUInt32LE(24),
    manifest.audio.sampleRate,
    `${filename} sample rate`,
  )
  assert.equal(buffer.readUInt16LE(34), 16, `${filename} bit depth`)
}

assert.equal(manifest.title, 'Wayfaring Stranger')
assert.equal(
  manifest.artist,
  'American Traditional · Guitar Tab Studio 初学编配',
)
assert.equal(manifest.provenance.compositionCredit, 'American traditional melody')
assert.equal(manifest.provenance.assetLicense, 'Apache-2.0')
assert.match(manifest.provenance.recordingCredit, /deterministic/u)
assert.equal(sha256(abc), manifest.source.sha256)
assert.equal(manifest.measures.length, 64)
assert.equal(manifest.bpm, 88)
assert.deepEqual(manifest.timeSignature, [4, 4])

validatePcmWav(guitar, manifest.audio.filename)
validatePcmWav(vocals, manifest.audio.vocalsFilename)
assert.equal(guitar.length, 15_394_954)
assert.equal(vocals.length, 15_394_954)
assert.equal(
  sha256(guitar),
  '37e75840946e3bcdece68c1df7dd8f164a53ccd2942e650411f32704a928483e',
)
assert.equal(
  sha256(vocals),
  '8bc008b2badbad083b46ab95977ab51466049ee6abf0a479fc0a48a0cec627be',
)
assert.notEqual(sha256(guitar), sha256(vocals))

const tuning = { 1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40 }
const ticksPerMeasure =
  manifest.ticksPerBeat * manifest.timeSignature[0]
for (const [measureIndex, measure] of manifest.measures.entries()) {
  assert.ok(measure.chord, `Measure ${measureIndex + 1} chord`)
  for (const event of measure.events) {
    assert.ok(event.tick >= 0 && event.tick < ticksPerMeasure)
    assert.ok(event.durationTicks > 0)
    assert.ok(event.string >= 1 && event.string <= 6)
    assert.ok(event.fret >= 0 && event.fret <= 24)
    assert.match(event.sourcePitch, /^[A-G]#?\d$/u)
    assert.ok(Number.isInteger(tuning[event.string] + event.fret))
  }
}

console.log(
  `Public-domain beginner demo verified: ${manifest.measures.length} measures, ` +
  `${manifest.measures.flatMap((measure) => measure.events).length} authored events.`,
)
