import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repositoryRoot = resolve(scriptDirectory, '..')
const assetRoot = resolve(repositoryRoot, 'apps/api/assets')
const abcFilename = 'wayfaring-stranger.abc'
const manifestFilename = 'wayfaring-stranger-beginner.json'
const abcPath = resolve(assetRoot, abcFilename)
const manifestPath = resolve(assetRoot, manifestFilename)
const abc = readFileSync(abcPath, 'utf8')

const TICKS_PER_BEAT = 12
const TICKS_PER_EIGHTH = TICKS_PER_BEAT / 2
const TICKS_PER_MEASURE = TICKS_PER_BEAT * 4
const BPM = 88
const SAMPLE_RATE = 44_100

const melodyPositions = {
  D: { string: 4, fret: 0, pitch: 'D3' },
  E: { string: 4, fret: 2, pitch: 'E3' },
  G: { string: 3, fret: 0, pitch: 'G3' },
  A: { string: 3, fret: 2, pitch: 'A3' },
  B: { string: 2, fret: 0, pitch: 'B3' },
  d: { string: 2, fret: 3, pitch: 'D4' },
  e: { string: 1, fret: 0, pitch: 'E4' },
}

const alternatingBass = {
  Em: [
    { string: 6, fret: 0, sourcePitch: 'E2' },
    { string: 5, fret: 2, sourcePitch: 'B2' },
  ],
  Am: [
    { string: 5, fret: 0, sourcePitch: 'A2' },
    { string: 6, fret: 0, sourcePitch: 'E2' },
  ],
  C: [
    { string: 5, fret: 3, sourcePitch: 'C3' },
    { string: 6, fret: 3, sourcePitch: 'G2' },
  ],
  D: [
    { string: 4, fret: 0, sourcePitch: 'D3' },
    { string: 5, fret: 0, sourcePitch: 'A2' },
  ],
  G: [
    { string: 6, fret: 3, sourcePitch: 'G2' },
    { string: 4, fret: 0, sourcePitch: 'D3' },
  ],
  B7: [
    { string: 5, fret: 2, sourcePitch: 'B2' },
    { string: 6, fret: 2, sourcePitch: 'F#2' },
  ],
}

const lyricCycles = [
  {
    section: '第 1 节',
    phrases: [
      {
        text: 'I am a poor wayfaring stranger',
        syllables: ['I', 'am', 'a', 'poor', 'way', 'far', 'ing', 'stran', 'ger'],
      },
      {
        text: 'While journeying through this world of woe',
        syllables: ['While', 'jour', 'ney', 'ing', 'through', 'this', 'world', 'of', 'woe'],
      },
      {
        text: "Yet there's no sickness, toil, or danger",
        syllables: ['Yet', "there's", 'no', 'sick', 'ness', 'toil', '~', 'or', 'dan', '~', 'ger'],
      },
      {
        text: 'In that bright world to which I go',
        syllables: ['In', 'that', 'bright', 'world', 'to', 'which', 'I', '~', 'go'],
      },
      {
        text: "I'm going there to see my father",
        syllables: ["I'm", 'go', 'ing', 'there', '~', 'to', 'see', 'my', 'fa', '~', 'ther'],
      },
      {
        text: "I'm going there no more to roam",
        syllables: ["I'm", 'go', 'ing', 'there', '~', 'no', 'more', '~', 'to', '~', 'roam'],
      },
      {
        text: "I'm just a-going over Jordan",
        syllables: ["I'm", 'just', 'a', 'go', 'ing', 'o', 'ver', 'Jor', 'dan'],
      },
      {
        text: "I'm just a-going over home",
        syllables: ["I'm", 'just', 'a', 'go', 'ing', 'o', 'ver', '~', 'home'],
      },
    ],
  },
  {
    section: '第 2 节',
    phrases: [
      {
        text: 'I know dark clouds will gather round me',
        syllables: ['I', 'know', 'dark', 'clouds', 'will', 'gath', 'er', 'round', 'me'],
      },
      {
        text: 'I know my way is rough and steep',
        syllables: ['I', 'know', 'my', 'way', 'is', 'rough', 'and', '~', 'steep'],
      },
      {
        text: 'Yet brighter fields lie just before me',
        syllables: ['Yet', 'bright', 'er', 'fields', '~', 'lie', 'just', 'be', 'fore', '~', 'me'],
      },
      {
        text: "Where God's redeemed their vigils keep",
        syllables: ['Where', "God's", 're', 'deemed', 'their', 'vig', 'ils', '~', 'keep'],
      },
      {
        text: "I'm going there to see my mother",
        syllables: ["I'm", 'go', 'ing', 'there', '~', 'to', 'see', 'my', 'moth', '~', 'er'],
      },
      {
        text: "She said she'd meet me when I come",
        syllables: ['She', 'said', "she'd", 'meet', '~', 'me', 'when', '~', 'I', '~', 'come'],
      },
      {
        text: "I'm just a-going over Jordan",
        syllables: ["I'm", 'just', 'a', 'go', 'ing', 'o', 'ver', 'Jor', 'dan'],
      },
      {
        text: "I'm just a-going over home",
        syllables: ["I'm", 'just', 'a', 'go', 'ing', 'o', 'ver', '~', 'home'],
      },
    ],
  },
]

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function abcBody(document) {
  const lines = document.split(/\r?\n/u)
  const keyIndex = lines.findIndex((line) => line.startsWith('K:'))
  if (keyIndex < 0) throw new Error('ABC source is missing a key header')
  return lines
    .slice(keyIndex + 1)
    .filter((line) => !line.trimStart().startsWith('%'))
    .join('')
    .replace(/\s+/gu, '')
}

function parseAbc(document) {
  const body = abcBody(document)
  const measures = []
  let currentChord
  let measureChord
  let measureEvents = []
  let measureTick = 0
  let tripletRemaining = 0
  let pendingTiePitch
  let cursor = 0

  const finishMeasure = () => {
    if (measureTick === 0 && measureEvents.length === 0) return
    if (measureTick !== TICKS_PER_MEASURE) {
      throw new Error(
        `ABC measure ${measures.length + 1} has ${measureTick} ticks; expected ${TICKS_PER_MEASURE}`,
      )
    }
    if (!measureChord) {
      throw new Error(`ABC measure ${measures.length + 1} has no chord`)
    }
    measures.push({
      chord: measureChord,
      events: measureEvents,
    })
    measureEvents = []
    measureTick = 0
    measureChord = currentChord
  }

  while (cursor < body.length) {
    const character = body[cursor]
    if (character === '"') {
      const end = body.indexOf('"', cursor + 1)
      if (end < 0) throw new Error('ABC chord annotation is unterminated')
      currentChord = body.slice(cursor + 1, end)
      if (measureTick === 0) measureChord = currentChord
      cursor = end + 1
      continue
    }
    if (character === '(') {
      if (body.slice(cursor, cursor + 2) !== '(3') {
        throw new Error(`Unsupported ABC tuplet at offset ${cursor}`)
      }
      tripletRemaining = 3
      cursor += 2
      continue
    }
    if (character === ':') {
      cursor += 1
      continue
    }
    if (character === '|') {
      finishMeasure()
      cursor += 1
      continue
    }
    if (!/[A-Ga-gz]/u.test(character)) {
      throw new Error(`Unsupported ABC token "${character}" at offset ${cursor}`)
    }

    cursor += 1
    const lengthStart = cursor
    while (cursor < body.length && /\d/u.test(body[cursor])) cursor += 1
    const length = Number(body.slice(lengthStart, cursor) || '1')
    const tied = body[cursor] === '-'
    if (tied) cursor += 1
    const tripletScale = tripletRemaining > 0 ? 2 / 3 : 1
    const durationTicks = length * TICKS_PER_EIGHTH * tripletScale
    if (!Number.isInteger(durationTicks)) {
      throw new Error(`ABC token ${character}${length} has fractional ticks`)
    }
    if (tripletRemaining > 0) tripletRemaining -= 1

    if (character === 'z') {
      if (pendingTiePitch) throw new Error('ABC tie cannot continue into a rest')
      measureTick += durationTicks
      continue
    }

    const position = melodyPositions[character]
    if (!position) {
      throw new Error(`No beginner guitar position for ABC pitch ${character}`)
    }
    const tieFromPrevious = Boolean(pendingTiePitch)
    if (pendingTiePitch && pendingTiePitch !== position.pitch) {
        throw new Error(
          `ABC tie changed pitch from ${pendingTiePitch} to ${position.pitch}`,
        )
    }
    measureEvents.push({
      tick: measureTick,
      durationTicks,
      string: position.string,
      fret: position.fret,
      sourcePitch: position.pitch,
      part: 'melody',
      ...(tieFromPrevious ? { tieFromPrevious: true } : {}),
      ...(tied ? { tieToNext: true } : {}),
    })
    pendingTiePitch = tied ? position.pitch : undefined
    measureTick += durationTicks
  }

  if (measureTick !== 0 || measureEvents.length > 0) finishMeasure()
  if (pendingTiePitch) throw new Error('ABC source ends with an unresolved tie')
  if (tripletRemaining !== 0) throw new Error('ABC source ends inside a triplet')
  if (measures.length !== 32) {
    throw new Error(`Expected 32 ABC measures, received ${measures.length}`)
  }
  return measures
}

function addBass(measures) {
  return measures.map((measure) => {
    const bass = alternatingBass[measure.chord]
    if (!bass) throw new Error(`No alternating bass for ${measure.chord}`)
    const events = [
      ...measure.events,
      ...bass.map((note, index) => ({
        tick: index * TICKS_PER_BEAT * 2,
        durationTicks: 20,
        ...note,
        part: 'bass',
      })),
    ].sort(
      (left, right) =>
        left.tick - right.tick ||
        (left.part === right.part ? 0 : left.part === 'bass' ? -1 : 1),
    )
    return { ...measure, events }
  })
}

function createSongMeasures(tuneMeasures) {
  return lyricCycles.flatMap((cycle, cycleIndex) => {
    const measures = tuneMeasures.map((measure) => ({
      chord: measure.chord,
      events: measure.events.map((event) => ({ ...event })),
    }))
    cycle.phrases.forEach((phrase, phraseIndex) => {
      const phraseMeasures = measures.slice(
        phraseIndex * 4,
        phraseIndex * 4 + 4,
      )
      const attacks = phraseMeasures.flatMap((measure) =>
        measure.events.filter(
          (event) => event.part === 'melody' && !event.tieFromPrevious,
        ),
      )
      if (attacks.length !== phrase.syllables.length) {
        throw new Error(
          `Lyric phrase ${cycleIndex + 1}.${phraseIndex + 1} has ${
            phrase.syllables.length
          } syllables for ${attacks.length} melody attacks`,
        )
      }
      let previousSyllable = 'ah'
      phrase.syllables.forEach((syllable, index) => {
        const event = attacks[index]
        if (syllable === '~') {
          event.vocalSyllable = previousSyllable
          event.melisma = true
          return
        }
        event.lyricSyllable = syllable
        event.vocalSyllable = syllable
        previousSyllable = syllable
      })
      phraseMeasures[0].lyric = phrase.text
    })
    measures[0].section = cycle.section
    measures[16].section = '副歌'
    return measures
  })
}

const tuneMeasures = addBass(parseAbc(abc))
const melodyFingerprint = sha256(
  tuneMeasures
    .flatMap((measure, measureIndex) =>
      measure.events
        .filter((event) => event.part === 'melody')
        .map(
          (event) =>
            `${measureIndex + 1}:${event.tick}:${event.durationTicks}:${event.sourcePitch}:${event.tieFromPrevious ? 1 : 0}:${event.tieToNext ? 1 : 0}`,
        ),
    )
    .join('|'),
)
const measures = createSongMeasures(tuneMeasures)
const songFingerprint = sha256(
  measures
    .flatMap((measure, measureIndex) =>
      measure.events.map(
        (event) =>
          `${measureIndex + 1}:${event.tick}:${event.durationTicks}:${event.sourcePitch}:${event.lyricSyllable ?? ''}:${event.melisma ? 1 : 0}`,
      ),
    )
    .join('|'),
)

const manifest = {
  schemaVersion: 3,
  id: 'wayfaring-stranger-complete-v1',
  title: 'Wayfaring Stranger',
  artist: 'American Traditional · Guitar Tab Studio 初学编配',
  bpm: BPM,
  key: 'E minor',
  timeSignature: [4, 4],
  ticksPerBeat: TICKS_PER_BEAT,
  source: {
    format: 'abc',
    filename: abcFilename,
    sha256: sha256(abc),
    melodyFingerprint,
    songFingerprint,
  },
  audio: {
    filename: 'wayfaring-stranger-beginner.wav',
    vocalsFilename: 'wayfaring-stranger-guide-vocals.wav',
    sampleRate: SAMPLE_RATE,
    channels: 1,
    bitsPerSample: 16,
  },
  provenance: {
    compositionCredit: 'American traditional melody',
    arrangementCredit:
      'Guitar Tab Studio melody with documented alternating-bass accompaniment',
    recordingCredit:
      'Guitar Tab Studio deterministic guitar recording and local synthetic guide vocal',
    assetLicense: 'Apache-2.0',
    note:
      'Two public-domain verses and repeated refrains are synchronized to the melody; every audible guitar event is included in the TAB. The vocal is a synthetic guide, not a human performance.',
    references: [
      'https://hymnary.org/tune/wayfaring_stranger',
      'https://hymnary.org/text/i_am_a_poor_wayfaring_stranger',
      'https://natunelist.net/the-wayfaring-stranger/',
    ],
  },
  measures,
}

const beatDuration = 60 / manifest.bpm
const measureDuration = beatDuration * manifest.timeSignature[0]
const duration = measureDuration * manifest.measures.length
const frameCount = Math.round(duration * SAMPLE_RATE)
const mix = new Float64Array(frameCount)
const openStringMidi = [64, 59, 55, 50, 45, 40]

function seedValue(label) {
  let value = 2166136261
  for (const character of label) {
    value ^= character.charCodeAt(0)
    value = Math.imul(value, 16777619)
  }
  return value >>> 0
}

function randomSource(seed) {
  let state = seed || 1
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 0xffffffff
  }
}

function frequencyFor(event) {
  const midi = openStringMidi[event.string - 1] + event.fret
  return 440 * 2 ** ((midi - 69) / 12)
}

function addPluck(event, start, noteDuration, gain, seed) {
  const frequency = frequencyFor(event)
  const delayLength = Math.max(2, Math.round(SAMPLE_RATE / frequency))
  const delay = new Float64Array(delayLength)
  const random = randomSource(seed)
  for (let index = 0; index < delay.length; index += 1) {
    delay[index] = random() * 2 - 1
  }

  const startFrame = Math.max(0, Math.round(start * SAMPLE_RATE))
  const renderFrames = Math.min(
    frameCount - startFrame,
    Math.round((noteDuration + 1.1) * SAMPLE_RATE),
  )
  const releaseStart = Math.max(
    1,
    Math.round(noteDuration * SAMPLE_RATE),
  )
  let previous = 0
  for (let frame = 0; frame < renderFrames; frame += 1) {
    const index = frame % delayLength
    const next = (index + 1) % delayLength
    const value = delay[index]
    delay[index] = (delay[index] + delay[next]) * 0.5 * 0.9965
    const softened = value * 0.78 + previous * 0.22
    previous = softened
    const attack = Math.min(1, frame / (SAMPLE_RATE * 0.004))
    const release =
      frame < releaseStart
        ? 1
        : Math.max(
            0,
            1 - (frame - releaseStart) / (renderFrames - releaseStart),
          )
    const transient =
      frame < SAMPLE_RATE * 0.012
        ? (random() * 2 - 1) *
          (1 - frame / (SAMPLE_RATE * 0.012)) *
          0.1
        : 0
    mix[startFrame + frame] +=
      (softened + transient) * attack * release * gain
  }
}

const indexedEvents = manifest.measures.flatMap((measure, measureIndex) =>
  measure.events.map((event, eventIndex) => ({
    event,
    eventIndex,
    measureIndex,
    absoluteTick:
      measureIndex * TICKS_PER_MEASURE + event.tick,
  })),
)
const playbackEvents = indexedEvents
  .filter(({ event }) => event.part === 'bass')
  .map((indexed) => ({
    ...indexed,
    durationTicks: indexed.event.durationTicks,
  }))
const melodySegments = indexedEvents.filter(
  ({ event }) => event.part === 'melody',
)
melodySegments.forEach((indexed, index) => {
  if (indexed.event.tieFromPrevious) return
  let durationTicks = indexed.event.durationTicks
  let segment = indexed
  let nextIndex = index + 1
  while (segment.event.tieToNext) {
    const next = melodySegments[nextIndex]
    if (
      !next ||
      !next.event.tieFromPrevious ||
      next.event.sourcePitch !== indexed.event.sourcePitch ||
      next.absoluteTick !== segment.absoluteTick + segment.event.durationTicks
    ) {
      throw new Error(
        `Broken playback tie after ${indexed.event.sourcePitch} at ${indexed.absoluteTick}`,
      )
    }
    durationTicks += next.event.durationTicks
    segment = next
    nextIndex += 1
  }
  playbackEvents.push({ ...indexed, durationTicks })
})

playbackEvents
  .sort((left, right) => left.absoluteTick - right.absoluteTick)
  .forEach(({ event, eventIndex, measureIndex, absoluteTick, durationTicks }) => {
    const start = (absoluteTick / TICKS_PER_BEAT) * beatDuration
    const noteDuration = (durationTicks / TICKS_PER_BEAT) * beatDuration
    addPluck(
      event,
      start,
      noteDuration,
      event.part === 'melody' ? 0.3 : 0.19,
      seedValue(`${manifest.id}:${measureIndex}:${eventIndex}`),
    )
  })

const room = Float64Array.from(mix)
for (const [delaySeconds, gain] of [
  [0.037, 0.11],
  [0.061, 0.075],
  [0.097, 0.045],
]) {
  const delayFrames = Math.round(delaySeconds * SAMPLE_RATE)
  for (let frame = delayFrames; frame < frameCount; frame += 1) {
    room[frame] += mix[frame - delayFrames] * gain
  }
}

let previousInput = 0
let previousOutput = 0
let maximum = 0
for (let frame = 0; frame < frameCount; frame += 1) {
  const highPassed = room[frame] - previousInput + 0.994 * previousOutput
  previousInput = room[frame]
  previousOutput = highPassed
  const fadeIn = Math.min(1, frame / (SAMPLE_RATE * 0.008))
  const fadeOut = Math.min(
    1,
    (frameCount - frame - 1) / (SAMPLE_RATE * 0.65),
  )
  room[frame] = Math.tanh(highPassed * 1.22) * fadeIn * fadeOut
  maximum = Math.max(maximum, Math.abs(room[frame]))
}

const normalization = maximum > 0 ? 0.88 / maximum : 1
const dataBytes = frameCount * 2
const wav = Buffer.alloc(44 + dataBytes)
wav.write('RIFF', 0)
wav.writeUInt32LE(36 + dataBytes, 4)
wav.write('WAVE', 8)
wav.write('fmt ', 12)
wav.writeUInt32LE(16, 16)
wav.writeUInt16LE(1, 20)
wav.writeUInt16LE(1, 22)
wav.writeUInt32LE(SAMPLE_RATE, 24)
wav.writeUInt32LE(SAMPLE_RATE * 2, 28)
wav.writeUInt16LE(2, 32)
wav.writeUInt16LE(16, 34)
wav.write('data', 36)
wav.writeUInt32LE(dataBytes, 40)
for (let frame = 0; frame < frameCount; frame += 1) {
  const sample = Math.max(-1, Math.min(1, room[frame] * normalization))
  wav.writeInt16LE(Math.round(sample * 32767), 44 + frame * 2)
}

const manifestBytes = `${JSON.stringify(manifest, null, 2)}\n`
const outputPath = resolve(assetRoot, manifest.audio.filename)
if (process.argv.includes('--check')) {
  if (readFileSync(manifestPath, 'utf8') !== manifestBytes) {
    throw new Error('Built-in demo manifest is stale; regenerate it')
  }
  if (!readFileSync(outputPath).equals(wav)) {
    throw new Error('Built-in demo WAV is stale; regenerate it')
  }
} else {
  writeFileSync(manifestPath, manifestBytes)
  writeFileSync(outputPath, wav)
}

process.stdout.write(
  [
    `${relative(repositoryRoot, manifestPath)} ${measures.length} measures`,
    `${relative(repositoryRoot, outputPath)} ${duration.toFixed(3)}s ${sha256(wav)}`,
    `melody ${melodyFingerprint}`,
  ].join('\n') + '\n',
)
