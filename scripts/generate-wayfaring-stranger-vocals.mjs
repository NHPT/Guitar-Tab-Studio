import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repositoryRoot = resolve(scriptDirectory, '..')
const assetRoot = resolve(repositoryRoot, 'apps/api/assets')
const manifestPath = resolve(assetRoot, 'wayfaring-stranger-beginner.json')
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
const outputPath = resolve(assetRoot, manifest.audio.vocalsFilename)
const sampleRate = manifest.audio.sampleRate
const ticksPerMeasure =
  manifest.ticksPerBeat * manifest.timeSignature[0]
const beatDuration = 60 / manifest.bpm
const duration =
  manifest.measures.length * manifest.timeSignature[0] * beatDuration
const frameCount = Math.round(duration * sampleRate)
const temporaryRoot = resolve(assetRoot, '.wayfaring-vocal-build')

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function wavData(buffer) {
  if (
    buffer.toString('ascii', 0, 4) !== 'RIFF' ||
    buffer.toString('ascii', 8, 12) !== 'WAVE'
  ) {
    throw new Error('Guide vocal is not a WAV file')
  }
  let cursor = 12
  let channels
  let sourceRate
  let bitsPerSample
  let data
  while (cursor + 8 <= buffer.length) {
    const chunk = buffer.toString('ascii', cursor, cursor + 4)
    const length = buffer.readUInt32LE(cursor + 4)
    const start = cursor + 8
    if (chunk === 'fmt ') {
      if (buffer.readUInt16LE(start) !== 1) {
        throw new Error('Guide vocal must use PCM encoding')
      }
      channels = buffer.readUInt16LE(start + 2)
      sourceRate = buffer.readUInt32LE(start + 4)
      bitsPerSample = buffer.readUInt16LE(start + 14)
    }
    if (chunk === 'data') data = buffer.subarray(start, start + length)
    cursor = start + length + (length % 2)
  }
  if (
    channels !== 1 ||
    sourceRate !== sampleRate ||
    bitsPerSample !== 16 ||
    !data
  ) {
    throw new Error('Guide vocal must be 44.1 kHz mono 16-bit PCM')
  }
  return data
}

function midiForPitch(pitch) {
  const match = /^([A-G])(#?)(\d)$/u.exec(pitch)
  if (!match) throw new Error(`Unsupported vocal pitch ${pitch}`)
  const semitones = {
    C: 0,
    D: 2,
    E: 4,
    F: 5,
    G: 7,
    A: 9,
    B: 11,
  }
  return (Number(match[3]) + 1) * 12 + semitones[match[1]] + (match[2] ? 1 : 0)
}

function frequencyForPitch(pitch) {
  return 440 * 2 ** ((midiForPitch(pitch) - 69) / 12)
}

function pronunciation(syllable, melisma) {
  if (melisma) {
    const normalized = syllable.toLowerCase()
    if (/(?:o|oa|oe|woe|home)$/u.test(normalized)) return 'oh'
    if (/(?:i|y|ee|ing|me)$/u.test(normalized)) return 'ee'
    if (/(?:u|oo|through)$/u.test(normalized)) return 'oo'
    if (/(?:ay|way)$/u.test(normalized)) return 'ay'
    return 'ah'
  }
  const replacements = {
    beau: 'byoo',
    ger: 'jer',
    jour: 'jur',
    Jor: 'jor',
    ney: 'nee',
    tious: 'shus',
  }
  return replacements[syllable] ?? syllable
}

function vowelFor(syllable) {
  const normalized = syllable.toLowerCase()
  if (/(?:oo|through|you|u)/u.test(normalized)) return 'u'
  if (/(?:ee|ea|i|y|me|ney|ing)/u.test(normalized)) return 'i'
  if (/(?:o|oa|woe|home|jor)/u.test(normalized)) return 'o'
  if (/(?:e|ai|ay)/u.test(normalized)) return 'e'
  return 'a'
}

function playbackEvents() {
  const indexed = manifest.measures.flatMap((measure, measureIndex) =>
    measure.events.map((event) => ({
      event,
      absoluteTick: measureIndex * ticksPerMeasure + event.tick,
    })),
  )
  const melody = indexed.filter(({ event }) => event.part === 'melody')
  return melody.flatMap((indexedEvent, index) => {
    const { event } = indexedEvent
    if (event.tieFromPrevious || !event.vocalSyllable) return []
    let durationTicks = event.durationTicks
    let segment = indexedEvent
    let nextIndex = index + 1
    while (segment.event.tieToNext) {
      const next = melody[nextIndex]
      if (
        !next ||
        !next.event.tieFromPrevious ||
        next.event.sourcePitch !== event.sourcePitch ||
        next.absoluteTick !==
          segment.absoluteTick + segment.event.durationTicks
      ) {
        throw new Error(
          `Broken vocal tie after ${event.sourcePitch} at ${indexedEvent.absoluteTick}`,
        )
      }
      durationTicks += next.event.durationTicks
      segment = next
      nextIndex += 1
    }
    return [{ ...indexedEvent, durationTicks }]
  })
}

function validateExisting() {
  if (!existsSync(outputPath)) {
    throw new Error('Built-in guide vocal is missing; regenerate it')
  }
  const wav = readFileSync(outputPath)
  const data = wavData(wav)
  const samples = new Int16Array(
    data.buffer,
    data.byteOffset,
    Math.floor(data.byteLength / 2),
  )
  if (Math.abs(samples.length - frameCount) > 1) {
    throw new Error('Built-in guide vocal duration does not match the song')
  }
  let peak = 0
  let energy = 0
  for (const sample of samples) {
    peak = Math.max(peak, Math.abs(sample))
    energy += sample * sample
  }
  if (peak < 1000 || Math.sqrt(energy / samples.length) < 100) {
    throw new Error('Built-in guide vocal is silent')
  }
  return wav
}

if (process.argv.includes('--check')) {
  const wav = validateExisting()
  process.stdout.write(
    `${relative(repositoryRoot, outputPath)} ${duration.toFixed(3)}s ${sha256(wav)}\n`,
  )
  process.exit(0)
}

if (!existsSync('/usr/bin/say')) {
  throw new Error('macOS system voice is required to regenerate guide vocals')
}
const python = resolve(repositoryRoot, '.venv/bin/python')
if (!existsSync(python)) {
  throw new Error('Project Python environment is required to fit guide vocals')
}

rmSync(temporaryRoot, { recursive: true, force: true })
mkdirSync(temporaryRoot, { recursive: true })
try {
  const events = playbackEvents()
  const sourceKeys = new Map()
  const tasks = events.map(({ event, absoluteTick, durationTicks }) => {
    const spoken = pronunciation(event.vocalSyllable, event.melisma)
    const sourceKey = sha256(
      `${spoken}:${event.melisma ? 'vowel' : 'syllable'}`,
    )
    sourceKeys.set(sourceKey, spoken)
    return {
      sourceKey,
      targetPitch: frequencyForPitch(event.sourcePitch),
      vowel: vowelFor(event.vocalSyllable),
      targetFrames: Math.max(
        Math.round(sampleRate * 0.16),
        Math.round(
          (durationTicks / manifest.ticksPerBeat) *
            beatDuration *
            sampleRate *
            0.94,
        ),
      ),
      startFrame: Math.round(
        (absoluteTick / manifest.ticksPerBeat) *
          beatDuration *
          sampleRate,
      ),
    }
  })

  for (const [sourceKey, spoken] of sourceKeys) {
    const result = spawnSync(
      '/usr/bin/say',
      [
        '-v',
        'Samantha',
        '-r',
        '175',
        '--file-format=WAVE',
        `--data-format=LEI16@${sampleRate}`,
        '-o',
        resolve(temporaryRoot, `${sourceKey}.wav`),
        `[[pbas 45]]${spoken}`,
      ],
      {
        cwd: repositoryRoot,
        encoding: 'utf8',
        timeout: 15_000,
      },
    )
    if (result.status !== 0) {
      throw new Error(`System guide-vocal synthesis failed for "${spoken}"`)
    }
  }

  const tasksPath = resolve(temporaryRoot, 'tasks.json')
  writeFileSync(
    tasksPath,
    `${JSON.stringify({ sampleRate, frameCount, events: tasks })}\n`,
    'utf8',
  )
  const result = spawnSync(
    python,
    [
      resolve(repositoryRoot, 'scripts/render-guide-vocals.py'),
      tasksPath,
      temporaryRoot,
      outputPath,
    ],
    {
      cwd: repositoryRoot,
      encoding: 'utf8',
      timeout: 5 * 60 * 1000,
    },
  )
  if (result.status !== 0) {
    throw new Error(
      `Guide-vocal rendering failed: ${result.stderr.trim()}`,
    )
  }
  const wav = validateExisting()
  process.stdout.write(
    `${relative(repositoryRoot, outputPath)} ${duration.toFixed(3)}s ${sha256(wav)}\n`,
  )
} finally {
  rmSync(temporaryRoot, { recursive: true, force: true })
}
