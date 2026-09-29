import assert from 'node:assert/strict'
import test from 'node:test'
import {
  createStudioProject,
  detectPlatform,
  normalizeSharedUrl,
  sourceFromLink,
  tabFromWorker,
} from './pipeline.js'
import type { WorkerAnalysis } from './pipeline.js'

test('detects supported Chinese media platforms', () => {
  assert.equal(detectPlatform('https://www.bilibili.com/video/BV1xx')?.id, 'bilibili')
  assert.equal(detectPlatform('https://v.douyin.com/example')?.id, 'douyin')
  assert.equal(detectPlatform('https://music.163.com/song?id=123')?.id, 'netease')
  assert.equal(detectPlatform('https://y.qq.com/n/ryqq/songDetail/example'), null)
  assert.equal(detectPlatform('https://qishui.douyin.com/s/example'), null)
})

test('extracts supported URLs from app share text', () => {
  const shared =
    '分享歌曲：示例标题 https://music.163.com/song?id=12345 （来自网易云音乐）'
  assert.equal(normalizeSharedUrl(shared), 'https://music.163.com/song?id=12345')
  assert.equal(sourceFromLink(shared).kind, 'netease')
  assert.equal(detectPlatform('https://bili2233.cn/example')?.id, 'bilibili')
  assert.equal(detectPlatform('https://163.fm/example')?.id, 'netease')
})

test('rejects unsupported or malformed URLs', () => {
  assert.equal(detectPlatform('not-a-url'), null)
  assert.equal(detectPlatform('https://example.com/song'), null)
  assert.throws(() => sourceFromLink('https://example.com/song'))
})

test('demo project has playable tab timing and independent stems', () => {
  const project = createStudioProject({
    kind: 'upload',
    label: '测试',
    filename: 'sample.wav',
  })

  assert.equal(project.tab.measures.length, 8)
  assert.equal(project.tracks.length, 6)
  assert.equal(project.visualization?.length, 720)
  assert.equal(project.techniqueModel?.version, 'demo-1')
  assert.ok(project.duration > 0)
  assert.ok(
    project.tab.measures.every((measure) =>
      measure.beats.every((beat) =>
        beat.notes.every(
          (note) =>
            note.fret >= 0 &&
            note.string >= 1 &&
            note.string <= 6 &&
            (note.positionConfidence ?? 0) > 0 &&
            (note.techniqueConfidence ?? 0) > 0 &&
            (note.techniqueCandidates?.length ?? 0) > 0,
        ),
      ),
    ),
  )
  const beatStyles = new Set<string | undefined>(
    project.tab.measures.flatMap((measure) => measure.beats.map((beat) => beat.style)),
  )
  assert.ok(
    ['pick', 'strum', 'arpeggio', 'rasgueado', 'tremolo'].every((style) =>
      beatStyles.has(style),
    ),
  )

  const techniques = new Set<string>(
    project.tab.measures.flatMap((measure) =>
      measure.beats.flatMap((beat) => beat.notes.map((note) => note.technique)),
    ),
  )
  assert.ok(
    ['hammer-on', 'pull-off', 'slide', 'harmonic', 'palm-mute', 'dead-note', 'slap', 'tremolo'].every(
      (technique) => techniques.has(technique),
    ),
  )
})

test('maps worker technique candidates and only valid note relations', () => {
  const analysis: WorkerAnalysis = {
    version: 3,
    bpm: 120,
    stems: {},
    notes: [
      {
        start: 0.1,
        end: 0.3,
        pitch: 57,
        velocity: 80,
        string: 3,
        fret: 2,
        technique: 'pick',
        confidence: 0.88,
        position_confidence: 0.73,
        position_source: 'playable-optimizer-v2',
        technique_confidence: 0.64,
        technique_source: 'acoustic-heuristic-v2',
        technique_evidence: ['audible-onset'],
        technique_candidates: [
          { technique: 'pick', confidence: 0.64, evidence: ['audible-onset'] },
        ],
        related_note_index: null,
      },
      {
        start: 0.32,
        end: 0.58,
        pitch: 59,
        velocity: 68,
        string: 3,
        fret: 4,
        technique: 'hammer-on',
        confidence: 0.84,
        position_confidence: 0.79,
        position_source: 'playable-optimizer-v2',
        technique_confidence: 0.91,
        technique_source: 'transition-heuristic-v2',
        technique_evidence: ['same-string', 'ascending-fret'],
        technique_candidates: [
          {
            technique: 'hammer-on',
            confidence: 0.91,
            evidence: ['same-string', 'ascending-fret'],
          },
        ],
        related_note_index: 0,
      },
    ],
    chords: [],
    warnings: [],
  }

  const notes = tabFromWorker(analysis).measures.flatMap((measure) =>
    measure.beats.flatMap((beat) => beat.notes),
  )
  assert.equal(notes[0].relatedNoteId, undefined)
  assert.equal(notes[1].relatedNoteId, 'worker-1')
  assert.equal(notes[1].positionConfidence, 0.79)
  assert.equal(notes[1].positionSource, 'playable-optimizer-v2')
  assert.equal(notes[1].techniqueConfidence, 0.91)
  assert.equal(notes[1].techniqueCandidates?.[0].technique, 'hammer-on')
})

test('uses the detected beat grid instead of treating recording lead-in as a rest', () => {
  const analysis: WorkerAnalysis = {
    version: 6,
    bpm: 120,
    beats: [0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6],
    onsets: [0.6, 2.1],
    stems: {},
    notes: [
      {
        start: 0.61,
        end: 0.9,
        pitch: 45,
        velocity: 80,
        string: 5,
        fret: 0,
        technique: 'pick',
        confidence: 0.9,
      },
      {
        start: 2.09,
        end: 2.4,
        pitch: 52,
        velocity: 78,
        string: 4,
        fret: 2,
        technique: 'pick',
        confidence: 0.88,
      },
    ],
    chords: [],
    warnings: [],
  }

  const tab = tabFromWorker(analysis)

  assert.equal(tab.measures[0].start, 0.6)
  assert.equal(tab.measures[0].duration, 2)
  assert.deepEqual(
    tab.measures[0].beats.map((beat) => beat.at),
    [0.6, 1.1, 1.6, 2.1],
  )
  assert.equal(tab.measures[0].beats[0].notes.length, 1)
  assert.equal(tab.measures[0].beats[3].notes.length, 1)
  assert.equal(tab.measures[0].beats[0].notes[0].notationAt, 0.6)
  assert.equal(tab.measures[0].beats[3].notes[0].notationAt, 2.1)
})

test('keeps near-simultaneous chord plucks visible instead of marking a strum', () => {
  const analysis: WorkerAnalysis = {
    version: 4,
    bpm: 120,
    stems: {},
    notes: [
      {
        start: 0.1,
        end: 0.8,
        pitch: 40,
        velocity: 80,
        string: 6,
        fret: 0,
        technique: 'pick',
        confidence: 0.9,
      },
      {
        start: 0.11,
        end: 0.8,
        pitch: 47,
        velocity: 78,
        string: 5,
        fret: 2,
        technique: 'pick',
        confidence: 0.88,
      },
      {
        start: 0.12,
        end: 0.8,
        pitch: 52,
        velocity: 76,
        string: 4,
        fret: 2,
        technique: 'pick',
        confidence: 0.86,
      },
    ],
    chords: [],
    warnings: [],
  }

  const firstBeat = tabFromWorker(analysis).measures[0].beats[0]
  assert.equal(firstBeat.style, 'pick')
  assert.equal(firstBeat.notes.length, 3)
})

test('maps physical string order to the correct strum direction', () => {
  const analysisFor = (
    strings: Array<1 | 2 | 3 | 4 | 5 | 6>,
  ): WorkerAnalysis => ({
    version: 8,
    bpm: 120,
    stems: {},
    notes: strings.map((string, index) => ({
      start: 0.08 + index * 0.045,
      end: 0.5,
      pitch: 40 + index * 5,
      velocity: 80,
      string,
      fret: 2,
      technique: 'pick',
      confidence: 0.9,
    })),
    chords: [],
    warnings: [],
  })

  const down = tabFromWorker(analysisFor([6, 5, 4])).measures[0].beats[0]
  const up = tabFromWorker(analysisFor([4, 5, 6])).measures[0].beats[0]

  assert.equal(down.style, 'strum')
  assert.equal(down.direction, 'down')
  assert.equal(up.style, 'strum')
  assert.equal(up.direction, 'up')
})

test('does not infer tremolo from three ordinary repeated picks', () => {
  const analysis: WorkerAnalysis = {
    version: 8,
    bpm: 120,
    stems: {},
    notes: [0.05, 0.22, 0.39].map((start) => ({
      start,
      end: start + 0.1,
      pitch: 61,
      velocity: 76,
      string: 2,
      fret: 2,
      technique: 'pick',
      confidence: 0.86,
    })),
    chords: [],
    warnings: [],
  }

  assert.equal(tabFromWorker(analysis).measures[0].beats[0].style, 'pick')
})

test('does not label a non-directional legato phrase as an arpeggio', () => {
  const starts = [0.05, 0.13, 0.19, 0.22]
  const strings = [2, 4, 2, 5] as const
  const analysis: WorkerAnalysis = {
    version: 5,
    bpm: 120,
    stems: {},
    notes: starts.map((start, index) => ({
      start,
      end: start + 0.25,
      pitch: 60 - index,
      velocity: 76,
      string: strings[index],
      fret: index + 1,
      technique: index === 2 ? 'hammer-on' : 'pick',
      confidence: 0.86,
      related_note_index: index === 2 ? 0 : null,
    })),
    chords: [],
    warnings: [],
  }

  const firstBeat = tabFromWorker(analysis).measures[0].beats[0]
  assert.equal(firstBeat.style, 'pick')
  assert.equal(firstBeat.direction, undefined)
})

test('does not advertise unsupported protected music platforms', () => {
  assert.throws(
    () => sourceFromLink('https://y.qq.com/n/ryqq/songDetail/example'),
    /当前仅支持哔哩哔哩、抖音和网易云音乐/,
  )
})
