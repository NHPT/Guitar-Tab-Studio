import assert from 'node:assert/strict'
import { chmodSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import {
  createStudioProject,
  describeMediaAcquisitionFailure,
  describeWorkerFailure,
  detectPlatform,
  executeWorker,
  normalizeSharedUrl,
  parseWorkerProgress,
  resolveInferenceExecution,
  sourceFromLink,
  tabFromWorker,
  transcriptionWorkerArguments,
} from './pipeline.js'
import type { WorkerAnalysis } from './pipeline.js'

test('detects supported public media platforms', () => {
  assert.equal(detectPlatform('https://www.bilibili.com/video/BV1xx')?.id, 'bilibili')
  assert.equal(detectPlatform('https://v.douyin.com/example')?.id, 'douyin')
  assert.equal(detectPlatform('https://music.163.com/song?id=123')?.id, 'netease')
  assert.equal(
    detectPlatform('https://y.qq.com/n/ryqq/songDetail/004Ti8rT003TaZ')?.id,
    'qqmusic',
  )
  assert.equal(
    detectPlatform('https://y.qq.com/n/ryqq_v2/songDetail/004Ti8rT003TaZ')?.id,
    'qqmusic',
  )
  assert.equal(detectPlatform('https://youtu.be/example')?.id, 'youtube')
  assert.equal(detectPlatform('https://music.youtube.com/watch?v=example')?.id, 'youtube')
  assert.equal(detectPlatform('https://vimeo.com/123456')?.id, 'vimeo')
  assert.equal(detectPlatform('https://soundcloud.com/artist/track')?.id, 'soundcloud')
  assert.equal(detectPlatform('https://artist.bandcamp.com/track/example')?.id, 'bandcamp')
  assert.equal(detectPlatform('https://www.jamendo.com/track/123/example')?.id, 'jamendo')
  assert.equal(detectPlatform('https://audius.co/artist/track')?.id, 'audius')
  assert.equal(detectPlatform('https://archive.org/details/example')?.id, 'internet-archive')
  assert.equal(
    detectPlatform('https://commons.wikimedia.org/wiki/File:Example.ogg')?.id,
    'wikimedia',
  )
  assert.equal(detectPlatform('https://qishui.douyin.com/s/example'), null)
})

test('accepts only structured worker progress and caps incomplete jobs below 100', () => {
  assert.deepEqual(
    parseWorkerProgress(
      '{"type":"gts-progress","progress":63,"status":"transcribing","stageLabel":"拍点检测完成"}',
    ),
    {
      progress: 63,
      status: 'transcribing',
      stageLabel: '拍点检测完成',
    },
  )
  assert.equal(parseWorkerProgress('Predicting MIDI...'), null)
  assert.equal(
    parseWorkerProgress(
      '{"type":"gts-progress","progress":100,"status":"completed","stageLabel":"完成"}',
    ),
    null,
  )
})

test('preserves UTF-8 progress text split across worker output chunks', {
  skip: process.platform === 'win32' && 'requires an executable script fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-worker-${randomUUID()}`)
  const worker = resolve(root, 'worker.mjs')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/usr/bin/env node
const line = Buffer.from('{"type":"gts-progress","progress":65,"status":"transcribing","stageLabel":"正在分析中文曲名"}\\n')
const marker = line.indexOf(Buffer.from('正在'))
process.stdout.write(line.subarray(0, marker + 1))
setTimeout(() => process.stdout.write(line.subarray(marker + 1)), 20)
`,
  )
  chmodSync(worker, 0o755)
  const progress: Array<{ stageLabel: string }> = []
  try {
    await executeWorker(
      worker,
      process.execPath,
      [],
      resolve(root, 'output'),
      undefined,
      undefined,
      5_000,
      (event) => progress.push(event),
    )
    assert.equal(progress[0]?.stageLabel, '正在分析中文曲名')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('preserves UTF-8 worker errors split across stderr chunks', {
  skip: process.platform === 'win32' && 'requires an executable script fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-worker-${randomUUID()}`)
  const worker = resolve(root, 'worker.mjs')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/usr/bin/env node
const line = Buffer.from('RuntimeError: 模型加载失败\\n')
const marker = line.indexOf(Buffer.from('模型'))
process.stderr.write(line.subarray(0, marker + 1))
setTimeout(() => {
  process.stderr.write(line.subarray(marker + 1))
  process.exitCode = 1
}, 20)
`,
  )
  chmodSync(worker, 0o755)
  try {
    await assert.rejects(
      executeWorker(
        worker,
        process.execPath,
        [],
        resolve(root, 'output'),
        undefined,
        undefined,
        5_000,
      ),
      /模型加载失败/u,
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('normalizes the selected transcription provider into worker arguments', () => {
  assert.deepEqual(transcriptionWorkerArguments({}), [
    '--transcription-provider',
    'basic-pitch',
  ])
  assert.deepEqual(
    transcriptionWorkerArguments({
      GTS_TRANSCRIPTION_PROVIDER: 'muscriptor',
      GTS_TRANSCRIPTION_MODEL: 'large',
      GTS_TRANSCRIPTION_VARIANT: 'large',
    }),
    [
      '--transcription-provider',
      'muscriptor',
      '--transcription-model',
      'large',
      '--transcription-variant',
      'large',
    ],
  )
  assert.deepEqual(
    transcriptionWorkerArguments({
      GTS_TRANSCRIPTION_PROVIDER: 'guitar-specialist',
    }),
    ['--transcription-provider', 'guitar-specialist'],
  )
  assert.deepEqual(
    transcriptionWorkerArguments({
      GTS_TRANSCRIPTION_PROVIDER: 'multitrack-research',
      GTS_TRANSCRIPTION_MODEL: '/models/yourmt3.ckpt',
    }),
    [
      '--transcription-provider',
      'multitrack-research',
      '--transcription-model',
      '/models/yourmt3.ckpt',
    ],
  )
})

test('turns worker tracebacks into copyable user-facing errors', () => {
  assert.equal(
    describeWorkerFailure(
      [
        '  File "/private/runtime/demucs/audio.py", line 260, in encode_wav',
        '    file.write(content)',
        'OSError: [Errno 28] No space left on device',
        'RuntimeError: Command failed (1): /private/runtime/python -m demucs.separate',
      ].join('\n'),
      1,
    ),
    '可用磁盘空间不足，无法完成音轨分离。请释放更多磁盘空间后重试。',
  )
  assert.equal(
    describeWorkerFailure(
      'RuntimeError: Command failed (1): python -m demucs.separate',
      1,
    ),
    '音轨分离进程异常退出。请确认磁盘空间充足后重试。',
  )
})

test('extracts supported URLs from app share text', () => {
  const shared =
    '分享歌曲：示例标题 https://music.163.com/song?id=12345 （来自网易云音乐）'
  assert.equal(normalizeSharedUrl(shared), 'https://music.163.com/song?id=12345')
  assert.equal(sourceFromLink(shared).kind, 'netease')
  assert.equal(detectPlatform('https://bili2233.cn/example')?.id, 'bilibili')
  assert.equal(detectPlatform('https://163.fm/example')?.id, 'netease')
  assert.equal(
    normalizeSharedUrl(
      'https://y.qq.com/n/ryqq_v2/songDetail/004Ti8rT003TaZ',
    ),
    'https://y.qq.com/n/ryqq/songDetail/004Ti8rT003TaZ',
  )
})

test('reports the actual media acquisition failure category', () => {
  const youtube = detectPlatform('https://youtu.be/example')
  assert.ok(youtube)
  assert.match(
    describeMediaAcquisitionFailure(
      youtube,
      'ERROR: Sign in to confirm you are not a bot',
    ),
    /登录会话.*人机验证/u,
  )
  assert.match(
    describeMediaAcquisitionFailure(youtube, 'ERROR: This video is DRM protected'),
    /DRM/u,
  )
  assert.match(
    describeMediaAcquisitionFailure(youtube, 'ERROR: Unsupported URL'),
    /链接格式/u,
  )
})

test('rejects unsupported or malformed URLs', () => {
  assert.equal(detectPlatform('not-a-url'), null)
  assert.equal(detectPlatform('https://example.com/song'), null)
  assert.throws(() => sourceFromLink('https://example.com/song'))
})

test('demo project uses the requested Country Roads recording and generated TAB', () => {
  const project = createStudioProject({
    kind: 'netease',
    label: 'John Denver · Take Me Home, Country Roads',
    url: 'https://music.163.com/#/song?id=1478037',
  })

  assert.equal(project.schemaVersion, 2)
  assert.equal(project.guitarParts.length, 1)
  assert.equal(project.primaryGuitarPartId, project.guitarParts[0].id)
  assert.ok(project.tab.measures.length > 100)
  assert.equal(project.tracks.length, 7)
  assert.ok(project.visualization)
  assert.equal(project.title, 'Take Me Home, Country Roads')
  assert.equal(project.artist, 'John Denver')
  assert.equal(project.bpm, 161.5)
  assert.equal(project.key, 'A major')
  assert.equal(project.transcriptionModel?.version, '0.4')
  assert.equal(
    project.techniqueModel?.name,
    'unreviewed-technique-candidates',
  )
  assert.ok(Math.abs(project.duration - 194.14667) < 0.001)
  assert.equal(project.tab.capo, 0)
  assert.deepEqual(
    project.tab.measures.flatMap((measure) =>
      measure.section ? [[measure.number, measure.section]] : [],
    ),
    [
      [1, '前奏'],
    ],
  )
  assert.deepEqual(
    [...new Set(project.tab.measures.map((measure) => measure.chord))].sort(),
    ['A', 'A7', 'B7', 'C#m', 'D', 'D7', 'E', 'E7', 'F#7', 'F#m', 'N.C.'],
  )
  assert.ok(
    project.tab.measures.flatMap((measure) =>
      measure.beats.flatMap((beat) => beat.notes),
    ).length > 200,
  )
  assert.deepEqual(
    project.tracks
      .filter((track) => track.available)
      .map((track) => track.id),
    ['vocals', 'guitar-1', 'guitar-2', 'bass', 'drums', 'piano', 'other'],
  )
  assert.equal(
    project.tracks.find((track) => track.role === 'vocals')?.url,
    '/media/builtin/take-me-home-country-roads-vocals.m4a',
  )
  assert.equal(
    project.tracks.find((track) => track.role === 'vocals')?.available,
    true,
  )
  assert.equal(
    project.tracks.find((track) => track.id === 'guitar-1')?.url,
    '/media/builtin/take-me-home-country-roads-guitar-1.m4a',
  )
  assert.ok(
    project.tab.measures.every((measure) =>
      measure.beats.every((beat) =>
        beat.notes.every(
          (note) =>
            note.fret >= 0 &&
            note.fret <= 24 &&
            note.string >= 1 &&
            note.string <= 6 &&
            (note.positionConfidence ?? 0) > 0 &&
            note.confidence >= 0 &&
            note.confidence <= 1,
        ),
      ),
    ),
  )
  const beatStyles = new Set<string | undefined>(
    project.tab.measures.flatMap((measure) => measure.beats.map((beat) => beat.style)),
  )
  assert.ok(beatStyles.has('pick'))

  const techniques = new Set<string>(
    project.tab.measures.flatMap((measure) =>
      measure.beats.flatMap((beat) => beat.notes.map((note) => note.technique)),
    ),
  )
  assert.ok(techniques.has('unknown'))
  assert.ok(project.tab.measures.every((measure) =>
    measure.beats.every((beat) =>
      beat.notes.every((note) =>
        (note.notationAt ?? note.at) >= measure.start &&
        (note.notationAt ?? note.at) < measure.start + measure.duration,
      ),
    ),
  ))
})

test('builds independently namespaced TAB documents for guitar parts', () => {
  const analysis: WorkerAnalysis = {
    version: 10,
    bpm: 120,
    stems: {},
    notes: [],
    chords: [],
    warnings: [],
    guitar_parts: [
      {
        id: 'guitar-2',
        track_id: 'guitar-2',
        name: '节奏吉他',
        role: 'rhythm',
        role_confidence: 0.86,
        notes: [
          {
            start: 0,
            end: 0.4,
            pitch: 45,
            velocity: 84,
            string: 5,
            fret: 0,
            technique: 'pick',
            confidence: 0.9,
          },
        ],
        chords: [{ start: 0, duration: 2, chord: 'A' }],
      },
    ],
  }

  const tab = tabFromWorker(analysis, analysis.guitar_parts![0])
  const notes = tab.measures.flatMap((measure) =>
    measure.beats.flatMap((beat) => beat.notes),
  )

  assert.equal(notes[0].id, 'guitar-2-1')
  assert.equal(tab.measures[0].chord, 'A')
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

test('builds complete measures for every supported meter family', () => {
  const fixtures = [
    { signature: [2, 4] as const, interval: 0.5 },
    { signature: [3, 4] as const, interval: 0.5 },
    { signature: [4, 4] as const, interval: 0.5 },
    { signature: [6, 8] as const, interval: 0.25 },
    { signature: [12, 8] as const, interval: 0.25 },
  ]
  for (const { signature, interval } of fixtures) {
    const beats = Array.from(
      { length: signature[0] * 2 + 1 },
      (_, index) => index * interval,
    )
    const analysis: WorkerAnalysis = {
      version: 9,
      bpm: 60 / interval,
      time_signature: [signature[0], signature[1]],
      metadata: { duration: signature[0] * interval * 2 },
      beats,
      stems: {},
      notes: [],
      chords: [],
      warnings: [],
    }

    const tab = tabFromWorker(analysis)

    assert.deepEqual(tab.timeSignature, signature)
    assert.ok(tab.measures.length >= 2)
    assert.ok(
      tab.measures.every(
        (measure) => measure.beats.length === signature[0],
      ),
    )
  }
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

test('rejects only sources without a dedicated media adapter', () => {
  assert.equal(
    sourceFromLink(
      'https://y.qq.com/n/ryqq_v2/songDetail/004Ti8rT003TaZ',
    ).kind,
    'qqmusic',
  )
  assert.throws(
    () => sourceFromLink('https://qishui.douyin.com/s/example'),
    /QQ 音乐.*Wikimedia Commons/u,
  )
})

test('resolves isolated shadow execution and candidate fallback models', () => {
  assert.deepEqual(
    resolveInferenceExecution({
      mode: 'shadow',
      promotionId: 'promotion-shadow',
      modelVersion: 'model-shadow',
      artifactPath: '/models/shadow.bin',
      baselineArtifactPath: '/models/champion.bin',
      baselineFallbackArtifactPath: '/models/previous.bin',
    }),
    {
      mode: 'shadow',
      promotionId: 'promotion-shadow',
      modelVersion: 'model-shadow',
      primaryModelPath: '/models/champion.bin',
      fallbackModelPaths: ['/models/previous.bin', undefined],
      shadowModelPath: '/models/shadow.bin',
    },
  )
  assert.deepEqual(
    resolveInferenceExecution({
      mode: 'canary',
      promotionId: 'promotion-canary',
      modelVersion: 'model-canary',
      artifactPath: '/models/canary.bin',
      baselineArtifactPath: '/models/champion.bin',
      baselineFallbackArtifactPath: '/models/previous.bin',
    }),
    {
      mode: 'canary',
      promotionId: 'promotion-canary',
      modelVersion: 'model-canary',
      primaryModelPath: '/models/canary.bin',
      fallbackModelPaths: [
        '/models/champion.bin',
        '/models/previous.bin',
        undefined,
      ],
    },
  )
  assert.deepEqual(
    resolveInferenceExecution({
      mode: 'champion',
      promotionId: 'promotion-champion',
      modelVersion: 'model-champion',
      artifactPath: '/models/champion.bin',
      baselineArtifactPath: '/models/previous.bin',
    }),
    {
      mode: 'champion',
      promotionId: 'promotion-champion',
      modelVersion: 'model-champion',
      primaryModelPath: '/models/champion.bin',
      fallbackModelPaths: ['/models/previous.bin', undefined],
    },
  )
  assert.deepEqual(
    resolveInferenceExecution({
      mode: 'canary',
      promotionId: 'promotion-invalid',
    }),
    { mode: 'baseline', fallbackModelPaths: [] },
  )
})

test('terminates a worker when inference is cancelled or times out', async () => {
  const root = resolve(process.cwd(), 'data', `.test-worker-${randomUUID()}`)
  const script = resolve(root, 'worker.mjs')
  mkdirSync(root, { recursive: true })
  writeFileSync(script, 'setTimeout(() => undefined, 30_000)\n', 'utf8')
  try {
    const controller = new AbortController()
    const cancelled = executeWorker(
      process.execPath,
      process.execPath,
      [script],
      resolve(root, 'cancelled'),
      undefined,
      controller.signal,
      10_000,
    )
    setTimeout(() => controller.abort(), 30)
    await assert.rejects(cancelled, /任务已取消/)

    await assert.rejects(
      executeWorker(
        process.execPath,
        process.execPath,
        [script],
        resolve(root, 'timed-out'),
        undefined,
        undefined,
        20,
      ),
      /推理任务超时/,
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
