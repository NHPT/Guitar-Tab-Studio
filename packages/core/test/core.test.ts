import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ANALYSIS_SCHEMA_VERSION,
  arrangementPreservesPitch,
  authorizedMediaProvider,
  authorizedMediaProviderForContent,
  countInBeatDurations,
  CoreValidationError,
  createEasyArrangement,
  detectStandardAudioSignature,
  MemoryProjectRepository,
  parseModelPackageManifest,
  parseProjectRevision,
  PRACTICE_RHYTHMS,
  migrateStudioProject,
  parseAnalysisDocument,
  practiceTempoProfile,
  requiresContentInspection,
  requiresAuthorizedMediaDecode,
  rhythmCompatibility,
  rhythmCompatibilityForTempoProfile,
  rhythmEventPosition,
  uploadContentValidationError,
  uploadPrefilterError,
} from '../src/index.js'

function legacyProject() {
  return {
    id: 'project-1',
    title: 'Legacy project',
    artist: 'Fixture',
    source: {
      kind: 'upload',
      label: 'fixture',
      filename: 'fixture.wav',
    },
    bpm: 120,
    key: 'E minor',
    duration: 2,
    createdAt: '2026-09-29T00:00:00.000Z',
    analysisMode: 'inference',
    tracks: [],
    tab: {
      tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
      capo: 0,
      measures: [],
    },
    warnings: [],
  }
}

test('migrates an unversioned project to schema version 1', () => {
  const project = migrateStudioProject(legacyProject())
  assert.equal(project.schemaVersion, 1)
  assert.equal(project.source.kind, 'upload')
  assert.deepEqual(project.tab.tuning, ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'])
  assert.deepEqual(project.tab.timeSignature, [4, 4])
})

test('deduplicates legacy tracks and withholds ungated technique labels', () => {
  const track = {
    id: 'guitar-1',
    name: '吉他 1',
    role: 'guitar',
    color: '#19c37d',
    url: '/media/guitar-1.m4a',
    waveform: [0.2],
    available: false,
  }
  const project = migrateStudioProject({
    ...legacyProject(),
    techniqueModel: {
      name: 'stringtrace-technique-baseline',
      version: 'heuristic-v4',
      kind: 'heuristic',
      labels: ['unknown', 'slap'],
    },
    tracks: [track, track],
    tab: {
      ...legacyProject().tab,
      measures: [{
        number: 1,
        start: 0,
        duration: 2,
        chord: 'D',
        beats: [{
          at: 0,
          duration: 0.5,
          style: 'tremolo',
          notes: [{
            id: 'unreviewed',
            string: 4,
            fret: 0,
            at: 0,
            duration: 0.2,
            technique: 'slap',
            confidence: 0.8,
            techniqueConfidence: 0.9,
            techniqueSource: 'acoustic-heuristic-v2',
          }],
        }],
      }],
    },
  })
  assert.equal(project.tracks.length, 1)
  assert.equal(project.tracks[0].available, true)
  assert.equal(project.tab.measures[0].beats[0].style, 'pick')
  assert.equal(project.tab.measures[0].beats[0].notes[0].technique, 'unknown')
  assert.equal(project.tab.measures[0].beats[0].notes[0].techniqueConfidence, 0)
})

test('preserves structured project provenance', () => {
  const project = migrateStudioProject({
    ...legacyProject(),
    provenance: {
      compositionCredit: 'Traditional folk song',
      arrangementCredit: 'Project arrangement',
      recordingCredit: 'Project recording',
      assetLicense: 'Apache-2.0',
      note: 'Independent arrangement and recording.',
      references: ['https://example.com/catalog'],
    },
  })
  assert.deepEqual(project.provenance, {
    compositionCredit: 'Traditional folk song',
    arrangementCredit: 'Project arrangement',
    recordingCredit: 'Project recording',
    assetLicense: 'Apache-2.0',
    note: 'Independent arrangement and recording.',
    references: ['https://example.com/catalog'],
  })
})

test('accepts legacy and current protected-media confirmation records', () => {
  for (const policyVersion of [1, 2] as const) {
    const project = migrateStudioProject({
      ...legacyProject(),
      source: {
        ...legacyProject().source,
        authorizedMedia: {
          provider: 'qqmusic',
          purpose: 'analysis-and-score-generation',
          decodedOutput: 'ephemeral',
          policyVersion,
        },
      },
    })
    assert.equal(project.source.authorizedMedia?.policyVersion, policyVersion)
  }
  assert.throws(
    () =>
      migrateStudioProject({
        ...legacyProject(),
        source: {
          ...legacyProject().source,
          authorizedMedia: {
            provider: 'qqmusic',
            purpose: 'analysis-and-score-generation',
            decodedOutput: 'ephemeral',
            policyVersion: 3,
          },
        },
      }),
    /expected 1 or 2/u,
  )
})

test('preserves authored ties in project TAB notes', () => {
  const project = migrateStudioProject({
    ...legacyProject(),
    tab: {
      ...legacyProject().tab,
      measures: [
        {
          number: 1,
          start: 0,
          duration: 2,
          chord: 'Em',
          beats: [
            {
              at: 0,
              duration: 1,
              notes: [
                {
                  id: 'tied-note',
                  string: 4,
                  fret: 2,
                  at: 0,
                  duration: 1,
                  technique: 'pick',
                  confidence: 1,
                  tieToNext: true,
                },
              ],
            },
          ],
        },
      ],
    },
  })
  assert.equal(project.tab.measures[0].beats[0].notes[0].tieToNext, true)
})

test('preserves explicit time signatures and filters incompatible rhythms', () => {
  const project = migrateStudioProject({
    ...legacyProject(),
    tab: {
      ...legacyProject().tab,
      timeSignature: [6, 8],
    },
  })
  const sixEightRock = PRACTICE_RHYTHMS.find(
    (rhythm) => rhythm.id === 'six-eight-rock',
  )
  const house = PRACTICE_RHYTHMS.find((rhythm) => rhythm.id === 'house')
  assert.ok(PRACTICE_RHYTHMS.length >= 30)
  assert.ok(sixEightRock)
  assert.ok(house)
  assert.deepEqual(project.tab.timeSignature, [6, 8])
  assert.equal(
    rhythmCompatibility(sixEightRock, 100, project.tab.timeSignature)
      .compatible,
    true,
  )
  assert.equal(
    rhythmCompatibility(house, 122, project.tab.timeSignature).reason,
    'time-signature',
  )
  assert.equal(
    rhythmCompatibility(sixEightRock, 170, project.tab.timeSignature).reason,
    'tempo-high',
  )
})

test('anchors rhythm steps to the detected beat grid within each measure', () => {
  const rock = PRACTICE_RHYTHMS.find((rhythm) => rhythm.id === 'rock-basic')
  assert.ok(rock)
  const measure = {
    number: 1,
    start: 0.6,
    duration: 2,
    chord: 'Em',
    beats: [
      { at: 0.6, duration: 0.46, notes: [] },
      { at: 1.06, duration: 0.52, notes: [] },
      { at: 1.58, duration: 0.43, notes: [] },
      { at: 2.01, duration: 0.59, notes: [] },
    ],
  }
  assert.equal(rhythmEventPosition(measure, rock, 0), 0.6)
  assert.equal(rhythmEventPosition(measure, rock, 4), 1.06)
  assert.equal(rhythmEventPosition(measure, rock, 8), 1.58)
  assert.equal(rhythmEventPosition(measure, rock, 12), 2.01)
  assert.equal(rhythmEventPosition(measure, rock, 2), 0.83)
  assert.deepEqual(
    countInBeatDurations(120, 4),
    [0.5, 0.5, 0.5, 0.5],
  )
  for (const rhythm of PRACTICE_RHYTHMS.filter(
    (candidate) =>
      candidate.timeSignature[0] === 4 &&
      candidate.timeSignature[1] === 4,
  )) {
    const stepsPerBeat = rhythm.stepsPerBar / measure.beats.length
    measure.beats.forEach((beat, beatIndex) => {
      assert.equal(
        rhythmEventPosition(measure, rhythm, beatIndex * stepsPerBeat),
        beat.at,
      )
    })
  }
  const profile = practiceTempoProfile([measure])
  assert.ok(profile)
  assert.ok(profile.low < profile.typical)
  assert.ok(profile.typical < profile.high)
  const house = PRACTICE_RHYTHMS.find((rhythm) => rhythm.id === 'house')
  assert.ok(house)
  assert.equal(
    rhythmCompatibilityForTempoProfile(house, profile, [4, 4]).compatible,
    false,
  )
})

test('scales count-in intervals with every project tempo and playback speed', () => {
  for (const bpm of [60, 96, 117.45, 120, 123.05, 180]) {
    const sourceDuration = countInBeatDurations(bpm, 4)[0]
    for (const speed of [0.5, 0.75, 1, 1.25]) {
      assert.ok(
        Math.abs(sourceDuration / speed - 60 / (bpm * speed)) < 1e-12,
      )
    }
  }
})

test('supports count-in and rhythm downbeats for common time signatures', () => {
  const fixtures = [
    {
      rhythmId: 'waltz-classic',
      bpm: 90,
      beats: [
        { at: 0, duration: 0.66, notes: [] },
        { at: 0.66, duration: 0.68, notes: [] },
        { at: 1.34, duration: 0.66, notes: [] },
      ],
    },
    {
      rhythmId: 'six-eight-rock',
      bpm: 120,
      beats: Array.from({ length: 6 }, (_, index) => ({
        at: index * 0.25,
        duration: 0.25,
        notes: [],
      })),
    },
  ]

  fixtures.forEach((fixture, fixtureIndex) => {
    const rhythm = PRACTICE_RHYTHMS.find(
      (candidate) => candidate.id === fixture.rhythmId,
    )
    assert.ok(rhythm)
    assert.equal(
      countInBeatDurations(fixture.bpm, rhythm.timeSignature[0]).length,
      rhythm.timeSignature[0],
    )
    const measure = {
      number: fixtureIndex + 1,
      start: 0,
      duration: fixture.beats.reduce(
        (total, beat) => total + beat.duration,
        0,
      ),
      chord: 'N.C.',
      beats: fixture.beats,
    }
    const stepsPerBeat = rhythm.stepsPerBar / measure.beats.length
    measure.beats.forEach((beat, beatIndex) => {
      assert.equal(
        rhythmEventPosition(measure, rhythm, beatIndex * stepsPerBeat),
        beat.at,
      )
    })
  })
})

test('rejects unsupported project versions and malformed project fields', () => {
  assert.throws(
    () => migrateStudioProject({ ...legacyProject(), schemaVersion: 2 }),
    (error: unknown) =>
      error instanceof CoreValidationError &&
      error.path === 'project.schemaVersion',
  )
  assert.throws(
    () =>
      migrateStudioProject({
        ...legacyProject(),
        source: { kind: 'unsupported', label: 'fixture' },
      }),
    /project\.source\.kind/,
  )
})

test('provides an in-memory project repository for non-HTTP clients', () => {
  const repository = new MemoryProjectRepository([legacyProject()])
  const project = repository.getProject('project-1')
  assert.equal(project?.schemaVersion, 1)
  assert.equal(repository.listProjects().length, 1)
  if (!project) assert.fail('project was not seeded')
  repository.saveProject({ ...project, title: 'Updated project' })
  assert.equal(repository.getProject(project.id)?.title, 'Updated project')
  repository.removeProject(project.id)
  assert.equal(repository.getProject(project.id), undefined)
})

test('creates a lower-position arrangement without changing any pitch', () => {
  const project = migrateStudioProject({
    ...legacyProject(),
    tab: {
      ...legacyProject().tab,
      measures: [
        {
          number: 1,
          start: 0,
          duration: 2,
          chord: 'C',
          beats: [
            {
              at: 0,
              duration: 1,
              notes: [
                {
                  id: 'high-position-e4',
                  string: 2,
                  fret: 5,
                  at: 0,
                  duration: 1,
                  technique: 'pick',
                  confidence: 0.9,
                },
              ],
            },
          ],
        },
      ],
    },
  })
  const arrangement = createEasyArrangement(
    project,
    '2026-09-29T01:00:00.000Z',
  )
  const arranged = arrangement.tab.measures[0].beats[0].notes[0]
  assert.equal(arrangement.strategyVersion, 'pitch-preserving-v1')
  assert.equal(arranged.string, 1)
  assert.equal(arranged.fret, 0)
  assert.equal(arrangementPreservesPitch(project.tab, arrangement.tab), true)
})

test('validates immutable project revision snapshots', () => {
  const project = migrateStudioProject(legacyProject())
  const revision = parseProjectRevision({
    schemaVersion: 1,
    id: 'revision-1',
    projectId: project.id,
    createdAt: '2026-09-29T01:00:00.000Z',
    label: '人工校订',
    kind: 'manual',
    project,
  })
  assert.equal(revision.project.title, project.title)
  assert.throws(
    () => parseProjectRevision({ ...revision, projectId: 'other' }),
    /must match/,
  )
})

test('normalizes supported legacy analysis documents to schema version 9', () => {
  const analysis = parseAnalysisDocument({
    version: 8,
    bpm: 120,
    time_signature: [6, 8],
    vocal_melody: [
      { start: 1, end: 1.5, pitch: 69, confidence: 0.9 },
    ],
    lyrics: [{ at: 1, text: 'sample' }],
    transcription_model: {
      name: 'hybrid',
      version: 5,
      kind: 'hybrid',
    },
    stems: {},
    stem_tracks: [
      {
        id: 'guitar-1',
        name: '吉他 1',
        role: 'guitar',
        path: '/managed/guitar-1.m4a',
        level: 0.2,
        waveform: [0.1, 0.4],
        separation_confidence: 0.82,
        separation_method: 'stereo-spatial-mask-v1',
      },
    ],
    notes: [
      {
        start: 0.1,
        end: 0.4,
        pitch: 57,
        velocity: 80,
        string: 3,
        fret: 2,
        technique: 'pick',
        confidence: 0.9,
      },
    ],
  })
  assert.equal(analysis.version, ANALYSIS_SCHEMA_VERSION)
  assert.equal(analysis.transcription_model?.kind, 'trained')
  assert.equal(analysis.transcription_model?.version, '5')
  assert.deepEqual(analysis.time_signature, [6, 8])
  assert.equal(analysis.vocal_melody?.[0].pitch, 69)
  assert.equal(analysis.lyrics?.[0].text, 'sample')
  assert.equal(analysis.notes[0].string, 3)
  assert.equal(analysis.stem_tracks?.[0].id, 'guitar-1')
  assert.equal(analysis.stem_tracks?.[0].separation_confidence, 0.82)
  assert.deepEqual(analysis.chords, [])
  assert.deepEqual(analysis.warnings, [])
})

test('rejects future analysis versions and invalid guitar strings', () => {
  assert.throws(
    () =>
      parseAnalysisDocument({
        version: ANALYSIS_SCHEMA_VERSION + 1,
        bpm: 120,
        notes: [],
      }),
    /analysis\.version/,
  )
  assert.throws(
    () =>
      parseAnalysisDocument({
        version: 9,
        bpm: 120,
        notes: [
          {
            start: 0,
            end: 1,
            pitch: 60,
            velocity: 80,
            string: 7,
            fret: 1,
            technique: 'pick',
            confidence: 0.8,
          },
        ],
      }),
    /analysis\.notes\[0\]\.string/,
  )
})

test('shares media extension and signature policy across runtimes', () => {
  assert.equal(uploadPrefilterError('clip.WAV', 'application/octet-stream'), null)
  for (const extension of [
    'mflac0',
    'mgg0',
    'mgg1',
    'kgg',
    'kgma',
  ]) {
    assert.equal(
      requiresContentInspection(`encrypted.${extension}`),
      true,
    )
    assert.equal(
      uploadPrefilterError(
        `encrypted.${extension}`,
        'application/octet-stream',
      ),
      null,
    )
  }
  assert.equal(authorizedMediaProvider('encrypted.ncm'), 'netease')
  assert.equal(authorizedMediaProvider('encrypted.qmcflac'), 'qqmusic')
  assert.equal(authorizedMediaProvider('encrypted.mflac0'), 'qqmusic')
  assert.equal(authorizedMediaProvider('encrypted.kgg'), 'kugou')
  assert.equal(authorizedMediaProvider('encrypted.kgma'), 'kugou')
  assert.equal(
    authorizedMediaProviderForContent(
      'encrypted.mflac0',
      Buffer.from('yeelion-kuwo-tme'),
    ),
    'kuwo',
  )
  assert.equal(
    authorizedMediaProviderForContent(
      'encrypted.mgg1',
      Buffer.from('7cd532eb86027f4ba8afa68e0fff9914', 'hex'),
    ),
    'kugou',
  )
  assert.equal(
    requiresAuthorizedMediaDecode(
      'encrypted.ncm',
      new Uint8Array([0, 1, 2, 3]),
    ),
    true,
  )
  assert.match(
    uploadContentValidationError(
      'encrypted.ncm',
      'application/octet-stream',
      new Uint8Array([0, 1, 2, 3]),
    ) ?? '',
    /专有加密容器/,
  )
  const wav = new Uint8Array(12)
  wav.set(Buffer.from('RIFF'), 0)
  wav.set(Buffer.from('WAVE'), 8)
  assert.equal(detectStandardAudioSignature(wav), true)
  assert.equal(requiresAuthorizedMediaDecode('export.mflac', wav), false)
  assert.equal(
    uploadContentValidationError(
      'export.mflac',
      'application/octet-stream',
      wav,
    ),
    null,
  )
})

test('rejects model package traversal and unsupported analysis schemas', () => {
  const manifest = {
    schemaVersion: 1,
    id: 'gts-tab',
    version: '1.0.0',
    createdAt: '2026-09-29T00:00:00.000Z',
    application: { minimumVersion: '0.1.0' },
    runtime: {
      platform: 'darwin',
      arch: 'arm64',
      analysisSchemaVersion: 9,
    },
    files: [
      {
        path: '../model.bin',
        role: 'transcription-model',
        sizeBytes: 1,
        sha256: 'a'.repeat(64),
        executable: false,
      },
    ],
  }
  assert.throws(() => parseModelPackageManifest(manifest), /stay in package/)
  manifest.files[0].path = 'model.bin'
  manifest.runtime.analysisSchemaVersion = 10
  assert.throws(
    () => parseModelPackageManifest(manifest),
    /analysisSchemaVersion/,
  )
})
