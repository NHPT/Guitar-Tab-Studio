import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import test from 'node:test'
import {
  classifyBeatErrors,
  evaluateBeatGold,
  listBeatEvaluationRuns,
  saveBeatEvaluationRun,
} from './beat-evaluation.js'
import {
  beatGoldReleaseArchivePath,
  createBeatGoldRelease,
  listBeatGoldReleases,
} from './beat-gold-releases.js'
import {
  beatTrainingStatus,
  createBeatTrainingDataset,
  createBeatTrainingJob,
} from './beat-training.js'
import {
  createBeatBackup,
  restoreBeatBackup,
  verifyBeatBackup,
} from './beat-backups.js'
import { reviewLayerStatuses } from './review-layers.js'
import {
  adjudicateBeatReview,
  beatGoldStatus,
  beatReviewContentDigest,
  BeatReviewRequestError,
  compareBeatReviews,
  getBeatReview,
  listBeatAdjudication,
  listBeatGold,
  listBeatReviews,
  regenerateBeatReviewDraftsFromStructure,
  saveBeatReviewDraft,
  submitBeatReview,
  type BeatReviewPackage,
} from './beat-reviews.js'
import {
  beatStructureAudioPath,
  beatStructureContentDigest,
  getBeatStructureReview,
  listBeatStructureReviews,
  reopenBeatStructureReview,
  saveBeatStructureDraft,
  submitBeatStructureReview,
  type BeatStructureReviewPackage,
} from './beat-structure-reviews.js'

function sha256(value: Buffer): string {
  return createHash('sha256').update(value).digest('hex')
}

async function fixtureRoot(): Promise<string> {
  const root = resolve(process.cwd(), '.test-beat-reviews')
  const reviewerRoot = resolve(root, 'reviews', 'reviewer-a')
  const sourceRoot = resolve(root, 'sources')
  await rm(root, { recursive: true, force: true })
  await mkdir(reviewerRoot, { recursive: true })
  await mkdir(resolve(root, 'analysis'), { recursive: true })
  await mkdir(sourceRoot, { recursive: true })
  const audio = Buffer.from('RIFF-beat-review-fixture')
  const sourceAudio = Buffer.from('OggS-full-song-structure-fixture')
  await writeFile(resolve(sourceRoot, 'song.media'), sourceAudio)
  await writeFile(resolve(reviewerRoot, 'song-01.wav'), audio)
  const review: BeatReviewPackage = {
    schema_version: 2,
    review_kind: 'beat-grid',
    status: 'draft',
    corpus_version: 'beat-pilot-v2-test',
    clip_id: 'song-01',
    group_id: 'song-recording',
    annotator_alias: 'reviewer-a',
    audio: 'song-01.wav',
    source_audio_sha256: 'a'.repeat(64),
    audio_sha256: sha256(audio),
    audio_source_start: 6,
    audio_source_end: 14,
    audio_duration: 8,
    evaluation_start: 2,
    evaluation_end: 6,
    evaluation_duration: 4,
    time_signature: [4, 4],
    pickup_beats: 0,
    beats: [
      { at: 0.5, beat: 4, measure: 7, downbeat: false },
      { at: 1.5, beat: 1, measure: 8, downbeat: true },
      { at: 2.5, beat: 2, measure: 8, downbeat: false },
      { at: 3.5, beat: 3, measure: 8, downbeat: false },
      { at: 4.5, beat: 4, measure: 8, downbeat: false },
      { at: 5.5, beat: 1, measure: 9, downbeat: true },
      { at: 6.5, beat: 2, measure: 9, downbeat: false },
    ],
    tags: ['train', 'acoustic'],
    checks: {
      beat_times: false,
      downbeats: false,
      meter: false,
      completeness: false,
    },
    candidate_source: { kind: 'worker-analysis' },
    submission: null,
    adjudication: null,
    verification_status: 'draft',
    benchmark_eligible: false,
  }
  await writeFile(
    resolve(reviewerRoot, 'song-01.json'),
    `${JSON.stringify(review, null, 2)}\n`,
  )
  await writeFile(
    resolve(root, 'pilot-manifest.json'),
    `${JSON.stringify({
      schema_version: 2,
      kind: 'beat-gold-pilot-v2',
      corpus_version: 'beat-pilot-v2-test',
      corpus_sha256: 'b'.repeat(64),
      immutable: true,
      evaluation_duration: 4,
      context: { pre_roll: 2, post_roll: 2 },
      verification_status: 'draft',
      benchmark_eligible: false,
      recordings: 1,
      clips: 1,
      split_counts: { train: 1, validation: 0, test: 0 },
      items: [
        {
          clip_id: 'song-01',
          recording_id: 'song',
          split: 'train',
          audio_source_start: 6,
          audio_source_end: 14,
          audio_duration: 8,
          evaluation_start: 2,
          evaluation_end: 6,
          evaluation_duration: 4,
          audio_sha256: sha256(audio),
        },
      ],
    }, null, 2)}\n`,
  )
  await writeFile(
    resolve(root, 'analysis', 'song.json'),
    `${JSON.stringify({
      source_audio_sha256: sha256(sourceAudio),
      duration: 15.5,
      bpm: 60,
      time_signature: [4, 4],
      pickup_beats: 0,
      beats: Array.from({ length: 15 }, (_, index) => index + 0.5),
    }, null, 2)}\n`,
  )
  return root
}

test('classifies beat timing failures without gold-data leakage', () => {
  const clean = {
    true_positive: 16,
    false_positive: 0,
    false_negative: 0,
    precision: 1,
    recall: 1,
    f1: 1,
    mean_error_ms: 5,
    median_error_ms: 5,
    p95_error_ms: 8,
    tolerance_ms: 70,
  }
  const halfTempo = {
    ...clean,
    true_positive: 8,
    false_negative: 8,
    recall: 0.5,
    f1: 2 / 3,
  }
  assert.deepEqual(
    classifyBeatErrors(16, 8, halfTempo, clean, true, true),
    ['tempo-half'],
  )
  assert.deepEqual(
    classifyBeatErrors(
      16,
      16,
      { ...clean, median_error_ms: 25, p95_error_ms: 60 },
      { ...clean, f1: 0.5 },
      false,
      false,
    ),
    ['timing-offset', 'downbeat-phase', 'meter', 'pickup'],
  )
})

test('lists only manifest-backed beat reviews and verifies audio', async () => {
  const root = await fixtureRoot()
  try {
    assert.deepEqual(beatGoldStatus(root), {
      enabled: true,
      reviewers: ['reviewer-a'],
      manifest: {
        corpusVersion: 'beat-pilot-v2-test',
        corpusSha256: 'b'.repeat(64),
        immutable: true,
        evaluationDuration: 4,
        context: { preRoll: 2, postRoll: 2 },
        recordings: 1,
        clips: 1,
        splitCounts: { train: 1, validation: 0, test: 0 },
        verificationStatus: 'draft',
        benchmarkEligible: false,
      },
    })
    const reviews = listBeatReviews('reviewer-a', root)
    assert.equal(reviews.length, 1)
    assert.equal(reviews[0].clipId, 'song-01')
    assert.equal(reviews[0].split, 'train')
    const document = getBeatReview('song-01', 'reviewer-a', root)
    assert.equal(document.review.audio_sha256, sha256(Buffer.from('RIFF-beat-review-fixture')))
    assert.match(document.audioUrl, /reviewer-a\/song-01\.wav$/u)
    assert.throws(
      () => listBeatReviews('../reviewer-a', root),
      BeatReviewRequestError,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('rebuilds untouched clip drafts from submitted full-song structure', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const originalClip = await readFile(
      resolve(root, 'reviews', 'reviewer-a', 'song-01.json'),
      'utf8',
    )
    const reviews = listBeatStructureReviews('reviewer-a', root)
    assert.equal(reviews.length, 1)
    assert.equal(reviews[0].recordingId, 'song')
    assert.equal(reviews[0].status, 'draft')
    const document = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.equal(document.candidateBeats.length, 15)
    assert.equal(document.review.first_downbeat_at, 0.5)
    assert.equal(
      beatStructureAudioPath('song', root, sourceRoot),
      resolve(sourceRoot, 'song.media'),
    )

    const draft = saveBeatStructureDraft(
      'song',
      'reviewer-a',
      {
        time_signature: [3, 4],
        pickup_beats: 1,
        phase_status: 'confirmed',
        first_downbeat_at: 1.5,
        meter_changes: [{ at: 14.5, time_signature: [4, 4] }],
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: false,
        },
      },
      root,
      sourceRoot,
    )
    assert.deepEqual(draft.review.time_signature, [3, 4])
    assert.equal(draft.review.pickup_beats, 1)
    assert.equal(draft.review.first_downbeat_at, 1.5)
    assert.equal(draft.review.meter_changes[0].at, 14.5)
    const reloadedDraft = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.equal(reloadedDraft.review.pickup_beats, 1)
    assert.equal(
      beatStructureContentDigest(reloadedDraft.review),
      beatStructureContentDigest(draft.review),
    )
    assert.throws(
      () =>
        submitBeatStructureReview(
          'song',
          'reviewer-a',
          { checks: draft.review.checks },
          root,
          sourceRoot,
        ),
      /提交前需完成检查/u,
    )
    const submitted = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
      { adminUserId: 'user-a', adminUsername: 'owner' },
    )
    assert.equal(submitted.review.status, 'submitted')
    assert.equal(
      submitted.review.submission?.content_sha256,
      beatStructureContentDigest(submitted.review),
    )
    const regeneratedClip = getBeatReview(
      'song-01',
      'reviewer-a',
      root,
    )
    assert.notEqual(
      await readFile(
        resolve(root, 'reviews', 'reviewer-a', 'song-01.json'),
        'utf8',
      ),
      originalClip,
    )
    assert.deepEqual(regeneratedClip.review.time_signature, [3, 4])
    assert.equal(regeneratedClip.review.pickup_beats, 1)
    assert.equal(regeneratedClip.review.beats.length, 8)
    assert.equal(regeneratedClip.review.beats.at(-1)?.at, 7.5)
    assert.deepEqual(regeneratedClip.review.checks, {
      beat_times: false,
      downbeats: false,
      meter: false,
      completeness: false,
    })
    assert.equal(
      regeneratedClip.review.candidate_source.kind,
      'reviewed-full-song-structure',
    )
    assert.equal(
      regeneratedClip.review.candidate_source.structure_content_sha256,
      submitted.review.submission?.content_sha256,
    )
    const backup = createBeatBackup(root)
    restoreBeatBackup(backup.id, root)
    const restoredStructure = JSON.parse(
      await readFile(
        resolve(
          root,
          '.restores',
          backup.id,
          '.test-beat-reviews',
          'structure-reviews',
          'reviewer-a',
          'song.json',
        ),
        'utf8',
      ),
    ) as { status: string }
    assert.equal(restoredStructure.status, 'submitted')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('does not overwrite clip drafts after manual checks have started', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const touched = saveBeatReviewDraft(
      'song-01',
      'reviewer-a',
      {
        checks: {
          beat_times: true,
          downbeats: false,
          meter: false,
          completeness: false,
        },
      },
      root,
    )
    submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        excluded_candidate_beats: [8.5],
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    const preserved = getBeatReview('song-01', 'reviewer-a', root)
    assert.deepEqual(preserved.review.beats, touched.review.beats)
    assert.equal(preserved.review.checks.beat_times, true)
    assert.equal(
      preserved.review.candidate_source.kind,
      'worker-analysis',
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('explicitly rebuilds a touched clip draft during a controlled migration', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    saveBeatReviewDraft(
      'song-01',
      'reviewer-a',
      {
        checks: {
          beat_times: true,
          downbeats: false,
          meter: false,
          completeness: false,
        },
      },
      root,
    )
    const structure = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    const result = regenerateBeatReviewDraftsFromStructure(
      {
        recordingId: 'song',
        reviewer: 'reviewer-a',
        corpusVersion: structure.review.corpus_version,
        timeSignature: structure.review.time_signature,
        pickupBeats: structure.review.pickup_beats,
        firstDownbeatAt: structure.review.first_downbeat_at ?? 0,
        meterChanges: structure.review.meter_changes,
        candidateBeats: structure.candidateBeats,
        contentSha256:
          structure.review.submission?.content_sha256 ?? '',
      },
      root,
      { overwriteTouched: true },
    )
    assert.deepEqual(result, { regenerated: 1, skipped: 0 })
    const regenerated = getBeatReview('song-01', 'reviewer-a', root)
    assert.equal(regenerated.review.checks.beat_times, false)
    assert.equal(
      regenerated.review.candidate_source.kind,
      'reviewed-full-song-structure',
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('numbers regenerated measures from the absolute first downbeat index', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const saved = saveBeatStructureDraft(
      'song',
      'reviewer-a',
      { first_downbeat_at: 9.5 },
      root,
      sourceRoot,
    )
    assert.equal(saved.review.first_downbeat_at, 9.5)
    submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    const regenerated = getBeatReview('song-01', 'reviewer-a', root)
    const firstDownbeat = regenerated.review.beats.find(
      (beat) => beat.at === 3.5,
    )
    assert.deepEqual(firstDownbeat, {
      at: 3.5,
      beat: 1,
      measure: 1,
      downbeat: true,
    })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('saves and submits quarter-beat pickups and rejects invalid values', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { pickup_beats: 0.1 },
          root,
          sourceRoot,
        ),
      /0\.25 拍为步进/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { pickup_beats: -0.25 },
          root,
          sourceRoot,
        ),
      /不能为负数/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { pickup_beats: 4 },
          root,
          sourceRoot,
        ),
      /必须小于每小节拍数/u,
    )

    const saved = saveBeatStructureDraft(
      'song',
      'reviewer-a',
      { pickup_beats: 0.25 },
      root,
      sourceRoot,
    )
    assert.equal(saved.review.pickup_beats, 0.25)
    assert.equal(
      getBeatStructureReview('song', 'reviewer-a', root, sourceRoot)
        .review.pickup_beats,
      0.25,
    )

    const submitted = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    assert.equal(submitted.review.pickup_beats, 0.25)
    assert.equal(
      getBeatStructureReview('song', 'reviewer-a', root, sourceRoot)
        .review.pickup_beats,
      0.25,
    )
    assert.equal(
      getBeatReview('song-01', 'reviewer-a', root).review.pickup_beats,
      0.25,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('filters excluded full-song candidate beats and protects structure anchors', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const initial = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.equal(initial.rawCandidateBeats.length, 15)
    assert.equal(initial.candidateBeats.length, 15)
    assert.deepEqual(initial.review.excluded_candidate_beats, [])
    assert.deepEqual(initial.review.added_candidate_beats, [])

    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { excluded_candidate_beats: [2.75] },
          root,
          sourceRoot,
        ),
      /不在原始候选中/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { excluded_candidate_beats: [2.5, 2.5] },
          root,
          sourceRoot,
        ),
      /不能重复/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { excluded_candidate_beats: [0.5] },
          root,
          sourceRoot,
        ),
      /首个强拍必须选择候选拍点/u,
    )

    const saved = saveBeatStructureDraft(
      'song',
      'reviewer-a',
      { excluded_candidate_beats: [2.5] },
      root,
      sourceRoot,
    )
    assert.deepEqual(saved.review.excluded_candidate_beats, [2.5])
    assert.equal(saved.rawCandidateBeats.length, 15)
    assert.equal(saved.candidateBeats.length, 14)
    assert.equal(saved.candidateBeats.includes(2.5), false)
    assert.notEqual(
      beatStructureContentDigest(saved.review),
      beatStructureContentDigest(initial.review),
    )

    const submitted = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    assert.equal(submitted.review.status, 'submitted')
    assert.equal(submitted.candidateBeats.includes(2.5), false)
    assert.equal(
      submitted.review.submission?.content_sha256,
      beatStructureContentDigest(submitted.review),
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('adds full-song candidate beats and persists their effective order', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const initial = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { added_candidate_beats: [2.5] },
          root,
          sourceRoot,
        ),
      /与现有候选拍点重复/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { added_candidate_beats: [-0.1] },
          root,
          sourceRoot,
        ),
      /超出音频范围/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { added_candidate_beats: [2.75, 2.7505] },
          root,
          sourceRoot,
        ),
      /不能重复/u,
    )
    assert.throws(
      () =>
        saveBeatStructureDraft(
          'song',
          'reviewer-a',
          { added_candidate_beats: [2.51] },
          root,
          sourceRoot,
        ),
      /间隔不足 0\.03 秒/u,
    )

    const saved = saveBeatStructureDraft(
      'song',
      'reviewer-a',
      {
        added_candidate_beats: [2.75],
        first_downbeat_at: 2.75,
        meter_changes: [
          { at: 2.75, time_signature: [3, 4] },
        ],
      },
      root,
      sourceRoot,
    )
    assert.deepEqual(saved.review.added_candidate_beats, [2.75])
    assert.equal(saved.review.first_downbeat_at, 2.75)
    assert.deepEqual(saved.review.meter_changes, [
      { at: 2.75, time_signature: [3, 4] },
    ])
    assert.deepEqual(saved.candidateBeats.slice(1, 5), [
      1.5,
      2.5,
      2.75,
      3.5,
    ])
    assert.notEqual(
      beatStructureContentDigest(saved.review),
      beatStructureContentDigest(initial.review),
    )

    const reloaded = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.deepEqual(reloaded.review.added_candidate_beats, [2.75])
    assert.equal(reloaded.candidateBeats.includes(2.75), true)

    const submitted = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    assert.equal(submitted.review.status, 'submitted')
    assert.deepEqual(submitted.review.added_candidate_beats, [2.75])
    assert.equal(
      submitted.review.submission?.content_sha256,
      beatStructureContentDigest(submitted.review),
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('loads legacy submitted structure reviews without an added-beats field', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const submitted = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
    )
    const legacy = structuredClone(
      submitted.review,
    ) as BeatStructureReviewPackage
    delete legacy.added_candidate_beats
    if (!legacy.submission) throw new Error('submission missing')
    legacy.submission.content_sha256 = beatStructureContentDigest(legacy)
    await writeFile(
      resolve(root, 'structure-reviews', 'reviewer-a', 'song.json'),
      `${JSON.stringify(legacy, null, 2)}\n`,
    )

    const loaded = getBeatStructureReview(
      'song',
      'reviewer-a',
      root,
      sourceRoot,
    )
    assert.equal(loaded.review.added_candidate_beats, undefined)
    assert.deepEqual(loaded.candidateBeats, loaded.rawCandidateBeats)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('reopens submitted structure and invalidates linked clips and gold', async () => {
  const root = await fixtureRoot()
  const sourceRoot = resolve(root, 'sources')
  try {
    const leftRoot = resolve(root, 'reviews', 'reviewer-a')
    const rightRoot = resolve(root, 'reviews', 'reviewer-b')
    await mkdir(rightRoot, { recursive: true })
    await writeFile(
      resolve(rightRoot, 'song-01.wav'),
      await readFile(resolve(leftRoot, 'song-01.wav')),
    )
    const right = JSON.parse(
      await readFile(resolve(leftRoot, 'song-01.json'), 'utf8'),
    ) as BeatReviewPackage
    right.annotator_alias = 'reviewer-b'
    await writeFile(
      resolve(rightRoot, 'song-01.json'),
      `${JSON.stringify(right, null, 2)}\n`,
    )

    const structure = submitBeatStructureReview(
      'song',
      'reviewer-a',
      {
        checks: {
          meter: true,
          downbeat_phase: true,
          pickup: true,
          meter_changes: true,
        },
      },
      root,
      sourceRoot,
      { adminUserId: 'structure-user', adminUsername: 'structure-owner' },
    )
    assert.equal(structure.review.status, 'submitted')

    const checks = {
      beat_times: true,
      downbeats: true,
      meter: true,
      completeness: true,
    }
    const left = submitBeatReview(
      'song-01',
      'reviewer-a',
      { checks },
      root,
      { adminUserId: 'user-a', adminUsername: 'reviewer-a-user' },
    )
    submitBeatReview(
      'song-01',
      'reviewer-b',
      { checks },
      root,
      { adminUserId: 'user-b', adminUsername: 'reviewer-b-user' },
    )
    adjudicateBeatReview(
      'song-01',
      'reviewer-a',
      'reviewer-b',
      {
        beats: left.review.beats,
        time_signature: left.review.time_signature,
        pickup_beats: left.review.pickup_beats,
        checks,
      },
      { adminUserId: 'user-c', adminUsername: 'adjudicator' },
      root,
    )
    assert.equal(listBeatGold(root).length, 1)

    const reopened = reopenBeatStructureReview(
      'song',
      'reviewer-a',
      'correct downbeat phase',
      root,
      sourceRoot,
      { adminUserId: 'owner-id', adminUsername: 'owner' },
    )
    assert.equal(reopened.clipsInvalidated, 1)
    assert.equal(reopened.goldInvalidated, 1)
    assert.equal(reopened.document.review.status, 'draft')
    assert.deepEqual(reopened.document.review.checks, {
      meter: false,
      downbeat_phase: false,
      pickup: false,
      meter_changes: false,
    })
    assert.equal(
      reopened.document.review.submission_history?.[0].invalidation_reason,
      'correct downbeat phase',
    )

    const invalidated = getBeatReview('song-01', 'reviewer-a', root).review
    assert.equal(invalidated.status, 'draft')
    assert.equal(invalidated.checks.beat_times, true)
    assert.equal(invalidated.checks.completeness, true)
    assert.equal(invalidated.checks.downbeats, false)
    assert.equal(invalidated.checks.meter, false)
    assert.equal(invalidated.submission, null)
    assert.equal(invalidated.verification_status, 'draft')
    assert.equal(
      invalidated.structure_invalidation_history?.[0].reason,
      'correct downbeat phase',
    )
    assert.equal(
      getBeatReview('song-01', 'reviewer-b', root).review.status,
      'submitted',
    )
    assert.equal(listBeatGold(root).length, 0)
    assert.throws(
      () =>
        reopenBeatStructureReview(
          'song',
          'reviewer-a',
          'repeat',
          root,
          sourceRoot,
        ),
      /只有已提交/u,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('preserves full-song marker phase and submits only the evaluation window', async () => {
  const root = await fixtureRoot()
  try {
    const draft = saveBeatReviewDraft(
      'song-01',
      'reviewer-a',
      {
        beats: [
          { at: 0.5, beat: 4, measure: 7, downbeat: false },
          { at: 1.5, beat: 1, measure: 8, downbeat: true },
          { at: 2.6, beat: 2, measure: 8, downbeat: false },
          { at: 3.5, beat: 3, measure: 8, downbeat: false },
          { at: 4.5, beat: 4, measure: 8, downbeat: false },
          { at: 5.5, beat: 1, measure: 9, downbeat: true },
          { at: 6.5, beat: 2, measure: 9, downbeat: false },
        ],
        checks: {
          beat_times: true,
          downbeats: true,
          meter: false,
          completeness: true,
        },
      },
      root,
    )
    assert.deepEqual(
      draft.review.beats.map((beat) => [
        beat.at,
        beat.beat,
        beat.measure,
        beat.downbeat,
      ]),
      [
        [0.5, 4, 7, false],
        [1.5, 1, 8, true],
        [2.6, 2, 8, false],
        [3.5, 3, 8, false],
        [4.5, 4, 8, false],
        [5.5, 1, 9, true],
        [6.5, 2, 9, false],
      ],
    )
    assert.throws(
      () =>
        submitBeatReview(
          'song-01',
          'reviewer-a',
          {
            checks: draft.review.checks,
          },
          root,
        ),
      /提交前需完成检查/u,
    )
    const submitted = submitBeatReview(
      'song-01',
      'reviewer-a',
      {
        checks: {
          beat_times: true,
          downbeats: true,
          meter: true,
          completeness: true,
        },
      },
      root,
    )
    assert.equal(submitted.review.status, 'submitted')
    assert.equal(
      submitted.review.submission?.content_sha256,
      beatReviewContentDigest(submitted.review),
    )
    assert.throws(
      () =>
        saveBeatReviewDraft(
          'song-01',
          'reviewer-a',
          { beats: draft.review.beats },
          root,
        ),
      /不能直接修改/u,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('requires two accounts and a third-party adjudicator for gold', async () => {
  const root = await fixtureRoot()
  try {
    const leftRoot = resolve(root, 'reviews', 'reviewer-a')
    const rightRoot = resolve(root, 'reviews', 'reviewer-b')
    await mkdir(rightRoot, { recursive: true })
    await writeFile(
      resolve(rightRoot, 'song-01.wav'),
      await readFile(resolve(leftRoot, 'song-01.wav')),
    )
    const right = JSON.parse(
      await readFile(resolve(leftRoot, 'song-01.json'), 'utf8'),
    ) as BeatReviewPackage
    right.annotator_alias = 'reviewer-b'
    right.beats = right.beats.map((marker) => ({
      ...marker,
      at: marker.at + 0.01,
    }))
    await writeFile(
      resolve(rightRoot, 'song-01.json'),
      `${JSON.stringify(right, null, 2)}\n`,
    )
    const checks = {
      beat_times: true,
      downbeats: true,
      meter: true,
      completeness: true,
    }
    const leftSubmitted = submitBeatReview(
      'song-01',
      'reviewer-a',
      { checks },
      root,
      { adminUserId: 'user-a', adminUsername: 'reviewer-a-user' },
    )
    submitBeatReview(
      'song-01',
      'reviewer-b',
      { checks },
      root,
      { adminUserId: 'user-b', adminUsername: 'reviewer-b-user' },
    )
    const comparison = compareBeatReviews(
      'song-01',
      'reviewer-a',
      'reviewer-b',
      root,
    )
    assert.equal(comparison.independentAccounts, true)
    assert.equal(comparison.beatMetrics.f1, 1)
    assert.equal(comparison.differences[0].deltaMs, 10)
    assert.equal(listBeatAdjudication('reviewer-a', 'reviewer-b', root)[0].ready, true)
    assert.throws(
      () =>
        adjudicateBeatReview(
          'song-01',
          'reviewer-a',
          'reviewer-b',
          {
            beats: leftSubmitted.review.beats,
            time_signature: [4, 4],
            pickup_beats: 0,
            checks,
          },
          { adminUserId: 'user-a', adminUsername: 'reviewer-a-user' },
          root,
        ),
      /独立于两名复核员/u,
    )
    const gold = adjudicateBeatReview(
      'song-01',
      'reviewer-a',
      'reviewer-b',
      {
        beats: leftSubmitted.review.beats,
        time_signature: [4, 4],
        pickup_beats: 0,
        checks,
      },
      { adminUserId: 'user-c', adminUsername: 'adjudicator-user' },
      root,
    )
    assert.equal(gold.status, 'adjudicated')
    assert.equal(gold.verification_status, 'adjudicated_gold')
    assert.equal(gold.benchmark_eligible, true)
    assert.equal(listBeatGold(root).length, 1)
    const report = evaluateBeatGold('candidate-v1', root)
    assert.equal(report.coverage.complete, true)
    assert.equal(report.aggregate.beatMetrics.f1, 1)
    assert.equal(report.gate.passed, false)
    saveBeatEvaluationRun('candidate-v1', root)
    assert.equal(listBeatEvaluationRuns(root).length, 1)
    const release = createBeatGoldRelease('gold-v1', root)
    assert.equal(release.clips, 1)
    assert.equal(listBeatGoldReleases(root).length, 1)
    assert.match(beatGoldReleaseArchivePath('gold-v1', root), /gold-v1\.tar\.gz$/u)
    const dataset = createBeatTrainingDataset('train-v1', root)
    assert.equal(dataset.clips, 1)
    assert.equal(dataset.readyForTraining, false)
    const job = createBeatTrainingJob(
      {
        datasetVersion: 'train-v1',
        codeRevision: 'test-revision',
      },
      root,
    )
    assert.equal(job.status, 'blocked')
    assert.equal(beatTrainingStatus(root).jobs.length, 1)
    const layers = reviewLayerStatuses(root)
    assert.equal(layers.find((layer) => layer.id === 'beat')?.status, 'active')
    assert.equal(
      layers.find((layer) => layer.id === 'onset')?.status,
      'scaffolded',
    )
    assert.equal(
      layers.find((layer) => layer.id === 'pitch')?.status,
      'blocked',
    )
    const backup = createBeatBackup(root)
    assert.equal(verifyBeatBackup(backup.id, root).verified, true)
    assert.equal(restoreBeatBackup(backup.id, root).restored, true)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
