import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import test from 'node:test'
import {
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
  saveBeatReviewDraft,
  submitBeatReview,
  type BeatReviewPackage,
} from './beat-reviews.js'

function sha256(value: Buffer): string {
  return createHash('sha256').update(value).digest('hex')
}

async function fixtureRoot(): Promise<string> {
  const root = resolve(process.cwd(), '.test-beat-reviews')
  const reviewerRoot = resolve(root, 'reviews', 'reviewer-a')
  await rm(root, { recursive: true, force: true })
  await mkdir(reviewerRoot, { recursive: true })
  await mkdir(resolve(root, 'analysis'), { recursive: true })
  const audio = Buffer.from('RIFF-beat-review-fixture')
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
      time_signature: [4, 4],
      pickup_beats: 0,
      beats: Array.from({ length: 15 }, (_, index) => index + 0.5),
    }, null, 2)}\n`,
  )
  return root
}

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
