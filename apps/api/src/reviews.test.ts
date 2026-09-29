import assert from 'node:assert/strict'
import { createHash, randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import {
  approveReview,
  getReview,
  listReviews,
  reopenReview,
  reviewContentDigest,
  saveReviewDraft,
} from './reviews.js'
import type { ReviewPackage } from './reviews.js'

const REVIEW_AUDIO = Buffer.from('RIFF')
const REVIEW_AUDIO_SHA256 = createHash('sha256')
  .update(REVIEW_AUDIO)
  .digest('hex')

function fixture(): ReviewPackage {
  return {
    schema_version: 1,
    status: 'draft',
    review_id: 'project-1-m1',
    project_id: 'project-1',
    project_sha256: 'a'.repeat(64),
    source_audio_sha256: 'b'.repeat(64),
    review_audio: 'real-reference-01.wav',
    review_audio_sha256: REVIEW_AUDIO_SHA256,
    source_kind: 'upload',
    measure_numbers: [1],
    source_start: 10,
    source_end: 12,
    duration: 2,
    capo: 0,
    bpm: 120,
    beats: [0, 0.5, 1, 1.5],
    time_signature: [4, 4],
    events: [
      {
        onset: 0.25,
        offset: 0.5,
        string: 6,
        fret: 3,
        technique: 'unknown',
        confidence: 1,
        candidate_provenance: {
          project_note_id: 'note-1',
          position_source: 'manual',
          technique_source: 'unknown',
        },
      },
    ],
    checks: {
      timing: false,
      string_fret: false,
      completeness: false,
      technique: false,
    },
    approval: null,
  }
}

function withReviewRoot(run: (root: string) => void): void {
  const root = resolve(process.cwd(), 'data', `.test-reviews-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  writeFileSync(
    resolve(root, 'real-reference-01.json'),
    `${JSON.stringify(fixture(), null, 2)}\n`,
    'utf8',
  )
  writeFileSync(resolve(root, 'real-reference-01.wav'), REVIEW_AUDIO)
  try {
    run(root)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

test('matches the Python canonical review digest', () => {
  assert.equal(
    reviewContentDigest(fixture()),
    '09e40a04b5fd3e4340d03b5136cb26fe6b959a19ac6774b4cd9c21b5aa81888d',
  )
})

test('lists and reads only allowlisted review packages', () => {
  withReviewRoot((root) => {
    writeFileSync(resolve(root, 'notes.json'), '{}', 'utf8')
    const projectRoot = resolve(root, 'projects')
    mkdirSync(projectRoot)
    writeFileSync(
      resolve(projectRoot, 'project-1.json'),
      JSON.stringify({
        title: 'Independent song',
        artist: 'Independent artist',
      }),
      'utf8',
    )
    assert.deepEqual(
      listReviews(root).map((review) => review.name),
      ['real-reference-01'],
    )
    assert.deepEqual(
      listReviews(root, projectRoot).map((review) => ({
        title: review.projectTitle,
        artist: review.projectArtist,
      })),
      [{ title: 'Independent song', artist: 'Independent artist' }],
    )
    assert.equal(
      getReview('real-reference-01', root).audioUrl,
      '/media/training/reviews/real-reference-01.wav',
    )
    assert.throws(() => getReview('../real-reference-01', root), /不存在/)
    assert.throws(() => getReview('notes', root), /不存在/)
    writeFileSync(resolve(root, 'real-reference-01.wav'), Buffer.from('changed'))
    assert.throws(
      () => getReview('real-reference-01', root),
      /审核音频哈希不匹配/,
    )
  })
})

test('isolates residual-onset review packages and media paths', () => {
  withReviewRoot((root) => {
    const residual = {
      ...fixture(),
      review_kind: 'residual-onset-training' as const,
      display_title: 'P1 · Palm mute · Mic + amp',
      display_subtitle: 'Training stem · 120.0-132.0s',
      review_id: 'residual-onset-p1-palm-mute-01',
      project_id: 'guitar-techs-p1-palm-mute',
      review_audio: 'residual-onset-01.wav',
      training_audio: 'residual-onset-01.training.wav',
      training_audio_sha256: createHash('sha256')
        .update(Buffer.from('TRAINING'))
        .digest('hex'),
      training_provenance: {
        source_manifest: 'technique-mixture-demucs-v1/manifest-combined.jsonl',
        source_manifest_sha256: 'd'.repeat(64),
        source_track_id: 'guitar-techs-P1-palm-mute-micamp',
        review_source_track_id: 'guitar-techs-P1-palm-mute',
        source_group_id: 'guitar-techs-P1-palm-mute',
        source_split: 'train' as const,
        source_dataset: 'Guitar-TECHS',
        source_license: 'CC-BY-4.0',
        audio_condition: 'procedural-mixture-demucs-guitar-stem',
        training_source_audio_sha256: 'e'.repeat(64),
        audio_alignment: 'sample-aligned-no-offset' as const,
        review_audio_processing:
          'rms-normalized--20-dbfs-soft-limited--1-dbfs' as const,
        review_audio_gain_db: 55,
      },
    }
    writeFileSync(
      resolve(root, 'residual-onset-01.json'),
      `${JSON.stringify(residual, null, 2)}\n`,
      'utf8',
    )
    writeFileSync(resolve(root, 'residual-onset-01.wav'), REVIEW_AUDIO)
    writeFileSync(
      resolve(root, 'residual-onset-01.training.wav'),
      Buffer.from('TRAINING'),
    )

    const summaries = listReviews(root)
    assert.deepEqual(
      summaries.map((review) => review.name),
      ['real-reference-01', 'residual-onset-01'],
    )
    assert.equal(summaries[1].projectTitle, residual.display_title)
    assert.equal(summaries[1].projectArtist, residual.display_subtitle)
    assert.equal(summaries[1].reviewKind, 'residual-onset-training')
    assert.equal(
      getReview(
        'residual-onset-01',
        root,
        '/media/training/reviews/residual-onset',
      ).audioUrl,
      '/media/training/reviews/residual-onset/residual-onset-01.wav',
    )
    const approved = approveReview(
      'residual-onset-01',
      {
        reviewerAlias: 'reviewer-1',
        events: residual.events,
        checks: {
          timing: true,
          string_fret: false,
          completeness: true,
          technique: false,
        },
      },
      root,
      '/media/training/reviews/residual-onset',
    )
    assert.equal(approved.review.status, 'approved')
    assert.equal(approved.review.checks.string_fret, false)
    writeFileSync(
      resolve(root, 'residual-onset-01.training.wav'),
      Buffer.from('changed'),
    )
    assert.throws(
      () =>
        getReview(
          'residual-onset-01',
          root,
          '/media/training/reviews/residual-onset',
        ),
      /训练音频哈希不匹配/,
    )
  })
})

test('saves a validated draft and orders events by onset', () => {
  withReviewRoot((root) => {
    const saved = saveReviewDraft(
      'real-reference-01',
      {
        capo: 2,
        events: [
          {
            onset: 1,
            offset: 1.2,
            string: 2,
            fret: 7,
            technique: 'slide',
            confidence: 1,
          },
          fixture().events[0],
        ],
        checks: {
          timing: true,
          string_fret: false,
          completeness: false,
          technique: false,
        },
      },
      root,
    )
    assert.equal(saved.review.capo, 2)
    assert.deepEqual(
      saved.review.events.map((event) => event.onset),
      [0.25, 1],
    )
    assert.equal(saved.review.checks.timing, true)
    assert.equal(
      JSON.parse(
        readFileSync(resolve(root, 'real-reference-01.json'), 'utf8'),
      ).status,
      'draft',
    )
  })
})

test('saves ordered excluded ranges and rejects invalid overlaps', () => {
  withReviewRoot((root) => {
    const review = fixture()
    const saved = saveReviewDraft(
      'real-reference-01',
      {
        events: review.events,
        checks: review.checks,
        excluded_ranges: [
          { start: 1.2, end: 1.4, reason: 'second uncertain passage' },
          { start: 0.6, end: 0.9, reason: ' first uncertain passage ' },
        ],
      },
      root,
    )
    assert.deepEqual(saved.review.excluded_ranges, [
      { start: 0.6, end: 0.9, reason: 'first uncertain passage' },
      { start: 1.2, end: 1.4, reason: 'second uncertain passage' },
    ])

    assert.throws(
      () =>
        saveReviewDraft(
          'real-reference-01',
          {
            events: review.events,
            checks: review.checks,
            excluded_ranges: [
              { start: 0.5, end: 1, reason: 'uncertain' },
              { start: 0.9, end: 1.2, reason: 'overlap' },
            ],
          },
          root,
        ),
      /不能重叠/,
    )
    assert.throws(
      () =>
        saveReviewDraft(
          'real-reference-01',
          {
            events: review.events,
            checks: review.checks,
            excluded_ranges: [
              { start: 1.8, end: 2.1, reason: 'outside clip' },
            ],
          },
          root,
        ),
      /时间范围无效/,
    )
  })
})

test('requires complete checks and binds approved content', () => {
  withReviewRoot((root) => {
    const review = fixture()
    assert.throws(
      () =>
        approveReview(
          'real-reference-01',
          {
            reviewerAlias: 'reviewer-1',
            events: review.events,
            checks: {
              timing: true,
              string_fret: true,
              completeness: false,
              technique: false,
            },
          },
          root,
        ),
      /批准前必须完成检查/,
    )

    const approved = approveReview(
      'real-reference-01',
      {
        reviewerAlias: 'reviewer-1',
        events: review.events,
        checks: {
          timing: true,
          string_fret: true,
          completeness: true,
          technique: false,
        },
      },
      root,
    )
    assert.equal(approved.review.status, 'approved')
    assert.equal(
      approved.review.approval?.content_sha256,
      reviewContentDigest(approved.review),
    )
    assert.throws(
      () =>
        saveReviewDraft(
          'real-reference-01',
          { events: review.events, checks: review.checks },
          root,
        ),
      /不能再修改/,
    )

    const changed = JSON.parse(
      readFileSync(resolve(root, 'real-reference-01.json'), 'utf8'),
    ) as ReviewPackage
    changed.events[0].fret = 12
    writeFileSync(
      resolve(root, 'real-reference-01.json'),
      JSON.stringify(changed),
      'utf8',
    )
    assert.throws(() => getReview('real-reference-01', root), /发生变化/)
  })
})

test('rejects invalid event boundaries and reviewer aliases', () => {
  withReviewRoot((root) => {
    const review = fixture()
    assert.throws(
      () =>
        saveReviewDraft(
          'real-reference-01',
          {
            events: [{ ...review.events[0], onset: -1 }],
            checks: review.checks,
          },
          root,
        ),
      /时间范围无效/,
    )
    assert.throws(
      () =>
        approveReview(
          'real-reference-01',
          {
            reviewerAlias: '../../owner',
            events: review.events,
            checks: {
              timing: true,
              string_fret: true,
              completeness: true,
              technique: false,
            },
          },
          root,
        ),
      /审核人代号/,
    )
  })
})

test('reopens an approved review with audit history and resets timing', () => {
  withReviewRoot((root) => {
    const review = fixture()
    approveReview(
      'real-reference-01',
      {
        reviewerAlias: 'reviewer-1',
        events: review.events,
        checks: {
          timing: true,
          string_fret: true,
          completeness: true,
          technique: true,
        },
      },
      root,
    )

    const reopened = reopenReview(
      'real-reference-01',
      'correct event onset',
      root,
    )
    assert.equal(reopened.review.status, 'draft')
    assert.equal(reopened.review.approval, null)
    assert.equal(reopened.review.checks.timing, false)
    assert.equal(reopened.review.checks.string_fret, true)
    assert.equal(reopened.review.approval_history?.length, 1)
    assert.equal(
      reopened.review.approval_history?.[0].reason,
      'correct event onset',
    )

    const correctedEvents = reopened.review.events.map((event, index) =>
      index === 0 ? { ...event, onset: 0.24 } : event,
    )
    const saved = saveReviewDraft(
      'real-reference-01',
      {
        capo: reopened.review.capo,
        events: correctedEvents,
        checks: reopened.review.checks,
      },
      root,
    )
    assert.equal(saved.review.events[0].onset, 0.24)
  })
})
