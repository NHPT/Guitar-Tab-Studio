import assert from 'node:assert/strict'
import test from 'node:test'
import { projectFromOpticalScore } from './score-import.js'

test('converts optical TAB events into an editable constant-tempo project', () => {
  const project = projectFromOpticalScore(
    {
      version: 1,
      title: 'practice-score',
      pageCount: 2,
      staffCount: 3,
      timeSignature: [4, 4],
      bpm: 90,
      confidence: 0.82,
      measures: [
        {
          chord: 'Em',
          source: { page: 1, staff: 1 },
          events: [
            {
              position: 0,
              duration: 0.5,
              string: 6,
              fret: 0,
              confidence: 0.94,
              source: 'ocr',
            },
            {
              position: 0.5,
              duration: 0.5,
              string: 5,
              fret: 2,
              confidence: 0.89,
              source: 'ocr',
            },
          ],
        },
        {
          chord: 'G',
          source: { page: 1, staff: 1 },
          events: [],
        },
      ],
      lyrics: [{ position: 0.5, text: 'home' }],
      warnings: [],
    },
    'practice.pdf',
  )

  assert.equal(project.title, 'practice-score')
  assert.equal(project.source.filename, 'practice.pdf')
  assert.equal(project.tab.measures.length, 2)
  assert.equal(project.tab.measures[0].beats.length, 4)
  assert.equal(project.tab.measures[0].beats[0].notes.length, 2)
  assert.equal(project.tab.measures[0].beats[0].notes[1].fret, 2)
  assert.equal(
    project.tab.measures[0].beats[0].notes[0].positionSource,
    'source-score-omr-v1',
  )
  assert.equal(project.tab.lyrics?.[0].text, 'home')
  assert.equal(project.tracks[0].available, true)
  assert.equal(project.tracks[0].url, null)
  assert.equal(project.duration, 16 / 3)
})
