import { expect, test, type Page } from '@playwright/test'
import type { StudioProject } from '../apps/web/src/types'

async function seekExactly(page: Page, seconds: number, duration: number) {
  await page.locator('.waveform').evaluate((element, ratio) => {
    const bounds = element.getBoundingClientRect()
    const event = new MouseEvent('click', {
      bubbles: true,
      clientY: bounds.top + bounds.height / 2,
    })
    // Native click coordinates round to pixels; retain a fractional position
    // here to test score timing independently of the waveform's seek resolution.
    Object.defineProperty(event, 'clientX', {
      value: bounds.left + bounds.width * ratio,
    })
    element.dispatchEvent(event)
  }, seconds / duration)
}

test('places notes on the playhead at their notation time across unequal measures and resizing', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const allNotes = project.tab.measures.flatMap(measure =>
    measure.beats.flatMap(beat => beat.notes),
  )
  const sample = [
    allNotes[4],
    allNotes[8],
    allNotes[18],
    allNotes[Math.floor(allNotes.length / 3)],
    allNotes[Math.floor((allNotes.length * 2) / 3)],
    allNotes.at(-2),
  ].filter((note): note is NonNullable<typeof note> => Boolean(note))
  await page.goto('/')
  await expect(page.locator('.tab-glyph').first()).toBeVisible()
  const originalSize = page.viewportSize()!
  for (const width of [originalSize.width, Math.max(360, originalSize.width - 150)]) {
    await page.setViewportSize({ ...originalSize, width })
    for (const note of sample) {
      const notationAt = Number(
        await page
          .locator(`[data-note-id="${note.id}"]`)
          .getAttribute('data-notation-at'),
      )
      await seekExactly(page, notationAt, project.duration)
      await expect.poll(() =>
        page.locator(`[data-note-id="${note.id}"] text`).evaluate(text => {
          const glyph = text.getBoundingClientRect()
          const playhead = document.querySelector('.score-playhead')!.getBoundingClientRect()
          return Math.abs((glyph.left + glyph.right - playhead.left - playhead.right) / 2)
        }),
      ).toBeLessThanOrEqual(2)
    }
  }
})

test('marks three evenly spaced subdivisions as a triplet and rejects uneven attacks', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const measure = project.tab.measures.find((candidate) =>
    candidate.beats.length >= 3
  )!
  const baseNote = measure.beats[0].notes[0]
  const patterns = [[0, 1 / 3, 2 / 3], [0, 0.25, 0.5], [0.23, 0.33, 0.69]]
  patterns.forEach((positions, index) => {
    const beat = measure.beats[index]
    beat.style = 'pick'
    beat.notes = positions.map((position, noteIndex) => ({
      ...baseNote,
      id: `rhythm-${index}-${noteIndex}`,
      at: beat.at + position * beat.duration,
      notationAt: beat.at + position * beat.duration,
      duration: beat.duration * 0.2,
      technique: 'pick',
    }))
  })
  await page.route('**/api/projects/demo', route => route.fulfill({ json: project }))
  await page.goto('/')
  const slots = page.locator(
    `[data-measure="${measure.number}"] .rhythm-beat-slot`,
  )
  await expect(slots.nth(0)).toHaveAttribute('data-tuplet', '3')
  await expect(slots.nth(1)).not.toHaveAttribute('data-tuplet', '3')
  await expect(slots.nth(2)).not.toHaveAttribute('data-tuplet', '3')
})

test('quantizes expressive attacks as an ordered beat pattern with leading rests', async ({
  page,
}) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const measure = project.tab.measures.find((candidate) =>
    candidate.beats.length >= 2
  )!
  const base = measure.beats.flatMap((beat) => beat.notes)[0]
  expect(base).toBeTruthy()
  measure.beats.forEach((beat) => {
    beat.notes = []
    beat.style = 'pick'
  })
  const patterns = [
    [0.07, 0.40, 0.66, 0.81],
    [0.24, 0.51, 0.76],
  ]
  patterns.forEach((positions, beatIndex) => {
    const beat = measure.beats[beatIndex]
    beat.notes = positions.map((position, noteIndex) => {
      const at = beat.at + position * beat.duration
      return {
        ...base,
        id: `expressive-${beatIndex}-${noteIndex}`,
        at,
        notationAt: at,
        duration: 0.05,
        technique: 'pick' as const,
      }
    })
  })
  await page.route('**/api/projects/demo', route => route.fulfill({ json: project }))
  await page.goto('/')

  const expectedFractions = [
    [0, 0.25, 0.5, 0.75],
    [0.25, 0.5, 0.75],
  ]
  for (const [beatIndex, fractions] of expectedFractions.entries()) {
    const beat = measure.beats[beatIndex]
    for (const [noteIndex, fraction] of fractions.entries()) {
      const glyph = page.locator(
        `[data-note-id="expressive-${beatIndex}-${noteIndex}"]`,
      )
      await expect(glyph).toBeVisible()
      const notationAt = Number(await glyph.getAttribute('data-notation-at'))
      expect(notationAt).toBeCloseTo(
        beat.at + fraction * beat.duration,
        6,
      )
    }
  }
})

test('shows an inferred strum with every fret and preserves its distinct performance times', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const allNotes = project.tab.measures.flatMap(m => m.beats.flatMap(b => b.notes))
  const firstBeat = project.tab.measures[0].beats[0]
  const strumNotes = [...new Map(
    allNotes.map((note) => [note.string, note] as const),
  ).values()].slice(0, 3).map((note, index) => ({
    ...note,
    id: `synthetic-strum-${index}`,
    at: firstBeat.at + index * 0.02,
    notationAt: firstBeat.at,
    technique: 'strum-down' as const,
  }))
  firstBeat.style = 'strum'
  firstBeat.direction = 'down'
  firstBeat.notes = strumNotes
  const syntheticAnchor = strumNotes[0]
  expect(syntheticAnchor).toBeTruthy()
  const notes = strumNotes.filter(note =>
    note.technique === syntheticAnchor.technique &&
    note.notationAt === syntheticAnchor.notationAt,
  )
  expect(notes.length).toBeGreaterThanOrEqual(3)
  expect(new Set(notes.map(note => note.at)).size).toBeGreaterThan(1)
  await page.route('**/api/projects/demo', route =>
    route.fulfill({ json: project }),
  )
  await page.goto('/')
  const notationAt = Number(
    await page
      .locator(`[data-note-id="${syntheticAnchor.id}"]`)
      .getAttribute('data-notation-at'),
  )
  await seekExactly(page, notationAt, project.duration)
  const arrow = page.locator(
    `.strum-mark[data-gesture-at="${notationAt}"]`,
  )
  await expect(arrow).toBeVisible()
  await expect(arrow).toHaveAttribute('data-direction', 'down')
  const xs: number[] = []
  for (const note of notes) {
    const glyph = page.locator(`[data-note-id="${note.id}"]`)
    await expect(glyph).toBeVisible()
    await expect(glyph).toHaveAttribute('data-note-at', String(note.at))
    xs.push(await glyph.locator('text').evaluate(el => {
      const bounds = el.getBoundingClientRect()
      return (bounds.left + bounds.right) / 2
    }))
  }
  expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(1)
})

test('notates attack spacing as quarters and dotted eighths independently of short decay', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const measure = project.tab.measures.find((candidate) =>
    candidate.beats.length >= 4
  )!
  const base = measure.beats[0].notes[0]
  measure.beats.forEach(b => { b.notes = []; b.style = 'pick' })
  const attacks = [measure.beats[0].at, measure.beats[1].at,
    measure.beats[1].at + measure.beats[1].duration * 0.75, measure.beats[2].at]
  attacks.forEach((at, index) => {
    const beat = measure.beats[index === 0 ? 0 : index === 3 ? 2 : 1]
    beat.notes.push({ ...base, id: `spacing-${index}`, at, notationAt: at, duration: 0.07, technique: 'pick' })
  })
  const finalBeat = measure.beats[3]
  for (const [index, fraction] of [0, 0.875].entries()) {
    const at = finalBeat.at + finalBeat.duration * fraction
    finalBeat.notes.push({
      ...base,
      id: `separated-${index}`,
      at,
      notationAt: at,
      duration: 0.07,
      technique: 'pick',
    })
  }
  await page.route('**/api/projects/demo', route => route.fulfill({ json: project }))
  await page.goto('/')
  const slots = page.locator(
    `[data-measure="${measure.number}"] .rhythm-beat-slot`,
  )
  await expect(slots.nth(0).locator('.rhythm-event')).toHaveAttribute('data-duration-beats', '1')
  await expect(slots.nth(0).locator('.rhythm-event')).toHaveAttribute('data-flags', '0')
  await expect(slots.nth(1).locator('.rhythm-event').nth(0)).toHaveAttribute('data-duration-beats', '0.75')
  await expect(slots.nth(1).locator('.rhythm-dot')).toHaveCount(1)
  expect(
    Number(
      await slots.nth(1).locator('.rhythm-dot').getAttribute('cy'),
    ),
  ).toBeLessThan(110)
  await expect(slots.nth(1).locator('.rhythm-event').nth(1)).toHaveAttribute('data-flags', '2')
  await expect(slots.nth(1).locator('.rhythm-beam')).toHaveCount(1)
  const dottedPairBeamLevels = await slots
    .nth(1)
    .locator('.rhythm-beam, .rhythm-flag')
    .evaluateAll(elements =>
      elements.map(element => ({
        level: Number(element.getAttribute('data-beam-level')),
        y: Number(element.getAttribute('y1')),
      })),
    )
  expect(dottedPairBeamLevels.find(beam => beam.level === 1)!.y).toBeLessThan(
    dottedPairBeamLevels.find(beam => beam.level === 0)!.y,
  )
  await expect(slots.nth(3).locator('.rhythm-beam')).toHaveCount(0)
})

test('keeps sustained beats distinct from a following silent beat', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const measure = project.tab.measures.find((candidate) =>
    candidate.beats.length >= 4
  )!
  const base = measure.beats[0].notes[0]
  measure.beats.forEach(b => { b.notes = []; b.style = 'pick' })
  const at = measure.beats[0].at
  measure.beats[0].notes = [{ ...base, id: 'ringing', at, notationAt: at,
    duration: measure.beats[2].at + measure.beats[2].duration * 0.9 - at }]
  await page.route('**/api/projects/demo', route => route.fulfill({ json: project }))
  await page.goto('/')
  const slots = page.locator(
    `[data-measure="${measure.number}"] .rhythm-beat-slot`,
  )
  for (const index of [1, 2]) {
    await expect(slots.nth(index)).toHaveClass(/is-sustain/)
    await expect(slots.nth(index).locator('.rhythm-rest')).toHaveCount(0)
    await expect(slots.nth(index).locator('[data-sustain-note="ringing"]')).toHaveCount(1)
  }
  await expect(slots.nth(3)).toHaveClass(/is-rest/)
})

test('keeps each opening 4/4 measure at four beats including continuations and rests', async ({ page }) => {
  const response = await page.request.get('http://127.0.0.1:8787/api/projects/demo')
  const project = await response.json() as StudioProject
  const fullMeasures = project.tab.measures
    .filter((measure) => measure.beats.length === 4)
    .slice(0, 3)
  await page.goto('/')
  await expect(page.locator('.tab-glyph').first()).toBeVisible()
  for (const measure of fullMeasures) {
    const durations = await page.locator(
      `[data-measure="${measure.number}"] .rhythm-event, ` +
      `[data-measure="${measure.number}"] .rhythm-continuation, ` +
      `[data-measure="${measure.number}"] .rhythm-rest-event`,
    ).evaluateAll(elements => elements.map(el => Number(el.getAttribute('data-duration-beats'))))
    expect(durations.reduce((sum, duration) => sum + duration, 0)).toBeCloseTo(4, 6)
  }
  expect(await page.locator('.rhythm-continuation').count()).toBeGreaterThan(0)
  const overfull = await page.locator('.score-measure').evaluateAll(measures =>
    measures.map(measure => ({
      measure: measure.getAttribute('data-measure'),
      beats: [...measure.querySelectorAll(
        '.rhythm-event, .rhythm-continuation, .rhythm-rest-event',
      )]
        .reduce((sum, el) => sum + Number(el.getAttribute('data-duration-beats')), 0),
    })).filter(measure => measure.beats > 4.0001),
  )
  expect(overfull).toEqual([])
})

test('closes every repaired project measure at four beats without horizontal rewind', async ({ page }) => {
  const projectId = '6f579959-aebf-4d01-96c0-80de5a1eb8b7'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json() as StudioProject
  expect(project.tab.measures).toHaveLength(12)
  expect(project.tracks.filter(track => track.name === '吉他 1')).toHaveLength(1)
  expect(project.tracks.filter(track => track.name === '吉他 2')).toHaveLength(1)

  await page.goto(`/?project=${projectId}`)
  await expect(page.locator('.score-measure')).toHaveCount(12)
  const measures = await page.locator('.score-measure').evaluateAll(elements =>
    elements.map((measure) => {
      const rhythm = [...measure.querySelectorAll<SVGGElement>(
        '.rhythm-event, .rhythm-continuation, .rhythm-rest-event',
      )]
      const positioned = [...measure.querySelectorAll<SVGGElement>(
        '.rhythm-event, .rhythm-continuation',
      )]
        .map(element => ({
          position: Number(element.dataset.rhythmPosition),
          x: Number(element.dataset.rhythmX),
        }))
        .sort((left, right) => left.position - right.position)
      return {
        number: Number(measure.getAttribute('data-measure')),
        total: rhythm.reduce(
          (sum, element) =>
            sum + Number(element.getAttribute('data-duration-beats')),
          0,
        ),
        positions: positioned,
      }
    }),
  )
  expect(measures).toHaveLength(12)
  measures.forEach((measure) => expect(measure.total).toBeCloseTo(4, 9))
  for (const measure of measures) {
    for (let index = 1; index < measure.positions.length; index += 1) {
      expect(
        measure.positions[index].x,
        `measure ${measure.number} rewinds at rhythm position ${measure.positions[index].position}`,
      ).toBeGreaterThanOrEqual(measure.positions[index - 1].x - 0.5)
    }
  }
})

test('uses the corrected quarter-note grid for the repaired country project', async ({ page }) => {
  const projectId = '228e3288-429f-49c6-924e-fda060e366be'
  const response = await page.request.get(
    `http://127.0.0.1:8787/api/projects/${projectId}`,
  )
  const project = await response.json() as StudioProject
  expect(project.bpm).toBe(81.35)
  expect(project.tab.timeSignature).toEqual([4, 4])
  expect(project.tab.measures).toHaveLength(65)
  expect(project.tab.measures.every(measure => measure.duration > 2.5)).toBe(true)
  const beatDurations = project.tab.measures.flatMap(measure =>
    measure.beats.map(beat => beat.duration),
  )
  expect(Math.max(...beatDurations) - Math.min(...beatDurations)).toBeLessThan(
    1e-9,
  )
  const durationJumps = project.tab.measures.slice(1).map(
    (measure, index) =>
      Math.abs(
        measure.duration / project.tab.measures[index].duration - 1,
      ),
  )
  expect(Math.max(...durationJumps)).toBeLessThan(0.006)
  expect(project.tracks.filter(track => track.name === '吉他 1')).toHaveLength(1)
  expect(project.tracks.filter(track => track.name === '吉他 2')).toHaveLength(1)

  await page.goto(`/?project=${projectId}`)
  await expect(page.locator('.score-measure')).toHaveCount(65)
  const openingPatterns = await page
    .locator('[data-measure="1"] .rhythm-beat-slot')
    .evaluateAll(slots =>
      slots.map(slot =>
        [...slot.querySelectorAll('.rhythm-event')].map(event =>
          Number(event.getAttribute('data-duration-beats')),
        ),
      ),
    )
  expect(openingPatterns[0]).toEqual([0.5, 0.25, 0.25])
  expect(openingPatterns[1]).toEqual(openingPatterns[0])
  expect(openingPatterns[2]).toEqual(openingPatterns[0])
  expect(openingPatterns[3]).toEqual([0.25, 0.25, 0.25, 0.25])
  const measures = await page.locator('.score-measure').evaluateAll(elements =>
    elements.map(measure => {
      const events = [...measure.querySelectorAll(
        '.rhythm-event, .rhythm-continuation, .rhythm-rest-event',
      )]
      const positions = [...measure.querySelectorAll<SVGGElement>(
        '.rhythm-event, .rhythm-continuation',
      )]
        .map(element => ({
          position: Number(element.dataset.rhythmPosition),
          x: Number(element.dataset.rhythmX),
        }))
        .sort((left, right) => left.position - right.position)
      return {
        number: Number(measure.getAttribute('data-measure')),
        total: events.reduce(
          (sum, event) =>
            sum + Number(event.getAttribute('data-duration-beats')),
          0,
        ),
        rewinds: positions.slice(1).filter(
          (position, index) => position.x < positions[index].x - 0.5,
        ).length,
      }
    }),
  )
  expect(measures).toHaveLength(65)
  measures.forEach((measure) => expect(measure.total).toBeCloseTo(4, 9))
  expect(measures.filter(measure => measure.rewinds > 0)).toEqual([])

  const scoreVelocities = await page.locator('.score-track').evaluate(
    (track, sourceMeasures) => {
      const trackLeft = track.getBoundingClientRect().left
      const anchors: Array<{
        at: number
        x: number
        event: boolean
      }> = []
      for (const measure of sourceMeasures) {
        const notation = track.querySelector<SVGSVGElement>(
          `[data-measure="${measure.number}"] .tab-notation`,
        )
        const matrix = notation?.getScreenCTM()
        if (!notation || !matrix) continue
        anchors.push({
          at: measure.start,
          x: new DOMPoint(Number(notation.dataset.timelineStart), 0)
            .matrixTransform(matrix).x - trackLeft,
          event: false,
        })
      }
      for (const glyph of track.querySelectorAll<SVGGraphicsElement>(
        '.tab-glyph[data-notation-at], .strum-mark[data-notation-at]',
      )) {
        const bounds = (glyph.querySelector('text') ?? glyph).getBoundingClientRect()
        anchors.push({
          at: Number(glyph.dataset.notationAt),
          x: (bounds.left + bounds.right) / 2 - trackLeft,
          event: true,
        })
      }
      const ordered = anchors
        .sort(
          (left, right) =>
            left.at - right.at || Number(right.event) - Number(left.event),
        )
        .filter(
          (anchor, index, all) =>
            index === 0 || anchor.at !== all[index - 1].at,
        )
      return ordered.slice(1).flatMap((anchor, index) => {
        const previous = ordered[index]
        const elapsed = anchor.at - previous.at
        return elapsed > 0.0001
          ? [(anchor.x - previous.x) / elapsed]
          : []
      })
    },
    project.tab.measures.map((measure) => ({
      number: measure.number,
      start: measure.start,
    })),
  )
  expect(Math.max(...scoreVelocities) / Math.min(...scoreVelocities)).toBeLessThan(
    1.005,
  )
})

test('renders standard TAB stems, strum arrows, numbered melody and timed lyrics', async ({
  page,
}) => {
  const projectId = '228e3288-429f-49c6-924e-fda060e366be'
  await page.goto(`/?project=${projectId}`)
  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.tab-notation.has-vocal-guide')).toHaveCount(65)
  expect(await page.locator('.numbered-note').count()).toBeGreaterThan(200)
  expect(await page.locator('.lyric-token').count()).toBeGreaterThan(150)
  const numberedMeasureTotals = await page
    .locator('.score-measure')
    .evaluateAll(measures =>
      measures.map(measure =>
        [...measure.querySelectorAll('[data-numbered-duration]')].reduce(
          (sum, event) =>
            sum + Number(event.getAttribute('data-numbered-duration')),
          0,
        ),
      ),
    )
  expect(numberedMeasureTotals).toEqual(
    Array.from({ length: 65 }, () => 4),
  )
  expect(await page.locator('.numbered-note.is-rest').count()).toBeGreaterThan(
    100,
  )

  const flags = await page.locator('.rhythm-flag').evaluateAll(elements =>
    elements.map(element => ({
      tag: element.tagName.toLowerCase(),
      y1: element.getAttribute('y1'),
      y2: element.getAttribute('y2'),
    })),
  )
  expect(flags.length).toBeGreaterThan(0)
  expect(flags.every(flag =>
    flag.tag === 'line' && flag.y1 === flag.y2
  )).toBe(true)
  const stackedBeams = await page
    .locator('.rhythm-event[data-flags="2"]')
    .first()
    .locator('.rhythm-beam, .rhythm-flag')
    .evaluateAll(elements =>
      elements.map(element => ({
        level: Number(element.getAttribute('data-beam-level')),
        y: Number(element.getAttribute('y1')),
      })),
    )
  expect(stackedBeams.find(beam => beam.level === 1)!.y).toBeLessThan(
    stackedBeams.find(beam => beam.level === 0)!.y,
  )

  const strumStem = page.locator('.strum-mark').first()
  await expect(strumStem).toBeVisible()
  const strumGeometry = await strumStem.evaluate(element => {
    const line = element.querySelector('line')!
    return {
      direction: element.getAttribute('data-direction'),
      y1: Number(line.getAttribute('y1')),
      y2: Number(line.getAttribute('y2')),
    }
  })
  expect(Math.min(strumGeometry.y1, strumGeometry.y2)).toBe(40)
  expect(Math.max(strumGeometry.y1, strumGeometry.y2)).toBe(114)

  const guideSpacing = await page.locator('.score-measure').evaluateAll(measures =>
    measures.flatMap(measure => {
      const notes = [
        ...measure.querySelectorAll<SVGTextElement>(
          '.numbered-note .numbered-digit',
        ),
      ]
        .map(note => {
          const bounds = note.getBoundingClientRect()
          return (bounds.left + bounds.right) / 2
        })
        .sort((left, right) => left - right)
      return notes.slice(1).map((left, index) => left - notes[index])
    }),
  )
  expect(Math.min(...guideSpacing)).toBeGreaterThanOrEqual(9)
  const unmatchedLyrics = await page.locator('.score-measure').evaluateAll(measures =>
    measures.flatMap(measure => {
      const noteCenters = [
        ...measure.querySelectorAll<SVGTextElement>(
          '.numbered-note .numbered-digit',
        ),
      ].map(note => {
        const bounds = note.getBoundingClientRect()
        return (bounds.left + bounds.right) / 2
      })
      return [...measure.querySelectorAll<SVGTextElement>('.lyric-token')]
        .filter(lyric => {
          const bounds = lyric.getBoundingClientRect()
          const center = (bounds.left + bounds.right) / 2
          return !noteCenters.some(noteCenter =>
            Math.abs(noteCenter - center) < 1
          )
        })
        .map(lyric => lyric.textContent)
    }),
  )
  expect(unmatchedLyrics).toEqual([])
  const guideFonts = await page.evaluate(() => ({
    numbered: Number.parseFloat(
      getComputedStyle(
        document.querySelector('.numbered-note .numbered-digit')!,
      ).fontSize,
    ),
    lyric: Number.parseFloat(
      getComputedStyle(document.querySelector('.lyric-token')!).fontSize,
    ),
  }))
  expect(guideFonts.numbered).toBeGreaterThanOrEqual(16)
  expect(guideFonts.lyric).toBeGreaterThanOrEqual(13)
  const overflow = await page.locator('.score-measure').evaluateAll(measures =>
    measures.filter(measure =>
      measure.scrollWidth > measure.clientWidth + 1 ||
      measure.scrollHeight > measure.clientHeight + 1
    ).length,
  )
  expect(overflow).toBe(0)
})

test('closes measures correctly across simple and compound meters', async ({ page }) => {
  const response = await page.request.get(
    'http://127.0.0.1:8787/api/projects/demo',
  )
  const project = await response.json() as StudioProject
  const baseNote = project.tab.measures[0].beats[0].notes[0]
  let routedProject = project
  await page.route('**/api/projects/demo', route =>
    route.fulfill({ json: routedProject }),
  )
  const signatures = [
    [2, 4],
    [3, 4],
    [4, 4],
    [6, 8],
    [12, 8],
  ] as const
  for (const signature of signatures) {
    const beatDuration = 0.25
    const measureDuration = signature[0] * beatDuration
    const measures = Array.from({ length: 2 }, (_, measureIndex) => {
      const start = measureIndex * measureDuration
      return {
        number: measureIndex + 1,
        start,
        duration: measureDuration,
        chord: 'N.C.',
        beats: Array.from({ length: signature[0] }, (_, beatIndex) => {
          const at = start + beatIndex * beatDuration
          return {
            at,
            duration: beatDuration,
            style: 'pick' as const,
            notes: [{
              ...baseNote,
              id: `meter-${signature.join('-')}-${measureIndex}-${beatIndex}`,
              at,
              notationAt: at,
              duration: 0.08,
              technique: 'pick' as const,
            }],
          }
        }),
      }
    })
    routedProject = {
      ...project,
      duration: measureDuration * measures.length,
      tab: {
        ...project.tab,
        timeSignature: [signature[0], signature[1]],
        measures,
      },
    }

    await page.goto('/')
    const rendered = page.locator('.score-measure')
    await expect(rendered).toHaveCount(2)
    const totals = await rendered.evaluateAll(elements =>
      elements.map(measure =>
        [...measure.querySelectorAll(
          '.rhythm-event, .rhythm-continuation, .rhythm-rest-event',
        )].reduce(
          (sum, event) =>
            sum + Number(event.getAttribute('data-duration-beats')),
          0,
        ),
      ),
    )
    const denominatorBeatScale = 4 / signature[1]
    expect(
      totals.map(total => total / denominatorBeatScale),
    ).toEqual([signature[0], signature[0]])
    await expect(page.locator('.rhythm-beat-slot')).toHaveCount(
      signature[0] * measures.length,
    )
  }
})
