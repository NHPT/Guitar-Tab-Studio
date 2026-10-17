import { expect, test } from '@playwright/test'

test('lists, switches and removes local projects from the sidebar', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  await page.addInitScript(() => {
    const projects = Array.from({ length: 12 }, (_, index) => ({
      schemaVersion: 1,
      id: `local-project-${index + 1}`,
      title: `Local project ${index + 1}`,
      artist: 'Local',
      source: {
        kind: 'upload',
        label: `Recording ${index + 1}`,
        filename: `recording-${index + 1}.wav`,
      },
      bpm: 120,
      key: 'E minor',
      duration: 2,
      createdAt: `2026-10-14T00:${String(index).padStart(2, '0')}:00.000Z`,
      analysisMode: 'inference',
      tracks: [],
      tab: {
        tuning: ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],
        capo: 0,
        timeSignature: [4, 4],
        measures: [
          {
            number: 1,
            start: 0,
            duration: 2,
            chord: 'Em',
            beats: [0, 0.5, 1, 1.5].map((at, beatIndex) => ({
              id: `beat-${index}-${beatIndex}`,
              at,
              duration: 0.5,
              notes: [],
            })),
          },
        ],
      },
      warnings: [],
    }))
    const demoProject = {
      ...projects[0],
      id: 'built-in-demo',
      title: 'Built-in demo',
      source: {
        kind: 'upload',
        label: 'Built-in recording',
        filename: 'built-in.wav',
      },
      analysisMode: 'demo',
    }
    const projectEntries = () =>
      projects.map((project) => ({
        id: project.id,
        title: project.title,
        sourceLabel: project.source.label,
        openedAt: project.createdAt,
      }))

    Object.defineProperty(window, 'gtsDesktop', {
      configurable: false,
      value: {
        async invoke(request: {
          method: string
          payload?: { id?: string }
        }) {
          switch (request.method) {
            case 'capabilities':
              return {
                ffmpeg: true,
                ffprobe: true,
                ytDlp: true,
                worker: true,
                mode: 'inference',
              }
            case 'platforms':
              return []
            case 'demo-project':
              return demoProject
            case 'project':
              return projects.find(
                (project) => project.id === request.payload?.id,
              )
            case 'local-projects':
              return projectEntries()
            case 'remove-project': {
              const index = projects.findIndex(
                (project) => project.id === request.payload?.id,
              )
              if (index >= 0) projects.splice(index, 1)
              return { removed: index >= 0 }
            }
            case 'project-revisions':
              return []
            default:
              throw new Error(`Unexpected desktop method: ${request.method}`)
          }
        },
        onOpenMedia() {
          return () => undefined
        },
        onOpenProject() {
          return () => undefined
        },
        onTranscriptionProgress() {
          return () => undefined
        },
      },
    })
  })

  await page.goto('/')
  const projectList = page.locator('.project-list')
  await expect(projectList.locator('.session-item')).toHaveCount(13)
  const defaultHeight = await projectList.evaluate(
    (element) => element.clientHeight,
  )
  expect(defaultHeight).toBeGreaterThanOrEqual(460)
  expect(defaultHeight).toBeLessThanOrEqual(468)
  expect(
    await projectList.evaluate(
      (element) => element.scrollHeight > element.clientHeight,
    ),
  ).toBe(true)

  const resizer = page.getByRole('separator', {
    name: '调整项目列表高度',
  })
  const resizerBounds = await resizer.boundingBox()
  expect(resizerBounds).not.toBeNull()
  await page.mouse.move(
    resizerBounds!.x + resizerBounds!.width / 2,
    resizerBounds!.y + resizerBounds!.height / 2,
  )
  await page.mouse.down()
  await page.mouse.move(
    resizerBounds!.x + resizerBounds!.width / 2,
    resizerBounds!.y + resizerBounds!.height / 2 - 120,
  )
  await page.mouse.up()
  const resizedHeight = await projectList.evaluate(
    (element) => element.clientHeight,
  )
  expect(resizedHeight).toBeLessThanOrEqual(defaultHeight - 110)
  expect(
    await page.evaluate(() =>
      Number(window.localStorage.getItem('gts-project-list-height')),
    ),
  ).toBe(resizedHeight)

  await page.reload()
  await expect(projectList.locator('.session-item')).toHaveCount(13)
  expect(
    await projectList.evaluate((element) => element.clientHeight),
  ).toBe(resizedHeight)
  expect(
    await projectList.evaluate(
      (element) => element.scrollHeight > element.clientHeight,
    ),
  ).toBe(true)

  await projectList.getByRole('button', { name: /Local project 2/u }).click()
  await expect(projectList.locator('.session-item')).toHaveCount(13)
  await expect(
    projectList.getByRole('button', { name: /Built-in demo/u }),
  ).toBeVisible()
  await projectList.getByRole('button', { name: /Built-in demo/u }).click()
  await expect(page.getByRole('heading', { name: 'Built-in demo' })).toBeVisible()
  await expect(projectList.locator('.session-item')).toHaveCount(13)

  await projectList
    .getByRole('button', { name: /Local project 2/u })
    .click({ button: 'right' })
  const menu = page.getByRole('menu')
  await expect(menu.getByRole('menuitem', { name: '删除项目' })).toBeVisible()
  page.once('dialog', (dialog) => dialog.accept())
  await menu.getByRole('menuitem', { name: '删除项目' }).click()
  await expect(projectList.locator('.session-item')).toHaveCount(12)
  await expect(
    projectList.getByRole('button', { name: /Local project 2/u }),
  ).toHaveCount(0)

  await page.locator('.workspace').click({ button: 'right' })
  await expect(page.getByRole('menu')).toHaveCount(0)
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-local-projects.png`,
  })
})
