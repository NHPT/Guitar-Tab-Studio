import { expect, test } from '@playwright/test'

test('keeps the cloud-backed workspace usable on phone and tablet layouts', async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile')
  await page.goto('/')
  await expect(
    page.getByRole('heading', {
      name: '国际歌',
    }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: '导入歌曲' })).toBeVisible()

  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(1)

  await page.getByRole('button', { name: '导入歌曲' }).click()
  const dialog = page.getByRole('dialog', { name: '导入歌曲' })
  const dialogBounds = await dialog.boundingBox()
  expect(dialogBounds).not.toBeNull()
  expect(dialogBounds!.width).toBeLessThanOrEqual(390)
  await dialog.getByRole('button', { name: '关闭窗口' }).click()

  await page.getByRole('button', { name: '完整曲谱' }).click()
  const measures = await page.locator('.score-measure').evaluateAll(
    (items) =>
      items.slice(0, 2).map((item) => {
        const bounds = item.getBoundingClientRect()
        return {
          top: Math.round(bounds.top),
          left: Math.round(bounds.left),
          right: Math.round(bounds.right),
        }
      }),
  )
  expect(measures[1].top).toBeGreaterThan(measures[0].top)
  expect(measures.every((measure) => measure.left >= 0)).toBe(true)
  expect(measures.every((measure) => measure.right <= 390)).toBe(true)
})
