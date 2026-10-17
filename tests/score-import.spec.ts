import { expect, test } from '@playwright/test'

function tabPdf(): Buffer {
  const content = [
    '0 0 0 RG 1.4 w',
    ...[280, 255, 230, 205, 180, 155].map(
      (y) => `50 ${y} m 1150 ${y} l S`,
    ),
    '2 w',
    ...[50, 600, 1150].map((x) => `${x} 148 m ${x} 287 l S`),
    '1 1 1 rg',
    ...[
      [160, 142],
      [280, 167],
      [410, 192],
      [680, 142],
      [800, 167],
      [930, 192],
    ].map(([x, y]) => `${x} ${y} 42 28 re f`),
    '0 0 0 rg',
    'BT /F1 28 Tf 170 150 Td (3) Tj ET',
    'BT /F1 28 Tf 290 175 Td (2) Tj ET',
    'BT /F1 28 Tf 420 200 Td (0) Tj ET',
    'BT /F1 28 Tf 690 150 Td (3) Tj ET',
    'BT /F1 28 Tf 810 175 Td (2) Tj ET',
    'BT /F1 28 Tf 940 200 Td (0) Tj ET',
    'BT /F1 26 Tf 120 345 Td (G) Tj ET',
    'BT /F1 26 Tf 680 345 Td (C) Tj ET',
    'BT /F1 24 Tf 170 95 Td (country roads take me home) Tj ET',
  ].join('\n')
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1200 420] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    `<< /Length ${Buffer.byteLength(content)} >>\nstream\n${content}\nendstream`,
  ]
  let pdf = '%PDF-1.4\n'
  const offsets = [0]
  objects.forEach((object, index) => {
    offsets.push(Buffer.byteLength(pdf))
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`
  })
  const xref = Buffer.byteLength(pdf)
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`
  for (const offset of offsets.slice(1)) {
    pdf += `${String(offset).padStart(10, '0')} 00000 n \n`
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\n`
  pdf += `startxref\n${xref}\n%%EOF\n`
  return Buffer.from(pdf)
}

test('imports a PDF TAB as an editable and playable project', async ({
  page,
}) => {
  test.setTimeout(45_000)
  await page.goto('/')
  await page.getByRole('button', { name: '导入歌曲' }).click()
  await page.getByRole('button', { name: '图片 / PDF' }).click()
  await page.locator('input[accept*=".pdf"]').setInputFiles({
    name: 'playable-tab.pdf',
    mimeType: 'application/pdf',
    buffer: tabPdf(),
  })

  await expect(
    page.getByRole('heading', { name: 'playable-tab' }),
  ).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('.score-measure')).toHaveCount(2)
  await expect(page.locator('.tab-glyph')).toHaveCount(6)
  await expect(page.locator('.track-title strong')).toHaveText(['吉他'])

  await page.getByRole('button', { name: '完整曲谱' }).click()
  await expect(page.locator('.chord-diagram figcaption')).toHaveText(['G', 'C'])
  const firstNote = page.locator('.tab-glyph').first()
  await firstNote.click()
  await expect(page.locator('.inspector')).toBeVisible()

  await page.getByLabel('预备小节数').selectOption('0')
  await page.getByRole('button', { name: '播放', exact: true }).click()
  await expect
    .poll(() => page.locator('.time-readout').textContent())
    .not.toContain('0:00 /')
})
