const MAX_SCORE_PAGES = 24

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob)
        } else {
          reject(new Error('PDF 页面转换失败'))
        }
      },
      'image/png',
    )
  })
}

export async function prepareScorePages(file: File): Promise<File[]> {
  const isPdf =
    file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
  if (!isPdf) return [file]

  const [{ getDocument, GlobalWorkerOptions }, { default: pdfWorkerUrl }] =
    await Promise.all([
      import('pdfjs-dist'),
      import('pdfjs-dist/build/pdf.worker.min.mjs?url'),
    ])
  GlobalWorkerOptions.workerSrc = pdfWorkerUrl
  const loadingTask = getDocument({
    data: new Uint8Array(await file.arrayBuffer()),
  })
  try {
    const document = await loadingTask.promise
    if (document.numPages > MAX_SCORE_PAGES) {
      throw new Error(`PDF 最多支持 ${MAX_SCORE_PAGES} 页`)
    }
    const baseName = file.name.replace(/\.pdf$/iu, '') || 'score'
    const pages: File[] = []
    for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
      const page = await document.getPage(pageNumber)
      const baseViewport = page.getViewport({ scale: 1 })
      const scale = Math.max(
        1.75,
        Math.min(3, 2200 / Math.max(baseViewport.width, baseViewport.height)),
      )
      const viewport = page.getViewport({ scale })
      const canvas = window.document.createElement('canvas')
      canvas.width = Math.ceil(viewport.width)
      canvas.height = Math.ceil(viewport.height)
      const context = canvas.getContext('2d', {
        alpha: false,
        willReadFrequently: false,
      })
      if (!context) throw new Error('浏览器无法创建 PDF 页面画布')
      context.fillStyle = '#ffffff'
      context.fillRect(0, 0, canvas.width, canvas.height)
      await page.render({
        canvas,
        canvasContext: context,
        viewport,
      }).promise
      const blob = await canvasBlob(canvas)
      pages.push(
        new File(
          [blob],
          `${baseName}-page-${String(pageNumber).padStart(2, '0')}.png`,
          { type: 'image/png' },
        ),
      )
      page.cleanup()
    }
    return pages
  } finally {
    await loadingTask.destroy()
  }
}
