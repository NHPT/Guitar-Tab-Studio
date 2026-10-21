import type { DesktopUpdateProgress } from '@gts/core'

export interface UpdateDownloadMetrics {
  percent: number
  transferred: number
  total: number
  bytesPerSecond: number
}

function finiteNonNegative(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0
}

export function normalizeUpdateProgress(
  version: string,
  metrics: UpdateDownloadMetrics,
): DesktopUpdateProgress {
  const totalBytes = finiteNonNegative(metrics.total)
  const transferredBytes = Math.min(
    finiteNonNegative(metrics.transferred),
    totalBytes || Number.MAX_SAFE_INTEGER,
  )
  const measuredPercent =
    totalBytes > 0 ? (transferredBytes / totalBytes) * 100 : metrics.percent

  return {
    phase: 'downloading',
    version,
    percent: Math.min(100, finiteNonNegative(measuredPercent)),
    transferredBytes,
    totalBytes,
    bytesPerSecond: finiteNonNegative(metrics.bytesPerSecond),
  }
}
