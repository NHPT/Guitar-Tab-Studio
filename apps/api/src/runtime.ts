import { statfsSync } from 'node:fs'
import { resolve } from 'node:path'

const MIN_ANALYSIS_FREE_BYTES = 2 * 1024 * 1024 * 1024

export function runtimeDataRoot(): string {
  return resolve(process.env.GTS_DATA_ROOT ?? resolve(process.cwd(), 'data'))
}

export function analysisRequiredBytes(inputBytes: number): number {
  return Math.max(MIN_ANALYSIS_FREE_BYTES, inputBytes * 4)
}

export function availableStorageBytes(): number {
  const fileSystem = statfsSync(runtimeDataRoot())
  return fileSystem.bavail * fileSystem.bsize
}

export function hasAnalysisCapacity(
  inputBytes: number,
  availableBytes = availableStorageBytes(),
): boolean {
  return availableBytes >= analysisRequiredBytes(inputBytes)
}
