import { resolve } from 'node:path'

const stableDataSegments = ['@gts', 'desktop'] as const

export function desktopDataRoot(
  appDataRoot: string,
  configuredRoot?: string,
): string {
  return configuredRoot
    ? resolve(configuredRoot)
    : resolve(appDataRoot, ...stableDataSegments)
}
