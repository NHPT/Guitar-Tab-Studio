export function acceptsPrereleaseUpdates(version: string): boolean {
  return version.includes('-')
}
