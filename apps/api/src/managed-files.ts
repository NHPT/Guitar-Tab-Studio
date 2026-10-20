import { unlinkSync } from 'node:fs'

export function unlinkIfExists(path: string): boolean {
  try {
    unlinkSync(path)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false
    throw error
  }
}
