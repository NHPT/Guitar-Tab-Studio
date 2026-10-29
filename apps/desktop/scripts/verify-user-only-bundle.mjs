import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { resolve } from 'node:path'

const webRoot = resolve('../web/dist-desktop')
const apiEntry = resolve('build-resources/api/index.mjs')
const desktopEntry = resolve(webRoot, 'desktop.html')
const forbiddenMarkers = [
  '/api/admin',
  '/admin/reviews',
  'AdminWorkspace',
  'gts_admin_session',
]

function filesIn(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name)
    return entry.isDirectory() ? filesIn(path) : [path]
  })
}

if (!existsSync(desktopEntry)) {
  throw new Error('Desktop renderer entry is missing')
}
if (!existsSync(apiEntry)) {
  throw new Error('Desktop API bundle is missing')
}

for (const path of [...filesIn(webRoot), apiEntry]) {
  if (!statSync(path).isFile()) continue
  const source = readFileSync(path, 'utf8')
  for (const marker of forbiddenMarkers) {
    if (source.includes(marker)) {
      throw new Error(
        `User-only desktop bundle contains forbidden marker ${marker} in ${path}`,
      )
    }
  }
}

process.stdout.write('Verified user-only desktop renderer and API bundles\n')
