import { spawn } from 'node:child_process'
import { existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repositoryRoot = resolve(scriptDirectory, '..')
const dataRoot = resolve(
  process.env.GTS_DESKTOP_DATA_ROOT ??
    resolve(repositoryRoot, 'apps/desktop/data/preview-user-data'),
)
const debugPort = process.env.GTS_DESKTOP_DEBUG_PORT ?? '9222'

if (!/^\d{1,5}$/u.test(debugPort) || Number(debugPort) > 65_535) {
  throw new Error(`Invalid GTS_DESKTOP_DEBUG_PORT: ${debugPort}`)
}

const executable =
  process.platform === 'darwin'
    ? resolve(
        repositoryRoot,
        `apps/desktop/release/mac-${process.arch}`,
        'Guitar Tab Studio.app/Contents/MacOS/Guitar Tab Studio',
      )
    : process.platform === 'win32'
      ? resolve(
          repositoryRoot,
          'apps/desktop/release/win-unpacked/Guitar Tab Studio.exe',
        )
      : undefined

if (!executable || !existsSync(executable)) {
  throw new Error(
    'Packaged desktop preview is unavailable for this platform. Run the desktop package:dir command first.',
  )
}

mkdirSync(dataRoot, { recursive: true })
const child = spawn(
  executable,
  [`--remote-debugging-port=${debugPort}`],
  {
    detached: true,
    stdio: 'ignore',
    env: {
      ...process.env,
      GTS_DESKTOP_DATA_ROOT: dataRoot,
    },
  },
)

await new Promise((resolveSpawn, rejectSpawn) => {
  child.once('spawn', resolveSpawn)
  child.once('error', rejectSpawn)
})
child.unref()

process.stdout.write(
  `Started Guitar Tab Studio with persistent preview data on port ${debugPort}\n`,
)
